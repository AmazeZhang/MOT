#!/usr/bin/env python3
"""S01 fixed-detection CV+E visual replay with causal, source-ID keyed memory."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_p1_common_state_mechanism import read_detection_cache, write_tracker_row
from run_e1_mature_reemit_factorial import u_extra_rows
from mot_research.association.appearance_core import (
    CausalAppearanceMemory, fuse_appearance_score, select_high_detections,
)
from mot_research.analysis.runtime_record import peak_rss_kib
from mot_research.tracking.first_round_adapter import FirstRoundAssociationAdapter
from mot_research.tracking.natural_motion_policy import (
    NaturalMotionFirstRoundPolicy, TrackerDetectionHistory,
)

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"G": (0.0, None), "Z": (0.0, "ema"),
              "L": (0.25, "last"), "M": (0.25, "ema")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def all_detection_ids(path: Path) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    seen = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame, did = int(row["frame"]), int(row["det_id"])
            if (frame, did) in seen:
                raise ValueError("duplicate source detection key")
            seen.add((frame, did))
            result[frame].append(did)
    return dict(result)


def load_features(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]]:
    result = {}
    with np.load(path, allow_pickle=False) as saved:
        keys, boxes, scores = saved["keys"], saved["xyxy"], saved["scores"]
        values, valid = saved["features"], saved["valid"]
        if (keys.ndim != 2 or keys.shape[1] != 2 or values.shape != (len(keys), 2048)
                or values.dtype != np.float32 or len(valid) != len(keys)):
            raise ValueError("feature cache schema/dimension changed")
        for key, box, score, value, ok in zip(keys, boxes, scores, values, valid):
            k = (int(key[0]), int(key[1]))
            if k in result or (not ok and np.any(value != 0)):
                raise ValueError("duplicate feature key or invalid feature not zeroed")
            result[k] = (value.copy(), box.copy(), float(score))
    return result


def archived_rows(path: Path, last_frame: int) -> list[list[float]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            values = [float(item) for item in line.strip().split(",")]
            if int(values[0]) > last_frame:
                break
            rows.append(values)
    return rows


class VisualPolicy:
    def __init__(self, cv_policy, memory, weight, assoc_module):
        self.cv_policy, self.memory, self.weight = cv_policy, memory, weight
        self.assoc_module = assoc_module
        self.det_ids: tuple[int, ...] = ()
        self.det_features = np.empty((0, 2048), np.float64)
        self.trace: dict = {}
        self.begun = False

    def begin_without_association(self, frame: int, tracker_ids: tuple[int, ...]) -> None:
        if self.memory is not None:
            self.memory.begin_frame(frame, tracker_ids)
        self.begun = True

    def __call__(self, request, upstream_associate):
        if self.begun:
            raise RuntimeError("appearance memory begin_frame repeated")
        if len(request.detections) != len(self.det_ids):
            raise RuntimeError("source high detection IDs misaligned with association")
        if self.memory is None:
            past = np.zeros((len(request.tracker_ids), 2048), np.float64)
            written = np.full(len(request.tracker_ids), -1, np.int64)
        else:
            past, written = self.memory.begin_frame(request.frame_id, request.tracker_ids)
        self.begun = True
        self.trace["requested_det_ids"] = list(self.det_ids)
        self.trace["requested_tracker_ids"] = list(request.tracker_ids)
        self.trace["memory_last_write_frames"] = written.tolist()

        def associate(dets, trks, threshold, velocities, observations, inertia):
            used_edges = 0
            fast_path = False
            if len(dets) and len(trks):
                iou = self.assoc_module.iou_batch(dets, trks)
                eligible = iou > threshold
                fast_path = bool(eligible.sum(1).max() == 1 and eligible.sum(0).max() == 1)
            self.trace["fast_path"] = fast_path
            if self.weight == 0:
                result = upstream_associate(dets, trks, threshold, velocities, observations, inertia)
                self.trace["base_first_pairs"] = self._pairs(result[0], request.tracker_ids)
                self.trace["first_pairs"] = self.trace["base_first_pairs"]
            else:
                base_result = upstream_associate(dets, trks, threshold, velocities, observations, inertia)
                self.trace["base_first_pairs"] = self._pairs(base_result[0], request.tracker_ids)
                original_lap = self.assoc_module.linear_assignment

                def visual_lap(cost):
                    nonlocal used_edges
                    iou_matrix = self.assoc_module.iou_batch(dets, trks)
                    fused, used = fuse_appearance_score(
                        -cost, self.det_features, past, iou_matrix >= 0.3,
                        weight=self.weight)
                    used_edges = int(used.sum())
                    return original_lap(-fused)

                self.assoc_module.linear_assignment = visual_lap
                try:
                    result = upstream_associate(dets, trks, threshold, velocities, observations, inertia)
                finally:
                    self.assoc_module.linear_assignment = original_lap
                self.trace["first_pairs"] = self._pairs(result[0], request.tracker_ids)
            self.trace["appearance_eligible_edges"] = used_edges
            return result

        return self.cv_policy(request, associate)

    def _pairs(self, pairs, tracker_ids):
        return [[int(self.det_ids[d]), int(tracker_ids[t])] for d, t in pairs]


def run_condition(condition: str, lock: dict, sequence: str, last_frame: int,
                  cached: dict, source_ids: dict, feature_map: dict, output: Path,
                  ocsort_module, assoc_module, height: int, width: int, fps: float) -> dict:
    weight, memory_mode = CONDITIONS[condition]
    tracker = ocsort_module.OCSort(det_thresh=.6, max_age=30, min_hits=3,
        iou_threshold=.3, delta_t=3, asso_func="iou", inertia=.2, use_byte=False)
    history = TrackerDetectionHistory(history_length=8, score_threshold=.6)
    cv_policy = NaturalMotionFirstRoundPolicy(mode="constant_velocity", history=history,
        image_width=width, image_height=height, fps=fps,
        minimum_observations=2, maximum_frame_gap=31)
    memory = None if memory_mode is None else CausalAppearanceMemory(
        2048, mode=memory_mode, alpha=.9)
    visual = VisualPolicy(cv_policy, memory, weight, assoc_module)
    adapter = FirstRoundAssociationAdapter(tracker, ocsort_module, visual)
    scale = min(800 / height, 1440 / width)
    paths = [output / "trackers" / condition / f"{sequence}.txt",
             output / "traces" / condition / f"{sequence}.jsonl"]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    last_original = {}
    mature_ids = set()
    counts = Counter()
    started = time.perf_counter()
    cpu_started = time.process_time()
    with paths[0].open("w", newline="") as track_handle, paths[1].open("w") as trace_handle:
        writer = csv.writer(track_handle, lineterminator="\n")
        for frame in range(1, last_frame + 1):
            ids = source_ids.get(frame, [])
            rows = np.asarray(cached.get(frame, []), np.float64).reshape(-1, 5)
            if len(ids) != len(rows):
                raise RuntimeError(f"{sequence}:{frame}: source rows/IDs mismatch")
            high_ids, high_rows, _ = select_high_detections(ids, rows)
            features = []
            for did, row in zip(high_ids, high_rows):
                key = (frame, did)
                if key not in feature_map:
                    raise RuntimeError(f"missing high feature record {key}")
                value, box, score = feature_map[key]
                if np.max(np.abs(row[:4] - box)) > 1e-5 or abs(row[4] - score) > 1e-8:
                    raise RuntimeError(f"source/cache feature box or score changed: {key}")
                features.append(value)
            visual.det_ids = high_ids
            visual.det_features = np.stack(features) if features else np.empty((0, 2048), np.float32)
            visual.trace = {"sequence": sequence, "frame": frame, "condition": condition}
            visual.begun = False
            before_hits = {int(t.id) + 1: int(t.hits) for t in tracker.trackers}
            mature_before = set(mature_ids)
            events: list[tuple[int, int, str]] = []

            def record(tid, did, stage):
                events.append((int(tid), int(did), stage))

            tracker._s01_source_det_ids = ids
            tracker._s01_record = record
            tracker_input = None if len(rows) == 0 else rows.copy()
            if tracker_input is not None:
                tracker_input[:, :4] *= scale
            else:
                visual.begin_without_association(frame, tuple(int(t.id) + 1 for t in tracker.trackers))
            original = adapter.update(tracker_input, (height, width), (800, 1440), frame_id=frame)
            original = np.asarray(original, np.float64).reshape(-1, 5)
            if tracker_input is not None and not visual.begun:
                raise RuntimeError("first-round association was not invoked")
            if len({did for _, did, _ in events}) != len(events):
                raise RuntimeError("duplicate detection assigned within frame")
            if len({tid for tid, _, _ in events}) != len(events):
                raise RuntimeError("same track updated twice within frame")
            mapping = {tid: did for tid, did, _ in events}
            births = {tid: np.asarray([*rows[ids.index(did), :]], np.float64)
                      for tid, did, stage in events if stage == "birth"}
            if tracker_input is not None:
                history.record_after_update(frame, tracker.trackers, births)
            if memory is not None:
                live = [int(t.id) + 1 for t in tracker.trackers]
                memory.finish_frame(frame, live, mapping, high_ids, visual.det_features)

            extra, details = u_extra_rows(tracker, original, mature_before,
                                          before_hits, tracker_input is not None)
            output_ids = []
            for row in original:
                write_tracker_row(writer, frame, row)
                output_ids.append(int(row[4]))
            for row, detail in zip(extra, details):
                if float(detail["score"]) >= .8:
                    write_tracker_row(writer, frame, row)
                    output_ids.append(int(row[4]))
            if len(output_ids) != len(set(output_ids)):
                raise RuntimeError("duplicate output track ID")
            for row in original:
                last_original[int(row[4])] = (frame, row[:4].copy())
            for t in tracker.trackers:
                tid = int(t.id) + 1
                if tid in {int(row[4]) for row in original} and t.hit_streak >= tracker.min_hits:
                    mature_ids.add(tid)
            visual.trace.update({
                "requested_det_ids": visual.trace.get("requested_det_ids", []),
                "requested_tracker_ids": visual.trace.get("requested_tracker_ids", []),
                "base_first_pairs": visual.trace.get("base_first_pairs", []),
                "first_pairs": visual.trace.get("first_pairs", []),
                "final_actual_updates": [[tid, did, stage] for tid, did, stage in events],
                "output_ids": output_ids,
                "appearance_eligible_edges": visual.trace.get("appearance_eligible_edges", 0),
                "fast_path": visual.trace.get("fast_path", False),
            })
            trace_handle.write(json.dumps(visual.trace, separators=(",", ":")) + "\n")
            counts["frames"] += 1
            counts["output_rows"] += len(output_ids)
            counts["actual_updates"] += len(events)
            counts["appearance_eligible_edges"] += visual.trace["appearance_eligible_edges"]
            counts["changed_first_round_frames"] += visual.trace["first_pairs"] != visual.trace["base_first_pairs"]
    return {"condition": condition, "sequence": sequence, "counts": dict(counts),
            "track_sha256": sha256(paths[0]), "trace_sha256": sha256(paths[1]),
            "wall_seconds": time.perf_counter() - started,
            "cpu_core_hours": (time.process_time() - cpu_started) / 3600,
            "process_peak_rss_kib": peak_rss_kib()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("tracker replay must use CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    for rel_path, expected in lock["code_hashes"].items():
        actual = sha256(ROOT / rel_path)
        if actual != expected:
            raise RuntimeError(f"code SHA256 differs from lock: {rel_path}")
    if args.sequence not in lock["selected_sequences"] or args.last_frame < 1:
        raise ValueError("sequence/frame outside lock")
    output = Path(lock["output_root"])
    feature_file = Path(lock["feature_cache_path"])
    if sha256(feature_file) != lock["feature_cache_sha256"]:
        raise RuntimeError("feature cache SHA256 differs from lock")
    cache_file = Path(lock["detection_root"]) / f"{args.sequence}.csv"
    if sha256(cache_file) != lock["input_detection_sha256"][args.sequence]:
        raise RuntimeError("source detection hash changed")
    sys.path.insert(0, str((ROOT / "third_party/OC_SORT").resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module
    seq_dir = Path(lock["train_root"]) / args.sequence
    cfg = configparser.ConfigParser()
    if not cfg.read(seq_dir / "seqinfo.ini"):
        raise FileNotFoundError("seqinfo.ini")
    info = cfg["Sequence"]
    height, width, fps = info.getint("imHeight"), info.getint("imWidth"), info.getfloat("frameRate")
    cached, _ = read_detection_cache(cache_file)
    ids = all_detection_ids(cache_file)
    features = load_features(feature_file)
    results = {}
    for condition in args.conditions:
        results[condition] = run_condition(condition, lock, args.sequence, args.last_frame,
            cached, ids, features, output, ocsort_module, assoc_module, height, width, fps)
    if "G" in results:
        reference = archived_rows(Path(lock["e3_reference_tracker"]), args.last_frame)
        produced = archived_rows(output / "trackers/G" / f"{args.sequence}.txt", args.last_frame)
        if len(reference) != len(produced):
            raise RuntimeError("G/E3 E output row count mismatch")
        for i, (a, b) in enumerate(zip(reference, produced)):
            if a[0:2] != b[0:2] or np.max(np.abs(np.asarray(a[2:6]) - b[2:6])) > 1e-4:
                raise RuntimeError(f"G/E3 E output mismatch at row {i}")
        results["G"]["archived_E3_E_rows_exact_ID_box_atol_1e-4"] = len(reference)
    if "G" in results and "Z" in results:
        for stream in ("trackers", "traces"):
            if stream == "trackers":
                if (output / stream / "G" / f"{args.sequence}.txt").read_bytes() != \
                        (output / stream / "Z" / f"{args.sequence}.txt").read_bytes():
                    raise RuntimeError("G/Z output differs")
            else:
                g = [json.loads(x) for x in (output / stream / "G" / f"{args.sequence}.jsonl").read_text().splitlines()]
                z = [json.loads(x) for x in (output / stream / "Z" / f"{args.sequence}.jsonl").read_text().splitlines()]
                if len(g) != len(z) or len(g) != args.last_frame:
                    raise RuntimeError("G/Z trace frame count differs")
                for a, b in zip(g, z):
                    for field in ("base_first_pairs", "first_pairs", "final_actual_updates", "output_ids"):
                        if a[field] != b[field]:
                            raise RuntimeError(f"G/Z {field} differs at frame {a['frame']}")
        results["Z"]["G_equivalence"] = "PASS"
    result_file = output / "engineering_checks.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    result_file.write_text(json.dumps({"status": "ENGINEERING_PASS", "results": results}, indent=2) + "\n")
    print(json.dumps({"status": "ENGINEERING_PASS", "results": results}, indent=2))


if __name__ == "__main__":
    main()
