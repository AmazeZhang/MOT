#!/usr/bin/env python3
"""Engineering-only exact S01 M replay and one visual-memory-write fork."""

from __future__ import annotations

import argparse
import configparser
import copy
import csv
import hashlib
import io
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from analyze_p1_common_state_mechanism import read_detection_cache, write_tracker_row
from run_e1_mature_reemit_factorial import u_extra_rows
from run_s01_visual_baselines import VisualPolicy, all_detection_ids, load_features
from mot_research.association.appearance_core import CausalAppearanceMemory, select_high_detections
from mot_research.tracking.first_round_adapter import FirstRoundAssociationAdapter
from mot_research.tracking.natural_motion_policy import (
    NaturalMotionFirstRoundPolicy, TrackerDetectionHistory,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/s03_memory_clone_smoke_20261121.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA mismatch {path}: {actual} != {expected}")


class CVTap:
    """Observe the first round after CV adaptation without changing assignment."""

    def __init__(self, base, visual, assoc_module, frame: int, track_id: int,
                 positive_det_id: int):
        self.base = base
        self.visual = visual
        self.assoc_module = assoc_module
        self.frame = frame
        self.track_id = track_id
        self.positive_det_id = positive_det_id
        self.capture = None

    def __call__(self, request, associate):
        def observed(dets, trks, threshold, velocities, observations, inertia):
            if request.frame_id == self.frame:
                dids = self.visual.det_ids
                tids = request.tracker_ids
                if self.positive_det_id not in dids or self.track_id not in tids:
                    raise RuntimeError("S02 event positive detection or target track absent")
                didx = dids.index(self.positive_det_id)
                tidx = tids.index(self.track_id)
                iou = self.assoc_module.iou_batch(dets, trks)
                value = float(iou[didx, tidx])
                self.capture = {"frame": self.frame, "track_id": self.track_id,
                    "owner_positive_det_id": self.positive_det_id,
                    "first_round_iou_after_CV": value,
                    "first_round_legal_at_0_3": value >= 0.3,
                    "requested_detection_count": len(dids),
                    "requested_tracker_count": len(tids)}
            return associate(dets, trks, threshold, velocities, observations, inertia)
        return self.base(request, observed)


class MRunner:
    def __init__(self, sequence: str, cached: dict, source_ids: dict,
                 features: dict, height: int, width: int, fps: float,
                 ocsort_module, assoc_module, capture: tuple[int, int, int] | None):
        self.sequence = sequence
        self.cached, self.source_ids, self.features = cached, source_ids, features
        self.height, self.width, self.fps = height, width, fps
        self.ocsort_module, self.assoc_module = ocsort_module, assoc_module
        self.capture_spec = capture
        self.tracker = ocsort_module.OCSort(det_thresh=.6, max_age=30,
            min_hits=3, iou_threshold=.3, delta_t=3, asso_func="iou",
            inertia=.2, use_byte=False)
        self.history = TrackerDetectionHistory(history_length=8, score_threshold=.6)
        self.memory = CausalAppearanceMemory(2048, mode="ema", alpha=.9)
        self.mature_ids: set[int] = set()
        self.next_count = int(ocsort_module.KalmanBoxTracker.count)
        self._attach_policy()

    def _attach_policy(self) -> None:
        cv = NaturalMotionFirstRoundPolicy(mode="constant_velocity", history=self.history,
            image_width=self.width, image_height=self.height, fps=self.fps,
            minimum_observations=2, maximum_frame_gap=31)
        self.visual = VisualPolicy(cv, self.memory, .25, self.assoc_module)
        self.tap = None
        if self.capture_spec is not None:
            self.tap = CVTap(cv, self.visual, self.assoc_module, *self.capture_spec)
            self.visual.cv_policy = self.tap
        self.adapter = FirstRoundAssociationAdapter(self.tracker, self.ocsort_module, self.visual)

    def fork(self) -> "MRunner":
        other = object.__new__(MRunner)
        for name in ("sequence", "cached", "source_ids", "features", "height", "width",
                     "fps", "ocsort_module", "assoc_module", "capture_spec", "next_count"):
            setattr(other, name, getattr(self, name))
        other.tracker = copy.deepcopy(self.tracker)
        other.history = copy.deepcopy(self.history)
        other.memory = copy.deepcopy(self.memory)
        other.mature_ids = set(self.mature_ids)
        other._attach_policy()
        other.adapter.association_calls = self.adapter.association_calls
        return other

    def step(self, frame: int, suppress_track_id: int | None = None) -> tuple[str, str]:
        self.ocsort_module.KalmanBoxTracker.count = self.next_count
        ids = self.source_ids.get(frame, [])
        rows = np.asarray(self.cached.get(frame, []), np.float64).reshape(-1, 5)
        if len(ids) != len(rows):
            raise RuntimeError(f"source rows/IDs mismatch: {self.sequence}/{frame}")
        high_ids, high_rows, _ = select_high_detections(ids, rows)
        feat = []
        for did, row in zip(high_ids, high_rows):
            value, box, score = self.features[(frame, did)]
            if np.max(np.abs(row[:4] - box)) > 1e-5 or abs(row[4] - score) > 1e-8:
                raise RuntimeError("feature/source detection mismatch")
            feat.append(value)
        self.visual.det_ids = high_ids
        self.visual.det_features = np.stack(feat) if feat else np.empty((0, 2048), np.float32)
        self.visual.trace = {"sequence": self.sequence, "frame": frame, "condition": "M"}
        self.visual.begun = False
        before_hits = {int(t.id) + 1: int(t.hits) for t in self.tracker.trackers}
        mature_before = set(self.mature_ids)
        events: list[tuple[int, int, str]] = []

        def record(tid, did, stage):
            events.append((int(tid), int(did), stage))

        self.tracker._s01_source_det_ids = ids
        self.tracker._s01_record = record
        tracker_input = None if not len(rows) else rows.copy()
        if tracker_input is not None:
            tracker_input[:, :4] *= min(800 / self.height, 1440 / self.width)
        else:
            self.visual.begin_without_association(
                frame, tuple(int(t.id) + 1 for t in self.tracker.trackers))
        original = self.adapter.update(tracker_input, (self.height, self.width),
                                       (800, 1440), frame_id=frame)
        original = np.asarray(original, np.float64).reshape(-1, 5)
        if tracker_input is not None and not self.visual.begun:
            raise RuntimeError("first association not invoked")
        if len({did for _, did, _ in events}) != len(events) or \
                len({tid for tid, _, _ in events}) != len(events):
            raise RuntimeError("duplicate final update")
        mapping = {tid: did for tid, did, _ in events}
        births = {tid: np.asarray([*rows[ids.index(did), :]], np.float64)
                  for tid, did, stage in events if stage == "birth"}
        if tracker_input is not None:
            self.history.record_after_update(frame, self.tracker.trackers, births)
        live = [int(t.id) + 1 for t in self.tracker.trackers]
        if suppress_track_id is not None:
            if suppress_track_id not in mapping:
                raise RuntimeError("target write absent at selected frame")
            mapping = {tid: did for tid, did in mapping.items() if tid != suppress_track_id}
        self.memory.finish_frame(frame, live, mapping, high_ids, self.visual.det_features)
        self.next_count = int(self.ocsort_module.KalmanBoxTracker.count)

        extra, details = u_extra_rows(self.tracker, original, mature_before,
                                      before_hits, tracker_input is not None)
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
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
        current_ids = {int(row[4]) for row in original}
        for track in self.tracker.trackers:
            tid = int(track.id) + 1
            if tid in current_ids and track.hit_streak >= self.tracker.min_hits:
                self.mature_ids.add(tid)
        self.visual.trace.update({
            "requested_det_ids": self.visual.trace.get("requested_det_ids", []),
            "requested_tracker_ids": self.visual.trace.get("requested_tracker_ids", []),
            "base_first_pairs": self.visual.trace.get("base_first_pairs", []),
            "first_pairs": self.visual.trace.get("first_pairs", []),
            "final_actual_updates": [[tid, did, stage] for tid, did, stage in events],
            "output_ids": output_ids,
            "appearance_eligible_edges": self.visual.trace.get("appearance_eligible_edges", 0),
            "fast_path": self.visual.trace.get("fast_path", False),
        })
        return output.getvalue(), json.dumps(self.visual.trace, separators=(",", ":")) + "\n"


def tracker_equal(a: MRunner, b: MRunner) -> bool:
    if a.next_count != b.next_count or a.mature_ids != b.mature_ids or \
            a.history.snapshot() != b.history.snapshot() or \
            len(a.tracker.trackers) != len(b.tracker.trackers):
        return False
    for x, y in zip(a.tracker.trackers, b.tracker.trackers):
        if (x.id, x.hits, x.time_since_update, x.hit_streak) != \
                (y.id, y.hits, y.time_since_update, y.hit_streak):
            return False
        for name in ("x", "P"):
            if not np.array_equal(getattr(x.kf, name), getattr(y.kf, name)):
                return False
        if not np.array_equal(x.last_observation, y.last_observation):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S03 smoke requires CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if lock["protocol_sha256"] != sha256(PROTOCOL) or \
            lock["smoke_code_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("S03 smoke code/protocol differs from lock")
    for rel, expected in lock["dependency_code_sha256"].items():
        require_hash(ROOT / rel, expected)
    actual_versions = {"python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "pyyaml": yaml.__version__}
    if actual_versions != lock["environment_versions"]:
        raise RuntimeError(f"environment changed: {actual_versions}")
    plan = yaml.safe_load(PROTOCOL.read_text())
    inp = plan["inputs"]
    source_frozen = Path(inp["s01_frozen_inputs"])
    source_events = Path(inp["s02_events"])
    require_hash(source_frozen, inp["s01_frozen_inputs_sha256"])
    require_hash(source_events, inp["s02_events_sha256"])
    frozen = json.loads(source_frozen.read_text())
    events = [json.loads(line) for line in source_events.read_text().splitlines()]
    rank = {seq: i for i, seq in enumerate(frozen["sequences"])}
    actionable = sorted((e for e in events if e["condition"] == "M" and e["actionable_competition"]),
                        key=lambda e: (rank[e["sequence"]], e["frame"], e["track_id"]))
    event = actionable[0]
    if (event["sequence"], event["frame"], event["track_id"],
            event["owner_positive_det_id"]) != (
            inp["selected_sequence"], inp["selected_frame"],
            inp["selected_track_id"], inp["selected_owner_positive_det_id"]):
        raise RuntimeError("smoke event selection differs from frozen protocol")
    seq = event["sequence"]
    artifacts = {(a["sequence"], a["condition"], a["kind"]): a for a in frozen["artifacts"]}
    feature = next(a for a in frozen["feature_artifacts"] if a["sequence"] == seq)
    check = next(a for a in frozen["engineering_checks"] if a["sequence"] == seq)
    source_track = Path(artifacts[(seq, "M", "trackers")]["path"])
    source_trace = Path(artifacts[(seq, "M", "traces")]["path"])
    for path, expected in ((source_track, artifacts[(seq, "M", "trackers")]["sha256"]),
                           (source_trace, artifacts[(seq, "M", "traces")]["sha256"]),
                           (Path(feature["path"]), feature["sha256"]),
                           (Path(check["path"]), check["sha256"])):
        require_hash(path, expected)
    tracking_lock_path = Path(check["path"]).parent / "runtime_lock.json"
    require_hash(tracking_lock_path, check["runtime_lock_sha256"])
    tracking_lock = json.loads(tracking_lock_path.read_text())
    for rel, expected in tracking_lock["code_hashes"].items():
        require_hash(ROOT / rel, expected)
    det_path = Path(tracking_lock["detection_root"]) / f"{seq}.csv"
    require_hash(det_path, tracking_lock["input_detection_sha256"][seq])
    image_dir = Path(tracking_lock["train_root"]) / seq
    cfg = configparser.ConfigParser()
    if not cfg.read(image_dir / "seqinfo.ini"):
        raise FileNotFoundError("seqinfo.ini")
    info = cfg["Sequence"]
    height, width, fps, length = (info.getint("imHeight"), info.getint("imWidth"),
                                  info.getfloat("frameRate"), info.getint("seqLength"))
    cached, _ = read_detection_cache(det_path)
    ids = all_detection_ids(det_path)
    features = load_features(Path(feature["path"]))
    sys.path.insert(0, str((ROOT / "third_party/OC_SORT").resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module

    output = Path(plan["execution"]["output_root"])
    if output.exists() or str(output) != lock["output_root"]:
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_rows: dict[int, str] = defaultdict(str)
    for line in source_track.read_text().splitlines(keepends=True):
        source_rows[int(line.split(",", 1)[0])] += line
    source_trace_lines = source_trace.read_text().splitlines(keepends=True)
    if len(source_trace_lines) != length:
        raise RuntimeError("source trace length differs from seqinfo")
    runner = MRunner(seq, cached, ids, features, height, width, fps, ocsort_module,
                     assoc_module, (event["frame"], event["track_id"], event["owner_positive_det_id"]))
    snapshot = None
    with (output / "full_replay_tracker.txt").open("w") as track_handle, \
            (output / "full_replay_trace.jsonl").open("w") as trace_handle:
        for frame in range(1, length + 1):
            if frame == event["frame"]:
                snapshot = runner.fork()
            rows, trace = runner.step(frame)
            if rows != source_rows.get(frame, "") or trace != source_trace_lines[frame - 1]:
                raise RuntimeError(f"full replay differs at frame {frame}")
            track_handle.write(rows)
            trace_handle.write(trace)
    require_hash(output / "full_replay_tracker.txt", artifacts[(seq, "M", "trackers")]["sha256"])
    require_hash(output / "full_replay_trace.jsonl", artifacts[(seq, "M", "traces")]["sha256"])
    if snapshot is None:
        raise RuntimeError("event snapshot not captured")

    factual, suppressed = snapshot.fork(), snapshot.fork()
    future = min(length, event["frame"] + inp["branch_future_frames"])
    differing_frames = 0
    with (output / "factual_branch_tracker.txt").open("w") as ftrack, \
            (output / "factual_branch_trace.jsonl").open("w") as ftrace, \
            (output / "suppressed_branch_tracker.txt").open("w") as btrack, \
            (output / "suppressed_branch_trace.jsonl").open("w") as btrace_handle:
        for frame in range(event["frame"], future + 1):
            arows, atrace = factual.step(frame)
            brows, btrace_text = suppressed.step(frame, event["track_id"] if frame == event["frame"] else None)
            if arows != source_rows.get(frame, "") or atrace != source_trace_lines[frame - 1]:
                raise RuntimeError(f"factual clone differs from S01 at frame {frame}")
            if frame == event["frame"]:
                if arows != brows or atrace != btrace_text or not tracker_equal(factual, suppressed):
                    raise RuntimeError("event frame tracker/output/association changed")
                t = event["track_id"]
                if factual.memory._memory[t][0] != frame or suppressed.memory._memory[t][0] >= frame:
                    raise RuntimeError("target memory write was not isolated")
                for tid, value in factual.memory._memory.items():
                    if tid != t and (tid not in suppressed.memory._memory or
                            value[0] != suppressed.memory._memory[tid][0] or
                            not np.array_equal(value[1], suppressed.memory._memory[tid][1])):
                        raise RuntimeError("other track memory changed at intervention")
            elif arows != brows:
                differing_frames += 1
            ftrack.write(arows); ftrace.write(atrace)
            btrack.write(brows); btrace_handle.write(btrace_text)
    capture = runner.tap.capture
    if capture is None:
        raise RuntimeError("actual CV-adapted first-round matrix not captured")
    result = {"status": "ENGINEERING_PASS", "claim_level": "no_GT_scoring",
        "sequence": seq, "frames_replayed": length, "event": event,
        "branch_last_frame": future, "first_round_legal_screen": capture,
        "future_frames_with_different_output_rows": differing_frames,
        "source_tracker_sha256": sha256(source_track),
        "source_trace_sha256": sha256(source_trace),
        "full_replay_tracker_sha256": sha256(output / "full_replay_tracker.txt"),
        "full_replay_trace_sha256": sha256(output / "full_replay_trace.jsonl"),
        "factual_branch_tracker_sha256": sha256(output / "factual_branch_tracker.txt"),
        "suppressed_branch_tracker_sha256": sha256(output / "suppressed_branch_tracker.txt"),
        "runtime_lock_sha256": sha256(args.runtime_lock),
        "protocol_sha256": sha256(PROTOCOL), "GT_read": False}
    (output / "engineering_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
