#!/usr/bin/env python3
"""Paired, GT-privileged first-round association forks on fixed train-fit gaps."""

from __future__ import annotations

import configparser
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_p1_common_state_mechanism import read_detection_cache
from mot_research.analysis.failure_events import (
    Candidate, hungarian_iou_matches, read_detections, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, NaturalMotionFirstRoundPolicy,
    TrackerDetectionHistory,
)
from run_e1_mature_reemit_factorial import u_extra_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e4_oracle_association_headroom_20261119_retry3.yaml"
CODE_PATHS = (
    "scripts/run_e4_oracle_association_headroom.py",
    "scripts/run_e1_mature_reemit_factorial.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "src/mot_research/data/motion_samples.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
)


class CapturePolicy:
    def __init__(self, base):
        self.base = base
        self.tracker_ids = None
        self.detections = None
        self.called = 0

    def __call__(self, request, associate):
        self.called += 1
        self.tracker_ids = request.tracker_ids
        self.detections = np.asarray(request.detections).copy()
        return self.base(request, associate)


class ForcePairPolicy(CapturePolicy):
    def __init__(self, base, target_id: int, det_index: int):
        super().__init__(base)
        self.target_id = target_id
        self.det_index = det_index
        self.forced = None

    def __call__(self, request, associate):
        result = super().__call__(request, associate)
        if self.target_id not in request.tracker_ids:
            raise RuntimeError("target track absent in oracle first round")
        if not 0 <= self.det_index < len(request.detections):
            raise RuntimeError("positive detection index absent")
        track_index = request.tracker_ids.index(self.target_id)
        pairs = [(int(d), int(t)) for d, t in result[0]
                 if int(d) != self.det_index and int(t) != track_index]
        pairs.append((self.det_index, track_index))
        if len({d for d, _ in pairs}) != len(pairs) or len({t for _, t in pairs}) != len(pairs):
            raise RuntimeError("oracle pair violates one-to-one")
        self.forced = (self.det_index, track_index)
        used_d, used_t = {d for d, _ in pairs}, {t for _, t in pairs}
        return (np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                np.asarray(sorted(set(range(len(request.detections))) - used_d), dtype=np.int64),
                np.asarray(sorted(set(range(len(request.trackers))) - used_t), dtype=np.int64))


def output_candidates(rows) -> list[Candidate]:
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    ids = [int(row[4]) for row in values]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate output track ID")
    return [Candidate(int(row[4]), row[:4].copy()) for row in values]


def same_outputs(actual: list[Candidate], reference: list[Candidate], atol: float = 1e-4) -> bool:
    a = {item.candidate_id: item.box_xyxy for item in actual}
    b = {item.candidate_id: item.box_xyxy for item in reference}
    return a.keys() == b.keys() and all(np.max(np.abs(a[k] - b[k])) <= atol for k in a)


def build_candidates(sequence: str, gt: dict, c_outputs: dict, detections: dict,
                     length: int, cap: int) -> tuple[list[dict], int]:
    last_anchor = {}
    used_episode = set()
    candidates = []
    for frame in range(1, length + 1):
        anchors = hungarian_iou_matches(gt.get(frame, {}), c_outputs.get(frame, []), .5)[0]
        positives = hungarian_iou_matches(gt.get(frame, {}), detections.get(frame, []), .5)[0]
        for gt_id, (det_id, _) in positives.items():
            if gt_id not in last_anchor or gt_id in used_episode:
                continue
            anchor_frame, track_id = last_anchor[gt_id]
            if not (2 <= frame - anchor_frame <= 30 and frame + 4 <= length):
                continue
            used_episode.add(gt_id)
            rank = hashlib.sha256(
                f"e4:{sequence}:{frame:06d}:{gt_id}:{track_id}".encode("ascii")
            ).hexdigest()
            candidates.append({"sequence": sequence, "frame": frame, "gt_id": int(gt_id),
                               "track_id": int(track_id), "anchor_frame": anchor_frame,
                               "positive_det_id": int(det_id), "rank": rank})
        for gt_id, (track_id, _) in anchors.items():
            last_anchor[gt_id] = (frame, int(track_id))
            used_episode.discard(gt_id)
    count = len(candidates)
    candidates.sort(key=lambda x: (x["rank"], x["frame"], x["gt_id"]))
    return candidates[:cap], count


def step_e(runner, mature_ids: set[int], values, frame: int, height: int, width: int,
           inp: dict) -> tuple[list[Candidate], list[Candidate]]:
    tracker = runner.tracker
    before_hits = {int(t.id) + 1: int(t.hits) for t in tracker.trackers}
    # OC-SORT scales the Nx5 input in place. Every fork needs its own copy.
    independent_values = None if values is None else values.copy()
    original = runner.update(independent_values, (height, width),
                             tuple(inp["input_size"]), frame_id=frame)
    original = np.asarray(original, dtype=np.float64).reshape(-1, 5)
    extra, details = u_extra_rows(tracker, original, set(mature_ids), before_hits,
                                  values is not None)
    if len(extra) != len(details):
        raise RuntimeError("extra row/detail mismatch")
    selected = [row for row, item in zip(extra, details) if float(item["score"]) >= .8]
    c = output_candidates(original)
    e = output_candidates(list(original) + selected)
    original_ids = {item.candidate_id for item in c}
    for t in tracker.trackers:
        track_id = int(t.id) + 1
        if track_id in original_ids and t.hit_streak >= tracker.min_hits:
            mature_ids.add(track_id)
    return c, e


def target_positive_update(runner, target_id: int, before_hits: int,
                           positive_box: np.ndarray) -> bool:
    tracker = next((t for t in runner.tracker.trackers if int(t.id) + 1 == target_id), None)
    if tracker is None or tracker.hits != before_hits + 1:
        return False
    observed = np.asarray(tracker.last_observation, dtype=np.float64).reshape(-1)
    return observed.size >= 4 and np.max(np.abs(observed[:4] - positive_box)) <= 1e-3


def fork_runner(parent, ocsort_module):
    """Copy only mutable tracker/policy state; preserve the shared source module."""
    tracker = copy.deepcopy(parent.tracker)
    policy = copy.deepcopy(parent.policy)
    fork = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
    fork.source_frame_id = parent.source_frame_id
    return fork


def score_branches(gt: dict, anchors: dict[int, int], target_gt_id: int,
                   baseline: list[list[Candidate]], oracle: list[list[Candidate]],
                   first_frame: int) -> dict:
    scores = {"baseline": Counter(), "oracle": Counter()}
    for offset, (a, b) in enumerate(zip(baseline, oracle)):
        frame = first_frame + offset
        for name, output in (("baseline", a), ("oracle", b)):
            matches = hungarian_iou_matches(gt.get(frame, {}), output, .5)[0]
            for gid, tid in anchors.items():
                if gid in gt.get(frame, {}) and matches.get(gid, (None, 0.0))[0] == tid:
                    scores[name][gid] += 1
    deltas = {gid: scores["oracle"][gid] - scores["baseline"][gid] for gid in anchors}
    return {"target_correct_frames_baseline": scores["baseline"][target_gt_id],
            "target_correct_frames_oracle": scores["oracle"][target_gt_id],
            "target_delta": deltas[target_gt_id],
            "all_anchor_net_delta": sum(deltas.values()),
            "other_anchor_harmed": any(delta < 0 for gid, delta in deltas.items()
                                       if gid != target_gt_id),
            "other_anchor_harmed_count": sum(delta < 0 for gid, delta in deltas.items()
                                             if gid != target_gt_id),
            "anchor_count": len(anchors)}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E4 requires CUDA_VISIBLE_DEVICES='' and CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    source = protocol["source"]
    for path, expected in ((Path(source["e3_fit_summary"]), source["e3_fit_summary_sha256"]),
                           (ROOT / source["e3_protocol"], source["e3_protocol_sha256"]),
                           (ROOT / source["fit_split_manifest"], source["fit_split_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
    e3 = json.loads(Path(source["e3_fit_summary"]).read_text())
    split = yaml.safe_load((ROOT / source["fit_split_manifest"]).read_text())
    sequences = split["split"]["fit_sequences"]
    if e3["status"] != "PASS" or e3["partition"] != "fit" or \
            [item["sequence"] for item in e3["sequences"]] != sequences:
        raise RuntimeError("E3 fit source incomplete or split changed")
    inp = protocol["inputs"]
    commit = subprocess.run(["git", "-C", str(ROOT / inp["oc_sort_root"]), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if commit != inp["oc_sort_commit"]:
        raise RuntimeError("OC-SORT commit changed")
    output = Path(protocol["outputs"]["run_dir"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(p.exists() for p in (output, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir(parents=True)
    start = time.monotonic()
    cpu_start = time.process_time()
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL),
                           "E3_fit_summary_sha256": source["e3_fit_summary_sha256"],
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        sys.path.insert(0, str((ROOT / inp["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort
        seq_summaries = []
        for seq, e3_record in zip(sequences, e3["sequences"]):
            seq_dir = Path(inp["train_root"]) / seq
            cache_file = Path(inp["detection_root"]) / f"{seq}.csv"
            info_file = seq_dir / "seqinfo.ini"
            gt_file = seq_dir / "gt/gt.txt"
            if (sha256_file(cache_file) != e3_record["cache_sha256"] or
                    sha256_file(info_file) != e3_record["seqinfo_sha256"] or
                    sha256_file(gt_file) != e3_record["gt_sha256"]):
                raise RuntimeError(f"{seq}: E3 input hash mismatch")
            c_path = Path(source["e3_fit_summary"]).parent / "trackers/C" / f"{seq}.txt"
            e_path = Path(source["e3_fit_summary"]).parent / "trackers/E" / f"{seq}.txt"
            for path, cond in ((c_path, "C"), (e_path, "E")):
                if sha256_file(path) != e3_record["tracker_sha256"][cond]:
                    raise RuntimeError(f"{seq}: E3 {cond} output hash mismatch")
            gt = read_gt(gt_file)
            c_ref, e_ref = read_tracker(c_path), read_tracker(e_path)
            det_for_gt = read_detections(cache_file, .6)
            config = configparser.ConfigParser()
            if not config.read(info_file):
                raise FileNotFoundError(info_file)
            info = config["Sequence"]
            length, height, width = info.getint("seqLength"), info.getint("imHeight"), info.getint("imWidth")
            fps = info.getfloat("frameRate")
            selected, all_candidate_count = build_candidates(
                seq, gt, c_ref, det_for_gt, length,
                protocol["scope"]["maximum_candidate_episodes_per_video"])
            by_frame = defaultdict(list)
            for item in selected:
                by_frame[item["frame"]].append(item)
            cached, det_ids = read_detection_cache(cache_file)
            scale = min(inp["input_size"][0] / height, inp["input_size"][1] / width)

            def frame_input(frame: int):
                values = np.asarray(cached.get(frame, []), dtype=np.float64)
                if values.size == 0:
                    return None
                result = values.reshape(-1, 5).copy()
                result[:, :4] *= scale
                return result

            tracker = ocsort.OCSort(**inp["tracker"])
            history = TrackerDetectionHistory(history_length=inp["cv_history"]["length"],
                                              score_threshold=.6)
            policy = NaturalMotionFirstRoundPolicy(
                mode="constant_velocity", history=history, image_width=width,
                image_height=height, fps=fps,
                minimum_observations=inp["cv_history"]["minimum_observations"],
                maximum_frame_gap=inp["cv_history"]["maximum_frame_gap"])
            parent = NaturalHistoryTrackerRunner(tracker, ocsort, policy)
            parent_mature = set()
            counts = Counter()
            event_path = partial / f"{seq}.jsonl"
            with event_path.open("w") as handle:
                for frame in range(1, length + 1):
                    frame_values = frame_input(frame)
                    for event in by_frame.get(frame, []):
                        counts["selected"] += 1
                        target_id = event["track_id"]
                        if target_id not in parent_mature:
                            counts["excluded_not_mature"] += 1
                            continue
                        target_before = next((t for t in parent.tracker.trackers
                                              if int(t.id) + 1 == target_id), None)
                        if target_before is None:
                            counts["excluded_not_alive"] += 1
                            continue
                        high_ids = [int(did) for did, values in zip(det_ids.get(frame, []), cached.get(frame, []))
                                    if float(values[4]) > .6]
                        if event["positive_det_id"] not in high_ids:
                            raise RuntimeError("positive detection missing from high pool")
                        det_index = high_ids.index(event["positive_det_id"])
                        positive = np.asarray([v for v, did in zip(cached[frame], det_ids[frame])
                                               if int(did) == event["positive_det_id"]][0][:4], dtype=np.float64)
                        pre_state = pickle.dumps((parent.tracker, parent.policy.history,
                                                  parent.source_frame_id), protocol=5)
                        pre_count = ocsort.KalmanBoxTracker.count
                        before_hits = int(target_before.hits)
                        baseline = fork_runner(parent, ocsort)
                        baseline_mature = set(parent_mature)
                        capture = CapturePolicy(baseline.policy)
                        baseline.adapter.policy = capture
                        a_c, a_e = step_e(baseline, baseline_mature, frame_values, frame, height, width, inp)
                        baseline.adapter.policy = baseline.policy
                        baseline_count = ocsort.KalmanBoxTracker.count
                        if capture.called != 1 or target_id not in capture.tracker_ids:
                            counts["excluded_not_first_round_live"] += 1
                            ocsort.KalmanBoxTracker.count = pre_count
                            continue
                        if det_index >= len(capture.detections) or \
                                np.max(np.abs(capture.detections[det_index, :4] - positive)) > 1e-3:
                            raise RuntimeError("positive detection/request index mismatch")
                        if target_positive_update(baseline, target_id, before_hits, positive):
                            counts["excluded_already_correctly_updated"] += 1
                            ocsort.KalmanBoxTracker.count = pre_count
                            continue
                        ocsort.KalmanBoxTracker.count = pre_count
                        oracle = fork_runner(parent, ocsort)
                        oracle_mature = set(parent_mature)
                        force = ForcePairPolicy(oracle.policy, target_id, det_index)
                        oracle.adapter.policy = force
                        b_c, b_e = step_e(oracle, oracle_mature, frame_values, frame, height, width, inp)
                        oracle.adapter.policy = oracle.policy
                        oracle_count = ocsort.KalmanBoxTracker.count
                        if force.called != 1 or force.forced is None or not target_positive_update(
                                oracle, target_id, before_hits, positive):
                            raise RuntimeError("forced pair did not update target with positive detection")
                        if not same_outputs(a_c, c_ref.get(frame, [])) or \
                                not same_outputs(a_e, e_ref.get(frame, [])):
                            raise RuntimeError("baseline fork does not replay E3 C/E at event frame")
                        base_outputs, oracle_outputs = [a_e], [b_e]
                        for future in range(frame + 1, frame + 5):
                            # OC-SORT's birth ID counter is module-global, not stored
                            # on the copied tracker. Keep each branch's counter separate.
                            ocsort.KalmanBoxTracker.count = baseline_count
                            _, a = step_e(baseline, baseline_mature, frame_input(future),
                                          future, height, width, inp)
                            baseline_count = ocsort.KalmanBoxTracker.count
                            ocsort.KalmanBoxTracker.count = oracle_count
                            _, b = step_e(oracle, oracle_mature, frame_input(future),
                                          future, height, width, inp)
                            oracle_count = ocsort.KalmanBoxTracker.count
                            if not same_outputs(a, e_ref.get(future, [])):
                                raise RuntimeError(f"baseline fork does not replay E3 E at {seq}:{future}")
                            base_outputs.append(a)
                            oracle_outputs.append(b)
                        ocsort.KalmanBoxTracker.count = pre_count
                        if pickle.dumps((parent.tracker, parent.policy.history,
                                         parent.source_frame_id), protocol=5) != pre_state:
                            raise RuntimeError("parent changed during fork")
                        anchors = hungarian_iou_matches(
                            gt.get(event["anchor_frame"], {}),
                            c_ref.get(event["anchor_frame"], []), .5)[0]
                        anchor_map = {gid: int(pair[0]) for gid, pair in anchors.items()}
                        if anchor_map.get(event["gt_id"]) != target_id:
                            raise RuntimeError("target anchor changed")
                        score = score_branches(gt, anchor_map, event["gt_id"],
                                               base_outputs, oracle_outputs, frame)
                        baseline_first = hungarian_iou_matches(gt.get(frame, {}), a_e, .5)[0]
                        oracle_first = hungarian_iou_matches(gt.get(frame, {}), b_e, .5)[0]
                        result = {**event, "score": score,
                                  "baseline_first_correct": baseline_first.get(event["gt_id"], (None,))[0] == target_id,
                                  "oracle_first_correct": oracle_first.get(event["gt_id"], (None,))[0] == target_id,
                                  "checks": {"forced_pair_once": True, "parent_isolated": True,
                                             "baseline_E3_replay": True}}
                        handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                        counts["eligible_failures"] += 1
                        counts["target_gain_events"] += score["target_delta"] > 0
                        counts["target_harm_events"] += score["target_delta"] < 0
                        counts["target_zero_events"] += score["target_delta"] == 0
                        counts["target_correct_frame_delta"] += score["target_delta"]
                        counts["all_anchor_net_delta"] += score["all_anchor_net_delta"]
                        counts["other_anchor_harm_events"] += score["other_anchor_harmed"]
                        counts["baseline_first_correct"] += result["baseline_first_correct"]
                        counts["oracle_first_correct"] += result["oracle_first_correct"]
                    c_now, e_now = step_e(parent, parent_mature, frame_values, frame, height, width, inp)
                    if not same_outputs(c_now, c_ref.get(frame, [])) or \
                            not same_outputs(e_now, e_ref.get(frame, [])):
                        raise RuntimeError(f"parent C/E replay mismatch {seq}:{frame}")
                    if time.monotonic() - start > protocol["budget"]["max_wall_seconds"]:
                        raise TimeoutError("E4 wall budget exceeded")
            if counts["selected"] != len(selected):
                raise RuntimeError("selected event coverage mismatch")
            record = {"sequence": seq, "frames": length,
                      "candidate_episodes": all_candidate_count,
                      "selected_episodes": len(selected),
                      "counts": dict(counts),
                      "event_rows_sha256": sha256_file(event_path),
                      "C_reference_sha256": e3_record["tracker_sha256"]["C"],
                      "E_reference_sha256": e3_record["tracker_sha256"]["E"],
                      "GT_sha256": e3_record["gt_sha256"],
                      "cache_sha256": e3_record["cache_sha256"]}
            seq_summaries.append(record)
            (partial / "sequence_progress.json").write_text(json.dumps(seq_summaries, indent=2) + "\n")
            print(f"{seq}: candidates={all_candidate_count} selected={len(selected)} eligible_failures={counts['eligible_failures']} target_delta={counts['target_correct_frame_delta']}", flush=True)
        verify_code_unchanged(ROOT, provenance)
        total = Counter()
        for item in seq_summaries:
            total["candidate_episodes"] += item["candidate_episodes"]
            total["selected_episodes"] += item["selected_episodes"]
            total.update(item["counts"])
        n = total["eligible_failures"]
        gate = protocol["decision"]["sufficient_headroom_all_required"]
        passed = (n >= gate["minimum_eligible_failures"] and
                  total["target_correct_frame_delta"] >= gate["minimum_target_correct_frame_gain_per_event"] * n and
                  total["target_harm_events"] <= gate["maximum_target_harmed_event_fraction"] * n and
                  total["other_anchor_harm_events"] <= gate["maximum_other_anchor_harmed_event_fraction"] * n and
                  total["all_anchor_net_delta"] >= gate["minimum_all_anchor_net_correct_frame_gain_per_event"] * n)
        core_hours = (time.process_time() - cpu_start) / 3600
        if core_hours > protocol["budget"]["max_cpu_core_hours"]:
            raise RuntimeError("E4 CPU budget exceeded")
        summary = {"status": "PASS", "protocol_sha256": provenance["protocol_sha256"],
                   "E3_fit_summary_sha256": source["e3_fit_summary_sha256"],
                   "sequences": seq_summaries, "totals": dict(total),
                   "headroom_gate_pass": passed,
                   "target_gain_per_event": total["target_correct_frame_delta"] / n if n else None,
                   "all_anchor_net_gain_per_event": total["all_anchor_net_delta"] / n if n else None,
                   "wall_seconds": time.monotonic() - start, "cpu_core_hours": core_hours,
                   "peak_rss_kib": peak_rss_kib(),
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "totals": dict(total),
                          "headroom_gate_pass": passed, "output": str(output)}, indent=2), flush=True)
    except BaseException as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
