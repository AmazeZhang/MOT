#!/usr/bin/env python3
"""Read-only, GT-labelled diagnostic over frozen S01 natural tracker traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from mot_research.analysis.failure_events import (
    hungarian_iou_matches, read_detections, read_gt,
)
from mot_research.association.appearance_core import normalize_features

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("G", "L", "M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch: {path}: {actual} != {expected}")


def stable_anchor(history: list[tuple[int, int | None]], frame: int) -> int | None:
    if len(history) < 3 or frame - history[-1][0] > 30:
        return None
    labels = [label for _, label in history[-3:]]
    return labels[0] if labels[0] is not None and labels.count(labels[0]) == 3 else None


def feature_map(path: Path) -> dict[tuple[int, int], np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        keys = saved["keys"]
        values = saved["features"]
        valid = saved["valid"]
        if keys.ndim != 2 or keys.shape[1] != 2 or values.shape != (len(keys), 2048):
            raise RuntimeError(f"invalid S01 feature shape: {path}")
        if not np.all(valid):
            raise RuntimeError(f"invalid features present: {path}")
        return {(int(key[0]), int(key[1])): value.astype(np.float64)
                for key, value in zip(keys, values)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    value, ok = normalize_features(b[None, :])
    if not ok[0]:
        raise RuntimeError("S01 supposedly valid feature is zero")
    return float(np.clip(np.dot(a, value[0]), -1, 1))


def evaluate_future(event: dict, traces: list[dict], positives: dict[int, dict[int, int]],
                    labels: dict[int, dict[int, int]], last_frame: int) -> dict:
    tid, owner, onset = event["track_id"], event["anchor_gt_id"], event["frame"]
    opportunities = wrong_positive = wrong_labeled = 0
    first_return = None
    reason = "horizon_end"
    for frame in range(onset + 1, min(onset + 30, last_frame) + 1):
        trace = traces[frame - 1]
        if "memory_last_write_frames" not in trace:
            # S01's empty-detection path records no pre-association request.
            # It cannot establish liveness, and supplies no positive candidate.
            continue
        if tid not in trace["requested_tracker_ids"]:
            reason = "track_not_live"
            break
        positive = owner in positives[frame]
        opportunities += int(positive)
        update = next((did for track_id, did, _ in trace["final_actual_updates"]
                       if track_id == tid), None)
        if update is None:
            continue
        label = labels[frame].get(update)
        if label == owner:
            first_return = frame - onset
            reason = "returned_to_anchor"
            break
        if label is not None:
            wrong_labeled += 1
            wrong_positive += int(positive)
    return {"future_opportunity_frames": opportunities,
            "future_wrong_labeled_updates": wrong_labeled,
            "future_wrong_updates_with_owner_positive": wrong_positive,
            "first_return_gap_frames": first_return,
            "end_reason": reason}


def analyze_condition(sequence: str, condition: str, traces: list[dict],
                      positives: dict[int, dict[int, int]],
                      labels: dict[int, dict[int, int]],
                      features: dict[tuple[int, int], np.ndarray]) -> tuple[dict, list[dict]]:
    history: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    memories: dict[int, np.ndarray] = {}
    first_onset_tracks: set[int] = set()
    events: list[dict] = []
    counts = Counter()
    for frame, trace in enumerate(traces, start=1):
        if trace["frame"] != frame or trace["sequence"] != sequence or trace["condition"] != condition:
            raise RuntimeError(f"trace identity/order changed: {sequence}/{condition}/{frame}")
        live = trace["requested_tracker_ids"]
        anchor = {tid: stable_anchor(history[tid], frame) for tid in live}
        if condition != "G" and "memory_last_write_frames" in trace:
            written = trace["memory_last_write_frames"]
            if len(written) != len(live):
                raise RuntimeError("memory trace length mismatch")
            for tid, recorded in zip(live, written):
                expected = history[tid][-1][0] if history[tid] else -1
                if recorded != expected:
                    raise RuntimeError(f"memory last-write mismatch: {sequence}/{condition}/{frame}/{tid}")
        base = {tuple(pair) for pair in trace["base_first_pairs"]}
        actual = {tuple(pair) for pair in trace["first_pairs"]}
        if base != actual:
            counts["changed_first_round_frames"] += 1
            def score(pairs: set[tuple[int, int]]) -> tuple[int, int]:
                correct = wrong = 0
                for did, tid in pairs:
                    target, label = anchor.get(tid), labels[frame].get(did)
                    if target is not None and label is not None:
                        correct += int(target == label)
                        wrong += int(target != label)
                return correct, wrong
            base_correct, base_wrong = score(base)
            actual_correct, actual_wrong = score(actual)
            delta = actual_correct - base_correct
            counts["changed_pair_correct_delta_sum"] += delta
            counts["changed_pair_better_frames"] += int(delta > 0)
            counts["changed_pair_worse_frames"] += int(delta < 0)
            counts["changed_pair_equal_frames"] += int(delta == 0)
            counts["changed_pair_base_wrong"] += base_wrong
            counts["changed_pair_actual_wrong"] += actual_wrong
        for tid, did, stage in trace["final_actual_updates"]:
            counts["actual_updates"] += 1
            label = labels[frame].get(did)
            counts["GT_labeled_updates"] += int(label is not None)
            target = anchor.get(tid)
            if target is not None and stage != "birth":
                counts["stable_anchor_updates"] += 1
                if label is None:
                    counts["stable_anchor_unlabelled_updates"] += 1
                elif label == target:
                    counts["stable_anchor_clean_updates"] += 1
                elif label != target:
                    counts["foreign_GT_updates_after_anchor"] += 1
                    if tid not in first_onset_tracks:
                        first_onset_tracks.add(tid)
                        positive = positives[frame].get(target)
                        competitor = any(other != tid and owner == label
                                         for other, owner in anchor.items())
                        event = {"sequence": sequence, "condition": condition,
                            "frame": frame, "track_id": tid, "anchor_gt_id": target,
                            "foreign_gt_id": label, "selected_det_id": did,
                            "owner_positive_det_id": positive,
                            "foreign_competitor": competitor,
                            "actionable_competition": positive is not None and competitor,
                            "first_round_pair_changed": base != actual,
                            "memory_cosine_margin_owner_minus_wrong": None}
                        if condition != "G" and positive is not None and tid in memories:
                            memory = memories[tid]
                            event["memory_cosine_margin_owner_minus_wrong"] = (
                                cosine(memory, features[(frame, positive)]) -
                                cosine(memory, features[(frame, did)]))
                        events.append(event)
            history[tid].append((frame, label))
            history[tid] = history[tid][-3:]
            if condition != "G":
                value, valid = normalize_features(features[(frame, did)][None, :])
                if not valid[0]:
                    raise RuntimeError("invalid feature during memory replay")
                if condition == "M" and tid in memories:
                    value, valid = normalize_features((.9 * memories[tid] + .1 * value[0])[None, :])
                    if not valid[0]:
                        raise RuntimeError("invalid EMA memory")
                memories[tid] = value[0]
    for event in events:
        event.update(evaluate_future(event, traces, positives, labels, len(traces)))
        counts["first_onsets"] += 1
        if event["actionable_competition"]:
            counts["actionable_competition_first_onsets"] += 1
            counts["actionable_with_future_wrong_and_owner_positive"] += int(
                event["future_wrong_updates_with_owner_positive"] > 0)
            margin = event["memory_cosine_margin_owner_minus_wrong"]
            if margin is not None:
                counts["actionable_memory_margin_positive"] += int(margin > 0)
                counts["actionable_memory_margin_nonpositive"] += int(margin <= 0)
        counts["owner_positive_first_onsets"] += int(event["owner_positive_det_id"] is not None)
        counts["foreign_competitor_first_onsets"] += int(event["foreign_competitor"])
    return dict(counts), events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S02 diagnostic requires CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    protocol_path = ROOT / lock["protocol_path"]
    if (protocol_path.parent != ROOT / "manifests/protocols" or
            lock["protocol_sha256"] != sha256(protocol_path) or
            lock["analysis_code_sha256"] != sha256(Path(__file__))):
        raise RuntimeError("S02 protocol or analysis code changed after lock")
    for rel_path, expected in lock["dependency_code_sha256"].items():
        require_hash(ROOT / rel_path, expected)
    expected_versions = lock["environment_versions"]
    actual_versions = {"python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "pyyaml": yaml.__version__}
    if actual_versions != expected_versions:
        raise RuntimeError(f"S02 dependency versions changed: {actual_versions}")
    protocol = yaml.safe_load(protocol_path.read_text())
    inp = protocol["inputs"]
    for path_key, hash_key in (("s01_protocol", "s01_protocol_sha256"),
                               ("s01_summary", "s01_summary_sha256"),
                               ("frozen_inputs", "frozen_inputs_sha256"),
                               ("s01_runtime_lock", "s01_runtime_lock_sha256")):
        path = Path(inp[path_key]); path = path if path.is_absolute() else ROOT / path
        require_hash(path, inp[hash_key])
    frozen = json.loads(Path(inp["frozen_inputs"]).read_text())
    if lock["sequences"] != frozen["sequences"] or lock["output_root"] != protocol["execution"]["output_root"]:
        raise RuntimeError("S02 sequence list or output root differs from lock")
    artifact = {(a["sequence"], a["condition"], a["kind"]): a for a in frozen["artifacts"]}
    feature_artifact = {a["sequence"]: a for a in frozen["feature_artifacts"]}
    check_artifact = {a["sequence"]: a for a in frozen["engineering_checks"]}
    output = Path(lock["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_record = {"status": "source_artifacts_frozen_before_S02_GT_read",
                    "protocol_sha256": sha256(protocol_path),
                    "runtime_lock_sha256": sha256(args.runtime_lock),
                    "s01_frozen_inputs_sha256": sha256(Path(inp["frozen_inputs"])),
                    "sequences": frozen["sequences"], "artifacts": []}
    for sequence in frozen["sequences"]:
        check = check_artifact[sequence]
        check_path = Path(check["path"])
        require_hash(check_path, check["sha256"])
        tracking_lock = check_path.parent / "runtime_lock.json"
        require_hash(tracking_lock, check["runtime_lock_sha256"])
        tlock = json.loads(tracking_lock.read_text())
        detection = Path(tlock["detection_root"]) / f"{sequence}.csv"
        require_hash(detection, tlock["input_detection_sha256"][sequence])
        feature = Path(feature_artifact[sequence]["path"])
        require_hash(feature, feature_artifact[sequence]["sha256"])
        input_record["artifacts"].extend([
            {"path": str(detection), "sha256": sha256(detection)},
            {"path": str(feature), "sha256": sha256(feature)},
            {"path": str(tracking_lock), "sha256": sha256(tracking_lock)}])
        for condition in CONDITIONS:
            a = artifact[(sequence, condition, "traces")]
            path = Path(a["path"])
            require_hash(path, a["sha256"])
            input_record["artifacts"].append({"path": str(path), "sha256": a["sha256"]})
    (output / "frozen_inputs.json").write_text(json.dumps(input_record, indent=2) + "\n")
    s01_lock = json.loads((ROOT / inp["s01_runtime_lock"]).read_text())
    per_video = []
    totals = {condition: Counter() for condition in CONDITIONS}
    with (output / "events.jsonl").open("w") as event_handle, \
            (output / "per_video.jsonl").open("w") as video_handle:
        for sequence in frozen["sequences"]:
            gt_path = Path(yaml.safe_load((ROOT / inp["s01_protocol"]).read_text())["inputs"]["train_root"]) / sequence / "gt/gt.txt"
            require_hash(gt_path, s01_lock["gt_file_sha256"][sequence])
            gt = read_gt(gt_path)
            tracking_lock = Path(check_artifact[sequence]["path"]).parent / "runtime_lock.json"
            tlock = json.loads(tracking_lock.read_text())
            detections = read_detections(Path(tlock["detection_root"]) / f"{sequence}.csv", .6)
            positives, labels = {}, {}
            reference_trace = Path(artifact[(sequence, "M", "traces")]["path"])
            frames = sum(1 for _ in reference_trace.open())
            for frame in range(1, frames + 1):
                match, _ = hungarian_iou_matches(gt.get(frame, {}), detections.get(frame, []), .5)
                positives[frame] = {gid: did for gid, (did, _) in match.items()}
                labels[frame] = {did: gid for gid, did in positives[frame].items()}
            features = feature_map(Path(feature_artifact[sequence]["path"]))
            row = {"sequence": sequence, "frames": frames, "conditions": {}}
            for condition in CONDITIONS:
                trace_path = Path(artifact[(sequence, condition, "traces")]["path"])
                traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
                if len(traces) != frames:
                    raise RuntimeError(f"trace frame count mismatch: {sequence}/{condition}")
                counts, events = analyze_condition(sequence, condition, traces, positives, labels, features)
                row["conditions"][condition] = counts
                totals[condition].update(counts)
                for event in events:
                    event_handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            per_video.append(row)
            video_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "per_video.json").write_text(json.dumps(per_video, indent=2) + "\n")
    gate = protocol["decision"]["minimum_observational_support_to_plan_controlled_memory_test"]
    m = totals["M"]
    videos = sum(row["conditions"]["M"].get("actionable_competition_first_onsets", 0) > 0 for row in per_video)
    support = (m["actionable_competition_first_onsets"] >= gate["actionable_competition_first_onsets"]
        and videos >= gate["videos_with_actionable_competition"]
        and m["actionable_with_future_wrong_and_owner_positive"] >=
            gate["actionable_onsets_with_later_wrong_update_and_owner_positive"])
    summary = {"status": "COMPLETE_DIAGNOSTIC", "scope": "reused_train_fit_read_only",
        "protocol_sha256": sha256(protocol_path), "runtime_lock_sha256": sha256(args.runtime_lock),
        "frozen_inputs_sha256": sha256(output / "frozen_inputs.json"),
        "events_sha256": sha256(output / "events.jsonl"),
        "per_video_sha256": sha256(output / "per_video.json"),
        "sequences": frozen["sequences"], "videos": len(per_video),
        "frames": sum(row["frames"] for row in per_video),
        "totals": {condition: dict(value) for condition, value in totals.items()},
        "M_videos_with_actionable_competition": videos,
        "decision_gate": {"minimums": gate, "observational_support_met": support,
            "action": "plan_controlled_test_only" if support else "defer_candidate_A_training"},
        "limitations": ["IoU-0.5 GT assignment is a diagnostic identity proxy, not TrackEval events",
            "observational persistence does not establish that a memory write caused later errors",
            "condition-specific track IDs are not paired after trajectories diverge",
            "reused train-fit data and appearance training overlap may bias interpretation"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "frames": summary["frames"],
        "totals": summary["totals"], "decision_gate": summary["decision_gate"],
        "summary_sha256": sha256(output / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
