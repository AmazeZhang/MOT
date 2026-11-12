#!/usr/bin/env python3
"""Run the frozen D4 direction ablation or D5 simultaneous D2+D3 diagnostic."""

from __future__ import annotations

import argparse
import configparser
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from mot_research.analysis.d2_first_association import (
    association_maps,
    compare_tracker_replay,
    evaluate_association_counterfactual,
)
from mot_research.analysis.failure_events import (
    hungarian_iou_matches,
    read_detections,
    read_gt,
    sha256_file,
)


RESULT_COLUMNS = [
    "event_id", "sequence", "outcome", "candidate_class", "diagnostic",
    "focus_still_negative_separate_D2_D3", "D2_positive_match", "D3_positive_match",
    "D2_recoverable", "D3_recoverable", "after_frame", "gt_id", "intended_track_id",
    "positive_detection_id", "status", "non_applicable_reason", "live_tracks",
    "first_round_detections", "baseline_positive_detection_track_id",
    "intervention_positive_detection_track_id", "baseline_positive_match",
    "intervention_positive_match", "recoverable", "harmed", "track_xyxy_max_abs_change",
    "detection_xyxy_max_abs_change", "baseline_intended_positive_iou",
    "intervention_intended_positive_iou", "baseline_intended_direction_cost",
    "intervention_intended_direction_cost", "baseline_intended_total_score",
    "intervention_intended_total_score", "intended_total_score_change",
    "baseline_intended_gate_eligible", "intervention_intended_gate_eligible",
    "baseline_best_eligible_other_track_id", "baseline_best_eligible_other_score",
    "baseline_intended_minus_best_eligible_other",
    "intervention_best_eligible_other_track_id", "intervention_best_eligible_other_score",
    "intervention_intended_minus_best_eligible_other", "baseline_eligible_other_track_count",
    "intervention_eligible_other_track_count", "baseline_one_to_one_iou_shortcut",
    "intervention_one_to_one_iou_shortcut", "association_pairs_changed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", choices=("direction_ablation", "joint_d2_d3"), required=True)
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--baseline-tracker-root", type=Path, required=True)
    parser.add_argument("--parent-events", type=Path, required=True)
    parser.add_argument("--parent-events-sha256", required=True)
    parser.add_argument("--contention-events", type=Path, required=True)
    parser.add_argument("--contention-events-sha256", required=True)
    parser.add_argument("--d2-events", type=Path, required=True)
    parser.add_argument("--d2-events-sha256", required=True)
    parser.add_argument("--d3-events", type=Path, required=True)
    parser.add_argument("--d3-events-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument("--replay-atol", type=float, default=1e-4)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def association_scores(
    association_module: object,
    detections: np.ndarray,
    trackers: np.ndarray,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    vdc_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the upstream solver's IoU, direction term, and total score matrices."""
    iou = association_module.iou_batch(detections, trackers)
    if len(detections) == 0 or len(trackers) == 0:
        return iou, np.zeros_like(iou), iou.copy()
    direction_y, direction_x = association_module.speed_direction_batch(
        detections, previous_obs
    )
    inertia_y = np.repeat(velocities[:, 0, None], direction_y.shape[1], axis=1)
    inertia_x = np.repeat(velocities[:, 1, None], direction_x.shape[1], axis=1)
    cosine = np.clip(inertia_x * direction_x + inertia_y * direction_y, -1, 1)
    angle = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
    valid = np.repeat(
        (previous_obs[:, 4] >= 0).astype(float)[:, None], direction_x.shape[1], axis=1
    )
    scores = np.repeat(detections[:, -1, None], trackers.shape[0], axis=1)
    direction = (valid * angle).T * vdc_weight * scores
    return iou, direction, iou + direction


def uses_shortcut(iou: np.ndarray, threshold: float) -> bool:
    if min(iou.shape, default=0) == 0:
        return False
    adjacency = (iou > threshold).astype(np.int32)
    return bool(adjacency.sum(1).max() == 1 and adjacency.sum(0).max() == 1)


def read_tracker_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    keys: list[list[int]] = []
    boxes: list[list[float]] = []
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 columns")
            keys.append([int(row[0]), int(row[1])])
            x, y, width, height = map(float, row[2:6])
            boxes.append([x, y, x + width, y + height])
    return np.asarray(keys, dtype=np.int64), np.asarray(boxes, dtype=np.float64)


def grouped_counts(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row[key])
        groups[group][str(row["status"])] += 1
        if row["status"] != "applicable":
            continue
        groups[group]["baseline_positive_match"] += int(bool(row["baseline_positive_match"]))
        groups[group]["intervention_positive_match"] += int(
            bool(row["intervention_positive_match"])
        )
        groups[group]["recoverable"] += int(bool(row["recoverable"]))
        groups[group]["harmed"] += int(bool(row["harmed"]))
        groups[group]["pair_set_changed"] += int(int(row["association_pairs_changed"]) > 0)
        delta = float(row["intended_total_score_change"])
        groups[group]["score_improved"] += int(delta > 1e-12)
        groups[group]["score_unchanged"] += int(abs(delta) <= 1e-12)
        groups[group]["score_worsened"] += int(delta < -1e-12)
        groups[group]["gate_gained"] += int(
            not bool(row["baseline_intended_gate_eligible"])
            and bool(row["intervention_intended_gate_eligible"])
        )
        groups[group]["gate_lost"] += int(
            bool(row["baseline_intended_gate_eligible"])
            and not bool(row["intervention_intended_gate_eligible"])
        )
    return {group: dict(counts) for group, counts in sorted(groups.items())}


def assignment_runtime() -> dict[str, object]:
    try:
        import lap  # noqa: F401

        return {
            "backend": "lap.lapjv",
            "lap_version": importlib.metadata.version("lap"),
        }
    except ImportError:
        return {
            "backend": "scipy.optimize.linear_sum_assignment",
            "scipy_version": scipy.__version__,
        }


def best_other_fields(
    prefix: str,
    iou: np.ndarray,
    total: np.ndarray,
    positive_index: int,
    target_index: int,
    tracker_ids: list[int],
    threshold: float,
    target_score: float,
) -> dict[str, object]:
    eligible = [
        index for index in range(len(tracker_ids))
        if index != target_index and iou[positive_index, index] >= threshold
    ]
    fields: dict[str, object] = {f"{prefix}_eligible_other_track_count": len(eligible)}
    if eligible:
        best = max(eligible, key=lambda index: float(total[positive_index, index]))
        best_score = float(total[positive_index, best])
        fields.update({
            f"{prefix}_best_eligible_other_track_id": tracker_ids[best],
            f"{prefix}_best_eligible_other_score": best_score,
            f"{prefix}_intended_minus_best_eligible_other": target_score - best_score,
        })
    return fields


def main() -> None:
    args = parse_args()
    inputs = {
        args.parent_events: args.parent_events_sha256,
        args.contention_events: args.contention_events_sha256,
        args.d2_events: args.d2_events_sha256,
        args.d3_events: args.d3_events_sha256,
        args.protocol: args.protocol_sha256,
    }
    for path, expected in inputs.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")

    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    failed = args.run_dir.with_name(args.run_dir.name + ".failed")
    if args.run_dir.exists() or partial.exists() or failed.exists():
        raise FileExistsError(f"refusing to overwrite output near {args.run_dir}")
    partial.mkdir(parents=True)
    try:
        parent = [
            row for row in csv.DictReader(args.parent_events.open(newline=""))
            if row["outcome"] in {"identity_change_failure", "success_control"}
        ]
        if len(parent) != 165:
            raise RuntimeError(f"expected 165 parent events, got {len(parent)}")
        contention = {
            row["event_id"]: row
            for row in csv.DictReader(args.contention_events.open(newline=""))
        }
        d2 = {row["event_id"]: row for row in csv.DictReader(args.d2_events.open(newline=""))}
        d3 = {row["event_id"]: row for row in csv.DictReader(args.d3_events.open(newline=""))}
        parent_ids = {row["event_id"] for row in parent}
        if parent_ids != set(d2) or parent_ids != set(d3):
            raise RuntimeError("parent, D2, and D3 event ID sets differ")

        by_sequence_frame: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in parent:
            by_sequence_frame[event["sequence"]][int(event["after_frame"])].append(event)

        sys.path.insert(0, str(args.oc_sort_root.resolve()))
        import trackers.ocsort_tracker.association as association_module
        import trackers.ocsort_tracker.ocsort as ocsort_module

        original_associate = ocsort_module.associate
        tracker_args = {
            "det_thresh": 0.6,
            "iou_threshold": 0.3,
            "asso_func": "iou",
            "delta_t": 3,
            "inertia": 0.2,
            "use_byte": False,
        }
        results: list[dict[str, object]] = []
        sequence_replays: list[dict[str, object]] = []
        input_hashes: dict[str, object] = {"sequences": {}}

        for sequence in sorted(by_sequence_frame):
            sequence_dir = args.train_root / sequence
            seqinfo_path = sequence_dir / "seqinfo.ini"
            config = configparser.ConfigParser()
            if not config.read(seqinfo_path):
                raise FileNotFoundError(seqinfo_path)
            info = config["Sequence"]
            sequence_length = info.getint("seqLength")
            image_height, image_width = info.getint("imHeight"), info.getint("imWidth")
            scale = min(args.input_height / image_height, args.input_width / image_width)
            cache_path = args.detection_root / f"{sequence}.csv"
            baseline_path = args.baseline_tracker_root / f"{sequence}.txt"
            gt_path = sequence_dir / "gt" / "gt.txt"
            cache_all: dict[int, list[list[float]]] = defaultdict(list)
            with cache_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    cache_all[int(row["frame"])].append([
                        float(row["x1"]), float(row["y1"]), float(row["x2"]),
                        float(row["y2"]), float(row["score"]),
                    ])
            high_candidates = read_detections(cache_path, 0.6)
            gt = read_gt(gt_path)
            tracker = ocsort_module.OCSort(**tracker_args)
            current_frame = 0

            def capturing_associate(
                detections: np.ndarray,
                trackers: np.ndarray,
                iou_threshold: float,
                velocities: np.ndarray,
                previous_obs: np.ndarray,
                vdc_weight: float,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                baseline_result = original_associate(
                    detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight
                )
                frame_events = by_sequence_frame[sequence].get(current_frame, [])
                if not frame_events:
                    return baseline_result
                candidates = high_candidates.get(current_frame, [])
                detection_ids = [item.candidate_id for item in candidates]
                if len(candidates) != len(detections):
                    raise RuntimeError(f"{sequence} frame {current_frame}: candidate count mismatch")
                if candidates:
                    candidate_values = np.stack(
                        [np.r_[item.box_xyxy, item.score] for item in candidates]
                    )
                    if not np.allclose(candidate_values, detections, atol=1e-7, rtol=0):
                        raise RuntimeError(f"{sequence} frame {current_frame}: candidate mismatch")
                tracker_ids = [item.id + 1 for item in tracker.trackers]
                if len(tracker_ids) != len(trackers):
                    raise RuntimeError(f"{sequence} frame {current_frame}: tracker alignment mismatch")
                tracker_index = {track_id: index for index, track_id in enumerate(tracker_ids)}
                detection_index = {det_id: index for index, det_id in enumerate(detection_ids)}
                gt_matches, _ = hungarian_iou_matches(gt[current_frame], candidates, 0.5)
                baseline_iou, baseline_direction, baseline_total = association_scores(
                    association_module, detections, trackers, velocities, previous_obs, vdc_weight
                )
                for event in frame_events:
                    event_id = event["event_id"]
                    gt_id = int(event["gt_id"])
                    intended_track_id = int(event["before_tracker_id"])
                    positive = gt_matches.get(gt_id)
                    d2_positive = parse_bool(d2[event_id]["oracle_positive_match"])
                    d3_positive = parse_bool(d3[event_id]["D3_GT_detection_positive_match"])
                    d2_recoverable = parse_bool(d2[event_id]["recoverable"])
                    d3_recoverable = parse_bool(d3[event_id]["recoverable"])
                    baseline_parent = parse_bool(d2[event_id]["baseline_positive_match"])
                    if baseline_parent != parse_bool(d3[event_id]["baseline_positive_match"]):
                        raise RuntimeError(f"{event_id}: D2/D3 baseline mismatch")
                    focus = (
                        event["outcome"] == "identity_change_failure"
                        and not baseline_parent and not d2_positive and not d3_positive
                    )
                    row: dict[str, object] = {
                        "event_id": event_id,
                        "sequence": sequence,
                        "outcome": event["outcome"],
                        "candidate_class": contention[event_id]["event_candidate_class"],
                        "diagnostic": args.diagnostic,
                        "focus_still_negative_separate_D2_D3": focus,
                        "D2_positive_match": d2_positive,
                        "D3_positive_match": d3_positive,
                        "D2_recoverable": d2_recoverable,
                        "D3_recoverable": d3_recoverable,
                        "after_frame": current_frame,
                        "gt_id": gt_id,
                        "intended_track_id": intended_track_id,
                        "live_tracks": len(tracker_ids),
                        "first_round_detections": len(detection_ids),
                    }
                    reason = ""
                    if positive is None:
                        reason = "positive_detection_mapping_missing"
                    elif intended_track_id not in tracker_index:
                        reason = "intended_history_track_not_alive"
                    if reason:
                        row.update({"status": "not_applicable", "non_applicable_reason": reason})
                        results.append(row)
                        continue

                    positive_detection_id, _ = positive
                    positive_index = detection_index[positive_detection_id]
                    target_index = tracker_index[intended_track_id]
                    intervention_detections = detections.copy()
                    intervention_trackers = trackers.copy()
                    intervention_weight = vdc_weight
                    if args.diagnostic == "direction_ablation":
                        intervention_weight = 0.0
                    else:
                        target_gt = np.asarray(gt[current_frame][gt_id], dtype=np.float64)
                        intervention_trackers[target_index, :4] = target_gt
                        intervention_detections[positive_index, :4] = target_gt
                        if not np.array_equal(
                            intervention_trackers[target_index, :4],
                            intervention_detections[positive_index, :4],
                        ):
                            raise RuntimeError(f"{event_id}: joint target pair coordinates differ")
                        if intervention_detections[positive_index, 4] != detections[positive_index, 4]:
                            raise RuntimeError(f"{event_id}: joint intervention changed score")

                    intervention_result = original_associate(
                        intervention_detections,
                        intervention_trackers,
                        iou_threshold,
                        velocities,
                        previous_obs,
                        intervention_weight,
                    )
                    intervention_iou, intervention_direction, intervention_total = association_scores(
                        association_module,
                        intervention_detections,
                        intervention_trackers,
                        velocities,
                        previous_obs,
                        intervention_weight,
                    )
                    evaluated = evaluate_association_counterfactual(
                        intended_track_id,
                        positive_detection_id,
                        tracker_ids,
                        detection_ids,
                        baseline_result[0],
                        intervention_result[0],
                    )
                    if bool(evaluated["baseline_positive_match"]) != baseline_parent:
                        raise RuntimeError(f"{event_id}: replay baseline mismatch with D2/D3")
                    _, baseline_d2t = association_maps(
                        baseline_result[0], tracker_ids, detection_ids
                    )
                    _, intervention_d2t = association_maps(
                        intervention_result[0], tracker_ids, detection_ids
                    )
                    baseline_target_score = float(baseline_total[positive_index, target_index])
                    intervention_target_score = float(
                        intervention_total[positive_index, target_index]
                    )
                    row.update({
                        "positive_detection_id": positive_detection_id,
                        "status": "applicable",
                        "non_applicable_reason": "",
                        "baseline_positive_detection_track_id": baseline_d2t.get(
                            positive_detection_id
                        ),
                        "intervention_positive_detection_track_id": intervention_d2t.get(
                            positive_detection_id
                        ),
                        "baseline_positive_match": evaluated["baseline_positive_match"],
                        "intervention_positive_match": evaluated["oracle_positive_match"],
                        "recoverable": evaluated["recoverable"],
                        "harmed": evaluated["harmed"],
                        "track_xyxy_max_abs_change": float(np.max(np.abs(
                            intervention_trackers[target_index, :4] - trackers[target_index, :4]
                        ))),
                        "detection_xyxy_max_abs_change": float(np.max(np.abs(
                            intervention_detections[positive_index, :4]
                            - detections[positive_index, :4]
                        ))),
                        "baseline_intended_positive_iou": float(
                            baseline_iou[positive_index, target_index]
                        ),
                        "intervention_intended_positive_iou": float(
                            intervention_iou[positive_index, target_index]
                        ),
                        "baseline_intended_direction_cost": float(
                            baseline_direction[positive_index, target_index]
                        ),
                        "intervention_intended_direction_cost": float(
                            intervention_direction[positive_index, target_index]
                        ),
                        "baseline_intended_total_score": baseline_target_score,
                        "intervention_intended_total_score": intervention_target_score,
                        "intended_total_score_change": (
                            intervention_target_score - baseline_target_score
                        ),
                        "baseline_intended_gate_eligible": bool(
                            baseline_iou[positive_index, target_index] >= iou_threshold
                        ),
                        "intervention_intended_gate_eligible": bool(
                            intervention_iou[positive_index, target_index] >= iou_threshold
                        ),
                        "baseline_one_to_one_iou_shortcut": uses_shortcut(
                            baseline_iou, iou_threshold
                        ),
                        "intervention_one_to_one_iou_shortcut": uses_shortcut(
                            intervention_iou, iou_threshold
                        ),
                        "association_pairs_changed": evaluated["association_pairs_changed"],
                    })
                    row.update(best_other_fields(
                        "baseline", baseline_iou, baseline_total, positive_index, target_index,
                        tracker_ids, iou_threshold, baseline_target_score,
                    ))
                    row.update(best_other_fields(
                        "intervention", intervention_iou, intervention_total, positive_index,
                        target_index, tracker_ids, iou_threshold, intervention_target_score,
                    ))
                    results.append(row)
                return baseline_result

            ocsort_module.associate = capturing_associate
            replay_keys: list[list[int]] = []
            replay_boxes: list[list[float]] = []
            try:
                for frame_id in range(1, sequence_length + 1):
                    current_frame = frame_id
                    values = np.asarray(cache_all.get(frame_id, []), dtype=np.float64)
                    tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                    if tracker_input is not None:
                        tracker_input[:, :4] *= scale
                    tracks = tracker.update(
                        tracker_input,
                        (image_height, image_width),
                        (args.input_height, args.input_width),
                    )
                    for x1, y1, x2, y2, track_id in tracks:
                        replay_keys.append([frame_id, int(track_id)])
                        replay_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            finally:
                ocsort_module.associate = original_associate

            reference_keys, reference_boxes = read_tracker_rows(baseline_path)
            comparison = compare_tracker_replay(
                reference_keys,
                reference_boxes,
                np.asarray(replay_keys, dtype=np.int64),
                np.asarray(replay_boxes, dtype=np.float64),
                args.replay_atol,
            )
            if comparison["status"] != "PASS":
                raise RuntimeError(f"baseline replay failed for {sequence}: {comparison}")
            sequence_replays.append({"sequence": sequence, **comparison})
            input_hashes["sequences"][sequence] = {
                "seqinfo": sha256_file(seqinfo_path),
                "gt": sha256_file(gt_path),
                "detections": sha256_file(cache_path),
                "baseline_tracker": sha256_file(baseline_path),
            }

        if len(results) != 165 or len({str(row["event_id"]) for row in results}) != 165:
            raise RuntimeError("result coverage mismatch")
        if sum(bool(row["focus_still_negative_separate_D2_D3"]) for row in results) != 21:
            raise RuntimeError("focused D2/D3-negative event coverage is not 21")
        if any(row["status"] != "applicable" for row in results):
            raise RuntimeError("frozen diagnostics require all 165 events to be applicable")
        if args.diagnostic == "direction_ablation":
            if any(float(row["track_xyxy_max_abs_change"]) != 0.0 for row in results):
                raise RuntimeError("direction ablation changed a track coordinate")
            if any(float(row["detection_xyxy_max_abs_change"]) != 0.0 for row in results):
                raise RuntimeError("direction ablation changed a detection coordinate")
        else:
            if any(abs(float(row["intervention_intended_positive_iou"]) - 1.0) > 1e-12 for row in results):
                raise RuntimeError("joint intervention did not produce unit target-pair IoU")

        results.sort(key=lambda row: str(row["event_id"]))
        result_path = partial / "event_counterfactual.csv"
        with result_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(results)

        applicable = [row for row in results if row["status"] == "applicable"]
        failures = [row for row in applicable if row["outcome"] == "identity_change_failure"]
        focused = [
            row for row in applicable if row["focus_still_negative_separate_D2_D3"]
        ]
        joint_recovery = {str(row["event_id"]) for row in failures if row["recoverable"]}
        d2_recovery = {str(row["event_id"]) for row in failures if row["D2_recoverable"]}
        d3_recovery = {str(row["event_id"]) for row in failures if row["D3_recoverable"]}
        summary = {
            "status": "PASS",
            "diagnostic": args.diagnostic,
            "events": len(results),
            "applicable": len(applicable),
            "focused_still_negative_separate_D2_D3": len(focused),
            "baseline_positive_match": sum(bool(row["baseline_positive_match"]) for row in applicable),
            "intervention_positive_match": sum(
                bool(row["intervention_positive_match"]) for row in applicable
            ),
            "recoverable": sum(bool(row["recoverable"]) for row in applicable),
            "harmed": sum(bool(row["harmed"]) for row in applicable),
            "by_outcome": grouped_counts(results, "outcome"),
            "by_candidate_class": grouped_counts(results, "candidate_class"),
            "by_focus_still_negative_separate_D2_D3": grouped_counts(
                results, "focus_still_negative_separate_D2_D3"
            ),
            "failure_recovery_sets": {
                "D2": len(d2_recovery),
                "D3": len(d3_recovery),
                "current_diagnostic": len(joint_recovery),
                "current_intersection_D2": len(joint_recovery & d2_recovery),
                "current_intersection_D3": len(joint_recovery & d3_recovery),
                "current_new_beyond_separate_D2_D3": len(
                    joint_recovery - (d2_recovery | d3_recovery)
                ),
                "union_D2_D3_current": len(joint_recovery | d2_recovery | d3_recovery),
            },
            "baseline_one_to_one_shortcut_events": sum(
                bool(row["baseline_one_to_one_iou_shortcut"]) for row in applicable
            ),
            "intervention_one_to_one_shortcut_events": sum(
                bool(row["intervention_one_to_one_iou_shortcut"]) for row in applicable
            ),
            "sequence_replays": sequence_replays,
            "assignment": assignment_runtime(),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "result_file": result_path.name,
            "input_hashes_file": "input_hashes.json",
            "runtime_file": "runtime.json",
            "limitations": [
                "All event labels and positive-detection mappings are privileged GT diagnostics.",
                "Counterfactual decisions are not written back into tracker state.",
                "First-round assignments are not closed-loop recovery or official tracking metrics.",
                "The selected four train videos do not establish validation generalization.",
            ],
        }
        with (partial / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

        repo_root = Path(__file__).resolve().parents[1]
        input_hashes.update({
            "parent_events": sha256_file(args.parent_events),
            "contention_events": sha256_file(args.contention_events),
            "d2_events": sha256_file(args.d2_events),
            "d3_events": sha256_file(args.d3_events),
            "protocol": sha256_file(args.protocol),
            "analysis_script": sha256_file(Path(__file__)),
            "d2_helper": sha256_file(repo_root / "src/mot_research/analysis/d2_first_association.py"),
            "failure_event_helper": sha256_file(
                repo_root / "src/mot_research/analysis/failure_events.py"
            ),
            "upstream_ocsort": sha256_file(args.oc_sort_root / "trackers/ocsort_tracker/ocsort.py"),
            "upstream_association": sha256_file(
                args.oc_sort_root / "trackers/ocsort_tracker/association.py"
            ),
        })
        with (partial / "input_hashes.json").open("w") as handle:
            json.dump(input_hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")

        upstream_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.oc_sort_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        runtime = {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "assignment": assignment_runtime(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_policy": "CPU-only diagnostic; physical GPU 0 forbidden and no CUDA device exposed",
            "upstream_oc_sort_commit": upstream_commit,
            "cwd": str(Path.cwd()),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (partial / "runtime.json").open("w") as handle:
            json.dump(runtime, handle, indent=2)
            handle.write("\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        if partial.exists():
            failure = {
                "status": "FAIL",
                "diagnostic": args.diagnostic,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            with (partial / "failure.json").open("w") as handle:
                json.dump(failure, handle, indent=2)
                handle.write("\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
