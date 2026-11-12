#!/usr/bin/env python3
"""Evaluate privileged correction of one existing positive detection's XYXY."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy

from mot_research.analysis.d2_first_association import (
    association_maps,
    compare_tracker_replay,
    evaluate_association_counterfactual,
)
from mot_research.analysis.d3_detection_coordinate import replace_detection_xyxy
from mot_research.analysis.failure_events import (
    hungarian_iou_matches,
    pairwise_iou,
    read_detections,
    read_gt,
    sha256_file,
)


RESULT_COLUMNS = [
    "event_id", "sequence", "outcome", "candidate_class", "D2_oracle_non_recovery",
    "after_frame", "gt_id", "intended_track_id", "positive_detection_id", "status",
    "non_applicable_reason", "live_tracks", "first_round_detections",
    "baseline_positive_detection_track_id", "D3_positive_detection_track_id",
    "baseline_positive_match", "D3_GT_detection_positive_match", "recoverable", "harmed",
    "original_detection_target_GT_iou", "detection_xyxy_max_abs_change",
    "baseline_intended_positive_iou", "D3_intended_positive_iou",
    "baseline_intended_positive_total_score", "D3_intended_positive_total_score",
    "intended_total_score_change", "baseline_best_eligible_other_track_id",
    "baseline_best_eligible_other_score", "baseline_intended_minus_best_eligible_other",
    "D3_best_eligible_other_track_id", "D3_best_eligible_other_score",
    "D3_intended_minus_best_eligible_other", "baseline_intended_gate_eligible",
    "D3_intended_gate_eligible", "baseline_eligible_other_track_count",
    "D3_eligible_other_track_count", "baseline_one_to_one_iou_shortcut",
    "D3_one_to_one_iou_shortcut", "association_pairs_changed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument("--replay-atol", type=float, default=1e-4)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def association_scores(
    association_module: object,
    detections: np.ndarray,
    trackers: np.ndarray,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    vdc_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    iou = association_module.iou_batch(detections, trackers)
    if len(detections) == 0 or len(trackers) == 0:
        return iou, iou.copy()
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
    direction_cost = (valid * angle).T * vdc_weight * scores
    return iou, iou + direction_cost


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
        delta = float(row["intended_total_score_change"])
        groups[group]["baseline_positive_match"] += int(bool(row["baseline_positive_match"]))
        groups[group]["D3_GT_detection_positive_match"] += int(
            bool(row["D3_GT_detection_positive_match"])
        )
        groups[group]["recoverable"] += int(bool(row["recoverable"]))
        groups[group]["harmed"] += int(bool(row["harmed"]))
        groups[group]["score_improved"] += int(delta > 1e-12)
        groups[group]["score_unchanged"] += int(abs(delta) <= 1e-12)
        groups[group]["score_worsened"] += int(delta < -1e-12)
        groups[group]["gate_gained"] += int(
            not bool(row["baseline_intended_gate_eligible"])
            and bool(row["D3_intended_gate_eligible"])
        )
        groups[group]["gate_lost"] += int(
            bool(row["baseline_intended_gate_eligible"])
            and not bool(row["D3_intended_gate_eligible"])
        )
        groups[group]["pair_set_changed"] += int(int(row["association_pairs_changed"]) > 0)
    return {group: dict(counts) for group, counts in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    for path, expected in {
        args.parent_events: args.parent_events_sha256,
        args.contention_events: args.contention_events_sha256,
        args.d2_events: args.d2_events_sha256,
        args.protocol: args.protocol_sha256,
    }.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    if args.run_dir.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
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
        d2 = {
            row["event_id"]: row
            for row in csv.DictReader(args.d2_events.open(newline=""))
        }
        parent_ids = {row["event_id"] for row in parent}
        if parent_ids != set(d2):
            raise RuntimeError("D3 parent and D2 event IDs differ")
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
                tracker_index = {track_id: index for index, track_id in enumerate(tracker_ids)}
                detection_index = {det_id: index for index, det_id in enumerate(detection_ids)}
                gt_matches, _ = hungarian_iou_matches(gt[current_frame], candidates, 0.5)
                baseline_iou, baseline_total = association_scores(
                    association_module,
                    detections,
                    trackers,
                    velocities,
                    previous_obs,
                    vdc_weight,
                )
                for event in frame_events:
                    event_id = event["event_id"]
                    gt_id = int(event["gt_id"])
                    intended_track_id = int(event["before_tracker_id"])
                    positive = gt_matches.get(gt_id)
                    row: dict[str, object] = {
                        "event_id": event_id,
                        "sequence": sequence,
                        "outcome": event["outcome"],
                        "candidate_class": contention[event_id]["event_candidate_class"],
                        "D2_oracle_non_recovery": event["outcome"] == "identity_change_failure"
                        and not parse_bool(d2[event_id]["oracle_positive_match"]),
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
                    positive_detection_id, original_gt_iou = positive
                    positive_index = detection_index[positive_detection_id]
                    target_index = tracker_index[intended_track_id]
                    corrected_detections = replace_detection_xyxy(
                        detections, positive_index, gt[current_frame][gt_id]
                    )
                    if corrected_detections[positive_index, 4] != detections[positive_index, 4]:
                        raise RuntimeError(f"{event_id}: D3 changed detection score")
                    D3_result = original_associate(
                        corrected_detections,
                        trackers,
                        iou_threshold,
                        velocities,
                        previous_obs,
                        vdc_weight,
                    )
                    D3_iou, D3_total = association_scores(
                        association_module,
                        corrected_detections,
                        trackers,
                        velocities,
                        previous_obs,
                        vdc_weight,
                    )
                    evaluated = evaluate_association_counterfactual(
                        intended_track_id,
                        positive_detection_id,
                        tracker_ids,
                        detection_ids,
                        baseline_result[0],
                        D3_result[0],
                    )
                    d2_row = d2[event_id]
                    if bool(evaluated["baseline_positive_match"]) != parse_bool(
                        d2_row["baseline_positive_match"]
                    ):
                        raise RuntimeError(f"{event_id}: baseline mismatch with D2")
                    baseline_t2d, baseline_d2t = association_maps(
                        baseline_result[0], tracker_ids, detection_ids
                    )
                    D3_t2d, D3_d2t = association_maps(
                        D3_result[0], tracker_ids, detection_ids
                    )
                    del baseline_t2d, D3_t2d
                    baseline_target_score = float(
                        baseline_total[positive_index, target_index]
                    )
                    D3_target_score = float(D3_total[positive_index, target_index])
                    baseline_other = [
                        index for index in range(len(tracker_ids))
                        if index != target_index
                        and baseline_iou[positive_index, index] >= iou_threshold
                    ]
                    D3_other = [
                        index for index in range(len(tracker_ids))
                        if index != target_index
                        and D3_iou[positive_index, index] >= iou_threshold
                    ]
                    baseline_best = max(
                        baseline_other,
                        key=lambda index: float(baseline_total[positive_index, index]),
                    ) if baseline_other else None
                    D3_best = max(
                        D3_other,
                        key=lambda index: float(D3_total[positive_index, index]),
                    ) if D3_other else None
                    row.update({
                        "positive_detection_id": positive_detection_id,
                        "status": "applicable",
                        "non_applicable_reason": "",
                        "baseline_positive_detection_track_id": baseline_d2t.get(positive_detection_id),
                        "D3_positive_detection_track_id": D3_d2t.get(positive_detection_id),
                        "baseline_positive_match": evaluated["baseline_positive_match"],
                        "D3_GT_detection_positive_match": evaluated["oracle_positive_match"],
                        "recoverable": evaluated["recoverable"],
                        "harmed": evaluated["harmed"],
                        "original_detection_target_GT_iou": float(original_gt_iou),
                        "detection_xyxy_max_abs_change": float(np.max(np.abs(
                            corrected_detections[positive_index, :4]
                            - detections[positive_index, :4]
                        ))),
                        "baseline_intended_positive_iou": float(
                            baseline_iou[positive_index, target_index]
                        ),
                        "D3_intended_positive_iou": float(D3_iou[positive_index, target_index]),
                        "baseline_intended_positive_total_score": baseline_target_score,
                        "D3_intended_positive_total_score": D3_target_score,
                        "intended_total_score_change": D3_target_score - baseline_target_score,
                        "baseline_intended_gate_eligible": bool(
                            baseline_iou[positive_index, target_index] >= iou_threshold
                        ),
                        "D3_intended_gate_eligible": bool(
                            D3_iou[positive_index, target_index] >= iou_threshold
                        ),
                        "baseline_eligible_other_track_count": len(baseline_other),
                        "D3_eligible_other_track_count": len(D3_other),
                        "baseline_one_to_one_iou_shortcut": uses_shortcut(
                            baseline_iou, iou_threshold
                        ),
                        "D3_one_to_one_iou_shortcut": uses_shortcut(D3_iou, iou_threshold),
                        "association_pairs_changed": evaluated["association_pairs_changed"],
                    })
                    if baseline_best is not None:
                        score = float(baseline_total[positive_index, baseline_best])
                        row.update({
                            "baseline_best_eligible_other_track_id": tracker_ids[baseline_best],
                            "baseline_best_eligible_other_score": score,
                            "baseline_intended_minus_best_eligible_other": baseline_target_score - score,
                        })
                    if D3_best is not None:
                        score = float(D3_total[positive_index, D3_best])
                        row.update({
                            "D3_best_eligible_other_track_id": tracker_ids[D3_best],
                            "D3_best_eligible_other_score": score,
                            "D3_intended_minus_best_eligible_other": D3_target_score - score,
                        })
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
            raise RuntimeError("D3 result coverage mismatch")
        results.sort(key=lambda row: str(row["event_id"]))
        result_path = partial / "event_d3_detection_coordinate.csv"
        with result_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(results)
        applicable = [row for row in results if row["status"] == "applicable"]
        summary = {
            "status": "PASS",
            "scope": "DanceTrack train-only privileged D3 existing-detection coordinate correction",
            "events": len(results),
            "applicable": len(applicable),
            "not_applicable": len(results) - len(applicable),
            "baseline_positive_match": sum(bool(row["baseline_positive_match"]) for row in applicable),
            "D3_GT_detection_positive_match": sum(
                bool(row["D3_GT_detection_positive_match"]) for row in applicable
            ),
            "recoverable": sum(bool(row["recoverable"]) for row in applicable),
            "harmed": sum(bool(row["harmed"]) for row in applicable),
            "by_outcome": grouped_counts(results, "outcome"),
            "by_candidate_class": grouped_counts(results, "candidate_class"),
            "by_D2_oracle_non_recovery": grouped_counts(results, "D2_oracle_non_recovery"),
            "baseline_one_to_one_shortcut_events": sum(
                bool(row["baseline_one_to_one_iou_shortcut"]) for row in applicable
            ),
            "D3_one_to_one_shortcut_events": sum(
                bool(row["D3_one_to_one_iou_shortcut"]) for row in applicable
            ),
            "sequence_replays": sequence_replays,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "parent_events_sha256": args.parent_events_sha256,
            "contention_events_sha256": args.contention_events_sha256,
            "d2_events_sha256": args.d2_events_sha256,
            "result_file": result_path.name,
            "input_hashes_file": "input_hashes.json",
            "runtime_file": "runtime.json",
            "limitations": [
                "GT detection coordinates are privileged and unavailable at inference.",
                "Only an existing positive detection is corrected; no detection is added.",
                "Independent first-round counterfactuals are not written back.",
                "Pairwise margins are descriptive under global Hungarian assignment.",
                "No validation data is used.",
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
            "protocol": sha256_file(args.protocol),
            "analysis_script": sha256_file(Path(__file__)),
            "d3_helper": sha256_file(repo_root / "src/mot_research/analysis/d3_detection_coordinate.py"),
            "d2_helper": sha256_file(repo_root / "src/mot_research/analysis/d2_first_association.py"),
            "failure_event_helper": sha256_file(repo_root / "src/mot_research/analysis/failure_events.py"),
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
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "upstream_oc_sort_commit": upstream_commit,
            "cwd": str(Path.cwd()),
        }
        with (partial / "runtime.json").open("w") as handle:
            json.dump(runtime, handle, indent=2)
            handle.write("\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(summary, indent=2))
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


if __name__ == "__main__":
    main()
