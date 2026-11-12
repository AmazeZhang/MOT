#!/usr/bin/env python3
"""Complete the frozen D1 v2 OC-SORT association-score margin measurements."""

from __future__ import annotations

import argparse
import configparser
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
    "event_id", "sequence", "outcome", "candidate_class", "D2_oracle_non_recovery",
    "after_frame", "gt_id", "intended_track_id", "positive_detection_id",
    "baseline_positive_detection_track_id", "clean_positive_detection_track_id",
    "baseline_positive_match", "D1_clean_history_positive_match", "recoverable", "harmed",
    "baseline_intended_positive_iou", "clean_intended_positive_iou",
    "baseline_intended_positive_total_score", "clean_intended_positive_total_score",
    "intended_total_score_change", "best_other_track_id_all",
    "best_other_positive_total_score_all", "baseline_intended_minus_best_other_all",
    "clean_intended_minus_best_other_all", "best_eligible_other_track_id",
    "best_eligible_other_positive_total_score", "baseline_intended_minus_best_eligible_other",
    "clean_intended_minus_best_eligible_other", "baseline_intended_gate_eligible",
    "clean_intended_gate_eligible", "eligible_other_track_count",
    "baseline_one_to_one_iou_shortcut", "clean_one_to_one_iou_shortcut",
    "association_pairs_changed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--baseline-tracker-root", type=Path, required=True)
    parser.add_argument("--parent-events", type=Path, required=True)
    parser.add_argument("--parent-events-sha256", required=True)
    parser.add_argument("--parent-d1-events", type=Path, required=True)
    parser.add_argument("--parent-d1-events-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument("--replay-atol", type=float, default=1e-4)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def association_scores(
    association_module: object,
    detections: np.ndarray,
    trackers: np.ndarray,
    velocities: np.ndarray,
    previous_obs: np.ndarray,
    vdc_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce upstream first-round IoU and raw IoU+direction score matrices."""
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
    detection_scores = np.repeat(
        detections[:, -1, None], trackers.shape[0], axis=1
    )
    direction_cost = (valid * angle).T * vdc_weight * detection_scores
    return iou, iou + direction_cost


def uses_one_to_one_shortcut(iou: np.ndarray, threshold: float) -> bool:
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


def assert_parent_reproduction(
    parent: dict[str, str],
    evaluated: dict[str, object],
    baseline_iou: float,
    clean_iou: float,
    prediction_change: float,
) -> None:
    comparisons = {
        "baseline_positive_match": bool(evaluated["baseline_positive_match"]),
        "D1_clean_history_positive_match": bool(evaluated["oracle_positive_match"]),
        "recoverable": bool(evaluated["recoverable"]),
        "harmed": bool(evaluated["harmed"]),
    }
    for key, actual in comparisons.items():
        if actual != parse_bool(parent[key]):
            raise RuntimeError(f"{parent['event_id']}: parent mismatch for {key}")
    if int(evaluated["association_pairs_changed"]) != int(parent["association_pairs_changed"]):
        raise RuntimeError(f"{parent['event_id']}: association-pair mismatch")
    float_checks = [
        (baseline_iou, float(parent["baseline_positive_iou"]), 1e-12, "baseline IoU"),
        (clean_iou, float(parent["D1_clean_history_positive_iou"]), 1e-12, "clean IoU"),
        (
            prediction_change,
            float(parent["prediction_xyxy_max_abs_change"]),
            1e-9,
            "prediction change",
        ),
    ]
    for actual, expected, tolerance, label in float_checks:
        if abs(actual - expected) > tolerance:
            raise RuntimeError(
                f"{parent['event_id']}: {label} mismatch {actual} vs {expected}"
            )
    baseline_owner = optional_int(parent["baseline_positive_detection_track_id"])
    clean_owner = optional_int(parent["oracle_positive_detection_track_id"])
    if evaluated["baseline_positive_detection_track_id"] != baseline_owner:
        raise RuntimeError(f"{parent['event_id']}: baseline positive owner mismatch")
    if evaluated["oracle_positive_detection_track_id"] != clean_owner:
        raise RuntimeError(f"{parent['event_id']}: clean positive owner mismatch")


def grouped_summary(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row[key])
        delta = float(row["intended_total_score_change"])
        groups[group]["events"] += 1
        groups[group]["score_improved"] += int(delta > 1e-12)
        groups[group]["score_unchanged"] += int(abs(delta) <= 1e-12)
        groups[group]["score_worsened"] += int(delta < -1e-12)
        groups[group]["gate_gained"] += int(
            not bool(row["baseline_intended_gate_eligible"])
            and bool(row["clean_intended_gate_eligible"])
        )
        groups[group]["gate_lost"] += int(
            bool(row["baseline_intended_gate_eligible"])
            and not bool(row["clean_intended_gate_eligible"])
        )
        groups[group]["pair_set_changed"] += int(int(row["association_pairs_changed"]) > 0)
    return {group: dict(counts) for group, counts in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    for path, expected in {
        args.parent_events: args.parent_events_sha256,
        args.parent_d1_events: args.parent_d1_events_sha256,
        args.protocol: args.protocol_sha256,
    }.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}")
    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    if args.run_dir.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
    partial.mkdir(parents=True)
    try:
        parent_rows = list(csv.DictReader(args.parent_d1_events.open(newline="")))
        if len(parent_rows) != 165 or len({row["event_id"] for row in parent_rows}) != 165:
            raise RuntimeError("expected 165 unique parent D1 rows")
        original_parent_ids = {
            row["event_id"]
            for row in csv.DictReader(args.parent_events.open(newline=""))
            if row["outcome"] in {"identity_change_failure", "success_control"}
        }
        if original_parent_ids != {row["event_id"] for row in parent_rows}:
            raise RuntimeError("parent event population mismatch")
        by_sequence_after: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_sequence_before: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_sequence_gap: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in parent_rows:
            sequence = row["sequence"]
            by_sequence_after[sequence][int(row["after_frame"])].append(row)
            by_sequence_before[sequence][int(row["before_frame"])].append(row)
            for frame in range(int(row["gap_start_frame"]), int(row["gap_end_frame"]) + 1):
                by_sequence_gap[sequence][frame].append(row)

        sys.path.insert(0, str(args.oc_sort_root.resolve()))
        import trackers.ocsort_tracker.association as association_module
        import trackers.ocsort_tracker.ocsort as ocsort_module

        original_associate = ocsort_module.associate
        results: list[dict[str, object]] = []
        sequence_replays: list[dict[str, object]] = []
        input_hashes: dict[str, object] = {"sequences": {}}
        tracker_args = {
            "det_thresh": 0.6,
            "iou_threshold": 0.3,
            "asso_func": "iou",
            "delta_t": 3,
            "inertia": 0.2,
            "use_byte": False,
        }

        for sequence in sorted(by_sequence_after):
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
            clean_states: dict[str, object] = {}
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
                frame_events = by_sequence_after[sequence].get(current_frame, [])
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
                gt_detection_matches, _ = hungarian_iou_matches(
                    gt[current_frame], candidates, 0.5
                )
                baseline_iou, baseline_total = association_scores(
                    association_module,
                    detections,
                    trackers,
                    velocities,
                    previous_obs,
                    vdc_weight,
                )
                for parent in frame_events:
                    event_id = parent["event_id"]
                    gt_id = int(parent["gt_id"])
                    intended_track_id = int(parent["intended_track_id"])
                    positive = gt_detection_matches.get(gt_id)
                    if positive is None or positive[0] != int(parent["positive_detection_id"]):
                        raise RuntimeError(f"{event_id}: positive detection reproduction mismatch")
                    if intended_track_id not in tracker_index or event_id not in clean_states:
                        raise RuntimeError(f"{event_id}: target/clean state missing")
                    positive_detection_id = positive[0]
                    positive_index = detection_index[positive_detection_id]
                    target_index = tracker_index[intended_track_id]
                    clean = copy.deepcopy(clean_states[event_id])
                    clean_prediction = clean.predict()[0]
                    clean_trackers = trackers.copy()
                    clean_velocities = velocities.copy()
                    clean_previous_obs = previous_obs.copy()
                    clean_trackers[target_index, :4] = clean_prediction
                    clean_velocities[target_index] = (
                        clean.velocity if clean.velocity is not None else np.asarray((0, 0))
                    )
                    clean_previous_obs[target_index] = ocsort_module.k_previous_obs(
                        clean.observations, clean.age, clean.delta_t
                    )
                    clean_result = original_associate(
                        detections,
                        clean_trackers,
                        iou_threshold,
                        clean_velocities,
                        clean_previous_obs,
                        vdc_weight,
                    )
                    clean_iou, clean_total = association_scores(
                        association_module,
                        detections,
                        clean_trackers,
                        clean_velocities,
                        clean_previous_obs,
                        vdc_weight,
                    )
                    evaluated = evaluate_association_counterfactual(
                        intended_track_id,
                        positive_detection_id,
                        tracker_ids,
                        detection_ids,
                        baseline_result[0],
                        clean_result[0],
                    )
                    baseline_positive_iou = float(baseline_iou[positive_index, target_index])
                    clean_positive_iou = float(clean_iou[positive_index, target_index])
                    prediction_change = float(np.max(np.abs(
                        trackers[target_index, :4] - clean_trackers[target_index, :4]
                    )))
                    assert_parent_reproduction(
                        parent,
                        evaluated,
                        baseline_positive_iou,
                        clean_positive_iou,
                        prediction_change,
                    )
                    other_indices = [
                        index for index in range(len(tracker_ids)) if index != target_index
                    ]
                    best_other_index = max(
                        other_indices,
                        key=lambda index: float(baseline_total[positive_index, index]),
                    ) if other_indices else None
                    eligible_other_indices = [
                        index for index in other_indices
                        if baseline_iou[positive_index, index] >= iou_threshold
                    ]
                    best_eligible_index = max(
                        eligible_other_indices,
                        key=lambda index: float(baseline_total[positive_index, index]),
                    ) if eligible_other_indices else None
                    baseline_target_score = float(
                        baseline_total[positive_index, target_index]
                    )
                    clean_target_score = float(clean_total[positive_index, target_index])
                    baseline_t2d, baseline_d2t = association_maps(
                        baseline_result[0], tracker_ids, detection_ids
                    )
                    clean_t2d, clean_d2t = association_maps(
                        clean_result[0], tracker_ids, detection_ids
                    )
                    del baseline_t2d, clean_t2d
                    row: dict[str, object] = {
                        "event_id": event_id,
                        "sequence": sequence,
                        "outcome": parent["outcome"],
                        "candidate_class": parent["candidate_class"],
                        "D2_oracle_non_recovery": parse_bool(parent["D2_oracle_non_recovery"]),
                        "after_frame": current_frame,
                        "gt_id": gt_id,
                        "intended_track_id": intended_track_id,
                        "positive_detection_id": positive_detection_id,
                        "baseline_positive_detection_track_id": baseline_d2t.get(positive_detection_id),
                        "clean_positive_detection_track_id": clean_d2t.get(positive_detection_id),
                        "baseline_positive_match": evaluated["baseline_positive_match"],
                        "D1_clean_history_positive_match": evaluated["oracle_positive_match"],
                        "recoverable": evaluated["recoverable"],
                        "harmed": evaluated["harmed"],
                        "baseline_intended_positive_iou": baseline_positive_iou,
                        "clean_intended_positive_iou": clean_positive_iou,
                        "baseline_intended_positive_total_score": baseline_target_score,
                        "clean_intended_positive_total_score": clean_target_score,
                        "intended_total_score_change": clean_target_score - baseline_target_score,
                        "baseline_intended_gate_eligible": baseline_positive_iou >= iou_threshold,
                        "clean_intended_gate_eligible": clean_positive_iou >= iou_threshold,
                        "eligible_other_track_count": len(eligible_other_indices),
                        "baseline_one_to_one_iou_shortcut": uses_one_to_one_shortcut(
                            baseline_iou, iou_threshold
                        ),
                        "clean_one_to_one_iou_shortcut": uses_one_to_one_shortcut(
                            clean_iou, iou_threshold
                        ),
                        "association_pairs_changed": evaluated["association_pairs_changed"],
                    }
                    if best_other_index is not None:
                        best_other_score = float(
                            baseline_total[positive_index, best_other_index]
                        )
                        row.update({
                            "best_other_track_id_all": tracker_ids[best_other_index],
                            "best_other_positive_total_score_all": best_other_score,
                            "baseline_intended_minus_best_other_all": baseline_target_score - best_other_score,
                            "clean_intended_minus_best_other_all": clean_target_score - best_other_score,
                        })
                    if best_eligible_index is not None:
                        best_eligible_score = float(
                            baseline_total[positive_index, best_eligible_index]
                        )
                        row.update({
                            "best_eligible_other_track_id": tracker_ids[best_eligible_index],
                            "best_eligible_other_positive_total_score": best_eligible_score,
                            "baseline_intended_minus_best_eligible_other": baseline_target_score - best_eligible_score,
                            "clean_intended_minus_best_eligible_other": clean_target_score - best_eligible_score,
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
                    live_by_id = {item.id + 1: item for item in tracker.trackers}
                    for event in by_sequence_before[sequence].get(frame_id, []):
                        track_id = int(event["intended_track_id"])
                        if track_id not in live_by_id:
                            raise RuntimeError(f"{event['event_id']}: anchor track missing")
                        clean_states[event["event_id"]] = copy.deepcopy(live_by_id[track_id])
                    for event in by_sequence_gap[sequence].get(frame_id, []):
                        state = clean_states.get(event["event_id"])
                        if state is None:
                            raise RuntimeError(f"{event['event_id']}: clean state missing in gap")
                        state.predict()
                        state.update(None)
                    for x1, y1, x2, y2, track_id in tracks:
                        replay_keys.append([frame_id, int(track_id)])
                        replay_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            finally:
                ocsort_module.associate = original_associate
            reference_keys, reference_boxes = read_tracker_rows(baseline_path)
            replay_comparison = compare_tracker_replay(
                reference_keys,
                reference_boxes,
                np.asarray(replay_keys, dtype=np.int64),
                np.asarray(replay_boxes, dtype=np.float64),
                args.replay_atol,
            )
            if replay_comparison["status"] != "PASS":
                raise RuntimeError(f"baseline replay failed for {sequence}: {replay_comparison}")
            sequence_replays.append({"sequence": sequence, **replay_comparison})
            input_hashes["sequences"][sequence] = {
                "seqinfo": sha256_file(seqinfo_path),
                "gt": sha256_file(gt_path),
                "detections": sha256_file(cache_path),
                "baseline_tracker": sha256_file(baseline_path),
            }

        if len(results) != 165 or len({str(row["event_id"]) for row in results}) != 165:
            raise RuntimeError("score addendum coverage mismatch")
        results.sort(key=lambda row: str(row["event_id"]))
        if [row["event_id"] for row in results] != sorted(row["event_id"] for row in parent_rows):
            raise RuntimeError("score addendum event ordering/population mismatch")
        result_path = partial / "event_d1_score_margins.csv"
        with result_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(results)

        input_hashes.update({
            "parent_events": sha256_file(args.parent_events),
            "parent_d1_events": sha256_file(args.parent_d1_events),
            "protocol": sha256_file(args.protocol),
            "analysis_script": sha256_file(Path(__file__)),
            "d1_helper": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/mot_research/analysis/d1_history_correspondence.py"
            ),
            "d2_helper": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/mot_research/analysis/d2_first_association.py"
            ),
            "failure_event_helper": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/mot_research/analysis/failure_events.py"
            ),
            "upstream_ocsort": sha256_file(
                args.oc_sort_root / "trackers/ocsort_tracker/ocsort.py"
            ),
            "upstream_association": sha256_file(
                args.oc_sort_root / "trackers/ocsort_tracker/association.py"
            ),
        })
        with (partial / "input_hashes.json").open("w") as handle:
            json.dump(input_hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")
        upstream_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.oc_sort_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        runtime = {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "upstream_oc_sort_commit": upstream_commit,
            "cwd": str(Path.cwd()),
        }
        with (partial / "runtime.json").open("w") as handle:
            json.dump(runtime, handle, indent=2)
            handle.write("\n")
        summary = {
            "status": "PASS",
            "scope": "D1 v2 score-margin addendum on the unchanged 165 train events",
            "events": len(results),
            "parent_event_reproductions_passed": len(results),
            "by_outcome": grouped_summary(results, "outcome"),
            "by_candidate_class": grouped_summary(results, "candidate_class"),
            "by_D2_oracle_non_recovery": grouped_summary(
                results, "D2_oracle_non_recovery"
            ),
            "baseline_one_to_one_shortcut_events": sum(
                bool(row["baseline_one_to_one_iou_shortcut"]) for row in results
            ),
            "clean_one_to_one_shortcut_events": sum(
                bool(row["clean_one_to_one_iou_shortcut"]) for row in results
            ),
            "sequence_replays": sequence_replays,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "parent_d1_events_sha256": args.parent_d1_events_sha256,
            "result_file": result_path.name,
            "input_hashes_file": "input_hashes.json",
            "runtime_file": "runtime.json",
            "limitations": [
                "This is a post-audit supplement and does not replace or change the parent D1 v2 outcomes.",
                "Pairwise positive-detection margins are descriptive because Hungarian optimizes a global assignment.",
                "OC-SORT bypasses total scores when the IoU adjacency is already one-to-one.",
                "No counterfactual association is written back and no closed-loop metric is measured.",
                "No validation data is used.",
            ],
        }
        with (partial / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(summary, indent=2))
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


if __name__ == "__main__":
    main()
