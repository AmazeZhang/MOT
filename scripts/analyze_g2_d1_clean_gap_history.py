#!/usr/bin/env python3
"""Evaluate D1 by suppressing wrong-identity target-track updates during each gap."""

from __future__ import annotations

import argparse
import configparser
import copy
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mot_research.analysis.d1_history_correspondence import summarize_gap_updates
from mot_research.analysis.d2_first_association import (
    compare_tracker_replay,
    evaluate_association_counterfactual,
)
from mot_research.analysis.failure_events import (
    hungarian_iou_matches,
    pairwise_iou,
    read_detections,
    read_gt,
    sha256_file,
)


RESULT_COLUMNS = [
    "event_id", "sequence", "outcome", "candidate_class", "before_frame",
    "gap_start_frame", "gap_end_frame", "after_frame", "missing_length", "gt_id",
    "intended_track_id", "positive_detection_id", "D2_oracle_positive_match",
    "D2_oracle_non_recovery", "status", "non_applicable_reason", "live_tracks",
    "first_round_detections", "baseline_target_detection_id",
    "baseline_positive_detection_track_id", "oracle_target_detection_id",
    "oracle_positive_detection_track_id", "baseline_positive_match",
    "D1_clean_history_positive_match", "recoverable", "harmed",
    "baseline_gap_updates", "baseline_gap_updates_target_owner",
    "baseline_gap_updates_other_owner", "baseline_gap_updates_unmapped_owner",
    "baseline_positive_iou", "D1_clean_history_positive_iou",
    "prediction_xyxy_max_abs_change", "association_pairs_changed",
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


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def grouped_counts(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row[key])
        groups[group][str(row["status"])] += 1
        if row["status"] == "applicable":
            groups[group]["baseline_positive_match"] += int(bool(row["baseline_positive_match"]))
            groups[group]["D1_clean_history_positive_match"] += int(
                bool(row["D1_clean_history_positive_match"])
            )
            groups[group]["recoverable"] += int(bool(row["recoverable"]))
            groups[group]["harmed"] += int(bool(row["harmed"]))
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
            raise RuntimeError(f"expected 165 resolved parent events, got {len(parent)}")
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
            raise RuntimeError("D1 parent and D2 event IDs differ")
        by_sequence_after: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_sequence_before: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_sequence_gap: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in parent:
            sequence = event["sequence"]
            by_sequence_after[sequence][int(event["after_frame"])].append(event)
            by_sequence_before[sequence][int(event["before_frame"])].append(event)
            for frame in range(int(event["gap_start_frame"]), int(event["gap_end_frame"]) + 1):
                by_sequence_gap[sequence][frame].append(event)

        sys.path.insert(0, str(args.oc_sort_root.resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module

        original_associate = ocsort_module.associate
        results: list[dict[str, object]] = []
        sequence_summaries: list[dict[str, object]] = []
        input_hashes: dict[str, dict[str, str]] = {}
        tracker_args = {
            "det_thresh": 0.6,
            "iou_threshold": 0.3,
            "asso_func": "iou",
            "delta_t": 3,
            "inertia": 0.2,
            "use_byte": False,
        }
        replay_root = partial / "baseline_replay"
        replay_root.mkdir()

        for sequence in sorted(by_sequence_after):
            sequence_dir = args.train_root / sequence
            config = configparser.ConfigParser()
            if not config.read(sequence_dir / "seqinfo.ini"):
                raise FileNotFoundError(sequence_dir / "seqinfo.ini")
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
            track_history: dict[int, list[dict[str, int | None]]] = defaultdict(list)
            clean_states: dict[str, object] = {}
            anchor_failures: dict[str, str] = {}
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
                    values = np.stack([np.r_[item.box_xyxy, item.score] for item in candidates])
                    if not np.allclose(values, detections, atol=1e-7, rtol=0):
                        raise RuntimeError(f"{sequence} frame {current_frame}: candidate mismatch")
                tracker_ids = [item.id + 1 for item in tracker.trackers]
                if len(tracker_ids) != len(trackers):
                    raise RuntimeError(f"{sequence} frame {current_frame}: tracker mismatch")
                tracker_index = {track_id: index for index, track_id in enumerate(tracker_ids)}
                detection_index = {det_id: index for index, det_id in enumerate(detection_ids)}
                gt_detection_matches, _ = hungarian_iou_matches(gt[current_frame], candidates, 0.5)
                for event in frame_events:
                    event_id = event["event_id"]
                    gt_id = int(event["gt_id"])
                    intended_track_id = int(event["before_tracker_id"])
                    positive = gt_detection_matches.get(gt_id)
                    positive_detection_id = None if positive is None else positive[0]
                    d2_row = d2[event_id]
                    row: dict[str, object] = {
                        "event_id": event_id,
                        "sequence": sequence,
                        "outcome": event["outcome"],
                        "candidate_class": contention[event_id]["event_candidate_class"],
                        "before_frame": int(event["before_frame"]),
                        "gap_start_frame": int(event["gap_start_frame"]),
                        "gap_end_frame": int(event["gap_end_frame"]),
                        "after_frame": current_frame,
                        "missing_length": int(event["missing_length"]),
                        "gt_id": gt_id,
                        "intended_track_id": intended_track_id,
                        "positive_detection_id": positive_detection_id,
                        "D2_oracle_positive_match": parse_bool(d2_row["oracle_positive_match"]),
                        "D2_oracle_non_recovery": event["outcome"] == "identity_change_failure"
                        and not parse_bool(d2_row["oracle_positive_match"]),
                        "live_tracks": len(tracker_ids),
                        "first_round_detections": len(detection_ids),
                    }
                    row.update(summarize_gap_updates(
                        track_history.get(intended_track_id, []),
                        int(event["gap_start_frame"]),
                        int(event["gap_end_frame"]),
                        gt_id,
                    ))
                    reason = anchor_failures.get(event_id)
                    if positive_detection_id is None:
                        reason = "positive_detection_mapping_missing"
                    elif intended_track_id not in tracker_index:
                        reason = "intended_history_track_not_alive"
                    elif event_id not in clean_states:
                        reason = reason or "trusted_anchor_state_missing"
                    if reason:
                        oracle_matches = baseline_result[0]
                        row.update(evaluate_association_counterfactual(
                            intended_track_id,
                            -1 if positive_detection_id is None else positive_detection_id,
                            tracker_ids,
                            detection_ids,
                            baseline_result[0],
                            oracle_matches,
                        ))
                        row["status"] = "not_applicable"
                        row["non_applicable_reason"] = reason
                    else:
                        target_index = tracker_index[intended_track_id]
                        clean = copy.deepcopy(clean_states[event_id])
                        clean_prediction = clean.predict()[0]
                        oracle_trackers = trackers.copy()
                        oracle_velocities = velocities.copy()
                        oracle_previous_obs = previous_obs.copy()
                        oracle_trackers[target_index, :4] = clean_prediction
                        oracle_velocities[target_index] = (
                            clean.velocity if clean.velocity is not None else np.asarray((0, 0))
                        )
                        oracle_previous_obs[target_index] = ocsort_module.k_previous_obs(
                            clean.observations, clean.age, clean.delta_t
                        )
                        oracle_result = original_associate(
                            detections,
                            oracle_trackers,
                            iou_threshold,
                            oracle_velocities,
                            oracle_previous_obs,
                            vdc_weight,
                        )
                        evaluated = evaluate_association_counterfactual(
                            intended_track_id,
                            positive_detection_id,
                            tracker_ids,
                            detection_ids,
                            baseline_result[0],
                            oracle_result[0],
                        )
                        evaluated["D1_clean_history_positive_match"] = evaluated.pop(
                            "oracle_positive_match"
                        )
                        row.update(evaluated)
                        positive_index = detection_index[positive_detection_id]
                        row["baseline_positive_iou"] = float(pairwise_iou(
                            trackers[target_index : target_index + 1, :4],
                            detections[positive_index : positive_index + 1, :4],
                        )[0, 0])
                        row["D1_clean_history_positive_iou"] = float(pairwise_iou(
                            oracle_trackers[target_index : target_index + 1, :4],
                            detections[positive_index : positive_index + 1, :4],
                        )[0, 0])
                        row["prediction_xyxy_max_abs_change"] = float(np.max(np.abs(
                            trackers[target_index, :4] - oracle_trackers[target_index, :4]
                        )))
                    results.append(row)
                return baseline_result

            ocsort_module.associate = capturing_associate
            replay_path = replay_root / f"{sequence}.txt"
            try:
                with replay_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
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
                        candidates = high_candidates.get(frame_id, [])
                        gt_matches, _ = hungarian_iou_matches(gt[frame_id], candidates, 0.5)
                        detection_to_gt = {
                            detection_id: gt_id
                            for gt_id, (detection_id, _) in gt_matches.items()
                        }
                        candidate_values = [np.r_[item.box_xyxy, item.score] for item in candidates]
                        for live_track in tracker.trackers:
                            if live_track.time_since_update != 0:
                                continue
                            matching = [
                                item.candidate_id
                                for item, value in zip(candidates, candidate_values)
                                if np.allclose(live_track.last_observation, value, atol=1e-7, rtol=0)
                            ]
                            candidate_id = matching[0] if len(matching) == 1 else None
                            track_history[live_track.id + 1].append({
                                "frame": frame_id,
                                "detection_id": candidate_id,
                                "gt_id": None if candidate_id is None else detection_to_gt.get(candidate_id),
                            })
                        live_by_id = {item.id + 1: item for item in tracker.trackers}
                        for event in by_sequence_before[sequence].get(frame_id, []):
                            track_id = int(event["before_tracker_id"])
                            if track_id not in live_by_id:
                                anchor_failures[event["event_id"]] = "trusted_anchor_track_missing"
                            else:
                                clean_states[event["event_id"]] = copy.deepcopy(live_by_id[track_id])
                        for event in by_sequence_gap[sequence].get(frame_id, []):
                            state = clean_states.get(event["event_id"])
                            if state is not None:
                                state.predict()
                                state.update(None)
                        for x1, y1, x2, y2, track_id in tracks:
                            writer.writerow([
                                frame_id, int(track_id), f"{x1:.6f}", f"{y1:.6f}",
                                f"{x2-x1:.6f}", f"{y2-y1:.6f}", 1, -1, -1, -1,
                            ])
            finally:
                ocsort_module.associate = original_associate
            baseline_keys, baseline_boxes = read_tracker_rows(baseline_path)
            replay_keys, replay_boxes = read_tracker_rows(replay_path)
            replay_comparison = compare_tracker_replay(
                baseline_keys, baseline_boxes, replay_keys, replay_boxes, args.replay_atol
            )
            if replay_comparison["status"] != "PASS":
                raise RuntimeError(f"baseline replay failed for {sequence}: {replay_comparison}")
            subset = [row for row in results if row["sequence"] == sequence]
            sequence_summaries.append({
                "sequence": sequence,
                "events": len(subset),
                "applicable": sum(row["status"] == "applicable" for row in subset),
                "recoverable": sum(bool(row.get("recoverable")) for row in subset),
                "harmed": sum(bool(row.get("harmed")) for row in subset),
                "baseline_replay_sha256": sha256_file(replay_path),
                "baseline_reference_sha256": sha256_file(baseline_path),
                **replay_comparison,
            })
            input_hashes[sequence] = {
                "gt": sha256_file(gt_path),
                "detections": sha256_file(cache_path),
                "baseline_tracker": sha256_file(baseline_path),
            }

        if len(results) != 165 or len({str(row["event_id"]) for row in results}) != 165:
            raise RuntimeError("D1 v2 result coverage mismatch")
        results.sort(key=lambda row: str(row["event_id"]))
        with (partial / "event_d1_clean_gap_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(results)
        applicable = [row for row in results if row["status"] == "applicable"]
        non_applicable = Counter(
            str(row["non_applicable_reason"])
            for row in results if row["status"] != "applicable"
        )
        summary = {
            "status": "PASS",
            "scope": "DanceTrack train-only privileged D1 clean-gap-history first association",
            "events": len(results),
            "applicable": len(applicable),
            "not_applicable": len(results) - len(applicable),
            "non_applicable_reasons": dict(non_applicable),
            "baseline_positive_match": sum(bool(row["baseline_positive_match"]) for row in applicable),
            "D1_clean_history_positive_match": sum(
                bool(row["D1_clean_history_positive_match"]) for row in applicable
            ),
            "recoverable": sum(bool(row["recoverable"]) for row in applicable),
            "harmed": sum(bool(row["harmed"]) for row in applicable),
            "events_with_baseline_gap_update": sum(
                int(row["baseline_gap_updates"]) > 0 for row in applicable
            ),
            "events_with_other_owner_gap_update": sum(
                int(row["baseline_gap_updates_other_owner"]) > 0 for row in applicable
            ),
            "by_outcome": grouped_counts(results, "outcome"),
            "by_candidate_class": grouped_counts(results, "candidate_class"),
            "by_D2_oracle_non_recovery": grouped_counts(results, "D2_oracle_non_recovery"),
            "baseline_replay_atol": args.replay_atol,
            "baseline_replay_max_abs_xyxy_error": max(
                float(item["max_abs_xyxy_error"] or 0.0) for item in sequence_summaries
            ),
            "sequence_summaries": sequence_summaries,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "parent_events_sha256": args.parent_events_sha256,
            "contention_events_sha256": args.contention_events_sha256,
            "d2_events_sha256": args.d2_events_sha256,
            "input_hashes_file": "input_hashes.json",
            "limitations": [
                "GT-derived gap correspondence is privileged and unavailable at inference.",
                "V2 is exploratory because it was designed after v1 failed its success-control sanity check.",
                "Only target-track updates during the frozen gap are cleaned; earlier history remains unchanged.",
                "Independent first-round counterfactuals are not written back and do not measure closed-loop metrics.",
                "No validation data is used.",
            ],
        }
        with (partial / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        with (partial / "input_hashes.json").open("w") as handle:
            json.dump(input_hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(summary, indent=2))
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


if __name__ == "__main__":
    main()
