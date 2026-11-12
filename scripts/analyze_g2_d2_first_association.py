#!/usr/bin/env python3
"""Replay OC-SORT and evaluate privileged GT-position first-round associations."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mot_research.analysis.d2_first_association import (
    compare_tracker_replay,
    evaluate_association_counterfactual,
)
from mot_research.analysis.failure_events import (
    Candidate,
    hungarian_iou_matches,
    pairwise_iou,
    read_detections,
    read_gt,
    sha256_file,
)


RESULT_COLUMNS = [
    "event_id", "sequence", "outcome", "candidate_class", "after_frame", "gt_id",
    "intended_track_id", "positive_detection_id", "status", "non_applicable_reason",
    "live_tracks", "first_round_detections", "baseline_target_detection_id",
    "baseline_positive_detection_track_id", "oracle_target_detection_id",
    "oracle_positive_detection_track_id", "baseline_positive_match",
    "oracle_positive_match", "recoverable", "harmed",
    "baseline_prompt_positive_iou", "oracle_prompt_positive_iou",
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
    parser.add_argument("--contention-events", type=Path, required=True)
    parser.add_argument("--contention-events-sha256", required=True)
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


def nested_counts(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group = str(row[key])
        grouped[group][str(row["status"])] += 1
        if row["status"] == "applicable":
            grouped[group]["baseline_positive_match"] += int(bool(row["baseline_positive_match"]))
            grouped[group]["oracle_positive_match"] += int(bool(row["oracle_positive_match"]))
            grouped[group]["recoverable"] += int(bool(row["recoverable"]))
            grouped[group]["harmed"] += int(bool(row["harmed"]))
    return {group: dict(counts) for group, counts in sorted(grouped.items())}


def main() -> None:
    args = parse_args()
    checks = {
        args.parent_events: args.parent_events_sha256,
        args.contention_events: args.contention_events_sha256,
        args.protocol: args.protocol_sha256,
    }
    for path, expected in checks.items():
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
        by_sequence_frame: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in parent:
            by_sequence_frame[event["sequence"]][int(event["after_frame"])].append(event)

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

        for sequence in sorted(by_sequence_frame):
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
                    cache_all[int(row["frame"])].append(
                        [
                            float(row["x1"]), float(row["y1"]),
                            float(row["x2"]), float(row["y2"]), float(row["score"]),
                        ]
                    )
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
                detection_ids = [candidate.candidate_id for candidate in candidates]
                if len(candidates) != len(detections):
                    raise RuntimeError(f"{sequence} frame {current_frame}: candidate count mismatch")
                if candidates:
                    candidate_values = np.stack(
                        [np.r_[candidate.box_xyxy, candidate.score] for candidate in candidates]
                    )
                    if not np.allclose(candidate_values, detections, atol=1e-7, rtol=0):
                        raise RuntimeError(f"{sequence} frame {current_frame}: candidate order/value mismatch")
                tracker_ids = [item.id + 1 for item in tracker.trackers]
                if len(tracker_ids) != len(trackers):
                    raise RuntimeError(f"{sequence} frame {current_frame}: tracker alignment mismatch")
                frame_gt = gt[current_frame]
                gt_detection_matches, _ = hungarian_iou_matches(frame_gt, candidates, 0.5)
                detection_index = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
                tracker_index = {track_id: index for index, track_id in enumerate(tracker_ids)}
                for event in frame_events:
                    gt_id = int(event["gt_id"])
                    positive = gt_detection_matches.get(gt_id)
                    positive_detection_id = positive[0] if positive else -1
                    intended_track_id = int(event["before_tracker_id"])
                    row: dict[str, object] = {
                        "event_id": event["event_id"],
                        "sequence": sequence,
                        "outcome": event["outcome"],
                        "candidate_class": contention[event["event_id"]]["event_candidate_class"],
                        "after_frame": current_frame,
                        "gt_id": gt_id,
                        "intended_track_id": intended_track_id,
                        "positive_detection_id": positive_detection_id if positive else None,
                        "live_tracks": len(tracker_ids),
                        "first_round_detections": len(detection_ids),
                    }
                    if positive is None:
                        row.update({
                            "status": "not_applicable",
                            "non_applicable_reason": "positive_detection_mapping_missing",
                        })
                    elif intended_track_id not in tracker_index:
                        row.update(
                            evaluate_association_counterfactual(
                                intended_track_id,
                                positive_detection_id,
                                tracker_ids,
                                detection_ids,
                                baseline_result[0],
                                baseline_result[0],
                            )
                        )
                    else:
                        target_index = tracker_index[intended_track_id]
                        oracle_trackers = trackers.copy()
                        oracle_trackers[target_index, :4] = frame_gt[gt_id]
                        oracle_result = original_associate(
                            detections,
                            oracle_trackers,
                            iou_threshold,
                            velocities,
                            previous_obs,
                            vdc_weight,
                        )
                        row.update(
                            evaluate_association_counterfactual(
                                intended_track_id,
                                positive_detection_id,
                                tracker_ids,
                                detection_ids,
                                baseline_result[0],
                                oracle_result[0],
                            )
                        )
                        positive_index = detection_index[positive_detection_id]
                        row["baseline_prompt_positive_iou"] = float(
                            pairwise_iou(
                                trackers[target_index : target_index + 1, :4],
                                detections[positive_index : positive_index + 1, :4],
                            )[0, 0]
                        )
                        row["oracle_prompt_positive_iou"] = float(
                            pairwise_iou(
                                oracle_trackers[target_index : target_index + 1, :4],
                                detections[positive_index : positive_index + 1, :4],
                            )[0, 0]
                        )
                    results.append(row)
                return baseline_result

            ocsort_module.associate = capturing_associate
            replay_path = replay_root / f"{sequence}.txt"
            try:
                with replay_path.open("w", newline="") as handle:
                    # Preserve the upstream replay script's csv dialect/CRLF bytes so
                    # the diagnostic baseline can be checked byte-for-byte.
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
                        for x1, y1, x2, y2, track_id in tracks:
                            writer.writerow([
                                frame_id, int(track_id), f"{x1:.6f}", f"{y1:.6f}",
                                f"{x2-x1:.6f}", f"{y2-y1:.6f}", 1, -1, -1, -1,
                            ])
            finally:
                ocsort_module.associate = original_associate
            replay_sha = sha256_file(replay_path)
            baseline_sha = sha256_file(baseline_path)
            baseline_keys, baseline_boxes = read_tracker_rows(baseline_path)
            replay_keys, replay_boxes = read_tracker_rows(replay_path)
            replay_comparison = compare_tracker_replay(
                baseline_keys,
                baseline_boxes,
                replay_keys,
                replay_boxes,
                args.replay_atol,
            )
            if replay_comparison["status"] != "PASS":
                raise RuntimeError(
                    f"baseline replay failed frozen tolerance for {sequence}: {replay_comparison}"
                )
            sequence_results = [row for row in results if row["sequence"] == sequence]
            sequence_summaries.append({
                "sequence": sequence,
                "events": len(sequence_results),
                "applicable": sum(row["status"] == "applicable" for row in sequence_results),
                "recoverable": sum(bool(row.get("recoverable")) for row in sequence_results),
                "baseline_replay_sha256": replay_sha,
                "baseline_reference_sha256": baseline_sha,
                **replay_comparison,
            })
            input_hashes[sequence] = {
                "gt": sha256_file(gt_path),
                "detections": sha256_file(cache_path),
                "baseline_tracker": baseline_sha,
            }

        if len(results) != 165 or len({str(row["event_id"]) for row in results}) != 165:
            raise RuntimeError("D2 result coverage mismatch")
        results.sort(key=lambda row: str(row["event_id"]))
        with (partial / "event_d2_first_association.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(results)
        applicable = [row for row in results if row["status"] == "applicable"]
        summary = {
            "status": "PASS",
            "scope": "DanceTrack train-only privileged D2 first-association counterfactual",
            "events": len(results),
            "applicable": len(applicable),
            "not_applicable": len(results) - len(applicable),
            "baseline_positive_match": sum(bool(row["baseline_positive_match"]) for row in applicable),
            "oracle_positive_match": sum(bool(row["oracle_positive_match"]) for row in applicable),
            "recoverable": sum(bool(row["recoverable"]) for row in applicable),
            "harmed": sum(bool(row["harmed"]) for row in applicable),
            "baseline_replay_atol": args.replay_atol,
            "baseline_replay_max_abs_xyxy_error": max(
                float(item["max_abs_xyxy_error"] or 0.0) for item in sequence_summaries
            ),
            "by_outcome": nested_counts(results, "outcome"),
            "by_candidate_class": nested_counts(results, "candidate_class"),
            "sequence_summaries": sequence_summaries,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "parent_events_sha256": args.parent_events_sha256,
            "contention_events_sha256": args.contention_events_sha256,
            "input_hashes_file": "input_hashes.json",
            "limitations": [
                "GT current position is privileged and unavailable at inference.",
                "Only independent first-round association counterfactuals are evaluated; no oracle result is written back.",
                "No closed-loop tracker metric or method improvement is measured.",
                "D1 history correction and D3 detection-coordinate correction are not included.",
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
