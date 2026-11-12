#!/usr/bin/env python3
"""Split frozen G2 gap frames into raw-candidate absence and contention."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from mot_research.analysis.failure_events import (
    classify_event_candidate_contention,
    read_detections,
    read_gt,
    sha256_file,
)


COLUMNS = [
    "event_id", "sequence", "gt_id", "outcome", "missing_length",
    "no_raw_candidate_frames", "candidate_contention_frames",
    "event_candidate_class", "max_raw_candidates_per_gap_frame",
    "mean_best_raw_detection_iou", "max_best_raw_detection_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-events", type=Path, required=True)
    parser.add_argument("--parent-events-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def nested_counts(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row[key])][str(row["event_candidate_class"])] += 1
    return {group: dict(counts) for group, counts in sorted(grouped.items())}


def main() -> None:
    args = parse_args()
    if sha256_file(args.parent_events) != args.parent_events_sha256:
        raise RuntimeError("parent event SHA256 does not match frozen addendum")
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise RuntimeError("contention protocol SHA256 does not match frozen value")
    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    if args.run_dir.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
    partial.mkdir(parents=True)
    try:
        parent_events = list(csv.DictReader(args.parent_events.open(newline="")))
        if len(parent_events) != 797:
            raise RuntimeError(f"expected 797 frozen parent events, got {len(parent_events)}")
        by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
        for event in parent_events:
            by_sequence[event["sequence"]].append(event)
        rows: list[dict[str, object]] = []
        input_hashes: dict[str, dict[str, str]] = {}
        for sequence, events in sorted(by_sequence.items()):
            gt_path = args.train_root / sequence / "gt" / "gt.txt"
            detection_path = args.detection_root / f"{sequence}.csv"
            gt = read_gt(gt_path)
            detections = read_detections(detection_path, args.score_threshold)
            rows.extend(
                classify_event_candidate_contention(
                    event, gt, detections, matching_iou=args.iou_threshold
                )
                for event in events
            )
            input_hashes[sequence] = {
                "gt": sha256_file(gt_path),
                "detections": sha256_file(detection_path),
            }
        rows.sort(key=lambda row: str(row["event_id"]))
        with (partial / "event_candidate_contention.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        frame_totals = {
            "no_raw_candidate_frames": sum(int(row["no_raw_candidate_frames"]) for row in rows),
            "candidate_contention_frames": sum(int(row["candidate_contention_frames"]) for row in rows),
        }
        summary = {
            "status": "PASS",
            "scope": "DanceTrack train only; post-extraction exploratory addendum",
            "events": len(rows),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "parent_events": str(args.parent_events.resolve()),
            "parent_events_sha256": args.parent_events_sha256,
            "parameters": {
                "eligible_detection_score": f"> {args.score_threshold}",
                "target_iou": args.iou_threshold,
            },
            "event_class_counts": dict(Counter(str(row["event_candidate_class"]) for row in rows)),
            "gap_frame_counts": frame_totals,
            "event_class_by_outcome": nested_counts(rows, "outcome"),
            "event_class_by_sequence": nested_counts(rows, "sequence"),
            "limitations": [
                "Candidate contention is defined by diagnostic GT matching and is not available to the online tracker.",
                "A raw candidate at IoU 0.5 does not imply correct detector identity or usable association evidence.",
                "This addendum is exploratory and uses no validation data.",
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
