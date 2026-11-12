#!/usr/bin/env python3
"""Extract frozen G2 train-only bounded detection-gap recovery events."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from mot_research.analysis.failure_events import (
    extract_recovery_events,
    read_detections,
    read_gt,
    read_sequence_info,
    read_tracker,
    sample_events,
    sha256_file,
)


EVENT_COLUMNS = [
    "event_id", "sequence", "gt_id", "before_frame", "gap_start_frame",
    "gap_end_frame", "after_frame", "missing_length", "outcome",
    "before_tracker_id", "after_tracker_id", "before_tracker_iou",
    "after_tracker_iou", "max_raw_detection_iou_during_gap",
    "localization_ambiguity_proxy", "max_neighbor_gt_iou",
    "neighbor_overlap_proxy", "min_normalized_boundary_distance",
    "boundary_proximity_proxy", "max_normalized_center_second_difference",
    "history_id_pollution_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--history-window", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=20261111)
    parser.add_argument("--sample-per-stratum", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def proxy_counts(events: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for outcome in ("identity_change_failure", "success_control", "unresolved"):
        subset = [event for event in events if event["outcome"] == outcome]
        result[outcome] = {
            "events": len(subset),
            "localization_ambiguity_proxy": sum(bool(event["localization_ambiguity_proxy"]) for event in subset),
            "neighbor_overlap_proxy": sum(bool(event["neighbor_overlap_proxy"]) for event in subset),
            "boundary_proximity_proxy": sum(bool(event["boundary_proximity_proxy"]) for event in subset),
            "history_id_pollution_proxy": sum(bool(event["history_id_pollution_proxy"]) for event in subset),
        }
    return result


def main() -> None:
    args = parse_args()
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise RuntimeError("protocol SHA256 does not match the frozen value")
    if args.run_dir.exists() or args.run_dir.with_name(args.run_dir.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite {args.run_dir}")
    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    partial.mkdir(parents=True)
    all_events: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    try:
        for sequence in args.sequences:
            sequence_dir = args.train_root / sequence
            gt_path = sequence_dir / "gt" / "gt.txt"
            detection_path = args.detection_root / f"{sequence}.csv"
            tracker_path = args.tracker_root / f"{sequence}.txt"
            for path in (gt_path, detection_path, tracker_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            info = read_sequence_info(sequence_dir)
            events, counters = extract_recovery_events(
                info,
                read_gt(gt_path),
                read_detections(detection_path, args.score_threshold),
                read_tracker(tracker_path),
                matching_iou=args.iou_threshold,
                max_gap=args.max_gap,
                history_window=args.history_window,
            )
            counts = Counter(str(event["outcome"]) for event in events)
            sequence_rows.append({
                "sequence": sequence,
                "events": len(events),
                "identity_change_failure": counts["identity_change_failure"],
                "success_control": counts["success_control"],
                "unresolved": counts["unresolved"],
                **counters,
            })
            all_events.extend(events)
            input_hashes[sequence] = {
                "gt": sha256_file(gt_path),
                "detections": sha256_file(detection_path),
                "tracker": sha256_file(tracker_path),
            }
        sampled = sample_events(all_events, args.sample_seed, args.sample_per_stratum)
        write_csv(partial / "all_recovery_events.csv", all_events, EVENT_COLUMNS)
        write_csv(partial / "sampled_events.csv", sampled, EVENT_COLUMNS + ["sample_stratum"])
        sequence_columns = [
            "sequence", "gt_identities", "events", "identity_change_failure",
            "success_control", "unresolved", "diagnostic_consecutive_match_id_changes",
        ]
        write_csv(partial / "sequence_summary.csv", sequence_rows, sequence_columns)
        with (partial / "input_hashes.json").open("w") as handle:
            json.dump(input_hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")
        outcome_counts = Counter(str(event["outcome"]) for event in all_events)
        sample_counts = Counter(str(event["outcome"]) for event in sampled)
        summary = {
            "status": "PASS",
            "scope": "DanceTrack train only",
            "sequences": args.sequences,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "parameters": {
                "eligible_detection_score": f"> {args.score_threshold}",
                "matching_iou": args.iou_threshold,
                "max_gap": args.max_gap,
                "history_window": args.history_window,
                "sample_seed": args.sample_seed,
                "sample_per_stratum": args.sample_per_stratum,
            },
            "event_counts": dict(outcome_counts),
            "sample_counts": dict(sample_counts),
            "sample_shortfall": {
                outcome: max(0, args.sample_per_stratum - sample_counts[outcome])
                for outcome in ("identity_change_failure", "success_control")
            },
            "proxy_counts_by_outcome": proxy_counts(all_events),
            "sequence_summary": sequence_rows,
            "limitations": [
                "Events and features are diagnostic proxies, not official TrackEval event labels.",
                "GT overlap is not treated as ground-truth occlusion.",
                "Proxy co-occurrence does not establish a causal failure mechanism.",
                "The sampled events require visual review before category claims.",
                "No validation data or validation metrics were used.",
            ],
            "input_hashes_file": "input_hashes.json",
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
