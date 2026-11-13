#!/usr/bin/env python3
"""Evaluate one tracker output on the frozen P1 common recovery events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mot_research.analysis.failure_events import read_gt, read_tracker
from mot_research.evaluation.closed_loop_events import evaluate_method_recovery_events


OUTCOME_FIELDS = (
    "event_id", "sequence", "gt_id", "gap_bin", "diagnostic_anchor",
    "baseline_stratum", "before_tracker_id", "after_tracker_id",
    "precondition_match_present", "reconnect_success", "persistence_censored",
    "persistence_success",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def parse_optional_bool(value: str) -> bool | None:
    return None if value == "" else parse_bool(value)


def parse_optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            events.append({
                **row,
                "gt_id": int(row["gt_id"]),
                "before_frame": int(row["before_frame"]),
                "after_frame": int(row["after_frame"]),
                "diagnostic_anchor": parse_bool(row["diagnostic_anchor"]),
                "persistence_censored": parse_bool(row["persistence_censored"]),
                "baseline_before_tracker_id": int(row["baseline_before_tracker_id"]),
                "baseline_after_tracker_id": parse_optional_int(
                    row["baseline_after_tracker_id"]
                ),
                "baseline_reconnect_success": parse_bool(
                    row["baseline_reconnect_success"]
                ),
                "baseline_persistence_success": parse_optional_bool(
                    row["baseline_persistence_success"]
                ),
            })
    ids = [str(row["event_id"]) for row in events]
    if not events or len(ids) != len(set(ids)):
        raise ValueError("event file must contain unique, non-empty event IDs")
    return events


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    values = list(rows)
    counts: Counter[str] = Counter()
    counts["events"] = len(values)
    for row in values:
        counts["precondition_match_present"] += int(row["precondition_match_present"])
        counts["reconnect_success"] += int(row["reconnect_success"])
        if row["persistence_censored"]:
            counts["persistence_censored"] += 1
            continue
        counts["persistence_evaluable"] += 1
        counts["persistence_success"] += int(bool(row["persistence_success"]))
        if row["baseline_stratum"] == "success_control":
            counts["success_controls"] += 1
            counts["success_control_harms"] += int(not row["persistence_success"])
        elif row["baseline_stratum"] == "recovery_failure":
            counts["baseline_failures"] += 1
            counts["baseline_failure_recoveries"] += int(bool(row["persistence_success"]))
    return dict(sorted(counts.items()))


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if key == "anchor_split":
            value = "anchor" if row["diagnostic_anchor"] else "non_anchor"
        else:
            value = str(row[key])
        groups[value].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def assert_baseline_self(
    events: list[dict[str, Any]], outcomes: list[dict[str, Any]]
) -> None:
    by_id = {row["event_id"]: row for row in outcomes}
    for event in events:
        outcome = by_id[event["event_id"]]
        expected = {
            "before_tracker_id": event["baseline_before_tracker_id"],
            "after_tracker_id": event["baseline_after_tracker_id"],
            "reconnect_success": event["baseline_reconnect_success"],
            "persistence_censored": event["persistence_censored"],
            "persistence_success": event["baseline_persistence_success"],
        }
        actual = {key: outcome[key] for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"baseline self mismatch for {event['event_id']}: {actual} != {expected}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--events-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matching-iou", type=float, default=0.5)
    parser.add_argument("--persistence-frames", type=int, default=5)
    parser.add_argument("--expect-baseline-self", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("recovery evaluation requires CUDA_VISIBLE_DEVICES empty")
    if sha256_file(args.events) != args.events_sha256:
        raise RuntimeError("event CSV SHA256 mismatch")
    events = read_events(args.events)
    partial = args.output_dir.with_name(args.output_dir.name + ".partial")
    failed = args.output_dir.with_name(args.output_dir.name + ".failed")
    if any(path.exists() for path in (args.output_dir, partial, failed)):
        raise FileExistsError(f"refusing to overwrite output near {args.output_dir}")
    partial.mkdir(parents=True)
    try:
        by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            by_sequence[str(event["sequence"])].append(event)
        outcomes = []
        input_hashes: dict[str, Any] = {"sequences": {}}
        for sequence, sequence_events in sorted(by_sequence.items()):
            gt_path = args.train_root / sequence / "gt" / "gt.txt"
            tracker_path = args.tracker_root / f"{sequence}.txt"
            sequence_outcomes, _ = evaluate_method_recovery_events(
                events=sequence_events,
                gt_by_frame=read_gt(gt_path),
                tracker_by_frame=read_tracker(tracker_path),
                matching_iou=args.matching_iou,
                persistence_frames=args.persistence_frames,
            )
            outcomes.extend(sequence_outcomes)
            input_hashes["sequences"][sequence] = {
                "gt": sha256_file(gt_path),
                "tracker": sha256_file(tracker_path),
            }
        outcomes.sort(key=lambda row: str(row["event_id"]))
        if [row["event_id"] for row in outcomes] != sorted(
            row["event_id"] for row in events
        ):
            raise RuntimeError("outcome event IDs differ from fixed event file")
        if args.expect_baseline_self:
            assert_baseline_self(events, outcomes)

        outcome_path = partial / "outcomes.csv"
        with outcome_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTCOME_FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in outcomes:
                writer.writerow({key: row[key] for key in OUTCOME_FIELDS})
        overall = summarize(outcomes)
        if args.expect_baseline_self:
            expected = {
                "events": 831,
                "precondition_match_present": 831,
                "reconnect_success": 159,
                "persistence_censored": 51,
                "persistence_evaluable": 780,
                "persistence_success": 81,
                "success_controls": 81,
                "success_control_harms": 0,
                "baseline_failures": 699,
                "baseline_failure_recoveries": 0,
            }
            if overall != expected:
                raise RuntimeError(f"baseline aggregate mismatch: {overall} != {expected}")
        summary = {
            "execution_status": "PASS",
            "method_id": args.method_id,
            "events_sha256": args.events_sha256,
            "matching_iou": args.matching_iou,
            "persistence_frames": args.persistence_frames,
            "baseline_self_exact": args.expect_baseline_self,
            "overall": overall,
            "by_sequence": grouped(outcomes, "sequence"),
            "by_gap": grouped(outcomes, "gap_bin"),
            "by_anchor_split": grouped(outcomes, "anchor_split"),
            "by_baseline_stratum": grouped(outcomes, "baseline_stratum"),
            "outcomes_sha256": sha256_file(outcome_path),
            "limitations": [
                "This is a fixed-denominator diagnostic, not an official TrackEval metric.",
                "GT is used only after tracking to match outputs for evaluation.",
                "Motion-dev is development evidence and contains four selected anchors.",
            ],
        }
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        repo_root = Path(__file__).resolve().parents[1]
        input_hashes.update({
            "events": sha256_file(args.events),
            "evaluator": sha256_file(Path(__file__)),
            "event_helper": sha256_file(
                repo_root / "src/mot_research/evaluation/closed_loop_events.py"
            ),
        })
        (partial / "input_hashes.json").write_text(
            json.dumps(input_hashes, indent=2, sort_keys=True) + "\n"
        )
        (partial / "runtime.json").write_text(json.dumps({
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        os.replace(partial, args.output_dir)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        if partial.exists():
            (partial / "failure.json").write_text(json.dumps({
                "execution_status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }, indent=2) + "\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
