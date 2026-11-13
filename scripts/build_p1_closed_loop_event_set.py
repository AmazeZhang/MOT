#!/usr/bin/env python3
"""Build the fixed B0 common recovery-event denominator for P1 evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mot_research.analysis.failure_events import (
    read_detections,
    read_gt,
    read_tracker,
)
from mot_research.data.motion_split import validate_motion_split
from mot_research.evaluation.closed_loop_events import build_common_recovery_events


FIELDS = (
    "event_id", "sequence", "gt_id", "before_frame", "gap_start_frame",
    "gap_end_frame", "after_frame", "missing_length", "gap_bin",
    "diagnostic_anchor", "reappearance_detection_id", "persistence_censored",
    "baseline_before_tracker_id", "baseline_after_tracker_id",
    "baseline_reconnect_success", "baseline_persistence_success", "baseline_stratum",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--baseline-tracker-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matching-iou", type=float, default=0.5)
    parser.add_argument("--maximum-gap", type=int, default=31)
    parser.add_argument("--persistence-frames", type=int, default=5)
    return parser.parse_args()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("event-set construction requires CUDA_VISIBLE_DEVICES empty")
    verify_hash(args.split_manifest, args.split_manifest_sha256)
    verify_hash(args.protocol, args.protocol_sha256)
    protocol = yaml.safe_load(args.protocol.read_text())
    expected = {
        "matching_iou": args.matching_iou,
        "maximum_gap": args.maximum_gap,
        "persistence_frames": args.persistence_frames,
    }
    if protocol.get("event_definition") != expected:
        raise RuntimeError("CLI event definition differs from frozen protocol")

    partial = args.output_dir.with_name(args.output_dir.name + ".partial")
    failed = args.output_dir.with_name(args.output_dir.name + ".failed")
    if any(path.exists() for path in (args.output_dir, partial, failed)):
        raise FileExistsError(f"refusing to overwrite output near {args.output_dir}")
    partial.mkdir(parents=True)
    try:
        split = yaml.safe_load(args.split_manifest.read_text())
        split_validation = validate_motion_split(split, args.train_root)
        anchors = set(split["selection"]["diagnostic_anchor_dev"])
        events: list[dict[str, Any]] = []
        per_sequence = []
        input_hashes: dict[str, Any] = {"sequences": {}}
        totals: Counter[str] = Counter()
        for sequence in split["motion_dev"]["sequences"]:
            gt_path = args.train_root / sequence / "gt" / "gt.txt"
            detection_path = args.detection_root / f"{sequence}.csv"
            tracker_path = args.baseline_tracker_root / f"{sequence}.txt"
            sequence_events, counters = build_common_recovery_events(
                sequence=sequence,
                gt_by_frame=read_gt(gt_path),
                detections_by_frame=read_detections(detection_path, 0.6),
                baseline_tracker_by_frame=read_tracker(tracker_path),
                diagnostic_anchor=sequence in anchors,
                matching_iou=args.matching_iou,
                maximum_gap=args.maximum_gap,
                persistence_frames=args.persistence_frames,
            )
            events.extend(sequence_events)
            totals.update(counters)
            per_sequence.append({
                "sequence": sequence,
                "diagnostic_anchor": sequence in anchors,
                "event_count": len(sequence_events),
                "counters": counters,
            })
            input_hashes["sequences"][sequence] = {
                "gt": sha256_file(gt_path),
                "detections": sha256_file(detection_path),
                "registered_baseline_tracker": sha256_file(tracker_path),
            }

        event_path = partial / "events.csv"
        with event_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for event in events:
                writer.writerow({key: event[key] for key in FIELDS})
        strata = Counter(str(event["baseline_stratum"]) for event in events)
        gaps = Counter(str(event["gap_bin"]) for event in events)
        anchor = Counter(
            "anchor" if event["diagnostic_anchor"] else "non_anchor" for event in events
        )
        cross = Counter(
            f"{event['gap_bin']}|{event['baseline_stratum']}" for event in events
        )
        summary = {
            "execution_status": "PASS",
            "scope": "fixed B0 common recovery-event denominator for P1 motion-dev",
            "method_results_read": False,
            "official_val_used": False,
            "event_count": len(events),
            "event_definition": expected,
            "baseline_strata": dict(sorted(strata.items())),
            "gap_bins": dict(sorted(gaps.items())),
            "anchor_split": dict(sorted(anchor.items())),
            "gap_by_baseline_stratum": dict(sorted(cross.items())),
            "construction_counters": dict(sorted(totals.items())),
            "per_sequence": per_sequence,
            "events_sha256": sha256_file(event_path),
            "split_validation": split_validation,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "limitations": [
                "GT is used only for offline event construction and method evaluation.",
                "The denominator is conditioned on a registered B0 match before each gap.",
                "These diagnostic recovery events do not replace official TrackEval metrics.",
                "Motion-dev includes four selected difficult anchors and is development evidence.",
            ],
        }
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        repo_root = Path(__file__).resolve().parents[1]
        input_hashes.update({
            "split_manifest": sha256_file(args.split_manifest),
            "protocol": sha256_file(args.protocol),
            "builder": sha256_file(Path(__file__)),
            "event_helper": sha256_file(
                repo_root / "src/mot_research/evaluation/closed_loop_events.py"
            ),
            "failure_event_helpers": sha256_file(
                repo_root / "src/mot_research/analysis/failure_events.py"
            ),
        })
        (partial / "input_hashes.json").write_text(
            json.dumps(input_hashes, indent=2, sort_keys=True) + "\n"
        )
        (partial / "runtime.json").write_text(json.dumps({
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
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
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
