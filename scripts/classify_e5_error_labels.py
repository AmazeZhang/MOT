#!/usr/bin/env python3
"""Reproduce nuanced GT-positive and rightful-pair labels for E5 diagnostics."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mot_research.analysis.failure_events import pairwise_iou, read_detections, read_gt
from mot_research.analysis.runtime_record import sha256_file

ROOT = Path("/media/imc/data/mot-data/runs/e5_baseline_pair_trace_addendum_20261120")
DATA = Path("/media/imc/data/mot-data/datasets/dancetrack/train")
DETECTIONS = Path("/media/imc/data/mot-data/cache/detections/ocsort_yolox_dance_bb2f686_b8d1/train")


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("read-only label check requires CUDA_VISIBLE_DEVICES='' and CPU only")
    joined_path = ROOT / "joined_events.jsonl"
    joined_summary_path = ROOT / "joined_summary.json"
    trace_summary_path = ROOT / "summary.json"
    output_path = ROOT / "positive_label_and_displacement_check.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite label check")
    joined_summary = json.loads(joined_summary_path.read_text())
    trace_summary = json.loads(trace_summary_path.read_text())
    if sha256_file(joined_path) != joined_summary["joined_events_sha256"]:
        raise RuntimeError("joined rows changed")
    events = [json.loads(line) for line in joined_path.read_text().splitlines()]
    if len(events) != 576:
        raise RuntimeError("unexpected event coverage")
    per_sequence = {row["sequence"]: row for row in trace_summary["sequences"]}
    counts = Counter()
    for seq in per_sequence:
        seq_rows = [row for row in events if row["sequence"] == seq]
        gt = read_gt(DATA / seq / "gt/gt.txt")
        high = read_detections(DETECTIONS / f"{seq}.csv", .6)
        trace_path = ROOT / f"{seq}.pairs.jsonl"
        if sha256_file(trace_path) != per_sequence[seq]["pairs_sha256"]:
            raise RuntimeError("pair trace changed")
        trace_rows = {(row["sequence"], row["frame"], row["track_id"], row["anchor_frame"]): row
                      for row in map(json.loads, trace_path.read_text().splitlines())}
        for row in seq_rows:
            key = (seq, row["frame"], row["track_id"], row["anchor_frame"])
            if key not in trace_rows:
                raise RuntimeError("joined event missing pair trace")
            expected_high_ids = [item.candidate_id for item in high.get(row["frame"], [])]
            if trace_rows[key]["high_detection_ids"] != expected_high_ids:
                raise RuntimeError("cached high-detection order mismatch")
            target_gt = row["target_gt_id"]
            frame_gt = gt.get(row["frame"], {})
            if target_gt is None:
                category = "no_prior_C_GT_anchor"
            elif target_gt not in frame_gt:
                category = "target_absent_from_current_GT"
            elif row["positive_detection_id"] is not None:
                category = "assigned_target_positive_high_detection"
            else:
                detections = high.get(row["frame"], [])
                if detections:
                    boxes = np.stack([item.box_xyxy for item in detections])
                    best = float(pairwise_iou(frame_gt[target_gt][None, :], boxes).max())
                else:
                    best = 0.0
                category = ("no_high_detection_IoU_0.5" if best < .5 else
                            "overlap_exists_but_Hungarian_assigns_elsewhere")
            counts[f"target_label_{category}"] += 1
            if row["positive_detection_id"] is None and category == "assigned_target_positive_high_detection":
                raise RuntimeError("positive label inconsistency")
            if row["other_anchor_harm_group"] == "both":
                geo = row["selected"]["geometry"]["was_paired_to_GT_owners_prior_C_track"]
                app = row["selected"]["appearance"]["was_paired_to_GT_owners_prior_C_track"]
                counts["joint_harm"] += 1
                counts["joint_harm_both_branches_displace_rightful_pair"] += geo and app
                counts["joint_harm_either_branch_displaces_rightful_pair"] += geo or app
                counts["joint_harm_geometry_displaces_rightful_pair"] += geo
                counts["joint_harm_appearance_displaces_rightful_pair"] += app
            counts["events"] += 1
    if (counts["events"] != 576 or counts["joint_harm"] != 414 or
            counts["target_label_no_prior_C_GT_anchor"] != 47 or
            counts["target_label_target_absent_from_current_GT"] != 107 or
            counts["target_label_no_high_detection_IoU_0.5"] != 64 or
            counts["target_label_overlap_exists_but_Hungarian_assigns_elsewhere"] != 125 or
            counts["target_label_assigned_target_positive_high_detection"] != 233 or
            counts["joint_harm_both_branches_displace_rightful_pair"] != 398 or
            counts["joint_harm_either_branch_displaces_rightful_pair"] != 402):
        raise RuntimeError("unexpected refined diagnostic counts")
    result = {"status": "PASS", "source_hashes": {
        "joined_events": sha256_file(joined_path),
        "joined_summary": sha256_file(joined_summary_path),
        "pair_trace_summary": sha256_file(trace_summary_path)},
        "counts": dict(counts), "generated_utc": datetime.now(timezone.utc).isoformat()}
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "counts": dict(counts),
                      "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
