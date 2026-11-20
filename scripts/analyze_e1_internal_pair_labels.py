#!/usr/bin/env python3
"""Post-hoc descriptive CV-own-ID labeling of E1 first-round pairs."""

from __future__ import annotations

import csv
import json
import os
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mot_research.analysis.failure_events import (
    hungarian_iou_matches, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import runtime_record, sha256_file, verify_code_unchanged

ROOT = Path(__file__).resolve().parents[1]
PAIR_RUN = Path("/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119_internal_pairs_addendum")
OUTPUT = Path("/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119_internal_pair_labels")
EVENTS = Path("/media/imc/data/mot-data/runs/p1_closed_loop_common_events_v1_retry1_20261112/events.csv")
GT_ROOT = Path("/media/imc/data/mot-data/datasets/dancetrack/train")
CV_ROOT = Path("/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119/trackers/C")
PRIOR = Path("/media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113/input_hashes.json")
CODE_PATHS = (
    "scripts/analyze_e1_internal_pair_labels.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/analysis/runtime_record.py",
)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("pair labels require CUDA_VISIBLE_DEVICES='' and CPU only")
    pair_summary = json.loads((PAIR_RUN / "summary.json").read_text())
    if pair_summary["status"] != "PASS":
        raise RuntimeError("pair addendum incomplete")
    if sha256_file(PAIR_RUN / "event_internal_pairs.csv") != pair_summary["csv_sha256"]:
        raise RuntimeError("pair addendum CSV hash mismatch")
    prior = json.loads(PRIOR.read_text())
    pairs = list(csv.DictReader((PAIR_RUN / "event_internal_pairs.csv").open(newline="")))
    events = {row["event_id"]: row for row in csv.DictReader(EVENTS.open(newline=""))}
    if len(pairs) != 831 or set(events) != {r["event_id"] for r in pairs}:
        raise RuntimeError("event coverage mismatch")
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    failed = OUTPUT.with_name(OUTPUT.name + ".failed")
    if any(p.exists() for p in (OUTPUT, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    partial.mkdir(parents=True)
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"pair_summary_sha256": sha256_file(PAIR_RUN / "summary.json"),
                           "events_sha256": sha256_file(EVENTS),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        by_sequence = defaultdict(list)
        for row in pairs:
            by_sequence[row["sequence"]].append(row)
        rows = []
        summaries = []
        for sequence, seq_pairs in sorted(by_sequence.items()):
            gt_path = GT_ROOT / sequence / "gt" / "gt.txt"
            if sha256_file(gt_path) != prior["sequences"][sequence]["gt"]:
                raise RuntimeError(f"{sequence}: GT hash mismatch")
            gt = read_gt(gt_path)
            cv_path = CV_ROOT / f"{sequence}.txt"
            cv = read_tracker(cv_path)
            before_frames = {int(events[r["event_id"]]["before_frame"]) for r in seq_pairs}
            matches = {frame: hungarian_iou_matches(gt.get(frame, {}),
                                                     cv.get(frame, []), .5)[0]
                       for frame in before_frames}
            for pair in seq_pairs:
                event = events[pair["event_id"]]
                before_frame = int(event["before_frame"])
                gt_id = int(event["gt_id"])
                matched = matches[before_frame].get(gt_id)
                cv_own_id = None if matched is None else int(matched[0])
                owner = (None if pair["CV_positive_detection_first_round_owner"] == ""
                         else int(pair["CV_positive_detection_first_round_owner"]))
                rows.append({"event_id": pair["event_id"], "sequence": sequence,
                             "CV_before_tracker_id": cv_own_id,
                             "CV_before_precondition_present": cv_own_id is not None,
                             "positive_detection_first_round_owner": owner,
                             "CV_own_ID_first_round_positive_pair":
                                 cv_own_id is not None and owner == cv_own_id,
                             "B0_ID_first_round_positive_pair":
                                 pair["CV_first_round_positive_pair"] == "True"})
            sequence_rows = [r for r in rows if r["sequence"] == sequence]
            summaries.append({"sequence": sequence, "events": len(sequence_rows),
                              "CV_before_precondition_present": sum(r["CV_before_precondition_present"]
                                                                     for r in sequence_rows),
                              "CV_own_ID_first_round_positive_pair": sum(r["CV_own_ID_first_round_positive_pair"]
                                                                         for r in sequence_rows),
                              "CV_gt_sha256": sha256_file(gt_path),
                              "CV_tracker_sha256": sha256_file(cv_path)})
        if len(rows) != 831 or len({r["event_id"] for r in rows}) != 831:
            raise RuntimeError("labeled pair coverage mismatch")
        path = partial / "event_cv_own_pair.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["event_id"]))
        verify_code_unchanged(ROOT, provenance)
        summary = {"status": "PASS", "scope": "post-hoc descriptive internal pair label; not E1 gate",
                   "pair_summary_sha256": provenance["pair_summary_sha256"],
                   "event_count": len(rows),
                   "CV_before_precondition_present": sum(r["CV_before_precondition_present"] for r in rows),
                   "CV_own_ID_first_round_positive_pair": sum(r["CV_own_ID_first_round_positive_pair"] for r in rows),
                   "CV_B0_ID_first_round_positive_pair": sum(r["B0_ID_first_round_positive_pair"] for r in rows),
                   "sequences": summaries, "csv_sha256": sha256_file(path),
                   "E1_main_gate_changed": False,
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, OUTPUT)
        print(json.dumps({"status": "PASS", "CV_before_precondition_present": summary["CV_before_precondition_present"],
                          "CV_own_ID_first_round_positive_pair": summary["CV_own_ID_first_round_positive_pair"],
                          "output": str(OUTPUT)}, indent=2))
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
