#!/usr/bin/env python3
"""Append the protocol's intended-track margin from archived R2 score matrices."""

from __future__ import annotations

import csv
import json
import math
import os
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from mot_research.analysis.runtime_record import (
    runtime_record, sha256_file, verify_code_unchanged,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/media/imc/data/mot-data/runs/r2_rescore_p1_20261118")
OUTPUT = Path("/media/imc/data/mot-data/runs/r2_rescore_p1_20261118_secondary_margin")
CODE_PATHS = (
    "scripts/derive_r2_target_margin.py",
    "src/mot_research/analysis/runtime_record.py",
)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R2 margin addendum requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    source_summary = json.loads((RUN / "summary.json").read_text())
    source_csv = RUN / "event_rule_rows.csv"
    source_matrices = RUN / "frame_scores.jsonl"
    if sha256_file(source_csv) != source_summary["event_rule_rows_sha256"]:
        raise RuntimeError("source event CSV hash mismatch")
    if sha256_file(source_matrices) != source_summary["frame_scores_sha256"]:
        raise RuntimeError("source score matrix hash mismatch")
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    failed = OUTPUT.with_name(OUTPUT.name + ".failed")
    if any(path.exists() for path in (OUTPUT, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    partial.mkdir(parents=True)
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"source_summary_sha256": sha256_file(RUN / "summary.json"),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        event_rows = list(csv.DictReader(source_csv.open(newline="")))
        by_frame = defaultdict(list)
        for row in event_rows:
            by_frame[(row["sequence"], int(row["frame"]), row["variant"])].append(row)
        output_rows = []
        seen_frames = set()
        with source_matrices.open() as handle:
            for line in handle:
                matrix = json.loads(line)
                key = (matrix["sequence"], matrix["frame"], matrix["variant"])
                if key in seen_frames or key not in by_frame:
                    raise RuntimeError(f"unexpected/duplicate matrix {key}")
                seen_frames.add(key)
                track_ids = matrix["tracker_ids"]
                det_ids = matrix["detection_ids"]
                for row in by_frame[key]:
                    rule = row["rule"]
                    intended = int(row["target_track_id"])
                    positive = int(row["positive_detection_id"])
                    det_index = det_ids.index(positive)
                    col = track_ids.index(intended) if intended in track_ids else None
                    target_eligible = col is not None and matrix["eligibility"][det_index][col]
                    if target_eligible != (row["target_eligible"] == "True"):
                        raise RuntimeError(f"eligibility mismatch: {row['event_id']} {rule}")
                    margin = None
                    strongest_other_id = None
                    if target_eligible:
                        target_score = matrix["scores"][rule][det_index][col]
                        if not math.isclose(target_score, float(row["target_score"]),
                                            rel_tol=1e-12, abs_tol=1e-12):
                            raise RuntimeError(f"target score mismatch: {row['event_id']} {rule}")
                        other = [(matrix["scores"][rule][det_index][j], track_ids[j])
                                 for j in range(len(track_ids))
                                 if j != col and matrix["eligibility"][det_index][j]]
                        if other:
                            best_score, strongest_other_id = min(other,
                                key=lambda item: (-item[0], item[1]))
                            margin = target_score - best_score
                    output_rows.append({
                        "event_id": row["event_id"], "sequence": row["sequence"],
                        "frame": row["frame"], "variant": row["variant"], "rule": rule,
                        "target_track_id": intended, "target_eligible": target_eligible,
                        "strongest_other_track_id": strongest_other_id,
                        "target_minus_strongest_other": margin,
                    })
        if seen_frames != set(by_frame) or len(output_rows) != len(event_rows):
            raise RuntimeError("incomplete secondary margin coverage")
        if len({(r["event_id"], r["variant"], r["rule"]) for r in output_rows}) != len(output_rows):
            raise RuntimeError("duplicate secondary margin key")
        output_rows.sort(key=lambda r: (r["event_id"], r["variant"], r["rule"]))
        path = partial / "target_margin.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output_rows)
        verify_code_unchanged(ROOT, provenance)
        (partial / "summary.json").write_text(json.dumps({
            "status": "PASS", "scope": "post-hoc correction of secondary margin only",
            "source_summary_sha256": provenance["source_summary_sha256"],
            "source_event_rows_sha256": source_summary["event_rule_rows_sha256"],
            "source_score_matrices_sha256": source_summary["frame_scores_sha256"],
            "row_count": len(output_rows), "frame_matrix_count": len(seen_frames),
            "margin_defined": sum(r["target_minus_strongest_other"] is not None for r in output_rows),
            "target_margin_sha256": sha256_file(path),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "primary_gate_changed": False,
        }, indent=2) + "\n")
        os.replace(partial, OUTPUT)
        print(f"secondary margin PASS: {OUTPUT}")
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
