#!/usr/bin/env python3
"""Describe E1 extra-output risk using only frozen outputs and GT labels."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from mot_research.analysis.failure_events import (
    hungarian_iou_matches, pairwise_iou, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e1_mature_reemit_factorial_20261119.yaml"
OUTPUT = Path("/media/imc/data/mot-data/runs/e2_reemit_risk_20261119")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("risk analysis requires CUDA_VISIBLE_DEVICES='' and CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    run = Path(protocol["outputs"]["run_dir"])
    eval_run = run.with_name(run.name + "_eval")
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "PASS" or summary["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("E1 run or protocol hash mismatch")
    indexed = {(item["mode"], item["sequence"]): item for item in summary["sequences"]}
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows = []
    try:
        for condition, mode, original in (("B", "upstream", "A"),
                                          ("D", "constant_velocity", "C")):
            for seq in protocol["scope"]["sequences"]:
                source = indexed[mode, seq]
                extra_path = run / "extra_outputs" / f"{mode}_{seq}.jsonl"
                old_path = run / "trackers" / original / f"{seq}.txt"
                full_path = run / "trackers" / condition / f"{seq}.txt"
                for path, key in ((extra_path, "U_extra_sha256"),
                                  (old_path, "original_sha256"),
                                  (full_path, "U_sha256")):
                    if sha256_file(path) != source[key]:
                        raise RuntimeError(f"source hash mismatch: {path}")
                extras = [json.loads(line) for line in extra_path.open()]
                by_frame = defaultdict(list)
                for item in extras:
                    by_frame[int(item["frame"])].append(item)
                gt = read_gt(Path(protocol["inputs"]["train_root"]) / seq / "gt/gt.txt")
                old = read_tracker(old_path)
                full = read_tracker(full_path)
                last_old = {}
                previous_frame = 0
                for frame in sorted(set(by_frame) | set(old)):
                    if frame < previous_frame:
                        raise RuntimeError("frame order error")
                    previous_frame = frame
                    matched = hungarian_iou_matches(gt.get(frame, {}), full.get(frame, []), .5)[0]
                    matched_ids = {int(pair[0]) for pair in matched.values()}
                    for item in by_frame.get(frame, []):
                        tid = int(item["track_id"])
                        if tid not in last_old:
                            raise RuntimeError(f"{seq}:{frame}:{tid} has no mature output")
                        last_frame, last_box = last_old[tid]
                        box = np.asarray(item["box_xyxy"], dtype=np.float64)
                        iou = float(pairwise_iou(box[None], last_box[None])[0, 0])
                        rows.append({"condition": condition, "sequence": seq,
                                     "frame": frame, "track_id": tid,
                                     "score": float(item["score"]),
                                     "hit_streak": int(item["hit_streak"]),
                                     "gap_since_original_output": frame - last_frame,
                                     "iou_last_original_output": iou,
                                     "TP_proxy": int(tid in matched_ids)})
                    for candidate in old.get(frame, []):
                        last_old[int(candidate.candidate_id)] = (frame, candidate.box_xyxy)
        by_condition = {}
        for condition in ("B", "D"):
            group = [row for row in rows if row["condition"] == condition]
            score_bins = [("(0.6,0.7)", lambda x: x < .7),
                          ("[0.7,0.8)", lambda x: .7 <= x < .8),
                          ("[0.8,0.9)", lambda x: .8 <= x < .9),
                          ("[0.9,1]", lambda x: x >= .9)]
            gap_bins = [("1", lambda x: x == 1), ("2", lambda x: x == 2),
                        ("3-5", lambda x: 3 <= x <= 5),
                        ("6-10", lambda x: 6 <= x <= 10),
                        ("11+", lambda x: x >= 11)]
            def count(subset):
                return {"total": len(subset), "TP_proxy": sum(r["TP_proxy"] for r in subset),
                        "FP_proxy": sum(1 - r["TP_proxy"] for r in subset)}
            by_condition[condition] = {
                "all": count(group),
                "by_score": {label: count([r for r in group if test(r["score"])])
                             for label, test in score_bins},
                "by_gap": {label: count([r for r in group if test(r["gap_since_original_output"])])
                           for label, test in gap_bins},
                "by_streak": {str(streak): count([r for r in group if r["hit_streak"] == streak])
                              for streak in sorted({r["hit_streak"] for r in group})},
                "by_sequence": {seq: count([r for r in group if r["sequence"] == seq])
                                for seq in protocol["scope"]["sequences"]},
            }
        expected = json.loads((eval_run / "summary.json").read_text())["extra_output_proxy"]
        for condition in ("B", "D"):
            if by_condition[condition]["all"] != expected[condition]:
                raise RuntimeError(f"proxy count mismatch for {condition}")
        with (OUTPUT / "rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        result = {"status": "PASS", "source_protocol_sha256": sha256_file(PROTOCOL),
                  "e1_run_summary_sha256": sha256_file(run / "summary.json"),
                  "e1_eval_summary_sha256": sha256_file(eval_run / "summary.json"),
                  "script_sha256": sha256_file(Path(__file__)),
                  "rows_sha256": sha256_file(OUTPUT / "rows.csv"),
                  "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                  "conditions": by_condition}
        (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(by_condition, indent=2))
    except BaseException:
        # Keep partial evidence for debugging instead of presenting it as a result.
        (OUTPUT / "INCOMPLETE").write_text("risk analysis did not finish\n")
        raise


if __name__ == "__main__":
    main()
