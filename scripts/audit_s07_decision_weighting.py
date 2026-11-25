#!/usr/bin/env python3
"""Post-result denominator audit for S07 component-decision accuracies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def choose(rows: list[dict], field: str) -> str:
    best = max((float(row[field]), -index, row["action"])
               for index, row in enumerate(rows))
    return best[2] if best[0] > 0 else "M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("denominator audit is CPU-only")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["epoch"], row["split"], row["sequence"],
                 row["component_id"])].append(row)
    results = []
    for model in ("Local", "Context"):
        for epoch in (5, 10, 15, 20, 25, 30):
            for split in ("train18", "internal_dev6"):
                components = [(key, value) for key, value in grouped.items()
                              if key[:3] == (model, epoch, split)]
                for policy in ("no_extra", "risk"):
                    correctness = []
                    by_sequence: dict[str, list[bool]] = defaultdict(list)
                    all_m_correctness = []
                    all_m_by_sequence: dict[str, list[bool]] = defaultdict(list)
                    for key, component_rows in components:
                        target = choose(component_rows, f"label_{policy}_score")
                        predicted = choose(component_rows, f"predicted_{policy}_score")
                        correct = predicted == target
                        all_m_correct = target == "M"
                        correctness.append(correct)
                        all_m_correctness.append(all_m_correct)
                        by_sequence[key[3]].append(correct)
                        all_m_by_sequence[key[3]].append(all_m_correct)
                    per_sequence = {sequence: sum(values) / len(values)
                                    for sequence, values in sorted(by_sequence.items())}
                    all_m_per_sequence = {sequence: sum(values) / len(values)
                                          for sequence, values in sorted(all_m_by_sequence.items())}
                    results.append({
                        "model": model, "epoch": epoch, "split": split, "policy": policy,
                        "components": len(correctness),
                        "pooled_accuracy_as_executed": sum(correctness) / len(correctness),
                        "video_balanced_accuracy_protocol_intended":
                            sum(per_sequence.values()) / len(per_sequence),
                        "per_sequence_accuracy": per_sequence,
                        "always_M_pooled_accuracy": sum(all_m_correctness) / len(all_m_correctness),
                        "always_M_video_balanced_accuracy":
                            sum(all_m_per_sequence.values()) / len(all_m_per_sequence),
                        "always_M_per_sequence_accuracy": all_m_per_sequence,
                    })
    result = {
        "status": "COMPLETED_POST_RESULT_AUDIT_ADDENDUM",
        "discovered_after_result_inspection": True,
        "reason": "protocol declared video-balanced weighting but decision accuracy implementation pooled components",
        "primary_MSE_affected": False,
        "original_metrics_overwritten": False,
        "predictions": str(args.predictions), "predictions_sha256": sha256(args.predictions),
        "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "rows": len(results),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
