#!/usr/bin/env python3
"""Apply the frozen AssA -> HOTA -> earlier-epoch S06 checkpoint rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates = {"Local": [], "Context": []}
    for mode in candidates:
        for epoch in range(5, 31, 5):
            method = f"{mode}_risk_e{epoch:02d}"
            summary_path = args.evaluation_root / method / "summary.json"
            summary = json.loads(summary_path.read_text())
            if summary["execution_status"] != "PASS" or summary["sequence_count"] != 6:
                raise RuntimeError(f"incomplete evaluation {method}")
            metric = summary["official_trackeval_combined"]
            checkpoint = args.training_root / mode / f"epoch_{epoch:02d}.pt"
            candidates[mode].append({
                "mode": mode, "epoch": epoch, "method": method,
                "AssA": metric["AssA"], "HOTA": metric["HOTA"],
                "IDF1": metric["IDF1"], "IDSW": metric["IDSW"],
                "evaluation_summary": str(summary_path),
                "evaluation_summary_sha256": sha(summary_path),
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            })
    selected = {}
    for mode, values in candidates.items():
        selected[mode] = max(values, key=lambda x: (x["AssA"], x["HOTA"], -x["epoch"]))
    output = {"status": "COMPLETED", "selection_rule": "max_AssA_then_HOTA_then_earlier_epoch",
              "policy_used_for_selection": "risk", "candidates": candidates,
              "selected": selected, "evaluation8_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
