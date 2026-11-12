#!/usr/bin/env python3
"""Aggregate completed P1 seed jobs and apply the frozen offline feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from mot_research.evaluation.motion_prediction import offline_feasibility_gate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-aggregate", type=Path, required=True)
    parser.add_argument("--queue-aggregate-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-mse-ratio-max", type=float, required=True)
    parser.add_argument("--every-seed-mse-ratio-max", type=float, required=True)
    parser.add_argument("--gap-2-31-mse-ratio-max", type=float, required=True)
    parser.add_argument("--macro-iou-drop-max", type=float, required=True)
    parser.add_argument("--seed-mse-relative-range-max", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, expected in {
        args.queue_aggregate: args.queue_aggregate_sha256,
        args.protocol: args.protocol_sha256,
    }.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    aggregate = json.loads(args.queue_aggregate.read_text())
    if not aggregate["complete"] or aggregate["passed_job_count"] != aggregate["required_job_count"]:
        raise RuntimeError("all seed jobs must complete successfully before research aggregation")
    seed_metrics = [job["metrics"] for job in aggregate["passed_jobs"]]
    if len(seed_metrics) != 3 or len({item["seed"] for item in seed_metrics}) != 3:
        raise RuntimeError("expected exactly three distinct completed seeds")
    if any(item["protocol_sha256"] != args.protocol_sha256 for item in seed_metrics):
        raise RuntimeError("seed metrics protocol hash mismatch")
    result = offline_feasibility_gate(
        seed_metrics,
        mean_mse_ratio_max=args.mean_mse_ratio_max,
        every_seed_mse_ratio_max=args.every_seed_mse_ratio_max,
        gap_2_31_mse_ratio_max=args.gap_2_31_mse_ratio_max,
        macro_iou_drop_max=args.macro_iou_drop_max,
        seed_mse_relative_range_max=args.seed_mse_relative_range_max,
    )
    result.update({
        "status": "PASS",
        "scope": "P1 offline multi-seed feasibility gate on frozen motion-dev",
        "seed_job_count": len(seed_metrics),
        "queue_aggregate": str(args.queue_aggregate.resolve()),
        "queue_aggregate_sha256": args.queue_aggregate_sha256,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": args.protocol_sha256,
        "limitations": [
            "The gate uses offline GT targets and GT-aligned detector histories.",
            "GO authorizes only the preregistered natural closed-loop test, not a tracking claim.",
            "Motion-dev contains four selected G2 anchors and is not final confirmation evidence.",
        ],
    })
    partial = args.output_dir.with_name(args.output_dir.name + ".partial")
    if args.output_dir.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite offline summary")
    partial.mkdir(parents=True)
    try:
        (partial / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (partial / "per_seed_projection.json").write_text(json.dumps([
            {
                "seed": item["seed"],
                "training": item["training"],
                "learned_macro_sequence": item["dev"]["learned_gru"]["macro_sequence"],
                "learned_gap_2_31": item["dev"]["learned_gru"]["by_gap"]["gap_2_31"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "dev_predictions_sha256": item["dev_predictions_sha256"],
            }
            for item in seed_metrics
        ], indent=2) + "\n")
        os.replace(partial, args.output_dir)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
