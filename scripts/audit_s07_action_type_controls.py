#!/usr/bin/env python3
"""Post-result train-only action-type constant controls for S07 labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights)))


def weights(rows: list[dict]) -> list[float]:
    counts = Counter(row["sequence"] for row in rows)
    return [1.0 / (len(counts) * counts[row["sequence"]]) for row in rows]


def score(rows: list[dict], predictor) -> dict:
    row_weights = weights(rows)
    errors = []
    for row in rows:
        predicted_g, predicted_h = predictor(row["action"])
        errors.append(((predicted_g - row["label_g"]) ** 2 +
                       (predicted_h - row["label_h"]) ** 2) / 2)
    return {"actions": len(rows), "video_balanced_mean_mse": weighted_mean(errors, row_weights)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("control audit is CPU-only")
    if args.output.exists():
        raise FileExistsError(args.output)
    all_rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    rows = [row for row in all_rows if row["model"] == "Local" and row["epoch"] == 5]
    assert len(rows) == 252
    # Labels and keys must repeat exactly for every checkpoint.
    reference = {(row["split"], row["component_id"], row["action"]):
                 (row["sequence"], row["label_g"], row["label_h"]) for row in rows}
    for row in all_rows:
        key = (row["split"], row["component_id"], row["action"])
        assert reference[key] == (row["sequence"], row["label_g"], row["label_h"])
    train = [row for row in rows if row["split"] == "train18"]
    train_non_m = [row for row in train if row["action"] != "M"]
    non_m_weights = weights(train_non_m)
    non_m_mean = tuple(weighted_mean([row[key] for row in train_non_m], non_m_weights)
                       for key in ("label_g", "label_h"))
    action_means = {}
    for action in sorted({row["action"] for row in train}):
        selected = [row for row in train if row["action"] == action]
        selected_weights = weights(selected)
        action_means[action] = tuple(weighted_mean([row[key] for row in selected], selected_weights)
                                     for key in ("label_g", "label_h"))
    controls = {}
    for split in ("train18", "internal_dev6"):
        selected = [row for row in rows if row["split"] == split]
        controls[split] = {
            "analytic_M_zero_shared_non_M_train_mean": score(
                selected, lambda action: (0.0, 0.0) if action == "M" else non_m_mean),
            "train_action_type_means_M_G_V": score(selected, lambda action: action_means[action]),
        }
    result = {
        "status": "COMPLETED_POST_RESULT_EXPLORATORY_ADDENDUM",
        "discovered_after_result_inspection": True,
        "confirmatory_control": False,
        "purpose": "test whether frozen action-agnostic controls were too weak",
        "original_results_overwritten": False,
        "prediction_source": str(args.predictions),
        "prediction_source_sha256": sha256(args.predictions),
        "train_video_balanced_shared_non_M_mean_g_h": list(non_m_mean),
        "train_video_balanced_action_type_means_g_h":
            {key: list(value) for key, value in action_means.items()},
        "controls": controls,
        "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
        "interpretation": "descriptive confound check only; not a frozen model-selection control",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
