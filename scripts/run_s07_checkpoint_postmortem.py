#!/usr/bin/env python3
"""CPU-only fixed-sample diagnostics for all frozen S06 selector checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from mot_research.models.component_action_selector import (
    CONTINUOUS_FEATURES,
    ComponentActionSelector,
    FeatureNormalization,
)


ROOT = Path(__file__).resolve().parents[1]
EPOCHS = (5, 10, 15, 20, 25, 30)
MODES = ("Local", "Context")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensors(row: dict, action: dict) -> tuple[torch.Tensor, ...]:
    continuous = torch.tensor(
        [[edge[name] for name in CONTINUOUS_FEATURES] for edge in row["edges"]],
        dtype=torch.float32,
    )
    binary = torch.tensor(
        [[float(edge["cosine_valid"]), float(value)]
         for edge, value in zip(row["edges"], action["differs_from_M"])],
        dtype=torch.float32,
    )
    detection = torch.tensor([edge["detection_row"] for edge in row["edges"]])
    tracker = torch.tensor([edge["tracker_row"] for edge in row["edges"]])
    selected_indices = set(action["selected_edge_indices"])
    selected = torch.tensor([i in selected_indices for i in range(len(row["edges"]))])
    component = torch.tensor(
        [np.log1p(row["detection_count"]), np.log1p(row["track_count"]),
         np.log1p(len(selected_indices))], dtype=torch.float32,
    )
    return continuous, binary, detection, tracker, selected, component


def weighted_mean(values: list[float], weights: list[float]) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=np.asarray(weights)))


def choose(rows: list[dict], field: str) -> str:
    # Original action order is M, G, V with earlier actions winning exact ties.
    best = max((float(row[field]), -index, row["action"])
               for index, row in enumerate(rows))
    return best[2] if best[0] > 0 else "M"


def summarize(records: list[dict], split: str) -> dict:
    rows = [row for row in records if row["split"] == split]
    counts = Counter(row["sequence"] for row in rows)
    weights = [1.0 / (len(counts) * counts[row["sequence"]]) for row in rows]
    by_component: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_component[row["component_id"]].append(row)
    m_rows = [row for row in rows if row["action"] == "M"]
    m_counts = Counter(row["sequence"] for row in m_rows)
    m_weights = [1.0 / (len(m_counts) * m_counts[row["sequence"]]) for row in m_rows]
    result = {
        "actions": len(rows),
        "components": len(by_component),
        "represented_sequences": sorted(counts),
    }
    for target in ("g", "h"):
        errors = [row[f"predicted_{target}"] - row[f"label_{target}"] for row in rows]
        result[f"{target}_mse"] = weighted_mean([value * value for value in errors], weights)
        result[f"{target}_mae"] = weighted_mean([abs(value) for value in errors], weights)
    result["mean_mse"] = (result["g_mse"] + result["h_mse"]) / 2
    result["mean_mae"] = (result["g_mae"] + result["h_mae"]) / 2
    result["M_zero_g_mae"] = weighted_mean(
        [abs(row["predicted_g"]) for row in m_rows], m_weights)
    result["M_zero_h_mae"] = weighted_mean(
        [abs(row["predicted_h"]) for row in m_rows], m_weights)
    for policy in ("no_extra", "risk"):
        true_key, predicted_key = f"label_{policy}_score", f"predicted_{policy}_score"
        agreements = []
        rejected = 0
        predicted_non_m = 0
        true_non_m = 0
        for component_rows in by_component.values():
            predicted = choose(component_rows, predicted_key)
            target = choose(component_rows, true_key)
            agreements.append(predicted == target)
            rejected += predicted == "M" and max(row[predicted_key] for row in component_rows) <= 0
            predicted_non_m += predicted != "M"
            true_non_m += target != "M"
        result[f"{policy}_decision_accuracy"] = sum(agreements) / len(agreements)
        result[f"{policy}_predicted_non_M"] = predicted_non_m
        result[f"{policy}_true_non_M"] = true_non_m
        result[f"{policy}_all_scores_rejected"] = rejected
    return result


def constant_summary(rows: list[dict], split: str, prediction: tuple[float, float]) -> dict:
    selected = [row for row in rows if row["split"] == split and row["fully_labelable"]
                for _action in row["actions"]]
    labels = [(action["g_div_k"], action["h_div_k"])
              for row in rows if row["split"] == split and row["fully_labelable"]
              for action in row["actions"]]
    sequences = [row["sequence"] for row in rows if row["split"] == split and row["fully_labelable"]
                 for _action in row["actions"]]
    counts = Counter(sequences)
    weights = [1.0 / (len(counts) * counts[sequence]) for sequence in sequences]
    g_mse = weighted_mean([(prediction[0] - label[0]) ** 2 for label in labels], weights)
    h_mse = weighted_mean([(prediction[1] - label[1]) ** 2 for label in labels], weights)
    return {"actions": len(selected), "prediction_g_h": list(prediction),
            "g_mse": g_mse, "h_mse": h_mse, "mean_mse": (g_mse + h_mse) / 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S07 is CPU-only and requires CUDA_VISIBLE_DEVICES=''")
    if args.output_root.exists():
        raise FileExistsError(f"immutable output already exists: {args.output_root}")
    if sha256(args.protocol) != args.protocol_sha256:
        raise RuntimeError("protocol hash mismatch")
    started = time.time()
    args.output_root.mkdir(parents=True)
    rows = [json.loads(line) for line in args.samples.read_text().splitlines()]
    training = json.loads(args.training_summary.read_text())
    if sha256(args.samples) != "ccd2f00185ec9f0c7868713f175713890fa4a35799131cbe94ea99a9a2b626bd":
        raise RuntimeError("frozen samples hash mismatch")
    checkpoints = {mode: {int(item["epoch"]): item for item in training["models"][mode]["checkpoints"]}
                   for mode in MODES}
    predictions_path = args.output_root / "predictions.jsonl"
    all_records: dict[tuple[str, int], list[dict]] = {}
    checkpoint_hashes = {}
    with predictions_path.open("w") as output:
        for mode in MODES:
            for epoch in EPOCHS:
                item = checkpoints[mode][epoch]
                path = Path(item["path"])
                actual_hash = sha256(path)
                if actual_hash != item["sha256"]:
                    raise RuntimeError(f"checkpoint hash mismatch: {path}")
                checkpoint_hashes[f"{mode}_e{epoch:02d}"] = actual_hash
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                normalization = FeatureNormalization.from_dict(checkpoint["normalization"])
                model = ComponentActionSelector(mode, normalization)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                records = []
                with torch.no_grad():
                    for row in rows:
                        if not row["fully_labelable"]:
                            continue
                        for action in row["actions"]:
                            prediction = model(*tensors(row, action))
                            g_hat, h_hat = (float(value) for value in prediction)
                            record = {
                                "run_id": "s07_s06_checkpoint_postmortem_cpu_attempt1",
                                "model": mode, "epoch": epoch,
                                "checkpoint_sha256": actual_hash,
                                "split": row["split"], "sequence": row["sequence"],
                                "component_id": row["component_id"], "frame": row["frame"],
                                "action": action["name"],
                                "label_g": action["g_div_k"], "label_h": action["h_div_k"],
                                "predicted_g": g_hat, "predicted_h": h_hat,
                                "label_no_extra_score": action["g_div_k"] - action["h_div_k"],
                                "predicted_no_extra_score": g_hat - h_hat,
                                "label_risk_score": action["g_div_k"] - 2 * action["h_div_k"],
                                "predicted_risk_score": g_hat - 2 * h_hat,
                            }
                            records.append(record)
                            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                all_records[(mode, epoch)] = records
    # Freeze the video-balanced mean from train18 labels only.
    train_actions = [(row["sequence"], action) for row in rows
                     if row["split"] == "train18" and row["fully_labelable"]
                     for action in row["actions"]]
    train_counts = Counter(sequence for sequence, _ in train_actions)
    train_weights = [1.0 / (len(train_counts) * train_counts[sequence])
                     for sequence, _ in train_actions]
    train_mean = tuple(weighted_mean([action[key] for _, action in train_actions], train_weights)
                       for key in ("g_div_k", "h_div_k"))
    metrics = []
    for (mode, epoch), records in all_records.items():
        for split in ("train18", "internal_dev6"):
            metrics.append({"model": mode, "epoch": epoch, "split": split,
                            **summarize(records, split)})
    metrics_path = args.output_root / "checkpoint_metrics.csv"
    fields = ["model", "epoch", "split", "actions", "components", "represented_sequences",
              "g_mse", "h_mse", "mean_mse", "g_mae", "h_mae", "mean_mae",
              "M_zero_g_mae", "M_zero_h_mae", "no_extra_decision_accuracy",
              "no_extra_predicted_non_M", "no_extra_true_non_M", "no_extra_all_scores_rejected",
              "risk_decision_accuracy", "risk_predicted_non_M", "risk_true_non_M",
              "risk_all_scores_rejected"]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            row = dict(row); row["represented_sequences"] = ";".join(row["represented_sequences"])
            writer.writerow(row)
    controls = {split: {
        "zero": constant_summary(rows, split, (0.0, 0.0)),
        "train_video_balanced_mean": constant_summary(rows, split, train_mean),
    } for split in ("train18", "internal_dev6")}
    comparisons = []
    indexed = {(row["model"], row["epoch"], row["split"]): row for row in metrics}
    for epoch in EPOCHS:
        for split in ("train18", "internal_dev6"):
            local = indexed[("Local", epoch, split)]
            context = indexed[("Context", epoch, split)]
            comparisons.append({"epoch": epoch, "split": split,
                                "Context_minus_Local_mean_mse": context["mean_mse"] - local["mean_mse"],
                                "Context_minus_Local_no_extra_accuracy":
                                    context["no_extra_decision_accuracy"] - local["no_extra_decision_accuracy"],
                                "Context_minus_Local_risk_accuracy":
                                    context["risk_decision_accuracy"] - local["risk_decision_accuracy"]})
    ended = time.time()
    result = {
        "schema_version": "s07-checkpoint-postmortem-v1",
        "status": "COMPLETED", "scientific_scope": "FIXED_SAMPLE_DIAGNOSTIC_ONLY",
        "run_id": "s07_s06_checkpoint_postmortem_cpu_attempt1",
        "protocol": str(args.protocol), "protocol_sha256": args.protocol_sha256,
        "inputs": {"samples": str(args.samples), "samples_sha256": sha256(args.samples),
                   "training_summary": str(args.training_summary),
                   "training_summary_sha256": sha256(args.training_summary)},
        "checkpoint_sha256": checkpoint_hashes,
        "coverage": {"checkpoint_count": len(all_records), "modes": list(MODES),
                     "epochs": list(EPOCHS), "train18_actions": 208,
                     "internal_dev6_actions": 44},
        "weighting": "each represented video equal; actions within video equal",
        "train18_video_balanced_target_mean_g_h": list(train_mean),
        "constant_controls": controls,
        "checkpoint_metrics": metrics,
        "same_epoch_comparisons": comparisons,
        "resources": {"CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
                      "torch_num_threads": torch.get_num_threads(),
                      "wall_seconds": ended - started, "cpu_core_hours_upper_bound":
                          (ended - started) * torch.get_num_threads() / 3600},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__},
        "outputs": {"predictions": str(predictions_path),
                    "predictions_sha256": sha256(predictions_path),
                    "checkpoint_metrics": str(metrics_path),
                    "checkpoint_metrics_sha256": sha256(metrics_path)},
        "limitations": ["No training, calibration, threshold scan, tracking, or TrackEval",
                        "Internal-dev6 is reused for prior checkpoint selection",
                        "Only seed13 exists", "Labels are offline stable-owner proxies"],
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    runtime = {
        "run_id": result["run_id"], "status": "COMPLETED",
        "exact_argv": sys.argv, "working_directory": str(Path.cwd()),
        "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
        "source_sha256": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
                          "src/mot_research/models/component_action_selector.py":
                              sha256(ROOT / "src/mot_research/models/component_action_selector.py")},
        "protocol_sha256": args.protocol_sha256, "started_unix": started, "ended_unix": ended,
        "exit_code": 0, "output_sha256": {"summary.json": sha256(summary_path),
                                          "predictions.jsonl": sha256(predictions_path),
                                          "checkpoint_metrics.csv": sha256(metrics_path)},
    }
    runtime_path = args.output_root / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "output": str(args.output_root),
                      "wall_seconds": result["resources"]["wall_seconds"],
                      "internal_dev6": [row for row in metrics if row["split"] == "internal_dev6"]},
                     indent=2))


if __name__ == "__main__":
    main()
