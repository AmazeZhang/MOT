#!/usr/bin/env python3
"""Train frozen S06 Local and Context seed-13 heads; save every fifth epoch."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from mot_research.models.component_action_selector import (
    CONTINUOUS_FEATURES, ComponentActionSelector, FeatureNormalization, parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalization(rows: list[dict]) -> FeatureNormalization:
    values = [[edge[name] for name in CONTINUOUS_FEATURES]
              for row in rows if row["split"] == "train18" and row["fully_labelable"]
              for edge in row["edges"]]
    array = np.asarray(values, dtype=np.float64)
    mean, std = array.mean(axis=0), array.std(axis=0)
    std[std < 1e-8] = 1.0
    return FeatureNormalization(tuple(mean.tolist()), tuple(std.tolist()))


def action_examples(rows: list[dict], split: str) -> list[tuple[dict, dict]]:
    return [(row, action) for row in rows if row["split"] == split and row["fully_labelable"]
            for action in row["actions"]]


def tensors(row: dict, action: dict, device: torch.device):
    continuous = torch.tensor([[edge[name] for name in CONTINUOUS_FEATURES]
                               for edge in row["edges"]], dtype=torch.float32, device=device)
    binary = torch.tensor([[float(edge["cosine_valid"]), float(value)]
                           for edge, value in zip(row["edges"], action["differs_from_M"])],
                          dtype=torch.float32, device=device)
    detection = torch.tensor([edge["detection_row"] for edge in row["edges"]], device=device)
    tracker = torch.tensor([edge["tracker_row"] for edge in row["edges"]], device=device)
    selected_set = set(action["selected_edge_indices"])
    selected = torch.tensor([i in selected_set for i in range(len(row["edges"]))], device=device)
    component = torch.tensor([np.log1p(row["detection_count"]), np.log1p(row["track_count"]),
                              np.log1p(len(selected_set))], dtype=torch.float32, device=device)
    target = torch.tensor([action["g_div_k"], action["h_div_k"]],
                          dtype=torch.float32, device=device)
    return continuous, binary, detection, tracker, selected, component, target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-lock", type=Path, required=True)
    args = p.parse_args()
    lock = json.loads(args.runtime_lock.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or lock["CUDA_VISIBLE_DEVICES"] != "1":
        raise RuntimeError("training requires CUDA_VISIBLE_DEVICES=1 (physical GPU1)")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or \
            lock.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    for path, expected in lock["code_hashes"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"training source changed: {path}")
    samples = Path(lock["samples_path"])
    if sha(samples) != lock["samples_sha256"]:
        raise RuntimeError("samples changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    norm = normalization(rows)
    train = action_examples(rows, "train18")
    counts = Counter(row["sequence"] for row, _ in train)
    weights = np.asarray([1.0 / counts[row["sequence"]] for row, _ in train], dtype=np.float64)
    weights /= weights.sum()
    output = Path(lock["output_root"])
    curves_path = output / "learning_curves.csv"
    with curves_path.open("w", newline="") as curve_file:
        writer = csv.DictWriter(curve_file, fieldnames=["model", "seed", "epoch", "split",
                                                       "metric", "value", "samples_seen",
                                                       "elapsed_seconds", "checkpoint"])
        writer.writeheader()
        for mode in ("Local", "Context"):
            seed = 13
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
            model = ComponentActionSelector(mode, norm).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            rng = np.random.default_rng(seed)
            begun = time.perf_counter()
            samples_seen = 0
            model_dir = output / mode
            model_dir.mkdir()
            for epoch in range(1, 31):
                model.train()
                order = rng.choice(len(train), size=len(train), replace=True, p=weights)
                epoch_loss = 0.0
                optimizer.zero_grad(set_to_none=True)
                for step, index in enumerate(order, 1):
                    row, action = train[int(index)]
                    *inputs, target = tensors(row, action, device)
                    prediction = model(*inputs)
                    loss = torch.mean((prediction - target) ** 2) / 128
                    loss.backward()
                    epoch_loss += float(loss.detach().cpu()) * 128
                    samples_seen += 1
                    if step % 128 == 0 or step == len(order):
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                checkpoint = ""
                if epoch % 5 == 0:
                    checkpoint = f"{mode}/epoch_{epoch:02d}.pt"
                    torch.save({"schema_version": "s06-selector-v1", "mode": mode,
                                "seed": seed, "epoch": epoch,
                                "normalization": norm.as_dict(),
                                "parameter_count": parameter_count(model),
                                "model_state_dict": model.state_dict()}, output / checkpoint)
                writer.writerow({"model": mode, "seed": seed, "epoch": epoch,
                                 "split": "train18", "metric": "video_balanced_resampled_MSE",
                                 "value": epoch_loss / len(order), "samples_seen": samples_seen,
                                 "elapsed_seconds": time.perf_counter() - begun,
                                 "checkpoint": checkpoint})
                curve_file.flush()
    summary = {"status": "COMPLETED", "seed": 13, "device": "cuda:0",
               "CUDA_VISIBLE_DEVICES": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "physical_gpus": [1],
               "logical_to_physical": {"cuda:0": 1}, "GPU0_used": False,
               "train_actions": len(train), "normalization": norm.as_dict(),
               "models": {mode: {"parameter_count": parameter_count(
                   ComponentActionSelector(mode, norm)),
                   "checkpoints": [{"epoch": epoch,
                                    "path": str(output / mode / f"epoch_{epoch:02d}.pt"),
                                    "sha256": sha(output / mode / f"epoch_{epoch:02d}.pt")}
                                   for epoch in range(5, 31, 5)]}
                          for mode in ("Local", "Context")},
               "learning_curves": str(curves_path),
               "learning_curves_sha256": sha(curves_path)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
