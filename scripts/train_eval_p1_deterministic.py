#!/usr/bin/env python3
"""Fixed-budget full P1 training and offline motion-dev evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--dev-samples", type=Path, required=True)
    parser.add_argument("--dev-sha256", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--normalization-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser.parse_args()


def safe_device_record() -> dict[str, object]:
    physical = os.environ.get("MOT_PHYSICAL_GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical is None or int(physical) not in range(1, 8):
        raise RuntimeError("physical GPU must be one of 1..7")
    if visible != physical or os.environ.get("MOT_PROCESS_DEVICE") != "cuda:0":
        raise RuntimeError("queue GPU mapping is inconsistent")
    return {"physical_gpu": int(physical), "CUDA_VISIBLE_DEVICES": visible, "process_device": "cuda:0"}


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    required = (
        "history_states", "history_frames", "history_valid", "target_frame", "fps",
        "target_delta", "target_state", "sequence_index", "gt_id", "frame_gap",
        "positive_detection_id",
    )
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key]) for key in required}


def torch_batch(arrays: dict[str, np.ndarray], indices: np.ndarray, device: object) -> dict[str, object]:
    import torch
    float_keys = {"history_states", "fps", "target_delta"}
    keys = ("history_states", "history_frames", "history_valid", "target_frame", "fps", "target_delta")
    return {
        key: torch.from_numpy(np.ascontiguousarray(arrays[key][indices])).to(
            device=device, dtype=torch.float32 if key in float_keys else None
        )
        for key in keys
    }


def predict_normalized_delta(model: object, arrays: dict[str, np.ndarray], batch_size: int, device: object) -> np.ndarray:
    import torch
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays["target_frame"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays["target_frame"])))
            batch = torch_batch(arrays, indices, device)
            output = model(
                batch["history_states"], batch["history_frames"], batch["history_valid"],
                batch["target_frame"], batch["fps"],
            )
            outputs.append(output.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    artifact_text = os.environ.get("MOT_JOB_ARTIFACT_DIR")
    if not artifact_text:
        raise RuntimeError("formal training must run inside the async queue")
    output_dir = Path(artifact_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, expected in {
        args.train_samples: args.train_sha256,
        args.dev_samples: args.dev_sha256,
        args.normalization: args.normalization_sha256,
        args.protocol: args.protocol_sha256,
    }.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    device_record = safe_device_record()
    import torch
    from mot_research.evaluation.motion_prediction import (
        constant_velocity_prediction,
        evaluate_prediction_set,
        zero_delta_prediction,
    )
    from mot_research.models.deterministic_motion import DeterministicMotionGRU, MotionNormalization

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal queue job must see exactly one CUDA device")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")

    normalization_mapping = json.loads(args.normalization.read_text())
    normalization = MotionNormalization.from_mapping(normalization_mapping)
    delta_std = np.asarray(normalization.delta_std, dtype=np.float64)
    train = load_arrays(args.train_samples)
    dev = load_arrays(args.dev_samples)
    model = DeterministicMotionGRU(normalization, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.MSELoss()
    generator = np.random.default_rng(args.seed)
    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        permutation = generator.permutation(len(train["target_frame"]))
        total_loss = 0.0
        total_samples = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            batch = torch_batch(train, indices, device)
            target = model.normalize_delta(batch["target_delta"])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["history_states"], batch["history_frames"], batch["history_valid"],
                batch["target_frame"], batch["fps"],
            )
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(indices)
            total_samples += len(indices)
        epoch_losses.append({"epoch": epoch + 1, "normalized_mse": total_loss / total_samples})

    learned_normalized_delta = predict_normalized_delta(model, dev, args.eval_batch_size, device)
    learned_delta = learned_normalized_delta * delta_std + np.asarray(normalization.delta_mean)
    last_dev = dev["history_states"][:, -1]
    predictions = {
        "learned_gru": last_dev + learned_delta,
        "zero_delta": zero_delta_prediction(dev["history_states"], dev["history_valid"]),
        "constant_velocity": constant_velocity_prediction(
            dev["history_states"], dev["history_frames"], dev["history_valid"], dev["target_frame"]
        ),
    }
    dev_metrics = {
        name: evaluate_prediction_set(prediction, dev, delta_std)
        for name, prediction in predictions.items()
    }
    train_normalized_delta = predict_normalized_delta(model, train, args.eval_batch_size, device)
    train_target = (train["target_delta"] - np.asarray(normalization.delta_mean)) / delta_std
    train_normalized_mse = float(np.mean((train_normalized_delta - train_target) ** 2))

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {"name": "DeterministicMotionGRU", "hidden_size": args.hidden_size, "num_layers": 1},
        "normalization": normalization_mapping,
        "seed": args.seed,
        "protocol_sha256": args.protocol_sha256,
        "train_samples_sha256": args.train_sha256,
        "dev_samples_sha256": args.dev_sha256,
    }, checkpoint_path)
    reloaded = DeterministicMotionGRU(normalization, hidden_size=args.hidden_size).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"])
    reload_prediction = predict_normalized_delta(reloaded, dev, args.eval_batch_size, device)
    reload_max_abs_error = float(np.max(np.abs(reload_prediction - learned_normalized_delta)))
    if reload_max_abs_error != 0.0:
        raise RuntimeError(f"checkpoint reload prediction mismatch: {reload_max_abs_error}")

    np.savez_compressed(
        output_dir / "dev_predictions.npz",
        sequence_index=dev["sequence_index"], gt_id=dev["gt_id"], target_frame=dev["target_frame"],
        frame_gap=dev["frame_gap"], target_state=dev["target_state"],
        learned_gru=predictions["learned_gru"], zero_delta=predictions["zero_delta"],
        constant_velocity=predictions["constant_velocity"],
    )
    metrics = {
        "execution_status": "PASS",
        "research_gate": "PENDING_MULTI_SEED_AGGREGATION",
        "seed": args.seed,
        "train_sample_count": int(len(train["target_frame"])),
        "dev_sample_count": int(len(dev["target_frame"])),
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "epoch_losses": epoch_losses,
            "final_full_train_normalized_mse": train_normalized_mse,
        },
        "dev": dev_metrics,
        "checkpoint_reload_max_abs_error": reload_max_abs_error,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dev_predictions_sha256": sha256_file(output_dir / "dev_predictions.npz"),
        "protocol_sha256": args.protocol_sha256,
        "train_samples_sha256": args.train_sha256,
        "dev_samples_sha256": args.dev_sha256,
        "normalization_sha256": args.normalization_sha256,
        "device": device_record,
        "torch_device_name": torch.cuda.get_device_name(0),
        "limitations": [
            "Offline GT targets and GT-aligned detector histories are used.",
            "Offline prediction metrics are not closed-loop tracking metrics.",
            "Motion-dev contains four deliberately selected G2 anchor sequences.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "runtime.json").write_text(json.dumps({
        "argv": sys.argv, "python": sys.executable, "python_version": sys.version,
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "MOT_PHYSICAL_GPU", "MOT_PROCESS_DEVICE", "MOT_QUEUE_ID",
            "MOT_JOB_ID", "MOT_ATTEMPT", "CUBLAS_WORKSPACE_CONFIG",
        )},
        "source_hashes": {
            "training_script": sha256_file(Path(__file__)),
            "model": sha256_file(Path(__file__).resolve().parents[1] / "src/mot_research/models/deterministic_motion.py"),
            "evaluation": sha256_file(Path(__file__).resolve().parents[1] / "src/mot_research/evaluation/motion_prediction.py"),
        },
    }, indent=2) + "\n")
    print(json.dumps({
        "execution_status": "PASS", "seed": args.seed,
        "train_normalized_mse": train_normalized_mse,
        "dev_macro_sequence": {name: value["macro_sequence"] for name, value in dev_metrics.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
