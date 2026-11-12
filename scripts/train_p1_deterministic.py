#!/usr/bin/env python3
"""Train the minimal P1 deterministic GRU, including the overfit smoke gate."""

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
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--samples-sha256", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--normalization-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--max-final-loss", type=float, default=0.05)
    parser.add_argument("--max-final-over-initial", type=float, default=0.1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def require_safe_device(device: str) -> dict[str, object]:
    physical_text = os.environ.get("MOT_PHYSICAL_GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device == "cuda":
        if physical_text is None or int(physical_text) not in range(1, 8):
            raise RuntimeError("CUDA training requires MOT_PHYSICAL_GPU in physical 1..7")
        if visible != physical_text:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must equal the queue physical GPU")
        if os.environ.get("MOT_PROCESS_DEVICE") != "cuda:0":
            raise RuntimeError("queue must map the selected physical GPU to cuda:0")
    elif visible not in {None, ""}:
        raise RuntimeError("CPU training requires no visible CUDA device")
    return {
        "physical_gpu": int(physical_text) if physical_text is not None else None,
        "CUDA_VISIBLE_DEVICES": visible,
        "process_device": "cuda:0" if device == "cuda" else "cpu",
    }


def main() -> None:
    args = parse_args()
    output_dir_text = args.output_dir or os.environ.get("MOT_JOB_ARTIFACT_DIR")
    if not output_dir_text:
        raise RuntimeError("output directory must be explicit or supplied by the queue")
    output_dir = Path(output_dir_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hashes = {
        args.samples: args.samples_sha256,
        args.normalization: args.normalization_sha256,
        args.protocol: args.protocol_sha256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")

    device_record = require_safe_device(args.device)
    import torch
    from mot_research.models.deterministic_motion import (
        DeterministicMotionGRU,
        MotionNormalization,
        deterministic_sample_indices,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("queue job must see exactly one CUDA device")
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    normalization_mapping = json.loads(args.normalization.read_text())
    normalization = MotionNormalization.from_mapping(normalization_mapping)
    with np.load(args.samples, allow_pickle=False) as archive:
        selected, subset_hash = deterministic_sample_indices(
            archive["sequence_index"], archive["gt_id"], archive["target_frame"],
            count=args.sample_count, seed=args.seed,
        )
        numpy_batch = {
            key: np.ascontiguousarray(archive[key][selected])
            for key in (
                "history_states", "history_frames", "history_valid", "target_frame",
                "fps", "target_delta", "sequence_index", "gt_id",
            )
        }

    float_keys = {"history_states", "fps", "target_delta"}
    tensors = {
        key: torch.from_numpy(value).to(
            device=device,
            dtype=torch.float32 if key in float_keys else None,
        )
        for key, value in numpy_batch.items()
    }
    model = DeterministicMotionGRU(
        normalization, hidden_size=args.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    targets = model.normalize_delta(tensors["target_delta"])

    def loss_value() -> torch.Tensor:
        predictions = model(
            tensors["history_states"], tensors["history_frames"],
            tensors["history_valid"], tensors["target_frame"], tensors["fps"],
        )
        return criterion(predictions, targets)

    model.train()
    with torch.no_grad():
        initial_loss = float(loss_value().item())
    losses: list[float] = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    model.eval()
    with torch.no_grad():
        final_predictions = model(
            tensors["history_states"], tensors["history_frames"],
            tensors["history_valid"], tensors["target_frame"], tensors["fps"],
        )
        final_loss = float(criterion(final_predictions, targets).item())

    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": {"name": "DeterministicMotionGRU", "hidden_size": args.hidden_size, "num_layers": 1},
        "normalization": normalization_mapping,
        "seed": args.seed,
        "subset_sample_key_sha256": subset_hash,
        "protocol_sha256": args.protocol_sha256,
        "samples_sha256": args.samples_sha256,
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = DeterministicMotionGRU(normalization, hidden_size=args.hidden_size).to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reload_predictions = reloaded(
            tensors["history_states"], tensors["history_frames"],
            tensors["history_valid"], tensors["target_frame"], tensors["fps"],
        )
    reload_max_abs_error = float((reload_predictions - final_predictions).abs().max().item())

    ratio = final_loss / initial_loss
    passed = (
        final_loss <= args.max_final_loss
        and ratio <= args.max_final_over_initial
        and reload_max_abs_error == 0.0
    )
    repo_root = Path(__file__).resolve().parents[1]
    metrics = {
        "status": "PASS" if passed else "FAIL",
        "scope": "engineering-only tiny-subset deterministic overfit smoke",
        "initial_normalized_mse": initial_loss,
        "final_normalized_mse": final_loss,
        "final_over_initial": ratio,
        "minimum_training_step_loss": min(losses),
        "acceptance": {
            "max_final_loss": args.max_final_loss,
            "max_final_over_initial": args.max_final_over_initial,
            "checkpoint_reload_max_abs_error_required": 0.0,
        },
        "checkpoint_reload_max_abs_error": reload_max_abs_error,
        "sample_count": args.sample_count,
        "subset_sample_key_sha256": subset_hash,
        "steps": args.steps,
        "seed": args.seed,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "device": device_record,
        "torch_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "protocol_sha256": args.protocol_sha256,
        "samples_sha256": args.samples_sha256,
        "normalization_sha256": args.normalization_sha256,
        "limitations": [
            "Training and evaluation use the same deliberately tiny subset.",
            "This validates plumbing and capacity only, not held-out prediction or tracking quality.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "loss_curve.json").write_text(json.dumps({
        "initial": initial_loss,
        "steps": losses,
        "final": final_loss,
    }) + "\n")
    runtime = {
        "argv": sys.argv,
        "python": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "MOT_PHYSICAL_GPU", "MOT_PROCESS_DEVICE",
            "MOT_QUEUE_ID", "MOT_JOB_ID", "MOT_ATTEMPT", "CUBLAS_WORKSPACE_CONFIG",
        )},
        "source_hashes": {
            "training_script": sha256_file(Path(__file__)),
            "model": sha256_file(repo_root / "src/mot_research/models/deterministic_motion.py"),
        },
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not passed:
        raise RuntimeError("overfit smoke acceptance criteria were not met")


if __name__ == "__main__":
    main()
