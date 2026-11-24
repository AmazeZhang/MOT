#!/usr/bin/env python3
"""Freeze the S06 seed-13 training lock and source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CODE = ["scripts/train_s06_component_selector.py",
        "src/mot_research/models/component_action_selector.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE).stdout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    if sha(args.protocol) != args.protocol_sha256:
        raise RuntimeError("protocol changed")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    snapshot = args.output_root / "source_snapshot"
    for path in [*CODE, args.protocol.relative_to(ROOT).as_posix()]:
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    patch = args.output_root / "source_dirty.patch"
    patch.write_text(git("diff", "--binary", "HEAD"))
    lock = {
        "run_id": args.run_id,
        "status": "locked_before_training",
        "seed": 13,
        "models": ["Local", "Context"],
        "protocol_path": str(args.protocol), "protocol_sha256": args.protocol_sha256,
        "samples_path": str(args.samples), "samples_sha256": sha(args.samples),
        "code_hashes": {path: sha(ROOT / path) for path in CODE},
        "project_commit": git("rev-parse", "HEAD").strip(),
        "project_status": git("status", "--short"),
        "dirty_patch_sha256": sha(patch),
        "output_root": str(args.output_root),
        "CUDA_VISIBLE_DEVICES": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "physical_gpus": [1],
        "logical_to_physical": {"cuda:0": 1}, "physical_GPU0_forbidden": True,
    }
    (args.output_root / "runtime_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "run_id": lock["run_id"],
                      "samples_sha256": lock["samples_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
