#!/usr/bin/env python3
"""Freeze S05-A tracker and evaluator inputs before TrackEval loads GT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TE_ROOT = ROOT / "third_party/TrackEval"
TE_FILES = [
    "trackeval/eval.py",
    "trackeval/datasets/_base_dataset.py",
    "trackeval/datasets/mot_challenge_2d_box.py",
    "trackeval/metrics/hota.py",
    "trackeval/metrics/clear.py",
    "trackeval/metrics/identity.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime_lock.exists() or args.output_root.exists():
        raise FileExistsError("evaluation lock or output already exists")
    protocol = args.protocol.resolve()
    sequences = ["dancetrack0074", "dancetrack0032", "dancetrack0069", "dancetrack0015",
                 "dancetrack0023", "dancetrack0080", "dancetrack0049", "dancetrack0083"]
    artifacts = []
    checks = []
    gt_hashes = {}
    for sequence in sequences:
        seq_root = args.tracking_root / sequence
        check_path = seq_root / "engineering_checks.json"
        runtime_path = seq_root / "runtime_lock.json"
        check = json.loads(check_path.read_text())
        runtime = json.loads(runtime_path.read_text())
        if check["status"] != "ENGINEERING_PASS" or set(check["results"]) != {"G", "L"}:
            raise RuntimeError(f"{sequence}: incomplete G/L engineering output")
        for condition in ("G", "L"):
            path = seq_root / "trackers" / condition / f"{sequence}.txt"
            actual = sha256(path)
            if actual != check["results"][condition]["track_sha256"]:
                raise RuntimeError(f"{sequence}/{condition}: tracker hash mismatch")
            artifacts.append({"sequence": sequence, "condition": condition,
                              "path": str(path), "sha256": actual})
        m_path = Path(runtime["reused_M_tracker"])
        if sha256(m_path) != runtime["reused_M_tracker_sha256"]:
            raise RuntimeError(f"{sequence}/M: reused tracker hash mismatch")
        artifacts.append({"sequence": sequence, "condition": "M", "path": str(m_path),
                          "sha256": runtime["reused_M_tracker_sha256"]})
        checks.append({"sequence": sequence, "path": str(check_path),
                       "sha256": sha256(check_path), "runtime_lock_sha256": sha256(runtime_path)})
        gt_path = Path(runtime["train_root"]) / sequence / "gt/gt.txt"
        gt_hashes[sequence] = sha256(gt_path)
    dirty_patch = subprocess.check_output(["git", "diff", "--binary"], cwd=TE_ROOT)
    lock = {
        "run_id": "s05a_trackeval_development8_attempt1",
        "status": "locked_before_evaluation",
        "protocol_path": str(protocol.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol),
        "evaluation_code_sha256": sha256(ROOT / "scripts/evaluate_s05a_stability.py"),
        "projection_helper_sha256": sha256(ROOT / "scripts/analyze_e1_mature_reemit.py"),
        "tracking_root": str(args.tracking_root),
        "output_root": str(args.output_root),
        "dataset_root": "/media/imc/data/mot-data/datasets/dancetrack/train",
        "sequences": sequences,
        "conditions": ["G", "L", "M"],
        "tracker_artifacts": artifacts,
        "engineering_checks": checks,
        "gt_file_sha256": gt_hashes,
        "trackeval_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TE_ROOT, text=True).strip(),
        "trackeval_dirty_patch_sha256": hashlib.sha256(dirty_patch).hexdigest(),
        "trackeval_file_sha256": {path: sha256(TE_ROOT / path) for path in TE_FILES},
        "cuda_visible_devices": "",
        "physical_gpus": [],
        "logical_to_physical": {},
        "GPU0_used": False,
    }
    args.runtime_lock.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_lock.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "artifacts": len(artifacts),
                      "runtime_lock": str(args.runtime_lock)}, sort_keys=True))


if __name__ == "__main__":
    main()
