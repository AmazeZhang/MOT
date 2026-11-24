#!/usr/bin/env python3
"""Freeze S05-B traces and analysis source before offline GT labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime_lock.exists() or args.output_root.exists():
        raise FileExistsError("analysis lock or output already exists")
    sequences = sorted(path.name for path in args.coverage_root.glob("dancetrack*")
                       if (path / "engineering_checks.json").exists())
    if len(sequences) != 24:
        raise RuntimeError(f"expected 24 completed sequences, got {len(sequences)}")
    inputs, per_sequence, gt_hashes = [], {}, {}
    dataset_root = Path("/media/imc/data/mot-data/datasets/dancetrack/train")
    for sequence in sequences:
        seq_root = args.coverage_root / sequence
        check_path = seq_root / "engineering_checks.json"
        runtime_path = seq_root / "runtime_lock.json"
        trace_path = seq_root / "traces/M" / f"{sequence}.jsonl"
        check = json.loads(check_path.read_text())
        runtime = json.loads(runtime_path.read_text())
        if check["status"] != "ENGINEERING_PASS" or not check["M_output_exact"]:
            raise RuntimeError(f"{sequence}: engineering gate did not pass")
        if sha256(trace_path) != check["result"]["trace_sha256"]:
            raise RuntimeError(f"{sequence}: trace hash mismatch")
        detection_path = Path(runtime["detection_root"]) / f"{sequence}.csv"
        if sha256(detection_path) != runtime["input_detection_sha256"][sequence]:
            raise RuntimeError(f"{sequence}: detection hash mismatch")
        for role, path in (("trace", trace_path), ("engineering_check", check_path),
                           ("runtime_lock", runtime_path), ("detection", detection_path)):
            inputs.append({"sequence": sequence, "role": role, "path": str(path),
                           "sha256": sha256(path)})
        per_sequence[sequence] = {"trace_path": str(trace_path),
                                  "detection_path": str(detection_path)}
        gt_hashes[sequence] = sha256(dataset_root / sequence / "gt/gt.txt")
    lock = {
        "run_id": "s05b_candidate_coverage_analysis_attempt1",
        "status": "locked_before_GT_labeling",
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256(ROOT / args.protocol),
        "analysis_code_sha256": sha256(ROOT / "scripts/analyze_s05_candidate_coverage.py"),
        "dependency_code_sha256": {
            "src/mot_research/analysis/failure_events.py": sha256(ROOT / "src/mot_research/analysis/failure_events.py")},
        "coverage_root": str(args.coverage_root),
        "output_root": str(args.output_root),
        "dataset_root": str(dataset_root),
        "sequences": sequences,
        "inputs": inputs,
        "per_sequence": per_sequence,
        "gt_file_sha256": gt_hashes,
        "cuda_visible_devices": "",
        "physical_gpus": [],
        "logical_to_physical": {},
        "GPU0_used": False,
    }
    args.runtime_lock.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "sequences": len(sequences),
                      "inputs": len(inputs)}, sort_keys=True))


if __name__ == "__main__":
    main()
