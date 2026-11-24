#!/usr/bin/env python3
"""Run S05-A G/L conditions while reusing hash-verified S04 M outputs."""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import sys

from run_s01_visual_baselines import (
    all_detection_ids,
    load_features,
    read_detection_cache,
    run_condition,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=("G", "L"), default=("G", "L"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S05-A tracking must use CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if lock["protocol_sha256"] != sha256(ROOT / lock["protocol_path"]):
        raise RuntimeError("S05 protocol hash changed after runtime lock")
    for rel_path, expected in lock["code_hashes"].items():
        if sha256(ROOT / rel_path) != expected:
            raise RuntimeError(f"code SHA256 differs from lock: {rel_path}")
    sequence = lock["selected_sequences"][0]
    cache_file = Path(lock["detection_root"]) / f"{sequence}.csv"
    feature_file = Path(lock["feature_cache_path"])
    if sha256(cache_file) != lock["input_detection_sha256"][sequence]:
        raise RuntimeError("detection cache changed")
    if sha256(feature_file) != lock["feature_cache_sha256"]:
        raise RuntimeError("feature cache changed")
    reused_m = Path(lock["reused_M_tracker"])
    if sha256(reused_m) != lock["reused_M_tracker_sha256"]:
        raise RuntimeError("reused S04 M tracker output changed")

    sys.path.insert(0, str((ROOT / "third_party/OC_SORT").resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module

    seq_dir = Path(lock["train_root"]) / sequence
    config = configparser.ConfigParser()
    if not config.read(seq_dir / "seqinfo.ini"):
        raise FileNotFoundError(seq_dir / "seqinfo.ini")
    info = config["Sequence"]
    last_frame = info.getint("seqLength")
    cached, _ = read_detection_cache(cache_file)
    source_ids = all_detection_ids(cache_file)
    features = load_features(feature_file)
    output = Path(lock["output_root"])
    results = {}
    for condition in args.conditions:
        results[condition] = run_condition(
            condition, lock, sequence, last_frame, cached, source_ids, features,
            output, ocsort_module, assoc_module, info.getint("imHeight"),
            info.getint("imWidth"), info.getfloat("frameRate"),
        )
    result = {
        "status": "ENGINEERING_PASS",
        "run_id": lock["run_id"],
        "CUDA_VISIBLE_DEVICES": "",
        "physical_gpus": [],
        "results": results,
        "reused_M": {
            "path": str(reused_m),
            "sha256": lock["reused_M_tracker_sha256"],
            "source_run_id": lock["reused_M_source_run_id"],
        },
    }
    result_path = output / "engineering_checks.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
