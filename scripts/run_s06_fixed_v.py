#!/usr/bin/env python3
"""Run the frozen fixed appearance-weight V=0.5 control in natural closed loop."""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import sys

import run_s01_visual_baselines as s01


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--runtime-lock", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("fixed V replay is CPU-only")
    lock = json.loads(args.runtime_lock.read_text())
    for path, expected in lock["code_hashes"].items():
        if s01.sha256(ROOT / path) != expected:
            raise RuntimeError(f"source changed: {path}")
    sequence = lock["selected_sequences"][0]
    cache = Path(lock["detection_root"]) / f"{sequence}.csv"
    feature = Path(lock["feature_cache_path"])
    if s01.sha256(cache) != lock["input_detection_sha256"][sequence] or \
            s01.sha256(feature) != lock["feature_cache_sha256"]:
        raise RuntimeError("input cache changed")
    sys.path.insert(0, str((ROOT / "third_party/OC_SORT").resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module
    cfg = configparser.ConfigParser(); seq_dir = Path(lock["train_root"]) / sequence
    if not cfg.read(seq_dir / "seqinfo.ini"):
        raise FileNotFoundError(seq_dir / "seqinfo.ini")
    info = cfg["Sequence"]
    cached, _ = s01.read_detection_cache(cache)
    ids = s01.all_detection_ids(cache); features = s01.load_features(feature)
    if "V" in s01.CONDITIONS:
        raise RuntimeError("V unexpectedly already registered")
    s01.CONDITIONS["V"] = (0.5, "ema")
    try:
        result = s01.run_condition("V", lock, sequence, info.getint("seqLength"),
            cached, ids, features, Path(lock["output_root"]), ocsort_module, assoc_module,
            info.getint("imHeight"), info.getint("imWidth"), info.getfloat("frameRate"))
    finally:
        del s01.CONDITIONS["V"]
    check = {"status": "ENGINEERING_PASS", "run_id": lock["run_id"],
             "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
             "result": result}
    path = Path(lock["output_root"]) / "engineering_checks.json"
    path.write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
