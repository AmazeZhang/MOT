#!/usr/bin/env python3
"""Run pinned official TrackEval for S05-A G/L/M development stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TE_ROOT = ROOT / "third_party/TrackEval"
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "IDSW", "Frag", "FP", "FN", "TP")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S05-A TrackEval requires CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    protocol = ROOT / lock["protocol_path"]
    if sha256(protocol) != lock["protocol_sha256"] or \
            sha256(Path(__file__)) != lock["evaluation_code_sha256"]:
        raise RuntimeError("protocol or evaluation source changed")
    if sha256(ROOT / "scripts/analyze_e1_mature_reemit.py") != lock["projection_helper_sha256"]:
        raise RuntimeError("metric projection helper changed")
    dirty_patch = subprocess.check_output(["git", "diff", "--binary"], cwd=TE_ROOT)
    if hashlib.sha256(dirty_patch).hexdigest() != lock["trackeval_dirty_patch_sha256"]:
        raise RuntimeError("TrackEval dirty patch changed")
    for path, expected in lock["trackeval_file_sha256"].items():
        if sha256(TE_ROOT / path) != expected:
            raise RuntimeError(f"TrackEval file changed: {path}")
    output = Path(lock["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frozen = {"status": "tracker_hashes_frozen_before_GT_load", "artifacts": [],
              "runtime_lock_sha256": sha256(args.runtime_lock),
              "sequences": lock["sequences"], "conditions": lock["conditions"]}
    for artifact in lock["tracker_artifacts"]:
        source = Path(artifact["path"])
        if sha256(source) != artifact["sha256"]:
            raise RuntimeError(f"tracker source changed: {source}")
        destination = output / "trackers" / artifact["condition"] / f"{artifact['sequence']}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != artifact["sha256"]:
            raise RuntimeError("staged tracker hash mismatch")
        frozen["artifacts"].append({**artifact, "staged_path": str(destination)})
    seqmap = output / "seqmap.txt"
    seqmap.write_text("name\n" + "".join(f"{sequence}\n" for sequence in lock["sequences"]))
    frozen["seqmap_sha256"] = sha256(seqmap)
    frozen_path = output / "frozen_inputs.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_sha = sha256(frozen_path)
    for sequence, expected in lock["gt_file_sha256"].items():
        if sha256(Path(lock["dataset_root"]) / sequence / "gt/gt.txt") != expected:
            raise RuntimeError(f"GT changed: {sequence}")

    from analyze_e1_mature_reemit import projection
    sys.path.insert(0, str(TE_ROOT.resolve()))
    import trackeval
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True, "PLOT_CURVES": False,
        "DISPLAY_LESS_PROGRESS": True, "OUTPUT_FOLDER": str(output / "official")})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({"GT_FOLDER": lock["dataset_root"],
        "TRACKERS_FOLDER": str((output / "trackers").resolve()),
        "OUTPUT_FOLDER": str(output / "official"), "TRACKERS_TO_EVAL": lock["conditions"],
        "TRACKER_DISPLAY_NAMES": lock["conditions"], "TRACKER_SUB_FOLDER": "",
        "OUTPUT_SUB_FOLDER": "", "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True, "SEQMAP_FILE": str(seqmap),
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt", "DO_PREPROC": True,
        "PRINT_CONFIG": False})
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    started = time.perf_counter()
    raw, messages = trackeval.Evaluator(evaluator_config).evaluate(
        [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
    results = {}
    for condition in lock["conditions"]:
        if messages["MotChallenge2DBox"][condition] != "Success":
            raise RuntimeError(f"TrackEval failed: {condition}")
        value = raw["MotChallenge2DBox"][condition]
        results[condition] = {"COMBINED_SEQ": projection(value["COMBINED_SEQ"]["pedestrian"]),
                              "per_sequence": {sequence: projection(value[sequence]["pedestrian"])
                                               for sequence in lock["sequences"]}}
    deltas = {}
    for name, a, b in (("L_minus_G", "L", "G"), ("M_minus_G", "M", "G"),
                       ("M_minus_L", "M", "L")):
        deltas[name] = {metric: results[a]["COMBINED_SEQ"][metric] -
                       results[b]["COMBINED_SEQ"][metric] for metric in METRICS}
    summary = {"status": "COMPLETED", "scope": "reused_development8_not_independent_test",
               "sequences": lock["sequences"], "frozen_inputs_sha256": frozen_sha,
               "runtime_lock_sha256": sha256(args.runtime_lock),
               "protocol_sha256": lock["protocol_sha256"], "trackeval_commit": lock["trackeval_commit"],
               "results": results, "deltas": deltas,
               "evaluation_wall_seconds": time.perf_counter() - started,
               "limitations": ["all eight sequences were previously used in R1 and S04",
                               "appearance weight training-data overlap remains unknown",
                               "descriptive development stability only"]}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "combined": {condition: results[condition]["COMBINED_SEQ"]
                      for condition in lock["conditions"]}, "deltas": deltas,
                      "summary_sha256": sha256(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
