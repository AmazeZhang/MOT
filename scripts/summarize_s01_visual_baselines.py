#!/usr/bin/env python3
"""Verify frozen S01 tracker artifacts, then evaluate with official TrackEval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/s01_visual_association_handoff_20261120.yaml"
SPLIT = ROOT / "manifests/protocols/r1_stage_b_20261118.yaml"
CONDITIONS = ("G", "Z", "L", "M")
METRIC_NAMES = ("HOTA", "DetA", "AssA", "IDF1", "IDSW", "Frag", "FP", "FN", "TP")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("TrackEval must run with CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if (lock["summary_code_sha256"] != sha256(Path(__file__)) or
            lock["protocol_sha256"] != sha256(PROTOCOL) or
            lock["tracking_root"] != str(args.tracking_root) or
            lock["evaluation_root"] != str(args.evaluation_root) or
            lock["sequences"] != args.sequences):
        raise RuntimeError("evaluation code, protocol, paths or sequence list differs from lock")
    if lock["projection_helper_sha256"] != sha256(ROOT / "scripts/analyze_e1_mature_reemit.py"):
        raise RuntimeError("metric projection helper differs from lock")
    trackeval_root = ROOT / "third_party/TrackEval"
    dirty_patch = subprocess.check_output(["git", "diff", "--binary"], cwd=trackeval_root)
    if hashlib.sha256(dirty_patch).hexdigest() != lock["trackeval_dirty_patch_sha256"]:
        raise RuntimeError("TrackEval dirty patch differs from lock")
    for rel_path, expected in lock["trackeval_file_sha256"].items():
        if sha256(trackeval_root / rel_path) != expected:
            raise RuntimeError(f"TrackEval source differs from lock: {rel_path}")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    if sha256(SPLIT) != protocol["inputs"]["split_manifest_sha256"]:
        raise RuntimeError("split SHA256 differs from protocol")
    fit = set(yaml.safe_load(SPLIT.read_text())["split"]["fit_sequences"])
    sequences = args.sequences
    if len(sequences) != len(set(sequences)) or not set(sequences).issubset(fit):
        raise ValueError("sequence list must be unique fit-only videos")
    if args.evaluation_root.exists():
        raise FileExistsError(args.evaluation_root)
    args.evaluation_root.mkdir(parents=True)
    input_manifest = {"status": "tracker_and_trace_hashes_frozen_before_GT_load",
                      "protocol_sha256": sha256(PROTOCOL),
                      "runtime_lock_sha256": sha256(args.runtime_lock),
                      "sequences": sequences, "conditions": list(CONDITIONS),
                      "tracking_root": str(args.tracking_root), "artifacts": [],
                      "feature_artifacts": [], "engineering_checks": []}
    costs = {c: {"wall_seconds": 0.0, "cpu_core_hours": 0.0,
                 "appearance_eligible_edges": 0, "changed_first_round_frames": 0,
                 "actual_updates": 0, "output_rows": 0} for c in CONDITIONS}
    for seq in sequences:
        seq_root = args.tracking_root / seq
        checks_path = seq_root / "engineering_checks.json"
        checks = json.loads(checks_path.read_text())
        if checks["status"] != "ENGINEERING_PASS" or set(checks["results"]) != set(CONDITIONS):
            raise RuntimeError(f"{seq}: incomplete four-condition engineering checks")
        if checks["results"]["Z"].get("G_equivalence") != "PASS":
            raise RuntimeError(f"{seq}: Z/G equivalence not passed")
        lock_path = seq_root / "runtime_lock.json"
        track_lock = json.loads(lock_path.read_text())
        feature_file = Path(track_lock["feature_cache_path"])
        if sha256(feature_file) != track_lock["feature_cache_sha256"]:
            raise RuntimeError(f"{seq}: feature cache differs from tracking lock")
        input_manifest["feature_artifacts"].append({"sequence": seq,
            "path": str(feature_file), "sha256": sha256(feature_file)})
        input_manifest["engineering_checks"].append({"sequence": seq,
            "path": str(checks_path), "sha256": sha256(checks_path),
            "runtime_lock_sha256": sha256(lock_path)})
        for condition in CONDITIONS:
            info = checks["results"][condition]
            for kind, suffix, key in (("trackers", "txt", "track_sha256"),
                                      ("traces", "jsonl", "trace_sha256")):
                path = seq_root / kind / condition / f"{seq}.{suffix}"
                actual = sha256(path)
                if actual != info[key]:
                    raise RuntimeError(f"{seq}/{condition}/{kind}: hash mismatch")
                input_manifest["artifacts"].append({"sequence": seq,
                    "condition": condition, "kind": kind,
                    "path": str(path), "sha256": actual})
                if kind == "trackers":
                    destination = args.evaluation_root / "trackers" / condition / f"{seq}.txt"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                    if sha256(destination) != actual:
                        raise RuntimeError("copied tracker differs from frozen artifact")
            counts = info["counts"]
            costs[condition]["wall_seconds"] += float(info["wall_seconds"])
            costs[condition]["cpu_core_hours"] += float(info["cpu_core_hours"])
            for key in ("appearance_eligible_edges", "changed_first_round_frames",
                        "actual_updates", "output_rows"):
                costs[condition][key] += int(counts.get(key, 0))
    seqmap = args.evaluation_root / "seqmap.txt"
    seqmap.write_text("name\n" + "".join(f"{seq}\n" for seq in sequences))
    input_manifest["seqmap_sha256"] = sha256(seqmap)
    frozen_path = args.evaluation_root / "frozen_inputs.json"
    frozen_path.write_text(json.dumps(input_manifest, indent=2) + "\n")
    frozen_sha = sha256(frozen_path)
    # No GT access occurs above this point. All inference outputs and trace
    # hashes are frozen before TrackEval loads annotations.
    for seq, expected in lock["gt_file_sha256"].items():
        gt = Path(protocol["inputs"]["train_root"]) / seq / "gt/gt.txt"
        if sha256(gt) != expected:
            raise RuntimeError(f"GT file differs from lock: {seq}")
    started = time.perf_counter()
    from analyze_e1_mature_reemit import projection
    sys.path.insert(0, str((ROOT / "third_party/TrackEval").resolve()))
    import trackeval

    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True, "PLOT_CURVES": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_FOLDER": str(args.evaluation_root / "official")})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({"GT_FOLDER": str(Path(protocol["inputs"]["train_root"]).resolve()),
        "TRACKERS_FOLDER": str((args.evaluation_root / "trackers").resolve()),
        "OUTPUT_FOLDER": str(args.evaluation_root / "official"),
        "TRACKERS_TO_EVAL": list(CONDITIONS), "TRACKER_DISPLAY_NAMES": list(CONDITIONS),
        "TRACKER_SUB_FOLDER": "", "OUTPUT_SUB_FOLDER": "",
        "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True,
        "SEQMAP_FILE": str(seqmap),
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "DO_PREPROC": True, "PRINT_CONFIG": False})
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    raw, messages = trackeval.Evaluator(evaluator_config).evaluate(
        [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
    results = {}
    for condition in CONDITIONS:
        if messages["MotChallenge2DBox"][condition] != "Success":
            raise RuntimeError(f"TrackEval {condition} failed")
        value = raw["MotChallenge2DBox"][condition]
        results[condition] = {"COMBINED_SEQ": projection(value["COMBINED_SEQ"]["pedestrian"]),
            "per_sequence": {seq: projection(value[seq]["pedestrian"]) for seq in sequences}}
    deltas = {}
    for name, a, b in (("L_minus_G", "L", "G"),
                       ("M_minus_G", "M", "G"), ("M_minus_L", "M", "L")):
        deltas[name] = {metric: results[a]["COMBINED_SEQ"][metric] -
                       results[b]["COMBINED_SEQ"][metric] for metric in METRIC_NAMES}
    summary = {"status": "COMPLETE_EXPLORATORY", "scope": "fit_only_development",
        "sequences": sequences, "frozen_inputs_sha256": frozen_sha,
        "runtime_lock_sha256": sha256(args.runtime_lock),
        "protocol_sha256": sha256(PROTOCOL), "trackeval_commit": protocol["baseline"]["trackeval_commit"],
        "results": results, "deltas": deltas, "costs": costs,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "limitations": ["train-fit data are reused development data, not independent test",
            "MOTIP engineering branch is separate and not compared by TrackEval",
            "G/Z/L/M are fixed simple baselines, not a new method"]}
    summary_path = args.evaluation_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "scope": sequences,
                      "combined": {c: results[c]["COMBINED_SEQ"] for c in CONDITIONS},
                      "deltas": deltas, "summary_sha256": sha256(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
