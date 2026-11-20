#!/usr/bin/env python3
"""Run frozen CV output controls on train-fit or train-heldout videos."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_e1_mature_reemit import projection
from analyze_p1_common_state_mechanism import read_detection_cache, write_tracker_row
from mot_research.analysis.failure_events import pairwise_iou
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, NaturalMotionFirstRoundPolicy,
    TrackerDetectionHistory,
)
from run_e1_mature_reemit_factorial import u_extra_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e3_short_gap_support_20261119.yaml"
CODE_PATHS = (
    "scripts/run_e3_short_gap_support.py",
    "scripts/run_e1_mature_reemit_factorial.py",
    "scripts/analyze_e1_mature_reemit.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "src/mot_research/data/motion_samples.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
)
CONDITIONS = ("C", "D", "E", "G")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("fit", "heldout"), required=True)
    return parser.parse_args()


def source_check(protocol: dict, partition: str) -> tuple[list[str], dict]:
    split_config = protocol["split"]
    r1_path = ROOT / split_config["manifest"]
    source_path = ROOT / split_config["source_split"]
    if (sha256_file(r1_path) != split_config["manifest_sha256"] or
            sha256_file(source_path) != split_config["source_split_sha256"]):
        raise RuntimeError("split manifest hash mismatch")
    r1 = yaml.safe_load(r1_path.read_text())
    source = yaml.safe_load(source_path.read_text())
    fit, heldout = r1["split"]["fit_sequences"], r1["split"]["heldout_sequences"]
    if (len(fit) != 24 or len(heldout) != 8 or set(fit) & set(heldout) or
            set(fit) | set(heldout) != set(source["motion_train"]["sequences"])):
        raise RuntimeError("fit/heldout is not the frozen motion-train partition")
    if sha256_file(ROOT / "manifests/protocols/e1_mature_reemit_factorial_20261119.yaml") != \
            protocol["prior_evidence"]["e1_protocol_sha256"]:
        raise RuntimeError("E1 protocol changed")
    if sha256_file(ROOT / "manifests/protocols/e2_reemit_guard_20261119.yaml") != \
            protocol["prior_evidence"]["e2_protocol_sha256"]:
        raise RuntimeError("E2 protocol changed")
    inp = protocol["inputs"]
    for root_key, expected in (("oc_sort_root", "oc_sort_commit"),
                               ("trackeval_root", "trackeval_commit")):
        actual = subprocess.run(["git", "-C", str(ROOT / inp[root_key]), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        if actual != inp[expected]:
            raise RuntimeError(f"{root_key} commit changed")
    g1_summary = json.loads(Path(inp["g1_baseline_summary"]).read_text())
    if g1_summary["status"] != "PASS" or g1_summary["sequences"] != 40:
        raise RuntimeError("G1 baseline summary incomplete")
    g1_by_seq = {item["sequence"]: item for item in g1_summary["sequence_summaries"]}
    return (fit if partition == "fit" else heldout), g1_by_seq


def evaluate(protocol: dict, partial: Path, sequences: list[str]) -> dict:
    inp = protocol["inputs"]
    seqmap = partial / "seqmap.txt"
    seqmap.write_text("name\n" + "".join(f"{seq}\n" for seq in sequences))
    sys.path.insert(0, str((ROOT / inp["trackeval_root"]).resolve()))
    import trackeval
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update({
        "USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False, "TIME_PROGRESS": False, "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True, "PLOT_CURVES": False, "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_FOLDER": str(partial / "official")})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(Path(inp["train_root"]).resolve()),
        "TRACKERS_FOLDER": str((partial / "trackers").resolve()),
        "OUTPUT_FOLDER": str(partial / "official"),
        "TRACKERS_TO_EVAL": list(CONDITIONS),
        "TRACKER_DISPLAY_NAMES": list(CONDITIONS),
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
            raise RuntimeError(f"TrackEval {condition} failed: {messages['MotChallenge2DBox'][condition]}")
        value = raw["MotChallenge2DBox"][condition]
        results[condition] = {
            "COMBINED_SEQ": projection(value["COMBINED_SEQ"]["pedestrian"]),
            "per_sequence": {seq: projection(value[seq]["pedestrian"]) for seq in sequences}}
    return results


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E3 requires CUDA_VISIBLE_DEVICES='' and CPU only")
    args = parse_args()
    protocol = yaml.safe_load(PROTOCOL.read_text())
    sequences, g1_by_seq = source_check(protocol, args.partition)
    if args.partition == "heldout":
        fit_dir = Path(protocol["outputs"]["fit_dir"])
        fit_summary = json.loads((fit_dir / "summary.json").read_text())
        if (fit_summary["status"] != "PASS" or
                fit_summary["protocol_sha256"] != sha256_file(PROTOCOL) or
                not fit_summary["fit_gate_pass"]):
            raise RuntimeError("fit stage did not authorize heldout run")
    inp = protocol["inputs"]
    output = Path(protocol["outputs"][f"{args.partition}_dir"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(path.exists() for path in (output, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir(parents=True)
    start = time.monotonic()
    cpu_start = time.process_time()
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL),
                           "partition": args.partition, "CUDA_VISIBLE_DEVICES": "",
                           "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for condition in CONDITIONS:
            (partial / "trackers" / condition).mkdir(parents=True)
        (partial / "extra_outputs").mkdir()
        sys.path.insert(0, str((ROOT / inp["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module
        summaries = []
        for seq in sequences:
            seq_dir = Path(inp["train_root"]) / seq
            seqinfo = seq_dir / "seqinfo.ini"
            cache_file = Path(inp["detection_root"]) / f"{seq}.csv"
            cache_sha = sha256_file(cache_file)
            if cache_sha != g1_by_seq[seq]["cache_sha256"]:
                raise RuntimeError(f"{seq}: detection cache differs from G1")
            config = configparser.ConfigParser()
            if not config.read(seqinfo):
                raise FileNotFoundError(seqinfo)
            info = config["Sequence"]
            length = info.getint("seqLength")
            height, width = info.getint("imHeight"), info.getint("imWidth")
            fps = info.getfloat("frameRate")
            scale = min(inp["input_size"][0] / height, inp["input_size"][1] / width)
            cached, _ = read_detection_cache(cache_file)
            tracker = ocsort_module.OCSort(**inp["tracker"])
            history = TrackerDetectionHistory(history_length=inp["cv_history"]["length"],
                                              score_threshold=.6)
            policy = NaturalMotionFirstRoundPolicy(
                mode="constant_velocity", history=history, image_width=width,
                image_height=height, fps=fps,
                minimum_observations=inp["cv_history"]["minimum_observations"],
                maximum_frame_gap=inp["cv_history"]["maximum_frame_gap"])
            runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
            paths = {c: partial / "trackers" / c / f"{seq}.txt" for c in CONDITIONS}
            handles = {c: path.open("w", newline="") for c, path in paths.items()}
            writers = {c: csv.writer(h, lineterminator="\n") for c, h in handles.items()}
            extra_file = partial / "extra_outputs" / f"{seq}.jsonl"
            last_original = {}
            mature_ids = set()
            counts = Counter()
            try:
                with extra_file.open("w") as extra_handle:
                    for frame in range(1, length + 1):
                        before_hits = {int(t.id) + 1: int(t.hits) for t in tracker.trackers}
                        mature_before = set(mature_ids)
                        values = np.asarray(cached.get(frame, []), dtype=np.float64)
                        tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                        if tracker_input is not None:
                            tracker_input[:, :4] *= scale
                        original = runner.update(tracker_input, (height, width),
                                                 tuple(inp["input_size"]), frame_id=frame)
                        original = np.asarray(original, dtype=np.float64).reshape(-1, 5)
                        original_ids = [int(row[4]) for row in original]
                        if len(original_ids) != len(set(original_ids)):
                            raise RuntimeError(f"{seq}:{frame}: duplicate original ID")
                        extra, details = u_extra_rows(tracker, original, mature_before,
                                                      before_hits, tracker_input is not None)
                        if len(extra) != len(details):
                            raise RuntimeError("U rows/detail mismatch")
                        for row in original:
                            for writer in writers.values():
                                write_tracker_row(writer, frame, row)
                            counts["C"] += 1
                        for row, item in zip(extra, details):
                            tid = int(item["track_id"])
                            if tid not in last_original:
                                raise RuntimeError(f"{seq}:{frame}:{tid}: no previous original output")
                            old_frame, old_box = last_original[tid]
                            gap = frame - old_frame
                            iou = float(pairwise_iou(np.asarray(item["box_xyxy"])[None],
                                                      old_box[None])[0, 0])
                            keep_e = float(item["score"]) >= .8
                            keep_g = keep_e or (gap <= protocol["G_rule"]["gap_from_last_original_C_output_at_most_frames"]
                                               and iou >= protocol["G_rule"]["iou_current_detection_to_last_original_C_output_at_least"])
                            write_tracker_row(writers["D"], frame, row)
                            counts["D_extra"] += 1
                            if keep_e:
                                write_tracker_row(writers["E"], frame, row)
                                counts["E_extra"] += 1
                            if keep_g:
                                write_tracker_row(writers["G"], frame, row)
                                counts["G_extra"] += 1
                            extra_handle.write(json.dumps({"sequence": seq, "frame": frame,
                                **item, "gap_since_last_C_output": gap,
                                "iou_last_C_output": iou,
                                "E_keep": keep_e, "G_keep": keep_g}, separators=(",", ":")) + "\n")
                        for row in original:
                            last_original[int(row[4])] = (frame, row[:4].copy())
                        for t in tracker.trackers:
                            tid = int(t.id) + 1
                            if tid in original_ids and t.hit_streak >= tracker.min_hits:
                                mature_ids.add(tid)
            finally:
                for handle in handles.values():
                    handle.close()
            if not counts["C"] <= counts["C"] + counts["E_extra"] <= \
                    counts["C"] + counts["G_extra"] <= counts["C"] + counts["D_extra"]:
                raise RuntimeError(f"{seq}: subset count order failed")
            record = {"sequence": seq, "frames": length,
                      "cache_sha256": cache_sha, "seqinfo_sha256": sha256_file(seqinfo),
                      "gt_sha256": sha256_file(seq_dir / "gt/gt.txt"),
                      "counts": dict(counts),
                      "tracker_sha256": {c: sha256_file(paths[c]) for c in CONDITIONS},
                      "extra_sha256": sha256_file(extra_file),
                      "runner": runner.snapshot()}
            summaries.append(record)
            (partial / "sequence_progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"{args.partition} {seq}: C={counts['C']} D+={counts['D_extra']} E+={counts['E_extra']} G+={counts['G_extra']}", flush=True)
            if time.monotonic() - start > protocol["budget"]["max_wall_seconds_per_partition"]:
                raise TimeoutError("E3 partition wall budget exceeded")
        verify_code_unchanged(ROOT, provenance)
        metrics = evaluate(protocol, partial, sequences)
        c = metrics["C"]["COMBINED_SEQ"]
        e = metrics["E"]["COMBINED_SEQ"]
        g = metrics["G"]["COMBINED_SEQ"]
        threshold = protocol["decision"]["fit_gate_all_required"]
        wins = sum(metrics["G"]["per_sequence"][seq]["HOTA"] >
                   metrics["E"]["per_sequence"][seq]["HOTA"] for seq in sequences)
        fit_gate = (g["HOTA"] - e["HOTA"] >= threshold["G_minus_E_HOTA_min_percentage_points"] and
                    g["AssA"] - e["AssA"] >= threshold["G_minus_E_AssA_min_percentage_points"] and
                    g["IDF1"] - e["IDF1"] >= threshold["G_minus_E_IDF1_min_percentage_points"] and
                    g["IDSW"] - e["IDSW"] <= threshold["G_minus_E_IDSW_max_count"] and
                    g["FP"] - e["FP"] <= threshold["G_minus_E_FP_max_count"] and
                    wins >= threshold["minimum_fit_videos_G_HOTA_above_E"])
        verify_code_unchanged(ROOT, provenance)
        core_hours = (time.process_time() - cpu_start) / 3600
        if core_hours > protocol["budget"]["max_cpu_core_hours_per_partition"]:
            raise RuntimeError("E3 CPU core-hour budget exceeded")
        result = {"status": "PASS", "partition": args.partition,
                  "protocol_sha256": provenance["protocol_sha256"],
                  "sequences": summaries, "metrics": metrics,
                  "G_minus_E": {k: g[k] - e[k] for k in ("HOTA", "DetA", "AssA", "IDF1", "IDSW", "Frag", "FP", "FN")},
                  "G_HOTA_win_videos": wins, "fit_gate_pass": fit_gate if args.partition == "fit" else None,
                  "wall_seconds": time.monotonic() - start, "cpu_core_hours": core_hours,
                  "peak_rss_kib": peak_rss_kib(),
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "partition": args.partition,
                          "metrics": {x: metrics[x]["COMBINED_SEQ"] for x in CONDITIONS},
                          "G_minus_E": result["G_minus_E"], "G_HOTA_win_videos": wins,
                          "fit_gate_pass": result["fit_gate_pass"], "output": str(output)}, indent=2), flush=True)
    except BaseException as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
