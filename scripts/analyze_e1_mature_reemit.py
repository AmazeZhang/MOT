#!/usr/bin/env python3
"""Evaluate frozen E1 four-condition outputs with TrackEval and fixed events."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from evaluate_p1_recovery_events import read_events, assert_baseline_self
from mot_research.analysis.failure_events import (
    hungarian_iou_matches, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.evaluation.closed_loop_events import evaluate_method_recovery_events

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e1_mature_reemit_factorial_20261119.yaml"
CODE_PATHS = (
    "scripts/analyze_e1_mature_reemit.py",
    "scripts/evaluate_p1_recovery_events.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/evaluation/closed_loop_events.py",
    "src/mot_research/analysis/runtime_record.py",
)
CONDITIONS = ("A", "B", "C", "D")


def projection(values):
    return {
        "HOTA": float(np.mean(values["HOTA"]["HOTA"])) * 100,
        "DetA": float(np.mean(values["HOTA"]["DetA"])) * 100,
        "AssA": float(np.mean(values["HOTA"]["AssA"])) * 100,
        "IDF1": float(values["Identity"]["IDF1"]) * 100,
        "IDSW": int(values["CLEAR"]["IDSW"]),
        "Frag": int(values["CLEAR"]["Frag"]),
        "FP": int(values["CLEAR"]["CLR_FP"]),
        "FN": int(values["CLEAR"]["CLR_FN"]),
        "TP": int(values["CLEAR"]["CLR_TP"]),
    }


def recovery_summary(rows):
    counts = Counter()
    counts["events"] = len(rows)
    for row in rows:
        counts["reconnect"] += bool(row["reconnect_success"])
        counts["censored"] += bool(row["persistence_censored"])
        if row["persistence_censored"]:
            continue
        counts["persistence_success"] += bool(row["persistence_success"])
        if row["baseline_stratum"] == "recovery_failure":
            counts["recovered_failure"] += bool(row["persistence_success"])
        elif row["baseline_stratum"] == "success_control":
            counts["harmed_control"] += not bool(row["persistence_success"])
    return dict(counts)


def gt_matches_for_windows(gt, outputs, events):
    frames = {frame for event in events
              for frame in range(int(event["after_frame"]), int(event["after_frame"]) + 6)}
    return {frame: hungarian_iou_matches(gt.get(frame, {}), outputs.get(frame, []), .5)[0]
            for frame in frames}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E1 evaluation requires CUDA_VISIBLE_DEVICES='' and CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    run = Path(protocol["outputs"]["run_dir"])
    run_summary = json.loads((run / "summary.json").read_text())
    if run_summary["status"] != "PASS" or run_summary["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("E1 tracker run failed or protocol changed")
    events_path = Path(protocol["events"]["path"])
    if sha256_file(events_path) != protocol["events"]["sha256"]:
        raise RuntimeError("fixed event hash mismatch")
    output = run.with_name(run.name + "_eval")
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(path.exists() for path in (output, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir(parents=True)
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL),
                           "run_summary_sha256": sha256_file(run / "summary.json"),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        sequences = protocol["scope"]["sequences"]
        seqmap = partial / "seqmap.txt"
        seqmap.write_text("name\n" + "".join(f"{seq}\n" for seq in sequences))
        trackeval_root = ROOT / protocol["inputs"]["trackeval_root"]
        trackeval_commit = subprocess.run(["git", "-C", str(trackeval_root),
                                           "rev-parse", "HEAD"], check=True,
                                          capture_output=True, text=True).stdout.strip()
        sys.path.insert(0, str(trackeval_root.resolve()))
        import trackeval
        evaluator_config = trackeval.Evaluator.get_default_eval_config()
        evaluator_config.update({
            "USE_PARALLEL": False, "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False, "PRINT_CONFIG": False,
            "TIME_PROGRESS": False, "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True, "PLOT_CURVES": False,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_FOLDER": str(partial / "official"),
        })
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update({
            "GT_FOLDER": str(Path(protocol["inputs"]["train_root"]).resolve()),
            "TRACKERS_FOLDER": str((run / "trackers").resolve()),
            "OUTPUT_FOLDER": str(partial / "official"),
            "TRACKERS_TO_EVAL": list(CONDITIONS),
            "TRACKER_DISPLAY_NAMES": list(CONDITIONS),
            "TRACKER_SUB_FOLDER": "", "OUTPUT_SUB_FOLDER": "",
            "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True,
            "SEQMAP_FILE": str(seqmap),
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "DO_PREPROC": True, "PRINT_CONFIG": False,
        })
        metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
                   trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                   trackeval.metrics.Identity({"PRINT_CONFIG": False})]
        raw, messages = trackeval.Evaluator(evaluator_config).evaluate(
            [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
        metric_results = {}
        for condition in CONDITIONS:
            if messages["MotChallenge2DBox"][condition] != "Success":
                raise RuntimeError(f"TrackEval {condition}: {messages['MotChallenge2DBox'][condition]}")
            result = raw["MotChallenge2DBox"][condition]
            metric_results[condition] = {
                "COMBINED_SEQ": projection(result["COMBINED_SEQ"]["pedestrian"]),
                "per_sequence": {seq: projection(result[seq]["pedestrian"]) for seq in sequences},
            }
        (partial / "trackeval_metrics.json").write_text(json.dumps({
            "trackeval_commit": trackeval_commit,
            "conditions": metric_results,
        }, indent=2) + "\n")
        events = read_events(events_path)
        by_sequence = defaultdict(list)
        for event in events:
            by_sequence[event["sequence"]].append(event)
        outcomes = {condition: [] for condition in CONDITIONS}
        delay_rows = []
        extra_counts = {"B": Counter(), "D": Counter()}
        run_by_key = {(x["mode"], x["sequence"]): x for x in run_summary["sequences"]}
        for seq in sequences:
            gt_path = Path(protocol["inputs"]["train_root"]) / seq / "gt" / "gt.txt"
            gt = read_gt(gt_path)
            tracker_outputs = {}
            matches = {}
            for condition in CONDITIONS:
                tracker_path = run / "trackers" / condition / f"{seq}.txt"
                mode = "upstream" if condition in ("A", "B") else "constant_velocity"
                expected_hash = (run_by_key[mode, seq]["original_sha256"] if condition in ("A", "C")
                                 else run_by_key[mode, seq]["U_sha256"])
                if sha256_file(tracker_path) != expected_hash:
                    raise RuntimeError(f"{condition}:{seq}: tracker hash mismatch")
                tracker_outputs[condition] = read_tracker(tracker_path)
                rows, _ = evaluate_method_recovery_events(
                    events=by_sequence[seq], gt_by_frame=gt,
                    tracker_by_frame=tracker_outputs[condition],
                    matching_iou=.5, persistence_frames=5)
                outcomes[condition].extend(rows)
                matches[condition] = gt_matches_for_windows(
                    gt, tracker_outputs[condition], by_sequence[seq])
                for event in by_sequence[seq]:
                    intended = int(event["baseline_before_tracker_id"])
                    after = int(event["after_frame"])
                    first = next((frame - after for frame in range(after, after + 6)
                                  if matches[condition].get(frame, {}).get(int(event["gt_id"]), (None,))[0]
                                  == intended), None)
                    delay_rows.append({"event_id": event["event_id"], "sequence": seq,
                                       "condition": condition, "delay_0_to_5_or_censored": first})
            for condition, mode in (("B", "upstream"), ("D", "constant_velocity")):
                extra_file = run / "extra_outputs" / f"{mode}_{seq}.jsonl"
                if sha256_file(extra_file) != run_by_key[mode, seq]["U_extra_sha256"]:
                    raise RuntimeError(f"{condition}:{seq}: extra output hash mismatch")
                extras = [json.loads(line) for line in extra_file.open()]
                by_frame = defaultdict(list)
                for row in extras:
                    by_frame[int(row["frame"])].append(int(row["track_id"]))
                for frame, ids in by_frame.items():
                    matched = hungarian_iou_matches(
                        gt.get(frame, {}), tracker_outputs[condition].get(frame, []), .5)[0]
                    matched_ids = {int(value[0]) for value in matched.values()}
                    extra_counts[condition]["TP_proxy"] += sum(track_id in matched_ids for track_id in ids)
                    extra_counts[condition]["FP_proxy"] += sum(track_id not in matched_ids for track_id in ids)
                extra_counts[condition]["total"] += len(extras)
        assert_baseline_self(events, outcomes["A"])
        for condition in CONDITIONS:
            if len(outcomes[condition]) != 831:
                raise RuntimeError(f"{condition}: incomplete event outcomes")
        with (partial / "event_outcomes.csv").open("w", newline="") as handle:
            fieldnames = ["condition", "event_id", "sequence", "baseline_stratum",
                          "reconnect_success", "persistence_censored", "persistence_success"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for condition in CONDITIONS:
                for row in sorted(outcomes[condition], key=lambda x: x["event_id"]):
                    writer.writerow({"condition": condition, **{key: row[key] for key in fieldnames[1:]}})
        with (partial / "reemit_delay.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(delay_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(delay_rows)
        contrasts = {}
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "IDSW", "Frag", "FP", "FN"):
            a, b, c, d = [metric_results[x]["COMBINED_SEQ"][metric] for x in CONDITIONS]
            contrasts[metric] = {"B_minus_A": b-a, "D_minus_C": d-c,
                                 "interaction": (d-c)-(b-a)}
        def beneficial(original, u):
            old = metric_results[original]["COMBINED_SEQ"]
            new = metric_results[u]["COMBINED_SEQ"]
            return (new["HOTA"] > old["HOTA"] + 1e-9 and
                    new["AssA"] >= old["AssA"] - 1e-9 and
                    new["IDF1"] >= old["IDF1"] - 1e-9)
        decision = {
            "stop_U_no_HOTA_gain_both": (contrasts["HOTA"]["B_minus_A"] <= 0 and
                                         contrasts["HOTA"]["D_minus_C"] <= 0),
            "eligible_as_simple_baseline": beneficial("A", "B") or beneficial("C", "D"),
            "note": "Development evidence only; U cannot change internal association."
        }
        verify_code_unchanged(ROOT, provenance)
        summary = {"status": "PASS", "scope": "8 motion-dev four-condition development factorial",
                   "protocol_sha256": provenance["protocol_sha256"],
                   "run_summary_sha256": provenance["run_summary_sha256"],
                   "trackeval_commit": trackeval_commit,
                   "metrics": metric_results, "contrasts": contrasts,
                   "recovery": {c: recovery_summary(outcomes[c]) for c in CONDITIONS},
                   "extra_output_proxy": {c: dict(extra_counts[c]) for c in ("B", "D")},
                   "decision": decision,
                   "event_outcomes_sha256": sha256_file(partial / "event_outcomes.csv"),
                   "delay_rows_sha256": sha256_file(partial / "reemit_delay.csv"),
                   "trackeval_metrics_sha256": sha256_file(partial / "trackeval_metrics.json"),
                   "wall_seconds": time.monotonic() - started,
                   "cpu_core_hours": (time.process_time() - cpu_started) / 3600,
                   "peak_rss_kib": peak_rss_kib(),
                   "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "metrics": {c: x["COMBINED_SEQ"] for c,x in metric_results.items()},
                          "decision": decision, "output": str(output)}, indent=2), flush=True)
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
