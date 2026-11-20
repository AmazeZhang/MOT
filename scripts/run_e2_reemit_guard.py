#!/usr/bin/env python3
"""Filter frozen E1 CV extra outputs, then evaluate two output-only controls."""

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

import yaml

from analyze_e1_mature_reemit import projection, recovery_summary
from evaluate_p1_recovery_events import read_events
from mot_research.analysis.failure_events import hungarian_iou_matches, read_gt, read_tracker
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.evaluation.closed_loop_events import evaluate_method_recovery_events

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e2_reemit_guard_20261119.yaml"
CODE_PATHS = (
    "scripts/run_e2_reemit_guard.py",
    "scripts/analyze_e1_mature_reemit.py",
    "scripts/evaluate_p1_recovery_events.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/evaluation/closed_loop_events.py",
    "src/mot_research/analysis/runtime_record.py",
)


def key(row: list[str]) -> tuple[int, int]:
    return int(row[0]), int(row[1])


def csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        return list(csv.reader(handle))


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E2 requires CUDA_VISIBLE_DEVICES='' and CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    origin = protocol["origin"]
    e1_protocol = ROOT / origin["e1_protocol"]
    e1 = yaml.safe_load(e1_protocol.read_text())
    source = Path(e1["outputs"]["run_dir"])
    e1_eval = source.with_name(source.name + "_eval")
    risk = Path("/media/imc/data/mot-data/runs/e2_reemit_risk_20261119/summary.json")
    for path, expected in ((source / "summary.json", origin["e1_run_summary_sha256"]),
                           (e1_eval / "summary.json", origin["e1_eval_summary_sha256"]),
                           (risk, origin["descriptive_risk_summary_sha256"]),
                           (Path(protocol["inputs"]["events"]), protocol["inputs"]["events_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
    e1_summary = json.loads((source / "summary.json").read_text())
    indexed = {item["sequence"]: item for item in e1_summary["sequences"]
               if item["mode"] == "constant_velocity"}
    output = Path(protocol["outputs"]["run_dir"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(path.exists() for path in (output, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir(parents=True)
    start = time.monotonic()
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL),
                           "e1_protocol_sha256": sha256_file(e1_protocol),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        sequences = protocol["scope"]["sequences"]
        for condition in ("E_score08", "F_streak2"):
            (partial / "trackers" / condition).mkdir(parents=True)
        rows_summary = {}
        for seq in sequences:
            source_info = indexed[seq]
            c_path = source / "trackers/C" / f"{seq}.txt"
            d_path = source / "trackers/D" / f"{seq}.txt"
            extra_path = source / "extra_outputs" / f"constant_velocity_{seq}.jsonl"
            for path, field in ((c_path, "original_sha256"), (d_path, "U_sha256"),
                                (extra_path, "U_extra_sha256")):
                if sha256_file(path) != source_info[field]:
                    raise RuntimeError(f"E1 source hash mismatch: {path}")
            c_rows, d_rows = csv_rows(c_path), csv_rows(d_path)
            c_by_key, d_by_key = {key(r): r for r in c_rows}, {key(r): r for r in d_rows}
            if len(c_by_key) != len(c_rows) or len(d_by_key) != len(d_rows):
                raise RuntimeError(f"{seq}: duplicate output key")
            if any(d_by_key.get(k) != r for k, r in c_by_key.items()):
                raise RuntimeError(f"{seq}: C not exact subset of D")
            extras = [json.loads(line) for line in extra_path.open()]
            extra_keys = {(int(x["frame"]), int(x["track_id"])) for x in extras}
            if len(extras) != len(extra_keys) or set(d_by_key) - set(c_by_key) != extra_keys:
                raise RuntimeError(f"{seq}: D extras mismatch provenance")
            allowed = {
                "E_score08": {(int(x["frame"]), int(x["track_id"])) for x in extras
                              if float(x["score"]) >= protocol["method"]["score_threshold"]},
                "F_streak2": {(int(x["frame"]), int(x["track_id"])) for x in extras
                              if int(x["hit_streak"]) >= protocol["method"]["streak_threshold"]},
            }
            rows_summary[seq] = {}
            for condition, keys in allowed.items():
                selected = [r for r in d_rows if key(r) in c_by_key or key(r) in keys]
                output_path = partial / "trackers" / condition / f"{seq}.txt"
                with output_path.open("w", newline="") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerows(selected)
                if len(selected) != len(c_rows) + len(keys):
                    raise RuntimeError(f"{seq}:{condition}: output count mismatch")
                rows_summary[seq][condition] = {
                    "original_rows": len(c_rows), "eligible_extra_rows": len(keys),
                    "output_sha256": sha256_file(output_path)}
            if time.monotonic() - start > protocol["budget"]["max_wall_seconds"]:
                raise TimeoutError("E2 wall budget exceeded")
        seqmap = partial / "seqmap.txt"
        seqmap.write_text("name\n" + "".join(f"{seq}\n" for seq in sequences))
        trackeval_root = ROOT / protocol["inputs"]["trackeval_root"]
        commit = subprocess.run(["git", "-C", str(trackeval_root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        if commit != protocol["inputs"]["trackeval_commit"]:
            raise RuntimeError("TrackEval commit changed")
        sys.path.insert(0, str(trackeval_root.resolve()))
        import trackeval
        evaluator_config = trackeval.Evaluator.get_default_eval_config()
        evaluator_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False,
                                 "PRINT_ONLY_COMBINED": False, "PRINT_CONFIG": False,
                                 "TIME_PROGRESS": False, "OUTPUT_SUMMARY": True,
                                 "OUTPUT_DETAILED": True, "PLOT_CURVES": False,
                                 "DISPLAY_LESS_PROGRESS": True,
                                 "OUTPUT_FOLDER": str(partial / "official")})
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update({
            "GT_FOLDER": str(Path(protocol["inputs"]["train_root"]).resolve()),
            "TRACKERS_FOLDER": str((partial / "trackers").resolve()),
            "OUTPUT_FOLDER": str(partial / "official"),
            "TRACKERS_TO_EVAL": ["E_score08", "F_streak2"],
            "TRACKER_DISPLAY_NAMES": ["E_score08", "F_streak2"],
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
        for condition in ("E_score08", "F_streak2"):
            if messages["MotChallenge2DBox"][condition] != "Success":
                raise RuntimeError(f"TrackEval {condition} failed")
            result = raw["MotChallenge2DBox"][condition]
            results[condition] = {
                "COMBINED_SEQ": projection(result["COMBINED_SEQ"]["pedestrian"]),
                "per_sequence": {seq: projection(result[seq]["pedestrian"]) for seq in sequences}}
        events = read_events(Path(protocol["inputs"]["events"]))
        by_sequence = defaultdict(list)
        for event in events:
            by_sequence[event["sequence"]].append(event)
        outcomes = {condition: [] for condition in results}
        proxies = {condition: Counter() for condition in results}
        for seq in sequences:
            gt = read_gt(Path(protocol["inputs"]["train_root"]) / seq / "gt/gt.txt")
            for condition in results:
                outputs = read_tracker(partial / "trackers" / condition / f"{seq}.txt")
                event_rows, _ = evaluate_method_recovery_events(
                    events=by_sequence[seq], gt_by_frame=gt,
                    tracker_by_frame=outputs, matching_iou=.5, persistence_frames=5)
                outcomes[condition].extend(event_rows)
                extra_path = source / "extra_outputs" / f"constant_velocity_{seq}.jsonl"
                extras = [json.loads(line) for line in extra_path.open()]
                keep = lambda x: (float(x["score"]) >= .8 if condition == "E_score08"
                                  else int(x["hit_streak"]) >= 2)
                by_frame = defaultdict(list)
                for item in extras:
                    if keep(item):
                        by_frame[int(item["frame"])].append(int(item["track_id"]))
                for frame, ids in by_frame.items():
                    matched = hungarian_iou_matches(gt.get(frame, {}), outputs.get(frame, []), .5)[0]
                    matched_ids = {int(value[0]) for value in matched.values()}
                    proxies[condition]["total"] += len(ids)
                    proxies[condition]["TP_proxy"] += sum(tid in matched_ids for tid in ids)
                    proxies[condition]["FP_proxy"] += sum(tid not in matched_ids for tid in ids)
        if any(len(outcomes[condition]) != 831 for condition in results):
            raise RuntimeError("event coverage mismatch")
        old = json.loads((e1_eval / "summary.json").read_text())["metrics"]
        utility = {}
        for condition, values in results.items():
            new = values["COMBINED_SEQ"]
            c, d = old["C"]["COMBINED_SEQ"], old["D"]["COMBINED_SEQ"]
            utility[condition] = (new["HOTA"] > c["HOTA"] and
                                  new["AssA"] >= c["AssA"] and
                                  new["IDF1"] >= c["IDF1"] and
                                  new["IDSW"] < d["IDSW"] and new["FP"] < d["FP"])
        verify_code_unchanged(ROOT, provenance)
        summary = {
            "status": "PASS", "protocol_sha256": provenance["protocol_sha256"],
            "E1_run_summary_sha256": origin["e1_run_summary_sha256"],
            "E1_eval_summary_sha256": origin["e1_eval_summary_sha256"],
            "trackeval_commit": commit, "source_rows": rows_summary,
            "metrics": results,
            "recovery": {c: recovery_summary(x) for c, x in outcomes.items()},
            "extra_output_proxy": {c: dict(x) for c, x in proxies.items()},
            "exploratory_utility_gate": utility,
            "wall_seconds": time.monotonic() - start,
            "peak_rss_kib": peak_rss_kib(),
            "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "metrics": {c: x["COMBINED_SEQ"] for c, x in results.items()},
                          "utility": utility, "output": str(output)}, indent=2))
    except BaseException as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
