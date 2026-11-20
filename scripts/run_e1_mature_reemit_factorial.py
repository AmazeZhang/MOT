#!/usr/bin/env python3
"""Run B0/CV once each and fork only the emitted tracker rows for E1."""

from __future__ import annotations

import configparser
import csv
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_p1_common_state_mechanism import (
    read_detection_cache, read_tracker_rows, write_tracker_row,
)
from mot_research.analysis.d2_first_association import compare_tracker_replay
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, NaturalMotionFirstRoundPolicy,
    TrackerDetectionHistory,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e1_mature_reemit_factorial_20261119.yaml"
CODE_PATHS = (
    "scripts/run_e1_mature_reemit_factorial.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/d2_first_association.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "src/mot_research/data/motion_samples.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
)


def u_extra_rows(tracker, original, mature_before, before_hits, processed):
    """Read, but never mutate, live tracker state to construct U's extra rows."""
    before_bytes = pickle.dumps(tracker.trackers, protocol=5)
    existing = {int(row[4]) for row in original}
    extra = []
    details = []
    if processed:
        for item in tracker.trackers:
            track_id = int(item.id) + 1
            if track_id in existing or track_id not in mature_before:
                continue
            if track_id not in before_hits or item.hits != before_hits[track_id] + 1:
                continue
            if item.time_since_update != 0:
                continue
            observation = np.asarray(item.last_observation, dtype=np.float64).reshape(-1)
            if observation.size < 5 or not np.isfinite(observation[:5]).all():
                raise RuntimeError(f"U track {track_id}: no real updated detection")
            if observation[4] <= tracker.det_thresh:
                raise RuntimeError(f"U track {track_id}: update below high detection threshold")
            box = observation[:4]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise RuntimeError(f"U track {track_id}: invalid box")
            extra.append(np.r_[box, track_id])
            details.append({"track_id": track_id, "box_xyxy": box.tolist(),
                            "score": float(observation[4]),
                            "hits": int(item.hits), "hit_streak": int(item.hit_streak),
                            "mature_before": True})
    if pickle.dumps(tracker.trackers, protocol=5) != before_bytes:
        raise RuntimeError("U generation changed tracker state")
    if len({int(row[4]) for row in extra}) != len(extra):
        raise RuntimeError("U generated duplicate track IDs")
    return extra, details


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E1 requires CUDA_VISIBLE_DEVICES='' and CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    inp = protocol["inputs"]
    e0_path = ROOT / protocol["trigger"]["e0_protocol"]
    e0_protocol = yaml.safe_load(e0_path.read_text())
    e0_run = Path(e0_protocol["outputs"]["run_dir"])
    e0_summary = json.loads((e0_run / "summary.json").read_text())
    if e0_summary["status"] != "PASS" or not e0_summary["E1_trigger"]:
        raise RuntimeError("E0 did not satisfy the frozen E1 trigger")
    if e0_summary["protocol_sha256"] != sha256_file(e0_path):
        raise RuntimeError("E0 protocol changed")
    if sha256_file(Path(inp["prior_hashes"])) != inp["prior_hashes_sha256"]:
        raise RuntimeError("prior input hashes changed")
    if sha256_file(Path(protocol["events"]["path"])) != protocol["events"]["sha256"]:
        raise RuntimeError("fixed events changed")
    oc_commit = subprocess.run(["git", "-C", str(ROOT / inp["oc_sort_root"]),
                                "rev-parse", "HEAD"], check=True,
                               capture_output=True, text=True).stdout.strip()
    if oc_commit != inp["oc_sort_commit"]:
        raise RuntimeError("OC-SORT commit changed")
    prior = json.loads(Path(inp["prior_hashes"]).read_text())
    output = Path(protocol["outputs"]["run_dir"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(p.exists() for p in (output, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir(parents=True)
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        sys.path.insert(0, str((ROOT / inp["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL),
                           "e0_summary_sha256": sha256_file(e0_run / "summary.json"),
                           "e0_protocol_sha256": sha256_file(e0_path),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "oc_sort_commit": oc_commit,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        for condition in ("A", "B", "C", "D"):
            (partial / "trackers" / condition).mkdir(parents=True)
        (partial / "extra_outputs").mkdir()
        summaries = []
        for mode, original_condition, u_condition in (
            ("upstream", "A", "B"), ("constant_velocity", "C", "D")
        ):
            for sequence in protocol["scope"]["sequences"]:
                seq_dir = Path(inp["train_root"]) / sequence
                seqinfo = seq_dir / "seqinfo.ini"
                cache = Path(inp["detection_root"]) / f"{sequence}.csv"
                reference_root = inp["b0_tracker_root"] if mode == "upstream" else inp["cv_tracker_root"]
                reference = Path(reference_root) / f"{sequence}.txt"
                if sha256_file(seqinfo) != prior["sequences"][sequence]["seqinfo"] or \
                   sha256_file(cache) != prior["sequences"][sequence]["detections"]:
                    raise RuntimeError(f"{sequence}: input hash mismatch")
                if mode == "upstream" and sha256_file(reference) != prior["sequences"][sequence]["b0_tracker"]:
                    raise RuntimeError(f"{sequence}: B0 reference hash mismatch")
                config = configparser.ConfigParser()
                if not config.read(seqinfo):
                    raise FileNotFoundError(seqinfo)
                info = config["Sequence"]
                length = info.getint("seqLength")
                height, width = info.getint("imHeight"), info.getint("imWidth")
                fps = info.getfloat("frameRate")
                scale = min(800 / height, 1440 / width)
                cached, _ = read_detection_cache(cache)
                tracker = ocsort_module.OCSort(**inp["tracker"])
                history = TrackerDetectionHistory(history_length=inp["cv_history"]["length"],
                                                  score_threshold=.6)
                policy = NaturalMotionFirstRoundPolicy(
                    mode=mode, history=history, image_width=width,
                    image_height=height, fps=fps,
                    minimum_observations=inp["cv_history"]["minimum_observations"],
                    maximum_frame_gap=inp["cv_history"]["maximum_frame_gap"])
                runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
                original_file = partial / "trackers" / original_condition / f"{sequence}.txt"
                u_file = partial / "trackers" / u_condition / f"{sequence}.txt"
                extra_file = partial / "extra_outputs" / f"{mode}_{sequence}.jsonl"
                mature_ids = set()
                counters = Counter()
                with original_file.open("w", newline="") as original_handle, \
                     u_file.open("w", newline="") as u_handle, extra_file.open("w") as extra_handle:
                    original_writer = csv.writer(original_handle, lineterminator="\n")
                    u_writer = csv.writer(u_handle, lineterminator="\n")
                    for frame in range(1, length + 1):
                        before_hits = {int(t.id) + 1: int(t.hits) for t in tracker.trackers}
                        mature_before = set(mature_ids)
                        values = np.asarray(cached.get(frame, []), dtype=np.float64)
                        tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                        if tracker_input is not None:
                            tracker_input[:, :4] *= scale
                        original = runner.update(tracker_input, (height, width),
                                                 (800, 1440), frame_id=frame)
                        original = np.asarray(original, dtype=np.float64).reshape(-1, 5)
                        original_ids = [int(row[4]) for row in original]
                        if len(original_ids) != len(set(original_ids)):
                            raise RuntimeError(f"{mode}:{sequence}:{frame}: duplicate original output ID")
                        extra, details = u_extra_rows(tracker, original, mature_before,
                                                      before_hits, tracker_input is not None)
                        for row in original:
                            write_tracker_row(original_writer, frame, row)
                            write_tracker_row(u_writer, frame, row)
                        for row in extra:
                            write_tracker_row(u_writer, frame, row)
                        for item in details:
                            extra_handle.write(json.dumps({"sequence": sequence, "frame": frame,
                                                            **item}, separators=(",", ":")) + "\n")
                        counters["original_rows"] += len(original)
                        counters["U_extra_rows"] += len(extra)
                        for t in tracker.trackers:
                            track_id = int(t.id) + 1
                            if track_id in original_ids and t.hit_streak >= tracker.min_hits:
                                mature_ids.add(track_id)
                ref_keys, ref_boxes = read_tracker_rows(reference)
                run_keys, run_boxes = read_tracker_rows(original_file)
                replay = compare_tracker_replay(ref_keys, ref_boxes, run_keys, run_boxes, 1e-4)
                if replay["status"] != "PASS":
                    raise RuntimeError(f"{mode}:{sequence}: original output replay mismatch: {replay}")
                summary = {"mode": mode, "sequence": sequence,
                           "original_condition": original_condition, "U_condition": u_condition,
                           "frames": length, "original_reference_replay": replay,
                           "counts": dict(counters),
                           "original_sha256": sha256_file(original_file),
                           "U_sha256": sha256_file(u_file),
                           "U_extra_sha256": sha256_file(extra_file),
                           "runner": runner.snapshot()}
                summaries.append(summary)
                (partial / "sequence_progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
                print(f"{mode} {sequence}: replay PASS, U extra {counters['U_extra_rows']}", flush=True)
                if time.monotonic() - started > protocol["budget"]["max_wall_seconds"]:
                    raise TimeoutError("E1 wall budget exceeded")
        verify_code_unchanged(ROOT, provenance)
        core_hours = (time.process_time() - cpu_started) / 3600
        if core_hours > protocol["budget"]["max_cpu_core_hours"]:
            raise RuntimeError("E1 CPU budget exceeded")
        result = {"status": "PASS", "protocol_sha256": provenance["protocol_sha256"],
                  "e0_summary_sha256": provenance["e0_summary_sha256"],
                  "sequence_runs": len(summaries),
                  "U_extra_rows": {condition: sum(s["counts"]["U_extra_rows"] for s in summaries
                                              if s["U_condition"] == condition)
                                   for condition in ("B", "D")},
                  "sequences": summaries, "wall_seconds": time.monotonic() - started,
                  "cpu_core_hours": core_hours, "peak_rss_kib": peak_rss_kib(),
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "U_extra_rows": result["U_extra_rows"],
                          "output": str(output)}, indent=2), flush=True)
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
