#!/usr/bin/env python3
"""Read-only parent replay to record normal first-round ownership of E5 action detections."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mot_research.analysis.failure_events import read_tracker
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from run_e4_oracle_association_headroom import same_outputs, step_e
from run_e5_nonoracle_appearance_local import CapturePolicy, make_runner, sequence_inputs

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "manifests/protocols/e5_baseline_pair_trace_addendum_20261120.yaml"
CODE_PATHS = (
    "scripts/derive_e5_baseline_pair_trace.py",
    "scripts/run_e5_nonoracle_appearance_local.py",
    "scripts/run_e4_oracle_association_headroom.py",
    "scripts/run_e1_mature_reemit_factorial.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "src/mot_research/data/motion_samples.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
)


def record_event(raw, capture, high_ids):
    if capture.called != 1 or capture.request is None:
        raise RuntimeError("selected event had no unique first-round call")
    request = capture.request
    if len(high_ids) != len(request.detections):
        raise RuntimeError("high detection index mismatch")
    target = int(raw["track_id"])
    if target not in request.tracker_ids:
        raise RuntimeError("E5 action target absent in first round")
    track_to_det = {}
    det_to_track = {}
    for det_index, track_index in capture.pairs:
        det_id = high_ids[det_index]
        track_id = int(request.tracker_ids[track_index])
        if det_id in det_to_track or track_id in track_to_det:
            raise RuntimeError("normal first-round pairing not one-to-one")
        det_to_track[det_id] = track_id
        track_to_det[track_id] = det_id
    picks = {}
    for method in ("geometry", "appearance"):
        index = int(raw["action_features"][method]["det_index"])
        det_id = high_ids[index]
        if det_id != raw["selected_detection_ids"][method]:
            raise RuntimeError("E5 selected detection index drift")
        picks[method] = {"detection_id": det_id, "baseline_first_round_owner_track_id": det_to_track.get(det_id)}
    return {
        "sequence": raw["sequence"], "frame": raw["frame"],
        "track_id": target, "anchor_frame": raw["anchor_frame"],
        "baseline_target_first_round_detection_id": track_to_det.get(target),
        "selected": picks,
        "normal_first_round_pairs": [{"detection_id": det, "track_id": track}
                                     for det, track in sorted(det_to_track.items())],
        "high_detection_ids": high_ids,
    }


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("pair trace requires CUDA_VISIBLE_DEVICES='' and CPU only")
    plan = yaml.safe_load(PLAN.read_text())
    inp = plan["inputs"]
    e5_run = Path(inp["e5_run"])
    e5_summary = e5_run / "summary.json"
    if sha256_file(e5_summary) != inp["e5_summary_sha256"]:
        raise RuntimeError("E5 summary changed")
    analysis_summary = Path(inp["e5_analysis_run"]) / "summary.json"
    if sha256_file(analysis_summary) != inp["e5_analysis_summary_sha256"]:
        raise RuntimeError("E5 read-only analysis changed")
    e5 = json.loads(e5_summary.read_text())
    e5_protocol = yaml.safe_load((ROOT / "manifests/protocols/e5_nonoracle_appearance_local_20261119.yaml").read_text())
    tracker_inp = e5_protocol["inputs"]
    output = Path(plan["outputs"]["run_dir"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(p.exists() for p in (output, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.mkdir(parents=True)
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"analysis_protocol_sha256": sha256_file(PLAN),
                           "e5_summary_sha256": inp["e5_summary_sha256"],
                           "analysis_summary_sha256": inp["e5_analysis_summary_sha256"],
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        sys.path.insert(0, str((ROOT / tracker_inp["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort
        seq_summaries = []
        for seq_info in e5["sequences"]:
            seq = seq_info["sequence"]
            seq_dir, length, height, width, fps, cache_path, cached, det_ids, frame_input = sequence_inputs(seq, tracker_inp)
            if sha256_file(cache_path) != seq_info["cache_sha256"] or \
                    sha256_file(seq_dir / "seqinfo.ini") != seq_info["seqinfo_sha256"]:
                raise RuntimeError(f"{seq}: E5 input changed")
            c_path = Path(inp["e3_fit_trackers"]) / "C" / f"{seq}.txt"
            e_path = Path(inp["e3_fit_trackers"]) / "E" / f"{seq}.txt"
            c_ref, e_ref = read_tracker(c_path), read_tracker(e_path)
            raw_path = e5_run / f"{seq}.raw.jsonl"
            if sha256_file(raw_path) != seq_info["raw_sha256"]:
                raise RuntimeError(f"{seq}: E5 raw events changed")
            events = [json.loads(line) for line in raw_path.read_text().splitlines()]
            by_frame = defaultdict(list)
            for event in events:
                by_frame[event["frame"]].append(event)
            runner = make_runner(ocsort, tracker_inp, height, width, fps)
            mature = set()
            traced = []
            for frame in range(1, length + 1):
                selected = by_frame.get(frame, [])
                capture = CapturePolicy(runner.policy) if selected else None
                if capture is not None:
                    runner.adapter.policy = capture
                c_now, e_now = step_e(runner, mature, frame_input(frame), frame, height, width, tracker_inp)
                if capture is not None:
                    runner.adapter.policy = runner.policy
                if not same_outputs(c_now, c_ref.get(frame, [])) or \
                        not same_outputs(e_now, e_ref.get(frame, [])):
                    raise RuntimeError(f"{seq}:{frame}: parent replay differs from E3")
                if selected:
                    high_ids = [int(did) for did, value in zip(det_ids.get(frame, []), cached.get(frame, []))
                                if float(value[4]) > .6]
                    for event in selected:
                        traced.append(record_event(event, capture, high_ids))
                if time.monotonic() - started > plan["budget"]["max_wall_seconds"]:
                    raise TimeoutError("E5 pair trace wall budget exceeded")
            if len(traced) != len(events):
                raise RuntimeError(f"{seq}: not all E5 action events traced")
            path = partial / f"{seq}.pairs.jsonl"
            with path.open("w") as handle:
                for row in traced:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            seq_summaries.append({"sequence": seq, "frames": length, "events": len(traced),
                                  "pairs_sha256": sha256_file(path),
                                  "C_reference_sha256": sha256_file(c_path),
                                  "E_reference_sha256": sha256_file(e_path)})
            (partial / "sequence_progress.json").write_text(json.dumps(seq_summaries, indent=2) + "\n")
            print(f"{seq}: traced={len(traced)} frames={length}", flush=True)
        verify_code_unchanged(ROOT, provenance)
        if len(seq_summaries) != 24 or sum(x["events"] for x in seq_summaries) != 576:
            raise RuntimeError("unexpected trace coverage")
        result = {"status": "PASS", "analysis_protocol_sha256": provenance["analysis_protocol_sha256"],
                  "e5_summary_sha256": inp["e5_summary_sha256"], "sequences": seq_summaries,
                  "total_events": 576, "wall_seconds": time.monotonic() - started,
                  "cpu_core_hours": (time.process_time() - cpu_started) / 3600,
                  "peak_rss_kib": peak_rss_kib(),
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "total_events": 576, "output": str(output)}, indent=2))
    except BaseException as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
