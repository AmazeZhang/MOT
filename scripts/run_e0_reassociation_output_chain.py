#!/usr/bin/env python3
"""Replay B0 and record the detection-to-output chain at fixed event windows."""

from __future__ import annotations

import configparser
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_p1_common_state_mechanism import (
    read_detection_cache, read_events, read_tracker_rows, write_tracker_row,
)
from mot_research.analysis.d2_first_association import compare_tracker_replay
from mot_research.analysis.failure_events import (
    Candidate, hungarian_iou_matches, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, TrackerDetectionHistory,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e0_reassociation_output_chain_20261119.yaml"
CODE_PATHS = (
    "scripts/run_e0_reassociation_output_chain.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/d2_first_association.py",
    "src/mot_research/analysis/failure_events.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/kalmanfilter.py",
)


def tracker_snapshot(tracker):
    return {"id": int(tracker.id) + 1, "hits": int(tracker.hits),
            "hit_streak": int(tracker.hit_streak),
            "time_since_update": int(tracker.time_since_update),
            "age": int(tracker.age)}


class ReadOnlyPolicy:
    def __init__(self, tracker, history, detection_ids, interested):
        self.tracker = tracker
        self.history = history
        self.detection_ids = detection_ids
        self.interested = interested
        self.calls = 0
        self.requests = {}

    def snapshot(self):
        return {"mode": "exact_B0_read_only", "association_calls": self.calls}

    def __call__(self, request, upstream_associate):
        self.calls += 1
        before = pickle.dumps(self.tracker.trackers, protocol=5)
        result = upstream_associate(
            request.detections, request.trackers, request.iou_threshold,
            request.velocities, request.previous_observations,
            request.velocity_direction_weight,
        )
        if pickle.dumps(self.tracker.trackers, protocol=5) != before:
            raise RuntimeError("B0 capture callback changed tracker state")
        if request.frame_id in self.interested:
            ids = self.detection_ids.get(request.frame_id, [])
            if len(ids) != len(request.detections):
                raise RuntimeError(f"detection ID mismatch at frame {request.frame_id}")
            pairs = [[ids[int(d)], int(request.tracker_ids[int(t)])]
                     for d, t in np.asarray(result[0], dtype=np.int64).reshape(-1, 2)]
            self.requests[request.frame_id] = {
                "first_round_pairs": pairs,
                "post_predict_tracks": [tracker_snapshot(t) for t in self.tracker.trackers],
                "first_round_unmatched_detection_ids": [ids[int(i)] for i in result[1]],
                "first_round_unmatched_track_ids": [int(request.tracker_ids[int(i)])
                                                    for i in result[2]],
            }
        return result


def map_updates(after_trackers, high_values, high_ids, first_pairs, frame_count,
                returned_ids, processed):
    values = np.asarray(high_values, dtype=np.float64).reshape(-1, 5)
    first = {track_id: det_id for det_id, track_id in first_pairs}
    update_map = {}
    used = set()
    states = {}
    for tracker in after_trackers:
        track_id = int(tracker.id) + 1
        state = tracker_snapshot(tracker)
        updated = processed and state["time_since_update"] == 0
        detection_id = None
        if updated:
            observation = np.asarray(tracker.last_observation, dtype=np.float64).reshape(-1)
            if observation.size >= 5 and observation[4] >= 0:
                candidates = [i for i, value in enumerate(values)
                              if np.max(np.abs(value - observation[:5])) <= 1e-4]
            else:
                box = np.asarray(tracker.get_state()[0], dtype=np.float64)
                candidates = [i for i, value in enumerate(values)
                              if np.max(np.abs(value[:4] - box)) <= 1e-4]
            if len(candidates) != 1:
                raise RuntimeError(f"track {track_id}: updated detection mapping {candidates}")
            detection_id = int(high_ids[candidates[0]])
            if detection_id in used:
                raise RuntimeError(f"detection {detection_id}: updated two tracks")
            used.add(detection_id)
        gate = bool(updated and (state["hit_streak"] >= 3 or frame_count <= 3))
        returned = track_id in returned_ids
        if gate != returned:
            raise RuntimeError(f"track {track_id}: output gate/actual return disagreement")
        state.update({"update_detection_id": detection_id,
                      "output_gate_pass": gate, "emitted": returned,
                      "first_round_detection_id": first.get(track_id)})
        states[track_id] = state
        if detection_id is not None:
            update_map[detection_id] = track_id
    return states, update_map


def label_event(event, frame_records, gt_matches, sequence_length):
    frame = int(event["after_frame"])
    record = frame_records[frame]
    intended = int(event["intended_track_id"])
    positive = int(event["positive_detection_id"])
    first_pairs = {(int(d), int(t)) for d, t in record["first_round_pairs"]}
    post_predict = {int(t["id"]): t for t in record["post_predict_tracks"]}
    after = {int(t["id"]): t for t in record["after_tracks"]}
    alive = intended in post_predict
    positive_in_pool = positive in record["high_detection_ids"]
    target_update = after.get(intended, {}).get("update_detection_id")
    target_emitted = intended in record["output_ids"]
    assigned = gt_matches.get(frame, {}).get(int(event["gt_id"]))
    assigned_id = None if assigned is None else int(assigned[0])
    reconnect = assigned_id == intended
    positive_final = target_update == positive
    if not alive:
        category = "target_track_absent_before_association"
    elif not positive_in_pool:
        category = "positive_detection_absent_from_high_pool"
    elif not positive_final:
        category = "target_not_finally_updated_by_positive_detection"
    elif not target_emitted:
        category = "target_updated_by_positive_but_not_emitted"
    elif not reconnect:
        category = "target_emitted_but_GT_not_assigned_to_target_ID"
    else:
        category = "no_break_reconnect"
    window_end = min(sequence_length, frame + 5)
    output_status = []
    for current in range(frame, window_end + 1):
        match = gt_matches.get(current, {}).get(int(event["gt_id"]))
        output_status.append({"frame": current,
                              "assigned_track_id": None if match is None else int(match[0]),
                              "target_emitted": intended in frame_records[current]["output_ids"]})
    reemit = next((item["frame"] - frame for item in output_status
                   if item["assigned_track_id"] == intended), None)
    subsequent = [item["assigned_track_id"] for item in output_status[1:]]
    return {
        "event_id": event["event_id"], "sequence": event["sequence"],
        "frame": frame, "gt_id": int(event["gt_id"]),
        "baseline_stratum": event["baseline_stratum"],
        "category": category, "target_alive": alive,
        "positive_detection_in_high_pool": positive_in_pool,
        "first_round_positive_pair": (positive, intended) in first_pairs,
        "final_positive_pair": positive_final,
        "OCR_positive_pair": positive_final and (positive, intended) not in first_pairs,
        "target_update_detection_id": target_update,
        "positive_detection_update_track_id": record["update_map"].get(str(positive)),
        "target_emitted": target_emitted,
        "output_gate_suppressed": positive_final and not target_emitted,
        "prior_mature": intended in record["mature_before_ids"],
        "hit_streak_before": record["before_tracks_by_id"].get(str(intended), {}).get("hit_streak"),
        "hit_streak_after_predict": post_predict.get(intended, {}).get("hit_streak"),
        "hit_streak_after_update": after.get(intended, {}).get("hit_streak"),
        "hits_after_update": after.get(intended, {}).get("hits"),
        "time_since_update_after": after.get(intended, {}).get("time_since_update"),
        "GT_assigned_output_id": assigned_id,
        "reconnect": reconnect,
        "first_same_ID_reemit_delay": reemit,
        "later_same_ID_lost_or_switched": reconnect and any(x != intended for x in subsequent),
        "window_censored_by_sequence_end": window_end < frame + 5,
        "window_output_status": output_status,
    }


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E0 requires CUDA_VISIBLE_DEVICES='' and CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    inp = protocol["inputs"]
    event_path = Path(protocol["events"]["path"])
    if sha256_file(event_path) != protocol["events"]["sha256"]:
        raise RuntimeError("events hash mismatch")
    if sha256_file(ROOT / protocol["scope"]["split_manifest"]) != protocol["scope"]["split_sha256"]:
        raise RuntimeError("split hash mismatch")
    hash_path = Path(inp["prior_input_hashes"])
    if sha256_file(hash_path) != inp["prior_input_hashes_sha256"]:
        raise RuntimeError("prior hashes mismatch")
    prior_hashes = json.loads(hash_path.read_text())
    oc_commit = subprocess.run(["git", "-C", str(ROOT / inp["oc_sort_root"]),
                                "rev-parse", "HEAD"], check=True,
                               capture_output=True, text=True).stdout.strip()
    if oc_commit != inp["oc_sort_commit"]:
        raise RuntimeError("OC-SORT commit mismatch")
    events = read_events(event_path)
    if len(events) != protocol["events"]["expected_count"]:
        raise RuntimeError("event count mismatch")
    if Counter(e["baseline_stratum"] for e in events) != protocol["events"]["expected_strata"]:
        raise RuntimeError("event strata mismatch")
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
                           "events_sha256": sha256_file(event_path),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "oc_sort_commit": oc_commit,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        by_sequence = defaultdict(list)
        for event in events:
            by_sequence[event["sequence"]].append(event)
        (partial / "b0_replay").mkdir()
        (partial / "window_frames").mkdir()
        (partial / "event_rows").mkdir()
        all_rows = []
        sequence_summaries = []
        for sequence in protocol["scope"]["sequences"]:
            seq_dir = Path(inp["train_root"]) / sequence
            seqinfo = seq_dir / "seqinfo.ini"
            cache = Path(inp["detection_root"]) / f"{sequence}.csv"
            reference = Path(inp["b0_tracker_root"]) / f"{sequence}.txt"
            gt_path = seq_dir / "gt" / "gt.txt"
            for key, path in (("seqinfo", seqinfo), ("detections", cache),
                              ("b0_tracker", reference), ("gt", gt_path)):
                if sha256_file(path) != prior_hashes["sequences"][sequence][key]:
                    raise RuntimeError(f"{sequence}: {key} input hash mismatch")
            config = configparser.ConfigParser()
            if not config.read(seqinfo):
                raise FileNotFoundError(seqinfo)
            info = config["Sequence"]
            length = info.getint("seqLength")
            height = info.getint("imHeight")
            width = info.getint("imWidth")
            scale = min(800 / height, 1440 / width)
            cached, detection_ids = read_detection_cache(cache)
            interested = {frame for event in by_sequence[sequence]
                          for frame in range(event["after_frame"],
                                             min(length, event["after_frame"] + 5) + 1)}
            tracker = ocsort_module.OCSort(**inp["tracker"])
            history = TrackerDetectionHistory(history_length=inp["history"]["length"],
                                              score_threshold=0.6)
            policy = ReadOnlyPolicy(tracker, history, detection_ids, interested)
            runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
            replay_path = partial / "b0_replay" / f"{sequence}.txt"
            window_path = partial / "window_frames" / f"{sequence}.jsonl"
            frame_records = {}
            mature_ids = set()
            with replay_path.open("w", newline="") as replay_handle, window_path.open("w") as window_handle:
                writer = csv.writer(replay_handle, lineterminator="\n")
                for frame in range(1, length + 1):
                    before = {str(t.id + 1): tracker_snapshot(t) for t in tracker.trackers}
                    mature_before = sorted(mature_ids)
                    values = np.asarray(cached.get(frame, []), dtype=np.float64)
                    tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                    if tracker_input is not None:
                        tracker_input[:, :4] *= scale
                    tracks = runner.update(tracker_input, (height, width),
                                           (800, 1440), frame_id=frame)
                    for track in tracks:
                        write_tracker_row(writer, frame, track)
                    output_ids = [int(t[4]) for t in tracks]
                    if len(output_ids) != len(set(output_ids)):
                        raise RuntimeError(f"{sequence}:{frame}: duplicate output track ID")
                    if frame not in interested:
                        for t in tracker.trackers:
                            if int(t.id) + 1 in output_ids and t.hit_streak >= tracker.min_hits:
                                mature_ids.add(int(t.id) + 1)
                        continue
                    req = policy.requests.pop(frame, {"first_round_pairs": [],
                                                      "post_predict_tracks": [],
                                                      "first_round_unmatched_detection_ids": [],
                                                      "first_round_unmatched_track_ids": []})
                    high_ids = detection_ids.get(frame, [])
                    high_values = [v for v in cached.get(frame, []) if v[4] > .6]
                    if len(high_ids) != len(high_values):
                        raise RuntimeError("high detection ID/value mismatch")
                    states, update_map = map_updates(tracker.trackers, high_values,
                                                      high_ids, req["first_round_pairs"],
                                                      tracker.frame_count, set(output_ids),
                                                      tracker_input is not None)
                    for track_id in output_ids:
                        if states[track_id]["hit_streak"] >= tracker.min_hits:
                            mature_ids.add(track_id)
                    record = {
                        "sequence": sequence, "frame": frame,
                        "high_detection_ids": high_ids,
                        "high_detections_xyxy_score": high_values,
                        "before_tracks_by_id": before,
                        "post_predict_tracks": req["post_predict_tracks"],
                        "first_round_pairs": req["first_round_pairs"],
                        "first_round_unmatched_detection_ids": req["first_round_unmatched_detection_ids"],
                        "first_round_unmatched_track_ids": req["first_round_unmatched_track_ids"],
                        "after_tracks": list(states.values()),
                        "update_map": {str(k): v for k, v in update_map.items()},
                        "output_rows_xyxy_id": np.asarray(tracks, dtype=float).tolist(),
                        "output_ids": output_ids,
                        "mature_before_ids": mature_before,
                        "mature_after_ids": sorted(mature_ids),
                    }
                    frame_records[frame] = record
                    window_handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if set(frame_records) != interested:
                raise RuntimeError(f"{sequence}: missing window frames")
            ref_keys, ref_boxes = read_tracker_rows(reference)
            replay_keys, replay_boxes = read_tracker_rows(replay_path)
            replay = compare_tracker_replay(ref_keys, ref_boxes, replay_keys,
                                            replay_boxes, 1e-4)
            if replay["status"] != "PASS":
                raise RuntimeError(f"{sequence}: B0 replay mismatch: {replay}")
            # All GT reads and labels occur after the complete B0 replay and comparison.
            gt = read_gt(gt_path)
            outputs = read_tracker(replay_path)
            gt_matches = {}
            for frame in interested:
                gt_matches[frame] = hungarian_iou_matches(
                    gt.get(frame, {}), outputs.get(frame, []), .5)[0]
            seq_rows = [label_event(event, frame_records, gt_matches, length)
                        for event in by_sequence[sequence]]
            all_rows.extend(seq_rows)
            event_file = partial / "event_rows" / f"{sequence}.jsonl"
            with event_file.open("w") as handle:
                for row in seq_rows:
                    handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            summary = {"sequence": sequence, "events": len(seq_rows),
                       "window_frames": len(interested), "b0_replay": replay,
                       "categories": dict(Counter(row["category"] for row in seq_rows)),
                       "window_sha256": sha256_file(window_path),
                       "event_rows_sha256": sha256_file(event_file),
                       "replay_sha256": sha256_file(replay_path)}
            sequence_summaries.append(summary)
            (partial / "sequence_progress.json").write_text(json.dumps(sequence_summaries, indent=2) + "\n")
            print(f"{sequence}: {len(seq_rows)} events, B0 replay PASS", flush=True)
            if time.monotonic() - started > protocol["budget"]["max_wall_seconds"]:
                raise TimeoutError("E0 wall budget exceeded")
        if len(all_rows) != 831 or len({r["event_id"] for r in all_rows}) != 831:
            raise RuntimeError("E0 event coverage mismatch")
        expected = protocol["label"]["reconcile_known_counts_as_checks"]
        checks = {
            "target_alive": sum(r["target_alive"] for r in all_rows),
            "first_round_positive_pair": sum(r["first_round_positive_pair"] for r in all_rows),
            "reconnect_total": sum(r["reconnect"] for r in all_rows),
            "first_round_positive_and_reconnect": sum(
                r["first_round_positive_pair"] and r["reconnect"] for r in all_rows),
        }
        if checks != expected:
            raise RuntimeError(f"historical count reconciliation failed: {checks} != {expected}")
        selected = {}
        for category in protocol["label"]["first_break_categories_in_priority_order"]:
            members = [r for r in all_rows if r["category"] == category]
            members.sort(key=lambda r: hashlib.sha256(r["event_id"].encode("ascii")).hexdigest())
            selected[category] = [r["event_id"] for r in members[:2]]
        trigger = any(r["category"] == "target_updated_by_positive_but_not_emitted"
                      and r["prior_mature"] for r in all_rows)
        verify_code_unchanged(ROOT, provenance)
        elapsed = time.monotonic() - started
        core_hours = (time.process_time() - cpu_started) / 3600
        if core_hours > protocol["budget"]["max_cpu_core_hours"]:
            raise RuntimeError("E0 CPU core-hour budget exceeded")
        summary = {
            "status": "PASS", "scope": "8 motion-dev B0 replay; exploratory fixed 831 events",
            "protocol_sha256": provenance["protocol_sha256"],
            "event_count": len(all_rows), "reconciled_counts": checks,
            "categories": dict(Counter(r["category"] for r in all_rows)),
            "prior_mature_suppressed": sum(
                r["category"] == "target_updated_by_positive_but_not_emitted"
                and r["prior_mature"] for r in all_rows),
            "E1_trigger": trigger, "fixed_case_ids": selected,
            "sequence_summaries": sequence_summaries,
            "wall_seconds": elapsed, "cpu_core_hours": core_hours,
            "peak_rss_kib": peak_rss_kib(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "categories": summary["categories"],
                          "E1_trigger": trigger, "output": str(output)}, indent=2), flush=True)
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "error": str(exc), "type": type(exc).__name__,
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
