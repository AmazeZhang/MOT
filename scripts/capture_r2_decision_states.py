#!/usr/bin/env python3
"""Capture frozen R2 event-frame candidates from a read-only B0 replay."""

from __future__ import annotations

import configparser
import csv
import json
import os
import pickle
import sys
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
from mot_research.analysis.runtime_record import (
    runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, NaturalMotionFirstRoundPolicy,
    TorchMotionPredictor, TrackerDetectionHistory,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/r2_decision_rules_p1_20261118.yaml"
CODE_PATHS = (
    "scripts/capture_r2_decision_states.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/d2_first_association.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "src/mot_research/data/motion_samples.py",
    "src/mot_research/models/deterministic_motion.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/kalmanfilter.py",
)


def to_list(array):
    return np.asarray(array, dtype=np.float64).tolist()


class CapturePolicy:
    def __init__(self, tracker, history, alternatives, frame_events, detection_ids,
                 sink, sequence):
        self.tracker = tracker
        self.history = history
        self.alternatives = alternatives
        self.frame_events = frame_events
        self.detection_ids = detection_ids
        self.sink = sink
        self.sequence = sequence
        self.seen = set()
        self.counters = Counter()

    def snapshot(self):
        return {"mode": "b0_read_only_capture", "counters": dict(self.counters),
                "alternatives": {k: v.snapshot() for k, v in self.alternatives.items()}}

    def __call__(self, request, upstream_associate):
        result = upstream_associate(
            request.detections, request.trackers, request.iou_threshold,
            request.velocities, request.previous_observations,
            request.velocity_direction_weight,
        )
        events = self.frame_events.get(request.frame_id)
        if not events:
            return result
        if request.frame_id in self.seen:
            raise RuntimeError(f"duplicate event frame {self.sequence}:{request.frame_id}")
        self.seen.add(request.frame_id)
        frame_detection_ids = self.detection_ids.get(request.frame_id, [])
        if len(frame_detection_ids) != len(request.detections):
            raise RuntimeError(f"detection IDs mismatch at {self.sequence}:{request.frame_id}")
        if tuple(int(t.id) + 1 for t in self.tracker.trackers) != request.tracker_ids:
            raise RuntimeError("KF states and association tracker IDs differ")
        state_before = pickle.dumps(self.tracker.trackers, protocol=5)
        proposals = {}
        for name, policy in self.alternatives.items():
            captured = []

            def capture(detections, trackers, threshold, velocities, previous, weight):
                captured.append(np.asarray(trackers).copy())
                return upstream_associate(detections, trackers, threshold,
                                          velocities, previous, weight)

            policy(request, capture)
            if len(captured) != 1:
                raise RuntimeError(f"{name}: expected one proposal callback")
            proposals[name] = to_list(captured[0][:, :4])
            if pickle.dumps(self.tracker.trackers, protocol=5) != state_before:
                raise RuntimeError(f"{name}: changed B0 tracker state")
        event_ids = [str(e["event_id"]) for e in events]
        for event in events:
            if int(event["positive_detection_id"]) not in frame_detection_ids:
                raise RuntimeError(f"{event['event_id']}: positive detection absent")
        observations = []
        for track_id in request.tracker_ids:
            observations.append([
                {"frame": obs.frame_id, "box": to_list(obs.box_xyxy), "score": obs.score}
                for obs in self.history.observations(track_id)
            ])
        row = {
            "sequence": self.sequence, "frame": request.frame_id,
            "event_ids": event_ids,
            "detection_ids": frame_detection_ids,
            "detections_xyxy_score": to_list(request.detections),
            "tracker_ids": list(request.tracker_ids),
            "predicted_kf_boxes_xyxy": to_list(request.trackers[:, :4]),
            "kf_x_after_predict": [to_list(t.kf.x.reshape(-1)) for t in self.tracker.trackers],
            "kf_P_after_predict": [to_list(t.kf.P) for t in self.tracker.trackers],
            "velocities": to_list(request.velocities),
            "previous_observations": to_list(request.previous_observations),
            "direction_weight": request.velocity_direction_weight,
            "iou_threshold": request.iou_threshold,
            "history_observations": observations,
            "proposals_xyxy": proposals,
            "b0_matches_rows": to_list(result[0]),
        }
        self.sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.counters["captured_frames"] += 1
        self.counters["captured_events"] += len(events)
        return result


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R2 capture requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    inputs = protocol["inputs"]
    events_spec = protocol["events"]
    event_path = Path(events_spec["path"])
    if sha256_file(event_path) != events_spec["sha256"]:
        raise RuntimeError("fixed event CSV hash mismatch")
    if sha256_file(Path(inputs["prior_hash_table"])) != inputs["prior_hash_table_sha256"]:
        raise RuntimeError("prior input hash table mismatch")
    if sha256_file(ROOT / protocol["scope"]["split_manifest"]) != protocol["scope"]["split_sha256"]:
        raise RuntimeError("motion-dev split hash mismatch")
    prior_hashes = json.loads(Path(inputs["prior_hash_table"]).read_text())
    for name, spec in inputs["checkpoints"].items():
        if sha256_file(Path(spec["path"])) != spec["sha256"]:
            raise RuntimeError(f"{name}: checkpoint hash mismatch")
    if not all(Path(inputs[k]).is_dir() for k in
               ("detection_root", "train_root", "b0_tracker_root")):
        raise FileNotFoundError("one or more input roots missing")
    import subprocess
    oc_commit = subprocess.run(["git", "-C", str(ROOT / inputs["oc_sort_root"]),
                                "rev-parse", "HEAD"], capture_output=True,
                               text=True, check=True).stdout.strip()
    if oc_commit != inputs["oc_sort_commit"]:
        raise RuntimeError(f"OC-SORT commit mismatch: {oc_commit}")
    output = Path(protocol["outputs"]["capture_run"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(p.exists() for p in (output, partial, failed)):
        raise FileExistsError(f"capture output already exists: {output}")
    partial.mkdir(parents=True)
    try:
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str((ROOT / inputs["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module
        run_meta = runtime_record(ROOT, CODE_PATHS)
        run_meta.update({"protocol_sha256": sha256_file(PROTOCOL),
                         "events_sha256": sha256_file(event_path),
                         "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                         "torch_version": torch.__version__,
                         "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(run_meta, indent=2) + "\n")
        events = read_events(event_path)
        if len(events) != events_spec["expected_unique"]:
            raise RuntimeError("event count mismatch")
        by_sequence = defaultdict(lambda: defaultdict(list))
        for event in events:
            by_sequence[event["sequence"]][event["after_frame"]].append(event)
        if set(by_sequence) != set(protocol["scope"]["sequence_order"]):
            raise RuntimeError("event sequence mismatch")
        predictors = {}
        for name, spec in inputs["checkpoints"].items():
            checkpoint = torch.load(spec["path"], map_location="cpu", weights_only=True)
            predictors[name] = TorchMotionPredictor(checkpoint, "cpu")
        (partial / "states").mkdir()
        (partial / "b0_replay").mkdir()
        summaries = []
        for sequence in protocol["scope"]["sequence_order"]:
            sequence_dir = Path(inputs["train_root"]) / sequence
            seqinfo = sequence_dir / "seqinfo.ini"
            cache = Path(inputs["detection_root"]) / f"{sequence}.csv"
            reference = Path(inputs["b0_tracker_root"]) / f"{sequence}.txt"
            prior = prior_hashes["sequences"][sequence]
            for key, path in (("seqinfo", seqinfo), ("detections", cache),
                              ("b0_tracker", reference)):
                if sha256_file(path) != prior[key]:
                    raise RuntimeError(f"{sequence}: input {key} hash mismatch")
            config = configparser.ConfigParser()
            if not config.read(seqinfo):
                raise FileNotFoundError(seqinfo)
            info = config["Sequence"]
            height, width = info.getint("imHeight"), info.getint("imWidth")
            fps, length = info.getfloat("frameRate"), info.getint("seqLength")
            input_height, input_width = protocol["inputs"]["input_size"]
            scale = min(input_height / height, input_width / width)
            cached, detection_ids = read_detection_cache(cache)
            hspec = inputs["history"]
            history = TrackerDetectionHistory(history_length=hspec["length"], score_threshold=0.6)
            alternatives = {"constant_velocity": NaturalMotionFirstRoundPolicy(
                mode="constant_velocity", history=history, image_width=width,
                image_height=height, fps=fps, minimum_observations=hspec["minimum_observations"],
                maximum_frame_gap=hspec["maximum_frame_gap"])}
            for name, predictor in predictors.items():
                alternatives[name] = NaturalMotionFirstRoundPolicy(
                    mode="learned", history=history, image_width=width,
                    image_height=height, fps=fps, predictor=predictor,
                    minimum_observations=hspec["minimum_observations"],
                    maximum_frame_gap=hspec["maximum_frame_gap"])
            tracker = ocsort_module.OCSort(**inputs["tracker"])
            replay_path = partial / "b0_replay" / f"{sequence}.txt"
            state_path = partial / "states" / f"{sequence}.jsonl"
            with state_path.open("w") as sink, replay_path.open("w", newline="") as replay_handle:
                policy = CapturePolicy(tracker, history, alternatives,
                                       by_sequence[sequence], detection_ids, sink, sequence)
                runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
                writer = csv.writer(replay_handle, lineterminator="\n")
                for frame in range(1, length + 1):
                    values = np.asarray(cached.get(frame, []), dtype=np.float64)
                    tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                    if tracker_input is not None:
                        tracker_input[:, :4] *= scale
                    tracks = runner.update(tracker_input, (height, width),
                                           (input_height, input_width), frame_id=frame)
                    for track in tracks:
                        write_tracker_row(writer, frame, track)
            if policy.seen != set(by_sequence[sequence]):
                raise RuntimeError(f"{sequence}: missing event frames {set(by_sequence[sequence]) - policy.seen}")
            ref_keys, ref_boxes = read_tracker_rows(reference)
            replay_keys, replay_boxes = read_tracker_rows(replay_path)
            replay = compare_tracker_replay(ref_keys, ref_boxes, replay_keys,
                                            replay_boxes, 1e-4)
            if replay["status"] != "PASS":
                raise RuntimeError(f"{sequence}: B0 replay mismatch {replay}")
            summaries.append({"sequence": sequence, "events": policy.counters["captured_events"],
                              "frames": policy.counters["captured_frames"], "b0_replay": replay,
                              "state_sha256": sha256_file(state_path),
                              "tracker_sha256": sha256_file(replay_path),
                              "runner": runner.snapshot()})
            (partial / "sequence_progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"{sequence}: {policy.counters['captured_events']} events, replay PASS", flush=True)
        if sum(x["events"] for x in summaries) != 831:
            raise RuntimeError("captured event coverage mismatch")
        verify_code_unchanged(ROOT, run_meta)
        (partial / "summary.json").write_text(json.dumps({
            "status": "PASS", "event_count": 831,
            "event_frame_count": sum(x["frames"] for x in summaries),
            "sequences": summaries, "protocol_sha256": run_meta["protocol_sha256"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        os.replace(partial, output)
        print(f"capture PASS: {output}", flush=True)
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
