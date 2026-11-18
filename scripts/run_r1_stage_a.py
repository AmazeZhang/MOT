#!/usr/bin/env python3
"""R1 Stage A: CPU-only, same-state paired association interventions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from mot_research.analysis.failure_events import Candidate, hungarian_iou_matches, read_gt
from mot_research.analysis.r1_components import DifferenceComponent, components, switched_result
from mot_research.analysis.runtime_record import (
    R1_SHARED_CODE_PATHS, peak_rss_kib, runtime_record, verify_code_unchanged,
)
from mot_research.tracking.first_round_adapter import FirstRoundAssociationAdapter
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner,
    NaturalMotionFirstRoundPolicy,
    TrackerDetectionHistory,
)
from run_p1_natural_closed_loop import load_cache, sha256_file


class ProbePolicy:
    """Run B0 while evaluating a CV proposal on the same first-round input."""

    def __init__(self, history, width, height, fps):
        self.history = history
        self.cv = NaturalMotionFirstRoundPolicy(
            mode="constant_velocity", history=history, image_width=width,
            image_height=height, fps=fps,
        )
        self.by_frame = {}

    def __call__(self, request, associate):
        a0 = associate(request.detections, request.trackers, request.iou_threshold,
                       request.velocities, request.previous_observations,
                       request.velocity_direction_weight)
        a1 = self.cv(request, associate)
        found = components(a0[0], a1[0], len(request.detections), len(request.trackers))
        self.by_frame[request.frame_id] = {
            "a0": a0, "a1": a1, "components": found,
            "tracker_ids": request.tracker_ids,
        }
        return a0

    def snapshot(self):
        return {"mode": "B0_with_same_state_CV_probe", "cv": self.cv.snapshot()}


class InterventionPolicy:
    def __init__(self, baseline, component):
        self.baseline = baseline
        self.component = component
        self.called = 0

    def __call__(self, request, associate):
        self.called += 1
        actual = associate(request.detections, request.trackers, request.iou_threshold,
                           request.velocities, request.previous_observations,
                           request.velocity_direction_weight)
        if not all(np.array_equal(x, y) for x, y in zip(actual, self.baseline["a0"])):
            raise RuntimeError("A0 proposal changed between discovery and intervention")
        if request.tracker_ids != self.baseline["tracker_ids"]:
            raise RuntimeError("tracker IDs changed between discovery and intervention")
        return switched_result(actual[0], self.component,
                               len(request.detections), len(request.trackers))


def output_candidates(output):
    values = np.asarray(output, dtype=np.float64).reshape(-1, 5)
    ids = [int(row[4]) for row in values]
    if len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
        raise RuntimeError("invalid or duplicate output tracker ID")
    return [Candidate(int(row[4]), row[:4]) for row in values]


def anchor_label(gt, reference, branches, frame, horizon):
    anchor_matches, _ = hungarian_iou_matches(gt.get(frame - 1, {}),
                                               output_candidates(reference[frame - 1]), 0.5)
    anchors = {gt_id: track_id for gt_id, (track_id, _) in anchor_matches.items()}
    scores = {name: Counter() for name in branches}
    missing = {name: Counter() for name in branches}
    wrong = {name: Counter() for name in branches}
    no_gt_frames = 0
    unanchored_visible = 0
    anchor_visible = 0
    for offset in range(horizon):
        frame_id = frame + offset
        visible = gt.get(frame_id, {})
        if not visible:
            no_gt_frames += 1
        unanchored_visible += len(set(visible) - set(anchors))
        anchor_visible += len(set(visible) & set(anchors))
        for name, outputs in branches.items():
            matches, _ = hungarian_iou_matches(visible,
                                                output_candidates(outputs[offset]), 0.5)
            for gt_id, original_id in anchors.items():
                if gt_id not in visible:
                    continue
                matched_id = matches.get(gt_id, (None, 0.0))[0]
                if matched_id == original_id:
                    scores[name][gt_id] += 1
                elif matched_id is None:
                    missing[name][gt_id] += 1
                else:
                    wrong[name][gt_id] += 1
    deltas = {str(gt_id): scores["a1"][gt_id] - scores["a0"][gt_id]
              for gt_id in sorted(anchors)}
    return {
        "anchor_count": len(anchors),
        "anchor_gt_to_original_tracker_id": {str(k): v for k, v in sorted(anchors.items())},
        "anchor_visible_frames": anchor_visible,
        "unanchored_gt_visible_frames": unanchored_visible,
        "no_gt_frames": no_gt_frames,
        "correct_frames": {name: sum(values.values()) for name, values in scores.items()},
        "missing_frames": {name: sum(values.values()) for name, values in missing.items()},
        "wrong_id_frames": {name: sum(values.values()) for name, values in wrong.items()},
        "per_anchor_correct_delta": deltas,
        "net_correct_frame_delta": sum(deltas.values()),
        "benefited_anchors": sum(value > 0 for value in deltas.values()),
        "harmed_anchors": sum(value < 0 for value in deltas.values()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--detection-root", type=Path, required=True)
    p.add_argument("--oc-sort-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    provenance = runtime_record(repo_root, ("scripts/run_r1_stage_a.py",) + R1_SHARED_CODE_PATHS)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Stage A is CPU-only; set CUDA_VISIBLE_DEVICES='' explicitly")
    protocol_bytes = args.protocol.read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    split = yaml.safe_load(args.split.read_text())
    if protocol["source_split"] != str(args.split):
        raise RuntimeError("split path differs from frozen protocol")
    ranked = sorted(split["motion_train"]["sequences"], key=lambda s:
                    hashlib.sha256(f"r1-stage-a-20261118:{s}".encode("ascii")).hexdigest())
    selection = protocol["selection"]
    if ranked[:24] != selection["fit_rank_order"] or ranked[24:] != selection["calibration_rank_order"]:
        raise RuntimeError("frozen fit/calibration split mismatch")
    if ranked[:4] != selection["smoke_sequences"]:
        raise RuntimeError("frozen smoke sequence mismatch")
    sys.path.insert(0, str(args.oc_sort_root.resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort
    from run_p1_natural_closed_loop import configparser

    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    failed = args.run_dir.with_name(args.run_dir.name + ".failed")
    if any(path.exists() for path in (args.run_dir, partial, failed)):
        raise FileExistsError("refusing to overwrite Stage A output")
    partial.mkdir(parents=True)
    start = time.monotonic()
    try:
        summaries = []
        hashes = {"protocol": sha256_file(args.protocol), "split": sha256_file(args.split),
                  "inputs": {}}
        horizon = int(protocol["action"]["horizon_frames_including_intervention"])
        max_per_seq = int(selection["max_components_per_sequence"])
        tracker_args = protocol["action"]["tracker"]
        input_h, input_w = protocol["action"]["input_size"]
        for sequence in selection["smoke_sequences"]:
            sequence_dir = args.train_root / sequence
            cache_path = args.detection_root / f"{sequence}.csv"
            gt_path = sequence_dir / "gt" / "gt.txt"
            info_path = sequence_dir / "seqinfo.ini"
            config = configparser.ConfigParser()
            if not config.read(info_path):
                raise FileNotFoundError(info_path)
            info = config["Sequence"]
            length, height, width = info.getint("seqLength"), info.getint("imHeight"), info.getint("imWidth")
            fps = info.getfloat("frameRate")
            cache = load_cache(cache_path, length)
            scale = min(input_h / height, input_w / width)
            hashes["inputs"][sequence] = {str(path): sha256_file(path) for path in
                                           (cache_path, gt_path, info_path)}

            def frame_input(frame_id):
                rows = np.asarray(cache.get(frame_id, []), dtype=np.float64)
                if rows.size == 0:
                    return None
                result = rows.reshape(-1, 5).copy()
                result[:, :4] *= scale
                return result

            def new_runner():
                tracker = ocsort.OCSort(**tracker_args)
                history = TrackerDetectionHistory(
                    history_length=int(protocol["action"]["history_length"]),
                    score_threshold=float(tracker_args["det_thresh"]),
                )
                probe = ProbePolicy(history, width, height, fps)
                return NaturalHistoryTrackerRunner(tracker, ocsort, probe), probe

            discovery, probe = new_runner()
            reference = {}
            candidates = []
            censored = 0
            for frame in range(1, length + 1):
                reference[frame] = discovery.update(frame_input(frame), (height, width),
                                                    (input_h, input_w), frame_id=frame).copy()
                output_candidates(reference[frame])
                record = probe.by_frame.get(frame)
                if record is None or frame < 2:
                    continue
                for part in record["components"]:
                    if frame + horizon - 1 > length:
                        censored += 1
                        continue
                    signature = part.signature(record["tracker_ids"])
                    rank = hashlib.sha256(
                        f"r1-stage-a-20261118:{sequence}:{frame:06d}:{signature}".encode("ascii")
                    ).hexdigest()
                    candidates.append((rank, frame, signature, part))
            candidates.sort(key=lambda value: (value[0], value[1], value[2]))
            selected = candidates[:max_per_seq]
            by_frame = {}
            for rank, frame, signature, part in selected:
                by_frame.setdefault(frame, []).append((rank, signature, part))
            gt = read_gt(gt_path)
            replay, replay_probe = new_runner()
            counts = Counter()
            sequence_start = time.monotonic()
            with (partial / f"{sequence}.jsonl").open("w") as output_handle:
                for frame in range(1, length + 1):
                    for rank, signature, part in by_frame.get(frame, []):
                        pre = copy.deepcopy(replay.tracker)
                        pre_bytes = pickle.dumps(replay.tracker, protocol=5)
                        pre_count = ocsort.KalmanBoxTracker.count
                        baseline = probe.by_frame[frame]
                        branches = {}
                        try:
                            for name in ("a0", "a1"):
                                ocsort.KalmanBoxTracker.count = pre_count
                                fork = copy.deepcopy(pre)
                                outputs = []
                                if name == "a1":
                                    intervention = InterventionPolicy(baseline, part)
                                    adapter = FirstRoundAssociationAdapter(fork, ocsort, intervention)
                                    first = adapter.update(frame_input(frame), (height, width),
                                                           (input_h, input_w), frame_id=frame)
                                    if intervention.called != 1:
                                        raise RuntimeError("intervention call count differs from one")
                                else:
                                    first = fork.update(frame_input(frame), (height, width), (input_h, input_w))
                                outputs.append(np.asarray(first).copy())
                                for future in range(frame + 1, frame + horizon):
                                    outputs.append(np.asarray(fork.update(
                                        frame_input(future), (height, width), (input_h, input_w))).copy())
                                for offset, value in enumerate(outputs):
                                    output_candidates(value)
                                    if name == "a0" and not np.array_equal(value, reference[frame + offset]):
                                        raise RuntimeError(f"A0 replay mismatch at {sequence}:{frame + offset}")
                                branches[name] = outputs
                        finally:
                            ocsort.KalmanBoxTracker.count = pre_count
                        if pickle.dumps(replay.tracker, protocol=5) != pre_bytes:
                            raise RuntimeError("parent tracker state changed during forks")
                        label = anchor_label(gt, reference, branches, frame, horizon)
                        involved_ids = {baseline["tracker_ids"][t]
                                        for _, t in part.a0_edges + part.a1_edges}
                        label["component_outside_anchor_count"] = sum(
                            track_id not in involved_ids
                            for track_id in label["anchor_gt_to_original_tracker_id"].values())
                        result = {"sequence": sequence, "frame": frame, "rank": rank,
                                  "component_signature": signature,
                                  "component_a0_edges": part.a0_edges,
                                  "component_a1_edges": part.a1_edges,
                                  "label": label, "checks": {"a0_exact": True,
                                  "parent_isolated": True, "one_to_one": True}}
                        output_handle.write(json.dumps(result, sort_keys=True) + "\n")
                        output_handle.flush()
                        counts["executed"] += 1
                        counts["positive_net"] += label["net_correct_frame_delta"] > 0
                        counts["negative_net"] += label["net_correct_frame_delta"] < 0
                        counts["zero_net"] += label["net_correct_frame_delta"] == 0
                        counts["with_anchors"] += label["anchor_count"] > 0
                        counts["benefited_anchors"] += label["benefited_anchors"]
                        counts["harmed_anchors"] += label["harmed_anchors"]
                    actual = replay.update(frame_input(frame), (height, width),
                                           (input_h, input_w), frame_id=frame)
                    if not np.array_equal(actual, reference[frame]):
                        raise RuntimeError(f"parent replay mismatch at {sequence}:{frame}")
                    original = probe.by_frame.get(frame)
                    replayed = replay_probe.by_frame.get(frame)
                    if (original is None) != (replayed is None):
                        raise RuntimeError("proposal call changed between passes")
                    if original is not None:
                        if [p.signature(original["tracker_ids"]) for p in original["components"]] != [
                            p.signature(replayed["tracker_ids"]) for p in replayed["components"]]:
                            raise RuntimeError("proposal components changed between passes")
                    if time.monotonic() - start > protocol["budget"]["maximum_wall_seconds"]:
                        raise RuntimeError("frozen wall-time budget exceeded")
            summary = {"sequence": sequence, "frames": length, "candidate_components": len(candidates),
                       "right_censored_components": censored, "selected": len(selected),
                       "counts": dict(counts), "elapsed_seconds": time.monotonic() - sequence_start,
                       "parent_exact_frames": length,
                       "peak_rss_kib": peak_rss_kib()}
            summaries.append(summary)
            (partial / f"{sequence}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
        totals = Counter()
        for row in summaries:
            totals.update(row["counts"])
        acceptance = protocol["acceptance"]
        gates = {
            "executed_minimum": totals["executed"] >= acceptance["minimum_executed_for_feasibility_review"],
            "positive_net_minimum": totals["positive_net"] >= acceptance["minimum_positive_net_for_stage_b_consideration"],
            "negative_net_minimum": totals["negative_net"] >= acceptance["minimum_negative_net_for_stage_b_consideration"],
        }
        verify_code_unchanged(repo_root, provenance)
        final = {"status": "complete", "device": "cpu", "CUDA_VISIBLE_DEVICES": "",
                 "physical_gpu": None, "git_commit": provenance["git_head"],
                 "runtime_provenance": provenance,
                 "hashes": hashes, "sequences": summaries, "totals": dict(totals),
                 "gates": gates, "elapsed_seconds": time.monotonic() - start,
                 "peak_rss_kib": peak_rss_kib()}
        (partial / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
        partial.rename(args.run_dir)
        print(json.dumps({"run_dir": str(args.run_dir), "totals": dict(totals), "gates": gates}), flush=True)
    except BaseException:
        (partial / "error.txt").write_text(traceback.format_exc())
        partial.rename(failed)
        raise


if __name__ == "__main__":
    main()
