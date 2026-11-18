#!/usr/bin/env python3
"""Collect frozen R1-B paired action labels on train-only video splits."""

from __future__ import annotations

import argparse
import configparser
import copy
import hashlib
import json
import math
import os
import pickle
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from mot_research.analysis.failure_events import pairwise_iou, read_gt
from mot_research.analysis.runtime_record import (
    R1_SHARED_CODE_PATHS, peak_rss_kib, runtime_record, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import NaturalHistoryTrackerRunner, TrackerDetectionHistory
from mot_research.tracking.first_round_adapter import FirstRoundAssociationAdapter
from run_p1_natural_closed_loop import load_cache, sha256_file
from run_r1_stage_a import ProbePolicy, InterventionPolicy, anchor_label, output_candidates


FEATURE_NAMES = (
    "component_a0_edges", "component_a1_edges", "component_tracker_count",
    "component_detection_count", "total_tracker_count", "total_detection_count",
    "component_mean_detection_score", "component_min_detection_score",
    "a0_mean_iou", "a1_mean_iou", "iou_difference",
    "mean_history_length", "mean_observation_gap",
)


def component_features(request, history, part):
    involved_t = sorted({t for _, t in part.a0_edges + part.a1_edges})
    involved_d = sorted({d for d, _ in part.a0_edges + part.a1_edges})
    detections = np.asarray(request.detections)
    trackers = np.asarray(request.trackers)
    matrix = pairwise_iou(detections[:, :4], trackers[:, :4])

    def mean_edge_iou(edges):
        return float(np.mean([matrix[d, t] for d, t in edges])) if edges else 0.0

    a0_iou = mean_edge_iou(part.a0_edges)
    a1_iou = mean_edge_iou(part.a1_edges)
    scores = detections[involved_d, 4] if involved_d else np.asarray([], dtype=np.float64)
    histories = [history.observations(request.tracker_ids[t]) for t in involved_t]
    gaps = [request.frame_id - items[-1].frame_id if items else 31 for items in histories]
    values = (
        len(part.a0_edges), len(part.a1_edges), len(involved_t), len(involved_d),
        len(trackers), len(detections),
        float(np.mean(scores)) if len(scores) else 0.0,
        float(np.min(scores)) if len(scores) else 0.0,
        a0_iou, a1_iou, a1_iou - a0_iou,
        float(np.mean([len(items) for items in histories])) if histories else 0.0,
        float(np.mean(gaps)) if gaps else 31.0,
    )
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise RuntimeError("invalid current/past-only feature vector")
    return dict(zip(FEATURE_NAMES, map(float, values)))


class FeatureProbePolicy(ProbePolicy):
    def __init__(self, history, width, height, fps):
        super().__init__(history, width, height, fps)
        self.features_by_frame = {}

    def __call__(self, request, associate):
        baseline = super().__call__(request, associate)
        record = self.by_frame[request.frame_id]
        self.features_by_frame[request.frame_id] = {
            part.signature(request.tracker_ids): component_features(request, self.history, part)
            for part in record["components"]
        }
        return baseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--stage-a-protocol", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--detection-root", type=Path, required=True)
    p.add_argument("--oc-sort-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    provenance = runtime_record(
        repo_root,
        ("scripts/run_r1_stage_b_labels.py", "scripts/run_r1_stage_a.py") + R1_SHARED_CODE_PATHS,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R1-B labels require CUDA_VISIBLE_DEVICES='' and CPU-only execution")
    protocol = yaml.safe_load(args.protocol.read_text())
    stage_a = yaml.safe_load(args.stage_a_protocol.read_text())
    split = yaml.safe_load(args.split.read_text())
    if sha256_file(args.stage_a_protocol) != protocol["stage_a_protocol_sha256"]:
        raise RuntimeError("Stage A protocol hash differs from frozen Stage B")
    if protocol["source_split"] != str(args.split):
        raise RuntimeError("Stage B source split path changed")
    fit = protocol["split"]["fit_sequences"]
    heldout = protocol["split"]["heldout_sequences"]
    if (set(fit) | set(heldout)) != set(split["motion_train"]["sequences"]) or set(fit) & set(heldout):
        raise RuntimeError("fit/heldout membership does not partition motion-train")
    if list(stage_a["selection"]["fit_rank_order"]) != fit or list(stage_a["selection"]["calibration_rank_order"]) != heldout:
        raise RuntimeError("Stage B split differs from Stage A")
    if tuple(protocol["features"]["names"]) != FEATURE_NAMES:
        raise RuntimeError("feature names differ from frozen protocol")
    sys.path.insert(0, str(args.oc_sort_root.resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort

    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    failed = args.run_dir.with_name(args.run_dir.name + ".failed")
    if any(path.exists() for path in (args.run_dir, partial, failed)):
        raise FileExistsError("refusing to overwrite Stage B label output")
    partial.mkdir(parents=True)
    start = time.monotonic()
    try:
        horizon = int(protocol["sampling"]["horizon_including_intervention"])
        if horizon != int(stage_a["action"]["horizon_frames_including_intervention"]):
            raise RuntimeError("Stage B horizon differs from Stage A")
        cap = int(protocol["sampling"]["per_sequence_cap"])
        tracker_args = stage_a["action"]["tracker"]
        input_h, input_w = stage_a["action"]["input_size"]
        all_summaries = []
        input_hashes = {}
        for sequence in fit + heldout:
            sequence_start = time.monotonic()
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
            input_hashes[sequence] = {str(path): sha256_file(path) for path in (cache_path, gt_path, info_path)}

            def frame_input(frame):
                rows = np.asarray(cache.get(frame, []), dtype=np.float64)
                if rows.size == 0:
                    return None
                result = rows.reshape(-1, 5).copy()
                result[:, :4] *= scale
                return result

            def new_runner():
                tracker = ocsort.OCSort(**tracker_args)
                history = TrackerDetectionHistory(
                    history_length=int(stage_a["action"]["history_length"]),
                    score_threshold=float(tracker_args["det_thresh"]),
                )
                probe = FeatureProbePolicy(history, width, height, fps)
                return NaturalHistoryTrackerRunner(tracker, ocsort, probe), probe

            discovery, probe = new_runner()
            reference = {}
            candidates = []
            tail_excluded = 0
            raw_components = 0
            for frame in range(1, length + 1):
                reference[frame] = discovery.update(frame_input(frame), (height, width),
                                                    (input_h, input_w), frame_id=frame).copy()
                output_candidates(reference[frame])
                proposal = probe.by_frame.get(frame)
                if proposal is None or frame < 2:
                    continue
                frame_candidates = []
                for part in proposal["components"]:
                    raw_components += 1
                    if frame + horizon - 1 > length:
                        tail_excluded += 1
                        continue
                    signature = part.signature(proposal["tracker_ids"])
                    rank = hashlib.sha256(
                        f"r1-stage-b-20261118:{sequence}:{frame:06d}:{signature}".encode("ascii")
                    ).hexdigest()
                    frame_candidates.append((rank, frame, signature, part))
                if frame_candidates:
                    candidates.append(min(frame_candidates, key=lambda item: (item[0], item[2])))
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            selected = candidates[:cap]
            selected_by_frame = {item[1]: item for item in selected}
            if len(selected_by_frame) != len(selected):
                raise RuntimeError("more than one selected action per frame")
            gt = read_gt(gt_path)
            replay, replay_probe = new_runner()
            counts = Counter()
            archive_ranks = {item[0] for item in selected[:5]}
            with (partial / f"{sequence}.jsonl").open("w") as labels_handle, \
                 (partial / f"{sequence}.audit.jsonl").open("w") as audit_handle:
                for frame in range(1, length + 1):
                    item = selected_by_frame.get(frame)
                    if item is not None:
                        rank, _, signature, part = item
                        before = pickle.dumps(replay.tracker, protocol=5)
                        pre = copy.deepcopy(replay.tracker)
                        pre_count = ocsort.KalmanBoxTracker.count
                        baseline = probe.by_frame[frame]
                        branches = {}
                        try:
                            for name in ("a0", "a1"):
                                ocsort.KalmanBoxTracker.count = pre_count
                                fork = copy.deepcopy(pre)
                                if name == "a1":
                                    policy = InterventionPolicy(baseline, part)
                                    adapter = FirstRoundAssociationAdapter(fork, ocsort, policy)
                                    first = adapter.update(frame_input(frame), (height, width),
                                                           (input_h, input_w), frame_id=frame)
                                    if policy.called != 1:
                                        raise RuntimeError("intervention not called exactly once")
                                else:
                                    first = fork.update(frame_input(frame), (height, width), (input_h, input_w))
                                outputs = [np.asarray(first).copy()]
                                for future in range(frame + 1, frame + horizon):
                                    outputs.append(np.asarray(fork.update(
                                        frame_input(future), (height, width), (input_h, input_w))).copy())
                                for offset, value in enumerate(outputs):
                                    output_candidates(value)
                                    if name == "a0" and not np.array_equal(value, reference[frame + offset]):
                                        raise RuntimeError(f"A0 branch mismatch at {sequence}:{frame + offset}")
                                branches[name] = outputs
                        finally:
                            ocsort.KalmanBoxTracker.count = pre_count
                        if pickle.dumps(replay.tracker, protocol=5) != before:
                            raise RuntimeError("parent state changed during paired branches")
                        future_label = anchor_label(gt, reference, branches, frame, horizon)
                        immediate_label = anchor_label(gt, reference,
                                                       {name: values[:1] for name, values in branches.items()},
                                                       frame, 1)
                        involved_ids = {baseline["tracker_ids"][t] for _, t in part.a0_edges + part.a1_edges}
                        anchored_ids = future_label["anchor_gt_to_original_tracker_id"].values()
                        direct_anchor_count = sum(track_id in involved_ids for track_id in anchored_ids)
                        row = {
                            "sequence": sequence, "group": "fit" if sequence in fit else "heldout",
                            "frame": frame, "rank": rank, "component_signature": signature,
                            "features": probe.features_by_frame[frame][signature],
                            "direct_anchor_count": direct_anchor_count,
                            "involved_tracker_ids": sorted(involved_ids),
                            "future_label": future_label,
                            "immediate_label": immediate_label,
                            "checks": {"a0_exact": True, "parent_isolated": True,
                                       "one_to_one": True, "one_component_per_frame": True},
                        }
                        labels_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        labels_handle.flush()
                        if rank in archive_ranks:
                            audit_handle.write(json.dumps({
                                "sequence": sequence, "frame": frame, "rank": rank,
                                "branches": {name: [value.tolist() for value in values]
                                             for name, values in branches.items()},
                                "reference_previous_output": reference[frame - 1].tolist(),
                            }, sort_keys=True) + "\n")
                            audit_handle.flush()
                        counts["executed"] += 1
                        counts["with_direct_anchor"] += direct_anchor_count > 0
                        counts["without_direct_anchor"] += direct_anchor_count == 0
                        net = future_label["net_correct_frame_delta"]
                        counts["future_positive"] += net > 0
                        counts["future_negative"] += net < 0
                        counts["future_zero"] += net == 0
                    actual = replay.update(frame_input(frame), (height, width),
                                           (input_h, input_w), frame_id=frame)
                    if not np.array_equal(actual, reference[frame]):
                        raise RuntimeError(f"parent baseline mismatch at {sequence}:{frame}")
                    if frame in selected_by_frame:
                        original = probe.by_frame[frame]
                        replayed = replay_probe.by_frame[frame]
                        if [p.signature(original["tracker_ids"]) for p in original["components"]] != [
                            p.signature(replayed["tracker_ids"]) for p in replayed["components"]]:
                            raise RuntimeError("proposal components changed between passes")
                    if time.monotonic() - start > protocol["budget"]["maximum_wall_seconds"]:
                        raise RuntimeError("Stage B wall budget exceeded")
            summary = {"sequence": sequence, "group": "fit" if sequence in fit else "heldout",
                       "frames": length, "raw_components": raw_components,
                       "tail_excluded": tail_excluded, "candidate_frames": len(candidates),
                       "selected": len(selected), "counts": dict(counts),
                       "parent_exact_frames": length,
                       "elapsed_seconds": time.monotonic() - sequence_start,
                       "peak_rss_kib": peak_rss_kib()}
            all_summaries.append(summary)
            (partial / f"{sequence}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            if sum(row["selected"] for row in all_summaries) > protocol["sampling"]["global_cap"]:
                raise RuntimeError("Stage B global sample cap exceeded")
        totals = Counter()
        for row in all_summaries:
            totals.update(row["counts"])
        verify_code_unchanged(repo_root, provenance)
        final = {"status": "complete", "device": "cpu", "CUDA_VISIBLE_DEVICES": "",
                 "physical_gpu": None, "protocol_sha256": sha256_file(args.protocol),
                 "runtime_provenance": provenance,
                 "stage_a_protocol_sha256": sha256_file(args.stage_a_protocol),
                 "split_sha256": sha256_file(args.split), "input_hashes": input_hashes,
                 "sequence_summaries": all_summaries, "totals": dict(totals),
                 "elapsed_seconds": time.monotonic() - start,
                 "peak_rss_kib": peak_rss_kib()}
        (partial / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
        partial.rename(args.run_dir)
        print(json.dumps({"run_dir": str(args.run_dir), "totals": dict(totals)}), flush=True)
    except BaseException:
        (partial / "error.txt").write_text(traceback.format_exc())
        partial.rename(failed)
        raise


if __name__ == "__main__":
    main()
