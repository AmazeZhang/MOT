#!/usr/bin/env python3
"""Paired non-GT appearance versus geometry association forks on train videos."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from analyze_p1_common_state_mechanism import read_detection_cache
from mot_research.analysis.failure_events import (
    hungarian_iou_matches, pairwise_iou, read_detections, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, NaturalMotionFirstRoundPolicy, TrackerDetectionHistory,
)
from run_e4_oracle_association_headroom import (
    ForcePairPolicy, fork_runner, output_candidates, same_outputs, step_e,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e5_nonoracle_appearance_local_20261119.yaml"
CODE_PATHS = (
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


class CapturePolicy:
    def __init__(self, base):
        self.base = base
        self.called = 0
        self.request = None
        self.pairs = None

    def __call__(self, request, associate):
        self.called += 1
        self.request = request
        result = self.base(request, associate)
        self.pairs = [(int(d), int(t)) for d, t in result[0]]
        return result


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("fit", "heldout"), required=True)
    return parser.parse_args()


def candidates_from_online_inputs(sequence, c_ref, cached, length, cap, distance_limit):
    last = {}
    used = set()
    events = []
    for frame in range(1, length + 1):
        high = np.asarray([v for v in cached.get(frame, []) if v[4] > .6], dtype=np.float64).reshape(-1, 5)
        for track_id, (old_frame, old_box) in last.items():
            gap = frame - old_frame
            if track_id in used or not 2 <= gap <= 30 or frame + 4 > length or len(high) < 2:
                continue
            old_center = (old_box[:2] + old_box[2:]) / 2
            det_centers = (high[:, :2] + high[:, 2:4]) / 2
            old_diag = max(float(np.linalg.norm(old_box[2:] - old_box[:2])), 1.0)
            distances = np.linalg.norm(det_centers - old_center, axis=1) / old_diag
            if int(np.sum(distances <= distance_limit)) < 2:
                continue
            used.add(track_id)
            rank = hashlib.sha256(f"e5:{sequence}:{frame:06d}:{track_id}".encode("ascii")).hexdigest()
            events.append({"sequence": sequence, "frame": frame, "track_id": int(track_id),
                           "anchor_frame": old_frame, "old_box": old_box.tolist(),
                           "rank": rank})
        for row in c_ref.get(frame, []):
            last[row.candidate_id] = (frame, row.box_xyxy.copy())
            used.discard(row.candidate_id)
    total = len(events)
    events.sort(key=lambda item: (item["rank"], item["frame"], item["track_id"]))
    return events[:cap], total


def crop_hist(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = .35 * (x2 - x1), .35 * (y2 - y1)
    left, top = max(0, int(np.floor(cx - hw))), max(0, int(np.floor(cy - hh)))
    right, bottom = min(width, int(np.ceil(cx + hw))), min(height, int(np.ceil(cy + hh)))
    if right <= left or bottom <= top:
        return None
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    total = float(hist.sum())
    return hist / total if total > 0 else None


def image_read(path, cache):
    if path not in cache:
        value = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if value is None:
            raise FileNotFoundError(path)
        cache[path] = value
        if len(cache) > 35:
            cache.pop(next(iter(cache)))
    return cache[path]


def choose_actions(event, capture, mature, seq_dir, images, protocol):
    request = capture.request
    track_id = event["track_id"]
    if track_id not in mature or request is None or track_id not in request.tracker_ids:
        return None, "not_mature_or_live"
    track_index = request.tracker_ids.index(track_id)
    matched_det = next((d for d, t in capture.pairs if t == track_index), None)
    dets = np.asarray(request.detections, dtype=np.float64).reshape(-1, 5)
    if len(dets) < 2:
        return None, "too_few_high_detections"
    old_box = np.asarray(event["old_box"], dtype=np.float64)
    old_center = (old_box[:2] + old_box[2:]) / 2
    centers = (dets[:, :2] + dets[:, 2:4]) / 2
    diag = max(float(np.linalg.norm(old_box[2:] - old_box[:2])), 1.0)
    dist = np.linalg.norm(centers - old_center, axis=1) / diag
    options = [i for i in range(len(dets)) if dist[i] <= protocol["scope"]["online_candidate_definition"]["center_distance_at_most_old_box_diagonals"]
               and i != matched_det]
    if not options:
        return None, "no_alternative_detection"
    old_path = seq_dir / "img1" / f"{event['anchor_frame']:08d}.jpg"
    now_path = seq_dir / "img1" / f"{event['frame']:08d}.jpg"
    old_hist = crop_hist(image_read(old_path, images), old_box)
    if old_hist is None:
        return None, "invalid_old_crop"
    now_image = image_read(now_path, images)
    old_iou = pairwise_iou(old_box[None, :], dets[options, :4])[0]
    predicted = np.asarray(request.trackers[track_index, :4], dtype=np.float64)
    pred_iou = pairwise_iou(predicted[None, :], dets[options, :4])[0]
    rows = []
    for pos, det_index in enumerate(options):
        current_hist = crop_hist(now_image, dets[det_index, :4])
        if current_hist is None:
            continue
        similarity = float(np.sqrt(old_hist * current_hist).sum())
        geo_iou = float(max(old_iou[pos], pred_iou[pos]))
        score = float(dets[det_index, 4])
        rows.append({"det_index": det_index, "similarity": similarity,
                     "geometry_iou": geo_iou, "detection_score": score,
                     "geometry_score": geo_iou + .05 * score,
                     "appearance_score": similarity + .1 * geo_iou})
    if not rows:
        return None, "invalid_current_crops"
    geometry = max(rows, key=lambda x: (x["geometry_score"], -x["det_index"]))
    appearance_rank = sorted(rows, key=lambda x: (-x["appearance_score"], x["det_index"]))
    appearance = appearance_rank[0]
    margin = (appearance["appearance_score"] - appearance_rank[1]["appearance_score"]
              if len(appearance_rank) > 1 else 1.0)
    rule = protocol["methods"]["appearance"]
    if (appearance["similarity"] < rule["minimum_best_similarity"] or
            margin < rule["minimum_best_minus_second_appearance_score"]):
        return {"rows": rows, "geometry": geometry, "appearance": appearance,
                "margin": margin}, "appearance_threshold_failed"
    return {"rows": rows, "geometry": geometry, "appearance": appearance,
            "margin": margin, "old_image_sha256": sha256_file(old_path),
            "current_image_sha256": sha256_file(now_path)}, "action"


def encode(outputs):
    return [[{"id": row.candidate_id, "box": row.box_xyxy.tolist()} for row in frame]
            for frame in outputs]


def decode(outputs):
    from mot_research.analysis.failure_events import Candidate
    return [[Candidate(int(row["id"]), np.asarray(row["box"], dtype=np.float64)) for row in frame]
            for frame in outputs]


def make_runner(ocsort, inp, height, width, fps):
    tracker = ocsort.OCSort(**inp["tracker"])
    history = TrackerDetectionHistory(history_length=inp["cv_history"]["length"], score_threshold=.6)
    policy = NaturalMotionFirstRoundPolicy(
        mode="constant_velocity", history=history, image_width=width, image_height=height,
        fps=fps, minimum_observations=inp["cv_history"]["minimum_observations"],
        maximum_frame_gap=inp["cv_history"]["maximum_frame_gap"])
    return NaturalHistoryTrackerRunner(tracker, ocsort, policy)


def sequence_inputs(seq, inp):
    seq_dir = Path(inp["train_root"]) / seq
    info_path = seq_dir / "seqinfo.ini"
    config = configparser.ConfigParser()
    if not config.read(info_path):
        raise FileNotFoundError(info_path)
    info = config["Sequence"]
    length, height, width = info.getint("seqLength"), info.getint("imHeight"), info.getint("imWidth")
    fps = info.getfloat("frameRate")
    cache_path = Path(inp["detection_root"]) / f"{seq}.csv"
    cached, det_ids = read_detection_cache(cache_path)
    scale = min(inp["input_size"][0] / height, inp["input_size"][1] / width)
    def frame_input(frame):
        values = np.asarray(cached.get(frame, []), dtype=np.float64)
        if values.size == 0:
            return None
        result = values.reshape(-1, 5).copy()
        result[:, :4] *= scale
        return result
    return seq_dir, length, height, width, fps, cache_path, cached, det_ids, frame_input


def baseline_replay(seq, ocsort, inp, length, height, width, fps, frame_input):
    runner = make_runner(ocsort, inp, height, width, fps)
    mature = set()
    c_ref, e_ref = {}, {}
    for frame in range(1, length + 1):
        c_ref[frame], e_ref[frame] = step_e(runner, mature, frame_input(frame), frame, height, width, inp)
    return c_ref, e_ref


def score_events(events, gt, c_ref, e_ref, high_detections):
    totals = Counter()
    scored = []
    for event in events:
        frame = event["frame"]
        anchor_matches = hungarian_iou_matches(gt.get(event["anchor_frame"], {}),
                                                c_ref.get(event["anchor_frame"], []), .5)[0]
        anchors = {gid: tid for gid, (tid, _) in anchor_matches.items()}
        target_gt = next((gid for gid, tid in anchors.items() if tid == event["track_id"]), None)
        baseline = [e_ref.get(f, []) for f in range(frame, frame + 5)]
        outputs = {"baseline": baseline, "geometry": decode(event["geometry_outputs"]),
                   "appearance": decode(event["appearance_outputs"])}
        matches = {name: [hungarian_iou_matches(gt.get(frame + offset, {}), rows, .5)[0]
                          for offset, rows in enumerate(branch)] for name, branch in outputs.items()}
        scores = {}
        for name in outputs:
            per_anchor = {gid: sum(gid in gt.get(frame + k, {}) and
                                   matches[name][k].get(gid, (None,))[0] == tid
                                   for k in range(5)) for gid, tid in anchors.items()}
            scores[name] = per_anchor
        method_scores = {}
        for name in ("geometry", "appearance"):
            deltas = {gid: scores[name][gid] - scores["baseline"][gid] for gid in anchors}
            method_scores[name] = {
                "all_anchor_net_delta": sum(deltas.values()),
                "target_delta": deltas.get(target_gt) if target_gt is not None else None,
                "other_anchor_harmed": any(value < 0 for gid, value in deltas.items() if gid != target_gt),
                "all_anchor_harmed": any(value < 0 for value in deltas.values()),
            }
            totals[f"{name}_all_anchor_net"] += method_scores[name]["all_anchor_net_delta"]
            totals[f"{name}_other_harm_events"] += method_scores[name]["other_anchor_harmed"]
            totals[f"{name}_all_harm_events"] += method_scores[name]["all_anchor_harmed"]
            if target_gt is not None:
                totals[f"{name}_target_net"] += method_scores[name]["target_delta"]
                totals[f"{name}_target_harm_events"] += method_scores[name]["target_delta"] < 0
        positive = hungarian_iou_matches(gt.get(frame, {}), high_detections.get(frame, []), .5)[0]
        positive_det = positive.get(target_gt, (None,))[0] if target_gt is not None else None
        selected_ids = event["selected_detection_ids"]
        correctness = {name: (selected_ids[name] == positive_det if positive_det is not None else None)
                       for name in ("geometry", "appearance")}
        for name in ("geometry", "appearance"):
            totals[f"{name}_correct_actions"] += correctness[name] is True
            totals[f"{name}_wrong_actions"] += correctness[name] is False
        totals["actions"] += 1
        totals["disagreeing_actions"] += selected_ids["geometry"] != selected_ids["appearance"]
        totals["target_anchored"] += target_gt is not None
        totals["target_positive_detection_exists"] += positive_det is not None
        scored.append({"sequence": event["sequence"], "frame": frame,
                       "track_id": event["track_id"], "anchor_frame": event["anchor_frame"],
                       "target_gt_id": target_gt, "positive_detection_id": positive_det,
                       "selected_detection_ids": selected_ids, "correctness": correctness,
                       "score": method_scores, "anchor_count": len(anchors),
                       "action_features": event["action_features"]})
    totals["appearance_minus_geometry_all_anchor_net"] = (
        totals["appearance_all_anchor_net"] - totals["geometry_all_anchor_net"])
    return scored, totals


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E5 requires CUDA_VISIBLE_DEVICES='' and CPU only")
    partition = args().partition
    protocol = yaml.safe_load(PROTOCOL.read_text())
    source = protocol["source"]
    for path, expected in ((ROOT / source["split_manifest"], source["split_sha256"]),
                           (Path(source["e3_fit_summary"]), source["e3_fit_summary_sha256"]),
                           (Path(source["e4_summary"]), source["e4_summary_sha256"])):
        if sha256_file(path) != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
    split = yaml.safe_load((ROOT / source["split_manifest"]).read_text())["split"]
    fit, heldout = split["fit_sequences"], split["heldout_sequences"]
    if len(fit) != 24 or len(heldout) != 8 or set(fit) & set(heldout):
        raise RuntimeError("unexpected split")
    e3 = json.loads(Path(source["e3_fit_summary"]).read_text())
    if [x["sequence"] for x in e3["sequences"]] != fit:
        raise RuntimeError("E3 fit sequence order changed")
    if partition == "heldout":
        fit_summary_path = Path(protocol["outputs"]["fit_dir"]) / "summary.json"
        fit_summary = json.loads(fit_summary_path.read_text())
        if (fit_summary["status"] != "PASS" or not fit_summary["fit_gate_pass"] or
                fit_summary["protocol_sha256"] != sha256_file(PROTOCOL)):
            raise RuntimeError("E5 fit gate did not authorize heldout")
    inp = protocol["inputs"]
    commit = subprocess.run(["git", "-C", str(ROOT / inp["oc_sort_root"]), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != inp["oc_sort_commit"]:
        raise RuntimeError("OC-SORT commit changed")
    output = Path(protocol["outputs"][f"{partition}_dir"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(path.exists() for path in (output, partial, failed)):
        raise FileExistsError(f"refusing overwrite: {output}")
    partial.mkdir(parents=True)
    start = time.monotonic()
    cpu_start = time.process_time()
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL), "partition": partition,
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat(),
                           "oc_sort_commit": commit, "opencv_version": cv2.__version__})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        sys.path.insert(0, str((ROOT / inp["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort
        summaries = []
        for seq in (fit if partition == "fit" else heldout):
            seq_dir, length, height, width, fps, cache_path, cached, det_ids, frame_input = sequence_inputs(seq, inp)
            if partition == "fit":
                e3_row = next(x for x in e3["sequences"] if x["sequence"] == seq)
                if sha256_file(cache_path) != e3_row["cache_sha256"] or \
                        sha256_file(seq_dir / "seqinfo.ini") != e3_row["seqinfo_sha256"]:
                    raise RuntimeError(f"{seq}: E3 input hash changed")
                c_path = Path(source["e3_fit_summary"]).parent / "trackers/C" / f"{seq}.txt"
                e_path = Path(source["e3_fit_summary"]).parent / "trackers/E" / f"{seq}.txt"
                if sha256_file(c_path) != e3_row["tracker_sha256"]["C"] or \
                        sha256_file(e_path) != e3_row["tracker_sha256"]["E"]:
                    raise RuntimeError(f"{seq}: E3 baseline changed")
                c_ref, e_ref = read_tracker(c_path), read_tracker(e_path)
            else:
                baseline_counter = ocsort.KalmanBoxTracker.count
                c_ref, e_ref = baseline_replay(seq, ocsort, inp, length, height, width, fps, frame_input)
                ocsort.KalmanBoxTracker.count = baseline_counter
            selected, online_count = candidates_from_online_inputs(
                seq, c_ref, cached, length,
                protocol["scope"]["sample_before_GT_and_fork_eligibility"]["cap_per_video"],
                protocol["scope"]["online_candidate_definition"]["center_distance_at_most_old_box_diagonals"])
            by_frame = defaultdict(list)
            for event in selected:
                by_frame[event["frame"]].append(event)
            parent = make_runner(ocsort, inp, height, width, fps)
            mature = set()
            images = {}
            count = Counter()
            raw_events = []
            for frame in range(1, length + 1):
                events = by_frame.get(frame, [])
                frame_values = frame_input(frame)
                pre_runner = fork_runner(parent, ocsort) if events else None
                pre_mature = set(mature) if events else None
                pre_counter = ocsort.KalmanBoxTracker.count
                capture = CapturePolicy(parent.policy) if events else None
                if capture is not None:
                    parent.adapter.policy = capture
                c_now, e_now = step_e(parent, mature, frame_values, frame, height, width, inp)
                if capture is not None:
                    parent.adapter.policy = parent.policy
                parent_counter = ocsort.KalmanBoxTracker.count
                if not same_outputs(c_now, c_ref.get(frame, [])) or \
                        not same_outputs(e_now, e_ref.get(frame, [])):
                    raise RuntimeError(f"{seq}:{frame}: parent C/E replay mismatch")
                for event in events:
                    count["selected"] += 1
                    if capture.called != 1:
                        count["no_first_round_call"] += 1
                        continue
                    choice, reason = choose_actions(event, capture, pre_mature, seq_dir, images, protocol)
                    if reason != "action":
                        count[reason] += 1
                        continue
                    high_ids = [int(did) for did, value in zip(det_ids.get(frame, []), cached.get(frame, []))
                                if float(value[4]) > .6]
                    if len(high_ids) != len(capture.request.detections):
                        raise RuntimeError("detection index mismatch")
                    branch_outputs = {}
                    for name in ("geometry", "appearance"):
                        pick = int(choice[name]["det_index"])
                        ocsort.KalmanBoxTracker.count = pre_counter
                        branch = fork_runner(pre_runner, ocsort)
                        branch_mature = set(pre_mature)
                        forced = ForcePairPolicy(branch.policy, event["track_id"], pick)
                        branch.adapter.policy = forced
                        _, first = step_e(branch, branch_mature, frame_values, frame, height, width, inp)
                        branch.adapter.policy = branch.policy
                        if forced.called != 1 or forced.forced is None:
                            raise RuntimeError(f"{seq}:{frame}: forced action missing")
                        outputs = [first]
                        for future in range(frame + 1, frame + 5):
                            _, rows = step_e(branch, branch_mature, frame_input(future), future, height, width, inp)
                            outputs.append(rows)
                        branch_outputs[name] = encode(outputs)
                    ocsort.KalmanBoxTracker.count = parent_counter
                    raw_events.append({**event, "action_features": choice,
                                       "selected_detection_ids": {
                                           name: high_ids[int(choice[name]["det_index"])]
                                           for name in ("geometry", "appearance")},
                                       "geometry_outputs": branch_outputs["geometry"],
                                       "appearance_outputs": branch_outputs["appearance"]})
                    count["actions"] += 1
                if time.monotonic() - start > protocol["budget"]["max_wall_seconds_per_partition"]:
                    raise TimeoutError("E5 partition wall budget exceeded")
            raw_path = partial / f"{seq}.raw.jsonl"
            with raw_path.open("w") as handle:
                for event in raw_events:
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")
            # Ground truth is opened only after every action and branch in this video is fixed.
            gt_path = seq_dir / "gt/gt.txt"
            gt = read_gt(gt_path)
            high = read_detections(cache_path, .6)
            scored, totals = score_events(raw_events, gt, c_ref, e_ref, high)
            scored_path = partial / f"{seq}.scored.jsonl"
            with scored_path.open("w") as handle:
                for row in scored:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            summary = {"sequence": seq, "frames": length,
                       "all_online_candidate_episodes": online_count,
                       "selected_episodes": len(selected), "counts": dict(count),
                       "scores": dict(totals), "cache_sha256": sha256_file(cache_path),
                       "seqinfo_sha256": sha256_file(seq_dir / "seqinfo.ini"),
                       "GT_sha256": sha256_file(gt_path),
                       "raw_sha256": sha256_file(raw_path),
                       "scored_sha256": sha256_file(scored_path)}
            summaries.append(summary)
            (partial / "sequence_progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"{seq}: candidates={online_count} selected={len(selected)} actions={totals['actions']} disagree={totals['disagreeing_actions']} app_minus_geo={totals['appearance_minus_geometry_all_anchor_net']}", flush=True)
        verify_code_unchanged(ROOT, provenance)
        total_counts, total_scores = Counter(), Counter()
        for row in summaries:
            total_counts.update(row["counts"])
            total_scores.update(row["scores"])
            total_counts["all_online_candidate_episodes"] += row["all_online_candidate_episodes"]
            total_counts["selected_episodes"] += row["selected_episodes"]
        actions = total_scores["actions"]
        if partition == "fit":
            gate = protocol["decision"]["fit_gate_all_required"]
            passed = (actions >= gate["minimum_paired_actions"] and
                      total_scores["disagreeing_actions"] >= gate["minimum_disagreeing_actions"] and
                      total_scores["appearance_minus_geometry_all_anchor_net"] >=
                      gate["minimum_appearance_minus_geometry_all_anchor_correct_frame_gain_per_action"] * actions and
                      total_scores["appearance_other_harm_events"] <= total_scores["geometry_other_harm_events"])
        else:
            gate = protocol["decision"]["heldout_success_all_required"]
            passed = (actions >= gate["minimum_paired_actions"] and
                      total_scores["appearance_minus_geometry_all_anchor_net"] > 0 and
                      total_scores["appearance_other_harm_events"] <= total_scores["geometry_other_harm_events"])
        result = {"status": "PASS", "partition": partition,
                  "protocol_sha256": provenance["protocol_sha256"],
                  "sequences": summaries, "counts": dict(total_counts), "scores": dict(total_scores),
                  "fit_gate_pass" if partition == "fit" else "heldout_success": passed,
                  "wall_seconds": time.monotonic() - start,
                  "cpu_core_hours": (time.process_time() - cpu_start) / 3600,
                  "peak_rss_kib": peak_rss_kib(),
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"partition": partition, "status": "PASS", "gate": passed,
                          "counts": dict(total_counts), "scores": dict(total_scores),
                          "output": str(output)}, indent=2), flush=True)
    except BaseException as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
