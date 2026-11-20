#!/usr/bin/env python3
"""Join the read-only E5 pair trace to the already scored E5 error breakdown."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mot_research.analysis.failure_events import hungarian_iou_matches, read_detections, read_gt, read_tracker
from mot_research.analysis.runtime_record import sha256_file

ROOT = Path(__file__).resolve().parents[1]
E5 = Path("/media/imc/data/mot-data/runs/e5_nonoracle_appearance_local_fit_20261119")
ERROR = Path("/media/imc/data/mot-data/runs/e5_readonly_error_analysis_20261120")
TRACE = Path("/media/imc/data/mot-data/runs/e5_baseline_pair_trace_addendum_20261120")
DATA = Path("/media/imc/data/mot-data/datasets/dancetrack/train")
DETECTIONS = Path("/media/imc/data/mot-data/cache/detections/ocsort_yolox_dance_bb2f686_b8d1/train")
C_ROOT = Path("/media/imc/data/mot-data/runs/e3_short_gap_support_fit_20261119/trackers/C")


def event_key(row):
    return row["sequence"], int(row["frame"]), int(row["track_id"]), int(row["anchor_frame"])


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("join requires CUDA_VISIBLE_DEVICES='' and CPU only")
    e5_summary_path = E5 / "summary.json"
    error_summary_path = ERROR / "summary.json"
    trace_summary_path = TRACE / "summary.json"
    e5 = json.loads(e5_summary_path.read_text())
    error = json.loads(error_summary_path.read_text())
    trace = json.loads(trace_summary_path.read_text())
    if len(e5["sequences"]) != 24 or len(trace["sequences"]) != 24:
        raise RuntimeError("unexpected sequence coverage")
    if trace["total_events"] != error["totals"]["actions"] != 576:
        raise RuntimeError("event denominator mismatch")
    output_path = TRACE / "joined_summary.json"
    rows_path = TRACE / "joined_events.jsonl"
    if output_path.exists() or rows_path.exists():
        raise FileExistsError("refusing to overwrite joined pair trace")
    totals = Counter()
    joined = []
    per_video = []
    for e5_seq, trace_seq in zip(e5["sequences"], trace["sequences"]):
        seq = e5_seq["sequence"]
        if seq != trace_seq["sequence"]:
            raise RuntimeError("sequence order mismatch")
        pair_path = TRACE / f"{seq}.pairs.jsonl"
        score_path = E5 / f"{seq}.scored.jsonl"
        if sha256_file(pair_path) != trace_seq["pairs_sha256"] or \
                sha256_file(score_path) != e5_seq["scored_sha256"]:
            raise RuntimeError("source file hash changed")
        if sha256_file(DATA / seq / "gt/gt.txt") != e5_seq["GT_sha256"] or \
                sha256_file(DETECTIONS / f"{seq}.csv") != e5_seq["cache_sha256"] or \
                sha256_file(C_ROOT / f"{seq}.txt") != trace_seq["C_reference_sha256"]:
            raise RuntimeError("dataset or C reference hash changed")
        pairs = {event_key(row): row for row in map(json.loads, pair_path.read_text().splitlines())}
        scores = {event_key(row): row for row in map(json.loads, score_path.read_text().splitlines())}
        if pairs.keys() != scores.keys() or len(pairs) != e5_seq["scores"]["actions"]:
            raise RuntimeError(f"{seq}: pair/score event key mismatch")
        gt = read_gt(DATA / seq / "gt/gt.txt")
        high = read_detections(DETECTIONS / f"{seq}.csv", .6)
        c_ref = read_tracker(C_ROOT / f"{seq}.txt")
        local = Counter()
        for key in sorted(pairs):
            pair, score = pairs[key], scores[key]
            frame, target = score["frame"], score["track_id"]
            positive = score["positive_detection_id"]
            baseline_target_det = pair["baseline_target_first_round_detection_id"]
            first_pairs = {row["detection_id"]: row["track_id"]
                           for row in pair["normal_first_round_pairs"]}
            if len(first_pairs) != len(pair["normal_first_round_pairs"]):
                raise RuntimeError("duplicate detection in baseline pair trace")
            anchors = hungarian_iou_matches(gt.get(score["anchor_frame"], {}),
                                             c_ref.get(score["anchor_frame"], []), .5)[0]
            anchor_tracks = {gid: tid for gid, (tid, _) in anchors.items()}
            positives = hungarian_iou_matches(gt.get(frame, {}), high.get(frame, []), .5)[0]
            det_to_gt = {det: gid for gid, (det, _) in positives.items()}
            if score["target_gt_id"] is not None:
                if anchor_tracks.get(score["target_gt_id"]) != target or \
                        positives.get(score["target_gt_id"], (None,))[0] != positive:
                    raise RuntimeError("target GT anchor/positive changed")
            same = score["selected_detection_ids"]["geometry"] == \
                score["selected_detection_ids"]["appearance"]
            harm_geo = bool(score["score"]["geometry"]["other_anchor_harmed"])
            harm_app = bool(score["score"]["appearance"]["other_anchor_harmed"])
            group = ("both" if harm_geo and harm_app else
                     "geometry_only" if harm_geo else "appearance_only" if harm_app else "neither")
            if positive is None:
                positive_pair = "no_target_positive_detection"
            elif baseline_target_det == positive:
                positive_pair = "normal_first_round_already_paired_target_positive"
            elif positive in first_pairs:
                positive_pair = "normal_first_round_positive_paired_other_track"
            else:
                positive_pair = "normal_first_round_positive_unpaired"
            if baseline_target_det is None:
                target_pair = "target_unpaired"
            elif baseline_target_det == positive and positive is not None:
                target_pair = "target_paired_positive"
            else:
                target_pair = "target_paired_other_detection"
            selected = {}
            for method in ("geometry", "appearance"):
                det_id = score["selected_detection_ids"][method]
                recorded = pair["selected"][method]
                if det_id != recorded["detection_id"] or \
                        recorded["baseline_first_round_owner_track_id"] != first_pairs.get(det_id):
                    raise RuntimeError("selected detection ownership mismatch")
                owner_track = first_pairs.get(det_id)
                if owner_track is None:
                    owner_type = "unpaired_detection"
                elif owner_track == target:
                    raise RuntimeError("E5 selected a detection already paired to target")
                else:
                    owner_type = "paired_to_other_track"
                owner_gt = det_to_gt.get(det_id)
                rightful = (owner_gt is not None and owner_gt in anchor_tracks and
                            owner_track == anchor_tracks[owner_gt])
                selected[method] = {"detection_id": det_id,
                                    "normal_first_round_owner_track_id": owner_track,
                                    "normal_first_round_owner_type": owner_type,
                                    "GT_owner_id": owner_gt,
                                    "was_paired_to_GT_owners_prior_C_track": bool(rightful)}
                local[f"{method}_chosen_{owner_type}"] += 1
                local[f"{method}_chosen_paired_to_GT_owners_prior_C_track"] += rightful
                local[f"{group}_{method}_chosen_{owner_type}"] += 1
                local[f"{group}_{method}_chosen_paired_to_GT_owners_prior_C_track"] += rightful
                if not same:
                    local[f"disagree_{method}_chosen_{owner_type}"] += 1
            local["actions"] += 1
            local["same"] += same
            local["disagree"] += not same
            local[f"harm_{group}"] += 1
            local[f"target_pair_{target_pair}"] += 1
            local[f"positive_pair_{positive_pair}"] += 1
            local[f"{group}_positive_pair_{positive_pair}"] += 1
            local[f"{group}_target_pair_{target_pair}"] += 1
            if not same:
                local[f"disagree_positive_pair_{positive_pair}"] += 1
            joined.append({"sequence": seq, "frame": frame, "track_id": target,
                           "anchor_frame": score["anchor_frame"],
                           "target_gt_id": score["target_gt_id"],
                           "positive_detection_id": positive,
                           "normal_first_round_target_detection_id": baseline_target_det,
                           "positive_pair_category": positive_pair,
                           "target_pair_category": target_pair,
                           "selected": selected,
                           "same_selection": same, "other_anchor_harm_group": group,
                           "appearance_minus_geometry_all_anchor_frames":
                               score["score"]["appearance"]["all_anchor_net_delta"] -
                               score["score"]["geometry"]["all_anchor_net_delta"]})
        totals.update(local)
        per_video.append({"sequence": seq, "counts": dict(local)})
    if len(joined) != 576 or totals["same"] != 529 or totals["disagree"] != 47 or \
            totals["harm_both"] != 414:
        raise RuntimeError("joined aggregate denominator mismatch")
    with rows_path.open("w") as handle:
        for row in joined:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {"status": "PASS", "analysis_type": "posthoc_read_only_pair_trace_join",
              "source_hashes": {"E5_summary": sha256_file(e5_summary_path),
                                "E5_error_summary": sha256_file(error_summary_path),
                                "pair_trace_summary": sha256_file(trace_summary_path)},
              "joined_events_sha256": sha256_file(rows_path),
              "totals": dict(totals), "per_video": per_video,
              "generated_utc": datetime.now(timezone.utc).isoformat()}
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "totals": dict(totals),
                      "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
