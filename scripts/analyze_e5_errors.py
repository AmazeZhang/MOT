#!/usr/bin/env python3
"""Read-only descriptive breakdown of the completed E5 fit branches."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_p1_common_state_mechanism import read_detection_cache
from mot_research.analysis.failure_events import (
    Candidate, hungarian_iou_matches, read_detections, read_gt, read_tracker,
)
from mot_research.analysis.runtime_record import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "manifests/protocols/e5_readonly_error_analysis_20261120.yaml"


def assert_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch: {path}: {actual} != {expected}")


def first_output(raw_rows: list[dict]) -> list[Candidate]:
    return [Candidate(int(row["id"]), np.asarray(row["box"], dtype=np.float64))
            for row in raw_rows]


def target_first_correct(gt_frame, output, gt_id, track_id) -> bool | None:
    if gt_id is None or gt_id not in gt_frame:
        return None
    matched = hungarian_iou_matches(gt_frame, output, .5)[0]
    return matched.get(gt_id, (None,))[0] == track_id


def chosen_owner(det_id, det_to_gt, target_gt, anchors) -> str:
    owner = det_to_gt.get(det_id)
    if owner is None:
        return "no_matched_GT"
    if target_gt is not None and owner == target_gt:
        return "target_GT"
    if owner in anchors:
        return "other_prior_C_anchor_GT"
    return "other_GT_without_prior_C_anchor"


def positive_pool_status(raw, scored, high_ids, high_values) -> str:
    positive = scored["positive_detection_id"]
    if positive is None:
        return "no_target_positive_detection"
    if positive not in high_ids:
        raise RuntimeError("scored positive detection missing from high cache")
    options = {high_ids[int(row["det_index"])] for row in raw["action_features"]["rows"]}
    if positive in options:
        return "in_recorded_alternative_pool"
    index = high_ids.index(positive)
    old = np.asarray(raw["old_box"], dtype=np.float64)
    current = np.asarray(high_values[index][:4], dtype=np.float64)
    distance = float(np.linalg.norm((current[:2] + current[2:]) / 2 -
                                    (old[:2] + old[2:]) / 2) /
                     max(float(np.linalg.norm(old[2:] - old[:2])), 1.0))
    if distance > 2.0:
        return "outside_old_box_spatial_radius"
    return "inside_radius_but_not_alternative"


def quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("analysis must set CUDA_VISIBLE_DEVICES='' and use CPU only")
    plan = yaml.safe_load(PLAN.read_text())
    inp = plan["inputs"]
    assert_sha(ROOT / inp["e5_protocol"], inp["e5_protocol_sha256"])
    run = Path(inp["e5_run"])
    summary_path = run / "summary.json"
    assert_sha(summary_path, inp["e5_summary_sha256"])
    e5 = json.loads(summary_path.read_text())
    if e5["status"] != "PASS" or e5["partition"] != "fit" or e5["fit_gate_pass"] is not False:
        raise RuntimeError("E5 fit result status changed")
    output = Path(plan["outputs"]["machine_summary"]).parent
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    totals = Counter()
    per_video = []
    differences = []
    disagreements = []
    joint_harm_rows = []
    scored_totals = Counter()
    for seq_info in e5["sequences"]:
        seq = seq_info["sequence"]
        seq_dir = Path(inp["train_root"]) / seq
        gt_path = seq_dir / "gt/gt.txt"
        cache_path = Path(inp["detection_root"]) / f"{seq}.csv"
        assert_sha(gt_path, seq_info["GT_sha256"])
        assert_sha(cache_path, seq_info["cache_sha256"])
        assert_sha(seq_dir / "seqinfo.ini", seq_info["seqinfo_sha256"])
        raw_path = run / f"{seq}.raw.jsonl"
        scored_path = run / f"{seq}.scored.jsonl"
        assert_sha(raw_path, seq_info["raw_sha256"])
        assert_sha(scored_path, seq_info["scored_sha256"])
        raws = [json.loads(line) for line in raw_path.read_text().splitlines()]
        scores = [json.loads(line) for line in scored_path.read_text().splitlines()]
        if len(raws) != len(scores) or len(raws) != seq_info["scores"]["actions"]:
            raise RuntimeError(f"{seq}: event row count mismatch")
        gt = read_gt(gt_path)
        high = read_detections(cache_path, .6)
        cached, det_ids = read_detection_cache(cache_path)
        c_ref = read_tracker(Path(inp["e3_fit_trackers"]) / "C" / f"{seq}.txt")
        e_ref = read_tracker(Path(inp["e3_fit_trackers"]) / "E" / f"{seq}.txt")
        local = Counter()
        local_scores = Counter()
        for raw, score in zip(raws, scores):
            identity = (seq, raw["frame"], raw["track_id"], raw["anchor_frame"])
            if identity != (score["sequence"], score["frame"], score["track_id"], score["anchor_frame"]):
                raise RuntimeError(f"{seq}: raw/scored identity mismatch")
            frame = raw["frame"]
            high_ids = [int(did) for did, value in zip(det_ids.get(frame, []), cached.get(frame, []))
                        if float(value[4]) > .6]
            high_values = [value for value in cached.get(frame, []) if float(value[4]) > .6]
            for method in ("geometry", "appearance"):
                selected_index = int(raw["action_features"][method]["det_index"])
                if high_ids[selected_index] != score["selected_detection_ids"][method]:
                    raise RuntimeError(f"{seq}:{frame}: selected detection index drift")
            gt_frame = gt.get(frame, {})
            anchors = hungarian_iou_matches(gt.get(raw["anchor_frame"], {}),
                                             c_ref.get(raw["anchor_frame"], []), .5)[0]
            anchor_ids = set(anchors)
            target_gt = score["target_gt_id"]
            if target_gt is not None and anchors[target_gt][0] != raw["track_id"]:
                raise RuntimeError(f"{seq}:{frame}: target anchor mismatch")
            positives = hungarian_iou_matches(gt_frame, high.get(frame, []), .5)[0]
            det_to_gt = {det_id: gid for gid, (det_id, _) in positives.items()}
            positive = positives.get(target_gt, (None,))[0] if target_gt is not None else None
            if positive != score["positive_detection_id"]:
                raise RuntimeError(f"{seq}:{frame}: target positive detection mismatch")
            ids = score["selected_detection_ids"]
            same = ids["geometry"] == ids["appearance"]
            if same and raw["geometry_outputs"] != raw["appearance_outputs"]:
                raise RuntimeError(f"{seq}:{frame}: same forced pair produced different branch outputs")
            first = {
                "baseline": target_first_correct(gt_frame, e_ref.get(frame, []), target_gt, raw["track_id"]),
                "geometry": target_first_correct(gt_frame, first_output(raw["geometry_outputs"][0]), target_gt, raw["track_id"]),
                "appearance": target_first_correct(gt_frame, first_output(raw["appearance_outputs"][0]), target_gt, raw["track_id"]),
            }
            harm_geo = bool(score["score"]["geometry"]["other_anchor_harmed"])
            harm_app = bool(score["score"]["appearance"]["other_anchor_harmed"])
            if harm_geo and harm_app:
                harm_group = "both"
            elif harm_geo:
                harm_group = "geometry_only"
            elif harm_app:
                harm_group = "appearance_only"
            else:
                harm_group = "neither"
            delta_geo = int(score["score"]["geometry"]["all_anchor_net_delta"])
            delta_app = int(score["score"]["appearance"]["all_anchor_net_delta"])
            difference = delta_app - delta_geo
            differences.append(difference)
            pool = positive_pool_status(raw, score, high_ids, high_values)
            owners = {method: chosen_owner(ids[method], det_to_gt, target_gt, anchor_ids)
                      for method in ("geometry", "appearance")}
            key = "same" if same else "disagree"
            local["actions"] += 1
            local[key] += 1
            local[f"harm_{harm_group}"] += 1
            local[f"{key}_harm_{harm_group}"] += 1
            local[f"{key}_target_anchored"] += target_gt is not None
            local[f"{key}_positive_exists"] += positive is not None
            local[f"{key}_positive_{pool}"] += 1
            local[f"{key}_baseline_first_target_correct"] += first["baseline"] is True
            local[f"{key}_geometry_first_target_correct"] += first["geometry"] is True
            local[f"{key}_appearance_first_target_correct"] += first["appearance"] is True
            local[f"{key}_app_minus_geo_frames"] += difference
            local[f"{key}_difference_positive"] += difference > 0
            local[f"{key}_difference_negative"] += difference < 0
            local[f"{key}_difference_zero"] += difference == 0
            local[f"{key}_geometry_correct_positive"] += score["correctness"]["geometry"] is True
            local[f"{key}_appearance_correct_positive"] += score["correctness"]["appearance"] is True
            for method in ("geometry", "appearance"):
                local[f"{key}_{method}_owner_{owners[method]}"] += 1
                local[f"both_harm_{method}_owner_{owners[method]}"] += harm_group == "both"
            if harm_group == "both":
                local["both_harm_target_anchored"] += target_gt is not None
                local["both_harm_positive_exists"] += positive is not None
                local["both_harm_baseline_first_target_correct"] += first["baseline"] is True
                local["both_harm_both_all_anchor_net_negative"] += delta_geo < 0 and delta_app < 0
            record = {"sequence": seq, "frame": frame, "track_id": raw["track_id"],
                      "target_gt_id": target_gt, "positive_detection_id": positive,
                      "positive_pool_status": pool, "selected_detection_ids": ids,
                      "owner": owners, "baseline_first_target_correct": first["baseline"],
                      "geometry_first_target_correct": first["geometry"],
                      "appearance_first_target_correct": first["appearance"],
                      "geometry_all_anchor_net": delta_geo, "appearance_all_anchor_net": delta_app,
                      "appearance_minus_geometry": difference, "harm_group": harm_group,
                      "appearance_similarity": raw["action_features"]["appearance"]["similarity"],
                      "appearance_margin": raw["action_features"]["margin"],
                      "geometry_selected_iou": raw["action_features"]["geometry"]["geometry_iou"],
                      "appearance_selected_iou": raw["action_features"]["appearance"]["geometry_iou"]}
            if not same:
                disagreements.append(record)
            if harm_group == "both":
                joint_harm_rows.append(record)
            # Recompute the E5 summary fields from scored rows as an input check.
            local_scores["actions"] += 1
            local_scores["disagreeing_actions"] += not same
            local_scores["target_anchored"] += target_gt is not None
            local_scores["target_positive_detection_exists"] += positive is not None
            for method in ("geometry", "appearance"):
                item = score["score"][method]
                local_scores[f"{method}_all_anchor_net"] += item["all_anchor_net_delta"]
                local_scores[f"{method}_other_harm_events"] += item["other_anchor_harmed"]
                local_scores[f"{method}_all_harm_events"] += item["all_anchor_harmed"]
                local_scores[f"{method}_correct_actions"] += score["correctness"][method] is True
                local_scores[f"{method}_wrong_actions"] += score["correctness"][method] is False
                if target_gt is not None:
                    local_scores[f"{method}_target_net"] += item["target_delta"]
                    local_scores[f"{method}_target_harm_events"] += item["target_delta"] < 0
        local_scores["appearance_minus_geometry_all_anchor_net"] = (
            local_scores["appearance_all_anchor_net"] - local_scores["geometry_all_anchor_net"])
        if dict(local_scores) != seq_info["scores"]:
            raise RuntimeError(f"{seq}: E5 score summary does not match event rows")
        scored_totals.update(local_scores)
        totals.update(local)
        per_video.append({"sequence": seq, "counts": dict(local)})
    if dict(scored_totals) != e5["scores"]:
        raise RuntimeError("E5 aggregate scores do not match event rows")
    if totals["actions"] != 576 or totals["disagree"] != 47 or totals["same"] != 529:
        raise RuntimeError("unexpected E5 action denominator")
    if len(disagreements) != 47 or len(joint_harm_rows) != totals["harm_both"]:
        raise RuntimeError("breakdown row coverage mismatch")
    disagreement_deltas = [r["appearance_minus_geometry"] for r in disagreements]
    result = {
        "status": "PASS", "analysis_type": "posthoc_read_only_descriptive",
        "plan_sha256": sha256_file(PLAN), "e5_summary_sha256": inp["e5_summary_sha256"],
        "e5_protocol_sha256": inp["e5_protocol_sha256"],
        "totals": dict(totals), "E5_score_totals_verified": dict(scored_totals),
        "disagreement_delta_distribution": {
            "count": len(disagreement_deltas), "sum": sum(disagreement_deltas),
            "min": min(disagreement_deltas), "p25": quantile(disagreement_deltas, .25),
            "median": quantile(disagreement_deltas, .5), "p75": quantile(disagreement_deltas, .75),
            "max": max(disagreement_deltas)},
        "per_video": per_video, "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output / "disagreements.jsonl").open("w") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output / "joint_other_anchor_harm.jsonl").open("w") as handle:
        for row in joint_harm_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result["disagreements_sha256"] = sha256_file(output / "disagreements.jsonl")
    result["joint_harm_sha256"] = sha256_file(output / "joint_other_anchor_harm.jsonl")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "totals": result["totals"],
                      "disagreement_delta_distribution": result["disagreement_delta_distribution"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
