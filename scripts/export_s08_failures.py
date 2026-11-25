#!/usr/bin/env python3
"""Stage F: Export bounded official IDSW failure inventory and case evidence for S08 T_SCRATCH."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import trackeval
import yaml
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s08_camel_baseline_20261125.yaml"
REPORT_DIR = REPO_ROOT / "reports/baselines/s08_camel_baseline_20261125"
RUNS_ROOT = DATA_ROOT / "runs/s08_camel_baseline_20261125"


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def box_iou_ltwh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


def load_dev8_raw_detections(state_pklz: Path, seq_to_vid: dict[str, int]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Map seq -> frame (1-indexed) -> list of frame-local detections from GT-free state file."""
    out: dict[str, dict[int, list[dict[str, Any]]]] = {}
    with zipfile.ZipFile(state_pklz, "r") as zf:
        for seq, vid in seq_to_vid.items():
            img_df = pickle.load(zf.open(f"{vid}_image.pkl"))
            det_df = pickle.load(zf.open(f"{vid}.pkl"))
            img_id_to_frame = {int(idx): int(row["frame"]) + 1 for idx, row in img_df.iterrows()}
            per_frame: dict[int, list[dict[str, Any]]] = {}
            for _, row in det_df.iterrows():
                fr = img_id_to_frame[int(row["image_id"])]
                per_frame.setdefault(fr, []).append({
                    "bbox_ltwh": [float(x) for x in row["bbox_ltwh"]],
                    "bbox_conf": float(row["bbox_conf"]),
                })
            out[seq] = per_frame
    return out


def extract_official_idsw_events(
    dataset: trackeval.datasets.MotChallenge2DBox,
    seq: str,
    raw_dets_by_frame: dict[int, list[dict[str, Any]]],
    expected_idsw: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay TrackEval MotChallenge2DBox + CLEAR exactly to locate official IDSW events."""
    raw_data = dataset.get_raw_seq_data("tracklab", seq)
    data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")

    # Reconstruct inverse ID maps identical to MotChallenge2DBox.get_preprocessed_seq_data
    unique_gt_ids = []
    unique_tracker_ids = []
    for t in range(raw_data["num_timesteps"]):
        gt_zero_marked = raw_data["gt_extras"][t]["zero_marked"]
        gt_keep = np.not_equal(gt_zero_marked, 0)
        unique_gt_ids.extend(np.unique(raw_data["gt_ids"][t][gt_keep]).tolist())
        unique_tracker_ids.extend(np.unique(raw_data["tracker_ids"][t]).tolist())
    unique_gt_ids = np.unique(unique_gt_ids) if unique_gt_ids else np.array([], dtype=int)
    unique_tracker_ids = np.unique(unique_tracker_ids) if unique_tracker_ids else np.array([], dtype=int)

    num_gt_ids = data["num_gt_ids"]
    prev_tracker_id = np.nan * np.zeros(num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)
    last_matched_t = np.full(num_gt_ids, -1, dtype=int)
    tracker_last_gt: dict[int, tuple[int, int]] = {}  # orig_pred_id -> (orig_gt_id, last_frame)

    # Also build frame-level match history for window evidence
    frame_records: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    total_idsw = 0

    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
        frame_1 = t + 1
        gt_dets_t = data["gt_dets"][t]
        trk_dets_t = data["tracker_dets"][t]

        if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
            frame_records[frame_1] = {"matches": [], "unmatched_gt": [int(unique_gt_ids[g]) for g in gt_ids_t]}
            continue

        similarity = data["similarity_scores"][t]
        score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < 0.5 - np.finfo("float").eps] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched = score_mat[match_rows, match_cols] > 0 + np.finfo("float").eps
        match_rows = match_rows[actually_matched]
        match_cols = match_cols[actually_matched]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]

        matches_frame = []
        current_gt_to_pred: dict[int, int] = {}
        for r, c, g_idx, trk_idx in zip(match_rows, match_cols, matched_gt_ids, matched_tracker_ids):
            orig_gt = int(unique_gt_ids[g_idx])
            orig_trk = int(unique_tracker_ids[trk_idx])
            current_gt_to_pred[orig_gt] = orig_trk
            matches_frame.append({
                "gt_id": orig_gt,
                "pred_id": orig_trk,
                "iou": float(similarity[r, c]),
                "gt_bbox_ltwh": [float(x) for x in gt_dets_t[r]],
                "pred_bbox_ltwh": [float(x) for x in trk_dets_t[c]],
            })

            prev_trk_idx = prev_tracker_id[g_idx]
            if not np.isnan(prev_trk_idx) and int(trk_idx) != int(prev_trk_idx):
                total_idsw += 1
                orig_prev_trk = int(unique_tracker_ids[int(prev_trk_idx)])
                prev_t = int(last_matched_t[g_idx])
                prev_frame_1 = prev_t + 1
                gt_box = np.asarray(gt_dets_t[r], dtype=float)

                # Check qualifying detections (conf >= 0.4) at frame_1
                qual_dets = [d for d in raw_dets_by_frame.get(frame_1, []) if d["bbox_conf"] >= 0.4]
                det_ious = [box_iou_ltwh(gt_box, np.asarray(d["bbox_ltwh"], dtype=float)) for d in qual_dets]
                max_det_iou = max(det_ious) if det_ious else 0.0
                has_qual_det = max_det_iou >= 0.5

                # Check competing GT boxes (IoU >= 0.3 with this GT box on same frame)
                competing_gts = []
                for j, other_g_idx in enumerate(gt_ids_t):
                    if int(other_g_idx) == int(g_idx):
                        continue
                    iou_other = box_iou_ltwh(gt_box, np.asarray(gt_dets_t[j], dtype=float))
                    if iou_other >= 0.3:
                        competing_gts.append({
                            "gt_id": int(unique_gt_ids[int(other_g_idx)]),
                            "iou_with_target_gt": round(iou_other, 6),
                        })

                # Other affected identities: who did new_pred_id previously follow, or who took prev_pred_id now?
                other_affected = []
                if orig_trk in tracker_last_gt and tracker_last_gt[orig_trk][0] != orig_gt:
                    other_affected.append({
                        "role": "new_pred_previously_matched_gt",
                        "pred_id": orig_trk,
                        "other_gt_id": tracker_last_gt[orig_trk][0],
                        "last_frame": tracker_last_gt[orig_trk][1],
                    })

                sort_key = sha256_str(f"13:{seq}:{frame_1}:{orig_gt}")
                events.append({
                    "sequence": seq,
                    "frame": frame_1,
                    "gt_id": orig_gt,
                    "sort_key_sha256": sort_key,
                    "event_source": "official_TrackEval_CLEAR_IDSW",
                    "prev_pred_id": orig_prev_trk,
                    "new_pred_id": orig_trk,
                    "prev_matched_frame": prev_frame_1,
                    "gap_since_prev_match_frames": int(frame_1 - prev_frame_1),
                    "matched_iou": float(similarity[r, c]),
                    "gt_bbox_ltwh": [float(x) for x in gt_dets_t[r]],
                    "new_pred_bbox_ltwh": [float(x) for x in trk_dets_t[c]],
                    "has_qualifying_det_iou_ge_0_5": has_qual_det,
                    "max_qualifying_det_iou": float(max_det_iou),
                    "has_competing_gt_iou_ge_0_3": len(competing_gts) > 0,
                    "competing_gt_details": competing_gts,
                    "other_affected_identities": other_affected,
                    "unknown_reason": "",
                })

        # Fill in if orig_prev_trk matched another GT on this same frame
        for ev in events:
            if ev["frame"] == frame_1:
                for m in matches_frame:
                    if m["pred_id"] == ev["prev_pred_id"] and m["gt_id"] != ev["gt_id"]:
                        ev["other_affected_identities"].append({
                            "role": "prev_pred_now_matched_other_gt",
                            "pred_id": ev["prev_pred_id"],
                            "other_gt_id": m["gt_id"],
                            "frame": frame_1,
                        })

        for g_idx, trk_idx in zip(matched_gt_ids, matched_tracker_ids):
            last_matched_t[g_idx] = t
            tracker_last_gt[int(unique_tracker_ids[int(trk_idx)])] = (int(unique_gt_ids[int(g_idx)]), frame_1)

        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        frame_records[frame_1] = {
            "matches": matches_frame,
            "num_gt": int(len(gt_ids_t)),
            "num_pred": int(len(tracker_ids_t)),
        }

    assert total_idsw == expected_idsw, f"{seq}: replayed IDSW {total_idsw} != official {expected_idsw}"
    return events, frame_records


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    dev8 = list(protocol["data"]["dev8"]["sequences"])
    split_audit = json.loads((REPORT_DIR / "split_audit.json").read_text())
    seq_to_vid = {x["sequence"]: x["video_id"] for x in split_audit["sequence_inventory"] if x["split_role"] == "dev8"}
    seq_nframes = {x["sequence"]: x["nframes"] for x in split_audit["sequence_inventory"] if x["split_role"] == "dev8"}

    t_summary_path = RUNS_ROOT / "e_t_scratch_dev8_v1/stage_e_summary.json"
    t_summary = json.loads(t_summary_path.read_text())
    gt_dir = Path(t_summary["eval_summary"]["gt_dir"])
    pred_dir = Path(t_summary["eval_summary"]["pred_dir"])
    detailed_csv = Path(t_summary["eval_summary"]["detailed_csv"])

    with detailed_csv.open(newline="") as f:
        official_rows = {r["seq"]: r for r in csv.DictReader(f)}

    raw_dets = load_dev8_raw_detections(Path(t_summary["state_file"]), seq_to_vid)

    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "BENCHMARK": "DanceTrack",
        "SPLIT_TO_EVAL": "val",
        "GT_FOLDER": str(gt_dir.parent),
        "TRACKERS_FOLDER": str(pred_dir.parent.parent),
        "TRACKERS_TO_EVAL": ["tracklab"],
        "TRACKER_SUB_FOLDER": "",
        "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt",
        "DO_PREPROC": False,
        "PRINT_CONFIG": False,
        "SEQ_INFO": seq_nframes,
    })
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)

    all_events: list[dict[str, Any]] = []
    sampled_events: list[dict[str, Any]] = []
    per_seq_counts: dict[str, Any] = {}
    evidence_dir = REPORT_DIR / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for seq in dev8:
        expected_idsw = int(float(official_rows[seq]["IDSW"]))
        events, frame_records = extract_official_idsw_events(dataset, seq, raw_dets[seq], expected_idsw)
        events_sorted = sorted(events, key=lambda e: e["sort_key_sha256"])
        chosen = events_sorted[:2]
        per_seq_counts[seq] = {
            "official_idsw": expected_idsw,
            "extracted_idsw": len(events),
            "sampled_count": len(chosen),
            "nframes": seq_nframes[seq],
        }
        all_events.extend(events)

        for idx, ev in enumerate(chosen, start=1):
            case_id = f"s08_idsw_{seq}_f{ev['frame']}_gt{ev['gt_id']}"
            ev["case_id"] = case_id
            w_start = max(1, ev["frame"] - 2)
            w_end = min(seq_nframes[seq], ev["frame"] + 2)
            window = {str(fr): frame_records.get(fr, {}) for fr in range(w_start, w_end + 1)}
            ev_path = evidence_dir / f"{case_id}.json"
            ev_payload = {
                "case_id": case_id,
                "sequence": seq,
                "event": ev,
                "window_frames_inclusive": [w_start, w_end],
                "window_records": window,
            }
            ev_path.write_text(json.dumps(ev_payload, indent=2) + "\n")
            ev["evidence_path"] = f"evidence/{ev_path.name}"
            sampled_events.append(ev)

    assert len(sampled_events) <= 16

    # Write failure_inventory.csv
    inv_fields = [
        "case_id",
        "sequence",
        "frame",
        "gt_id",
        "sort_key_sha256",
        "event_source",
        "prev_pred_id",
        "new_pred_id",
        "prev_matched_frame",
        "gap_since_prev_match_frames",
        "has_qualifying_det_iou_ge_0_5",
        "max_qualifying_det_iou",
        "has_competing_gt_iou_ge_0_3",
        "competing_gt_ids",
        "other_affected_identities",
        "unknown_reason",
        "evidence_ref",
    ]
    with (REPORT_DIR / "failure_inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_fields, lineterminator="\n")
        w.writeheader()
        for ev in sampled_events:
            w.writerow({
                "case_id": ev["case_id"],
                "sequence": ev["sequence"],
                "frame": ev["frame"],
                "gt_id": ev["gt_id"],
                "sort_key_sha256": ev["sort_key_sha256"],
                "event_source": ev["event_source"],
                "prev_pred_id": ev["prev_pred_id"],
                "new_pred_id": ev["new_pred_id"],
                "prev_matched_frame": ev["prev_matched_frame"],
                "gap_since_prev_match_frames": ev["gap_since_prev_match_frames"],
                "has_qualifying_det_iou_ge_0_5": ev["has_qualifying_det_iou_ge_0_5"],
                "max_qualifying_det_iou": repr(ev["max_qualifying_det_iou"]),
                "has_competing_gt_iou_ge_0_3": ev["has_competing_gt_iou_ge_0_3"],
                "competing_gt_ids": json.dumps([c["gt_id"] for c in ev["competing_gt_details"]]),
                "other_affected_identities": json.dumps(ev["other_affected_identities"]),
                "unknown_reason": ev["unknown_reason"],
                "evidence_ref": ev["evidence_path"],
            })

    # Write cases.jsonl
    with (REPORT_DIR / "cases.jsonl").open("w") as f:
        for ev in sampled_events:
            case_obj = {
                "case_id": ev["case_id"],
                "run_id": t_summary["run_id"],
                "condition": "T_SCRATCH",
                "baseline_condition": None,
                "sequence": ev["sequence"],
                "frame_range": [max(1, ev["frame"] - 2), min(seq_nframes[ev["sequence"]], ev["frame"] + 2)],
                "event_frame": ev["frame"],
                "sampling_stratum": "official_IDSW_per_sequence_max2",
                "sort_key_sha256": ev["sort_key_sha256"],
                "comparison_type": "single_condition_closed_loop_official_IDSW",
                "gt_id": ev["gt_id"],
                "gt_role": "offline_official_TrackEval_IDSW_diagnosis_only",
                "prev_pred_id": ev["prev_pred_id"],
                "new_pred_id": ev["new_pred_id"],
                "prev_matched_frame": ev["prev_matched_frame"],
                "gap_since_prev_match_frames": ev["gap_since_prev_match_frames"],
                "has_qualifying_det_iou_ge_0_5": ev["has_qualifying_det_iou_ge_0_5"],
                "has_competing_gt_iou_ge_0_3": ev["has_competing_gt_iou_ge_0_3"],
                "competing_gt_details": ev["competing_gt_details"],
                "other_affected_identities": ev["other_affected_identities"],
                "gt_bbox_ltwh": ev["gt_bbox_ltwh"],
                "new_pred_bbox_ltwh": ev["new_pred_bbox_ltwh"],
                "matched_iou": ev["matched_iou"],
                "evidence_path": ev["evidence_path"],
            }
            f.write(json.dumps(case_obj) + "\n")

    # Descriptive stats over ALL extracted official IDSW events and sampled subset
    total_all = len(all_events)
    all_gap_gt1 = sum(1 for e in all_events if e["gap_since_prev_match_frames"] > 1)
    all_det_present = sum(1 for e in all_events if e["has_qualifying_det_iou_ge_0_5"])
    all_competing_gt = sum(1 for e in all_events if e["has_competing_gt_iou_ge_0_3"])
    all_both_gap_and_crowd = sum(
        1 for e in all_events if e["gap_since_prev_match_frames"] > 1 and e["has_competing_gt_iou_ge_0_3"]
    )
    all_neither_gap_nor_crowd = sum(
        1 for e in all_events if e["gap_since_prev_match_frames"] == 1 and not e["has_competing_gt_iou_ge_0_3"]
    )

    diagnostics = {
        "schema_version": "mot-report-v1",
        "experiment_id": "s08_camel_baseline_20261125",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "condition": "T_SCRATCH",
        "run_id": t_summary["run_id"],
        "scope_id": "dev8",
        "sampling_rule": {
            "rule": "ascending_SHA256_of_13_colon_sequence_colon_frame_colon_gt_id",
            "max_per_video": 2,
            "max_total": 16,
            "total_official_idsw_candidates": total_all,
            "total_sampled_cases": len(sampled_events),
            "note": "Sampled subset (<=16) is illustrative only; do not extrapolate population rates from sampled cases.",
        },
        "statistics": [
            {
                "name": "total_official_idsw_events",
                "definition": "Total official TrackEval CLEAR IDSW events for T_SCRATCH on dev8",
                "value": total_all,
                "unit": "count",
                "numerator": total_all,
                "denominator": None,
                "scope": "dev8",
                "gt_role": "offline_official_TrackEval",
                "source": "e_t_scratch_dev8_v1/eval/results/tracklab/pedestrian_detailed.csv:COMBINED:IDSW",
            },
            {
                "name": "idsw_after_match_gap_gt_1_frame",
                "definition": "Official IDSW events where frame - prev_matched_frame > 1",
                "value": all_gap_gt1 / total_all if total_all else None,
                "unit": "fraction_of_all_idsw",
                "numerator": all_gap_gt1,
                "denominator": total_all,
                "scope": "dev8",
                "gt_role": "offline_official_TrackEval",
                "source": "replayed_TrackEval_CLEAR_matching",
            },
            {
                "name": "idsw_with_qualifying_detection_iou_ge_0_5",
                "definition": "Official IDSW events with a frame-local detection (conf>=0.4) having IoU>=0.5 with target GT (descriptive only, not valid tracker edge proof)",
                "value": all_det_present / total_all if total_all else None,
                "unit": "fraction_of_all_idsw",
                "numerator": all_det_present,
                "denominator": total_all,
                "scope": "dev8",
                "gt_role": "offline_diagnosis_only",
                "source": "states_dev8_gt_free.pklz + gt.txt",
            },
            {
                "name": "idsw_with_competing_gt_iou_ge_0_3",
                "definition": "Official IDSW events with another pedestrian GT box on the same frame having IoU>=0.3 with target GT (descriptive non-exclusive crowding flag)",
                "value": all_competing_gt / total_all if total_all else None,
                "unit": "fraction_of_all_idsw",
                "numerator": all_competing_gt,
                "denominator": total_all,
                "scope": "dev8",
                "gt_role": "offline_diagnosis_only",
                "source": "gt.txt",
            },
            {
                "name": "idsw_overlap_gap_gt_1_and_competing_gt",
                "definition": "Official IDSW events with both gap>1 and competing GT IoU>=0.3",
                "value": all_both_gap_and_crowd / total_all if total_all else None,
                "unit": "fraction_of_all_idsw",
                "numerator": all_both_gap_and_crowd,
                "denominator": total_all,
                "scope": "dev8",
                "gt_role": "offline_diagnosis_only",
                "source": "replayed_TrackEval_CLEAR_matching + gt.txt",
            },
            {
                "name": "idsw_neither_gap_gt_1_nor_competing_gt",
                "definition": "Official IDSW events with gap==1 and no competing GT IoU>=0.3",
                "value": all_neither_gap_nor_crowd / total_all if total_all else None,
                "unit": "fraction_of_all_idsw",
                "numerator": all_neither_gap_nor_crowd,
                "denominator": total_all,
                "scope": "dev8",
                "gt_role": "offline_diagnosis_only",
                "source": "replayed_TrackEval_CLEAR_matching + gt.txt",
            },
        ],
        "per_sequence_idsw_counts": per_seq_counts,
    }
    (REPORT_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "total_official_idsw": total_all,
        "sampled_cases": len(sampled_events),
        "all_gap_gt1": all_gap_gt1,
        "all_competing_gt": all_competing_gt,
        "all_both_overlap": all_both_gap_and_crowd,
    }, indent=2))


if __name__ == "__main__":
    main()
