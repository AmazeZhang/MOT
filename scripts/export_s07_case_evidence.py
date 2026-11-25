#!/usr/bin/env python3
"""Export descriptive GT-assisted evidence for existing S06 evaluation8 decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


SEQUENCES = ("dancetrack0074", "dancetrack0032", "dancetrack0049", "dancetrack0023")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[:, 2] += result[:, 0]
    result[:, 3] += result[:, 1]
    return result


def iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.empty((len(left), len(right)), dtype=np.float64)
    tl = np.maximum(left[:, None, :2], right[None, :, :2])
    br = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.maximum(0.0, br - tl)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = np.prod(left[:, 2:] - left[:, :2], axis=1)
    right_area = np.prod(right[:, 2:] - right[:, :2], axis=1)
    return intersection / np.maximum(left_area[:, None] + right_area[None, :] - intersection, 1e-12)


def match(ids_a: list[int], boxes_a: np.ndarray, ids_b: list[int], boxes_b: np.ndarray,
          threshold: float = 0.5) -> dict[int, dict]:
    scores = iou(boxes_a, boxes_b)
    if not scores.size:
        return {}
    rows, columns = linear_sum_assignment(-scores)
    return {ids_a[row]: {"id": ids_b[column], "iou": float(scores[row, column])}
            for row, column in zip(rows, columns) if scores[row, column] >= threshold}


def load_detections(path: Path) -> dict[int, tuple[list[int], np.ndarray]]:
    grouped: dict[int, list[tuple[int, list[float]]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            grouped.setdefault(frame, []).append((int(row["det_id"]),
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]))
    return {frame: ([item[0] for item in values], np.asarray([item[1] for item in values]))
            for frame, values in grouped.items()}


def load_mot(path: Path, gt: bool = False) -> dict[int, tuple[list[int], np.ndarray]]:
    grouped: dict[int, list[tuple[int, list[float]]]] = {}
    for line in path.read_text().splitlines():
        fields = line.split(",")
        frame, identity = int(float(fields[0])), int(float(fields[1]))
        if gt and len(fields) > 6 and int(float(fields[6])) != 1:
            continue
        grouped.setdefault(frame, []).append((identity, [float(value) for value in fields[2:6]]))
    return {frame: ([item[0] for item in values], xywh_to_xyxy([item[1] for item in values]))
            for frame, values in grouped.items()}


def at(mapping: dict[int, tuple[list[int], np.ndarray]], frame: int) -> tuple[list[int], np.ndarray]:
    return mapping.get(frame, ([], np.empty((0, 4), dtype=np.float64)))


def trajectory_summary(tracks: dict, gt: dict, identities: list[int], start: int, end: int) -> dict:
    observations = {identity: [] for identity in identities}
    visible = {identity: 0 for identity in identities}
    for frame in range(start, end + 1):
        track_ids, track_boxes = at(tracks, frame)
        gt_ids, gt_boxes = at(gt, frame)
        gt_to_track = {value["id"]: {"id": key, "iou": value["iou"]}
                       for key, value in match(track_ids, track_boxes, gt_ids, gt_boxes).items()}
        for identity in identities:
            if identity in gt_ids:
                visible[identity] += 1
            if identity in gt_to_track:
                observations[identity].append({"frame": frame, "track_id": gt_to_track[identity]["id"],
                                               "iou": gt_to_track[identity]["iou"]})
    result = {}
    for identity in identities:
        track_ids = [row["track_id"] for row in observations[identity]]
        result[str(identity)] = {
            "gt_visible_frames": visible[identity],
            "matched_frames_iou_ge_0_5": len(track_ids),
            "track_ids": sorted(set(track_ids)),
            "proxy_track_id_changes": sum(a != b for a, b in zip(track_ids, track_ids[1:])),
            "observations": observations[identity],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--followup-frames", type=int, default=30)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("case export is CPU-only")
    if args.output.exists():
        raise FileExistsError(args.output)
    source_cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    wanted = [row for row in source_cases if row["condition"] == "Context_no_extra_risk"
              and row["sequence"] in SEQUENCES and row["chosen"] != "M"]
    hashes = {str(args.cases): sha256(args.cases)}
    output = []
    for sequence in SEQUENCES:
        sequence_cases = [row for row in wanted if row["sequence"] == sequence]
        trace_path = (args.evaluation_root / "evaluation8_tracking_attempt1" /
                      "Context_no_extra_risk" / sequence / "traces/M" / f"{sequence}.jsonl")
        traces = {row["frame"]: row for row in
                  (json.loads(line) for line in trace_path.read_text().splitlines())}
        gt_path = args.dataset_root / sequence / "gt/gt.txt"
        detection_path = args.detection_root / f"{sequence}.csv"
        baseline_path = args.evaluation_root / "evaluation8_staged_attempt1/M" / f"{sequence}.txt"
        method_path = (args.evaluation_root / "evaluation8_staged_attempt1/Context_no_extra_risk" /
                       f"{sequence}.txt")
        for path in (trace_path, gt_path, detection_path, baseline_path, method_path):
            hashes[str(path)] = sha256(path)
        gt = load_mot(gt_path, gt=True)
        detections = load_detections(detection_path)
        baseline = load_mot(baseline_path)
        method = load_mot(method_path)
        for case in sequence_cases:
            frame = case["frame"]
            trace = traces[frame]
            det_ids, det_boxes = at(detections, frame)
            gt_ids, gt_boxes = at(gt, frame)
            detection_gt = match(det_ids, det_boxes, gt_ids, gt_boxes)
            prior_gt_ids, prior_gt_boxes = at(gt, max(1, frame - 1))
            baseline_prior_ids, baseline_prior_boxes = at(baseline, max(1, frame - 1))
            method_prior_ids, method_prior_boxes = at(method, max(1, frame - 1))
            method_tracker_gt = match(method_prior_ids, method_prior_boxes,
                                      prior_gt_ids, prior_gt_boxes)

            def pair_rows(pairs: list[list[int]], tracker_gt: dict) -> list[dict]:
                records = []
                for detection_id, tracker_id in pairs:
                    det_owner = detection_gt.get(detection_id)
                    tracker_owner = tracker_gt.get(tracker_id)
                    records.append({
                        "detection_id": detection_id, "tracker_id": tracker_id,
                        "detection_gt": None if det_owner is None else det_owner["id"],
                        "detection_gt_iou": None if det_owner is None else det_owner["iou"],
                        "tracker_prior_gt": None if tracker_owner is None else tracker_owner["id"],
                        "tracker_prior_gt_iou": None if tracker_owner is None else tracker_owner["iou"],
                        "proxy_correct": None if det_owner is None or tracker_owner is None
                        else det_owner["id"] == tracker_owner["id"],
                    })
                return records

            # run_s06_selector_closed_loop records proposals["G"] here.  It does
            # not retain the same-state M candidate pairs, so never label this M.
            reference_g_pairs = pair_rows(trace["base_first_pairs"], method_tracker_gt)
            method_pairs = pair_rows(trace["first_pairs"], method_tracker_gt)
            reference_g_set = {tuple(row) for row in trace["base_first_pairs"]}
            method_set = {tuple(row) for row in trace["first_pairs"]}
            changed_pairs = reference_g_set ^ method_set
            affected_det_ids = {row[0] for row in changed_pairs}
            affected_tracker_ids = {row[1] for row in changed_pairs}
            affected_gt = sorted({value["id"] for key, value in detection_gt.items()
                                  if key in affected_det_ids} |
                                 {value["id"] for key, value in method_tracker_gt.items()
                                  if key in affected_tracker_ids})
            end = min(frame + args.followup_frames, max(gt))
            output.append({
                "case_id": case["case_id"], "sequence": sequence, "frame": frame,
                "component_index": case["component_index"], "chosen": case["chosen"],
                "comparison_type": "descriptive_existing_trajectories_not_causal_attribution",
                "gt_role": "offline case diagnosis only; never used by the selector",
                "predictions": case["predictions"],
                "sequence_HOTA_delta_vs_M": case["sequence_HOTA_delta_vs_M"],
                "frame_pairing": {
                    "reference_G_same_method_state": reference_g_pairs,
                    "Context_no_extra_risk_applied": method_pairs,
                    "removed_vs_G": sorted([list(row) for row in reference_g_set - method_set]),
                    "added_vs_G": sorted([list(row) for row in method_set - reference_g_set]),
                    "same_state_M_candidate_pairs_available": False,
                    "reason": "retained S06 trace stores proposals['G'] and applied pairs, not proposals['M']",
                },
                "legality_evidence": {
                    "exact_legal_edge_mask_available": False,
                    "reason": "retained evaluation trace stores applied pairs and eligible-edge count, not the mask",
                    "appearance_eligible_edge_count": trace["appearance_eligible_edges"],
                    "reference_G_one_to_one": len({r[0] for r in reference_g_set}) == len(reference_g_set) == len({r[1] for r in reference_g_set}),
                    "method_one_to_one": len({r[0] for r in method_set}) == len(method_set) == len({r[1] for r in method_set}),
                },
                "affected_detection_ids": sorted(affected_det_ids),
                "affected_tracker_ids": sorted(affected_tracker_ids),
                "affected_gt_identities_iou_ge_0_5": affected_gt,
                "followup_window": [frame, end],
                "baseline_followup": trajectory_summary(baseline, gt, affected_gt, frame, end),
                "method_followup": trajectory_summary(method, gt, affected_gt, frame, end),
                "source_paths": {"trace": str(trace_path), "gt": str(gt_path),
                                 "detections": str(detection_path), "baseline": str(baseline_path),
                                 "method": str(method_path)},
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "status": "COMPLETED", "cases": len(output),
        "per_sequence": {sequence: sum(row["sequence"] == sequence for row in output)
                         for sequence in SEQUENCES},
        "followup_frames": args.followup_frames, "matching": "per-frame Hungarian IoU>=0.5",
        "exact_legal_edge_masks_available": False,
        "same_state_M_candidate_pairs_available": False,
        "attempt_lineage": {
            "attempt1": "INVALID: mislabeled trace base_first_pairs (proposals['G']) as baseline M",
            "attempt2": "authoritative; pairs are explicitly G-vs-applied and separate M trajectory is follow-up only"
        },
        "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
        "output": str(args.output), "output_sha256": sha256(args.output),
        "source_sha256": hashes,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
