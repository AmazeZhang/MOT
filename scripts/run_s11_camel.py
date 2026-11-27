#!/usr/bin/env python3
"""Execute S11 zero-retrain low-threshold calibration, 149-onset anatomy, and conflict-gated anchor probes."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb as tl_wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_s09_camel as s09  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s11_camel_onset_and_conflict_gate_20261127.yaml"
REPORT_DIR = REPO_ROOT / "reports/feasibility/s11_camel_onset_and_conflict_gate_20261127"
RUNS_ROOT = DATA_ROOT / "runs/s11_camel_onset_and_conflict_gate_20261127"

S09_REPORT_DIR = REPO_ROOT / "reports/diagnostics/s09_camel_calibration_and_attribution_20261126"
S09_RUNS_ROOT = DATA_ROOT / "runs/s09_camel_calibration_and_attribution_20261126"
S09_T_TWOSTAGE_CKPT = S09_RUNS_ROOT / "c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt"
S10_REPORT_DIR = REPO_ROOT / "reports/feasibility/s10_camel_anticontamination_20261127"


def load_dev8_gt_boxes(tracking_set: Any) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[str, int]]]:
    """Build read-only GT (person_ids, ltwh) and (seq, frame_1based) lookups indexed by image_id."""
    gt_by_image_id: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for img_id, grp in tracking_set.detections_gt.groupby("image_id"):
        gids = grp["person_id"].to_numpy(dtype=int)
        gboxes = np.stack(grp["bbox_ltwh"].to_list()).astype(float)
        gt_by_image_id[int(img_id)] = (gids, gboxes)
    vid_to_name = {int(k): str(v) for k, v in tracking_set.video_metadatas["name"].to_dict().items()}
    img_to_seq_fr = {
        int(idx): (vid_to_name[int(row["video_id"])], int(row["frame"]) + 1)
        for idx, row in tracking_set.image_metadatas.iterrows()
    }
    return gt_by_image_id, img_to_seq_fr


def compute_pairwise_iou_ltwh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute [A, B] pairwise IoU matrix for LTWH float arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    ax1 = boxes_a[:, 0:1]
    ay1 = boxes_a[:, 1:2]
    ax2 = ax1 + np.maximum(0.0, boxes_a[:, 2:3])
    ay2 = ay1 + np.maximum(0.0, boxes_a[:, 3:4])
    area_a = np.maximum(0.0, boxes_a[:, 2:3]) * np.maximum(0.0, boxes_a[:, 3:4])

    bx1 = boxes_b[None, :, 0]
    by1 = boxes_b[None, :, 1]
    bx2 = bx1 + np.maximum(0.0, boxes_b[None, :, 2])
    by2 = by1 + np.maximum(0.0, boxes_b[None, :, 3])
    area_b = np.maximum(0.0, boxes_b[None, :, 2]) * np.maximum(0.0, boxes_b[None, :, 3])

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


def compute_frame_conflict_signals(
    tracklets: list[Any],
    detections: list[Any],
    td_sim_matrix: Any,
) -> dict[str, Any]:
    """Compute purely GT-free pairwise IoU contention arrays and helper accessors for td_sim_matrix."""
    det_boxes = np.asarray(
        [d.bbox_ltwh.detach().cpu().numpy().astype(np.float64) for d in detections],
        dtype=np.float64,
    ).reshape(len(detections), 4)
    det_iou = compute_pairwise_iou_ltwh(det_boxes, det_boxes)
    if len(detections) > 0:
        np.fill_diagonal(det_iou, 0.0)
        det_max_other_iou = det_iou.max(axis=1)
    else:
        det_max_other_iou = np.empty((0,), dtype=np.float64)

    active_mask = np.array([getattr(t, "state", "init") == "active" for t in tracklets], dtype=bool)
    trk_boxes = np.asarray(
        [t.last_detection.bbox_ltwh.detach().cpu().numpy().astype(np.float64) for t in tracklets],
        dtype=np.float64,
    ).reshape(len(tracklets), 4)
    trk_iou = compute_pairwise_iou_ltwh(trk_boxes, trk_boxes)
    if len(tracklets) > 0:
        np.fill_diagonal(trk_iou, 0.0)
        # Only compare against other active tracklets per S11 active-track conflict definition
        masked_cols = np.where(active_mask[None, :], trk_iou, 0.0)
        trk_max_other_iou = masked_cols.max(axis=1)
    else:
        trk_max_other_iou = np.empty((0,), dtype=np.float64)

    if torch.is_tensor(td_sim_matrix) and td_sim_matrix.numel() > 0:
        sim_np = td_sim_matrix[: len(tracklets), : len(detections)].detach().cpu().numpy().astype(np.float64)
    else:
        sim_np = np.zeros((len(tracklets), len(detections)), dtype=np.float64)

    return {
        "det_max_other_iou": det_max_other_iou,
        "trk_max_other_iou": trk_max_other_iou,
        "sim_np": sim_np,
    }


def compute_two_sided_margin(sim_np: np.ndarray, trk_idx: int, det_idx: int) -> tuple[float, float, float]:
    """Return (assignment_margin, best_other_trk_sim, best_other_det_sim) strictly excluding (trk_idx, det_idx)."""
    sim_val = float(sim_np[trk_idx, det_idx])
    T, D = sim_np.shape
    if T > 1:
        col = sim_np[:, det_idx]
        best_other_trk = float(max(np.max(col[:trk_idx]) if trk_idx > 0 else -np.inf, np.max(col[trk_idx + 1 :]) if trk_idx + 1 < T else -np.inf))
    else:
        best_other_trk = -np.inf
    if D > 1:
        row = sim_np[trk_idx, :]
        best_other_det = float(max(np.max(row[:det_idx]) if det_idx > 0 else -np.inf, np.max(row[det_idx + 1 :]) if det_idx + 1 < D else -np.inf))
    else:
        best_other_det = -np.inf
    best_comp = max(best_other_trk, best_other_det)
    margin = float("inf") if best_comp == -np.inf else (sim_val - best_comp)
    return margin, best_other_trk, best_other_det


def normalize_part_embeddings(det: Any) -> Optional[np.ndarray]:
    """Extract L2-normalized [P, D] (6, 128) part embeddings from a Detection object."""
    emb = getattr(det, "embeddings", None)
    if emb is None:
        return None
    arr = emb.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(emb) else np.asarray(emb, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def get_part_visibility(det: Any, num_parts: int) -> np.ndarray:
    """Extract per-part visibility weights [P], defaulting to ones when absent."""
    vis = getattr(det, "visibility_scores", None)
    if vis is None:
        return np.ones((num_parts,), dtype=np.float64)
    arr = vis.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(vis) else np.asarray(vis, dtype=np.float64)
    return arr.reshape(-1)[:num_parts]


def part_cosine_distance(
    norm_emb_a: Optional[np.ndarray],
    norm_emb_b: Optional[np.ndarray],
    vis_a: Optional[np.ndarray] = None,
    vis_b: Optional[np.ndarray] = None,
) -> float:
    """Visibility-weighted normalized part-averaged cosine distance in [0, 2] (1 - cosine)."""
    if norm_emb_a is None or norm_emb_b is None:
        return 0.0
    cos_p = np.sum(norm_emb_a * norm_emb_b, axis=-1)
    if vis_a is not None and vis_b is not None:
        w = np.sqrt(np.clip(vis_a, 0.0, 1.0) * np.clip(vis_b, 0.0, 1.0))
        if float(w.sum()) > 1e-6:
            return float(1.0 - np.sum(w * cos_p) / np.sum(w))
    return float(1.0 - np.mean(cos_p))


def renorm_part_prototype(emb_list: list[np.ndarray]) -> np.ndarray:
    """Average normalized part vectors and re-normalize each part to unit length."""
    stacked = np.mean(np.stack(emb_list, axis=0), axis=0)
    norms = np.linalg.norm(stacked, axis=-1, keepdims=True)
    return stacked / np.maximum(norms, 1e-8)


def detect_two_cycles(
    prev_frame_pairs: list[tuple[int, int]],
    curr_pairs: list[tuple[int, int]],
) -> set[int]:
    """Identify track_ids involved in mutual A<->B identity exchange on current frame or t-1 -> t."""
    swapped_tids: set[int] = set()
    # 1. Same-frame mutual exchange: trk1(anchor A -> det GT B) & trk2(anchor B -> det GT A) with A != B
    # Tuple layout passed in: (track_id, anchor_gt, det_gt)
    pair_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tid, agt, dgt in curr_pairs:
        if agt is not None and dgt is not None and agt != dgt:
            pair_map[(agt, dgt)].append(tid)
    for (a, b), tids in pair_map.items():
        if (b, a) in pair_map:
            swapped_tids.update(tids)
            swapped_tids.update(pair_map[(b, a)])

    # 2. Temporal t-1 -> t swap: at t-1 trk1 had det_gt A and trk2 had det_gt B, at t they swapped det_gts
    prev_by_tid = {tid: prev_gt for tid, prev_gt in prev_frame_pairs if prev_gt is not None}
    curr_by_tid = {tid: dgt for tid, _agt, dgt in curr_pairs if dgt is not None}
    prev_gt_to_tids: dict[int, list[int]] = defaultdict(list)
    for tid, pgt in prev_by_tid.items():
        prev_gt_to_tids[pgt].append(tid)
    for tid1, d1 in curr_by_tid.items():
        p1 = prev_by_tid.get(tid1)
        if p1 is None or d1 == p1:
            continue
        for tid2 in prev_gt_to_tids.get(d1, []):
            if tid2 != tid1 and curr_by_tid.get(tid2) == p1:
                swapped_tids.add(tid1)
                swapped_tids.add(tid2)
    return swapped_tids


def evaluate_gate_condition(
    condition: str,
    sim_val: float,
    det_max_other_iou: float,
    trk_max_other_iou: float,
    assignment_margin: float,
) -> tuple[bool, bool]:
    """Return (should_skip_write, has_conflict) per frozen S11 gate definitions."""
    spatial_conflict = (det_max_other_iou >= 0.30) or (trk_max_other_iou >= 0.30)
    margin_conflict = assignment_margin < 0.10
    if condition == "G1_CONFLICT_IOU030_TAU065":
        return (spatial_conflict and sim_val < 0.65), spatial_conflict
    if condition == "G2_MARGIN_M010_TAU070":
        return (margin_conflict and sim_val < 0.70), margin_conflict
    if condition in ("G3_HYBRID_GATE_NO_PIN", "G4_HYBRID_GATE_EARLY_PIN5"):
        hybrid_conflict = spatial_conflict or margin_conflict
        return (hybrid_conflict and sim_val < 0.68), hybrid_conflict
    raise ValueError(f"Unknown gate condition: {condition}")


def append_gallery_with_pinning(
    trk: Any,
    det: Any,
    *,
    early_pin_k: int,
    has_conflict: bool,
) -> bool:
    """Append detection to trk.detections, pinning up to K earliest mature unconflicted detections."""
    if early_pin_k > 0 and int(trk.hits) >= 3 and (not has_conflict):
        pinned_count = sum(1 for d in trk.detections if getattr(d, "s11_pinned", False))
        if pinned_count < early_pin_k:
            det.s11_pinned = True
    trk.detections.append(det)
    while len(trk.detections) > int(trk.max_gallery_size):
        evict_idx = next((idx for idx, d in enumerate(trk.detections) if not getattr(d, "s11_pinned", False)), 0)
        trk.detections.pop(evict_idx)
    return bool(getattr(det, "s11_pinned", False))


class ConflictGatedCAMELTrack:
    """Zero-retrain GT-free Stage D wrapper implementing G1/G2/G3/G4 conflict-gated gallery writes."""

    def __init__(self, base_mod: Any, condition: str, min_hits_to_gate: int = 3, early_pin_k: int = 0) -> None:
        self.base = base_mod
        self.condition = condition
        self.min_hits_to_gate = int(min_hits_to_gate)
        self.early_pin_k = int(early_pin_k) if condition == "G4_HYBRID_GATE_EARLY_PIN5" else 0
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn
        self.gate_stats = {
            "total_frames": 0,
            "total_predicted_matches": 0,
            "mature_matches": 0,
            "conflict_flagged_mature": 0,
            "write_skipped": 0,
            "write_accepted": 0,
            "pinned_slots_assigned": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame | list[Any]:
        self.gate_stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(
            self.base.tracklets, detections
        )
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.base.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        signals = compute_frame_conflict_signals(self.base.tracklets, detections, td_sim_matrix)
        sim_np = signals["sim_np"]
        det_other_iou = signals["det_max_other_iou"]
        trk_other_iou = signals["trk_max_other_iou"]

        for m in matched:
            trk_idx, det_idx = int(m[0]), int(m[1])
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            sim_val = float(sim_np[trk_idx, det_idx])
            margin, _, _ = compute_two_sided_margin(sim_np, trk_idx, det_idx)
            d_iou = float(det_other_iou[det_idx])
            t_iou = float(trk_other_iou[trk_idx])

            should_skip, has_conflict = evaluate_gate_condition(self.condition, sim_val, d_iou, t_iou, margin)
            self.gate_stats["total_predicted_matches"] += 1
            if trk.hits >= self.min_hits_to_gate:
                self.gate_stats["mature_matches"] += 1
                if has_conflict:
                    self.gate_stats["conflict_flagged_mature"] += 1

            if trk.hits >= self.min_hits_to_gate and should_skip:
                trk.hits += 1
                trk.hit_streak += 1
                trk.time_wo_hits = 0
                trk.last_detection = det
                self.gate_stats["write_skipped"] += 1
            else:
                was_pinned = append_gallery_with_pinning(trk, det, early_pin_k=self.early_pin_k, has_conflict=has_conflict)
                trk.hits += 1
                trk.hit_streak += 1
                trk.time_wo_hits = 0
                trk.last_detection = det
                self.gate_stats["write_accepted"] += 1
                if was_pinned:
                    self.gate_stats["pinned_slots_assigned"] += 1

            det.similarity_with_tracklet = td_sim_matrix[trk_idx, det_idx]
            det.similarities = td_sim_matrix[: len(self.base.tracklets), det_idx]

        for i in unmatched_dets:
            if detections[int(i)].bbox_conf >= self.base.min_init_det_conf:
                self.base.tracklets.append(Tracklet(detections[int(i)], self.base.max_track_gallery_size))

        actives = []
        for trk in self.base.tracklets:
            self.base.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                actives.append(
                    {
                        "index": trk.last_detection.index.item(),
                        "track_id": trk.id,
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().item())
                        if trk.last_detection.similarity_with_tracklet is not None
                        else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {
                                self.base.tracklets[j].id: sim
                                for j, sim in enumerate(trk.last_detection.similarities.cpu().numpy())
                            }
                            if trk.last_detection.similarities is not None
                            else None,
                            "St": self.base.CAMEL.sim_threshold,
                        },
                    }
                )

        self.base.tracklets = [trk for trk in self.base.tracklets if trk.state != "dead"]
        self.base.frame_count += 1
        if actives:
            results = pd.DataFrame(actives).set_index("index", drop=True)
            assert set(results.index).issubset(detections_df.index)
            return results
        return []


class OnsetAnatomyReplayCAMELTrack:
    """Stage C read-only T_TWOSTAGE_TH050 anatomy probe: exact online updates + offline onset/anchor telemetry."""

    def __init__(
        self,
        base_mod: Any,
        gt_by_image_id: dict[int, tuple[np.ndarray, np.ndarray]],
        img_to_seq_fr: dict[int, tuple[str, int]],
    ) -> None:
        self.base = base_mod
        self.gt_by_image_id = gt_by_image_id
        self.img_to_seq_fr = img_to_seq_fr
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn

        self.events: list[dict[str, Any]] = []
        self.seen_cross_tracks: set[tuple[str, int]] = set()
        self.prev_frame_pairs_by_seq: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.last_gt_seen_frame: dict[tuple[str, int], int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame | list[Any]:
        for track in self.base.tracklets:
            track.forward()

        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(
            self.base.tracklets, detections
        )
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.base.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        signals = compute_frame_conflict_signals(self.base.tracklets, detections, td_sim_matrix)
        sim_np = signals["sim_np"]
        det_other_iou = signals["det_max_other_iou"]
        trk_other_iou = signals["trk_max_other_iou"]

        img_id = int(metadatas.index[0]) if hasattr(metadatas, "index") else int(detections[0].image_id.item())
        seq, fr = self.img_to_seq_fr[img_id]
        gt_ids_f, gt_boxes_f = self.gt_by_image_id.get(img_id, (np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)))
        det_boxes = [d.bbox_ltwh.detach().cpu().numpy().astype(float) for d in detections]
        det_to_gt, gt_to_det = s09.match_detections_to_gt_frame(det_boxes, gt_ids_f, gt_boxes_f, iou_thresh=0.5)

        curr_pairs: list[tuple[int, int, int]] = []
        matched_records: list[dict[str, Any]] = []

        for m in matched:
            trk_idx, det_idx = int(m[0]), int(m[1])
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            sim_val = float(sim_np[trk_idx, det_idx])
            margin, best_other_trk, best_other_det = compute_two_sided_margin(sim_np, trk_idx, det_idx)
            d_iou = float(det_other_iou[det_idx])
            t_iou = float(trk_other_iou[trk_idx])
            max_conflict_iou = max(d_iou, t_iou)

            # GT-free early clean anchor ReID cosine distance vs recent-5 sliding cosine distance
            det_norm_emb = normalize_part_embeddings(det)
            det_vis = get_part_visibility(det, det_norm_emb.shape[0]) if det_norm_emb is not None else None
            anchor_proto = getattr(trk, "s11_anchor_proto", None)
            anchor_vis = getattr(trk, "s11_anchor_vis", None)
            if anchor_proto is None:
                b_emb = normalize_part_embeddings(trk.detections[0])
                anchor_proto = b_emb
                anchor_vis = get_part_visibility(trk.detections[0], b_emb.shape[0]) if b_emb is not None else None
                trk.s11_anchor_embs = [b_emb] if b_emb is not None else []
                trk.s11_anchor_viss = [anchor_vis] if anchor_vis is not None else []
                trk.s11_anchor_proto = anchor_proto
                trk.s11_anchor_vis = anchor_vis
            anchor_reid_dist = part_cosine_distance(det_norm_emb, anchor_proto, det_vis, anchor_vis)

            rec_embs = [normalize_part_embeddings(d) for d in trk.detections[-5:]]
            rec_embs = [e for e in rec_embs if e is not None]
            if rec_embs and det_norm_emb is not None:
                rec_proto = renorm_part_prototype(rec_embs)
                recent_reid_dist = part_cosine_distance(det_norm_emb, rec_proto)
            else:
                recent_reid_dist = 0.0

            anchor_gt = getattr(trk, "s11_anchor_gt", None)
            det_gt = det_to_gt.get(det_idx)
            curr_pairs.append((int(trk.id), anchor_gt, det_gt))

            if anchor_gt is not None and det_gt is not None and det_gt == anchor_gt:
                label = "clean_match"
                sublabel = "clean_match"
            elif anchor_gt is not None and det_gt is not None and det_gt != anchor_gt:
                label = "cross_id_pollution"
                tkey = (seq, int(trk.id))
                if tkey not in self.seen_cross_tracks:
                    self.seen_cross_tracks.add(tkey)
                    sublabel = "onset_cross"
                else:
                    sublabel = "post_switch_cross"
            elif anchor_gt is not None and det_gt is None:
                label = "fp_pollution"
                sublabel = "fp_pollution"
            else:
                label = "unanchored"
                sublabel = "unanchored"

            prev_trk_match_fr = getattr(trk, "s11_prev_match_frame", fr - 1)
            trk_gap_frames = fr - prev_trk_match_fr
            gt_prev_fr = self.last_gt_seen_frame.get((seq, anchor_gt)) if anchor_gt is not None else None
            gt_gap_frames = (fr - gt_prev_fr) if gt_prev_fr is not None else trk_gap_frames

            true_det_idx = gt_to_det.get(anchor_gt) if anchor_gt is not None else None
            sim_to_true_det = float(sim_np[trk_idx, true_det_idx]) if true_det_idx is not None else None
            pos_minus_assigned = (sim_to_true_det - sim_val) if sim_to_true_det is not None else None

            rec = {
                "sequence": seq,
                "frame": fr,
                "track_id": int(trk.id),
                "det_index": int(det.index.item()),
                "anchor_gt": "" if anchor_gt is None else str(anchor_gt),
                "det_gt": "" if det_gt is None else str(det_gt),
                "label": label,
                "sublabel": sublabel,
                "hits_before": int(trk.hits),
                "gallery_len_before": len(trk.detections),
                "anchor_slots_count": len(getattr(trk, "s11_anchor_embs", [])),
                "sim_val": sim_val,
                "best_other_trk_sim": "" if math.isinf(best_other_trk) else best_other_trk,
                "best_other_det_sim": "" if math.isinf(best_other_det) else best_other_det,
                "assignment_margin": 99.0 if math.isinf(margin) else margin,
                "is_margin_inf": int(math.isinf(margin)),
                "det_max_other_iou": d_iou,
                "trk_max_other_iou": t_iou,
                "max_conflict_iou": max_conflict_iou,
                "anchor_reid_dist": anchor_reid_dist,
                "recent_reid_dist": recent_reid_dist,
                "trk_gap_frames": trk_gap_frames,
                "anchor_gt_gap_frames": gt_gap_frames,
                "true_det_present_in_frame": int(true_det_idx is not None),
                "sim_to_true_det": "" if sim_to_true_det is None else sim_to_true_det,
                "true_minus_assigned_sim": "" if pos_minus_assigned is None else pos_minus_assigned,
                "is_2cycle_swap": 0,  # backfilled below once all matches in frame are inspected
            }
            matched_records.append(rec)

            # Update GT-free online anchor prototype BEFORE changing gallery, only on unconflicted early slots (K<=5)
            has_conflict_gtfree = (max_conflict_iou >= 0.30) or (margin < 0.10)
            if det_norm_emb is not None and not has_conflict_gtfree and len(trk.s11_anchor_embs) < 5:
                trk.s11_anchor_embs.append(det_norm_emb)
                if det_vis is not None:
                    trk.s11_anchor_viss.append(det_vis)
                trk.s11_anchor_proto = renorm_part_prototype(trk.s11_anchor_embs)
                if trk.s11_anchor_viss:
                    trk.s11_anchor_vis = np.mean(np.stack(trk.s11_anchor_viss, axis=0), axis=0)

            # Standard unmodified CAMELTrack update
            trk.update(det)
            trk.s11_prev_match_frame = fr
            det.similarity_with_tracklet = td_sim_matrix[trk_idx, det_idx]
            det.similarities = td_sim_matrix[: len(self.base.tracklets), det_idx]

        swapped_tids = detect_two_cycles(self.prev_frame_pairs_by_seq[seq], curr_pairs)
        for rec in matched_records:
            if rec["track_id"] in swapped_tids:
                rec["is_2cycle_swap"] = 1
        self.events.extend(matched_records)
        self.prev_frame_pairs_by_seq[seq] = [(tid, dgt) for tid, _agt, dgt in curr_pairs if dgt is not None]
        for gid in gt_to_det.keys():
            self.last_gt_seen_frame[(seq, int(gid))] = fr

        for i in unmatched_dets:
            det = detections[int(i)]
            if det.bbox_conf >= self.base.min_init_det_conf:
                trk = Tracklet(det, self.base.max_track_gallery_size)
                trk.s11_anchor_gt = det_to_gt.get(int(i))
                trk.s11_prev_match_frame = fr
                b_emb = normalize_part_embeddings(det)
                if b_emb is not None:
                    b_vis = get_part_visibility(det, b_emb.shape[0])
                    trk.s11_anchor_embs = [b_emb]
                    trk.s11_anchor_viss = [b_vis]
                    trk.s11_anchor_proto = b_emb
                    trk.s11_anchor_vis = b_vis
                self.base.tracklets.append(trk)

        actives = []
        for trk in self.base.tracklets:
            self.base.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                actives.append(
                    {
                        "index": trk.last_detection.index.item(),
                        "track_id": trk.id,
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().item())
                        if trk.last_detection.similarity_with_tracklet is not None
                        else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {
                                self.base.tracklets[j].id: sim
                                for j, sim in enumerate(trk.last_detection.similarities.cpu().numpy())
                            }
                            if trk.last_detection.similarities is not None
                            else None,
                            "St": self.base.CAMEL.sim_threshold,
                        },
                    }
                )

        self.base.tracklets = [trk for trk in self.base.tracklets if trk.state != "dead"]
        self.base.frame_count += 1
        if actives:
            results = pd.DataFrame(actives).set_index("index", drop=True)
            assert set(results.index).issubset(detections_df.index)
            return results
        return []


def score_detector(y_true: np.ndarray, scores: np.ndarray, thresh: Optional[float] = None) -> dict[str, Any]:
    """Compute AUROC, AUPRC, TPR at 5% clean FPR, and optional F1 at preset threshold."""
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"count": int(len(y_true)), "pos": int(y_true.sum()), "auroc": None}
    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    valid_fpr = np.where(fpr <= 0.05)[0]
    idx5 = int(valid_fpr[-1]) if len(valid_fpr) > 0 else 0
    out: dict[str, Any] = {
        "count": int(len(y_true)),
        "pos": int(y_true.sum()),
        "neg": int((1 - y_true).sum()),
        "auroc": auroc,
        "auprc": auprc,
        "tpr_at_clean_fpr_5pct": float(tpr[idx5]),
        "actual_clean_fpr_at_5pct_cut": float(fpr[idx5]),
        "threshold_at_clean_fpr_5pct": float(thresholds[idx5]),
        "pos_mean": float(scores[y_true == 1].mean()),
        "neg_mean": float(scores[y_true == 0].mean()),
    }
    if thresh is not None:
        pred = scores >= thresh
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out["preset_threshold"] = thresh
        out["f1_at_preset"] = float(2 * prec * rec / max(prec + rec, 1e-12))
        out["recall_at_preset"] = float(rec)
        out["clean_fpr_at_preset"] = float(fp / max(int((y_true == 0).sum()), 1))
    return out


def cmd_stage_a_lock(_args: argparse.Namespace) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=False)
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    train32, dev8 = s09.load_split_lists()

    s09_audit = json.loads((S09_REPORT_DIR / "source_audit.json").read_text())
    s09_c2_sum = json.loads((S09_RUNS_ROOT / "c2_twostage_joint_seed13_e20_v1/stage_c2_summary.json").read_text())
    expected = {
        "t_twostage_joint_ckpt": (
            S09_T_TWOSTAGE_CKPT,
            protocol["reused_assets"]["t_twostage_joint_ckpt"]["sha256"],
        ),
        "states_dev8_gt_free": (
            s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
            s09_audit["reused_s08_assets_verified"]["states_dev8_gt_free"]["sha256"],
        ),
        "split_manifest": (s09.SPLIT_PATH, protocol["data"]["membership_sha256"]),
        "s09_protocol": (
            REPO_ROOT / "manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml",
            protocol["reused_assets"]["s09_protocol_sha256"],
        ),
        "s10_protocol": (
            REPO_ROOT / "manifests/protocols/s10_camel_anticontamination_20261127.yaml",
            protocol["reused_assets"]["s10_protocol_sha256"],
        ),
    }
    verified_assets = {}
    for k, (path, exp_sha) in expected.items():
        actual = s09.sha256_file(path)
        if actual != exp_sha:
            raise RuntimeError(f"Asset SHA256 mismatch for {k} ({path}): {actual} != {exp_sha}")
        verified_assets[k] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size, "verified": True}
    assert s09_c2_sum["last_checkpoint"]["sha256"] == verified_assets["t_twostage_joint_ckpt"]["sha256"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dir = REPORT_DIR / "resolved_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for th_label, th_val in [("010", 0.10), ("030", 0.30), ("040", 0.40), ("050", 0.50)]:
        cfg = s09.compose_s09_infer_config(
            run_dir=RUNS_ROOT / f"_resolved_th{th_label}",
            train_vids=train32,
            val_vids=dev8,
            val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
            checkpoint_path=S09_T_TWOSTAGE_CKPT,
            sim_threshold=th_val,
        )
        (cfg_dir / f"infer_th{th_label}.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    source_audit = {
        "schema_version": "s11-source-audit-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "upstream_cameltrack_commit": "bc1c261075b0beacfe71c2896453be83643153c8",
        "verified_assets": verified_assets,
        "s09_c2_summary_ckpt_sha256_verified": s09_c2_sum["last_checkpoint"]["sha256"],
        "embedding_and_anchor_dist_resolution": (
            "states_dev8_gt_free.pklz stores per-detection `embeddings` of shape (6, 128) float32 (unnormalized; part L2 norms ~5.8..24.7) "
            "and `visibility_scores` of shape (6,) float32 in [0, 1]. `anchor_reid_dist` L2-normalizes each of the 6 part vectors, averages up to K=5 "
            "earliest unconflicted (det/trk IoU < 0.30 and margin >= 0.10) detections per tracklet, re-normalizes per part, and computes visibility-weighted "
            "part-averaged cosine distance `1 - sum(w_p * cos_p) / sum(w_p)` with w_p = sqrt(vis_a_p * vis_b_p)."
        ),
        "trk_max_other_iou_definition": "Max LTWH IoU against `t2.last_detection.bbox_ltwh` over other `state == 'active'` tracklets in the current frame.",
    }
    (REPORT_DIR / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")

    runtime_lock = {
        "schema_version": "s11-runtime-lock-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_safety_probe": gpu_info,
        "scripts_sha256": {
            "scripts/run_s11_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s11_camel.py"),
            "scripts/run_s09_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s09_camel.py"),
            "tests/test_s11_camel.py": s09.sha256_file(REPO_ROOT / "tests/test_s11_camel.py"),
        },
    }
    (REPORT_DIR / "runtime_lock.json").write_text(json.dumps(runtime_lock, indent=2) + "\n")

    cmds = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
S11=/home/imc/yzy/MOT/scripts/run_s11_camel.py
CKPT={S09_T_TWOSTAGE_CKPT}

# Stage A
CUDA_VISIBLE_DEVICES='' $PY $S11 stage-a-lock
CUDA_VISIBLE_DEVICES='' $PY -m pytest /home/imc/yzy/MOT/tests/test_s11_camel.py -v

# Stage B: lower-threshold calibration sweep
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-th --condition T2_TH010 --sim-threshold 0.10 --ckpt $CKPT --run-id b_t2_th010_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-th --condition T2_TH030 --sim-threshold 0.30 --ckpt $CKPT --run-id b_t2_th030_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-th --condition T2_TH040 --sim-threshold 0.40 --ckpt $CKPT --run-id b_t2_th040_v1

# Stage C: 149-onset anatomy & anchor drift probe on T_TWOSTAGE_TH050
CUDA_VISIBLE_DEVICES=1 $PY $S11 stage-c-onset-anatomy --ckpt $CKPT --run-id c_onset_anatomy_v1

# Stage D: zero-retrain conflict-gated & early-pin5 closed-loop evals (sim_threshold=0.50)
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G1_CONFLICT_IOU030_TAU065 --ckpt $CKPT --run-id d_g1_conflict_iou030_tau065_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G2_MARGIN_M010_TAU070 --ckpt $CKPT --run-id d_g2_margin_m010_tau070_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G3_HYBRID_GATE_NO_PIN --ckpt $CKPT --run-id d_g3_hybrid_gate_no_pin_v1
CUDA_VISIBLE_DEVICES=1 $PY $S11 eval-conflict-gate --condition G4_HYBRID_GATE_EARLY_PIN5 --ckpt $CKPT --run-id d_g4_hybrid_gate_early_pin5_v1
"""
    cmd_path = REPORT_DIR / "commands.sh"
    cmd_path.write_text(cmds)
    cmd_path.chmod(0o755)
    print(json.dumps({"status": "PASS", "verified_assets": len(verified_assets)}, indent=2))


def _run_closed_loop_eval(
    *,
    run_id: str,
    condition: str,
    sim_threshold: float,
    ckpt_path: Path,
    wrap_fn: Any,
    extra_fn: Any,
) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    s09.setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    ckpt_sha = s09.sha256_file(ckpt_path)
    train32, dev8 = s09.load_split_lists()
    t0 = time.perf_counter()
    cfg = s09.compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        sim_threshold=sim_threshold,
    )
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    override_cfg = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    override_cfg["sim_threshold"] = float(sim_threshold)
    override_cfg["use_computed_sim_threshold"] = False
    base_mod.CAMEL = type(base_mod.CAMEL).load_from_checkpoint(ckpt_path, map_location=base_mod.device, **override_cfg)
    assert abs(float(base_mod.CAMEL.sim_threshold) - float(sim_threshold)) < 1e-12
    assert bool(base_mod.CAMEL.use_computed_sim_threshold) is False

    wrapped_mod = wrap_fn(base_mod)
    Tracklet.count = 1
    pipeline = Pipeline(models=[wrapped_mod])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    torch.cuda.synchronize()
    t_track0 = time.perf_counter()
    tracking_engine.track_dataset()
    torch.cuda.synchronize()
    t_track1 = time.perf_counter()
    peak_vram_mib = float(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    mot_dir = run_dir / "mot_pred"
    tracking_dataset.save_for_eval(
        tracker_state.detections_pred,
        tracker_state.image_metadatas,
        tracker_state.video_metadatas,
        str(mot_dir),
        "bbox_ltwh",
        False,
        is_ground_truth=False,
    )
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
    evaluator.run(tracker_state)
    t1 = time.perf_counter()

    detailed_csv = run_dir / "eval/results/tracklab/pedestrian_detailed.csv"
    with detailed_csv.open(newline="") as f:
        rows = {r["seq"]: r for r in csv.DictReader(f)}
    assert set(rows.keys()) == (set(dev8) | {"COMBINED"})

    extra = extra_fn(wrapped_mod)
    summary = {
        "schema_version": "s11-eval-v1",
        "status": "COMPLETED",
        "run_id": run_id,
        "condition": condition,
        "inference_mode": "online_closed_loop",
        "gt_role": "zero_gt_accessed_gt_free_states",
        "seed": "13",
        "scope_id": "dev8",
        "sim_threshold": float(sim_threshold),
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "state_file": str(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "state_file_sha256": s09.sha256_file(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "wall_seconds": t1 - t0,
        "tracking_seconds": t_track1 - t_track0,
        "peak_vram_mib": peak_vram_mib,
        "mot_dir": str(mot_dir),
        "mot_sha256_by_seq": {s: s09.sha256_file(mot_dir / f"{s}.txt") for s in dev8},
        "eval_summary": {
            "detailed_csv": str(detailed_csv),
            "summary_txt": str(run_dir / "eval/results/tracklab/pedestrian_summary.txt"),
            "gt_dir": str(run_dir / "eval/gt/DanceTrack-val"),
            "pred_dir": str(run_dir / "eval/pred/DanceTrack-val/tracklab"),
        },
        "combined_metrics_preview": {
            "HOTA": float(rows["COMBINED"]["HOTA___AUC"]) * 100.0,
            "DetA": float(rows["COMBINED"]["DetA___AUC"]) * 100.0,
            "AssA": float(rows["COMBINED"]["AssA___AUC"]) * 100.0,
            "IDF1": float(rows["COMBINED"]["IDF1"]) * 100.0,
            "MOTA": float(rows["COMBINED"]["MOTA"]) * 100.0,
            "IDSW": int(float(rows["COMBINED"]["IDSW"])),
            "FP": int(float(rows["COMBINED"]["CLR_FP"])),
            "FN": int(float(rows["COMBINED"]["CLR_FN"])),
            "Frag": int(float(rows["COMBINED"]["Frag"])),
        },
        **extra,
    }
    (run_dir / "stage_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "condition": condition, "combined": summary["combined_metrics_preview"], **extra}, indent=2))


def cmd_eval_th(args: argparse.Namespace) -> None:
    _run_closed_loop_eval(
        run_id=args.run_id,
        condition=args.condition,
        sim_threshold=float(args.sim_threshold),
        ckpt_path=Path(args.ckpt),
        wrap_fn=lambda m: m,
        extra_fn=lambda _m: {},
    )


def cmd_eval_conflict_gate(args: argparse.Namespace) -> None:
    _run_closed_loop_eval(
        run_id=args.run_id,
        condition=args.condition,
        sim_threshold=0.50,
        ckpt_path=Path(args.ckpt),
        wrap_fn=lambda m: ConflictGatedCAMELTrack(m, args.condition, min_hits_to_gate=3, early_pin_k=5 if args.condition == "G4_HYBRID_GATE_EARLY_PIN5" else 0),
        extra_fn=lambda mod: {"gate_stats": mod.gate_stats},
    )


def cmd_stage_c_onset_anatomy(args: argparse.Namespace) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    s09.setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    ckpt_path = Path(args.ckpt)
    ckpt_sha = s09.sha256_file(ckpt_path)
    train32, dev8 = s09.load_split_lists()
    t0 = time.perf_counter()

    cfg = s09.compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        sim_threshold=0.50,
    )
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    ov = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    ov["sim_threshold"] = 0.50
    ov["use_computed_sim_threshold"] = False
    base_mod.CAMEL = type(base_mod.CAMEL).load_from_checkpoint(ckpt_path, map_location=base_mod.device, **ov)

    gt_by_image_id, img_to_seq_fr = load_dev8_gt_boxes(tracking_set)
    probe = OnsetAnatomyReplayCAMELTrack(base_mod, gt_by_image_id, img_to_seq_fr)
    Tracklet.count = 1
    pipeline = Pipeline(models=[probe])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    torch.cuda.synchronize()
    tracking_engine.track_dataset()
    torch.cuda.synchronize()

    mot_dir = run_dir / "mot_pred"
    tracking_dataset.save_for_eval(
        tracker_state.detections_pred,
        tracker_state.image_metadatas,
        tracker_state.video_metadatas,
        str(mot_dir),
        "bbox_ltwh",
        False,
        is_ground_truth=False,
    )
    # Zero-GT-leakage verification against S09 authoritative T_TWOSTAGE_TH050 output
    s09_ref_summary = json.loads((S09_RUNS_ROOT / "c3_t_twostage_th050_v1/stage_eval_summary.json").read_text())
    replay_hashes = {s: s09.sha256_file(mot_dir / f"{s}.txt") for s in dev8}
    exact_mot_match = all(replay_hashes[s] == s09_ref_summary["mot_sha256_by_seq"][s] for s in dev8)
    assert exact_mot_match, "Zero-GT-leakage check failed: Stage C MOT output differs from S09 T_TWOSTAGE_TH050!"

    t1 = time.perf_counter()

    events_csv = run_dir / "c_onset_anatomy_events.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(probe.events[0].keys()))
        w.writeheader()
        w.writerows(probe.events)

    clean_ev = [e for e in probe.events if e["sublabel"] == "clean_match"]
    onset_ev = [e for e in probe.events if e["sublabel"] == "onset_cross"]
    post_ev = [e for e in probe.events if e["sublabel"] == "post_switch_cross"]
    fp_ev = [e for e in probe.events if e["sublabel"] == "fp_pollution"]
    cross_all = onset_ev + post_ev
    any_poll = cross_all + fp_ev

    # 100% onset mechanism anatomy
    n_on = len(onset_ev)
    gap_gt_1_trk = sum(1 for e in onset_ev if int(e["trk_gap_frames"]) > 1)
    gap_gt_1_gt = sum(1 for e in onset_ev if int(e["anchor_gt_gap_frames"]) > 1)
    comp_iou_ge_030 = sum(1 for e in onset_ev if float(e["max_conflict_iou"]) >= 0.30)
    det_iou_ge_030 = sum(1 for e in onset_ev if float(e["det_max_other_iou"]) >= 0.30)
    trk_iou_ge_030 = sum(1 for e in onset_ev if float(e["trk_max_other_iou"]) >= 0.30)
    two_cycle_cnt = sum(1 for e in onset_ev if int(e["is_2cycle_swap"]) == 1)
    margin_lt_010 = sum(1 for e in onset_ev if float(e["assignment_margin"]) < 0.10)
    # Split onset by GT true-detection availability & reversal magnitude
    true_present = [e for e in onset_ev if int(e["true_det_present_in_frame"]) == 1]
    true_absent = [e for e in onset_ev if int(e["true_det_present_in_frame"]) == 0]
    rev_strong = [e for e in true_present if float(e["true_minus_assigned_sim"]) < -0.05]
    rev_tie = [e for e in true_present if abs(float(e["true_minus_assigned_sim"])) <= 0.05]
    rev_hungarian_loss = [e for e in true_present if float(e["true_minus_assigned_sim"]) > 0.05]

    per_seq_onset: dict[str, int] = defaultdict(int)
    for e in onset_ev:
        per_seq_onset[e["sequence"]] += 1

    # Offline predictor AUROCs across slices
    def eval_predictors(pos_list: list[dict[str, Any]], neg_list: list[dict[str, Any]]) -> dict[str, Any]:
        sub = neg_list + pos_list
        y = np.array([0] * len(neg_list) + [1] * len(pos_list), dtype=int)
        s_1msim = np.array([1.0 - float(e["sim_val"]) for e in sub], dtype=np.float64)
        s_neg_margin = np.array([-float(e["assignment_margin"]) for e in sub], dtype=np.float64)
        s_iou = np.array([float(e["max_conflict_iou"]) for e in sub], dtype=np.float64)
        s_anchor = np.array([float(e["anchor_reid_dist"]) for e in sub], dtype=np.float64)
        s_recent = np.array([float(e["recent_reid_dist"]) for e in sub], dtype=np.float64)
        return {
            "one_minus_sim": score_detector(y, s_1msim, thresh=0.40),
            "neg_assignment_margin": score_detector(y, s_neg_margin, thresh=-0.10),
            "max_conflict_iou": score_detector(y, s_iou, thresh=0.30),
            "anchor_reid_dist": score_detector(y, s_anchor),
            "recent_reid_dist": score_detector(y, s_recent),
        }

    # Verify anchor_reid_dist gain conditioned on hits >= 3 (mature tracklets where anchor != det itself)
    clean_mat = [e for e in clean_ev if int(e["hits_before"]) >= 3]
    onset_mat = [e for e in onset_ev if int(e["hits_before"]) >= 3]
    post_mat = [e for e in post_ev if int(e["hits_before"]) >= 3]
    cross_mat = [e for e in cross_all if int(e["hits_before"]) >= 3]

    summary = {
        "schema_version": "s11-onset-anatomy-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "checkpoint_sha256": ckpt_sha,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "exact_mot_sha256_match_vs_s09_t_twostage_th050": exact_mot_match,
        "mot_sha256_by_seq": replay_hashes,
        "sublabel_counts": {
            "clean_match": len(clean_ev),
            "onset_cross": len(onset_ev),
            "post_switch_cross": len(post_ev),
            "fp_pollution": len(fp_ev),
            "unanchored": sum(1 for e in probe.events if e["sublabel"] == "unanchored"),
            "total_events": len(probe.events),
        },
        "onset_anatomy": {
            "total_onset": n_on,
            "per_sequence_counts": {s: per_seq_onset.get(s, 0) for s in dev8},
            "trk_gap_gt_1": gap_gt_1_trk,
            "trk_gap_eq_1": n_on - gap_gt_1_trk,
            "anchor_gt_gap_gt_1": gap_gt_1_gt,
            "competing_iou_ge_030_any": comp_iou_ge_030,
            "det_max_other_iou_ge_030": det_iou_ge_030,
            "trk_max_other_iou_ge_030": trk_iou_ge_030,
            "two_cycle_mutual_swap": two_cycle_cnt,
            "assignment_margin_lt_010": margin_lt_010,
            "sim_lt_065": sum(1 for e in onset_ev if float(e["sim_val"]) < 0.65),
            "hybrid_g3_condition_triggered_at_onset": sum(
                1 for e in onset_ev if (float(e["max_conflict_iou"]) >= 0.30 or float(e["assignment_margin"]) < 0.10) and float(e["sim_val"]) < 0.68
            ),
            "true_det_present_in_frame": len(true_present),
            "true_det_absent_in_frame": len(true_absent),
            "true_present_strong_reversal_lt_neg005": len(rev_strong),
            "true_present_tie_abs_le_005": len(rev_tie),
            "true_present_hungarian_higher_pos_gt_005": len(rev_hungarian_loss),
            "median_sim_val": float(np.median([float(e["sim_val"]) for e in onset_ev])) if onset_ev else None,
            "median_assignment_margin_finite": float(np.median([float(e["assignment_margin"]) for e in onset_ev if int(e["is_margin_inf"]) == 0])) if onset_ev else None,
            "median_max_conflict_iou": float(np.median([float(e["max_conflict_iou"]) for e in onset_ev])) if onset_ev else None,
        },
        "offline_detector_comparisons": {
            "onset_vs_clean_all": eval_predictors(onset_ev, clean_ev),
            "post_switch_vs_clean_all": eval_predictors(post_ev, clean_ev),
            "all_cross_id_vs_clean_all": eval_predictors(cross_all, clean_ev),
            "any_pollution_vs_clean_all": eval_predictors(any_poll, clean_ev),
            "onset_vs_clean_mature_ge3": eval_predictors(onset_mat, clean_mat),
            "post_switch_vs_clean_mature_ge3": eval_predictors(post_mat, clean_mat),
            "all_cross_id_vs_clean_mature_ge3": eval_predictors(cross_mat, clean_mat),
        },
        "events_csv": str(events_csv),
    }
    (run_dir / "stage_c_anatomy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "exact_mot_match": exact_mot_match,
        "sublabel_counts": summary["sublabel_counts"],
        "onset_anatomy": summary["onset_anatomy"],
        "anchor_auroc_post_switch": summary["offline_detector_comparisons"]["post_switch_vs_clean_all"]["anchor_reid_dist"]["auroc"],
        "recent_auroc_post_switch": summary["offline_detector_comparisons"]["post_switch_vs_clean_all"]["recent_reid_dist"]["auroc"],
        "anchor_auroc_all_cross": summary["offline_detector_comparisons"]["all_cross_id_vs_clean_all"]["anchor_reid_dist"]["auroc"],
        "anchor_auroc_onset": summary["offline_detector_comparisons"]["onset_vs_clean_all"]["anchor_reid_dist"]["auroc"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stage-a-lock")

    p_th = sub.add_parser("eval-th")
    p_th.add_argument("--condition", required=True)
    p_th.add_argument("--sim-threshold", type=float, required=True)
    p_th.add_argument("--ckpt", required=True)
    p_th.add_argument("--run-id", required=True)

    p_sc = sub.add_parser("stage-c-onset-anatomy")
    p_sc.add_argument("--ckpt", required=True)
    p_sc.add_argument("--run-id", required=True)

    p_cg = sub.add_parser("eval-conflict-gate")
    p_cg.add_argument(
        "--condition",
        choices=[
            "G1_CONFLICT_IOU030_TAU065",
            "G2_MARGIN_M010_TAU070",
            "G3_HYBRID_GATE_NO_PIN",
            "G4_HYBRID_GATE_EARLY_PIN5",
        ],
        required=True,
    )
    p_cg.add_argument("--ckpt", required=True)
    p_cg.add_argument("--run-id", required=True)

    args = parser.parse_args()
    dispatch = {
        "stage-a-lock": cmd_stage_a_lock,
        "eval-th": cmd_eval_th,
        "stage-c-onset-anatomy": cmd_stage_c_onset_anatomy,
        "eval-conflict-gate": cmd_eval_conflict_gate,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
