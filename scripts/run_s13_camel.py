#!/usr/bin/env python3
"""S13 runner: Stage A audits, Stage B M1 calibration & 0024/0053 attribution, Stage C N1/N2/N3, Stage D synergy & anatomy."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path("/home/imc/yzy/MOT")
THIRD_PARTY_CAMEL = Path("/media/imc/data/mot-data/third_party/CAMELTrack")
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(THIRD_PARTY_CAMEL) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_CAMEL))

import run_s09_camel as s09  # noqa: E402
import run_s11_camel as s11  # noqa: E402
import run_s12_camel as s12  # noqa: E402
from cameltrack.camel import Detections, Tracklets  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402
from cameltrack.train.callbacks import SimMetrics  # noqa: E402
from tracklab.datastruct import TrackerState  # noqa: E402
from tracklab.pipeline import Pipeline  # noqa: E402
from tracklab.utils import wandb as tl_wandb  # noqa: E402

DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s13_camel_calibration_and_dual_memory_20261128.yaml"
S12_PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s12_camel_pairwise_edge_and_assignment_20261127.yaml"
RUNS_ROOT = DATA_ROOT / "runs/s13_camel_calibration_and_dual_memory_20261128"
REPORT_DIR = REPO_ROOT / "reports/feasibility/s13_camel_calibration_and_dual_memory_20261128"
SHARED_DM_CACHE = DATA_ROOT / "cache/s12_camel/dm_s12_train32_dev8_shared"

S12_M1_CKPT = DATA_ROOT / "runs/s12_camel_pairwise_edge_and_assignment_20261127/c_m1_kin_edge_seed13_e20_v1/CAMEL/last.ckpt"
S12_M1_CKPT_SHA = "b3ca74bf5426729736cf641770f7f372ae4903a54b1a1485f778309a8ae94314"

S13_CONDITIONS: dict[str, dict[str, Any]] = {
    "N1_KIN_SCALE_ANCHORED": {
        "arch_variant": "n1_kin_scale_anchored",
        "lambda_center": 0.50,
        "p_occ_drop": 0.0,
        "occ_conflict_iou": 0.25,
        "tau_occ": 0.35,
        "beta": 10.0,
        "lambda_occ": 0.0,
        "early_anchor_k": 0,
        "p_hist_pollute": 0.0,
    },
    "N2_KIN_OCC_DROP_SUP": {
        "arch_variant": "n2_kin_occ_drop_sup",
        "lambda_center": 0.50,
        "p_occ_drop": 0.25,
        "occ_conflict_iou": 0.25,
        "tau_occ": 0.35,
        "beta": 10.0,
        "lambda_occ": 0.15,
        "early_anchor_k": 0,
        "p_hist_pollute": 0.0,
    },
    "N3_DUAL_ANCHOR_CAMEL": {
        "arch_variant": "n3_dual_anchor_camel",
        "lambda_center": 0.50,
        "p_occ_drop": 0.25,
        "occ_conflict_iou": 0.25,
        "tau_occ": 0.35,
        "beta": 10.0,
        "lambda_occ": 0.15,
        "early_anchor_k": 4,
        "recent_slots_for_proto": 8,
        "p_hist_pollute": 0.25,
    },
}


def compute_val_derived_threshold(val_th: float) -> float:
    """Zero-leakage threshold calibration anchored at Stage C val_th - 0.14 clipped to [0.35, 0.65]."""
    return float(min(0.65, max(0.35, round(float(val_th) - 0.14, 2))))


def _aggregate_part_proto(embs: torch.Tensor, vis: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute visibility-weighted L2-normalized part prototype over K slots.

    embs: [..., K, P, C], vis: [..., K, P], valid_mask: [..., K]
    Returns (proto [..., P, C], proto_vis [..., P], has_any [...])
    """
    dtype = embs.dtype
    m = valid_mask.unsqueeze(-1).to(dtype)  # [..., K, 1]
    v = vis.clamp(0.0, 1.0) * m
    norm_e = F.normalize(torch.nan_to_num(embs, nan=0.0), p=2, dim=-1, eps=1e-6)
    v_sum = v.sum(dim=-2)  # [..., P]
    # Fallback to uniform weight across valid slots for a part if its visibility sum is ~0
    slot_count = m.sum(dim=-2).clamp(min=1.0)  # [..., 1]
    weighted = (norm_e * v.unsqueeze(-1)).sum(dim=-3)  # [..., P, C]
    unweighted = (norm_e * m.unsqueeze(-1)).sum(dim=-3)
    raw_proto = torch.where(v_sum.unsqueeze(-1) > 1e-6, weighted, unweighted)
    proto = F.normalize(raw_proto, p=2, dim=-1, eps=1e-6)
    proto_vis = (v.sum(dim=-2) / slot_count).clamp(0.0, 1.0)
    has_any = valid_mask.any(dim=-1)
    return proto, proto_vis, has_any


def part_weighted_cosine_sim_matrix(
    proto_a: torch.Tensor,
    vis_a: torch.Tensor,
    valid_a: torch.Tensor,
    proto_b: torch.Tensor,
    vis_b: torch.Tensor,
    valid_b: torch.Tensor,
) -> torch.Tensor:
    """Pairwise [B, A, D_dim] part-visibility-weighted cosine similarity in [-1, 1], 0 when invalid."""
    # proto_a: [B, A, P, C], proto_b: [B, D_dim, P, C]
    cos_p = torch.einsum("bapc,bdpc->badp", proto_a, proto_b).clamp(-1.0, 1.0)
    w_p = torch.sqrt(vis_a.unsqueeze(2).clamp(0.0, 1.0) * vis_b.unsqueeze(1).clamp(0.0, 1.0))  # [B, A, D_dim, P]
    w_sum = w_p.sum(dim=-1)
    sim = torch.where(w_sum > 1e-6, (w_p * cos_p).sum(dim=-1) / w_sum.clamp(min=1e-6), cos_p.mean(dim=-1))
    pair_ok = valid_a.unsqueeze(2) & valid_b.unsqueeze(1)
    return torch.where(pair_ok, torch.nan_to_num(sim, 0.0).clamp(-1.0, 1.0), torch.zeros_like(sim))


def extract_dual_anchor_slots(
    tracks: Tracklets,
    k_anchor: int = 4,
    k_recent: int = 8,
    anchor_override: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select oldest<=k_anchor valid slots (largest age) and newest<=k_recent valid slots (smallest age)."""
    embs = tracks.feats["embeddings"]  # [B, T, S, P, C]
    vis = tracks.feats["visibility_scores"]  # [B, T, S, P]
    drop_app = tracks.feats["drop_app"][..., 0] > 0.5 if "drop_app" in tracks.feats else torch.zeros_like(tracks.feats_masks)
    valid = tracks.feats_masks & (~drop_app) & torch.isfinite(embs).all(dim=-1).all(dim=-1)
    B, T, S, P, C = embs.shape
    device = embs.device

    # Valid slots in CAMELDataset and Tracklet.padded_features are sorted age ascending:
    # smallest index -> newest (recent), largest valid index -> oldest (anchor).
    ages = torch.nan_to_num(tracks.feats["age"][..., 0], nan=0.0)
    rec_embs = torch.zeros((B, T, k_recent, P, C), dtype=embs.dtype, device=device)
    rec_vis = torch.zeros((B, T, k_recent, P), dtype=vis.dtype, device=device)
    rec_mask = torch.zeros((B, T, k_recent), dtype=torch.bool, device=device)

    anc_embs = torch.zeros((B, T, k_anchor, P, C), dtype=embs.dtype, device=device)
    anc_vis = torch.zeros((B, T, k_anchor, P), dtype=vis.dtype, device=device)
    anc_mask = torch.zeros((B, T, k_anchor), dtype=torch.bool, device=device)

    for b in range(B):
        for t in range(T):
            idx = torch.where(valid[b, t])[0]
            if len(idx) == 0:
                continue
            # Explicitly sort by age ascending to be invariant to upstream quirks
            idx = idx[torch.argsort(ages[b, t, idx], descending=False)]
            r_idx = idx[:k_recent]
            a_idx = idx[-k_anchor:]
            rec_embs[b, t, : len(r_idx)] = embs[b, t, r_idx]
            rec_vis[b, t, : len(r_idx)] = vis[b, t, r_idx]
            rec_mask[b, t, : len(r_idx)] = True
            anc_embs[b, t, : len(a_idx)] = embs[b, t, a_idx]
            anc_vis[b, t, : len(a_idx)] = vis[b, t, a_idx]
            anc_mask[b, t, : len(a_idx)] = True

    if anchor_override is not None:
        ov_e = anchor_override["embeddings"].to(device=device, dtype=embs.dtype)
        ov_v = anchor_override["visibility_scores"].to(device=device, dtype=vis.dtype)
        ov_m = anchor_override["masks"].to(device=device, dtype=torch.bool)
        has_ov = ov_m.any(dim=-1, keepdim=True)
        anc_embs = torch.where(has_ov.unsqueeze(-1).unsqueeze(-1), ov_e, anc_embs)
        anc_vis = torch.where(has_ov.unsqueeze(-1), ov_v, anc_vis)
        anc_mask = torch.where(has_ov, ov_m, anc_mask)

    return anc_embs, anc_vis, anc_mask, rec_embs, rec_vis, rec_mask


def compute_dual_anchor_reid_features(
    tracks: Tracklets,
    dets: Detections,
    k_anchor: int = 4,
    k_recent: int = 8,
    anchor_override: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute [B, T, D, 4] causal early-anchor consistency edge features:
    [sim_anchor_reid(i,j), sim_recent_reid(i,j), sim_anchor_reid - sim_recent_reid, anchor_recent_drift(i)].
    """
    anc_e, anc_v, anc_m, rec_e, rec_v, rec_m = extract_dual_anchor_slots(tracks, k_anchor=k_anchor, k_recent=k_recent, anchor_override=anchor_override)
    proto_anc, vis_anc, has_anc = _aggregate_part_proto(anc_e, anc_v, anc_m)  # [B, T, P, C]
    proto_rec, vis_rec, has_rec = _aggregate_part_proto(rec_e, rec_v, rec_m)  # [B, T, P, C]

    det_e = dets.feats["embeddings"]  # [B, D, 1, P, C]
    det_v = dets.feats["visibility_scores"]  # [B, D, 1, P]
    det_drop = dets.feats["drop_app"][..., 0] > 0.5 if "drop_app" in dets.feats else torch.zeros(det_e.shape[:3], dtype=torch.bool, device=det_e.device)
    det_valid = dets.feats_masks & (~det_drop) & torch.isfinite(det_e).all(dim=-1).all(dim=-1)
    proto_det, vis_det, has_det = _aggregate_part_proto(det_e, det_v, det_valid)  # [B, D, P, C]

    sim_anc = part_weighted_cosine_sim_matrix(proto_anc, vis_anc, has_anc, proto_det, vis_det, has_det)
    sim_rec = part_weighted_cosine_sim_matrix(proto_rec, vis_rec, has_rec, proto_det, vis_det, has_det)
    sim_diff = sim_anc - sim_rec

    # anchor_recent_drift(i) = 1 - cos(anchor_proto_i, recent_proto_i)
    cos_ar = torch.sum(proto_anc * proto_rec, dim=-1).clamp(-1.0, 1.0)
    w_ar = torch.sqrt(vis_anc.clamp(0.0, 1.0) * vis_rec.clamp(0.0, 1.0))
    w_ar_sum = w_ar.sum(dim=-1)
    sim_ar = torch.where(w_ar_sum > 1e-6, (w_ar * cos_ar).sum(dim=-1) / w_ar_sum.clamp(min=1e-6), cos_ar.mean(dim=-1))
    drift_i = torch.where(has_anc & has_rec, (1.0 - sim_ar).clamp(0.0, 2.0), torch.zeros_like(sim_ar))
    B, T, D = sim_anc.shape
    drift_td = drift_i.unsqueeze(2).expand(B, T, D)

    e_anc = torch.stack([sim_anc, sim_rec, sim_diff, drift_td], dim=-1)
    pair_ok = tracks.masks.unsqueeze(2) & dets.masks.unsqueeze(1)
    return torch.where(pair_ok.unsqueeze(-1), torch.nan_to_num(e_anc, 0.0), torch.zeros_like(e_anc))


def _box_iou_matrix_ltwh(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Tensorized [A, B] IoU matrix for LTWH tensors."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=boxes_a.dtype, device=boxes_a.device)
    ax1, ay1, aw, ah = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3].clamp(min=0.0), boxes_a[:, 3:4].clamp(min=0.0)
    bx1, by1, bw, bh = boxes_b[:, 0:1].T, boxes_b[:, 1:2].T, boxes_b[:, 2:3].T.clamp(min=0.0), boxes_b[:, 3:4].T.clamp(min=0.0)
    ix1, iy1 = torch.maximum(ax1, bx1), torch.maximum(ay1, by1)
    ix2, iy2 = torch.minimum(ax1 + aw, bx1 + bw), torch.minimum(ay1 + ah, by1 + bh)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
    union = aw * ah + bw * bh - inter
    return torch.where(union > 1e-8, inter / union.clamp(min=1e-8), torch.zeros_like(inter))


def inject_recent_history_pollution_(
    tracks: Tracklets,
    dets: Detections,
    *,
    p_pollute: float = 0.25,
    k_anchor_keep: int = 4,
    min_valid_slots: int = 6,
    recent_range: tuple[int, int] = (2, 4),
) -> int:
    """In-place training-only recent-window contamination on valid tracks with >=6 valid slots.
    Strictly preserves the oldest k_anchor_keep=4 slots untouched and only replaces the newest 2..4 slots
    with spatially nearest non-same-ID detection/recent-track features in the same sample.
    """
    embs = tracks.feats["embeddings"]  # [B, T, S, P, C]
    vis = tracks.feats["visibility_scores"]
    t_boxes = tracks.feats["bbox_ltwh"]  # [B, T, S, 4]
    ages = torch.nan_to_num(tracks.feats["age"][..., 0], nan=0.0)
    B, T, S = tracks.feats_masks.shape
    t_valid = tracks.masks  # [B, T]
    t_tgts = tracks.targets if tracks.targets.ndim == 2 else tracks.targets[:, :, 0]
    d_valid = dets.masks if dets.masks.ndim == 2 else dets.masks[:, :, 0]
    d_tgts = dets.targets if dets.targets.ndim == 2 else dets.targets[:, :, 0]
    d_boxes = dets.feats["bbox_ltwh"][:, :, 0]  # [B, D, 4]
    d_embs = dets.feats["embeddings"][:, :, 0]
    d_vis = dets.feats["visibility_scores"][:, :, 0]

    n_polluted = 0
    for b in range(B):
        valid_t_indices = torch.where(t_valid[b] & (t_tgts[b] >= 0))[0]
        valid_d_indices = torch.where(d_valid[b] & (d_tgts[b] >= 0))[0]
        if len(valid_t_indices) < 2 and len(valid_d_indices) < 2:
            continue
        for t in valid_t_indices.tolist():
            slots = torch.where(tracks.feats_masks[b, t])[0]
            if len(slots) < min_valid_slots:
                continue
            if float(torch.rand((), device=embs.device).item()) >= p_pollute:
                continue
            slots = slots[torch.argsort(ages[b, t, slots], descending=False)]  # newest at index 0
            max_pol = min(recent_range[1], len(slots) - k_anchor_keep)
            if max_pol < recent_range[0]:
                continue
            k_pol = int(torch.randint(recent_range[0], max_pol + 1, (), device=embs.device).item())
            target_slots = slots[:k_pol]  # strictly newest slots; oldest >=k_anchor_keep remain untouched!

            cur_id = t_tgts[b, t]
            cur_box = t_boxes[b, t, slots[0]]
            if not torch.isfinite(cur_box).all():
                continue

            # Find spatially closest different-ID detection in same sample (fallback to other valid track)
            other_d = valid_d_indices[d_tgts[b, valid_d_indices] != cur_id]
            donor_emb = None
            donor_vis = None
            if len(other_d) > 0:
                cand_boxes = d_boxes[b, other_d]
                finite_ok = torch.isfinite(cand_boxes).all(dim=-1)
                if finite_ok.any():
                    cand_boxes = cand_boxes[finite_ok]
                    other_d = other_d[finite_ok]
                    cur_c = torch.stack([cur_box[0] + 0.5 * cur_box[2], cur_box[1] + 0.5 * cur_box[3]])
                    cand_c = torch.stack([cand_boxes[:, 0] + 0.5 * cand_boxes[:, 2], cand_boxes[:, 1] + 0.5 * cand_boxes[:, 3]], dim=-1)
                    nn_idx = int(torch.argmin(((cand_c - cur_c) ** 2).sum(dim=-1)).item())
                    d_pick = int(other_d[nn_idx].item())
                    donor_emb = d_embs[b, d_pick].clone()
                    donor_vis = d_vis[b, d_pick].clone()
            if donor_emb is None:
                other_t = valid_t_indices[t_tgts[b, valid_t_indices] != cur_id]
                if len(other_t) == 0:
                    continue
                donor_t = int(other_t[0].item())
                d_slots = torch.where(tracks.feats_masks[b, donor_t])[0]
                if len(d_slots) == 0:
                    continue
                d_s = int(d_slots[torch.argmin(ages[b, donor_t, d_slots])].item())
                donor_emb = embs[b, donor_t, d_s].clone()
                donor_vis = vis[b, donor_t, d_s].clone()

            if torch.isfinite(donor_emb).all() and torch.isfinite(donor_vis).all():
                embs[b, t, target_slots] = donor_emb.unsqueeze(0)
                vis[b, t, target_slots] = donor_vis.unsqueeze(0)
                if "drop_app" in tracks.feats:
                    tracks.feats["drop_app"][b, t, target_slots] = 0.0
                n_polluted += 1
    return n_polluted


def apply_occlusion_drop_(
    tracks: Tracklets,
    dets: Detections,
    *,
    p_occ_drop: float = 0.25,
    occ_conflict_iou: float = 0.25,
) -> list[tuple[int, int, torch.Tensor]]:
    """Training-only targeted detection drop for occluded-conflict tracks.
    Only masks the sampled track's true positive detection d^+ without modifying other IDs in the frame,
    leaving remaining detections intact so L_occ penalizes high similarity to IoU>=occ_conflict_iou distractors.
    Returns list of (b, t_idx, orig_pos_box_ltwh) for sampled occluded tracks.
    """
    t_valid = tracks.masks  # [B, T]
    t_tgts = tracks.targets if tracks.targets.ndim == 2 else tracks.targets[:, :, 0]
    d_masks_ref = dets.masks if dets.masks.ndim == 2 else dets.masks[:, :, 0]
    d_tgts_ref = dets.targets if dets.targets.ndim == 2 else dets.targets[:, :, 0]
    d_boxes = dets.feats["bbox_ltwh"][:, :, 0]  # [B, D, 4]
    B, T = t_valid.shape

    dropped_records: list[tuple[int, int, int, torch.Tensor]] = []
    for b in range(B):
        valid_d = torch.where(d_masks_ref[b] & (d_tgts_ref[b] >= 0))[0]
        if len(valid_d) < 2:
            continue
        ious = _box_iou_matrix_ltwh(d_boxes[b, valid_d], d_boxes[b, valid_d])
        ious.fill_diagonal_(0.0)
        max_other_iou = ious.max(dim=1).values  # [len(valid_d)]
        det_iou_lookup = {int(valid_d[idx].item()): float(max_other_iou[idx].item()) for idx in range(len(valid_d))}

        for t in torch.where(t_valid[b] & (t_tgts[b] >= 0))[0].tolist():
            tid = t_tgts[b, t]
            pos_ds = torch.where(d_masks_ref[b] & (d_tgts_ref[b] == tid))[0]
            if len(pos_ds) != 1:
                continue
            d_plus = int(pos_ds[0].item())
            if det_iou_lookup.get(d_plus, 0.0) < occ_conflict_iou:
                continue
            # Ensure >=1 overlapping valid distractor detection still survives if we drop d_plus
            orig_box = d_boxes[b, d_plus].clone()
            rem_others = torch.where(d_masks_ref[b])[0]
            rem_others = rem_others[rem_others != d_plus]
            if len(rem_others) == 0:
                continue
            rem_ious = _box_iou_matrix_ltwh(orig_box.unsqueeze(0), d_boxes[b, rem_others]).squeeze(0)
            if not (rem_ious >= occ_conflict_iou).any():
                continue
            if float(torch.rand((), device=d_boxes.device).item()) >= p_occ_drop:
                continue
            d_masks_ref[b, d_plus] = False
            dets.feats_masks[b, d_plus, 0] = False
            if dets.targets.ndim == 3:
                dets.targets[b, d_plus, 0] = float("nan")
            else:
                dets.targets[b, d_plus] = float("nan")
            dropped_records.append((b, t, d_plus, orig_box))

    # Return (b, t, orig_pos_box) — L_occ will compute overlaps against REMAINING valid dets dynamically!
    return [(b, t, box) for b, t, _, box in dropped_records]


class S13PairwiseCAMEL(s12.S12PairwiseCAMEL):
    """S13 unified model for N1_KIN_SCALE_ANCHORED, N2_KIN_OCC_DROP_SUP, N3_DUAL_ANCHOR_CAMEL."""

    def __init__(
        self,
        *args,
        arch_variant: str = "n1_kin_scale_anchored",
        lambda_center: float = 0.50,
        p_occ_drop: float = 0.0,
        occ_conflict_iou: float = 0.25,
        tau_occ: float = 0.35,
        lambda_occ: float = 0.0,
        early_anchor_k: int = 0,
        recent_slots_for_proto: int = 8,
        p_hist_pollute: float = 0.0,
        beta: float = 10.0,
        **kwargs,
    ) -> None:
        super_variant = "kin_edge"
        super().__init__(*args, arch_variant=super_variant, beta=beta, **kwargs)
        self.s13_variant = str(arch_variant)
        self.lambda_center = float(lambda_center)
        self.p_occ_drop = float(p_occ_drop)
        self.occ_conflict_iou = float(occ_conflict_iou)
        self.tau_occ = float(tau_occ)
        self.lambda_occ = float(lambda_occ)
        self.early_anchor_k = int(early_anchor_k)
        self.recent_slots_for_proto = int(recent_slots_for_proto)
        self.p_hist_pollute = float(p_hist_pollute)
        self.save_hyperparameters(ignore=[k for k in locals() if "checkpoint_path" in k])

        if self.s13_variant == "n3_dual_anchor_camel":
            n_l, n_h = int(self.gaffe.n_layers), int(self.gaffe.n_heads)
            # Extend 13D e_kin inputs to 17D = [e_kin (13D), e_anc (4D)] with zero-init on new 4D weights
            for attr, out_dim in [("edge_attn_td", n_l * n_h), ("edge_attn_dt", n_l * n_h)]:
                old_mod = getattr(self, attr)
                new_in = nn.Linear(17, 64)
                with torch.no_grad():
                    new_in.weight.zero_()
                    new_in.weight[:, :13].copy_(old_mod[0].weight)
                    new_in.bias.copy_(old_mod[0].bias)
                new_mod = nn.Sequential(new_in, nn.GELU(), s12._zero_init_linear(nn.Linear(64, out_dim)))
                setattr(self, attr, new_mod)

            old_sim = self.edge_sim_mlp
            new_sim_in = nn.Linear(17 + 128, 64)
            with torch.no_grad():
                new_sim_in.weight.zero_()
                # old_sim layout: [e_kin(0:13), zt(13:77), zd(77:141)] -> new: [e_kin(0:13), e_anc(13:17), zt(17:81), zd(81:145)]
                new_sim_in.weight[:, :13].copy_(old_sim[0].weight[:, :13])
                new_sim_in.weight[:, 17:].copy_(old_sim[0].weight[:, 13:])
                new_sim_in.bias.copy_(old_sim[0].bias)
            self.edge_sim_mlp = nn.Sequential(new_sim_in, nn.GELU(), s12._zero_init_linear(nn.Linear(64, 1)))

    def forward(self, tracks: Tracklets, dets: Detections, anchor_override: dict[str, torch.Tensor] | None = None):
        feat_dict = s12.compute_pairwise_kin_doa_features(tracks, dets)
        tracks, dets = self.tokenize(tracks, dets)
        tracks, dets = self.merge(tracks, dets)

        e_pair = feat_dict["e_kin"]
        if self.s13_variant == "n3_dual_anchor_camel":
            e_anc = compute_dual_anchor_reid_features(
                tracks,
                dets,
                k_anchor=self.early_anchor_k,
                k_recent=self.recent_slots_for_proto,
                anchor_override=anchor_override,
            )
            e_pair = torch.cat([e_pair, e_anc], dim=-1)

        B, T, D, _ = e_pair.shape
        n_l, n_h = int(self.gaffe.n_layers), int(self.gaffe.n_heads)
        pair_valid = feat_dict["pair_valid"]

        b_td = self.edge_attn_td(e_pair).view(B, T, D, n_l, n_h).permute(0, 3, 4, 1, 2)
        b_dt = self.edge_attn_dt(e_pair).view(B, T, D, n_l, n_h).permute(0, 3, 4, 2, 1)
        tracks, dets = self._run_gaffe_with_bias(tracks, dets, b_td, b_dt, None, None)
        td_sim_base = self.similarity(tracks, dets)

        zt = F.relu(self.sim_proj_t(tracks.embs)).unsqueeze(2).expand(B, T, D, 64)
        zd = F.relu(self.sim_proj_d(dets.embs)).unsqueeze(1).expand(B, T, D, 64)
        delta_s = 0.5 * torch.tanh(self.edge_sim_mlp(torch.cat([e_pair, zt, zd], dim=-1)).squeeze(-1))
        delta_s = torch.where(pair_valid, delta_s, torch.zeros_like(delta_s))
        tracks._last_delta_s = delta_s
        tracks._last_pair_valid = pair_valid
        td_sim_matrix = torch.where(td_sim_base > -1e8, td_sim_base + delta_s, td_sim_base)
        return tracks, dets, td_sim_matrix

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        anchor_override = batch.get("track_anchor_feats", None)
        tracks, dets = self.predict_preprocess(batch)
        tracks, dets, td_sim_matrix = self.forward(tracks, dets, anchor_override=anchor_override)
        sim_threshold = self.computed_sim_threshold if (self.use_computed_sim_threshold and self.computed_sim_threshold) else self.sim_threshold
        association_matrix, association_result = self.association(td_sim_matrix, tracks.masks, dets.masks, sim_threshold=sim_threshold)
        return association_matrix, association_result, td_sim_matrix

    def _compute_s13_loss(
        self,
        tracks: Tracklets,
        dets: Detections,
        td_sim: torch.Tensor,
        occ_records: list[tuple[int, int, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        base_loss = self._compute_base_sim_loss_with_delta(tracks, dets)
        delta_s = tracks._last_delta_s  # [B, T, D]
        pair_valid = tracks._last_pair_valid  # [B, T, D]

        # N1 zero-mean centering penalty on delta_s
        center_terms = []
        raw_means = []
        for b in range(delta_s.shape[0]):
            pv = pair_valid[b]
            if pv.any():
                m_b = delta_s[b][pv].mean()
                center_terms.append(m_b * m_b)
                raw_means.append(float(m_b.detach().item()))
        l_center = torch.stack(center_terms).mean() if center_terms else td_sim.new_zeros(())
        mean_abs_ds = float(np.mean(np.abs(raw_means))) if raw_means else 0.0

        # N2/N3 targeted occlusion-drop distractor rejection loss
        occ_terms = []
        if occ_records and self.lambda_occ > 0.0:
            d_masks_ref = dets.masks if dets.masks.ndim == 2 else dets.masks[:, :, 0]
            d_boxes = dets.feats["bbox_ltwh"][:, :, 0]
            for b, t_idx, orig_box in occ_records:
                rem_ds = torch.where(d_masks_ref[b])[0]
                if len(rem_ds) == 0:
                    continue
                ious = _box_iou_matrix_ltwh(orig_box.unsqueeze(0), d_boxes[b, rem_ds]).squeeze(0)
                nbr_ds = rem_ds[ious >= self.occ_conflict_iou]
                if len(nbr_ds) == 0:
                    continue
                s_max_nbr = td_sim[b, t_idx, nbr_ds].max()
                occ_terms.append(F.softplus(self.beta * (s_max_nbr - self.tau_occ)) / self.beta)

        if occ_terms:
            l_occ = torch.stack(occ_terms).mean()
        else:
            l_occ = td_sim.new_zeros(())

        total = base_loss + self.lambda_center * l_center + self.lambda_occ * l_occ
        tracks._last_base_loss = base_loss.detach()
        tracks._last_center_loss = l_center.detach()
        tracks._last_occ_loss = l_occ.detach()
        tracks._last_delta_s_abs_mean = torch.tensor(mean_abs_ds, device=td_sim.device)
        return total, base_loss, l_center, l_occ, float(len(occ_terms))

    def training_step(self, batch, batch_idx):
        tracks, dets = self.train_val_preprocess(batch)
        pollute_count = 0
        if self.p_hist_pollute > 0.0:
            pollute_count = inject_recent_history_pollution_(tracks, dets, p_pollute=self.p_hist_pollute)
        occ_records: list[tuple[int, int, torch.Tensor]] = []
        if self.p_occ_drop > 0.0:
            occ_records = apply_occlusion_drop_(tracks, dets, p_occ_drop=self.p_occ_drop, occ_conflict_iou=self.occ_conflict_iou)
        tracks, dets, td_sim = self.forward(tracks, dets)
        loss, _, _, _, occ_active = self._compute_s13_loss(tracks, dets, td_sim, occ_records)
        self.log_loss(loss, "train")
        self.log_dict(
            {
                "train/base_sim_loss": tracks._last_base_loss,
                "train/center_loss": tracks._last_center_loss,
                "train/occ_loss": tracks._last_occ_loss,
                "train/delta_s_abs_mean": tracks._last_delta_s_abs_mean,
                "train/occ_drop_count": torch.tensor(float(len(occ_records)), device=loss.device),
                "train/occ_loss_active": torch.tensor(float(occ_active), device=loss.device),
                "train/hist_pollute_count": torch.tensor(float(pollute_count), device=loss.device),
            },
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        return {"loss": loss, "tracks": tracks, "dets": dets, "td_sim_matrix": td_sim}

    def validation_step(self, batch, batch_idx):
        # Clean validation without synthetic drop or pollution for unbiased val/computed_sim_threshold!
        tracks, dets = self.train_val_preprocess(batch)
        tracks, dets, td_sim = self.forward(tracks, dets)
        loss, _, _, _, _ = self._compute_s13_loss(tracks, dets, td_sim, [])
        self.log_loss(loss, "val")
        self.log_dict(
            {
                "val/base_sim_loss": tracks._last_base_loss,
                "val/center_loss": tracks._last_center_loss,
                "val/delta_s_abs_mean": tracks._last_delta_s_abs_mean,
            },
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        return {"loss": loss, "tracks": tracks, "dets": dets, "td_sim_matrix": td_sim}


class S13CAMELTrackWrapper:
    """Wraps upstream CAMELTrack for S13: exact B1 dustbin=sim_threshold calibration & N3 permanent early anchors."""

    def __init__(self, base_mod: Any, disambig_mode: str = "NONE", early_anchor_k: int = 0) -> None:
        self.base = base_mod
        self.mode = disambig_mode
        self.early_anchor_k = int(early_anchor_k)
        self.stats = {
            "total_frames": 0,
            "total_matches_initial": 0,
            "total_matches_final": 0,
            "dustbin_rejected": 0,
            "tracks_with_early_anchors": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    def _compute_frame_other_iou(self, detections: list[Any]) -> dict[int, float]:
        if len(detections) <= 1:
            return {i: 0.0 for i in range(len(detections))}
        boxes = torch.stack([d.bbox_ltwh for d in detections], dim=0).float()
        ious = _box_iou_matrix_ltwh(boxes, boxes)
        ious.fill_diagonal_(0.0)
        mx = ious.max(dim=1).values
        return {i: float(mx[i].item()) for i in range(len(detections))}

    def _record_early_anchor(self, trk: Tracklet, det: Any, max_other_iou: float) -> None:
        if self.early_anchor_k <= 0:
            return
        if not hasattr(trk, "_early_clean_anchors"):
            trk._early_clean_anchors = []
            trk._early_fallback_candidates = []
        if len(trk._early_clean_anchors) >= self.early_anchor_k:
            return
        conf = float(det.bbox_conf.item()) if torch.is_tensor(det.bbox_conf) else float(det.bbox_conf)
        entry = (det.embeddings.clone(), det.visibility_scores.clone(), conf, max_other_iou)
        if conf >= 0.70 and max_other_iou < 0.30:
            trk._early_clean_anchors.append(entry)
        elif trk.age < 8:
            trk._early_fallback_candidates.append(entry)

    def _build_anchor_override(self, tracklets: list[Tracklet], device: Any) -> dict[str, torch.Tensor] | None:
        if self.early_anchor_k <= 0 or not tracklets:
            return None
        T = len(tracklets)
        K = self.early_anchor_k
        e0 = tracklets[0].last_detection.embeddings
        v0 = tracklets[0].last_detection.visibility_scores
        P, C = e0.shape
        out_e = torch.zeros((1, T, K, P, C), dtype=torch.float32, device=device)
        out_v = torch.zeros((1, T, K, P), dtype=torch.float32, device=device)
        out_m = torch.zeros((1, T, K), dtype=torch.bool, device=device)
        for t_i, trk in enumerate(tracklets):
            clean = list(getattr(trk, "_early_clean_anchors", []))
            if len(clean) < K:
                fb = sorted(getattr(trk, "_early_fallback_candidates", []), key=lambda x: (x[3], -x[2]))
                need = K - len(clean)
                clean = clean + fb[:need]
            if not clean:
                # Bootstrap directly from earliest FIFO detections on birth before first update
                for d in trk.detections[:K]:
                    clean.append((d.embeddings, d.visibility_scores, 1.0, 0.0))
            for k_i, item in enumerate(clean[:K]):
                out_e[0, t_i, k_i] = item[0].to(device=device)
                out_v[0, t_i, k_i] = item[1].to(device=device)
                out_m[0, t_i, k_i] = True
        return {"embeddings": out_e, "visibility_scores": out_v, "masks": out_m}

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame | list[Any]:
        self.stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        other_iou = self._compute_frame_other_iou(detections) if self.early_anchor_k > 0 else {}
        T, D = len(self.base.tracklets), len(detections)
        if T == 0:
            matched, u_trks, u_dets, td_sim = np.empty((0, 2), dtype=int), [], list(range(D)), torch.empty((0, 0))
        elif D == 0:
            matched, u_trks, u_dets, td_sim = np.empty((0, 2), dtype=int), list(range(T)), [], torch.empty((0, 0))
        else:
            batch = self.base.build_camel_batch(self.base.tracklets, detections)
            ov = self._build_anchor_override(self.base.tracklets, self.base.device)
            if ov is not None:
                batch["track_anchor_feats"] = ov
            _, ass_res, td_sim_full = self.base.CAMEL.predict_step(batch, self.base.frame_count)
            td_sim = td_sim_full.squeeze(0)
            if self.mode == "B1_DUAL_SOFTMAX":
                dustbin = float(self.base.CAMEL.sim_threshold)
                matched, u_trks, u_dets, st = s12.dual_softmax_associate(td_sim[:T, :D], tau_ds=0.10, dustbin_score=dustbin)
                self.stats["total_matches_initial"] += len(matched) + st["dustbin_rejected"]
                self.stats["dustbin_rejected"] += st["dustbin_rejected"]
            else:
                matched = ass_res[0]["matched_td_indices"]
                u_trks = ass_res[0]["unmatched_trackers"]
                u_dets = ass_res[0]["unmatched_detections"]
                self.stats["total_matches_initial"] += len(matched)

        self.stats["total_matches_final"] += len(matched)
        for m in matched:
            trk = self.base.tracklets[int(m[0])]
            det_idx = int(m[1])
            det = detections[det_idx]
            trk.update(det)
            self._record_early_anchor(trk, det, other_iou.get(det_idx, 0.0))
            det.similarity_with_tracklet = td_sim[int(m[0]), det_idx]
            det.similarities = td_sim[:T, det_idx]

        for i in u_dets:
            det = detections[int(i)]
            if det.bbox_conf >= self.base.min_init_det_conf:
                trk = Tracklet(det, self.base.max_track_gallery_size)
                self._record_early_anchor(trk, det, other_iou.get(int(i), 0.0))
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
            return results
        return []


def compose_s13_train_config(run_dir: Path, train_vids: list[str], dev8_vids: list[str], dm_cache: Path, condition: str, max_epochs: int = 20, num_workers: int = 4) -> Any:
    spec = S13_CONDITIONS[condition]
    cfg = s09.compose_s09_train_config(
        run_dir=run_dir,
        train_vids=train_vids,
        val_vids=dev8_vids,
        train_state=s09.S08_STATES_DIR / "states_train32.pklz",
        val_state=s09.S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=dm_cache,
        stage1_cue=None,
        te_checkpoints=s12.S09_C1_CKPTS,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )
    OmegaConf.set_struct(cfg.modules.track.CAMEL, False)
    cfg.modules.track.CAMEL._target_ = "run_s13_camel.S13PairwiseCAMEL"
    for k, v in spec.items():
        cfg.modules.track.CAMEL[k] = v
    OmegaConf.set_struct(cfg.modules.track.CAMEL, True)
    return cfg


def extra_param_summary(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    base_params = (
        sum(p.numel() for p in model.gaffe.parameters())
        + sum(p.numel() for p in model.temp_encs.parameters())
    )
    extra = total - base_params
    return {
        "total_params": total,
        "base_params": base_params,
        "s13_extra_params": extra,
        "fraction_of_total": float(extra / max(total, 1)),
    }


def cmd_stage_a_lock(_: argparse.Namespace) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res_dir = REPORT_DIR / "resolved_configs"
    res_dir.mkdir(parents=True, exist_ok=True)

    train32, dev8 = s09.load_split_lists()
    dm_cache = s12.ensure_shared_dm_cache()

    assert S12_M1_CKPT.is_file() and s09.sha256_file(S12_M1_CKPT) == S12_M1_CKPT_SHA
    assert s09.sha256_file(S12_PROTOCOL_PATH) == "8095ff00d557541d7a36e0a0a9ebf554a071a743594d254eb3a995a891ac571a"

    # Save resolved configs & verify parameter counts
    param_table = {}
    for cond in S13_CONDITIONS:
        cfg_c = compose_s13_train_config(res_dir / cond.lower(), train32, dev8, dm_cache, cond)
        OmegaConf.save(cfg_c, res_dir / f"stage_c_{cond.lower()}.yaml")
        m_probe = instantiate(cfg_c.modules.track.CAMEL, _recursive_=False)
        param_table[cond] = extra_param_summary(m_probe)

    assert param_table["N1_KIN_SCALE_ANCHORED"]["s13_extra_params"] == param_table["N2_KIN_OCC_DROP_SUP"]["s13_extra_params"] == 146305
    assert param_table["N3_DUAL_ANCHOR_CAMEL"]["s13_extra_params"] > 146305

    # Audit train32 DM age sorting & occluded-conflict track counts across 200 batches
    cfg_n2 = compose_s13_train_config(res_dir / "probe_n2", train32, dev8, dm_cache, "N2_KIN_OCC_DROP_SUP", num_workers=0)
    pl.seed_everything(13, workers=True)
    td = instantiate(cfg_n2.dataset)
    dm = instantiate(cfg_n2.modules.track.datamodule_cfg, tracking_dataset=td, pipeline=Pipeline([]), datasets={})
    dm.prepare_data()
    dm.setup("fit")
    dl = dm.train_dataloader()
    m_probe_n2 = instantiate(cfg_n2.modules.track.CAMEL, _recursive_=False).train()

    monotone_violations = 0
    total_valid_trks = 0
    ge6_slots_trks = 0
    occ_eligible_trks = 0
    for bi, batch in enumerate(dl):
        if bi >= 100:
            break
        tracks, dets = m_probe_n2.train_val_preprocess(batch)
        ages = tracks.feats["age"][..., 0]
        for b in range(tracks.feats_masks.shape[0]):
            for t in range(tracks.feats_masks.shape[1]):
                s_idx = torch.where(tracks.feats_masks[b, t])[0]
                if len(s_idx) == 0:
                    continue
                total_valid_trks += 1
                if len(s_idx) >= 6:
                    ge6_slots_trks += 1
                a = ages[b, t, s_idx]
                if len(a) > 1 and (a[1:] < a[:-1]).any():
                    monotone_violations += 1
        # Check occ_conflict_iou >= 0.25 without mutating stats cleanly on copy
        recs = apply_occlusion_drop_(tracks, dets, p_occ_drop=1.0, occ_conflict_iou=0.25)
        occ_eligible_trks += len(recs)

    assert monotone_violations == 0, f"age sort violation: {monotone_violations}"
    audit = {
        "schema_version": "s13-source-audit-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "upstream_cameltrack_commit": "bc1c261075b0beacfe71c2896453be83643153c8",
        "tracklet_gallery_audit": (
            "Tracklet.detections appends chronologically and slices [-max_gallery_size:] (FIFO evicts index 0). "
            "Tracklet.padded_features stacks reversed(self.detections), so index 0 is newest (smallest age) and last valid slot is oldest (largest age)."
        ),
        "dataset_age_audit": (
            "CAMELDataset assigns age = img_id - image_id and calls sort_values(by='age', ascending=True). "
            "Empirical 100-batch check on train32 DM: monotone_violations=0 across all tracklets, matching online padded_features."
        ),
        "train32_100_batch_probe": {
            "total_valid_tracklets": total_valid_trks,
            "ge6_slots_tracklets": ge6_slots_trks,
            "ge6_slots_frac": ge6_slots_trks / max(total_valid_trks, 1),
            "occ_iou_ge_025_eligible_tracklets": occ_eligible_trks,
            "occ_eligible_frac": occ_eligible_trks / max(total_valid_trks, 1),
        },
        "verified_assets": {
            "s12_m1_ckpt": {"path": str(S12_M1_CKPT), "sha256": S12_M1_CKPT_SHA, "verified": True},
            "s12_protocol": {"path": str(S12_PROTOCOL_PATH), "sha256": s09.sha256_file(S12_PROTOCOL_PATH), "verified": True},
            "c1_app_ckpt": {"path": str(s12.S09_C1_CKPTS["app_encoder"]), "sha256": s09.sha256_file(s12.S09_C1_CKPTS["app_encoder"]), "verified": True},
            "c1_kp_ckpt": {"path": str(s12.S09_C1_CKPTS["kp_encoder"]), "sha256": s09.sha256_file(s12.S09_C1_CKPTS["kp_encoder"]), "verified": True},
            "c1_bbox_ckpt": {"path": str(s12.S09_C1_CKPTS["bbox_encoder"]), "sha256": s09.sha256_file(s12.S09_C1_CKPTS["bbox_encoder"]), "verified": True},
            "states_train32": {"path": str(s09.S08_STATES_DIR / "states_train32.pklz"), "sha256": s09.sha256_file(s09.S08_STATES_DIR / "states_train32.pklz"), "verified": True},
            "states_dev8_train_ref": {"path": str(s09.S08_STATES_DIR / "states_dev8_train_ref.pklz"), "sha256": s09.sha256_file(s09.S08_STATES_DIR / "states_dev8_train_ref.pklz"), "verified": True},
            "states_dev8_gt_free": {"path": str(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"), "sha256": s09.sha256_file(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"), "verified": True},
        },
        "model_parameters": param_table,
    }
    (REPORT_DIR / "source_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lock = {
        "schema_version": "s13-runtime-lock-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_probe": gpu_info,
    }
    (REPORT_DIR / "runtime_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "param_table": param_table, "audit_probe": audit["train32_100_batch_probe"]}, indent=2))


class S13EpochLogger(s09.S09EpochLogger):
    """Epoch logger appending S13 scale-anchoring, occ-drop, and hist-pollute metrics cleanly."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if int(trainer.current_epoch) == 0 and trainer.sanity_checking:
            return
        super().on_validation_epoch_end(trainer, pl_module)
        epoch = int(trainer.current_epoch) + 1
        step = int(trainer.global_step)
        samples_seen = step * 8
        elapsed = time.perf_counter() - self.t_start
        ckpt_id = f"epoch{epoch}_last" if epoch == int(trainer.max_epochs) else ""
        metrics = trainer.callback_metrics
        extra = [
            ("train", "train/base_sim_loss", "loss"),
            ("train", "train/center_loss", "loss"),
            ("train", "train/occ_loss", "loss"),
            ("train", "train/delta_s_abs_mean", "similarity"),
            ("train", "train/occ_drop_count", "count"),
            ("train", "train/occ_loss_active", "count"),
            ("train", "train/hist_pollute_count", "count"),
            ("val", "val/base_sim_loss", "loss"),
            ("val", "val/center_loss", "loss"),
            ("val", "val/delta_s_abs_mean", "similarity"),
        ]
        added = []
        for split, k, unit in extra:
            if k in metrics:
                val = float(metrics[k].detach().cpu().item()) if torch.is_tensor(metrics[k]) else float(metrics[k])
                added.append({
                    "run_id": self.run_dir.name,
                    "stage_tag": self.stage_tag,
                    "seed": 13,
                    "epoch": epoch,
                    "step": step,
                    "split": split,
                    "metric": k,
                    "value": repr(val),
                    "unit": unit,
                    "samples_seen": samples_seen,
                    "elapsed_seconds": round(elapsed, 3),
                    "checkpoint_id": ckpt_id,
                })
        if added:
            self.rows.extend(added)
            out_csv = self.run_dir / "learning_curves.csv"
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(self.rows[0].keys())).writerows(added)


def cmd_train_c(args: argparse.Namespace) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    s09.setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    dm_cache = s12.ensure_shared_dm_cache()
    train32, dev8 = s09.load_split_lists()
    cfg = compose_s13_train_config(run_dir, train32, dev8, dm_cache, args.condition, max_epochs=20, num_workers=4)

    t0 = time.perf_counter()
    pl.seed_everything(13, workers=True)
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL

    init_prefixes = ["gaffe"] + [f"temp_encs.{k}" for k in model.temp_encs.keys()]
    new_prefixes = ["edge_attn_td", "edge_attn_dt", "sim_proj_t", "sim_proj_d", "edge_sim_mlp"]
    before = {
        prefix: {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith(prefix + ".")}
        for prefix in (init_prefixes + new_prefixes)
    }
    params_info = extra_param_summary(model)

    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})
    epoch_cb = S13EpochLogger(run_dir, args.condition, t0)
    save_best_loss = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", dirpath="CAMEL", mode="min", filename="epoch={epoch}-loss={val/loss:.4f}", auto_insert_metric_name=False, save_last=True
    )
    logger = pl.loggers.WandbLogger(project="CAMEL", offline=True, save_dir=str(run_dir))
    tr_cfg = cfg.modules.track.train_cfg.pl_trainer
    trainer = pl.Trainer(
        max_epochs=int(tr_cfg.max_epochs),
        logger=logger,
        callbacks=[save_best_loss, pl.callbacks.LearningRateMonitor(), SimMetrics(), epoch_cb],
        accelerator="cuda",
        devices=1,
        precision=int(tr_cfg.precision),
        gradient_clip_val=float(tr_cfg.gradient_clip_val),
        accumulate_grad_batches=int(tr_cfg.accumulate_grad_batches),
        num_sanity_val_steps=0,
        log_every_n_steps=int(tr_cfg.log_every_n_steps),
        check_val_every_n_epoch=int(tr_cfg.check_val_every_n_epochs),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    last_ckpt = run_dir / "CAMEL/last.ckpt"
    assert last_ckpt.is_file()
    last_sha = s09.sha256_file(last_ckpt)

    current_params = dict(model.named_parameters())
    deltas = {}
    for prefix, ref_dict in before.items():
        n_changed, max_abs, l2_sq = 0, 0.0, 0.0
        for name, ref_t in ref_dict.items():
            diff = (current_params[name].detach().cpu() - ref_t).abs()
            m = float(diff.max().item())
            if m > 0.0:
                n_changed += 1
            max_abs = max(max_abs, m)
            l2_sq += float((diff.double() ** 2).sum().item())
        deltas[prefix] = {"num_tensors": len(ref_dict), "num_changed_tensors": n_changed, "max_abs_delta": max_abs, "l2_delta": float(np.sqrt(l2_sq))}
        if len(ref_dict) > 0 and n_changed == 0:
            raise RuntimeError(f"Zero parameter updates in {prefix}")

    reloaded = type(model).load_from_checkpoint(last_ckpt, map_location="cuda")
    sd_cur, sd_rel = model.state_dict(), reloaded.state_dict()
    max_reload_diff = max(float((sd_cur[k].cpu() - sd_rel[k].cpu()).abs().max().item()) for k in sd_cur)
    assert max_reload_diff == 0.0

    epoch20_comp_th = None
    for row in reversed(epoch_cb.rows):
        if row["metric"] == "val/computed_sim_threshold":
            epoch20_comp_th = float(row["value"])
            break
    assert epoch20_comp_th is not None
    th_cal = compute_val_derived_threshold(epoch20_comp_th)

    summary = {
        "schema_version": "s13-train-v1",
        "status": "COMPLETED",
        "condition": args.condition,
        "arch_variant": S13_CONDITIONS[args.condition]["arch_variant"],
        "run_id": args.run_id,
        "seed": 13,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "gpu_device_hours": (t1 - t0) / 3600.0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "completed_epochs": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "nonfinite_grad_steps": int(epoch_cb.nonfinite_grad_steps),
        "parameters": params_info,
        "module_deltas_init_to_end": deltas,
        "last_checkpoint": {
            "path": str(last_ckpt),
            "sha256": last_sha,
            "bytes": int(last_ckpt.stat().st_size),
            "reload_state_dict_max_abs_diff": max_reload_diff,
        },
        "epoch20_val_computed_sim_threshold": epoch20_comp_th,
        "val_derived_sim_threshold": th_cal,
        "learning_curves_csv": str(run_dir / "learning_curves.csv"),
    }
    (run_dir / "stage_c_train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "condition": args.condition,
        "nonfinite": summary["nonfinite_grad_steps"],
        "reload_diff": max_reload_diff,
        "epoch20_val_th": epoch20_comp_th,
        "th_cal": th_cal,
        "last_ckpt_sha256": last_sha,
    }, indent=2))


def cmd_eval_online(args: argparse.Namespace) -> None:
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
    sim_threshold = float(args.sim_threshold)
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
    OmegaConf.set_struct(cfg.modules.track.CAMEL, False)
    if args.model_family == "s12":
        cfg.modules.track.CAMEL._target_ = "run_s12_camel.S12PairwiseCAMEL"
        camel_cls = s12.S12PairwiseCAMEL
    else:
        cfg.modules.track.CAMEL._target_ = "run_s13_camel.S13PairwiseCAMEL"
        camel_cls = S13PairwiseCAMEL
    OmegaConf.set_struct(cfg.modules.track.CAMEL, True)

    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    ov = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    ov["sim_threshold"] = sim_threshold
    ov["use_computed_sim_threshold"] = False
    base_mod.CAMEL = camel_cls.load_from_checkpoint(ckpt_path, map_location=base_mod.device, **ov)
    assert abs(float(base_mod.CAMEL.sim_threshold) - sim_threshold) < 1e-12
    assert bool(base_mod.CAMEL.use_computed_sim_threshold) is False

    early_k = int(getattr(base_mod.CAMEL, "early_anchor_k", 0))
    use_wrapper = (args.disambig != "NONE") or (early_k > 0)
    wrapped_mod = S13CAMELTrackWrapper(base_mod, disambig_mode=args.disambig, early_anchor_k=early_k) if use_wrapper else base_mod

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

    n_frames = int(float(rows["COMBINED"]["CLR_Frames"]))
    tracking_sec = t_track1 - t_track0
    summary = {
        "schema_version": "s13-eval-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "condition": args.condition,
        "inference_mode": "online_closed_loop",
        "disambig_mode": args.disambig,
        "model_family": args.model_family,
        "early_anchor_k": early_k,
        "gt_role": "zero_gt_accessed_gt_free_states",
        "seed": "13",
        "scope_id": "dev8",
        "sim_threshold": sim_threshold,
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "wall_seconds": t1 - t0,
        "tracking_seconds": tracking_sec,
        "ms_per_frame": 1000.0 * tracking_sec / max(n_frames, 1),
        "peak_vram_mib": peak_vram_mib,
        "mot_dir": str(mot_dir),
        "mot_sha256_by_seq": {s: s09.sha256_file(mot_dir / f"{s}.txt") for s in dev8},
        "eval_summary": {
            "detailed_csv": str(detailed_csv),
            "summary_txt": str(run_dir / "eval/results/tracklab/pedestrian_summary.txt"),
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
        "wrapper_stats": wrapped_mod.stats if use_wrapper else None,
    }
    (run_dir / "stage_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "condition": args.condition, "sim_threshold": sim_threshold, "ms_per_frame": round(summary["ms_per_frame"], 2), "combined": summary["combined_metrics_preview"], "wrapper_stats": summary["wrapper_stats"]}, indent=2))


class S13AnatomyReplayWrapper(S13CAMELTrackWrapper):
    """Zero-drift S11/S12-compatible onset & post-switch tail telemetry wrapper on top of S13CAMELTrackWrapper."""

    def __init__(self, base_mod: Any, disambig_mode: str, early_anchor_k: int, gt_by_image_id: dict, img_to_seq_fr: dict) -> None:
        super().__init__(base_mod, disambig_mode=disambig_mode, early_anchor_k=early_anchor_k)
        self.gt_by_image_id = gt_by_image_id
        self.img_to_seq_fr = img_to_seq_fr
        self.events: list[dict[str, Any]] = []
        self.trk_meta: dict[int, dict[str, Any]] = {}
        self.last_gt_frame: dict[int, int] = {}

    def reset(self) -> None:
        super().reset()
        self.trk_meta = {}
        self.last_gt_frame = {}

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame | list[Any]:
        # Run exact same association + wrapper logic, recording pre-update anatomy events cleanly
        self.stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        other_iou = self._compute_frame_other_iou(detections) if self.early_anchor_k > 0 else {}
        T, D = len(self.base.tracklets), len(detections)
        if T == 0:
            matched, u_trks, u_dets, td_sim = np.empty((0, 2), dtype=int), [], list(range(D)), torch.empty((0, 0))
        elif D == 0:
            matched, u_trks, u_dets, td_sim = np.empty((0, 2), dtype=int), list(range(T)), [], torch.empty((0, 0))
        else:
            batch = self.base.build_camel_batch(self.base.tracklets, detections)
            ov = self._build_anchor_override(self.base.tracklets, self.base.device)
            if ov is not None:
                batch["track_anchor_feats"] = ov
            _, ass_res, td_sim_full = self.base.CAMEL.predict_step(batch, self.base.frame_count)
            td_sim = td_sim_full.squeeze(0)
            if self.mode == "B1_DUAL_SOFTMAX":
                dustbin = float(self.base.CAMEL.sim_threshold)
                matched, u_trks, u_dets, st = s12.dual_softmax_associate(td_sim[:T, :D], tau_ds=0.10, dustbin_score=dustbin)
                self.stats["dustbin_rejected"] += st["dustbin_rejected"]
            else:
                matched = ass_res[0]["matched_td_indices"]
                u_trks = ass_res[0]["unmatched_trackers"]
                u_dets = ass_res[0]["unmatched_detections"]

        # Read-only GT attribution after association decision (identical to S11/S12)
        if len(detections) > 0:
            img_id = int(detections[0].image_id.item())
            seq, fr = self.img_to_seq_fr[img_id]
            gt_ids, gt_boxes = self.gt_by_image_id.get(img_id, (np.empty(0, dtype=int), np.zeros((0, 4))))
            det_boxes = np.stack([d.bbox_ltwh.cpu().numpy() for d in detections]).astype(float)
            det_iou_gt = s11.compute_pairwise_iou_ltwh(det_boxes, gt_boxes) if len(gt_boxes) else np.zeros((D, 0))
            det_dd_iou = s11.compute_pairwise_iou_ltwh(det_boxes, det_boxes)
            if D > 0:
                np.fill_diagonal(det_dd_iou, 0.0)
            det_gt = [
                int(gt_ids[int(np.argmax(det_iou_gt[j]))]) if (len(gt_boxes) and float(det_iou_gt[j].max()) >= 0.5) else -1
                for j in range(D)
            ]
            gt_id_to_det_idx = {}
            for j, gid in enumerate(det_gt):
                if gid != -1 and gid not in gt_id_to_det_idx:
                    gt_id_to_det_idx[gid] = j
            if T > 0:
                trk_boxes = np.stack([t.last_detection.bbox_ltwh.cpu().numpy() for t in self.base.tracklets]).astype(float)
                trk_tt_iou = s11.compute_pairwise_iou_ltwh(trk_boxes, trk_boxes)
                np.fill_diagonal(trk_tt_iou, 0.0)
                sim_np = td_sim[:T, :D].detach().cpu().numpy()
                row_best_other = np.full(T, -np.inf)
                col_best_other = np.full(D, -np.inf)
                for m in matched:
                    ti, di = int(m[0]), int(m[1])
                    if D > 1:
                        r = sim_np[ti].copy()
                        r[di] = -np.inf
                        row_best_other[ti] = float(r.max())
                    if T > 1:
                        c = sim_np[:, di].copy()
                        c[ti] = -np.inf
                        col_best_other[di] = float(c.max())
            else:
                trk_tt_iou, sim_np = np.zeros((0, 0)), np.zeros((0, 0))
                row_best_other, col_best_other = np.zeros(0), np.zeros(0)

            for m in matched:
                ti, di = int(m[0]), int(m[1])
                trk = self.base.tracklets[ti]
                det = detections[di]
                meta = self.trk_meta.setdefault(trk.id, {"anchor_gt": -1, "has_switched": False, "last_det_gt": -1})
                if meta["anchor_gt"] == -1 and det_gt[di] != -1:
                    meta["anchor_gt"] = det_gt[di]
                anchor_gt = int(meta["anchor_gt"])
                cur_gt = int(det_gt[di])
                if cur_gt == -1:
                    sublabel = "fp_pollution"
                elif anchor_gt == -1:
                    sublabel = "unanchored"
                elif cur_gt == anchor_gt:
                    sublabel = "clean_match"
                elif not meta["has_switched"]:
                    sublabel = "onset_cross"
                    meta["has_switched"] = True
                else:
                    sublabel = "post_switch_cross"

                sim_val = float(sim_np[ti, di])
                best_other = max(float(row_best_other[ti]), float(col_best_other[di]))
                margin = float(sim_val - best_other) if np.isfinite(best_other) else float("inf")
                det_max_iou = float(det_dd_iou[di].max()) if D > 1 else 0.0
                trk_max_iou = float(trk_tt_iou[ti].max()) if T > 1 else 0.0
                true_present = int(anchor_gt in gt_id_to_det_idx) if anchor_gt != -1 else 0

                self.events.append({
                    "sequence": seq,
                    "frame": fr,
                    "track_id": int(trk.id),
                    "anchor_gt": anchor_gt,
                    "det_gt": cur_gt,
                    "sublabel": sublabel,
                    "hits_before": int(trk.hits),
                    "trk_gap_frames": int(trk.time_wo_hits),
                    "sim_val": sim_val,
                    "assignment_margin": margin if np.isfinite(margin) else 10.0,
                    "max_conflict_iou": max(det_max_iou, trk_max_iou),
                    "true_det_present_in_frame": true_present,
                })
                meta["last_det_gt"] = cur_gt
            for gid in gt_id_to_det_idx:
                self.last_gt_frame[gid] = fr

        for m in matched:
            trk = self.base.tracklets[int(m[0])]
            det_idx = int(m[1])
            det = detections[det_idx]
            trk.update(det)
            self._record_early_anchor(trk, det, other_iou.get(det_idx, 0.0))
            det.similarity_with_tracklet = td_sim[int(m[0]), det_idx]
            det.similarities = td_sim[:T, det_idx]

        for i in u_dets:
            det = detections[int(i)]
            if det.bbox_conf >= self.base.min_init_det_conf:
                trk = Tracklet(det, self.base.max_track_gallery_size)
                self._record_early_anchor(trk, det, other_iou.get(int(i), 0.0))
                self.base.tracklets.append(trk)
                if len(detections) > 0:
                    img_id = int(detections[0].image_id.item())
                    gt_ids, gt_boxes = self.gt_by_image_id.get(img_id, (np.empty(0, dtype=int), np.zeros((0, 4))))
                    if len(gt_boxes):
                        d_box = det.bbox_ltwh.cpu().numpy().reshape(1, 4).astype(float)
                        ious = s11.compute_pairwise_iou_ltwh(d_box, gt_boxes)[0]
                        if float(ious.max()) >= 0.5:
                            gid = int(gt_ids[int(np.argmax(ious))])
                            self.trk_meta[trk.id] = {"anchor_gt": gid, "has_switched": False, "last_det_gt": gid}

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
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().item()) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.base.tracklets[j].id: sim for j, sim in enumerate(trk.last_detection.similarities.cpu().numpy())} if trk.last_detection.similarities is not None else None,
                            "St": self.base.CAMEL.sim_threshold,
                        },
                    }
                )
        self.base.tracklets = [trk for trk in self.base.tracklets if trk.state != "dead"]
        self.base.frame_count += 1
        if actives:
            return pd.DataFrame(actives).set_index("index", drop=True)
        return []


def cmd_anatomy_replay(args: argparse.Namespace) -> None:
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
    sim_threshold = float(args.sim_threshold)
    train32, dev8 = s09.load_split_lists()
    target_vids = [v for v in dev8 if (not args.seqs or v in args.seqs)]

    cfg = s09.compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=target_vids,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        sim_threshold=sim_threshold,
    )
    OmegaConf.set_struct(cfg.modules.track.CAMEL, False)
    if args.model_family == "s12":
        cfg.modules.track.CAMEL._target_ = "run_s12_camel.S12PairwiseCAMEL"
        camel_cls = s12.S12PairwiseCAMEL
    else:
        cfg.modules.track.CAMEL._target_ = "run_s13_camel.S13PairwiseCAMEL"
        camel_cls = S13PairwiseCAMEL
    OmegaConf.set_struct(cfg.modules.track.CAMEL, True)

    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    gt_by_image_id, img_to_seq_fr = s11.load_dev8_gt_boxes(tracking_set)

    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    ov = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    ov["sim_threshold"] = sim_threshold
    ov["use_computed_sim_threshold"] = False
    base_mod.CAMEL = camel_cls.load_from_checkpoint(ckpt_path, map_location=base_mod.device, **ov)
    early_k = int(getattr(base_mod.CAMEL, "early_anchor_k", 0))
    probe = S13AnatomyReplayWrapper(base_mod, args.disambig, early_k, gt_by_image_id, img_to_seq_fr)

    Tracklet.count = 1
    pipeline = Pipeline(models=[probe])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    t0 = time.perf_counter()
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
    ref_root = Path(args.ref_root) if args.ref_root else RUNS_ROOT
    ref_summary = json.loads((ref_root / args.ref_online_run_id / "stage_eval_summary.json").read_text())
    replay_hashes = {s: s09.sha256_file(mot_dir / f"{s}.txt") for s in target_vids}
    if len(target_vids) == len(dev8):
        exact_mot_match = all(replay_hashes[s] == ref_summary["mot_sha256_by_seq"][s] for s in target_vids)
        assert exact_mot_match, f"Anatomy replay MOT mismatch vs {args.ref_online_run_id}"
    else:
        # Subsets shift sequential Tracklet.count offset; verify exact bijection & identical (frame, x, y, w, h, conf) lines
        ref_mot_dir = Path(ref_summary["mot_dir"])
        exact_mot_match = True
        for s in target_vids:
            r_lines = [ln.strip().split(",") for ln in (ref_mot_dir / f"{s}.txt").read_text().splitlines() if ln.strip()]
            c_lines = [ln.strip().split(",") for ln in (mot_dir / f"{s}.txt").read_text().splitlines() if ln.strip()]
            assert len(r_lines) == len(c_lines), f"Line count mismatch on {s}"
            id_map: dict[str, str] = {}
            for rl, cl in zip(r_lines, c_lines):
                assert rl[0] == cl[0] and rl[2:] == cl[2:], f"Box/conf mismatch on {s}: {rl} vs {cl}"
                if cl[1] in id_map:
                    assert id_map[cl[1]] == rl[1], f"ID bijection break on {s}"
                id_map[cl[1]] = rl[1]
    t1 = time.perf_counter()

    events_csv = run_dir / "anatomy_events.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(probe.events[0].keys()))
        w.writeheader()
        w.writerows(probe.events)

    onset_ev = [e for e in probe.events if e["sublabel"] == "onset_cross"]
    summary = {
        "schema_version": "s13-anatomy-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "ref_online_run_id": args.ref_online_run_id,
        "sequences": target_vids,
        "sim_threshold": sim_threshold,
        "checkpoint_sha256": ckpt_sha,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "exact_mot_match_vs_online": exact_mot_match,
        "sublabel_counts": {
            "clean_match": sum(1 for e in probe.events if e["sublabel"] == "clean_match"),
            "onset_cross": len(onset_ev),
            "post_switch_cross": sum(1 for e in probe.events if e["sublabel"] == "post_switch_cross"),
            "fp_pollution": sum(1 for e in probe.events if e["sublabel"] == "fp_pollution"),
            "unanchored": sum(1 for e in probe.events if e["sublabel"] == "unanchored"),
            "total_events": len(probe.events),
        },
        "onset_by_sequence": {
            s: sum(1 for e in onset_ev if e["sequence"] == s) for s in target_vids
        },
        "onset_summary": {
            "total_onset": len(onset_ev),
            "trk_gap_eq_1": sum(1 for e in onset_ev if int(e["trk_gap_frames"]) == 1),
            "competing_iou_ge_030": sum(1 for e in onset_ev if float(e["max_conflict_iou"]) >= 0.30),
            "assignment_margin_lt_010": sum(1 for e in onset_ev if float(e["assignment_margin"]) < 0.10),
            "true_det_absent_in_frame": sum(1 for e in onset_ev if int(e["true_det_present_in_frame"]) == 0),
            "true_det_present_in_frame": sum(1 for e in onset_ev if int(e["true_det_present_in_frame"]) == 1),
        },
        "events_csv": str(events_csv),
    }
    (run_dir / "stage_anatomy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "exact_mot_match": exact_mot_match, "sublabel_counts": summary["sublabel_counts"], "onset_summary": summary["onset_summary"], "onset_by_sequence": summary["onset_by_sequence"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage-a-lock")

    p_tc = sub.add_parser("train-c")
    p_tc.add_argument("--condition", choices=list(S13_CONDITIONS.keys()), required=True)
    p_tc.add_argument("--run-id", required=True)

    p_eo = sub.add_parser("eval-online")
    p_eo.add_argument("--condition", required=True)
    p_eo.add_argument("--ckpt", required=True)
    p_eo.add_argument("--sim-threshold", type=float, required=True)
    p_eo.add_argument("--disambig", choices=["NONE", "B1_DUAL_SOFTMAX"], default="NONE")
    p_eo.add_argument("--model-family", choices=["s12", "s13"], default="s13")
    p_eo.add_argument("--run-id", required=True)

    p_ar = sub.add_parser("anatomy-replay")
    p_ar.add_argument("--ckpt", required=True)
    p_ar.add_argument("--sim-threshold", type=float, required=True)
    p_ar.add_argument("--disambig", choices=["NONE", "B1_DUAL_SOFTMAX"], default="NONE")
    p_ar.add_argument("--model-family", choices=["s12", "s13"], default="s13")
    p_ar.add_argument("--ref-online-run-id", required=True)
    p_ar.add_argument("--ref-root", default="")
    p_ar.add_argument("--seqs", nargs="*", default=[])
    p_ar.add_argument("--run-id", required=True)

    args = parser.parse_args()
    {
        "stage-a-lock": cmd_stage_a_lock,
        "train-c": cmd_train_c,
        "eval-online": cmd_eval_online,
        "anatomy-replay": cmd_anatomy_replay,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
