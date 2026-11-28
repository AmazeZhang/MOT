#!/usr/bin/env python3
"""Execute S12 CAMELTrack pairwise kin+DOA edge bias (C') and assignment disambiguation (A')."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from scipy.optimize import linear_sum_assignment
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb as tl_wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_s09_camel as s09  # noqa: E402
import run_s11_camel as s11  # noqa: E402
from cameltrack.camel import CAMEL, Detections, Tracklets  # noqa: E402
from cameltrack.cameltrack import Tracklet  # noqa: E402
from cameltrack.train.callbacks import SimMetrics  # noqa: E402

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s12_camel_pairwise_edge_and_assignment_20261127.yaml"
REPORT_DIR = REPO_ROOT / "reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127"
RUNS_ROOT = DATA_ROOT / "runs/s12_camel_pairwise_edge_and_assignment_20261127"
CACHE_ROOT = DATA_ROOT / "cache/s12_camel"
SHARED_DM_CACHE = CACHE_ROOT / "dm_s12_train32_dev8_shared"

S09_RUNS_ROOT = DATA_ROOT / "runs/s09_camel_calibration_and_attribution_20261126"
S09_T_TWOSTAGE_CKPT = S09_RUNS_ROOT / "c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt"
S09_C1_CKPTS = {cue: S09_RUNS_ROOT / f"c1_{cue}_seed13_e20_v1/CAMEL/last.ckpt" for cue in s09.TE_CUES}

S12_ARCH_MAP = {
    "M0_NODE_CAP_CTRL": "node_cap_ctrl",
    "M1_C_KIN_EDGE": "kin_edge",
    "M2_C_KIN_DOA_EDGE": "kin_doa_edge",
    "M3_C_PRIME_PLUS_A_LOSS": "kin_doa_edge_a_loss",
}


def ensure_shared_dm_cache() -> Path:
    """Copy verified S09/S10 datamodule cache atomically into S12 isolated cache directory."""
    src_candidates = [
        DATA_ROOT / "cache/s10_camel/dm_s10_train32_dev8_shared",
        DATA_ROOT / "cache/s09_camel/dm_c2_twostage_joint_seed13_e20_v1_isolated",
    ]
    src_dir = next(d for d in src_candidates if (d / "camel_train.pklz").is_file())
    SHARED_DM_CACHE.mkdir(parents=True, exist_ok=True)
    for fname in ("camel_train.pklz", "camel_train_tracklets.json", "camel_val.pklz", "camel_val_tracklets.json"):
        src = src_dir / fname
        dst = SHARED_DM_CACHE / fname
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            tmp = SHARED_DM_CACHE / f".{fname}.tmp"
            shutil.copy2(src, tmp)
            tmp.replace(dst)
    return SHARED_DM_CACHE


def _zero_init_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


@torch.no_grad()
def compute_causal_extrapolation(
    bbox_ltwh: torch.Tensor,
    age: torch.Tensor,
    feats_masks: torch.Tensor,
    drop_bbox: Optional[torch.Tensor] = None,
    decay_alpha: float = 0.9,
) -> dict[str, torch.Tensor]:
    """Strictly causal exponentially weighted least-squares constant-velocity extrapolation to age=0.

    Uses only observed past boxes in `tracks.feats["bbox_ltwh"]` (age >= 1, excluding NaNs,
    non-positive dimensions, and DropoutFeatures `drop_bbox==1` slots). Safe on empty tracklets.
    """
    B, T, S, _ = bbox_ltwh.shape
    device = bbox_ltwh.device
    dtype = torch.float32

    boxes = torch.nan_to_num(bbox_ltwh.to(dtype), nan=0.0)
    ages = torch.nan_to_num(age.squeeze(-1).to(dtype), nan=1e6)
    finite_mask = torch.isfinite(bbox_ltwh).all(dim=-1) & torch.isfinite(age.squeeze(-1))
    pos_wh = (boxes[..., 2] > 1e-6) & (boxes[..., 3] > 1e-6)
    valid = feats_masks.bool() & finite_mask & pos_wh & (ages >= 1.0)
    if drop_bbox is not None:
        drop_flag = torch.nan_to_num(drop_bbox.squeeze(-1).to(dtype), nan=0.0) > 0.5
        valid = valid & (~drop_flag)

    trk_valid = valid.any(dim=-1)  # [B, T]
    # Most recent valid observation = minimum age
    score_recent = torch.where(valid, -ages, torch.full_like(ages, -1e9))
    last_idx = score_recent.argmax(dim=-1)  # [B, T]
    b_idx = torch.arange(B, device=device)[:, None]
    t_idx = torch.arange(T, device=device)[None, :]
    last_box = boxes[b_idx, t_idx, last_idx]  # [B, T, 4]
    dt_gap = torch.where(trk_valid, ages[b_idx, t_idx, last_idx].clamp(min=1.0), torch.ones((B, T), device=device, dtype=dtype))

    cx = boxes[..., 0] + 0.5 * boxes[..., 2]
    cy = boxes[..., 1] + 0.5 * boxes[..., 3]
    cx_last = last_box[..., 0] + 0.5 * last_box[..., 2]
    cy_last = last_box[..., 1] + 0.5 * last_box[..., 3]
    w_last = torch.where(trk_valid, last_box[..., 2].clamp(min=1e-4), torch.ones((B, T), device=device, dtype=dtype))
    h_last = torch.where(trk_valid, last_box[..., 3].clamp(min=1e-4), torch.ones((B, T), device=device, dtype=dtype))

    # Weighted linear regression over time t_s = -ages_s (<= -1)
    rel_age = (ages - dt_gap.unsqueeze(-1)).clamp(min=0.0, max=50.0)
    w_obs = torch.where(valid, torch.exp(-0.25 * rel_age), torch.zeros_like(rel_age))
    w_sum = w_obs.sum(dim=-1).clamp(min=1e-6)
    t_s = -ages
    t_bar = (w_obs * t_s).sum(dim=-1) / w_sum
    cx_bar = (w_obs * cx).sum(dim=-1) / w_sum
    cy_bar = (w_obs * cy).sum(dim=-1) / w_sum

    dt_c = (t_s - t_bar.unsqueeze(-1)) * valid.to(dtype)
    denom = (w_obs * (dt_c ** 2)).sum(dim=-1)
    n_valid = valid.sum(dim=-1)
    has_vel = (n_valid >= 2) & (denom > 0.25)

    safe_denom = denom.clamp(min=1e-4)
    vx = torch.where(has_vel, (w_obs * dt_c * (cx - cx_bar.unsqueeze(-1))).sum(dim=-1) / safe_denom, torch.zeros_like(denom))
    vy = torch.where(has_vel, (w_obs * dt_c * (cy - cy_bar.unsqueeze(-1))).sum(dim=-1) / safe_denom, torch.zeros_like(denom))
    # Clamp velocities in units of box sizes per frame to avoid rare outlier explosions
    vx = vx.clamp(min=-2.0 * w_last, max=2.0 * w_last)
    vy = vy.clamp(min=-2.0 * h_last, max=2.0 * h_last)

    damp_steps = (1.0 - torch.pow(torch.full_like(dt_gap, decay_alpha), dt_gap)) / (1.0 - decay_alpha)
    cx_hat = cx_last + vx * damp_steps
    cy_hat = cy_last + vy * damp_steps
    b_hat = torch.stack([cx_hat - 0.5 * w_last, cy_hat - 0.5 * h_last, w_last, h_last], dim=-1)
    b_last_safe = torch.stack([last_box[..., 0], last_box[..., 1], w_last, h_last], dim=-1)

    return {
        "b_hat": b_hat,
        "b_last": b_last_safe,
        "cx_hat": cx_hat,
        "cy_hat": cy_hat,
        "cx_last": cx_last,
        "cy_last": cy_last,
        "w_hat": w_last,
        "h_hat": h_last,
        "vx": vx,
        "vy": vy,
        "dt": dt_gap,
        "has_vel": has_vel,
        "trk_valid": trk_valid,
    }


def _box_iou_giou_pairwise(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (iou, giou, inter) of shape [..., A, B] for LTWH tensors a [..., A, 4], b [..., B, 4]."""
    ax1, ay1 = a[..., :, None, 0], a[..., :, None, 1]
    aw, ah = a[..., :, None, 2].clamp(min=1e-6), a[..., :, None, 3].clamp(min=1e-6)
    ax2, ay2 = ax1 + aw, ay1 + ah

    bx1, by1 = b[..., None, :, 0], b[..., None, :, 1]
    bw, bh = b[..., None, :, 2].clamp(min=1e-6), b[..., None, :, 3].clamp(min=1e-6)
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = torch.maximum(ax1, bx1), torch.maximum(ay1, by1)
    ix2, iy2 = torch.minimum(ax2, bx2), torch.minimum(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)
    area_a = aw * ah
    area_b = bw * bh
    union = (area_a + area_b - inter).clamp(min=1e-9)
    iou = inter / union

    cx1, cy1 = torch.minimum(ax1, bx1), torch.minimum(ay1, by1)
    cx2, cy2 = torch.maximum(ax2, bx2), torch.maximum(ay2, by2)
    c_area = ((cx2 - cx1).clamp(min=1e-6) * (cy2 - cy1).clamp(min=1e-6)).clamp(min=1e-9)
    giou = iou - (c_area - union) / c_area
    return iou, giou, inter


def _four_way_half_coverage(target: torch.Tensor, occluder: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute [..., A, B] overlap fraction of target A's L/R/T/B halves covered by occluder B."""
    tx1, ty1 = target[..., :, None, 0], target[..., :, None, 1]
    tw, th = target[..., :, None, 2].clamp(min=1e-6), target[..., :, None, 3].clamp(min=1e-6)
    tx2, ty2 = tx1 + tw, ty1 + th
    tcx, tcy = tx1 + 0.5 * tw, ty1 + 0.5 * th

    ox1, oy1 = occluder[..., None, :, 0], occluder[..., None, :, 1]
    ow, oh = occluder[..., None, :, 2].clamp(min=1e-6), occluder[..., None, :, 3].clamp(min=1e-6)
    ox2, oy2 = ox1 + ow, oy1 + oh

    iy_full = (torch.minimum(ty2, oy2) - torch.maximum(ty1, oy1)).clamp(min=0.0)
    ix_full = (torch.minimum(tx2, ox2) - torch.maximum(tx1, ox1)).clamp(min=0.0)
    half_area = (0.5 * tw * th).clamp(min=1e-9)

    cov_l = ((torch.minimum(tcx, ox2) - torch.maximum(tx1, ox1)).clamp(min=0.0) * iy_full) / half_area
    cov_r = ((torch.minimum(tx2, ox2) - torch.maximum(tcx, ox1)).clamp(min=0.0) * iy_full) / half_area
    cov_t = (ix_full * (torch.minimum(tcy, oy2) - torch.maximum(ty1, oy1)).clamp(min=0.0)) / half_area
    cov_b = (ix_full * (torch.minimum(ty2, oy2) - torch.maximum(tcy, oy1)).clamp(min=0.0)) / half_area
    return cov_l, cov_r, cov_t, cov_b


@torch.no_grad()
def compute_pairwise_kin_doa_features(tracks: Tracklets, dets: Detections) -> dict[str, torch.Tensor]:
    """Compute causal node, pairwise 13D kin, 13D DOA occlusion/front-back, and self spatial bias tensors."""
    B, T = tracks.masks.shape
    _, D = dets.masks.shape
    device = tracks.masks.device
    dtype = torch.float32

    t_drop = tracks.feats.get("drop_bbox")
    ext = compute_causal_extrapolation(tracks.feats["bbox_ltwh"], tracks.feats["age"], tracks.feats_masks, drop_bbox=t_drop)

    raw_d_boxes = dets.feats["bbox_ltwh"].squeeze(2).to(dtype)
    d_drop = dets.feats.get("drop_bbox")
    d_valid = dets.masks.bool() & torch.isfinite(raw_d_boxes).all(dim=-1) & (raw_d_boxes[..., 2] > 1e-6) & (raw_d_boxes[..., 3] > 1e-6)
    if d_drop is not None:
        d_valid = d_valid & (torch.nan_to_num(d_drop.squeeze(2).squeeze(-1).to(dtype), nan=0.0) <= 0.5)

    d_boxes = torch.where(d_valid.unsqueeze(-1), raw_d_boxes, torch.zeros_like(raw_d_boxes))
    d_boxes = torch.stack([d_boxes[..., 0], d_boxes[..., 1], d_boxes[..., 2].clamp(min=1e-4), d_boxes[..., 3].clamp(min=1e-4)], dim=-1)

    t_valid = ext["trk_valid"] & tracks.masks.bool()
    pair_valid = t_valid.unsqueeze(2) & d_valid.unsqueeze(1)  # [B, T, D]

    dcx = d_boxes[..., 0] + 0.5 * d_boxes[..., 2]
    dcy = d_boxes[..., 1] + 0.5 * d_boxes[..., 3]
    dw, dh = d_boxes[..., 2], d_boxes[..., 3]
    dy2 = d_boxes[..., 1] + d_boxes[..., 3]

    b_hat, b_last = ext["b_hat"], ext["b_last"]
    w_hat, h_hat = ext["w_hat"], ext["h_hat"]
    ty2_hat = b_hat[..., 1] + h_hat

    # 1. Pairwise 13D kinematic extrapolation residual features e_kin [B, T, D, 13]
    dx_hat = ((dcx.unsqueeze(1) - ext["cx_hat"].unsqueeze(2)) / w_hat.unsqueeze(2)).clamp(-10.0, 10.0)
    dy_hat = ((dcy.unsqueeze(1) - ext["cy_hat"].unsqueeze(2)) / h_hat.unsqueeze(2)).clamp(-10.0, 10.0)
    dx_last = ((dcx.unsqueeze(1) - ext["cx_last"].unsqueeze(2)) / w_hat.unsqueeze(2)).clamp(-10.0, 10.0)
    dy_last = ((dcy.unsqueeze(1) - ext["cy_last"].unsqueeze(2)) / h_hat.unsqueeze(2)).clamp(-10.0, 10.0)
    log_dw = torch.log((dw.unsqueeze(1) / w_hat.unsqueeze(2)).clamp(1e-2, 100.0)).clamp(-5.0, 5.0)
    log_dh = torch.log((dh.unsqueeze(1) / h_hat.unsqueeze(2)).clamp(1e-2, 100.0)).clamp(-5.0, 5.0)

    iou_hat, giou_hat, _ = _box_iou_giou_pairwise(b_hat, d_boxes)
    iou_last, _, _ = _box_iou_giou_pairwise(b_last, d_boxes)

    disp_x = dcx.unsqueeze(1) - ext["cx_last"].unsqueeze(2)
    disp_y = dcy.unsqueeze(1) - ext["cy_last"].unsqueeze(2)
    vx_e, vy_e = ext["vx"].unsqueeze(2), ext["vy"].unsqueeze(2)
    v_norm = torch.hypot(vx_e, vy_e)
    d_norm = torch.hypot(disp_x, disp_y)
    cos_dir = torch.where(
        ext["has_vel"].unsqueeze(2) & (v_norm > 1e-6) & (d_norm > 1e-6),
        (vx_e * disp_x + vy_e * disp_y) / (v_norm * d_norm).clamp(min=1e-9),
        torch.zeros_like(dx_hat),
    )
    speed_norm = (torch.hypot(ext["vx"], ext["vy"]) / h_hat).clamp(0.0, 10.0).unsqueeze(2).expand(B, T, D)
    log_dt = torch.log1p(ext["dt"]).unsqueeze(2).expand(B, T, D)
    has_vel_f = ext["has_vel"].to(dtype).unsqueeze(2).expand(B, T, D)

    e_kin = torch.stack(
        [dx_hat, dy_hat, dx_last, dy_last, log_dw, log_dh, iou_hat, giou_hat, iou_last, cos_dir, speed_norm, log_dt, has_vel_f],
        dim=-1,
    )
    e_kin = torch.where(pair_valid.unsqueeze(-1), torch.nan_to_num(e_kin, 0.0), torch.zeros_like(e_kin))

    # 2. Scene context & DOA directional occlusion + front/back features [B, T, D, 13]
    det_iou, _, _ = _box_iou_giou_pairwise(d_boxes, d_boxes)
    d_valid_pair = d_valid.unsqueeze(2) & d_valid.unsqueeze(1)
    eye_d = torch.eye(D, device=device, dtype=torch.bool).unsqueeze(0)
    other_d_mask = d_valid_pair & (~eye_d)
    det_max_other_iou = torch.where(other_d_mask, det_iou, torch.zeros_like(det_iou)).max(dim=2).values if D > 0 else torch.zeros((B, D), device=device, dtype=dtype)

    # Front mask on detections: k has bottom y2_k > y2_j
    d_front_mask = other_d_mask & (dy2.unsqueeze(1) > dy2.unsqueeze(2))  # [B, j, k]
    d_cov_l, d_cov_r, d_cov_t, d_cov_b = _four_way_half_coverage(d_boxes, d_boxes)
    det_front_l = torch.where(d_front_mask, d_cov_l, torch.zeros_like(d_cov_l)).max(dim=2).values if D > 0 else torch.zeros((B, D), device=device)
    det_front_r = torch.where(d_front_mask, d_cov_r, torch.zeros_like(d_cov_r)).max(dim=2).values if D > 0 else torch.zeros((B, D), device=device)
    det_front_t = torch.where(d_front_mask, d_cov_t, torch.zeros_like(d_cov_t)).max(dim=2).values if D > 0 else torch.zeros((B, D), device=device)
    det_front_b = torch.where(d_front_mask, d_cov_b, torch.zeros_like(d_cov_b)).max(dim=2).values if D > 0 else torch.zeros((B, D), device=device)
    det_front_rank = (
        torch.where(other_d_mask, (dy2.unsqueeze(2) > dy2.unsqueeze(1)).to(dtype), torch.zeros_like(det_iou)).sum(dim=2)
        / d_valid.sum(dim=1, keepdim=True).clamp(min=1).to(dtype)
    ) if D > 0 else torch.zeros((B, D), device=device)

    # Tracklet occlusion against front detections & other valid tracklets
    td_front_k_mask = t_valid.unsqueeze(2) & d_valid.unsqueeze(1) & (dy2.unsqueeze(1) > ty2_hat.unsqueeze(2))
    _, _, td_inter = _box_iou_giou_pairwise(b_hat, d_boxes)
    area_hat = (w_hat * h_hat).clamp(min=1e-9).unsqueeze(2)
    trk_front_occ_max = (
        torch.where(td_front_k_mask, (td_inter / area_hat).clamp(0.0, 1.0), torch.zeros_like(td_inter)).max(dim=2).values
        if D > 0 else torch.zeros((B, T), device=device)
    )

    trk_iou, _, _ = _box_iou_giou_pairwise(b_hat, b_hat)
    t_valid_pair = t_valid.unsqueeze(2) & t_valid.unsqueeze(1)
    eye_t = torch.eye(T, device=device, dtype=torch.bool).unsqueeze(0)
    other_t_mask = t_valid_pair & (~eye_t)
    trk_max_other_iou = torch.where(other_t_mask, trk_iou, torch.zeros_like(trk_iou)).max(dim=2).values if T > 0 else torch.zeros((B, T), device=device)
    trk_front_rank = (
        torch.where(other_t_mask, (ty2_hat.unsqueeze(2) > ty2_hat.unsqueeze(1)).to(dtype), torch.zeros_like(trk_iou)).sum(dim=2)
        / t_valid.sum(dim=1, keepdim=True).clamp(min=1).to(dtype)
    ) if T > 0 else torch.zeros((B, T), device=device)

    # Pairwise DOA directional occlusion of predicted track b_hat_i by candidate detection b_j
    pair_cov_l, pair_cov_r, pair_cov_t, pair_cov_b = _four_way_half_coverage(b_hat, d_boxes)
    delta_y2 = ((dy2.unsqueeze(1) - ty2_hat.unsqueeze(2)) / h_hat.unsqueeze(2)).clamp(-10.0, 10.0)

    e_occ = torch.stack(
        [
            delta_y2,
            det_front_rank.unsqueeze(1).expand(B, T, D),
            det_front_l.unsqueeze(1).expand(B, T, D),
            det_front_r.unsqueeze(1).expand(B, T, D),
            det_front_t.unsqueeze(1).expand(B, T, D),
            det_front_b.unsqueeze(1).expand(B, T, D),
            det_max_other_iou.unsqueeze(1).expand(B, T, D),
            pair_cov_l,
            pair_cov_r,
            pair_cov_t,
            pair_cov_b,
            trk_front_occ_max.unsqueeze(2).expand(B, T, D),
            trk_max_other_iou.unsqueeze(2).expand(B, T, D),
        ],
        dim=-1,
    )
    e_occ = torch.where(pair_valid.unsqueeze(-1), torch.nan_to_num(e_occ, 0.0), torch.zeros_like(e_occ))

    # 3. Single-node features for M0_NODE_CAP_CTRL [8D track, 6D det]
    node_t = torch.stack(
        [
            (ext["vx"] / w_hat).clamp(-10.0, 10.0),
            (ext["vy"] / h_hat).clamp(-10.0, 10.0),
            (torch.hypot(ext["vx"], ext["vy"]) / h_hat).clamp(0.0, 10.0),
            torch.log1p(ext["dt"]),
            ext["has_vel"].to(dtype),
            trk_front_occ_max,
            trk_max_other_iou,
            trk_front_rank,
        ],
        dim=-1,
    )
    node_t = torch.where(t_valid.unsqueeze(-1), torch.nan_to_num(node_t, 0.0), torch.zeros_like(node_t))

    node_d = torch.stack(
        [det_front_l, det_front_r, det_front_t, det_front_b, det_max_other_iou, det_front_rank],
        dim=-1,
    )
    node_d = torch.where(d_valid.unsqueeze(-1), torch.nan_to_num(node_d, 0.0), torch.zeros_like(node_d))

    # 4. Self spatial adjacency features [det-det, trk-trk] for M2/M3: [IoU, |delta_y2|]
    dd_dy2 = ((dy2.unsqueeze(2) - dy2.unsqueeze(1)).abs() / dh.unsqueeze(1).clamp(min=1e-4)).clamp(0.0, 10.0)
    self_dd = torch.stack([det_iou, dd_dy2], dim=-1)
    self_dd = torch.where(other_d_mask.unsqueeze(-1), torch.nan_to_num(self_dd, 0.0), torch.zeros_like(self_dd))

    tt_dy2 = ((ty2_hat.unsqueeze(2) - ty2_hat.unsqueeze(1)).abs() / h_hat.unsqueeze(1).clamp(min=1e-4)).clamp(0.0, 10.0)
    self_tt = torch.stack([trk_iou, tt_dy2], dim=-1)
    self_tt = torch.where(other_t_mask.unsqueeze(-1), torch.nan_to_num(self_tt, 0.0), torch.zeros_like(self_tt))

    return {
        "e_kin": e_kin,
        "e_occ": e_occ,
        "node_t": node_t,
        "node_d": node_d,
        "self_dd": self_dd,
        "self_tt": self_tt,
        "b_hat": b_hat,
        "pair_valid": pair_valid,
    }


class S12PairwiseCAMEL(CAMEL):
    """Unified S12 CAMEL supporting baseline / M0_NODE_CAP_CTRL / M1_C_KIN_EDGE / M2_C_KIN_DOA_EDGE / M3_C_PRIME_PLUS_A_LOSS."""

    def __init__(
        self,
        *args,
        arch_variant: str = "kin_doa_edge",
        lambda_margin: float = 0.5,
        lambda_rej: float = 0.25,
        margin_target: float = 0.15,
        tau_neg: float = 0.35,
        beta: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.arch_variant = str(arch_variant)
        self.lambda_margin = float(lambda_margin)
        self.lambda_rej = float(lambda_rej)
        self.margin_target = float(margin_target)
        self.tau_neg = float(tau_neg)
        self.beta = float(beta)
        self.save_hyperparameters(ignore=[k for k in locals() if "checkpoint_path" in k])

        emb_dim = int(self.gaffe.emb_dim)
        n_layers = int(self.gaffe.n_layers)
        n_heads = int(self.gaffe.n_heads)

        if self.arch_variant == "node_cap_ctrl":
            # Single-node residual MLPs with strictly zero-initialized output layers (>= M2 extra params)
            self.node_t_mlp = nn.Sequential(
                nn.Linear(8, 128),
                nn.GELU(),
                _zero_init_linear(nn.Linear(128, emb_dim)),
            )
            self.node_d_mlp = nn.Sequential(
                nn.Linear(6, 128),
                nn.GELU(),
                _zero_init_linear(nn.Linear(128, emb_dim)),
            )
        elif self.arch_variant in ("kin_edge", "kin_doa_edge", "kin_doa_edge_a_loss"):
            edge_dim = 13 if self.arch_variant == "kin_edge" else 26
            self.edge_attn_td = nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.GELU(),
                _zero_init_linear(nn.Linear(64, n_layers * n_heads)),
            )
            self.edge_attn_dt = nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.GELU(),
                _zero_init_linear(nn.Linear(64, n_layers * n_heads)),
            )
            if self.arch_variant in ("kin_doa_edge", "kin_doa_edge_a_loss"):
                self.self_dd_attn = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.GELU(),
                    _zero_init_linear(nn.Linear(32, n_layers * n_heads)),
                )
                self.self_tt_attn = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.GELU(),
                    _zero_init_linear(nn.Linear(32, n_layers * n_heads)),
                )
            self.sim_proj_t = nn.Linear(emb_dim, 64)
            self.sim_proj_d = nn.Linear(emb_dim, 64)
            self.edge_sim_mlp = nn.Sequential(
                nn.Linear(edge_dim + 128, 64),
                nn.GELU(),
                _zero_init_linear(nn.Linear(64, 1)),
            )

    def _run_gaffe_with_bias(
        self,
        tracks: Tracklets,
        dets: Detections,
        bias_td: Optional[torch.Tensor],
        bias_dt: Optional[torch.Tensor],
        bias_dd: Optional[torch.Tensor],
        bias_tt: Optional[torch.Tensor],
    ) -> tuple[Tracklets, Detections]:
        g = self.gaffe
        src = torch.cat([dets.tokens, tracks.tokens, g.cls.repeat(dets.masks.shape[0], 1, 1)], dim=1)
        src = g.src_norm(src)
        src = g.src_drop(src)

        B, L, _ = src.shape
        D = dets.masks.shape[1]
        T = tracks.masks.shape[1]
        n_layers = int(g.n_layers)
        n_heads = int(g.n_heads)

        src_mask = torch.cat(
            [dets.masks, tracks.masks, torch.ones((B, 1), device=src.device, dtype=torch.bool)],
            dim=1,
        )

        if bias_td is None:
            x = g.encoder(src, src_key_padding_mask=~src_mask)
        else:
            # Unified float attention bias [B, n_layers, n_heads, L, L] preserving diagonal 0.0
            bad = ~src_mask.unsqueeze(1).expand(B, L, L)
            eye = torch.eye(L, device=src.device, dtype=torch.bool).unsqueeze(0)
            bad = bad & (~eye)
            diag_idx = torch.arange(L, device=src.device)

            x = src
            for l_idx, layer in enumerate(g.encoder.layers):
                m_l = torch.zeros((B, n_heads, L, L), device=src.device, dtype=src.dtype)
                if T > 0 and D > 0:
                    m_l[:, :, :D, D : D + T] = bias_dt[:, l_idx]  # det queries -> track keys
                    m_l[:, :, D : D + T, :D] = bias_td[:, l_idx]  # track queries -> det keys
                if bias_dd is not None and D > 0:
                    m_l[:, :, :D, :D] = bias_dd[:, l_idx]
                if bias_tt is not None and T > 0:
                    m_l[:, :, D : D + T, D : D + T] = bias_tt[:, l_idx]
                m_l.masked_fill_(bad.unsqueeze(1), -1e9)
                m_l[:, :, diag_idx, diag_idx] = 0.0
                x = layer(x, src_mask=m_l.reshape(B * n_heads, L, L))
            if g.encoder.norm is not None:
                x = g.encoder.norm(x)

        tracks.embs = x[:, D : D + T]
        dets.embs = x[:, :D]
        return tracks, dets

    def forward(self, tracks: Tracklets, dets: Detections):
        feat_dict = compute_pairwise_kin_doa_features(tracks, dets) if self.arch_variant != "baseline" else None
        tracks, dets = self.tokenize(tracks, dets)
        tracks, dets = self.merge(tracks, dets)

        if self.arch_variant == "node_cap_ctrl":
            res_t = self.node_t_mlp(feat_dict["node_t"]) * tracks.masks.unsqueeze(-1).to(tracks.tokens.dtype)
            res_d = self.node_d_mlp(feat_dict["node_d"]) * dets.masks.unsqueeze(-1).to(dets.tokens.dtype)
            tracks.tokens = tracks.tokens + res_t
            dets.tokens = dets.tokens + res_d
            tracks, dets = self._run_gaffe_with_bias(tracks, dets, None, None, None, None)
            td_sim_matrix = self.similarity(tracks, dets)
        elif self.arch_variant in ("kin_edge", "kin_doa_edge", "kin_doa_edge_a_loss"):
            e_pair = feat_dict["e_kin"] if self.arch_variant == "kin_edge" else torch.cat([feat_dict["e_kin"], feat_dict["e_occ"]], dim=-1)
            B, T, D, _ = e_pair.shape
            n_l, n_h = int(self.gaffe.n_layers), int(self.gaffe.n_heads)
            pair_valid = feat_dict["pair_valid"]

            b_td = self.edge_attn_td(e_pair).view(B, T, D, n_l, n_h).permute(0, 3, 4, 1, 2)
            b_dt = self.edge_attn_dt(e_pair).view(B, T, D, n_l, n_h).permute(0, 3, 4, 2, 1)
            if self.arch_variant in ("kin_doa_edge", "kin_doa_edge_a_loss"):
                b_dd = self.self_dd_attn(feat_dict["self_dd"]).view(B, D, D, n_l, n_h).permute(0, 3, 4, 1, 2)
                b_tt = self.self_tt_attn(feat_dict["self_tt"]).view(B, T, T, n_l, n_h).permute(0, 3, 4, 1, 2)
            else:
                b_dd, b_tt = None, None

            tracks, dets = self._run_gaffe_with_bias(tracks, dets, b_td, b_dt, b_dd, b_tt)
            td_sim_base = self.similarity(tracks, dets)

            # Gated output pairwise residual delta_S = 0.5 * tanh(mlp([e_pair, z_t_64, z_d_64]))
            zt = F.relu(self.sim_proj_t(tracks.embs)).unsqueeze(2).expand(B, T, D, 64)
            zd = F.relu(self.sim_proj_d(dets.embs)).unsqueeze(1).expand(B, T, D, 64)
            delta_s = 0.5 * torch.tanh(self.edge_sim_mlp(torch.cat([e_pair, zt, zd], dim=-1)).squeeze(-1))
            delta_s = torch.where(pair_valid, delta_s, torch.zeros_like(delta_s))
            tracks._last_delta_s = delta_s
            td_sim_matrix = torch.where(td_sim_base > -1e8, td_sim_base + delta_s, td_sim_base)
        else:
            tracks, dets = self.gaffe(tracks, dets)
            td_sim_matrix = self.similarity(tracks, dets)

        return tracks, dets, td_sim_matrix

    def _compute_base_sim_loss_with_delta(self, tracks: Tracklets, dets: Detections) -> torch.Tensor:
        """Exact CAMEL NTXentLoss incorporating cross-domain edge residual 2*delta_s on (trk <-> det)."""
        delta_s = getattr(tracks, "_last_delta_s", None)
        if delta_s is None:
            return super().compute_loss(tracks, dets)

        t_embs = tracks.embs if not isinstance(tracks.embs, dict) else tracks.embs["default"]
        d_embs = dets.embs if not isinstance(dets.embs, dict) else dets.embs["default"]
        B = t_embs.shape[0]
        sim_losses = torch.zeros((B,), dtype=torch.float32, device=self.device)
        valid_b = torch.zeros((B,), dtype=torch.bool, device=self.device)

        for b in range(B):
            m_t = tracks.masks[b]
            m_d = dets.masks[b]
            if not (m_t.any() or m_d.any()):
                continue
            valid_b[b] = True
            t_inc = m_t & (tracks.targets[b] >= 0)
            d_inc = m_d & (dets.targets[b] >= 0)
            embs = torch.cat([t_embs[b, t_inc], d_embs[b, d_inc]], dim=0)
            labels = torch.cat([tracks.targets[b, t_inc], dets.targets[b, d_inc]], dim=0)
            if len(labels) < 2:
                continue
            idx_tuple = lmu.get_all_pairs_indices(labels)
            if all(len(x) <= 1 for x in idx_tuple):
                continue
            mat = self.sim_loss.distance(embs, None)
            n_t = int(t_inc.sum().item())
            n_d = int(d_inc.sum().item())
            if n_t > 0 and n_d > 0:
                ds_sub = delta_s[b][t_inc][:, d_inc].to(mat.dtype)
                pad_tl = torch.zeros((n_t, n_t), device=mat.device, dtype=mat.dtype)
                pad_br = torch.zeros((n_d, n_d), device=mat.device, dtype=mat.dtype)
                delta_mat = torch.cat(
                    [
                        torch.cat([pad_tl, 2.0 * ds_sub], dim=1),
                        torch.cat([2.0 * ds_sub.t(), pad_br], dim=1),
                    ],
                    dim=0,
                )
                mat = mat + delta_mat
            loss_dict = self.sim_loss.loss_method(mat, idx_tuple)
            sim_losses[b] = self.sim_loss.reducer(loss_dict, embs, labels)

        if not valid_b.any():
            return t_embs.sum() * 0.0
        return sim_losses[valid_b].mean().nan_to_num(0)

    def compute_loss(self, tracks: Tracklets, dets: Detections, *args):
        base_loss = self._compute_base_sim_loss_with_delta(tracks, dets)
        tracks._last_base_loss = base_loss.detach()
        if self.arch_variant != "kin_doa_edge_a_loss" or not args:
            return base_loss

        td_sim = args[0]  # [B, T, D]
        B = td_sim.shape[0]
        device = td_sim.device
        beta = self.beta
        m0 = self.margin_target
        tau_neg = self.tau_neg

        margin_terms: list[torch.Tensor] = []
        rej_terms: list[torch.Tensor] = []

        for b in range(B):
            t_idx = torch.where(tracks.masks[b] & (tracks.targets[b] >= 0))[0]
            d_idx = torch.where(dets.masks[b] & (dets.targets[b] >= 0))[0]
            if len(t_idx) == 0 or len(d_idx) == 0:
                continue
            S = td_sim[b][t_idx][:, d_idx]  # [nT, nD]
            t_tgt = tracks.targets[b, t_idx]
            d_tgt = dets.targets[b, d_idx]
            pos_mask = t_tgt[:, None] == d_tgt[None, :]
            neg_mask = ~pos_mask

            # Bidirectional margin loss on positive pairs
            row_has_neg = neg_mask.any(dim=1)
            col_has_neg = neg_mask.any(dim=0)
            row_max_neg = torch.where(neg_mask, S, torch.full_like(S, -1e9)).max(dim=1, keepdim=True).values
            col_max_neg = torch.where(neg_mask, S, torch.full_like(S, -1e9)).max(dim=0, keepdim=True).values

            pos_coords = torch.where(pos_mask)
            if len(pos_coords[0]) > 0:
                pi, pj = pos_coords
                s_pos = S[pi, pj]
                pair_losses = []
                valid_r = row_has_neg[pi]
                if valid_r.any():
                    pair_losses.append(F.softplus(beta * (m0 - (s_pos[valid_r] - row_max_neg[pi[valid_r], 0]))) / beta)
                valid_c = col_has_neg[pj]
                if valid_c.any():
                    pair_losses.append(F.softplus(beta * (m0 - (s_pos[valid_c] - col_max_neg[0, pj[valid_c]]))) / beta)
                if pair_losses:
                    margin_terms.append(torch.cat(pair_losses).mean())

            # Unmatched track/detection false-association rejection loss
            unm_t = ~pos_mask.any(dim=1)
            if unm_t.any():
                s_false_t = S[unm_t].max(dim=1).values
                rej_terms.append(F.softplus(beta * (s_false_t - tau_neg)) / beta)
            unm_d = ~pos_mask.any(dim=0)
            if unm_d.any():
                s_false_d = S[:, unm_d].max(dim=0).values
                rej_terms.append(F.softplus(beta * (s_false_d - tau_neg)) / beta)

        l_margin = torch.stack(margin_terms).mean() if margin_terms else td_sim.sum() * 0.0
        if rej_terms:
            all_rej = torch.cat(rej_terms)
            l_rej = all_rej.mean()
            rej_count = float(len(all_rej))
        else:
            l_rej = td_sim.sum() * 0.0
            rej_count = 0.0

        tracks._last_bidir_margin_loss = l_margin.detach()
        tracks._last_unmatched_rej_loss = l_rej.detach()
        tracks._last_unmatched_rej_count = torch.tensor(rej_count, device=device)
        return base_loss + self.lambda_margin * l_margin + self.lambda_rej * l_rej

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        tracks = out["tracks"]
        if hasattr(tracks, "_last_bidir_margin_loss"):
            self.log_dict(
                {
                    "train/base_sim_loss": tracks._last_base_loss,
                    "train/bidir_margin_loss": tracks._last_bidir_margin_loss,
                    "train/unmatched_rej_loss": tracks._last_unmatched_rej_loss,
                    "train/unmatched_rej_count": tracks._last_unmatched_rej_count,
                },
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        return out

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        tracks = out["tracks"]
        if hasattr(tracks, "_last_bidir_margin_loss"):
            self.log_dict(
                {
                    "val/base_sim_loss": tracks._last_base_loss,
                    "val/bidir_margin_loss": tracks._last_bidir_margin_loss,
                    "val/unmatched_rej_loss": tracks._last_unmatched_rej_loss,
                    "val/unmatched_rej_count": tracks._last_unmatched_rej_count,
                },
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        return out


class S12EpochLogger(s09.S09EpochLogger):
    """Epoch logger appending S12 component losses cleanly."""

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
            ("train", "train/bidir_margin_loss", "loss"),
            ("train", "train/unmatched_rej_loss", "loss"),
            ("train", "train/unmatched_rej_count", "count"),
            ("val", "val/base_sim_loss", "loss"),
            ("val", "val/bidir_margin_loss", "loss"),
            ("val", "val/unmatched_rej_loss", "loss"),
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
            self.write_curves()


def dual_softmax_associate(td_sim_matrix: torch.Tensor, *, tau_ds: float = 0.10, dustbin_score: float = 0.40) -> tuple[np.ndarray, list[int], list[int], dict[str, int]]:
    """Dustbin-augmented row & column dual-softmax assignment (strictly GT-free)."""
    T, D = td_sim_matrix.shape
    if T == 0:
        return np.empty((0, 2), dtype=int), [], list(range(D)), {"dustbin_rejected": 0, "valid_mutual_pairs": 0}
    if D == 0:
        return np.empty((0, 2), dtype=int), list(range(T)), [], {"dustbin_rejected": 0, "valid_mutual_pairs": 0}

    S = td_sim_matrix.detach().cpu().to(torch.float64)
    row_aug = torch.cat([S, torch.full((T, 1), float(dustbin_score), dtype=torch.float64)], dim=1)
    p_row = torch.softmax(row_aug / float(tau_ds), dim=1)
    p_row_ij, p_row_dust = p_row[:, :D], p_row[:, D : D + 1]

    col_aug = torch.cat([S, torch.full((1, D), float(dustbin_score), dtype=torch.float64)], dim=0)
    p_col = torch.softmax(col_aug / float(tau_ds), dim=0)
    p_col_ij, p_col_dust = p_col[:T, :], p_col[T : T + 1, :]

    valid = (p_row_ij >= p_row_dust) & (p_col_ij >= p_col_dust)
    s_cal = torch.sqrt((p_row_ij * p_col_ij).clamp(min=0.0))
    cost = torch.where(valid, -s_cal, torch.full_like(s_cal, 1e6)).numpy()

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_list: list[list[int]] = []
    dust_rej = 0
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if bool(valid[r, c].item()):
            matched_list.append([r, c])
        else:
            dust_rej += 1

    matched_arr = np.asarray(matched_list, dtype=int).reshape(-1, 2) if matched_list else np.empty((0, 2), dtype=int)
    m_trks = {int(m[0]) for m in matched_list}
    m_dets = {int(m[1]) for m in matched_list}
    return (
        matched_arr,
        [i for i in range(T) if i not in m_trks],
        [j for j in range(D) if j not in m_dets],
        {"dustbin_rejected": dust_rej, "valid_mutual_pairs": int(valid.sum().item())},
    )


def apply_b2_margin_2cycle_defer(
    tracklets: list[Any],
    detections: list[Any],
    matched: np.ndarray,
    unmatched_trks: list[int],
    unmatched_dets: list[int],
    td_sim_matrix: torch.Tensor,
) -> tuple[np.ndarray, list[int], list[int], dict[str, int]]:
    """Apply Stage B/D Step 1 (causal 2-cycle fix) and Step 2 (extreme ambiguity single-frame coast)."""
    T, D = len(tracklets), len(detections)
    if T == 0 or D == 0 or len(matched) == 0:
        return matched, list(unmatched_trks), list(unmatched_dets), {"twocycle_swaps": 0, "deferred_matches": 0}

    signals = s11.compute_frame_conflict_signals(tracklets, detections, td_sim_matrix)
    sim_np = signals["sim_np"]
    det_other_iou = signals["det_max_other_iou"]
    trk_other_iou = signals["trk_max_other_iou"]

    # Precompute causal constant-velocity extrapolated boxes [T, 4] in raw pixel coordinates
    det_boxes = np.asarray([d.bbox_ltwh.detach().cpu().numpy().astype(np.float64) for d in detections], dtype=np.float64)
    trk_last_boxes = np.asarray([t.last_detection.bbox_ltwh.detach().cpu().numpy().astype(np.float64) for t in tracklets], dtype=np.float64)
    S_max = max(len(t.detections) for t in tracklets)
    t_box_tensor = torch.zeros((1, T, S_max, 4), dtype=torch.float32)
    t_age_tensor = torch.full((1, T, S_max, 1), 1.0, dtype=torch.float32)
    t_mask = torch.zeros((1, T, S_max), dtype=torch.bool)
    for ti, trk in enumerate(tracklets):
        for k, d in enumerate(reversed(trk.detections)):
            t_box_tensor[0, ti, k] = d.bbox_ltwh.detach().cpu().to(torch.float32)
            a_val = float(d.age.detach().cpu().item()) if torch.is_tensor(d.age) else float(d.age)
            t_age_tensor[0, ti, k, 0] = max(1.0, a_val)
            t_mask[0, ti, k] = True
    b_hat_np = compute_causal_extrapolation(t_box_tensor, t_age_tensor, t_mask)["b_hat"][0].numpy()

    def _iou(ba: np.ndarray, bb: np.ndarray) -> float:
        return s09.box_iou_ltwh(ba, bb)

    # Step 1: 2-cycle kinematic disambiguation among mature active matches
    pairs = [[int(m[0]), int(m[1])] for m in matched.tolist()]
    n_pairs = len(pairs)
    candidates: list[tuple[float, int, int]] = []
    for a in range(n_pairs):
        ta, da = pairs[a]
        if tracklets[ta].hits < 3:
            continue
        ma, _, _ = s11.compute_two_sided_margin(sim_np, ta, da)
        for b in range(a + 1, n_pairs):
            tb, db = pairs[b]
            if tracklets[tb].hits < 3:
                continue
            mb, _, _ = s11.compute_two_sided_margin(sim_np, tb, db)
            if min(ma, mb) >= 0.10:
                continue
            spatial_ov = max(_iou(det_boxes[da], det_boxes[db]), _iou(trk_last_boxes[ta], trk_last_boxes[tb]))
            if spatial_ov < 0.30:
                continue
            if min(sim_np[ta, db], sim_np[tb, da]) < 0.30:
                continue
            orig_sum = _iou(b_hat_np[ta], det_boxes[da]) + _iou(b_hat_np[tb], det_boxes[db])
            swap_sum = _iou(b_hat_np[ta], det_boxes[db]) + _iou(b_hat_np[tb], det_boxes[da])
            gain = swap_sum - orig_sum
            if gain > 0.10:
                candidates.append((gain, a, b))

    swapped_indices: set[int] = set()
    twocycle_swaps = 0
    for _gain, a, b in sorted(candidates, reverse=True):
        if a in swapped_indices or b in swapped_indices:
            continue
        pairs[a][1], pairs[b][1] = pairs[b][1], pairs[a][1]
        swapped_indices.add(a)
        swapped_indices.add(b)
        twocycle_swaps += 1

    # Step 2: Extreme-ambiguity single-frame coast (mature active matches only)
    kept_pairs: list[list[int]] = []
    u_trks = list(unmatched_trks)
    u_dets = list(unmatched_dets)
    deferred = 0
    for ti, di in pairs:
        trk = tracklets[ti]
        sim_val = float(sim_np[ti, di])
        margin, _, _ = s11.compute_two_sided_margin(sim_np, ti, di)
        max_iou = max(float(det_other_iou[di]), float(trk_other_iou[ti]))
        if trk.hits >= 3 and getattr(trk, "state", "") == "active" and margin < 0.05 and max_iou >= 0.30 and sim_val < 0.65:
            u_trks.append(ti)
            u_dets.append(di)
            deferred += 1
        else:
            kept_pairs.append([ti, di])

    out_matched = np.asarray(kept_pairs, dtype=int).reshape(-1, 2) if kept_pairs else np.empty((0, 2), dtype=int)
    return out_matched, u_trks, u_dets, {"twocycle_swaps": twocycle_swaps, "deferred_matches": deferred}


class S12AssignmentDisambiguatedCAMELTrack:
    """GT-free online inference wrapper applying B1 (dual-softmax) or B2 (2-cycle + defer)."""

    def __init__(self, base_mod: Any, mode: str) -> None:
        assert mode in ("B1_DUAL_SOFTMAX", "B2_MARGIN_2CYCLE_DEFER")
        self.base = base_mod
        self.mode = mode
        self.stats = {
            "total_frames": 0,
            "total_matches_initial": 0,
            "total_matches_final": 0,
            "dustbin_rejected": 0,
            "twocycle_swaps": 0,
            "deferred_matches": 0,
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
        self.stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        T, D = len(self.base.tracklets), len(detections)
        if T == 0:
            matched, u_trks, u_dets, td_sim = np.empty((0, 2), dtype=int), [], list(range(D)), torch.empty((0, 0))
        elif D == 0:
            matched, u_trks, u_dets, td_sim = np.empty((0, 2), dtype=int), list(range(T)), [], torch.empty((0, 0))
        else:
            batch = self.base.build_camel_batch(self.base.tracklets, detections)
            _, ass_res, td_sim_full = self.base.CAMEL.predict_step(batch, self.base.frame_count)
            td_sim = td_sim_full.squeeze(0)
            if self.mode == "B1_DUAL_SOFTMAX":
                matched, u_trks, u_dets, st = dual_softmax_associate(td_sim[:T, :D], tau_ds=0.10, dustbin_score=0.40)
                self.stats["total_matches_initial"] += len(matched) + st["dustbin_rejected"]
                self.stats["dustbin_rejected"] += st["dustbin_rejected"]
            else:
                matched = ass_res[0]["matched_td_indices"]
                u_trks = ass_res[0]["unmatched_trackers"]
                u_dets = ass_res[0]["unmatched_detections"]
                self.stats["total_matches_initial"] += len(matched)
                matched, u_trks, u_dets, st2 = apply_b2_margin_2cycle_defer(
                    self.base.tracklets, detections, matched, u_trks, u_dets, td_sim[:T, :D]
                )
                self.stats["twocycle_swaps"] += st2["twocycle_swaps"]
                self.stats["deferred_matches"] += st2["deferred_matches"]

        self.stats["total_matches_final"] += len(matched)
        assert len(set([int(m[0]) for m in matched] + list(u_trks))) == T
        assert len(set([int(m[1]) for m in matched] + list(u_dets))) == D

        for m in matched:
            trk = self.base.tracklets[int(m[0])]
            det = detections[int(m[1])]
            trk.update(det)
            det.similarity_with_tracklet = td_sim[int(m[0]), int(m[1])]
            det.similarities = td_sim[:T, int(m[1])]

        for i in u_dets:
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


def compose_s12_train_config(run_dir: Path, train_vids: list[str], val_vids: list[str], dm_cache_dir: Path, condition: str, max_epochs: int = 20, num_workers: int = 4) -> Any:
    cfg = s09.compose_s09_train_config(
        run_dir=run_dir,
        train_vids=train_vids,
        val_vids=val_vids,
        train_state=s09.S08_STATES_DIR / "states_train32.pklz",
        val_state=s09.S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=dm_cache_dir,
        stage1_cue=None,
        te_checkpoints=S09_C1_CKPTS,
        max_epochs=max_epochs,
        num_workers=num_workers,
    )
    OmegaConf.set_struct(cfg.modules.track.CAMEL, False)
    cfg.modules.track.CAMEL._target_ = "run_s12_camel.S12PairwiseCAMEL"
    cfg.modules.track.CAMEL.arch_variant = S12_ARCH_MAP[condition]
    cfg.modules.track.CAMEL.lambda_margin = 0.5
    cfg.modules.track.CAMEL.lambda_rej = 0.25
    cfg.modules.track.CAMEL.margin_target = 0.15
    cfg.modules.track.CAMEL.tau_neg = 0.35
    cfg.modules.track.CAMEL.beta = 10.0
    OmegaConf.set_struct(cfg.modules.track.CAMEL, True)
    return cfg


def extra_param_summary(model: S12PairwiseCAMEL) -> dict[str, Any]:
    """Summarize total and newly added S12 parameters (excluding C1 temp_encs & vanilla GAFFE)."""
    ps = s09.param_summary(model)
    new_prefixes = ("node_t_mlp", "node_d_mlp", "edge_attn_td", "edge_attn_dt", "self_dd_attn", "self_tt_attn", "sim_proj_t", "sim_proj_d", "edge_sim_mlp")
    extra_params = sum(p.numel() for n, p in model.named_parameters() if n.split(".")[0] in new_prefixes)
    ps["s12_extra_params"] = {
        "total_params": int(extra_params),
        "fraction_of_total": float(extra_params / ps["total"]["total_params"]),
    }
    return ps


def cmd_stage_a_lock(_args: argparse.Namespace) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=False)
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    train32, dev8 = s09.load_split_lists()

    # 1. Verify all S08/S09/S10/S11 reused assets
    s09_audit = json.loads((REPO_ROOT / "reports/diagnostics/s09_camel_calibration_and_attribution_20261126/source_audit.json").read_text())
    expected = {
        "t_twostage_joint_ckpt": (S09_T_TWOSTAGE_CKPT, protocol["reused_assets"]["t_twostage_joint_ckpt"]["sha256"]),
        "c1_app_encoder_ckpt": (S09_C1_CKPTS["app_encoder"], protocol["reused_assets"]["c1_te_ckpts"]["app_encoder"]["sha256"]),
        "c1_kp_encoder_ckpt": (S09_C1_CKPTS["kp_encoder"], protocol["reused_assets"]["c1_te_ckpts"]["kp_encoder"]["sha256"]),
        "c1_bbox_encoder_ckpt": (S09_C1_CKPTS["bbox_encoder"], protocol["reused_assets"]["c1_te_ckpts"]["bbox_encoder"]["sha256"]),
        "split_manifest": (s09.SPLIT_PATH, protocol["data"]["membership_sha256"]),
        "states_train32": (s09.S08_STATES_DIR / "states_train32.pklz", s09_audit["reused_s08_assets_verified"]["states_train32"]["sha256"]),
        "states_dev8_train_ref": (s09.S08_STATES_DIR / "states_dev8_train_ref.pklz", s09_audit["reused_s08_assets_verified"]["states_dev8_train_ref"]["sha256"]),
        "states_dev8_gt_free": (s09.S08_STATES_DIR / "states_dev8_gt_free.pklz", s09_audit["reused_s08_assets_verified"]["states_dev8_gt_free"]["sha256"]),
        "s09_protocol": (REPO_ROOT / "manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml", protocol["reused_assets"]["s09_protocol_sha256"]),
        "s10_protocol": (REPO_ROOT / "manifests/protocols/s10_camel_anticontamination_20261127.yaml", protocol["reused_assets"]["s10_protocol_sha256"]),
        "s11_protocol": (REPO_ROOT / "manifests/protocols/s11_camel_onset_and_conflict_gate_20261127.yaml", protocol["reused_assets"]["s11_protocol_sha256"]),
    }
    verified_assets = {}
    for k, (path, exp_sha) in expected.items():
        actual = s09.sha256_file(path)
        if actual != exp_sha:
            raise RuntimeError(f"Asset SHA mismatch for {k} ({path}): {actual} != {exp_sha}")
        verified_assets[k] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size, "verified": True}

    dm_cache = ensure_shared_dm_cache()

    # 2. Check parameter counts across M0..M3 and write resolved YAML configs
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dir = REPORT_DIR / "resolved_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_b = s09.compose_s09_infer_config(
        run_dir=RUNS_ROOT / "_resolved_b",
        train_vids=train32,
        val_vids=dev8,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=S09_T_TWOSTAGE_CKPT,
        sim_threshold=0.40,
    )
    (cfg_dir / "stage_b_infer_th040.yaml").write_text(OmegaConf.to_yaml(cfg_b, resolve=True))

    param_table: dict[str, Any] = {}
    for cond in ("M0_NODE_CAP_CTRL", "M1_C_KIN_EDGE", "M2_C_KIN_DOA_EDGE", "M3_C_PRIME_PLUS_A_LOSS"):
        cfg_c = compose_s12_train_config(RUNS_ROOT / f"_resolved_{cond}", train32, dev8, dm_cache, cond)
        (cfg_dir / f"stage_c_{cond.lower()}.yaml").write_text(OmegaConf.to_yaml(cfg_c, resolve=True))
        m_c = instantiate(cfg_c.modules.track.CAMEL, _recursive_=False)
        param_table[cond] = extra_param_summary(m_c)

    m0_extra = param_table["M0_NODE_CAP_CTRL"]["s12_extra_params"]["total_params"]
    m2_extra = param_table["M2_C_KIN_DOA_EDGE"]["s12_extra_params"]["total_params"]
    assert m0_extra >= m2_extra, f"M0 extra params ({m0_extra}) < M2 extra params ({m2_extra})"

    # 3. 200-batch train32 unmatched / IoU>=0.30 / drop_bbox audit
    pl.seed_everything(13, workers=True)
    cfg_probe = compose_s12_train_config(RUNS_ROOT / "_stage_a_probe", train32, dev8, dm_cache, "M0_NODE_CAP_CTRL", num_workers=4)
    td = instantiate(cfg_probe.dataset)
    m_probe = instantiate(cfg_probe.modules.track.CAMEL, _recursive_=False)
    dm = instantiate(cfg_probe.modules.track.datamodule_cfg, tracking_dataset=td, pipeline=Pipeline([]), datasets={})
    dm.prepare_data()
    dm.setup("fit")
    tot_trks, unm_trks, tot_dets, unm_dets, tot_pairs, iou30_pairs, drop_trks = 0, 0, 0, 0, 0, 0, 0
    for bi, batch in enumerate(dm.train_dataloader()):
        if bi >= 200:
            break
        batch = m_probe.norm_coords(batch)
        t_valid = (~batch["track_targets"].isnan()).any(dim=-1)
        d_valid = (~batch["det_targets"].isnan()).any(dim=-1)
        t_id = batch["track_targets"][:, :, 0]
        d_id = batch["det_targets"][:, :, 0]
        t_boxes = batch["track_feats"]["bbox_ltwh"][:, :, 0]
        d_boxes = batch["det_feats"]["bbox_ltwh"][:, :, 0]
        if "drop_bbox" in batch["track_feats"]:
            drop_trks += int(((batch["track_feats"]["drop_bbox"][:, :, 0, 0] == 1) & t_valid).sum().item())
        for b in range(t_valid.shape[0]):
            ti = torch.where(t_valid[b])[0]
            di = torch.where(d_valid[b])[0]
            tot_trks += len(ti)
            tot_dets += len(di)
            tset = set(t_id[b, ti].tolist())
            dset = set(d_id[b, di].tolist())
            unm_trks += sum(1 for x in t_id[b, ti].tolist() if x not in dset)
            unm_dets += sum(1 for x in d_id[b, di].tolist() if x not in tset)
            tb, db = t_boxes[b, ti], d_boxes[b, di]
            if len(tb) > 0 and len(db) > 0:
                ix1 = torch.maximum(tb[:, None, 0], db[None, :, 0])
                iy1 = torch.maximum(tb[:, None, 1], db[None, :, 1])
                ix2 = torch.minimum(tb[:, None, 0] + tb[:, None, 2], db[None, :, 0] + db[None, :, 2])
                iy2 = torch.minimum(tb[:, None, 1] + tb[:, None, 3], db[None, :, 1] + db[None, :, 3])
                inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                union = (tb[:, None, 2] * tb[:, None, 3]).clamp(min=0) + (db[None, :, 2] * db[None, :, 3]).clamp(min=0) - inter
                iou = inter / union.clamp(min=1e-9)
                tot_pairs += int(iou.numel())
                iou30_pairs += int((iou >= 0.30).sum().item())

    source_audit = {
        "schema_version": "s12-source-audit-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "upstream_cameltrack_commit": "bc1c261075b0beacfe71c2896453be83643153c8",
        "gaffe_attention_interface": (
            "Upstream cameltrack/architecture/gaffe.py uses 4-layer nn.TransformerEncoder(batch_first=True, emb_dim=1024, n_heads=8) "
            "over [dets.tokens (0..D-1), tracks.tokens (D..D+T-1), cls (D+T)]. S12 injects per-layer head biases B^(l,h) into "
            "each TransformerEncoderLayer via unified float mask [B*n_heads, L, L] (setting padded keys to -1e9 and preserving diagonal 0.0)."
        ),
        "sim_loss_exact_spec": {
            "class": "pytorch_metric_learning.losses.NTXentLoss",
            "temperature": float(getattr(m_probe.sim_loss, "temperature", 0.07)),
            "distance": type(m_probe.sim_loss.distance).__name__,
            "reducer": type(m_probe.sim_loss.reducer).__name__,
            "similarity_metric": "norm_euclidean_sim_matrix (1 - ||u_hat - v_hat||_2 / 2 in [0, 1])",
        },
        "bbox_dropout_and_coords_resolution": (
            "DropoutFeatures sets drop_bbox=1 while preserving numeric bbox_ltwh; compute_causal_extrapolation masks drop_bbox==1 "
            "as invalid alongside NaNs and non-positive w/h. norm_coords='positive' scales both det and tracklet bbox_ltwh to [0,1] "
            "relative to im_width/im_height before CAMEL.forward during both training and inference."
        ),
        "train32_200_batch_audit": {
            "num_batches": 200,
            "total_tracklets": tot_trks,
            "unmatched_tracklets": unm_trks,
            "unmatched_tracklet_frac": unm_trks / max(tot_trks, 1),
            "total_dets": tot_dets,
            "unmatched_dets": unm_dets,
            "unmatched_det_frac": unm_dets / max(tot_dets, 1),
            "total_pairs": tot_pairs,
            "iou_ge_030_pairs": iou30_pairs,
            "iou_ge_030_pair_frac": iou30_pairs / max(tot_pairs, 1),
            "drop_bbox_tracklets": drop_trks,
            "drop_bbox_frac": drop_trks / max(tot_trks, 1),
        },
        "verified_assets": verified_assets,
        "shared_datamodule_cache": str(dm_cache),
        "model_parameters": param_table,
    }
    (REPORT_DIR / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")

    runtime_lock = {
        "schema_version": "s12-runtime-lock-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_safety_probe": gpu_info,
        "scripts_sha256": {
            "scripts/run_s12_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s12_camel.py"),
            "scripts/run_s11_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s11_camel.py"),
            "scripts/run_s09_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s09_camel.py"),
            "tests/test_s12_camel.py": s09.sha256_file(REPO_ROOT / "tests/test_s12_camel.py"),
        },
    }
    (REPORT_DIR / "runtime_lock.json").write_text(json.dumps(runtime_lock, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "m0_extra": m0_extra, "m2_extra": m2_extra, "audit": source_audit["train32_200_batch_audit"]}, indent=2))


def cmd_train_c(args: argparse.Namespace) -> None:
    s09.apply_s09_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    s09.setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    dm_cache = ensure_shared_dm_cache()
    train32, dev8 = s09.load_split_lists()
    cfg = compose_s12_train_config(run_dir, train32, dev8, dm_cache, args.condition, max_epochs=20, num_workers=4)

    t0 = time.perf_counter()
    pl.seed_everything(13, workers=True)
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL

    init_prefixes = ["gaffe"] + [f"temp_encs.{k}" for k in model.temp_encs.keys()]
    new_prefixes = [n for n in ("node_t_mlp", "node_d_mlp", "edge_attn_td", "edge_attn_dt", "self_dd_attn", "self_tt_attn", "sim_proj_t", "sim_proj_d", "edge_sim_mlp") if hasattr(model, n)]
    before = {
        prefix: {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith(prefix + ".")}
        for prefix in (init_prefixes + new_prefixes)
    }
    params_info = extra_param_summary(model)

    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})
    epoch_cb = S12EpochLogger(run_dir, args.condition, t0)
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

    summary = {
        "schema_version": "s12-train-v1",
        "status": "COMPLETED",
        "condition": args.condition,
        "arch_variant": S12_ARCH_MAP[args.condition],
        "run_id": args.run_id,
        "seed": 13,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "gpu_device_hours": (t1 - t0) / 3600.0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "completed_epochs": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "nonfinite_grad_steps": epoch_cb.nonfinite_grad_steps,
        "parameters": params_info,
        "module_deltas_init_to_end": deltas,
        "last_checkpoint": {
            "path": str(last_ckpt),
            "sha256": last_sha,
            "bytes": last_ckpt.stat().st_size,
            "reload_state_dict_max_abs_diff": max_reload_diff,
        },
        "epoch20_val_computed_sim_threshold": epoch20_comp_th,
        "learning_curves_csv": str(run_dir / "learning_curves.csv"),
        "datamodule_cache_dir": str(dm_cache),
    }
    (run_dir / "stage_c_train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "condition": args.condition, "nonfinite": summary["nonfinite_grad_steps"], "reload_diff": max_reload_diff, "last_ckpt_sha256": last_sha}, indent=2))


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
    if args.s12_model:
        OmegaConf.set_struct(cfg.modules.track.CAMEL, False)
        cfg.modules.track.CAMEL._target_ = "run_s12_camel.S12PairwiseCAMEL"
        OmegaConf.set_struct(cfg.modules.track.CAMEL, True)

    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    ov = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    ov["sim_threshold"] = sim_threshold
    ov["use_computed_sim_threshold"] = False
    camel_cls = S12PairwiseCAMEL if args.s12_model else type(base_mod.CAMEL)
    base_mod.CAMEL = camel_cls.load_from_checkpoint(ckpt_path, map_location=base_mod.device, **ov)
    assert abs(float(base_mod.CAMEL.sim_threshold) - sim_threshold) < 1e-12
    assert bool(base_mod.CAMEL.use_computed_sim_threshold) is False

    wrapped_mod = S12AssignmentDisambiguatedCAMELTrack(base_mod, args.disambig) if args.disambig != "NONE" else base_mod
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
        "schema_version": "s12-eval-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "condition": args.condition,
        "inference_mode": "online_closed_loop",
        "disambig_mode": args.disambig,
        "s12_model": bool(args.s12_model),
        "gt_role": "zero_gt_accessed_gt_free_states",
        "seed": "13",
        "scope_id": "dev8",
        "sim_threshold": sim_threshold,
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "state_file": str(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "state_file_sha256": s09.sha256_file(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "wall_seconds": t1 - t0,
        "tracking_seconds": tracking_sec,
        "ms_per_frame": 1000.0 * tracking_sec / max(n_frames, 1),
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
        "disambig_stats": wrapped_mod.stats if args.disambig != "NONE" else None,
    }
    (run_dir / "stage_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "condition": args.condition, "sim_threshold": sim_threshold, "ms_per_frame": round(summary["ms_per_frame"], 2), "combined": summary["combined_metrics_preview"], "disambig_stats": summary["disambig_stats"]}, indent=2))


def cmd_stage_d_anatomy_replay(args: argparse.Namespace) -> None:
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
    cfg.modules.track.CAMEL._target_ = "run_s12_camel.S12PairwiseCAMEL"
    OmegaConf.set_struct(cfg.modules.track.CAMEL, True)

    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    ov = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    ov["sim_threshold"] = sim_threshold
    ov["use_computed_sim_threshold"] = False
    base_mod.CAMEL = S12PairwiseCAMEL.load_from_checkpoint(ckpt_path, map_location=base_mod.device, **ov)

    gt_by_image_id, img_to_seq_fr = s11.load_dev8_gt_boxes(tracking_set)
    probe = s11.OnsetAnatomyReplayCAMELTrack(base_mod, gt_by_image_id, img_to_seq_fr)
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
    ref_summary = json.loads((RUNS_ROOT / args.ref_online_run_id / "stage_eval_summary.json").read_text())
    replay_hashes = {s: s09.sha256_file(mot_dir / f"{s}.txt") for s in dev8}
    exact_mot_match = all(replay_hashes[s] == ref_summary["mot_sha256_by_seq"][s] for s in dev8)
    assert exact_mot_match, f"Anatomy replay MOT mismatch against {args.ref_online_run_id}!"
    t1 = time.perf_counter()

    events_csv = run_dir / "d_onset_anatomy_events.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(probe.events[0].keys()))
        w.writeheader()
        w.writerows(probe.events)

    clean_ev = [e for e in probe.events if e["sublabel"] == "clean_match"]
    onset_ev = [e for e in probe.events if e["sublabel"] == "onset_cross"]
    post_ev = [e for e in probe.events if e["sublabel"] == "post_switch_cross"]
    fp_ev = [e for e in probe.events if e["sublabel"] == "fp_pollution"]
    true_present = [e for e in onset_ev if int(e["true_det_present_in_frame"]) == 1]

    summary = {
        "schema_version": "s12-onset-anatomy-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "ref_online_run_id": args.ref_online_run_id,
        "sim_threshold": sim_threshold,
        "checkpoint_sha256": ckpt_sha,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "exact_mot_match_vs_online": exact_mot_match,
        "sublabel_counts": {
            "clean_match": len(clean_ev),
            "onset_cross": len(onset_ev),
            "post_switch_cross": len(post_ev),
            "fp_pollution": len(fp_ev),
            "unanchored": sum(1 for e in probe.events if e["sublabel"] == "unanchored"),
            "total_events": len(probe.events),
        },
        "onset_anatomy": {
            "total_onset": len(onset_ev),
            "trk_gap_eq_1": sum(1 for e in onset_ev if int(e["trk_gap_frames"]) == 1),
            "trk_gap_gt_1": sum(1 for e in onset_ev if int(e["trk_gap_frames"]) > 1),
            "competing_iou_ge_030_any": sum(1 for e in onset_ev if float(e["max_conflict_iou"]) >= 0.30),
            "assignment_margin_lt_010": sum(1 for e in onset_ev if float(e["assignment_margin"]) < 0.10),
            "two_cycle_mutual_swap": sum(1 for e in onset_ev if int(e["is_2cycle_swap"]) == 1),
            "true_det_absent_in_frame": sum(1 for e in onset_ev if int(e["true_det_present_in_frame"]) == 0),
            "true_det_present_in_frame": len(true_present),
            "true_present_strong_reversal_lt_neg005": sum(1 for e in true_present if float(e["true_minus_assigned_sim"]) < -0.05),
            "true_present_tie_abs_le_005": sum(1 for e in true_present if abs(float(e["true_minus_assigned_sim"])) <= 0.05),
        },
        "events_csv": str(events_csv),
    }
    (run_dir / "stage_d_anatomy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "sim_threshold": sim_threshold, "exact_mot_match": exact_mot_match, "sublabel_counts": summary["sublabel_counts"], "onset_anatomy": summary["onset_anatomy"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage-a-lock")

    p_tc = sub.add_parser("train-c")
    p_tc.add_argument("--condition", choices=list(S12_ARCH_MAP.keys()), required=True)
    p_tc.add_argument("--run-id", required=True)

    p_eo = sub.add_parser("eval-online")
    p_eo.add_argument("--condition", required=True)
    p_eo.add_argument("--ckpt", required=True)
    p_eo.add_argument("--sim-threshold", type=float, required=True)
    p_eo.add_argument("--disambig", choices=["NONE", "B1_DUAL_SOFTMAX", "B2_MARGIN_2CYCLE_DEFER"], default="NONE")
    p_eo.add_argument("--s12-model", action="store_true")
    p_eo.add_argument("--run-id", required=True)

    p_da = sub.add_parser("stage-d-anatomy-replay")
    p_da.add_argument("--ckpt", required=True)
    p_da.add_argument("--sim-threshold", type=float, required=True)
    p_da.add_argument("--ref-online-run-id", required=True)
    p_da.add_argument("--run-id", required=True)

    args = parser.parse_args()
    dispatch = {
        "stage-a-lock": cmd_stage_a_lock,
        "train-c": cmd_train_c,
        "eval-online": cmd_eval_online,
        "stage-d-anatomy-replay": cmd_stage_d_anatomy_replay,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
