#!/usr/bin/env python3
"""S10 CAMELTrack skip-only decomposition, C0 write gate, and anti-contamination (C1/C2/C3) runner."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from collections import OrderedDict
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
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/home/imc/yzy/MOT/scripts")
import run_s09_camel as s09  # noqa: E402

import tracklab.main  # noqa: F401
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb as tl_wandb

from cameltrack.architecture.base_module import Module as CAMELBaseModule
from cameltrack.architecture.temporal_encoder import PositionalEncoding
from cameltrack.camel import CAMEL, Detections, Tracklets
from cameltrack.cameltrack import Tracklet
from cameltrack.train.callbacks import SimMetrics
from cameltrack.train.dataset import CAMELDataset
from cameltrack.train.transforms import Transform

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s10_camel_anticontamination_20261127.yaml"
REPORT_DIR = REPO_ROOT / "reports/feasibility/s10_camel_anticontamination_20261127"
RUNS_ROOT = DATA_ROOT / "runs/s10_camel_anticontamination_20261127"
CACHE_ROOT = DATA_ROOT / "cache/s10_camel"
SHARED_DM_CACHE = CACHE_ROOT / "dm_s10_train32_dev8_shared"
S09_RUNS_ROOT = DATA_ROOT / "runs/s09_camel_calibration_and_attribution_20261126"

S09_T_TWOSTAGE_CKPT = S09_RUNS_ROOT / "c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt"
S09_C1_CKPTS = {
    cue: S09_RUNS_ROOT / f"c1_{cue}_seed13_e20_v1/CAMEL/last.ckpt"
    for cue in s09.TE_CUES
}
SWAP_FEATURE_COLS = [
    "bbox_ltwh",
    "bbox_conf",
    "keypoints_xyc",
    "embeddings",
    "visibility_scores",
    "im_width",
    "im_height",
]


def apply_s10_runtime_compat_patches() -> None:
    """Enable S09 PyTorch 2.6 / pickle patches plus corrupted_mask feature extraction."""
    s09.apply_s09_runtime_compat_patches()
    if "corrupted_mask" not in CAMELDataset.feature_names:
        CAMELDataset.feature_names = list(CAMELDataset.feature_names) + ["corrupted_mask"]


class InitCorruptedMaskTransform(Transform):
    """Initialize boolean `corrupted_mask` column cleanly before swap/dropout transforms."""

    def __call__(self, tracks: pd.DataFrame, dets: Optional[pd.DataFrame] = None):
        if "corrupted_mask" not in tracks.columns:
            tracks["corrupted_mask"] = False
        if dets is not None and "corrupted_mask" not in dets.columns:
            dets["corrupted_mask"] = False
        return tracks, dets


def _center_xy(bbox_ltwh: np.ndarray) -> np.ndarray:
    return np.asarray([bbox_ltwh[0] + 0.5 * bbox_ltwh[2], bbox_ltwh[1] + 0.5 * bbox_ltwh[3]], dtype=np.float64)


def _vec_iou_dist_ltwh(box: np.ndarray, cboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized IoU and center Euclidean distance between one [4] box and [C, 4] boxes."""
    bx1, by1, bw, bh = box[0], box[1], box[2], box[3]
    bx2, by2 = bx1 + bw, by1 + bh
    cx1, cy1, cw, ch = cboxes[:, 0], cboxes[:, 1], cboxes[:, 2], cboxes[:, 3]
    cx2, cy2 = cx1 + cw, cy1 + ch

    ix1 = np.maximum(bx1, cx1)
    iy1 = np.maximum(by1, cy1)
    ix2 = np.minimum(bx2, cx2)
    iy2 = np.minimum(by2, cy2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = np.maximum(0.0, bw) * np.maximum(0.0, bh) + np.maximum(0.0, cw) * np.maximum(0.0, ch) - inter
    iou = np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)
    iou = np.nan_to_num(iou, nan=0.0)

    dx = (bx1 + 0.5 * bw) - (cx1 + 0.5 * cw)
    dy = (by1 + 0.5 * bh) - (cy1 + 0.5 * ch)
    dist = np.nan_to_num(np.hypot(dx, dy), nan=1e18)
    return iou, dist


def _extract_ltwh_array(tracks: pd.DataFrame) -> np.ndarray:
    """Extract [R, 4] float64 bounding boxes from DataFrame quickly."""
    col = tracks["bbox_ltwh"].to_numpy()
    try:
        return np.vstack(col).astype(np.float64, copy=False)
    except Exception:
        return np.asarray([np.asarray(x, dtype=np.float64) for x in col], dtype=np.float64)


def _build_frame_index(vids: Optional[np.ndarray], imgs: np.ndarray) -> dict[tuple[Any, int], np.ndarray]:
    """Map (video_id, image_id) -> 1D array of positional row indices in O(R log R)."""
    if vids is None:
        order = np.argsort(imgs, kind="mergesort")
        s_imgs = imgs[order]
        cuts = np.flatnonzero(s_imgs[1:] != s_imgs[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(order)]))
        return {(None, int(s_imgs[st])): order[st:en] for st, en in zip(starts, ends)}

    keys = list(zip(vids.tolist(), imgs.tolist()))
    buckets: dict[tuple[Any, int], list[int]] = {}
    for pos, (v, im) in enumerate(keys):
        k = (v, int(im))
        lst = buckets.get(k)
        if lst is None:
            buckets[k] = [pos]
        else:
            lst.append(pos)
    return {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}


def _build_track_groups(tids: np.ndarray, ages: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    """Group positional row indices by track_id, sorted ascending by age."""
    order = np.lexsort((ages, tids))
    s_tids = tids[order]
    cuts = np.flatnonzero(s_tids[1:] != s_tids[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    return [(s_tids[st], order[st:en]) for st, en in zip(starts, ends)]


def _apply_bulk_swaps(tracks: pd.DataFrame, dst_list: list[int], src_list: list[int]) -> None:
    """Apply collected (dst, src) feature overwrites in bulk without mutating donor/unswapped rows."""
    if not dst_list:
        return
    dst_idx = np.asarray(dst_list, dtype=np.int64)
    src_idx = np.asarray(src_list, dtype=np.int64)
    for col in SWAP_FEATURE_COLS:
        if col not in tracks.columns:
            continue
        arr = tracks[col].to_numpy().copy()
        if arr.dtype == object:
            arr[dst_idx] = [x.copy() if isinstance(x, np.ndarray) else x for x in arr[src_idx]]
        else:
            arr[dst_idx] = arr[src_idx]
        tracks[col] = arr
    corr = tracks["corrupted_mask"].to_numpy(dtype=bool).copy()
    corr[dst_idx] = True
    tracks["corrupted_mask"] = corr


class ContiguousTakeoverTransform(Transform):
    """C2/C3 training transform: contiguous cross-ID takeover of L in [2, 12] frames inside [-15, -1]."""

    def __init__(
        self,
        p_takeover: float = 0.55,
        min_len: int = 2,
        max_len: int = 12,
        tail_window: int = 15,
        max_length: int = 50,
        min_iou: float = 0.1,
    ) -> None:
        super().__init__()
        self.p_takeover = float(p_takeover)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.tail_window = int(tail_window)
        self.max_length = int(max_length)
        self.min_iou = float(min_iou)

    def __call__(self, tracks: pd.DataFrame, dets: Optional[pd.DataFrame] = None):
        if tracks.empty:
            return tracks, dets
        self.set_rng()
        if "corrupted_mask" not in tracks.columns:
            tracks["corrupted_mask"] = False

        tids = tracks["track_id"].to_numpy()
        ages = tracks["age"].to_numpy(dtype=np.float64)
        imgs = tracks["image_id"].to_numpy(dtype=np.int64)
        has_vid = "video_id" in tracks.columns
        vids = tracks["video_id"].to_numpy() if has_vid else None
        boxes = _extract_ltwh_array(tracks)

        frame_map = _build_frame_index(vids, imgs)
        track_groups = _build_track_groups(tids, ages)

        dst_list: list[int] = []
        src_list: list[int] = []

        for tid, pos_asc in track_groups:
            if self.rng.random() >= self.p_takeover:
                continue
            win_chron = pos_asc[: self.max_length][::-1]
            M = len(win_chron)
            l_hi = min(self.max_len, M // 3)
            if l_hi < self.min_len:
                continue
            L = int(self.rng.integers(self.min_len, l_hi + 1))
            end_lo = max(3 * L - 1, M - self.tail_window)
            end_hi = M - 1
            if end_lo > end_hi:
                continue
            end_pos = int(self.rng.integers(end_lo, end_hi + 1))
            start_pos = end_pos - L + 1
            target_pos = win_chron[start_pos : end_pos + 1]

            # Aggregate overlap stats per donor track_id across target frames
            donor_stats: dict[Any, list[Any]] = {}
            for tp in target_pos:
                v = vids[tp] if has_vid else None
                im = int(imgs[tp])
                fpos = frame_map.get((v, im))
                if fpos is None:
                    continue
                cpos = fpos[tids[fpos] != tid]
                if len(cpos) == 0:
                    continue
                ious, dists = _vec_iou_dist_ltwh(boxes[tp], boxes[cpos])
                ctids = tids[cpos]
                for cp, ctid, iou, dist in zip(cpos.tolist(), ctids.tolist(), ious.tolist(), dists.tolist()):
                    st = donor_stats.get(ctid)
                    if st is None:
                        donor_stats[ctid] = [1, iou, dist, [(tp, cp)]]
                    else:
                        # Handle rare duplicate rows in donor by keeping max count of covered frames or appending
                        st[0] += 1
                        if iou > st[1]:
                            st[1] = iou
                        st[2] += dist
                        st[3].append((tp, cp))

            if not donor_stats:
                continue

            best_overlap_key = (-1, -1.0, 0.0)
            best_overlap_pairs: Optional[list[tuple[int, int]]] = None
            best_nearest_key = (-1, -1e18)
            best_nearest_pairs: Optional[list[tuple[int, int]]] = None
            for cov, max_iou, sum_dist, pairs in donor_stats.values():
                mean_dist = sum_dist / cov
                if max_iou >= self.min_iou:
                    okey = (cov, max_iou, -mean_dist)
                    if okey > best_overlap_key:
                        best_overlap_key = okey
                        best_overlap_pairs = pairs
                nkey = (cov, -mean_dist)
                if nkey > best_nearest_key:
                    best_nearest_key = nkey
                    best_nearest_pairs = pairs

            chosen_pairs = best_overlap_pairs if best_overlap_pairs is not None else best_nearest_pairs
            if chosen_pairs:
                seen_tp: set[int] = set()
                for tp, cp in chosen_pairs:
                    if tp not in seen_tp:
                        seen_tp.add(tp)
                        dst_list.append(tp)
                        src_list.append(cp)

        _apply_bulk_swaps(tracks, dst_list, src_list)
        return tracks, dets


class MatchedIidSwapTransform(Transform):
    """C1 control transform: independent per-frame swaps within max_length window matched to C2 swap rate."""

    def __init__(
        self,
        p_iid: float = 0.085,
        max_length: int = 50,
        min_iou: float = 0.1,
    ) -> None:
        super().__init__()
        self.p_iid = float(p_iid)
        self.max_length = int(max_length)
        self.min_iou = float(min_iou)

    def __call__(self, tracks: pd.DataFrame, dets: Optional[pd.DataFrame] = None):
        if tracks.empty:
            return tracks, dets
        self.set_rng()
        if "corrupted_mask" not in tracks.columns:
            tracks["corrupted_mask"] = False

        tids = tracks["track_id"].to_numpy()
        ages = tracks["age"].to_numpy(dtype=np.float64)
        imgs = tracks["image_id"].to_numpy(dtype=np.int64)
        has_vid = "video_id" in tracks.columns
        vids = tracks["video_id"].to_numpy() if has_vid else None
        boxes = _extract_ltwh_array(tracks)

        frame_map = _build_frame_index(vids, imgs)
        track_groups = _build_track_groups(tids, ages)

        dst_list: list[int] = []
        src_list: list[int] = []

        for tid, pos_asc in track_groups:
            win = pos_asc[: self.max_length]
            rand_mask = self.rng.random(size=len(win)) < self.p_iid
            if not np.any(rand_mask):
                continue
            for tp in win[rand_mask].tolist():
                v = vids[tp] if has_vid else None
                im = int(imgs[tp])
                fpos = frame_map.get((v, im))
                if fpos is None:
                    continue
                cpos = fpos[tids[fpos] != tid]
                if len(cpos) == 0:
                    continue
                ious, dists = _vec_iou_dist_ltwh(boxes[tp], boxes[cpos])
                valid_iou = ious >= self.min_iou
                if np.any(valid_iou):
                    best_local = int(np.argmax(np.where(valid_iou, ious, -1.0)))
                else:
                    best_local = int(np.argmin(dists))
                dst_list.append(tp)
                src_list.append(int(cpos[best_local]))

        _apply_bulk_swaps(tracks, dst_list, src_list)
        return tracks, dets


class ConsistencyGateHead(nn.Module):
    """Lightweight strictly-causal multi-cue consistency estimator w_t in (0, 1)."""

    def __init__(self, in_dim: int = 704, hidden_dim: int = 128, n_heads: int = 4, n_layers: int = 2, ff_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs_tokens_cat: torch.Tensor, age: torch.Tensor, feats_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_tokens_cat: [B, N, S, in_dim], sorted age-ascending (0 is newest, S-1 oldest).
            age: [B, N, S, 1]
            feats_masks: bool [B, N, S]
        Returns:
            w_t: [B, N, S] in (0, 1), 0 on invalid padded slots
            logits: [B, N, S], 0 on invalid padded slots
        """
        B, N, S, _ = obs_tokens_cat.shape
        device = obs_tokens_cat.device
        dtype = obs_tokens_cat.dtype

        h = self.proj(obs_tokens_cat)
        h = self.pos_enc(h, age, feats_masks)
        h = self.norm(h)
        h = self.drop(h)

        # Flip to chronological order (0 is oldest, S-1 is newest) so strict causal mask prevents peeking ahead
        h_chron = h.flip(dims=[2]).reshape(B * N, S, -1)
        mask_chron = feats_masks.flip(dims=[2]).reshape(B * N, S)

        # Build unified float causal+padding mask (BN*n_heads, S, S) with diagonal always unmasked (0.0)
        causal_bad = torch.triu(torch.ones((S, S), dtype=torch.bool, device=device), diagonal=1)
        pad_bad = ~mask_chron.unsqueeze(1).expand(B * N, S, S)
        eye = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
        bad = (causal_bad.unsqueeze(0) | pad_bad) & (~eye)

        attn_mask = torch.zeros((B * N, S, S), device=device, dtype=dtype)
        attn_mask.masked_fill_(bad, -1e9)
        attn_mask = attn_mask.unsqueeze(1).expand(B * N, self.n_heads, S, S).reshape(B * N * self.n_heads, S, S)

        enc_chron = self.encoder(h_chron, mask=attn_mask)
        enc = enc_chron.reshape(B, N, S, -1).flip(dims=[2])
        logits = self.mlp(enc).squeeze(-1) * feats_masks.to(dtype)
        w_t = torch.sigmoid(logits) * feats_masks.to(dtype)
        return w_t, logits


def weighted_temporal_encode(
    temp_enc: Any,
    x: Tracklets,
    raw_tokens: torch.Tensor,
    obs_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run TemporalEncoder transformer using precomputed det_tokenizer tokens and optional consistency weights."""
    B, N, S, E, O = *x.feats_masks.shape, temp_enc.hidden_dim, temp_enc.output_dim
    output = torch.zeros((B, N, O), device=x.feats_masks.device, dtype=torch.float32)

    if obs_weights is None:
        tokens = temp_enc.pos_encoder(raw_tokens.clone(), x.feats["age"], x.feats_masks)
        tokens = temp_enc.src_norm(tokens)
        tokens = temp_enc.src_drop(tokens)
        src = tokens.permute(2, 0, 1, 3).reshape(S, B * N, E)
        cls = temp_enc.cls.expand(-1, B * N, -1)
        src = torch.cat([src, cls], dim=0)
        cls_mask = torch.ones((B * N, 1), dtype=torch.bool, device=x.feats_masks.device)
        src_mask = torch.cat([x.feats_masks.reshape(B * N, S), cls_mask], dim=1)
        src = temp_enc.encoder(src, src_key_padding_mask=~src_mask)
    else:
        tokens = temp_enc.pos_encoder(raw_tokens.clone(), x.feats["age"], x.feats_masks)
        tokens = temp_enc.src_norm(tokens)
        tokens = temp_enc.src_drop(tokens)
        src = tokens.permute(2, 0, 1, 3).reshape(S, B * N, E)
        cls = temp_enc.cls.expand(-1, B * N, -1)
        src = torch.cat([src, cls], dim=0)  # [S+1, B*N, E]

        n_heads = int(temp_enc.n_heads)
        flat_valid = x.feats_masks.reshape(B * N, S)
        log_w = torch.log(torch.clamp(obs_weights.reshape(B * N, S), min=1e-4, max=1.0))
        log_w = torch.where(flat_valid, log_w, torch.full_like(log_w, -1e9))

        attn_mask = torch.zeros((B * N, S + 1, S + 1), device=x.feats_masks.device, dtype=src.dtype)
        attn_mask[:, :, :S] = log_w.unsqueeze(1).expand(B * N, S + 1, S)
        # Preserve diagonal self-attention 0.0 so empty/fully-padded rows never hit NaN in softmax
        diag_idx = torch.arange(S + 1, device=x.feats_masks.device)
        attn_mask[:, diag_idx, diag_idx] = 0.0
        attn_mask = attn_mask.unsqueeze(1).expand(B * N, n_heads, S + 1, S + 1).reshape(B * N * n_heads, S + 1, S + 1)
        src = temp_enc.encoder(src, mask=attn_mask)

    src = src.permute(1, 0, 2).reshape(B, N, S + 1, E)
    tokens_out = src[:, :, -1, :]
    output[x.masks] = temp_enc.linear_out(tokens_out[x.masks])
    return output


class S10ConsistencyCAMEL(CAMEL):
    """CAMEL extended with lightweight causal multi-cue consistency head and attention-bias pooling (C3)."""

    def __init__(self, *args, cons_bce_weight: float = 0.2, cons_hidden_dim: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cons_bce_weight = float(cons_bce_weight)
        self.save_hyperparameters(ignore=[k for k in locals() if "checkpoint_path" in k])
        in_dim = sum(int(te.hidden_dim) for te in self.temp_encs.values())
        self.consistency_head = ConsistencyGateHead(in_dim=in_dim, hidden_dim=int(cons_hidden_dim))

    @torch.no_grad()
    def predict_consistency_weights(self, tracks: Tracklets) -> torch.Tensor:
        raw_tokens = [te.det_tokenizer(tracks) for te in self.temp_encs.values()]
        cat_tokens = torch.cat(raw_tokens, dim=-1)
        w_t, _ = self.consistency_head(cat_tokens, tracks.feats["age"], tracks.feats_masks)
        return w_t

    def tokenize(self, tracks: Tracklets, dets: Detections):
        tracks.tokens = {}
        dets.tokens = {}
        raw_track_tokens: dict[str, torch.Tensor] = {}
        for name, temp_enc in self.temp_encs.items():
            raw_track_tokens[name] = temp_enc.det_tokenizer(tracks)
            raw_det_tok = temp_enc.det_tokenizer(dets)
            dets.tokens[name] = weighted_temporal_encode(temp_enc, dets, raw_det_tok, obs_weights=None)

        cat_tokens = torch.cat([raw_track_tokens[name] for name in self.temp_encs.keys()], dim=-1)
        w_t, cons_logits = self.consistency_head(cat_tokens, tracks.feats["age"], tracks.feats_masks)
        tracks.cons_weights = w_t
        tracks.cons_logits = cons_logits

        for name, temp_enc in self.temp_encs.items():
            tracks.tokens[name] = weighted_temporal_encode(temp_enc, tracks, raw_track_tokens[name], obs_weights=w_t)
        return tracks, dets

    def compute_loss(self, tracks: Tracklets, dets: Detections, *args):
        sim_loss = super().compute_loss(tracks, dets, *args)
        tracks._last_sim_loss = sim_loss.detach()
        if hasattr(tracks, "cons_logits") and "corrupted_mask" in tracks.feats:
            valid_obs = tracks.feats_masks
            if valid_obs.any():
                m_t = tracks.feats["corrupted_mask"].squeeze(-1).nan_to_num(0.0).clamp(0.0, 1.0)
                target_clean = 1.0 - m_t
                cons_bce = F.binary_cross_entropy_with_logits(tracks.cons_logits[valid_obs], target_clean[valid_obs])
                cons_acc = ((tracks.cons_weights[valid_obs] >= 0.5) == (target_clean[valid_obs] >= 0.5)).float().mean()
            else:
                cons_bce = tracks.cons_logits.sum() * 0.0
                cons_acc = torch.tensor(1.0, device=sim_loss.device)
            tracks._last_cons_bce = cons_bce.detach()
            tracks._last_cons_acc = cons_acc.detach()
            return sim_loss + self.cons_bce_weight * cons_bce
        return sim_loss

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        tracks = out["tracks"]
        if hasattr(tracks, "_last_sim_loss"):
            self.log_dict(
                {
                    "train/sim_loss": tracks._last_sim_loss,
                    "train/cons_bce": tracks._last_cons_bce,
                    "train/cons_acc": tracks._last_cons_acc,
                },
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        return out

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        tracks = out["tracks"]
        if hasattr(tracks, "_last_sim_loss"):
            self.log_dict(
                {
                    "val/sim_loss": tracks._last_sim_loss,
                    "val/cons_bce": tracks._last_cons_bce,
                    "val/cons_acc": tracks._last_cons_acc,
                },
                on_epoch=True,
                on_step=False,
                logger=True,
            )
        return out


class S10EpochLogger(s09.S09EpochLogger):
    """Epoch logger adding S10 C3 consistency loss/accuracy curves cleanly."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if int(trainer.current_epoch) == 0 and trainer.sanity_checking:
            return
        super().on_validation_epoch_end(trainer, pl_module)
        # Replace extra metric units/append C3 auxiliary metrics if present
        epoch = int(trainer.current_epoch) + 1
        step = int(trainer.global_step)
        samples_seen = step * 8
        elapsed = time.perf_counter() - self.t_start
        ckpt_id = f"epoch{epoch}_last" if epoch == int(trainer.max_epochs) else ""
        metrics = trainer.callback_metrics
        extra = [
            ("train", "train/sim_loss", "loss"),
            ("train", "train/cons_bce", "bce"),
            ("train", "train/cons_acc", "ratio_0_1"),
            ("val", "val/sim_loss", "loss"),
            ("val", "val/cons_bce", "bce"),
            ("val", "val/cons_acc", "ratio_0_1"),
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


class SkipOnlyCleanGalleryCAMELTrack:
    """Stage B privileged D2_SKIP_ONLY: block cross-ID/FP writes cleanly with ZERO GT injection on misses."""

    def __init__(self, base_mod: Any, gt_by_image_id: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self.base = base_mod
        self.gt_by_image_id = gt_by_image_id
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn
        self.routing_stats = {
            "total_frames": 0,
            "total_predicted_matches": 0,
            "clean_matched_gallery_updates": 0,
            "cross_id_matches_skipped": 0,
            "fp_matches_skipped": 0,
            "unanchored_gallery_updates": 0,
            "gt_injected_on_miss": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        self.routing_stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(
            self.base.tracklets, detections
        )
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.base.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        img_id = int(metadatas.index[0]) if hasattr(metadatas, "index") else int(detections[0].image_id.item())
        gt_ids_f, gt_boxes_f = self.gt_by_image_id.get(img_id, (np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)))
        det_boxes = [d.bbox_ltwh.detach().cpu().numpy().astype(float) for d in detections]
        det_to_gt, _ = s09.match_detections_to_gt_frame(det_boxes, gt_ids_f, gt_boxes_f, iou_thresh=0.5)

        emitted_rows: list[dict[str, Any]] = []
        for m in matched:
            trk_idx, det_idx = int(m[0]), int(m[1])
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            sim_val = float(td_sim_matrix[trk_idx, det_idx].cpu().item())
            all_sims = {self.base.tracklets[j].id: float(s) for j, s in enumerate(td_sim_matrix[: len(self.base.tracklets), det_idx].cpu().numpy())}
            self.routing_stats["total_predicted_matches"] += 1

            anchor_gt = getattr(trk, "s10_gt_anchor", None)
            det_gt = det_to_gt.get(det_idx)
            if anchor_gt is not None and det_gt == anchor_gt:
                det_copy = copy.copy(det)
                if torch.is_tensor(det.age):
                    det_copy.age = det.age.clone().zero_()
                else:
                    det_copy.age = 0
                trk.detections.append(det_copy)
                trk.detections = trk.detections[-trk.max_gallery_size :]
                self.routing_stats["clean_matched_gallery_updates"] += 1
            elif anchor_gt is not None and det_gt is not None and det_gt != anchor_gt:
                self.routing_stats["cross_id_matches_skipped"] += 1
            elif anchor_gt is not None and det_gt is None:
                self.routing_stats["fp_matches_skipped"] += 1
            elif anchor_gt is None and det_gt is None:
                det_copy = copy.copy(det)
                if torch.is_tensor(det.age):
                    det_copy.age = det.age.clone().zero_()
                else:
                    det_copy.age = 0
                trk.detections.append(det_copy)
                trk.detections = trk.detections[-trk.max_gallery_size :]
                self.routing_stats["unanchored_gallery_updates"] += 1
            else:
                self.routing_stats["fp_matches_skipped"] += 1

            trk.hits += 1
            trk.hit_streak += 1
            trk.time_wo_hits = 0
            trk.last_detection = det
            self.base.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                emitted_rows.append(
                    {
                        "index": det.index.item(),
                        "track_id": trk.id,
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", sim_val),
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {"S": all_sims, "St": self.base.CAMEL.sim_threshold},
                    }
                )

        new_tracklets: list[Any] = []
        for i in unmatched_dets:
            det = detections[int(i)]
            if det.bbox_conf >= self.base.min_init_det_conf:
                trk = Tracklet(det, self.base.max_track_gallery_size)
                trk.s10_gt_anchor = det_to_gt.get(int(i))
                new_tracklets.append(trk)
                self.base.update_state(trk)
                if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                    emitted_rows.append(
                        {
                            "index": det.index.item(),
                            "track_id": trk.id,
                            "hits": trk.hits,
                            "age": trk.age,
                            "matched_with": None,
                            "time_since_update": trk.time_wo_hits,
                            "state": trk.state,
                            "costs": {"S": None, "St": self.base.CAMEL.sim_threshold},
                        }
                    )

        for u_idx in unmatched_trks:
            self.base.update_state(self.base.tracklets[int(u_idx)])

        self.base.tracklets.extend(new_tracklets)
        self.base.tracklets = [trk for trk in self.base.tracklets if trk.state != "dead"]
        self.base.frame_count += 1

        if emitted_rows:
            results = pd.DataFrame(emitted_rows).set_index("index", drop=True)
            assert set(results.index).issubset(detections_df.index)
            return results
        return pd.DataFrame(columns=["track_id", "hits", "age", "matched_with", "time_since_update", "state", "costs"])


class SimpleWriteGateCAMELTrack:
    """Stage B C0_SIMPLE_WRITE_GATE (GT-free): skip gallery write if mature trk.hits >= 3 and sim < tau_write."""

    def __init__(self, base_mod: Any, tau_write: float, min_hits_to_gate: int = 3) -> None:
        self.base = base_mod
        self.tau_write = float(tau_write)
        self.min_hits_to_gate = int(min_hits_to_gate)
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn
        self.gate_stats = {
            "total_frames": 0,
            "total_predicted_matches": 0,
            "mature_matches": 0,
            "write_skipped": 0,
            "write_accepted": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        self.gate_stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(
            self.base.tracklets, detections
        )
        sim_tensor = td_sim_matrix.detach().cpu() if torch.is_tensor(td_sim_matrix) and td_sim_matrix.numel() > 0 else torch.empty((0,))

        for m in matched:
            trk_idx, det_idx = int(m[0]), int(m[1])
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            sim_val = float(sim_tensor[trk_idx, det_idx].item())
            self.gate_stats["total_predicted_matches"] += 1
            was_mature = int(trk.hits) >= self.min_hits_to_gate
            if was_mature:
                self.gate_stats["mature_matches"] += 1
            if was_mature and sim_val < self.tau_write:
                self.gate_stats["write_skipped"] += 1
                trk.hits += 1
                trk.hit_streak += 1
                trk.time_wo_hits = 0
                trk.last_detection = det
            else:
                self.gate_stats["write_accepted"] += 1
                trk.update(det)
            det.similarity_with_tracklet = sim_tensor[trk_idx, det_idx]
            det.similarities = sim_tensor[: len(self.base.tracklets), det_idx]

        for i in unmatched_dets:
            if detections[i].bbox_conf >= self.base.min_init_det_conf:
                self.base.tracklets.append(Tracklet(detections[i], self.base.max_track_gallery_size))

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
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.item())
                        if trk.last_detection.similarity_with_tracklet is not None
                        else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.base.tracklets[j].id: float(sim) for j, sim in enumerate(trk.last_detection.similarities.numpy())}
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
        return pd.DataFrame(columns=["track_id", "hits", "age", "matched_with", "time_since_update", "state", "costs"])


def compute_candidate_w_last(base_mod: Any, cons_model: S10ConsistencyCAMEL, tracklets: list[Any], matched_dets: list[Any]) -> np.ndarray:
    """Strictly causal w_last for appending det[i] (at age=0) onto tracklets[i].detections (at age>=1)."""
    if not tracklets:
        return np.empty((0,), dtype=np.float32)
    T_max = max(len(t.detections) + 1 for t in tracklets)
    T_max = min(T_max, int(base_mod.max_track_gallery_size) + 1)
    device = cons_model.device
    feats: dict[str, torch.Tensor] = {}
    masks = torch.zeros((1, len(tracklets), T_max), dtype=torch.bool, device=device)

    all_cols = list(base_mod.input_columns) + ["index", "age", "im_width", "im_height"]
    for col in all_cols:
        per_trk = []
        for idx_t, (trk, det) in enumerate(zip(tracklets, matched_dets)):
            seq_dets = [det] + list(reversed(trk.detections))[: T_max - 1]
            if col == "age":
                det_age = torch.zeros_like(det.age) if torch.is_tensor(det.age) else torch.tensor([0.0], device=det.bbox_ltwh.device)
                if det_age.ndim == 0:
                    det_age = det_age.unsqueeze(0)
                vals = [
                    det_age
                    if k == 0
                    else (d.age if d.age.ndim > 0 else d.age.unsqueeze(0))
                    for k, d in enumerate(seq_dets)
                ]
            else:
                vals = [getattr(d, col) for d in seq_dets]
            stacked = torch.stack(vals)
            if stacked.shape[0] < T_max:
                pad_shape = (T_max - stacked.shape[0], *stacked.shape[1:])
                pad_val = float("nan") if stacked.is_floating_point() else 0
                pad = torch.full(pad_shape, pad_val, device=stacked.device, dtype=stacked.dtype)
                stacked = torch.cat([stacked, pad], dim=0)
            per_trk.append(stacked)
            if col == all_cols[0]:
                masks[0, idx_t, : len(seq_dets)] = True
        feats[col] = torch.stack(per_trk, dim=0).unsqueeze(0).to(device)

    det_feats = base_mod.build_detection_features(matched_dets)
    batch = {
        "det_feats": det_feats,
        "det_masks": torch.ones((1, len(matched_dets), 1), dtype=torch.bool, device=device),
        "track_feats": feats,
        "track_masks": masks,
    }
    tracks_obj, _ = cons_model.predict_preprocess(batch)
    w_t = cons_model.predict_consistency_weights(tracks_obj)
    return w_t[0, :, 0].detach().cpu().numpy().astype(np.float32)


class SoftConsistencyWriteFilterCAMELTrack:
    """Stage D C3_WITH_SOFT_WRITE_FILTER: skip write if mature trk.hits >= 3 and causal w_last < threshold."""

    def __init__(self, base_mod: Any, w_write_threshold: float = 0.50, min_hits_to_gate: int = 3) -> None:
        self.base = base_mod
        self.w_write_threshold = float(w_write_threshold)
        self.min_hits_to_gate = int(min_hits_to_gate)
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn
        self.gate_stats = {
            "total_frames": 0,
            "total_predicted_matches": 0,
            "mature_matches": 0,
            "write_skipped": 0,
            "write_accepted": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        self.gate_stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(
            self.base.tracklets, detections
        )
        sim_tensor = td_sim_matrix.detach().cpu() if torch.is_tensor(td_sim_matrix) and td_sim_matrix.numel() > 0 else torch.empty((0,))

        mature_pairs = [
            (int(m[0]), int(m[1]))
            for m in matched
            if int(self.base.tracklets[int(m[0])].hits) >= self.min_hits_to_gate
        ]
        w_last_map: dict[tuple[int, int], float] = {}
        if mature_pairs:
            m_trks = [self.base.tracklets[t_i] for t_i, _ in mature_pairs]
            m_dets = [detections[d_i] for _, d_i in mature_pairs]
            w_vals = compute_candidate_w_last(self.base, self.base.CAMEL, m_trks, m_dets)
            for pair, wv in zip(mature_pairs, w_vals.tolist()):
                w_last_map[pair] = float(wv)

        for m in matched:
            trk_idx, det_idx = int(m[0]), int(m[1])
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            self.gate_stats["total_predicted_matches"] += 1
            was_mature = (trk_idx, det_idx) in w_last_map
            if was_mature:
                self.gate_stats["mature_matches"] += 1
            w_last = w_last_map.get((trk_idx, det_idx), 1.0)
            if was_mature and w_last < self.w_write_threshold:
                self.gate_stats["write_skipped"] += 1
                trk.hits += 1
                trk.hit_streak += 1
                trk.time_wo_hits = 0
                trk.last_detection = det
            else:
                self.gate_stats["write_accepted"] += 1
                trk.update(det)
            det.similarity_with_tracklet = sim_tensor[trk_idx, det_idx]
            det.similarities = sim_tensor[: len(self.base.tracklets), det_idx]

        for i in unmatched_dets:
            if detections[i].bbox_conf >= self.base.min_init_det_conf:
                self.base.tracklets.append(Tracklet(detections[i], self.base.max_track_gallery_size))

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
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.item())
                        if trk.last_detection.similarity_with_tracklet is not None
                        else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.base.tracklets[j].id: float(sim) for j, sim in enumerate(trk.last_detection.similarities.numpy())}
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
        return pd.DataFrame(columns=["track_id", "hits", "age", "matched_with", "time_since_update", "state", "costs"])


def ensure_shared_dm_cache() -> Path:
    """Copy verified S09 datamodule cache into isolated S10 cache directory if not present."""
    src_dir = DATA_ROOT / "cache/s09_camel/dm_c2_twostage_joint_seed13_e20_v1_isolated"
    SHARED_DM_CACHE.mkdir(parents=True, exist_ok=True)
    for fname in ("camel_train.pklz", "camel_train_tracklets.json", "camel_val.pklz", "camel_val_tracklets.json"):
        src = src_dir / fname
        dst = SHARED_DM_CACHE / fname
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            tmp = SHARED_DM_CACHE / f".{fname}.tmp"
            shutil.copy2(src, tmp)
            tmp.replace(dst)
    return SHARED_DM_CACHE


def compose_s10_train_config(
    run_dir: Path,
    train_vids: list[str],
    val_vids: list[str],
    dm_cache_dir: Path,
    condition: str,
    p_iid: float = 0.085,
    max_epochs: int = 20,
    num_workers: int = 4,
) -> DictConfig:
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
    if condition == "C1_MATCHED_IID_SWAP":
        swap_entry = {"_target_": "run_s10_camel.MatchedIidSwapTransform", "p_iid": float(p_iid), "max_length": 50, "min_iou": 0.1}
    elif condition in ("C2_CONTIGUOUS_TAKEOVER", "C3_CONSISTENCY_GATED_TE"):
        swap_entry = {
            "_target_": "run_s10_camel.ContiguousTakeoverTransform",
            "p_takeover": 0.55,
            "min_len": 2,
            "max_len": 12,
            "tail_window": 15,
            "max_length": 50,
            "min_iou": 0.1,
        }
    else:
        raise ValueError(f"Unknown S10 train condition: {condition}")

    cfg.modules.track.datamodule_cfg.tracklet_transforms = OmegaConf.create({
        "train": {
            "_target_": "cameltrack.train.transforms.Compose",
            "transforms": [
                {"_target_": "run_s10_camel.InitCorruptedMaskTransform"},
                {"_target_": "cameltrack.train.transforms.MaxNumObs", "max_num_obs": 150},
                {"_target_": "cameltrack.train.transforms.DropoutFeatures", "p_drop_app": 0.4, "p_drop_bbox": 0.1, "p_drop_kps": 0.1},
                swap_entry,
                {
                    "_target_": "cameltrack.train.transforms.ProbabilisticTransform",
                    "probs": [0.4],
                    "transforms": [
                        {"_target_": "cameltrack.train.transforms.DropoutSporadic", "p_drop": 0.2, "sigma": 50, "method": "gaussian"},
                    ],
                },
            ],
        },
        "val": {
            "_target_": "cameltrack.train.transforms.Compose",
            "transforms": [{"_target_": "run_s10_camel.InitCorruptedMaskTransform"}],
        },
    })
    if condition == "C3_CONSISTENCY_GATED_TE":
        OmegaConf.set_struct(cfg.modules.track.CAMEL, False)
        cfg.modules.track.CAMEL._target_ = "run_s10_camel.S10ConsistencyCAMEL"
        cfg.modules.track.CAMEL.cons_bce_weight = 0.2
        cfg.modules.track.CAMEL.cons_hidden_dim = 128
        OmegaConf.set_struct(cfg.modules.track.CAMEL, True)
    return cfg


def _measure_single_swap_rate(
    dm_cache_dir: Path,
    train_vids: list[str],
    val_vids: list[str],
    condition: str,
    p_iid: float,
    num_batches: int = 200,
    tracking_dataset: Any = None,
) -> tuple[dict[str, Any], Any]:
    pl.seed_everything(13, workers=True)
    cfg = compose_s10_train_config(
        run_dir=RUNS_ROOT / "_stage_a_swap_probe",
        train_vids=train_vids,
        val_vids=val_vids,
        dm_cache_dir=dm_cache_dir,
        condition=condition,
        p_iid=p_iid,
        num_workers=4,
    )
    if tracking_dataset is None:
        tracking_dataset = instantiate(cfg.dataset)
    dm = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.train_dataloader()
    total_obs = 0
    swapped_obs = 0
    swapped_tracklets = 0
    total_tracklets = 0
    for b_idx, batch in enumerate(loader):
        if b_idx >= num_batches:
            break
        valid = ~batch["track_targets"].isnan()
        corr = batch["track_feats"]["corrupted_mask"].squeeze(-1).nan_to_num(0.0) > 0.5
        valid_corr = corr & valid
        total_obs += int(valid.sum().item())
        swapped_obs += int(valid_corr.sum().item())
        trk_valid = valid.any(dim=-1)
        trk_corr = valid_corr.any(dim=-1)
        total_tracklets += int(trk_valid.sum().item())
        swapped_tracklets += int(trk_corr.sum().item())
    return {
        "num_batches": num_batches,
        "total_valid_obs": total_obs,
        "swapped_obs": swapped_obs,
        "swap_frame_rate": swapped_obs / max(total_obs, 1),
        "total_valid_tracklets": total_tracklets,
        "swapped_tracklets": swapped_tracklets,
        "swapped_tracklet_frac": swapped_tracklets / max(total_tracklets, 1),
    }, tracking_dataset


def measure_swap_rates(dm_cache_dir: Path, train_vids: list[str], val_vids: list[str], p_iid: float, num_batches: int = 200) -> dict[str, Any]:
    """Measure empirical swap frame rates over num_batches for C2/C3 vs C1 (seed 13) without redundant C2 reruns."""
    c2_stats, tracking_dataset = _measure_single_swap_rate(
        dm_cache_dir, train_vids, val_vids, "C2_CONTIGUOUS_TAKEOVER", p_iid=p_iid, num_batches=num_batches
    )
    c1_stats, _ = _measure_single_swap_rate(
        dm_cache_dir, train_vids, val_vids, "C1_MATCHED_IID_SWAP", p_iid=p_iid, num_batches=num_batches, tracking_dataset=tracking_dataset
    )
    r_c2 = c2_stats["swap_frame_rate"]
    r_c1 = c1_stats["swap_frame_rate"]
    rel_diff = abs(r_c1 - r_c2) / max(r_c2, 1e-12)
    cur_p = float(p_iid)
    if rel_diff > 0.05:
        cur_p = min(0.50, max(0.005, cur_p * (r_c2 / max(r_c1, 1e-6))))
        c1_stats, _ = _measure_single_swap_rate(
            dm_cache_dir, train_vids, val_vids, "C1_MATCHED_IID_SWAP", p_iid=cur_p, num_batches=num_batches, tracking_dataset=tracking_dataset
        )
        r_c1 = c1_stats["swap_frame_rate"]
        rel_diff = abs(r_c1 - r_c2) / max(r_c2, 1e-12)
    return {
        "p_iid": cur_p,
        "rates": {"C2_CONTIGUOUS_TAKEOVER": c2_stats, "C1_MATCHED_IID_SWAP": c1_stats},
        "relative_diff_c1_vs_c2": rel_diff,
        "within_5pct": rel_diff <= 0.05,
    }


def cmd_stage_a_lock(args: argparse.Namespace) -> None:
    apply_s10_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=False)
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    train32, dev8 = s09.load_split_lists()

    # 1. Verify all S08 / S09 reused assets SHA256
    s08_audit = json.loads((REPO_ROOT / "reports/baselines/s08_camel_baseline_20261125/split_audit.json").read_text())
    expected = {
        "t_twostage_joint_ckpt": (S09_T_TWOSTAGE_CKPT, protocol["reused_assets"]["t_twostage_joint_ckpt"]["sha256"]),
        "c1_app_encoder_ckpt": (S09_C1_CKPTS["app_encoder"], protocol["reused_assets"]["c1_te_ckpts"]["app_encoder"]["sha256"]),
        "c1_kp_encoder_ckpt": (S09_C1_CKPTS["kp_encoder"], protocol["reused_assets"]["c1_te_ckpts"]["kp_encoder"]["sha256"]),
        "c1_bbox_encoder_ckpt": (S09_C1_CKPTS["bbox_encoder"], protocol["reused_assets"]["c1_te_ckpts"]["bbox_encoder"]["sha256"]),
        "split_manifest": (s09.SPLIT_PATH, protocol["data"]["membership_sha256"]),
        "states_train32": (s09.S08_STATES_DIR / "states_train32.pklz", s08_audit["isolated_state_files"]["train32_with_gt"]["sha256"]),
        "states_dev8_train_ref": (s09.S08_STATES_DIR / "states_dev8_train_ref.pklz", s08_audit["isolated_state_files"]["dev8_train_ref"]["sha256"]),
        "states_dev8_gt_free": (s09.S08_STATES_DIR / "states_dev8_gt_free.pklz", s08_audit["isolated_state_files"]["dev8_gt_free"]["sha256"]),
    }
    verified_assets = {}
    for k, (path, exp_sha) in expected.items():
        actual_sha = s09.sha256_file(path)
        if actual_sha != exp_sha:
            raise RuntimeError(f"Asset SHA mismatch for {k} ({path}): {actual_sha} != {exp_sha}")
        verified_assets[k] = {"path": str(path), "sha256": actual_sha, "bytes": path.stat().st_size, "verified": True}

    dm_cache = ensure_shared_dm_cache()

    # 2. Auto-calibrate p_iid in 2 steps so C1 matches C2 across 200 batches within <= 5%
    probe1 = measure_swap_rates(dm_cache, train32, dev8, p_iid=float(args.p_iid), num_batches=200)
    if not probe1["within_5pct"]:
        r_c2 = probe1["rates"]["C2_CONTIGUOUS_TAKEOVER"]["swap_frame_rate"]
        r_c1 = max(probe1["rates"]["C1_MATCHED_IID_SWAP"]["swap_frame_rate"], 1e-6)
        p_new = min(0.50, max(0.005, float(args.p_iid) * (r_c2 / r_c1)))
        probe1 = measure_swap_rates(dm_cache, train32, dev8, p_iid=p_new, num_batches=200)
    assert probe1["within_5pct"], f"Failed to match C1 vs C2 swap rate within 5%: {probe1}"
    calibrated_p_iid = float(probe1["p_iid"])

    # 3. Verify C3 parameter overhead (< 2%) & write resolved configs
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dir = REPORT_DIR / "resolved_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    placeholder = RUNS_ROOT / "_resolved_placeholder"

    cfg_b = s09.compose_s09_infer_config(
        run_dir=placeholder,
        train_vids=train32,
        val_vids=dev8,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=S09_T_TWOSTAGE_CKPT,
        sim_threshold=0.50,
    )
    (cfg_dir / "stage_b_skip_and_c0.yaml").write_text(OmegaConf.to_yaml(cfg_b, resolve=True))

    c_params: dict[str, Any] = {}
    for cond in ("C1_MATCHED_IID_SWAP", "C2_CONTIGUOUS_TAKEOVER", "C3_CONSISTENCY_GATED_TE"):
        cfg_c = compose_s10_train_config(
            run_dir=placeholder,
            train_vids=train32,
            val_vids=dev8,
            dm_cache_dir=dm_cache,
            condition=cond,
            p_iid=calibrated_p_iid,
        )
        (cfg_dir / f"stage_c_{cond.lower()}.yaml").write_text(OmegaConf.to_yaml(cfg_c, resolve=True))
        m_c = instantiate(cfg_c.modules.track.CAMEL, _recursive_=False)
        ps = s09.param_summary(m_c)
        if cond == "C3_CONSISTENCY_GATED_TE":
            cons_p = sum(p.numel() for p in m_c.consistency_head.parameters())
            ps["consistency_head"] = {"total_params": cons_p, "trainable_params": cons_p, "fraction_of_total": cons_p / ps["total"]["total_params"]}
            assert ps["consistency_head"]["fraction_of_total"] < 0.02
        c_params[cond] = ps

    source_audit = {
        "schema_version": "s10-source-audit-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "upstream_cameltrack_commit": "bc1c261075b0beacfe71c2896453be83643153c8",
        "temporal_encoder_pooling_note": (
            "Upstream cameltrack/architecture/temporal_encoder.py:93 calls nn.TransformerEncoder(src, src_key_padding_mask=~src_mask) "
            "on [S+1, B*N, E] where index S is the cls token and 0..S-1 are historical observations ordered age-ascending (0 newest). "
            "S10 C3 keeps all upstream temp_encs.{cue}.* parameter names untouched for exact S09 C1 loading, computes causal w_t in "
            "ConsistencyGateHead over concatenated det_tokenizer tokens (704 -> 128 -> 2-layer causal Transformer -> 1), and applies "
            "unified float attn_mask with log(clamp(w_t, 1e-4, 1.0)) key bias in TemporalEncoder (preserving diagonal 0.0 to avoid empty-row NaN)."
        ),
        "verified_assets": verified_assets,
        "shared_datamodule_cache": str(dm_cache),
        "swap_rate_calibration_200_batches": probe1,
        "model_parameters": c_params,
    }
    (REPORT_DIR / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")

    runtime_lock = {
        "schema_version": "s10-runtime-lock-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "calibrated_c1_p_iid": calibrated_p_iid,
        "gpu_safety_probe": gpu_info,
        "scripts_sha256": {
            "scripts/run_s10_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s10_camel.py"),
            "scripts/run_s09_camel.py": s09.sha256_file(REPO_ROOT / "scripts/run_s09_camel.py"),
            "tests/test_s10_camel.py": s09.sha256_file(REPO_ROOT / "tests/test_s10_camel.py"),
        },
    }
    (REPORT_DIR / "runtime_lock.json").write_text(json.dumps(runtime_lock, indent=2) + "\n")

    cmds = f"""#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
S10=/home/imc/yzy/MOT/scripts/run_s10_camel.py
T2_CKPT={S09_T_TWOSTAGE_CKPT}

# Stage A
CUDA_VISIBLE_DEVICES='' $PY $S10 stage-a-lock --p-iid {calibrated_p_iid}
CUDA_VISIBLE_DEVICES='' $PY -m pytest /home/imc/yzy/MOT/tests/test_s10_camel.py -q

# Stage B: D2_SKIP_ONLY + C0 zero-retrain baselines
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-skip-only --ckpt $T2_CKPT --run-id b_d2_skip_only_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-c0-write-gate --ckpt $T2_CKPT --tau-write 0.60 --run-id b_c0_tau060_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-c0-write-gate --ckpt $T2_CKPT --tau-write 0.65 --run-id b_c0_tau065_v1

# Stage C: Controlled 20-epoch training + dev8 online evaluation
CUDA_VISIBLE_DEVICES=1 $PY $S10 train-c --condition C1_MATCHED_IID_SWAP --p-iid {calibrated_p_iid} --run-id c1_matched_iid_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C1_MATCHED_IID_SWAP --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c1_matched_iid_seed13_e20_v1/CAMEL/last.ckpt --run-id c_eval_c1_th050_v1

CUDA_VISIBLE_DEVICES=1 $PY $S10 train-c --condition C2_CONTIGUOUS_TAKEOVER --run-id c2_contiguous_takeover_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C2_CONTIGUOUS_TAKEOVER --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c2_contiguous_takeover_seed13_e20_v1/CAMEL/last.ckpt --run-id c_eval_c2_th050_v1

CUDA_VISIBLE_DEVICES=1 $PY $S10 train-c --condition C3_CONSISTENCY_GATED_TE --run-id c3_consistency_gated_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C3_CONSISTENCY_GATED_TE --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c3_consistency_gated_seed13_e20_v1/CAMEL/last.ckpt --consistency-model --run-id c_eval_c3_th050_v1

# Stage D: Soft write filter + offline AUROC on real T_TWOSTAGE cross-ID writes
CUDA_VISIBLE_DEVICES=1 $PY $S10 eval-online --condition C3_WITH_SOFT_WRITE_FILTER --ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c3_consistency_gated_seed13_e20_v1/CAMEL/last.ckpt --consistency-model --soft-write-filter-w 0.50 --run-id d_c3_soft_write_filter_w050_v1
CUDA_VISIBLE_DEVICES=1 $PY $S10 stage-d-real-pollution-auroc --t2-ckpt $T2_CKPT --c3-ckpt /media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/c3_consistency_gated_seed13_e20_v1/CAMEL/last.ckpt --run-id d_real_pollution_auroc_v1

# Stage E
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s10_report.py
"""
    cmd_path = REPORT_DIR / "commands.sh"
    cmd_path.write_text(cmds)
    cmd_path.chmod(0o755)
    print(json.dumps({
        "status": "PASS",
        "calibrated_c1_p_iid": calibrated_p_iid,
        "swap_calibration": probe1,
        "c3_cons_fraction": c_params["C3_CONSISTENCY_GATED_TE"]["consistency_head"]["fraction_of_total"],
    }, indent=2))


def _save_eval_summary(
    *,
    run_dir: Path,
    run_id: str,
    condition: str,
    inference_mode: str,
    ckpt_path: Path,
    ckpt_sha: str,
    sim_threshold: float,
    gpu_info: dict[str, Any],
    t0: float,
    t_track0: float,
    t_track1: float,
    t1: float,
    peak_vram_mib: float,
    dev8: list[str],
    mot_dir: Path,
    extra: dict[str, Any],
) -> None:
    detailed_csv = run_dir / "eval/results/tracklab/pedestrian_detailed.csv"
    with detailed_csv.open(newline="") as f:
        rows = {r["seq"]: r for r in csv.DictReader(f)}
    assert set(rows.keys()) == (set(dev8) | {"COMBINED"})
    mot_hashes = {s: s09.sha256_file(mot_dir / f"{s}.txt") for s in dev8}
    summary = {
        "schema_version": "s10-eval-v1",
        "status": "COMPLETED",
        "run_id": run_id,
        "condition": condition,
        "inference_mode": inference_mode,
        "seed": "13",
        "scope_id": "dev8",
        "sim_threshold": sim_threshold,
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "state_file": str(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "state_file_sha256": s09.sha256_file(s09.S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "wall_seconds": t1 - t0,
        "tracking_seconds": t_track1 - t_track0,
        "peak_vram_mib": peak_vram_mib,
        "mot_dir": str(mot_dir),
        "mot_sha256_by_seq": mot_hashes,
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


def load_dev8_gt_boxes(tracking_set: Any) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[str, int]]]:
    """Build GT (person_ids, ltwh) and (seq, frame_1based) lookups indexed by image_id."""
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


def _run_wrapped_eval(
    *,
    run_id: str,
    condition: str,
    inference_mode: str,
    ckpt_path: Path,
    consistency_model: bool,
    wrap_fn: Any,
    extra_fn: Any,
) -> None:
    apply_s10_runtime_compat_patches()
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
        sim_threshold=0.50,
    )
    if consistency_model:
        cfg.modules.track.CAMEL._target_ = "run_s10_camel.S10ConsistencyCAMEL"

    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    base_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    override_cfg = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    override_cfg["sim_threshold"] = 0.50
    override_cfg["use_computed_sim_threshold"] = False
    camel_cls = S10ConsistencyCAMEL if consistency_model else type(base_mod.CAMEL)
    base_mod.CAMEL = camel_cls.load_from_checkpoint(ckpt_path, map_location=base_mod.device, **override_cfg)
    assert abs(float(base_mod.CAMEL.sim_threshold) - 0.50) < 1e-12
    assert bool(base_mod.CAMEL.use_computed_sim_threshold) is False

    wrapped_mod = wrap_fn(base_mod, tracking_set)
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

    _save_eval_summary(
        run_dir=run_dir,
        run_id=run_id,
        condition=condition,
        inference_mode=inference_mode,
        ckpt_path=ckpt_path,
        ckpt_sha=ckpt_sha,
        sim_threshold=0.50,
        gpu_info=gpu_info,
        t0=t0,
        t_track0=t_track0,
        t_track1=t_track1,
        t1=t1,
        peak_vram_mib=peak_vram_mib,
        dev8=dev8,
        mot_dir=mot_dir,
        extra=extra_fn(wrapped_mod),
    )


def cmd_eval_skip_only(args: argparse.Namespace) -> None:
    def wrap_fn(base_mod: Any, tracking_set: Any) -> Any:
        gt_by_image_id, _ = load_dev8_gt_boxes(tracking_set)
        return SkipOnlyCleanGalleryCAMELTrack(base_mod, gt_by_image_id)

    def extra_fn(mod: Any) -> dict[str, Any]:
        assert mod.routing_stats["gt_injected_on_miss"] == 0
        return {"gt_role": "privileged_skip_only_not_deployable", "routing_stats": mod.routing_stats}

    _run_wrapped_eval(
        run_id=args.run_id,
        condition="D2_SKIP_ONLY",
        inference_mode="privileged_skip_only_probe",
        ckpt_path=Path(args.ckpt),
        consistency_model=False,
        wrap_fn=wrap_fn,
        extra_fn=extra_fn,
    )


def cmd_eval_c0(args: argparse.Namespace) -> None:
    tau_write = float(args.tau_write)
    cond = f"C0_SIMPLE_WRITE_GATE_TAU{int(round(tau_write * 100)):03d}"

    def wrap_fn(base_mod: Any, _ts: Any) -> Any:
        return SimpleWriteGateCAMELTrack(base_mod, tau_write=tau_write, min_hits_to_gate=3)

    _run_wrapped_eval(
        run_id=args.run_id,
        condition=cond,
        inference_mode="online_closed_loop",
        ckpt_path=Path(args.ckpt),
        consistency_model=False,
        wrap_fn=wrap_fn,
        extra_fn=lambda mod: {"tau_write": tau_write, "min_hits_to_gate": 3, "gate_stats": mod.gate_stats},
    )


def cmd_eval_online(args: argparse.Namespace) -> None:
    soft_w = args.soft_write_filter_w

    def wrap_fn(base_mod: Any, _ts: Any) -> Any:
        if soft_w is not None:
            return SoftConsistencyWriteFilterCAMELTrack(base_mod, w_write_threshold=float(soft_w), min_hits_to_gate=3)
        return base_mod

    def extra_fn(mod: Any) -> dict[str, Any]:
        if soft_w is not None:
            return {"w_write_threshold": float(soft_w), "min_hits_to_gate": 3, "gate_stats": mod.gate_stats}
        return {}

    _run_wrapped_eval(
        run_id=args.run_id,
        condition=args.condition,
        inference_mode="online_closed_loop",
        ckpt_path=Path(args.ckpt),
        consistency_model=bool(args.consistency_model),
        wrap_fn=wrap_fn,
        extra_fn=extra_fn,
    )


def cmd_train_c(args: argparse.Namespace) -> None:
    apply_s10_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    s09.setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    dm_cache = ensure_shared_dm_cache()
    train32, dev8 = s09.load_split_lists()
    cfg = compose_s10_train_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        dm_cache_dir=dm_cache,
        condition=args.condition,
        p_iid=float(args.p_iid),
        max_epochs=20,
        num_workers=4,
    )

    t0 = time.perf_counter()
    pl.seed_everything(13, workers=True)
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL

    init_prefixes = ["gaffe"] + [f"temp_encs.{k}" for k in model.temp_encs.keys()]
    if hasattr(model, "consistency_head"):
        init_prefixes.append("consistency_head")
    before = {
        prefix: {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith(prefix + ".")}
        for prefix in init_prefixes
    }
    params_info = s09.param_summary(model)
    if hasattr(model, "consistency_head"):
        ch_p = sum(p.numel() for p in model.consistency_head.parameters())
        params_info["consistency_head"] = {"total_params": ch_p, "trainable_params": ch_p, "fraction_of_total": ch_p / params_info["total"]["total_params"]}

    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})
    epoch_cb = S10EpochLogger(run_dir, args.condition, t0)
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
        n_changed = 0
        max_abs = 0.0
        l2_sq = 0.0
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
        "schema_version": "s10-train-v1",
        "status": "COMPLETED",
        "condition": args.condition,
        "run_id": args.run_id,
        "seed": 13,
        "p_iid": float(args.p_iid) if args.condition == "C1_MATCHED_IID_SWAP" else None,
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
    print(json.dumps({"status": "COMPLETED", "condition": args.condition, "nonfinite": summary["nonfinite_grad_steps"], "last_ckpt_sha256": last_sha}, indent=2))


class RealPollutionScorerCAMELTrack:
    """Replay online T_TWOSTAGE_TH050 and score every matched write causally with C3 consistency head."""

    def __init__(self, base_mod: Any, c3_camel: S10ConsistencyCAMEL, gt_by_image_id: dict[int, tuple[np.ndarray, np.ndarray]], img_to_seq_fr: dict[int, tuple[str, int]]) -> None:
        self.base = base_mod
        self.c3_camel = c3_camel
        self.gt_by_image_id = gt_by_image_id
        self.img_to_seq_fr = img_to_seq_fr
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn
        self.events: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        for track in self.base.tracklets:
            track.forward()
        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(self.base.tracklets, detections)
        sim_tensor = td_sim_matrix.detach().cpu() if torch.is_tensor(td_sim_matrix) and td_sim_matrix.numel() > 0 else torch.empty((0,))

        img_id = int(metadatas.index[0]) if hasattr(metadatas, "index") else int(detections[0].image_id.item())
        seq, fr = self.img_to_seq_fr[img_id]
        gt_ids_f, gt_boxes_f = self.gt_by_image_id.get(img_id, (np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)))
        det_boxes = [d.bbox_ltwh.detach().cpu().numpy().astype(float) for d in detections]
        det_to_gt, _ = s09.match_detections_to_gt_frame(det_boxes, gt_ids_f, gt_boxes_f, iou_thresh=0.5)

        if len(matched) > 0:
            m_trks = [self.base.tracklets[int(m[0])] for m in matched]
            m_dets = [detections[int(m[1])] for m in matched]
            w_vals = compute_candidate_w_last(self.base, self.c3_camel, m_trks, m_dets)
        else:
            w_vals = np.empty((0,), dtype=np.float32)

        for (m, w_last) in zip(matched, w_vals.tolist()):
            trk_idx, det_idx = int(m[0]), int(m[1])
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            sim_val = float(sim_tensor[trk_idx, det_idx].item())
            anchor_gt = getattr(trk, "s10_gt_anchor", None)
            det_gt = det_to_gt.get(det_idx)
            if anchor_gt is not None and det_gt is not None and det_gt == anchor_gt:
                label = "clean_match"
            elif anchor_gt is not None and det_gt is not None and det_gt != anchor_gt:
                label = "cross_id_pollution"
            elif anchor_gt is not None and det_gt is None:
                label = "fp_pollution"
            else:
                label = "unanchored"

            self.events.append({
                "sequence": seq,
                "frame": fr,
                "track_id": int(trk.id),
                "anchor_gt": int(anchor_gt) if anchor_gt is not None else "",
                "det_gt": int(det_gt) if det_gt is not None else "",
                "label": label,
                "hits_before": int(trk.hits),
                "gallery_len_before": int(len(trk.detections)),
                "sim_val": sim_val,
                "w_last": float(w_last),
                "pollution_score_c3": 1.0 - float(w_last),
                "pollution_score_sim": 1.0 - sim_val,
            })

            trk.update(det)
            det.similarity_with_tracklet = sim_tensor[trk_idx, det_idx]
            det.similarities = sim_tensor[: len(self.base.tracklets), det_idx]

        for i in unmatched_dets:
            if detections[i].bbox_conf >= self.base.min_init_det_conf:
                trk = Tracklet(detections[i], self.base.max_track_gallery_size)
                trk.s10_gt_anchor = det_to_gt.get(int(i))
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
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.item())
                        if trk.last_detection.similarity_with_tracklet is not None
                        else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.base.tracklets[j].id: float(sim) for j, sim in enumerate(trk.last_detection.similarities.numpy())}
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
        return pd.DataFrame(columns=["track_id", "hits", "age", "matched_with", "time_since_update", "state", "costs"])


def _binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.50) -> dict[str, Any]:
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"count": int(len(y_true)), "pos": int(y_true.sum()), "auroc": None}
    preds = scores >= threshold
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "count": int(len(y_true)),
        "pos": int(y_true.sum()),
        "neg": int((1 - y_true).sum()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(prec),
        "recall_detection_rate": float(rec),
        "fpr_clean_false_alarm": float(fp / max(fp + tn, 1)),
        "f1": float(f1),
    }


def cmd_stage_d_real_auroc(args: argparse.Namespace) -> None:
    apply_s10_runtime_compat_patches()
    gpu_info = s09.enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    s09.setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    t2_ckpt = Path(args.t2_ckpt)
    c3_ckpt = Path(args.c3_ckpt)
    train32, dev8 = s09.load_split_lists()
    t0 = time.perf_counter()

    cfg_t2 = s09.compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=t2_ckpt,
        sim_threshold=0.50,
    )
    tl_wandb.init(cfg_t2)
    tracking_dataset = instantiate(cfg_t2.dataset)
    tracking_set = tracking_dataset.sets[cfg_t2.dataset.eval_set]
    base_mod = instantiate(cfg_t2.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    ov = OmegaConf.to_container(cfg_t2.modules.track.override_camel_cfg, resolve=True) or {}
    ov["sim_threshold"] = 0.50
    ov["use_computed_sim_threshold"] = False
    base_mod.CAMEL = type(base_mod.CAMEL).load_from_checkpoint(t2_ckpt, map_location=base_mod.device, **ov)

    # Load C3 consistency model separately on CUDA
    cfg_c3 = s09.compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=s09.S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=c3_ckpt,
        sim_threshold=0.50,
    )
    cfg_c3.modules.track.CAMEL._target_ = "run_s10_camel.S10ConsistencyCAMEL"
    c3_mod = instantiate(cfg_c3.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    c3_camel = S10ConsistencyCAMEL.load_from_checkpoint(c3_ckpt, map_location=c3_mod.device, **ov)
    c3_camel.eval()

    gt_by_image_id, img_to_seq_fr = load_dev8_gt_boxes(tracking_set)
    scorer = RealPollutionScorerCAMELTrack(base_mod, c3_camel, gt_by_image_id, img_to_seq_fr)
    Tracklet.count = 1
    pipeline = Pipeline(models=[scorer])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg_t2.state)
    tracking_engine = instantiate(cfg_t2.engine, modules=pipeline, tracker_state=tracker_state)
    tracking_engine.track_dataset()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    events_csv = run_dir / "d_real_pollution_scores.csv"
    with events_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scorer.events[0].keys()))
        w.writeheader()
        w.writerows(scorer.events)

    # Compute AUROC / F1 slices on offline labels
    def subset_metrics(sub_events: list[dict[str, Any]], pos_labels: set[str], sim_th: float = 0.60) -> dict[str, Any]:
        filtered = [e for e in sub_events if e["label"] in (pos_labels | {"clean_match"})]
        y = np.array([1 if e["label"] in pos_labels else 0 for e in filtered], dtype=int)
        s_c3 = np.array([float(e["pollution_score_c3"]) for e in filtered], dtype=np.float64)
        s_sim = np.array([float(e["pollution_score_sim"]) for e in filtered], dtype=np.float64)
        return {
            "c3_consistency_head_at_0_50": _binary_metrics(y, s_c3, threshold=0.50),
            "scalar_sim_baseline_at_tau060": _binary_metrics(y, s_sim, threshold=1.0 - sim_th),
            "scalar_sim_baseline_at_tau065": _binary_metrics(y, s_sim, threshold=1.0 - 0.65),
        }

    mature_events = [e for e in scorer.events if int(e["hits_before"]) >= 3]
    summary = {
        "schema_version": "s10-real-pollution-auroc-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "t2_ckpt_sha256": s09.sha256_file(t2_ckpt),
        "c3_ckpt_sha256": s09.sha256_file(c3_ckpt),
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "label_counts_all": {lbl: sum(1 for e in scorer.events if e["label"] == lbl) for lbl in ("clean_match", "cross_id_pollution", "fp_pollution", "unanchored")},
        "label_counts_mature_hits_ge_3": {lbl: sum(1 for e in mature_events if e["label"] == lbl) for lbl in ("clean_match", "cross_id_pollution", "fp_pollution", "unanchored")},
        "cross_id_vs_clean_all": subset_metrics(scorer.events, {"cross_id_pollution"}),
        "cross_id_vs_clean_mature": subset_metrics(mature_events, {"cross_id_pollution"}),
        "any_pollution_vs_clean_all": subset_metrics(scorer.events, {"cross_id_pollution", "fp_pollution"}),
        "any_pollution_vs_clean_mature": subset_metrics(mature_events, {"cross_id_pollution", "fp_pollution"}),
        "events_csv": str(events_csv),
    }
    (run_dir / "stage_d_auroc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "label_counts_all": summary["label_counts_all"],
        "cross_id_all_c3_auroc": summary["cross_id_vs_clean_all"]["c3_consistency_head_at_0_50"]["auroc"],
        "cross_id_all_sim_auroc": summary["cross_id_vs_clean_all"]["scalar_sim_baseline_at_tau060"]["auroc"],
        "cross_id_mature_c3_auroc": summary["cross_id_vs_clean_mature"]["c3_consistency_head_at_0_50"]["auroc"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_a = sub.add_parser("stage-a-lock")
    p_a.add_argument("--p-iid", type=float, default=0.085)

    p_skip = sub.add_parser("eval-skip-only")
    p_skip.add_argument("--ckpt", required=True)
    p_skip.add_argument("--run-id", required=True)

    p_c0 = sub.add_parser("eval-c0-write-gate")
    p_c0.add_argument("--ckpt", required=True)
    p_c0.add_argument("--tau-write", type=float, required=True)
    p_c0.add_argument("--run-id", required=True)

    p_tc = sub.add_parser("train-c")
    p_tc.add_argument("--condition", choices=["C1_MATCHED_IID_SWAP", "C2_CONTIGUOUS_TAKEOVER", "C3_CONSISTENCY_GATED_TE"], required=True)
    p_tc.add_argument("--p-iid", type=float, default=0.085)
    p_tc.add_argument("--run-id", required=True)

    p_eo = sub.add_parser("eval-online")
    p_eo.add_argument("--condition", required=True)
    p_eo.add_argument("--ckpt", required=True)
    p_eo.add_argument("--consistency-model", action="store_true")
    p_eo.add_argument("--soft-write-filter-w", type=float, default=None)
    p_eo.add_argument("--run-id", required=True)

    p_da = sub.add_parser("stage-d-real-pollution-auroc")
    p_da.add_argument("--t2-ckpt", required=True)
    p_da.add_argument("--c3-ckpt", required=True)
    p_da.add_argument("--run-id", required=True)

    args = parser.parse_args()
    dispatch = {
        "stage-a-lock": cmd_stage_a_lock,
        "eval-skip-only": cmd_eval_skip_only,
        "eval-c0-write-gate": cmd_eval_c0,
        "train-c": cmd_train_c,
        "eval-online": cmd_eval_online,
        "stage-d-real-pollution-auroc": cmd_stage_d_real_auroc,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
