"""Causal S06 edge features; no dataset GT or future state is accepted."""

from __future__ import annotations

import numpy as np

from mot_research.association.appearance_core import normalize_features


def upstream_direction_score(detections: np.ndarray, velocities: np.ndarray,
                             previous_observations: np.ndarray,
                             weight: float) -> np.ndarray:
    """Reproduce OC-SORT's score-shaped angle term (detections x trackers)."""
    dets = np.asarray(detections, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    obs = np.asarray(previous_observations, dtype=np.float64)
    if dets.ndim != 2 or dets.shape[1] < 5:
        raise ValueError("invalid OC-SORT direction inputs")
    if obs.size == 0:
        return np.empty((len(dets), 0), dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] < 5:
        raise ValueError("invalid OC-SORT observation inputs")
    if len(dets) == 0:
        return np.empty((0, len(obs)), dtype=np.float64)
    if vel.shape != (len(obs), 2):
        raise ValueError("invalid OC-SORT velocity inputs")
    cx1, cy1 = (dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2
    cx2, cy2 = (obs[:, 0] + obs[:, 2]) / 2, (obs[:, 1] + obs[:, 3]) / 2
    dx = cx1[None, :] - cx2[:, None]
    dy = cy1[None, :] - cy2[:, None]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6
    cosine = vel[:, 1, None] * (dx / norm) + vel[:, 0, None] * (dy / norm)
    angle = (np.pi / 2 - np.abs(np.arccos(np.clip(cosine, -1, 1)))) / np.pi
    valid = (obs[:, 4] >= 0).astype(np.float64)[:, None]
    return (valid * angle * float(weight) * dets[:, 4][None, :]).T


def cosine_matrix(detection_features: np.ndarray, track_features: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    det, valid_d = normalize_features(detection_features)
    trk, valid_t = normalize_features(track_features)
    if det.shape[1] != trk.shape[1]:
        raise ValueError("feature dimensions differ")
    valid = valid_d[:, None] & valid_t[None, :]
    values = np.zeros((len(det), len(trk)), dtype=np.float64)
    if values.size:
        raw = np.clip(det @ trk.T, -1, 1)
        values[valid] = raw[valid]
    return values, valid


def signed_competition_margins(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge score minus best competing edge in its row and column; 0 if none."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("scores must be a finite matrix")
    row = np.zeros_like(x)
    col = np.zeros_like(x)
    for d in range(x.shape[0]):
        for t in range(x.shape[1]):
            if x.shape[1] > 1:
                row[d, t] = x[d, t] - np.max(np.delete(x[d], t))
            if x.shape[0] > 1:
                col[d, t] = x[d, t] - np.max(np.delete(x[:, t], d))
    return row, col
