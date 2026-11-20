"""CPU-only reference primitives for S0/S1; no extractor, tracker, or GT access.

Rows are detections, columns are stable tracker IDs. Memory can only be read
before a frame's update and committed after ALL association rounds and births.
This is a simple baseline, not an implementation of Deep OC-SORT or MOTIP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _ids(values: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or v < 0 for v in ids):
        raise ValueError("IDs must be nonnegative integers")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs")
    return tuple(int(v) for v in ids)


def normalize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invalid/zero rows become zero vectors plus False, never usable evidence."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("features must have shape (N, D), D > 0")
    finite = np.isfinite(x).all(axis=1)
    clean = np.where(finite[:, None], x, 0.0)
    # Scale first so finite large-magnitude inputs do not overflow their norm.
    scale = np.max(np.abs(clean), axis=1)
    nonzero = finite & (scale > 0)
    scaled = np.zeros_like(clean)
    scaled[nonzero] = clean[nonzero] / scale[nonzero, None]
    norm = np.linalg.norm(scaled, axis=1)
    valid = nonzero & (norm > 0)
    out = np.zeros_like(clean)
    out[valid] = scaled[valid] / norm[valid, None]
    return out, valid


def select_high_detections(
    det_ids: Sequence[int], rows_xyxy_score: np.ndarray, threshold: float = 0.6,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Apply ONE mask to IDs and rows; return original row indices as provenance."""
    ids = _ids(det_ids)
    rows = np.asarray(rows_xyxy_score, dtype=np.float64)
    if rows.shape != (len(ids), 5) or not np.isfinite(rows).all():
        raise ValueError("expected finite (N, 5) rows aligned with IDs")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if np.any(rows[:, 2:4] <= rows[:, :2]):
        raise ValueError("invalid detection box")
    indices = np.flatnonzero(rows[:, 4] > threshold)
    return tuple(ids[i] for i in indices), rows[indices].copy(), indices


def fuse_appearance_score(
    base_score: np.ndarray, detection_features: np.ndarray,
    track_features: np.ndarray, eligible: np.ndarray, *, weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Add weight*cosine on eligible valid pairs; preserve scores elsewhere.

    The caller must retain the upstream fast path, LAP backend and post-LAP
    IoU rejection. 'eligible' controls the bonus, not assignment feasibility.
    At weight=0 return an exact copy without touching features (including NaNs).
    """
    base = np.asarray(base_score, dtype=np.float64)
    mask = np.asarray(eligible)
    if base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError("base score must be a finite matrix")
    if mask.dtype != np.bool_ or mask.shape != base.shape:
        raise ValueError("eligible must be a bool matrix aligned with base")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    out = base.copy()
    if weight == 0:
        return out, np.zeros(base.shape, dtype=bool)
    detections, valid_d = normalize_features(detection_features)
    tracks, valid_t = normalize_features(track_features)
    if (detections.shape[0], tracks.shape[0]) != base.shape:
        raise ValueError("feature row order/length differs from score axes")
    if detections.shape[1] != tracks.shape[1]:
        raise ValueError("feature dimensions differ")
    used = mask & valid_d[:, None] & valid_t[None, :]
    similarity = np.clip(detections @ tracks.T, -1.0, 1.0)
    out[used] += weight * similarity[used]
    return out, used


class CausalAppearanceMemory:
    """One instance per video AND condition. last or normalized EMA of matches.

    Invalid features skip writing; a missing detection never writes a prediction.
    Deleted tracks are purged, live-but-unmatched tracks retain their memory.
    Call finish_frame even for empty frames. No GT IDs are accepted by design.
    """

    def __init__(self, dimension: int, *, mode: str = "last", alpha: float = 0.9):
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be positive integer")
        if mode not in {"last", "ema"} or not np.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("mode must be last/ema and 0 <= alpha < 1")
        self.dimension, self.mode, self.alpha = dimension, mode, alpha
        self._memory: dict[int, tuple[int, np.ndarray]] = {}
        self._last_frame = 0
        self._pending: int | None = None

    def begin_frame(self, frame: int, track_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return independent pre-update features and last-write frames (-1 absent)."""
        ids = _ids(track_ids)
        if not isinstance(frame, (int, np.integer)) or isinstance(frame, (bool, np.bool_)):
            raise ValueError("frame must be an integer")
        if self._pending is not None or frame <= self._last_frame:
            raise ValueError("unfinished frame or non-increasing frame")
        features = np.zeros((len(ids), self.dimension), dtype=np.float64)
        written = np.full(len(ids), -1, dtype=np.int64)
        for i, tid in enumerate(ids):
            if tid in self._memory:
                written[i], features[i] = self._memory[tid]
                if written[i] >= frame:
                    raise RuntimeError("current/future appearance leaked into history")
        self._pending = int(frame)
        return features, written

    def finish_frame(
        self, frame: int, live_track_ids: Sequence[int],
        matched_track_to_det: Mapping[int, int], det_ids: Sequence[int],
        detection_features: np.ndarray,
    ) -> None:
        """Commit final real matches (including births), atomically after validation.

        det_ids identifies the full current-frame feature table. Match mapping
        must come from tracker update instrumentation, not box nearest-neighbors.
        """
        if self._pending is None or frame != self._pending:
            raise ValueError("finish_frame must match begin_frame")
        live = set(_ids(live_track_ids))
        ids = _ids(det_ids)
        track_keys = _ids(tuple(matched_track_to_det))
        chosen = _ids(tuple(matched_track_to_det.values()))
        if not set(track_keys).issubset(live) or not set(chosen).issubset(ids):
            raise ValueError("match references unknown detection or non-live track")
        features, valid = normalize_features(detection_features)
        if features.shape != (len(ids), self.dimension):
            raise ValueError("feature table has incorrect shape")
        index = {did: i for i, did in enumerate(ids)}
        updated = {tid: value for tid, value in self._memory.items() if tid in live}
        for tid, did in matched_track_to_det.items():
            i = index[did]
            if not valid[i]:
                continue
            value = features[i].copy()
            if self.mode == "ema" and tid in updated:
                mixed, ok = normalize_features(
                    (self.alpha * updated[tid][1] + (1 - self.alpha) * value)[None]
                )
                # Exact cancellation has no direction: keep last usable memory.
                if not ok[0]:
                    continue
                value = mixed[0]
            updated[tid] = (int(frame), value)
        self._memory = updated
        self._last_frame, self._pending = int(frame), None
