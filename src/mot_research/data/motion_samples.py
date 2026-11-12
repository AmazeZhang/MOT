"""Strictly-online motion sample construction from detector histories and GT targets."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np

from mot_research.analysis.failure_events import Candidate, hungarian_iou_matches


ARRAY_DTYPES = {
    "sequence_index": np.int32,
    "gt_id": np.int32,
    "target_frame": np.int32,
    "history_frames": np.int32,
    "history_valid": np.bool_,
    "history_states": np.float64,
    "target_state": np.float64,
    "target_delta": np.float64,
    "frame_gap": np.int32,
    "positive_detection_id": np.int32,
    "positive_detection_iou": np.float64,
    "image_width": np.int32,
    "image_height": np.int32,
    "fps": np.float64,
}


def box_xyxy_to_state(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Encode continuous XYXY boxes as normalized center/log-size state."""
    values = np.asarray(boxes, dtype=np.float64)
    widths = values[..., 2] - values[..., 0]
    heights = values[..., 3] - values[..., 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("boxes must have strictly positive continuous width and height")
    return np.stack(
        [
            (values[..., 0] + values[..., 2]) / (2.0 * image_width),
            (values[..., 1] + values[..., 3]) / (2.0 * image_height),
            np.log(widths / image_width),
            np.log(heights / image_height),
        ],
        axis=-1,
    )


def state_to_box_xyxy(states: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Decode normalized center/log-size state into continuous XYXY boxes."""
    values = np.asarray(states, dtype=np.float64)
    center_x = values[..., 0] * image_width
    center_y = values[..., 1] * image_height
    widths = np.exp(values[..., 2]) * image_width
    heights = np.exp(values[..., 3]) * image_height
    return np.stack(
        [
            center_x - widths / 2.0,
            center_y - heights / 2.0,
            center_x + widths / 2.0,
            center_y + heights / 2.0,
        ],
        axis=-1,
    )


def build_sequence_motion_samples(
    *,
    sequence_index: int,
    sequence_length: int,
    image_width: int,
    image_height: int,
    fps: float,
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    detections_by_frame: dict[int, list[Candidate]],
    history_length: int = 8,
    minimum_history_observations: int = 2,
    maximum_frame_gap: int = 31,
    matching_iou_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Build samples sequentially so target/current/future detections cannot leak."""
    if history_length < minimum_history_observations or minimum_history_observations < 1:
        raise ValueError("invalid history requirements")
    histories: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    rows: dict[str, list[Any]] = {key: [] for key in ARRAY_DTYPES}
    for frame_id in range(1, sequence_length + 1):
        frame_gt = gt_by_frame.get(frame_id, {})
        candidates = detections_by_frame.get(frame_id, [])
        matches, _ = hungarian_iou_matches(
            frame_gt, candidates, matching_iou_threshold
        )
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}

        # Samples are emitted before current matches are appended to history.
        for gt_id in sorted(frame_gt):
            history = histories.get(gt_id, [])
            if len(history) < minimum_history_observations:
                continue
            frame_gap = frame_id - history[-1][0]
            if frame_gap < 1 or frame_gap > maximum_frame_gap:
                continue
            selected = history[-history_length:]
            history_frames = np.zeros(history_length, dtype=np.int32)
            history_valid = np.zeros(history_length, dtype=np.bool_)
            history_states = np.zeros((history_length, 4), dtype=np.float64)
            offset = history_length - len(selected)
            for index, (observed_frame, observed_state) in enumerate(selected, start=offset):
                history_frames[index] = observed_frame
                history_valid[index] = True
                history_states[index] = observed_state
            target_state = box_xyxy_to_state(
                frame_gt[gt_id][None, :], image_width, image_height
            )[0]
            target_match = matches.get(gt_id)
            rows["sequence_index"].append(sequence_index)
            rows["gt_id"].append(gt_id)
            rows["target_frame"].append(frame_id)
            rows["history_frames"].append(history_frames)
            rows["history_valid"].append(history_valid)
            rows["history_states"].append(history_states)
            rows["target_state"].append(target_state)
            rows["target_delta"].append(target_state - selected[-1][1])
            rows["frame_gap"].append(frame_gap)
            rows["positive_detection_id"].append(target_match[0] if target_match else -1)
            rows["positive_detection_iou"].append(target_match[1] if target_match else np.nan)
            rows["image_width"].append(image_width)
            rows["image_height"].append(image_height)
            rows["fps"].append(fps)

        for gt_id, (detection_id, _) in matches.items():
            detection = candidate_by_id[detection_id]
            state = box_xyxy_to_state(
                detection.box_xyxy[None, :], image_width, image_height
            )[0]
            histories[gt_id].append((frame_id, state))

    sample_count = len(rows["target_frame"])
    arrays: dict[str, np.ndarray] = {}
    for key, dtype in ARRAY_DTYPES.items():
        if key in {"history_frames", "history_valid"}:
            shape = (sample_count, history_length)
        elif key == "history_states":
            shape = (sample_count, history_length, 4)
        elif key in {"target_state", "target_delta"}:
            shape = (sample_count, 4)
        else:
            shape = (sample_count,)
        arrays[key] = np.asarray(rows[key], dtype=dtype).reshape(shape)
    validate_motion_sample_arrays(arrays)
    return arrays


def concatenate_motion_samples(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("at least one sequence sample set is required")
    combined = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ARRAY_DTYPES
    }
    validate_motion_sample_arrays(combined)
    return combined


def validate_motion_sample_arrays(arrays: dict[str, np.ndarray]) -> None:
    count = len(arrays["target_frame"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("sample arrays have inconsistent leading dimensions")
    valid = arrays["history_valid"]
    frames = arrays["history_frames"]
    targets = arrays["target_frame"][:, None]
    if np.any(frames[valid] >= np.broadcast_to(targets, frames.shape)[valid]):
        raise ValueError("target or future frame leaked into history")
    valid_counts = valid.sum(axis=1)
    if np.any(valid_counts < 2):
        raise ValueError("sample has fewer than two valid history observations")
    last_indices = valid_counts - 1 + (valid.shape[1] - valid_counts)
    last_frames = frames[np.arange(count), last_indices]
    if not np.array_equal(arrays["target_frame"] - last_frames, arrays["frame_gap"]):
        raise ValueError("frame_gap differs from target minus last history frame")
    finite_keys = ["history_states", "target_state", "target_delta", "fps"]
    if any(not np.isfinite(arrays[key]).all() for key in finite_keys):
        raise ValueError("non-finite motion sample value")
    keys = np.stack(
        [arrays["sequence_index"], arrays["gt_id"], arrays["target_frame"]], axis=1
    )
    if len(np.unique(keys, axis=0)) != count:
        raise ValueError("duplicate sample key")


def fit_motion_normalization(arrays: dict[str, np.ndarray], std_floor: float) -> dict[str, list[float]]:
    """Fit population statistics from motion-train arrays only."""
    history_values = arrays["history_states"][arrays["history_valid"]]
    target_delta = arrays["target_delta"]
    state_mean = history_values.mean(axis=0)
    state_std = np.maximum(history_values.std(axis=0), std_floor)
    delta_mean = target_delta.mean(axis=0)
    delta_std = np.maximum(target_delta.std(axis=0), std_floor)
    values = np.r_[state_mean, state_std, delta_mean, delta_std]
    if not np.isfinite(values).all():
        raise ValueError("non-finite normalization statistics")
    return {
        "history_state_mean": state_mean.tolist(),
        "history_state_population_std": state_std.tolist(),
        "target_delta_mean": delta_mean.tolist(),
        "target_delta_population_std": delta_std.tolist(),
        "std_floor": float(std_floor),
        "fit_sample_count": int(len(target_delta)),
        "fit_valid_history_observation_count": int(len(history_values)),
    }


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, dtypes, shapes and C-order bytes independent of NPZ metadata."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(str(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()

