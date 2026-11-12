"""Offline motion-prediction comparators and metrics for the P1 feasibility gate."""

from __future__ import annotations

from typing import Any

import numpy as np


CORE_METRICS = (
    "normalized_delta_mse",
    "normalized_center_l2_mean",
    "box_iou_mean",
)


def last_two_observations(
    history_states: np.ndarray,
    history_frames: np.ndarray,
    history_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return previous/last observations from left-padded histories."""
    valid_counts = history_valid.sum(axis=1)
    if np.any(valid_counts < 2):
        raise ValueError("constant velocity requires at least two observations")
    width = history_valid.shape[1]
    last_index = width - 1
    previous_index = width - 2
    rows = np.arange(len(history_states))
    if not np.all(history_valid[:, last_index]) or not np.all(history_valid[:, previous_index]):
        raise ValueError("histories must be left-padded with valid observations on the right")
    return (
        history_states[rows, previous_index],
        history_states[rows, last_index],
        history_frames[rows, previous_index],
        history_frames[rows, last_index],
    )


def zero_delta_prediction(history_states: np.ndarray, history_valid: np.ndarray) -> np.ndarray:
    if not np.all(history_valid[:, -1]):
        raise ValueError("last history position must be valid")
    return np.asarray(history_states[:, -1], dtype=np.float64).copy()


def constant_velocity_prediction(
    history_states: np.ndarray,
    history_frames: np.ndarray,
    history_valid: np.ndarray,
    target_frame: np.ndarray,
) -> np.ndarray:
    previous, last, previous_frame, last_frame = last_two_observations(
        history_states, history_frames, history_valid
    )
    observation_gap = last_frame - previous_frame
    target_gap = target_frame - last_frame
    if np.any(observation_gap <= 0) or np.any(target_gap <= 0):
        raise ValueError("observation and target frame gaps must be positive")
    velocity = (last - previous) / observation_gap[:, None]
    return last + velocity * target_gap[:, None]


def state_to_normalized_xyxy(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    width = np.exp(states[:, 2])
    height = np.exp(states[:, 3])
    return np.stack(
        [
            states[:, 0] - width / 2,
            states[:, 1] - height / 2,
            states[:, 0] + width / 2,
            states[:, 1] + height / 2,
        ],
        axis=1,
    )


def paired_box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_box = state_to_normalized_xyxy(first)
    second_box = state_to_normalized_xyxy(second)
    intersection_width = np.maximum(
        0.0, np.minimum(first_box[:, 2], second_box[:, 2]) - np.maximum(first_box[:, 0], second_box[:, 0])
    )
    intersection_height = np.maximum(
        0.0, np.minimum(first_box[:, 3], second_box[:, 3]) - np.maximum(first_box[:, 1], second_box[:, 1])
    )
    intersection = intersection_width * intersection_height
    first_area = (first_box[:, 2] - first_box[:, 0]) * (first_box[:, 3] - first_box[:, 1])
    second_area = (second_box[:, 2] - second_box[:, 0]) * (second_box[:, 3] - second_box[:, 1])
    union = first_area + second_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    last_observation: np.ndarray,
    delta_std: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    last_observation = np.asarray(last_observation, dtype=np.float64)
    if prediction.shape != target.shape or target.shape != last_observation.shape:
        raise ValueError("prediction, target, and last observation shapes must match")
    if len(prediction) == 0:
        return {
            "sample_count": 0,
            "normalized_delta_mse": None,
            "state_mae_by_dimension": [None, None, None, None],
            "normalized_center_l2_mean": None,
            "normalized_center_l2_median": None,
            "box_iou_mean": None,
            "box_iou_median": None,
            "box_iou_at_least_0_5_fraction": None,
        }
    errors = prediction - target
    delta_error = ((prediction - last_observation) - (target - last_observation)) / delta_std
    iou = paired_box_iou(prediction, target)
    center_l2 = np.linalg.norm(errors[:, :2], axis=1)
    return {
        "sample_count": int(len(prediction)),
        "normalized_delta_mse": float(np.mean(delta_error ** 2)),
        "state_mae_by_dimension": np.mean(np.abs(errors), axis=0).tolist(),
        "normalized_center_l2_mean": float(np.mean(center_l2)),
        "normalized_center_l2_median": float(np.median(center_l2)),
        "box_iou_mean": float(np.mean(iou)),
        "box_iou_median": float(np.median(iou)),
        "box_iou_at_least_0_5_fraction": float(np.mean(iou >= 0.5)),
    }


def _gap_group(frame_gap: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "gap_1": frame_gap == 1,
        "gap_2_3": (frame_gap >= 2) & (frame_gap <= 3),
        "gap_4_7": (frame_gap >= 4) & (frame_gap <= 7),
        "gap_8_15": (frame_gap >= 8) & (frame_gap <= 15),
        "gap_16_31": (frame_gap >= 16) & (frame_gap <= 31),
        "gap_2_31": (frame_gap >= 2) & (frame_gap <= 31),
    }


def evaluate_prediction_set(
    prediction: np.ndarray,
    arrays: dict[str, np.ndarray],
    delta_std: np.ndarray,
) -> dict[str, Any]:
    target = arrays["target_state"]
    last = arrays["history_states"][:, -1]
    overall = prediction_metrics(prediction, target, last, delta_std)
    per_sequence: dict[str, Any] = {}
    for sequence in sorted(map(int, np.unique(arrays["sequence_index"]))):
        mask = arrays["sequence_index"] == sequence
        per_sequence[str(sequence)] = prediction_metrics(
            prediction[mask], target[mask], last[mask], delta_std
        )
    macro_sequence = {
        metric: float(np.mean([value[metric] for value in per_sequence.values()]))
        for metric in CORE_METRICS
    }
    by_gap = {}
    for name, mask in _gap_group(arrays["frame_gap"]).items():
        by_gap[name] = prediction_metrics(
            prediction[mask], target[mask], last[mask], delta_std
        )
    positive = arrays["positive_detection_id"] >= 0
    by_current_detection = {
        "present": prediction_metrics(prediction[positive], target[positive], last[positive], delta_std),
        "absent": prediction_metrics(prediction[~positive], target[~positive], last[~positive], delta_std),
    }
    return {
        "overall": overall,
        "macro_sequence": macro_sequence,
        "per_sequence": per_sequence,
        "by_gap": by_gap,
        "by_current_positive_detection": by_current_detection,
    }


def offline_feasibility_gate(
    seed_metrics: list[dict[str, Any]],
    *,
    mean_mse_ratio_max: float,
    every_seed_mse_ratio_max: float,
    gap_2_31_mse_ratio_max: float,
    macro_iou_drop_max: float,
    seed_mse_relative_range_max: float,
) -> dict[str, Any]:
    """Apply the frozen multi-seed gate without selecting a seed."""
    if len(seed_metrics) < 2:
        raise ValueError("multi-seed gate requires at least two seeds")
    comparator_names = ("zero_delta", "constant_velocity")
    reference = seed_metrics[0]["dev"]
    for metrics in seed_metrics[1:]:
        for name in comparator_names:
            if metrics["dev"][name] != reference[name]:
                raise ValueError(f"comparator metrics differ across seeds: {name}")

    learned_macro_mse = np.asarray([
        item["dev"]["learned_gru"]["macro_sequence"]["normalized_delta_mse"]
        for item in seed_metrics
    ], dtype=np.float64)
    baseline_macro_mse = {
        name: float(reference[name]["macro_sequence"]["normalized_delta_mse"])
        for name in comparator_names
    }
    best_macro_mse_name = min(baseline_macro_mse, key=baseline_macro_mse.get)
    best_macro_mse = baseline_macro_mse[best_macro_mse_name]

    learned_gap_mse = np.asarray([
        item["dev"]["learned_gru"]["by_gap"]["gap_2_31"]["normalized_delta_mse"]
        for item in seed_metrics
    ], dtype=np.float64)
    baseline_gap_mse = {
        name: float(reference[name]["by_gap"]["gap_2_31"]["normalized_delta_mse"])
        for name in comparator_names
    }
    best_gap_mse_name = min(baseline_gap_mse, key=baseline_gap_mse.get)
    best_gap_mse = baseline_gap_mse[best_gap_mse_name]

    learned_macro_iou = np.asarray([
        item["dev"]["learned_gru"]["macro_sequence"]["box_iou_mean"]
        for item in seed_metrics
    ], dtype=np.float64)
    baseline_macro_iou = {
        name: float(reference[name]["macro_sequence"]["box_iou_mean"])
        for name in comparator_names
    }
    best_macro_iou_name = max(baseline_macro_iou, key=baseline_macro_iou.get)
    best_macro_iou = baseline_macro_iou[best_macro_iou_name]

    mean_macro_mse = float(learned_macro_mse.mean())
    mean_gap_mse = float(learned_gap_mse.mean())
    mean_macro_iou = float(learned_macro_iou.mean())
    relative_range = float(np.ptp(learned_macro_mse) / mean_macro_mse)
    criteria = {
        "mean_macro_mse": bool(mean_macro_mse <= best_macro_mse * mean_mse_ratio_max),
        "every_seed_macro_mse": bool(np.all(learned_macro_mse <= best_macro_mse * every_seed_mse_ratio_max)),
        "mean_gap_2_31_mse": bool(mean_gap_mse <= best_gap_mse * gap_2_31_mse_ratio_max),
        "macro_iou_safety": bool(mean_macro_iou >= best_macro_iou - macro_iou_drop_max),
        "seed_stability": bool(relative_range <= seed_mse_relative_range_max),
    }
    sequence_wins = []
    for item in seed_metrics:
        learned_sequences = item["dev"]["learned_gru"]["per_sequence"]
        baseline_sequences = reference[best_macro_mse_name]["per_sequence"]
        wins = sum(
            learned_sequences[key]["normalized_delta_mse"] < baseline_sequences[key]["normalized_delta_mse"]
            for key in baseline_sequences
        )
        sequence_wins.append({"seed": int(item["seed"]), "wins": wins, "total": len(baseline_sequences)})
    return {
        "decision": "GO_TO_NATURAL_CLOSED_LOOP" if all(criteria.values()) else "NO_GO_STOP_MODEL_EXPANSION",
        "all_criteria_pass": all(criteria.values()),
        "criteria": criteria,
        "thresholds": {
            "mean_mse_ratio_max": mean_mse_ratio_max,
            "every_seed_mse_ratio_max": every_seed_mse_ratio_max,
            "gap_2_31_mse_ratio_max": gap_2_31_mse_ratio_max,
            "macro_iou_drop_max": macro_iou_drop_max,
            "seed_mse_relative_range_max": seed_mse_relative_range_max,
        },
        "learned": {
            "seeds": [int(item["seed"]) for item in seed_metrics],
            "macro_mse_by_seed": learned_macro_mse.tolist(),
            "macro_mse_mean": mean_macro_mse,
            "macro_mse_relative_range": relative_range,
            "gap_2_31_mse_by_seed": learned_gap_mse.tolist(),
            "gap_2_31_mse_mean": mean_gap_mse,
            "macro_iou_by_seed": learned_macro_iou.tolist(),
            "macro_iou_mean": mean_macro_iou,
            "sequence_mse_wins_vs_best_macro_mse_comparator": sequence_wins,
        },
        "comparators": {
            "macro_mse": baseline_macro_mse,
            "best_macro_mse_name": best_macro_mse_name,
            "gap_2_31_mse": baseline_gap_mse,
            "best_gap_2_31_mse_name": best_gap_mse_name,
            "macro_iou": baseline_macro_iou,
            "best_macro_iou_name": best_macro_iou_name,
        },
        "ratios_and_differences": {
            "learned_mean_over_best_macro_mse": mean_macro_mse / best_macro_mse,
            "learned_mean_over_best_gap_2_31_mse": mean_gap_mse / best_gap_mse,
            "learned_minus_best_macro_iou": mean_macro_iou - best_macro_iou,
        },
    }
