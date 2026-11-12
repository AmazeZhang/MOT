from __future__ import annotations

import numpy as np

from mot_research.evaluation.motion_prediction import (
    constant_velocity_prediction,
    evaluate_prediction_set,
    offline_feasibility_gate,
    paired_box_iou,
    prediction_metrics,
    zero_delta_prediction,
)


def fixture_arrays() -> dict[str, np.ndarray]:
    return {
        "history_states": np.asarray([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0.2, -2, -2], [0.2, 0.3, -2, -2]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.5, -2, -2], [0.5, 0.6, -2, -2]],
        ], dtype=np.float64),
        "history_frames": np.asarray([[0, 0, 1, 3], [0, 0, 2, 4]], dtype=np.int32),
        "history_valid": np.asarray([[False, False, True, True], [False, False, True, True]]),
        "target_frame": np.asarray([5, 5], dtype=np.int32),
        "target_state": np.asarray([[0.3, 0.4, -2, -2], [0.55, 0.65, -2, -2]]),
        "sequence_index": np.asarray([0, 1], dtype=np.int32),
        "frame_gap": np.asarray([2, 1], dtype=np.int32),
        "positive_detection_id": np.asarray([7, -1], dtype=np.int32),
    }


def test_comparators_follow_frozen_definitions() -> None:
    arrays = fixture_arrays()
    zero = zero_delta_prediction(arrays["history_states"], arrays["history_valid"])
    velocity = constant_velocity_prediction(
        arrays["history_states"], arrays["history_frames"], arrays["history_valid"], arrays["target_frame"]
    )
    assert np.array_equal(zero, arrays["history_states"][:, -1])
    assert np.allclose(velocity, arrays["target_state"])


def test_iou_and_metrics_are_exact_for_perfect_prediction() -> None:
    arrays = fixture_arrays()
    target = arrays["target_state"]
    assert np.array_equal(paired_box_iou(target, target), np.ones(2))
    metrics = prediction_metrics(
        target, target, arrays["history_states"][:, -1], np.ones(4)
    )
    assert metrics["normalized_delta_mse"] == 0.0
    assert metrics["box_iou_mean"] == 1.0


def test_evaluation_has_sequence_gap_and_detection_groups() -> None:
    arrays = fixture_arrays()
    result = evaluate_prediction_set(arrays["target_state"], arrays, np.ones(4))
    assert result["overall"]["sample_count"] == 2
    assert set(result["per_sequence"]) == {"0", "1"}
    assert result["by_gap"]["gap_1"]["sample_count"] == 1
    assert result["by_gap"]["gap_2_31"]["sample_count"] == 1
    assert result["by_gap"]["gap_4_7"]["normalized_delta_mse"] is None
    assert result["by_current_positive_detection"]["present"]["sample_count"] == 1
    assert result["by_current_positive_detection"]["absent"]["sample_count"] == 1


def synthetic_seed(seed: int, learned_mse: float, learned_gap: float, learned_iou: float) -> dict[str, object]:
    def method(mse: float, gap: float, iou: float) -> dict[str, object]:
        return {
            "macro_sequence": {"normalized_delta_mse": mse, "box_iou_mean": iou},
            "by_gap": {"gap_2_31": {"normalized_delta_mse": gap}},
            "per_sequence": {
                "0": {"normalized_delta_mse": mse},
                "1": {"normalized_delta_mse": mse + 0.01},
            },
        }
    return {
        "seed": seed,
        "dev": {
            "learned_gru": method(learned_mse, learned_gap, learned_iou),
            "zero_delta": method(1.0, 1.2, 0.80),
            "constant_velocity": method(1.1, 1.4, 0.78),
        },
    }


def test_offline_gate_passes_only_when_all_frozen_criteria_pass() -> None:
    seeds = [
        synthetic_seed(1, 0.90, 1.00, 0.802),
        synthetic_seed(2, 0.91, 1.02, 0.801),
        synthetic_seed(3, 0.89, 0.98, 0.800),
    ]
    result = offline_feasibility_gate(
        seeds, mean_mse_ratio_max=0.98, every_seed_mse_ratio_max=1.0,
        gap_2_31_mse_ratio_max=0.95, macro_iou_drop_max=0.005,
        seed_mse_relative_range_max=0.1,
    )
    assert result["decision"] == "GO_TO_NATURAL_CLOSED_LOOP"
    seeds[2] = synthetic_seed(3, 1.01, 0.98, 0.800)
    failed = offline_feasibility_gate(
        seeds, mean_mse_ratio_max=0.98, every_seed_mse_ratio_max=1.0,
        gap_2_31_mse_ratio_max=0.95, macro_iou_drop_max=0.005,
        seed_mse_relative_range_max=0.2,
    )
    assert not failed["criteria"]["every_seed_macro_mse"]
    assert failed["decision"] == "NO_GO_STOP_MODEL_EXPANSION"
