"""Pure helpers for D2 privileged first-association counterfactuals."""

from __future__ import annotations

from typing import Any

import numpy as np


def association_maps(
    matches: np.ndarray, tracker_ids: list[int], detection_ids: list[int]
) -> tuple[dict[int, int], dict[int, int]]:
    """Map stable tracker IDs to detection IDs and vice versa."""
    track_to_detection: dict[int, int] = {}
    detection_to_track: dict[int, int] = {}
    for detection_index, tracker_index in np.asarray(matches, dtype=int).reshape(-1, 2):
        tracker_id = tracker_ids[int(tracker_index)]
        detection_id = detection_ids[int(detection_index)]
        track_to_detection[tracker_id] = detection_id
        detection_to_track[detection_id] = tracker_id
    return track_to_detection, detection_to_track


def evaluate_association_counterfactual(
    intended_track_id: int,
    positive_detection_id: int,
    tracker_ids: list[int],
    detection_ids: list[int],
    baseline_matches: np.ndarray,
    oracle_matches: np.ndarray,
) -> dict[str, Any]:
    """Describe target/positive assignments before and after an oracle hint."""
    if intended_track_id not in tracker_ids:
        return {
            "status": "not_applicable",
            "non_applicable_reason": "intended_history_track_not_alive",
        }
    if positive_detection_id not in detection_ids:
        return {
            "status": "not_applicable",
            "non_applicable_reason": "positive_detection_not_in_first_round_pool",
        }
    baseline_t2d, baseline_d2t = association_maps(
        baseline_matches, tracker_ids, detection_ids
    )
    oracle_t2d, oracle_d2t = association_maps(oracle_matches, tracker_ids, detection_ids)
    baseline_positive = baseline_t2d.get(intended_track_id) == positive_detection_id
    oracle_positive = oracle_t2d.get(intended_track_id) == positive_detection_id
    baseline_pairs = set(baseline_t2d.items())
    oracle_pairs = set(oracle_t2d.items())
    return {
        "status": "applicable",
        "non_applicable_reason": "",
        "baseline_target_detection_id": baseline_t2d.get(intended_track_id),
        "baseline_positive_detection_track_id": baseline_d2t.get(positive_detection_id),
        "oracle_target_detection_id": oracle_t2d.get(intended_track_id),
        "oracle_positive_detection_track_id": oracle_d2t.get(positive_detection_id),
        "baseline_positive_match": baseline_positive,
        "oracle_positive_match": oracle_positive,
        "recoverable": (not baseline_positive) and oracle_positive,
        "harmed": baseline_positive and (not oracle_positive),
        "association_pairs_changed": len(baseline_pairs.symmetric_difference(oracle_pairs)),
    }


def compare_tracker_replay(
    reference_keys: np.ndarray,
    reference_boxes_xyxy: np.ndarray,
    replay_keys: np.ndarray,
    replay_boxes_xyxy: np.ndarray,
    atol: float,
) -> dict[str, Any]:
    """Apply the frozen G1 replay contract to tracker rows."""
    same_shape = (
        reference_keys.shape == replay_keys.shape
        and reference_boxes_xyxy.shape == replay_boxes_xyxy.shape
    )
    keys_exact = same_shape and np.array_equal(reference_keys, replay_keys)
    if same_shape and reference_boxes_xyxy.size:
        max_error = float(np.max(np.abs(reference_boxes_xyxy - replay_boxes_xyxy)))
    elif same_shape:
        max_error = 0.0
    else:
        max_error = None
    return {
        "status": "PASS" if keys_exact and max_error is not None and max_error <= atol else "FAIL",
        "frame_and_track_id_exact": keys_exact,
        "max_abs_xyxy_error": max_error,
        "atol": atol,
    }
