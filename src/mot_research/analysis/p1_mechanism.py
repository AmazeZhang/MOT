"""Pure helpers for the P1 common-state mechanism diagnosis.

The diagnosis is intentionally one-step and read-only: every method receives
the same B0 first-round association state.  These helpers contain no tracker
mutation and make the geometry/assignment definitions directly testable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from mot_research.analysis.d2_first_association import association_maps
from mot_research.analysis.failure_events import pairwise_iou


def box_geometry_metrics(
    prediction_xyxy: np.ndarray,
    target_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    """Measure center, size and IoU error without dataset-fitted scaling."""
    prediction = np.asarray(prediction_xyxy, dtype=np.float64).reshape(4)
    target = np.asarray(target_xyxy, dtype=np.float64).reshape(4)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("boxes must be finite")
    prediction_wh = prediction[2:] - prediction[:2]
    target_wh = target[2:] - target[:2]
    if np.any(prediction_wh <= 0) or np.any(target_wh <= 0):
        raise ValueError("boxes must have positive width and height")
    prediction_center = (prediction[:2] + prediction[2:]) / 2.0
    target_center = (target[:2] + target[2:]) / 2.0
    diagonal = float(np.hypot(image_width, image_height))
    return {
        "target_iou": float(pairwise_iou(prediction[None], target[None])[0, 0]),
        "center_error_normalized": float(
            np.linalg.norm(prediction_center - target_center) / diagonal
        ),
        "log_width_abs_error": float(abs(np.log(prediction_wh[0] / target_wh[0]))),
        "log_height_abs_error": float(abs(np.log(prediction_wh[1] / target_wh[1]))),
    }


def event_assignment_metrics(
    *,
    intended_track_id: int,
    positive_detection_id: int,
    tracker_ids: list[int],
    detection_ids: list[int],
    matches: np.ndarray,
    iou_matrix: np.ndarray,
    total_score_matrix: np.ndarray,
    iou_threshold: float,
) -> dict[str, Any]:
    """Describe the intended pair, its gate, competitors and final assignment."""
    if positive_detection_id not in detection_ids:
        return {
            "status": "not_applicable",
            "non_applicable_reason": "positive_detection_not_in_first_round_pool",
        }
    if intended_track_id not in tracker_ids:
        return {
            "status": "not_applicable",
            "non_applicable_reason": "intended_b0_track_not_alive",
        }
    if iou_matrix.shape != (len(detection_ids), len(tracker_ids)):
        raise ValueError("IoU matrix shape does not match IDs")
    if total_score_matrix.shape != iou_matrix.shape:
        raise ValueError("score matrix shape does not match IoU")

    detection_index = detection_ids.index(positive_detection_id)
    tracker_index = tracker_ids.index(intended_track_id)
    intended_iou = float(iou_matrix[detection_index, tracker_index])
    intended_score = float(total_score_matrix[detection_index, tracker_index])
    eligible_other: list[tuple[float, int]] = []
    for index, track_id in enumerate(tracker_ids):
        if track_id == intended_track_id:
            continue
        if float(iou_matrix[detection_index, index]) >= iou_threshold:
            eligible_other.append((float(total_score_matrix[detection_index, index]), track_id))
    eligible_other.sort(key=lambda item: (-item[0], item[1]))
    best_other_score = eligible_other[0][0] if eligible_other else None
    best_other_track_id = eligible_other[0][1] if eligible_other else None
    track_to_detection, detection_to_track = association_maps(
        matches, tracker_ids, detection_ids
    )
    return {
        "status": "applicable",
        "non_applicable_reason": "",
        "intended_positive_iou": intended_iou,
        "intended_positive_total_score": intended_score,
        "intended_gate_eligible": intended_iou >= iou_threshold,
        "eligible_other_track_count": len(eligible_other),
        "best_eligible_other_track_id": best_other_track_id,
        "best_eligible_other_score": best_other_score,
        "intended_minus_best_eligible_other": (
            None if best_other_score is None else intended_score - best_other_score
        ),
        "positive_detection_assigned_track_id": detection_to_track.get(
            positive_detection_id
        ),
        "intended_assigned_detection_id": track_to_detection.get(intended_track_id),
        "immediate_positive_match": (
            track_to_detection.get(intended_track_id) == positive_detection_id
        ),
    }


def summarize_method_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one method's long-format event rows."""
    values = list(rows)
    counts: Counter[str] = Counter()
    applicable = [row for row in values if row["status"] == "applicable"]
    for row in values:
        counts[f"status_{row['status']}"] += 1
        if row["status"] != "applicable":
            counts[f"reason_{row['non_applicable_reason']}"] += 1
    numeric_fields = (
        "target_iou",
        "center_error_normalized",
        "log_width_abs_error",
        "log_height_abs_error",
        "intended_positive_iou",
        "intended_positive_total_score",
    )
    metrics: dict[str, Any] = {}
    for field in numeric_fields:
        array = np.asarray([float(row[field]) for row in applicable], dtype=np.float64)
        metrics[field] = {
            "mean": float(array.mean()) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
        }
    margins = np.asarray(
        [
            float(row["intended_minus_best_eligible_other"])
            for row in applicable
            if row["intended_minus_best_eligible_other"] is not None
        ],
        dtype=np.float64,
    )
    metrics["intended_minus_best_eligible_other"] = {
        "count": int(len(margins)),
        "mean": float(margins.mean()) if len(margins) else None,
        "median": float(np.median(margins)) if len(margins) else None,
    }
    return {
        "events": len(values),
        "applicable": len(applicable),
        "counts": dict(sorted(counts.items())),
        "intended_gate_eligible": sum(
            bool(row["intended_gate_eligible"]) for row in applicable
        ),
        "immediate_positive_match": sum(
            bool(row["immediate_positive_match"]) for row in applicable
        ),
        "has_eligible_other": sum(
            int(row["eligible_other_track_count"]) > 0 for row in applicable
        ),
        "metrics": metrics,
    }


def grouped_method_summaries(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: summarize_method_rows(group)
        for name, group in sorted(groups.items())
    }


def diagnose_learned_vs_reference(
    method_summaries: dict[str, dict[str, Any]],
    *,
    learned_methods: list[str],
    reference_method: str,
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    """Apply frozen descriptive thresholds to locate the earliest disadvantage."""
    if not learned_methods or reference_method not in method_summaries:
        raise ValueError("learned methods and reference must be present")
    for method in learned_methods:
        if method not in method_summaries:
            raise ValueError(f"missing learned method summary: {method}")
    reference = method_summaries[reference_method]
    learned = [method_summaries[method] for method in learned_methods]
    reference_applicable = int(reference["applicable"])
    if any(int(item["applicable"]) != reference_applicable for item in learned):
        raise ValueError("common-state method applicability must be identical")
    if reference_applicable == 0:
        raise ValueError("diagnosis has no applicable events")

    def mean_metric(summary: dict[str, Any], name: str) -> float:
        value = summary["metrics"][name]["mean"]
        if value is None:
            raise ValueError(f"missing mean metric {name}")
        return float(value)

    learned_mean_iou = float(np.mean([
        mean_metric(item, "target_iou") for item in learned
    ]))
    learned_mean_center = float(np.mean([
        mean_metric(item, "center_error_normalized") for item in learned
    ]))
    reference_iou = mean_metric(reference, "target_iou")
    reference_center = mean_metric(reference, "center_error_normalized")
    learned_gate_rate = float(np.mean([
        int(item["intended_gate_eligible"]) / reference_applicable for item in learned
    ]))
    reference_gate_rate = int(reference["intended_gate_eligible"]) / reference_applicable
    learned_match_rate = float(np.mean([
        int(item["immediate_positive_match"]) / reference_applicable for item in learned
    ]))
    reference_match_rate = int(reference["immediate_positive_match"]) / reference_applicable

    minimum_consistent = int(thresholds["minimum_consistent_learned_seeds"])
    checks = {
        "geometry_iou": {
            "disadvantage": reference_iou - learned_mean_iou,
            "threshold": float(thresholds["target_iou_disadvantage_minimum"]),
            "consistent_seeds": sum(
                mean_metric(item, "target_iou") < reference_iou for item in learned
            ),
        },
        "geometry_center": {
            "disadvantage": learned_mean_center - reference_center,
            "threshold": float(thresholds["center_error_disadvantage_minimum"]),
            "consistent_seeds": sum(
                mean_metric(item, "center_error_normalized") > reference_center
                for item in learned
            ),
        },
        "gate": {
            "disadvantage": reference_gate_rate - learned_gate_rate,
            "threshold": float(thresholds["gate_rate_disadvantage_minimum"]),
            "consistent_seeds": sum(
                int(item["intended_gate_eligible"]) / reference_applicable
                < reference_gate_rate
                for item in learned
            ),
        },
        "assignment": {
            "disadvantage": reference_match_rate - learned_match_rate,
            "threshold": float(thresholds["immediate_match_rate_disadvantage_minimum"]),
            "consistent_seeds": sum(
                int(item["immediate_positive_match"]) / reference_applicable
                < reference_match_rate
                for item in learned
            ),
        },
    }
    for value in checks.values():
        value["triggered"] = (
            float(value["disadvantage"]) >= float(value["threshold"])
            and int(value["consistent_seeds"]) >= minimum_consistent
        )
    ordered = ["geometry_iou", "geometry_center", "gate", "assignment"]
    triggered = [name for name in ordered if bool(checks[name]["triggered"])]
    return {
        "reference_method": reference_method,
        "learned_methods": learned_methods,
        "applicable_events": reference_applicable,
        "checks": checks,
        "triggered_stages": triggered,
        "earliest_triggered_stage": triggered[0] if triggered else None,
        "interpretation": (
            "descriptive_mechanism_candidate_requires_event_level_review"
            if triggered
            else "no_registered_discriminating_mechanism_stop_learned_motion_branch"
        ),
    }
