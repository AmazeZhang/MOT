from __future__ import annotations

import numpy as np

from mot_research.analysis.p1_mechanism import (
    box_geometry_metrics,
    diagnose_learned_vs_reference,
    event_assignment_metrics,
    summarize_method_rows,
)


def test_box_geometry_metrics_is_exact_for_equal_boxes() -> None:
    box = np.asarray([10.0, 20.0, 30.0, 60.0])
    result = box_geometry_metrics(box, box, 100, 100)
    assert result == {
        "target_iou": 1.0,
        "center_error_normalized": 0.0,
        "log_width_abs_error": 0.0,
        "log_height_abs_error": 0.0,
    }


def test_event_assignment_metrics_exposes_gate_competition_and_owner() -> None:
    result = event_assignment_metrics(
        intended_track_id=10,
        positive_detection_id=100,
        tracker_ids=[10, 20],
        detection_ids=[100],
        matches=np.asarray([[0, 1]], dtype=int),
        iou_matrix=np.asarray([[0.4, 0.5]]),
        total_score_matrix=np.asarray([[0.45, 0.7]]),
        iou_threshold=0.3,
    )
    assert result["status"] == "applicable"
    assert result["intended_gate_eligible"] is True
    assert result["eligible_other_track_count"] == 1
    assert result["best_eligible_other_track_id"] == 20
    assert np.isclose(result["intended_minus_best_eligible_other"], -0.25)
    assert result["positive_detection_assigned_track_id"] == 20
    assert result["immediate_positive_match"] is False


def method_rows(target_iou: float, center: float, gate: int, match: int) -> list[dict[str, object]]:
    rows = []
    for index in range(10):
        rows.append({
            "status": "applicable",
            "non_applicable_reason": "",
            "target_iou": target_iou,
            "center_error_normalized": center,
            "log_width_abs_error": 0.1,
            "log_height_abs_error": 0.1,
            "intended_positive_iou": target_iou,
            "intended_positive_total_score": target_iou,
            "intended_minus_best_eligible_other": -0.1,
            "intended_gate_eligible": index < gate,
            "immediate_positive_match": index < match,
            "eligible_other_track_count": 1,
        })
    return rows


def test_frozen_diagnosis_selects_earliest_consistent_stage() -> None:
    summaries = {
        "constant_velocity": summarize_method_rows(method_rows(0.60, 0.02, 8, 7)),
        "learned_a": summarize_method_rows(method_rows(0.55, 0.03, 6, 5)),
        "learned_b": summarize_method_rows(method_rows(0.56, 0.03, 6, 5)),
        "learned_c": summarize_method_rows(method_rows(0.61, 0.02, 8, 7)),
    }
    result = diagnose_learned_vs_reference(
        summaries,
        learned_methods=["learned_a", "learned_b", "learned_c"],
        reference_method="constant_velocity",
        thresholds={
            "minimum_consistent_learned_seeds": 2,
            "target_iou_disadvantage_minimum": 0.01,
            "center_error_disadvantage_minimum": 0.005,
            "gate_rate_disadvantage_minimum": 0.02,
            "immediate_match_rate_disadvantage_minimum": 0.02,
        },
    )
    assert result["earliest_triggered_stage"] == "geometry_iou"
    assert result["checks"]["geometry_iou"]["consistent_seeds"] == 2
