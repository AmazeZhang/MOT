from __future__ import annotations

import numpy as np

from mot_research.analysis.failure_events import Candidate
from mot_research.evaluation.closed_loop_events import (
    build_common_recovery_events,
    evaluate_method_recovery_events,
    gap_bin,
)


BOX = np.asarray([0, 0, 10, 10], dtype=float)


def candidates(frames_to_id: dict[int, int]) -> dict[int, list[Candidate]]:
    return {
        frame: [Candidate(candidate_id=value, box_xyxy=BOX.copy(), score=0.9)]
        for frame, value in frames_to_id.items()
    }


def test_gap_bins_are_frozen_and_bounded() -> None:
    assert [gap_bin(value) for value in (1, 2, 3, 4, 7, 8, 15, 16, 31)] == [
        "gap1", "gap2_3", "gap2_3", "gap4_7", "gap4_7",
        "gap8_15", "gap8_15", "gap16_31", "gap16_31",
    ]


def test_common_events_use_baseline_population_and_fixed_five_frame_persistence() -> None:
    gt = {frame: {1: BOX.copy()} for frame in range(1, 9)}
    detections = candidates({1: 101, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107, 8: 108})
    baseline = candidates({1: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10})
    events, counters = build_common_recovery_events(
        sequence="seq", gt_by_frame=gt, detections_by_frame=detections,
        baseline_tracker_by_frame=baseline, diagnostic_anchor=True,
        persistence_frames=5,
    )
    assert len(events) == 1
    event = events[0]
    assert event["event_id"] == "seq:gt1:f1-3"
    assert event["gap_bin"] == "gap1"
    assert event["baseline_stratum"] == "success_control"
    assert event["baseline_persistence_success"] is True
    assert counters["included_success_control"] == 1

    harmed = candidates({1: 20, 3: 20, 4: 20, 5: 21, 6: 21, 7: 21})
    outcomes, summary = evaluate_method_recovery_events(
        events=events, gt_by_frame=gt, tracker_by_frame=harmed,
        persistence_frames=5,
    )
    assert outcomes[0]["reconnect_success"] is True
    assert outcomes[0]["persistence_success"] is False
    assert summary["success_control_harms"] == 1


def test_baseline_failure_recovery_and_right_censoring_are_separate() -> None:
    gt = {frame: {1: BOX.copy()} for frame in range(1, 7)}
    detections = candidates({1: 101, 3: 103, 4: 104, 5: 105, 6: 106})
    baseline = candidates({1: 10, 3: 11, 4: 11, 5: 11, 6: 11})
    events, _ = build_common_recovery_events(
        sequence="seq", gt_by_frame=gt, detections_by_frame=detections,
        baseline_tracker_by_frame=baseline, diagnostic_anchor=False,
        persistence_frames=3,
    )
    assert events[0]["baseline_stratum"] == "recovery_failure"
    recovered = candidates({1: 20, 3: 20, 4: 20, 5: 20, 6: 20})
    _, summary = evaluate_method_recovery_events(
        events=events, gt_by_frame=gt, tracker_by_frame=recovered,
        persistence_frames=3,
    )
    assert summary["baseline_failure_recoveries"] == 1

    censored_gt = {frame: {1: BOX.copy()} for frame in range(1, 5)}
    censored_detections = candidates({1: 1, 3: 3, 4: 4})
    censored_baseline = candidates({1: 10, 3: 10, 4: 10})
    censored, _ = build_common_recovery_events(
        sequence="seq2", gt_by_frame=censored_gt,
        detections_by_frame=censored_detections,
        baseline_tracker_by_frame=censored_baseline,
        diagnostic_anchor=False, persistence_frames=3,
    )
    assert censored[0]["persistence_censored"] is True
    assert censored[0]["baseline_stratum"] == "persistence_censored"
