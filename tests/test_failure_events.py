import numpy as np

from mot_research.analysis.d2_first_association import (
    compare_tracker_replay,
    evaluate_association_counterfactual,
)
from mot_research.analysis.d1_history_correspondence import (
    evaluate_history_owner_relabel,
    summarize_gap_updates,
    summarize_history_owner,
)
from mot_research.analysis.d3_detection_coordinate import replace_detection_xyxy
from mot_research.analysis.failure_events import (
    Candidate,
    SequenceInfo,
    classify_event_candidate_contention,
    extract_recovery_events,
    hungarian_iou_matches,
    review_crop_bounds,
    review_frames,
    sample_events,
)


def box(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.asarray([x1, y1, x2, y2], dtype=np.float64)


def test_bounded_gap_outcome_and_proxies() -> None:
    info = SequenceInfo("synthetic", 3, 100, 100)
    gt = {
        1: {7: box(10, 10, 30, 50), 8: box(25, 10, 45, 50)},
        2: {7: box(12, 10, 32, 50), 8: box(27, 10, 47, 50)},
        3: {7: box(14, 10, 34, 50), 8: box(29, 10, 49, 50)},
    }
    detections = {
        1: [Candidate(1, box(10, 10, 30, 50), 0.9), Candidate(2, box(25, 10, 45, 50), 0.9)],
        2: [Candidate(2, box(27, 10, 47, 50), 0.9)],
        3: [Candidate(3, box(14, 10, 34, 50), 0.9), Candidate(2, box(29, 10, 49, 50), 0.9)],
    }
    tracker = {
        1: [Candidate(10, box(10, 10, 30, 50)), Candidate(20, box(25, 10, 45, 50))],
        2: [Candidate(20, box(27, 10, 47, 50))],
        3: [Candidate(11, box(14, 10, 34, 50)), Candidate(20, box(29, 10, 49, 50))],
    }
    events, counters = extract_recovery_events(info, gt, detections, tracker)
    target = [event for event in events if event["gt_id"] == 7]
    assert len(target) == 1
    assert target[0]["outcome"] == "identity_change_failure"
    assert target[0]["missing_length"] == 1
    assert target[0]["neighbor_overlap_proxy"] is True
    assert target[0]["boundary_proximity_proxy"] is False
    assert counters["diagnostic_consecutive_match_id_changes"] == 0


def test_hungarian_matching_and_sampling_are_deterministic() -> None:
    gt = {1: box(0, 0, 10, 10), 2: box(20, 0, 30, 10)}
    candidates = [Candidate(5, box(0, 0, 10, 10)), Candidate(6, box(20, 0, 30, 10))]
    matches, best = hungarian_iou_matches(gt, candidates, 0.5)
    assert matches == {1: (5, 1.0), 2: (6, 1.0)}
    assert best == {1: 1.0, 2: 1.0}
    events = [
        {"event_id": f"f{index}", "outcome": "identity_change_failure"}
        for index in range(5)
    ] + [
        {"event_id": f"s{index}", "outcome": "success_control"}
        for index in range(5)
    ]
    assert sample_events(events, 13, 3) == sample_events(events, 13, 3)
    assert len(sample_events(events, 13, 3)) == 6


def test_review_frames_and_crop_stay_in_image() -> None:
    event = {
        "before_frame": "10",
        "gap_start_frame": "11",
        "gap_end_frame": "11",
        "after_frame": "12",
    }
    assert review_frames(event) == [10, 11, 12]
    bounds = review_crop_bounds([box(0, 0, 20, 50), box(5, 0, 25, 50)], 1920, 1080)
    assert bounds[0] == 0 and bounds[1] == 0
    assert bounds[2] <= 1920 and bounds[3] <= 1080


def test_candidate_contention_is_separate_from_raw_absence() -> None:
    gt = {
        1: {1: box(0, 0, 10, 10), 2: box(2, 0, 12, 10)},
        2: {1: box(0, 0, 10, 10), 2: box(20, 0, 30, 10)},
    }
    detections = {
        1: [Candidate(9, box(2, 0, 12, 10), 0.9)],
        2: [Candidate(10, box(20, 0, 30, 10), 0.9)],
    }
    event = {
        "event_id": "synthetic:gt1:f0-3",
        "sequence": "synthetic",
        "gt_id": 1,
        "outcome": "unresolved",
        "missing_length": 2,
        "gap_start_frame": 1,
        "gap_end_frame": 2,
    }
    result = classify_event_candidate_contention(event, gt, detections)
    assert result["candidate_contention_frames"] == 1
    assert result["no_raw_candidate_frames"] == 1
    assert result["event_candidate_class"] == "mixed"


def test_d2_counterfactual_recovers_positive_assignment() -> None:
    baseline = np.asarray([[0, 1], [1, 0]])
    oracle = np.asarray([[0, 0], [1, 1]])
    result = evaluate_association_counterfactual(
        intended_track_id=10,
        positive_detection_id=100,
        tracker_ids=[10, 20],
        detection_ids=[100, 200],
        baseline_matches=baseline,
        oracle_matches=oracle,
    )
    assert result["baseline_positive_match"] is False
    assert result["oracle_positive_match"] is True
    assert result["recoverable"] is True
    missing = evaluate_association_counterfactual(
        intended_track_id=30,
        positive_detection_id=100,
        tracker_ids=[10, 20],
        detection_ids=[100, 200],
        baseline_matches=baseline,
        oracle_matches=oracle,
    )
    assert missing["status"] == "not_applicable"
    assert missing["non_applicable_reason"] == "intended_history_track_not_alive"


def test_d1_latest_history_owner_and_relabel() -> None:
    history = [
        {"frame": 5, "detection_id": 0, "gt_id": 2},
        {"frame": 8, "detection_id": 1, "gt_id": None},
        {"frame": 9, "detection_id": 2, "gt_id": 7},
    ]
    owner = summarize_history_owner(history, evaluation_frame=10, window_frames=4)
    assert owner == {
        "owner_gt_id": 7,
        "owner_observation_frame": 9,
        "owner_age_frames": 1,
        "mapped_observations": 1,
        "unmapped_observations": 1,
        "unique_mapped_gt_ids": 1,
    }
    result = evaluate_history_owner_relabel(
        intended_track_id=4,
        target_gt_id=7,
        positive_detection_id=12,
        tracker_ids=[4, 9],
        detection_ids=[12, 13],
        baseline_matches=np.asarray([[0, 1], [1, 0]]),
        positive_track_owner=owner,
    )
    assert result["status"] == "applicable"
    assert result["baseline_positive_match"] is False
    assert result["D1_history_owner_positive_match"] is True
    assert result["recoverable"] is True
    assert result["harmed"] is False

    gap = summarize_gap_updates(history, 5, 8, target_gt_id=2)
    assert gap == {
        "baseline_gap_updates": 2,
        "baseline_gap_updates_target_owner": 1,
        "baseline_gap_updates_other_owner": 0,
        "baseline_gap_updates_unmapped_owner": 1,
    }


def test_d3_replaces_only_one_detection_box() -> None:
    detections = np.asarray([
        [1.0, 2.0, 11.0, 22.0, 0.91],
        [30.0, 40.0, 50.0, 70.0, 0.73],
    ])
    corrected = replace_detection_xyxy(
        detections, 1, np.asarray([31.0, 39.0, 52.0, 72.0])
    )
    assert np.array_equal(corrected[0], detections[0])
    assert np.array_equal(corrected[1, :4], [31.0, 39.0, 52.0, 72.0])
    assert corrected[1, 4] == detections[1, 4]
    assert np.array_equal(detections[1], [30.0, 40.0, 50.0, 70.0, 0.73])


def test_d2_replay_contract_requires_exact_keys_and_bounded_xyxy() -> None:
    keys = np.asarray([[1, 7], [2, 7]])
    boxes = np.asarray([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0]])
    close = boxes.copy()
    close[1, 2] += 0.00004
    assert compare_tracker_replay(keys, boxes, keys.copy(), close, 0.0001)["status"] == "PASS"
    wrong_keys = keys.copy()
    wrong_keys[1, 1] = 8
    assert compare_tracker_replay(keys, boxes, wrong_keys, close, 0.0001)["status"] == "FAIL"
