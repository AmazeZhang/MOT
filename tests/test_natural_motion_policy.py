from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mot_research.tracking.first_round_adapter import FirstRoundRequest
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner,
    NaturalMotionFirstRoundPolicy,
    TrackerDetectionHistory,
)


def tracker(track_id: int, frame_age: int, observation: list[float] | None) -> SimpleNamespace:
    if observation is None:
        return SimpleNamespace(
            id=track_id - 1,
            time_since_update=frame_age,
            last_observation=np.asarray([-1, -1, -1, -1, -1], dtype=float),
        )
    return SimpleNamespace(
        id=track_id - 1,
        time_since_update=frame_age,
        last_observation=np.asarray(observation, dtype=float),
    )


def request(frame_id: int, tracker_ids: tuple[int, ...], boxes: np.ndarray) -> FirstRoundRequest:
    boxes = np.asarray(boxes, dtype=float).reshape(len(tracker_ids), 5)
    return FirstRoundRequest(
        frame_id=frame_id,
        call_index=frame_id,
        tracker_ids=tracker_ids,
        detections=np.empty((0, 5), dtype=float),
        trackers=boxes,
        iou_threshold=0.3,
        velocities=np.zeros((len(tracker_ids), 2), dtype=float),
        previous_observations=np.full((len(tracker_ids), 5), -1.0),
        velocity_direction_weight=0.2,
    )


def capturing_upstream(captured: list[np.ndarray]):
    def associate(detections, trackers, threshold, velocities, observations, weight):
        del detections, threshold, velocities, observations, weight
        captured.append(np.asarray(trackers).copy())
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=int),
            np.arange(len(trackers), dtype=int),
        )
    return associate


def test_history_is_post_update_strictly_online_and_purges_deleted_tracks() -> None:
    history = TrackerDetectionHistory(history_length=3, score_threshold=0.6)
    assert history.observations(5) == ()
    history.record_after_update(1, [tracker(5, 0, [10, 10, 20, 30, 0.9])])
    history.record_after_update(2, [tracker(5, 1, None)])
    history.record_after_update(3, [tracker(5, 0, [12, 10, 22, 30, 0.8])])
    assert [item.frame_id for item in history.observations(5)] == [1, 3]
    history.record_after_update(4, [])
    assert history.observations(5) == ()
    assert history.counters["purged_deleted_track"] == 1


def test_zero_delta_replaces_only_eligible_row_and_preserves_input() -> None:
    history = TrackerDetectionHistory()
    history.record_after_update(1, [
        tracker(5, 0, [10, 10, 20, 30, 0.9]),
        tracker(7, 0, [30, 30, 40, 50, 0.9]),
    ])
    history.record_after_update(2, [
        tracker(5, 0, [12, 10, 22, 30, 0.9]),
        tracker(7, 1, None),
    ])
    original = np.asarray([[100, 100, 110, 120, 0], [200, 200, 210, 220, 0]], dtype=float)
    req = request(3, (5, 7), original)
    policy = NaturalMotionFirstRoundPolicy(
        mode="zero_delta", history=history, image_width=100, image_height=100, fps=20
    )
    captured: list[np.ndarray] = []
    policy(req, capturing_upstream(captured))
    assert np.array_equal(req.trackers, original)
    assert np.allclose(captured[0][0, :4], [12, 10, 22, 30])
    assert np.array_equal(captured[0][1], original[1])
    assert policy.counters["replaced_tracker_rows"] == 1
    assert policy.counters["fallback_insufficient_history"] == 1


def test_constant_velocity_uses_last_two_observation_frame_gaps() -> None:
    history = TrackerDetectionHistory()
    history.record_after_update(1, [tracker(5, 0, [10, 10, 20, 30, 0.9])])
    history.record_after_update(3, [tracker(5, 0, [12, 14, 22, 34, 0.9])])
    req = request(5, (5,), np.asarray([[99, 99, 109, 119, 0]], dtype=float))
    policy = NaturalMotionFirstRoundPolicy(
        mode="constant_velocity", history=history,
        image_width=100, image_height=100, fps=20,
    )
    captured: list[np.ndarray] = []
    policy(req, capturing_upstream(captured))
    assert np.allclose(captured[0][0, :4], [14, 18, 24, 38], atol=1e-12)


class FakePredictor:
    def predict_target_state(self, states, frames, valid, target_frames, fps):
        assert np.all(frames[valid] < np.repeat(target_frames, valid.sum(axis=1)))
        assert np.all(fps == 20)
        result = states[:, -1].copy()
        result[:, 0] += 0.1
        return result


def test_learned_policy_batches_rows_and_falls_back_on_invalid_prediction() -> None:
    history = TrackerDetectionHistory()
    history.record_after_update(1, [
        tracker(1, 0, [10, 10, 20, 20, 0.9]),
        tracker(2, 0, [30, 30, 40, 40, 0.9]),
    ])
    history.record_after_update(2, [
        tracker(1, 0, [11, 10, 21, 20, 0.9]),
        tracker(2, 0, [31, 30, 41, 40, 0.9]),
    ])
    req = request(3, (1, 2), np.asarray([
        [100, 100, 110, 110, 0], [200, 200, 210, 210, 0]
    ], dtype=float))
    policy = NaturalMotionFirstRoundPolicy(
        mode="learned", history=history, image_width=100, image_height=100,
        fps=20, predictor=FakePredictor(),
    )
    captured: list[np.ndarray] = []
    policy(req, capturing_upstream(captured))
    assert np.allclose(captured[0][0, :4], [21, 10, 31, 20], atol=1e-12)
    assert np.allclose(captured[0][1, :4], [41, 30, 51, 40], atol=1e-12)
    assert policy.counters["replaced_tracker_rows"] == 2


def test_gap_and_observation_validation_are_visible_fallbacks() -> None:
    history = TrackerDetectionHistory(score_threshold=0.6)
    history.record_after_update(1, [tracker(1, 0, [10, 10, 20, 20, 0.6])])
    assert history.observations(1) == ()
    history.record_after_update(2, [tracker(2, 0, [10, 10, 20, 20, 0.9])])
    history.record_after_update(3, [tracker(2, 0, [11, 10, 21, 20, 0.9])])
    req = request(40, (1, 2), np.asarray([
        [1, 1, 2, 2, 0], [3, 3, 4, 4, 0]
    ], dtype=float))
    policy = NaturalMotionFirstRoundPolicy(
        mode="zero_delta", history=history, image_width=100, image_height=100,
        fps=20, maximum_frame_gap=31,
    )
    captured: list[np.ndarray] = []
    policy(req, capturing_upstream(captured))
    assert np.array_equal(captured[0], req.trackers)
    assert history.counters["rejected_below_or_equal_score_threshold"] == 1
    assert policy.counters["fallback_zero_history"] == 1
    assert policy.counters["fallback_gap_above_maximum"] == 1


def test_actual_ocsort_upstream_mode_is_exact_and_records_after_update() -> None:
    pytest.importorskip("filterpy")
    root = Path(__file__).parents[1] / "third_party" / "OC_SORT"
    sys.path.insert(0, str(root))
    import trackers.ocsort_tracker.ocsort as ocsort_module

    frames = [
        np.asarray([[10, 10, 20, 30, 0.9]], dtype=float),
        np.asarray([[11, 10, 21, 30, 0.9]], dtype=float),
        np.empty((0, 5), dtype=float),
        np.asarray([[13, 10, 23, 30, 0.9]], dtype=float),
    ]
    tracker_args = dict(
        det_thresh=0.6, iou_threshold=0.3, asso_func="iou",
        delta_t=3, inertia=0.2, use_byte=False,
    )
    baseline = ocsort_module.OCSort(**tracker_args)
    expected = [
        baseline.update(value.copy(), (100, 100), (100, 100))
        for value in frames
    ]
    wrapped = ocsort_module.OCSort(**tracker_args)
    history = TrackerDetectionHistory()
    policy = NaturalMotionFirstRoundPolicy(
        mode="upstream", history=history, image_width=100, image_height=100, fps=20,
    )
    runner = NaturalHistoryTrackerRunner(wrapped, ocsort_module, policy)
    actual = [
        runner.update(value.copy(), (100, 100), (100, 100))
        for value in frames
    ]
    assert all(np.array_equal(first, second) for first, second in zip(expected, actual))
    assert [item.frame_id for item in history.observations(1)] == [1, 2, 4]
    assert runner.association_calls == 4


def test_actual_source_frame_gap_skips_none_history_without_compressing_time() -> None:
    pytest.importorskip("filterpy")
    root = Path(__file__).parents[1] / "third_party" / "OC_SORT"
    sys.path.insert(0, str(root))
    import trackers.ocsort_tracker.ocsort as ocsort_module

    tracker_instance = ocsort_module.OCSort(
        det_thresh=0.6, iou_threshold=0.3, asso_func="iou",
        delta_t=3, inertia=0.2, use_byte=False,
    )
    history = TrackerDetectionHistory()
    policy = NaturalMotionFirstRoundPolicy(
        mode="constant_velocity", history=history,
        image_width=100, image_height=100, fps=20,
    )
    runner = NaturalHistoryTrackerRunner(tracker_instance, ocsort_module, policy)
    runner.update(
        np.asarray([[10, 10, 20, 30, 0.9]], dtype=float),
        (100, 100), (100, 100), frame_id=1,
    )
    runner.update(None, (100, 100), (100, 100), frame_id=2)
    runner.update(
        np.asarray([[12, 10, 22, 30, 0.9]], dtype=float),
        (100, 100), (100, 100), frame_id=3,
    )
    assert [item.frame_id for item in history.observations(1)] == [1, 3]
    assert runner.snapshot()["last_source_frame_id"] == 3
    assert runner.birth_mapping_counters["upstream_none_frame_no_history_append"] == 1
