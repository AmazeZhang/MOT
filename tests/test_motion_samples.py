from __future__ import annotations

import numpy as np

from mot_research.analysis.failure_events import Candidate
from mot_research.data.motion_samples import (
    array_content_sha256,
    box_xyxy_to_state,
    build_sequence_motion_samples,
    fit_motion_normalization,
    state_to_box_xyxy,
)


def candidate(frame: int, box: list[float]) -> Candidate:
    return Candidate(frame, np.asarray(box, dtype=np.float64), 0.9)


def test_state_roundtrip() -> None:
    boxes = np.asarray([[10.5, 20.25, 50.75, 90.5], [0.0, 0.0, 1920.0, 1080.0]])
    states = box_xyxy_to_state(boxes, 1920, 1080)
    reconstructed = state_to_box_xyxy(states, 1920, 1080)
    assert np.allclose(reconstructed, boxes, atol=1e-12, rtol=0)


def test_current_and_future_detections_do_not_enter_history() -> None:
    gt = {
        frame: {1: np.asarray([frame, 0.0, frame + 10.0, 20.0])}
        for frame in range(1, 6)
    }
    detections = {
        frame: [candidate(frame, [frame, 0.0, frame + 10.0, 20.0])]
        for frame in range(1, 6)
    }
    baseline = build_sequence_motion_samples(
        sequence_index=0, sequence_length=5, image_width=100, image_height=100,
        fps=20.0, gt_by_frame=gt, detections_by_frame=detections,
    )
    changed = dict(detections)
    changed[5] = [candidate(5, [70.0, 70.0, 80.0, 90.0])]
    future_changed = build_sequence_motion_samples(
        sequence_index=0, sequence_length=5, image_width=100, image_height=100,
        fps=20.0, gt_by_frame=gt, detections_by_frame=changed,
    )
    target4 = np.where(baseline["target_frame"] == 4)[0][0]
    changed_target4 = np.where(future_changed["target_frame"] == 4)[0][0]
    assert np.array_equal(
        baseline["history_states"][target4],
        future_changed["history_states"][changed_target4],
    )
    assert baseline["history_frames"][target4, -1] == 3
    assert np.all(baseline["history_frames"][baseline["history_valid"]] < np.repeat(
        baseline["target_frame"], baseline["history_valid"].sum(axis=1)
    ))


def test_normalization_and_content_hash_are_deterministic() -> None:
    gt = {
        frame: {1: np.asarray([frame, 0.0, frame + 10.0, 20.0])}
        for frame in range(1, 6)
    }
    detections = {
        frame: [candidate(frame, [frame, 0.0, frame + 10.0, 20.0])]
        for frame in range(1, 6)
    }
    first = build_sequence_motion_samples(
        sequence_index=0, sequence_length=5, image_width=100, image_height=100,
        fps=20.0, gt_by_frame=gt, detections_by_frame=detections,
    )
    second = build_sequence_motion_samples(
        sequence_index=0, sequence_length=5, image_width=100, image_height=100,
        fps=20.0, gt_by_frame=gt, detections_by_frame=detections,
    )
    assert array_content_sha256(first) == array_content_sha256(second)
    stats = fit_motion_normalization(first, 1e-6)
    assert stats["fit_sample_count"] == len(first["target_frame"])
    assert min(stats["target_delta_population_std"]) >= 1e-6

