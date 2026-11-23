"""S04 isolation risks that the earlier summary-only state gate missed."""

from types import SimpleNamespace

import numpy as np

from mot_research.association.appearance_core import CausalAppearanceMemory
from mot_research.tracking.natural_motion_policy import (
    DetectionObservation, TrackerDetectionHistory,
)
from run_s04_bounded_memory_freeze import freeze_future_writes, state_equal


def test_future_freeze_blocks_only_target_memory_write():
    memory = CausalAppearanceMemory(2, mode='ema', alpha=.9)
    original = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    memory.begin_frame(1, (1, 2))
    memory.finish_frame(1, (1, 2), {1: 10, 2: 11}, (10, 11), original)
    prior_target = (memory._memory[1][0], memory._memory[1][1].copy())
    removed = []
    freeze_future_writes(SimpleNamespace(memory=memory), 1, removed)
    memory.begin_frame(2, (1, 2))
    memory.finish_frame(2, (1, 2), {1: 10, 2: 11}, (10, 11), original)
    assert removed == [(2, 10)]
    assert memory._memory[1][0] == prior_target[0]
    assert np.array_equal(memory._memory[1][1], prior_target[1])
    assert memory._memory[2][0] == 2


def test_exact_state_check_sees_history_content_hidden_by_summary():
    left = TrackerDetectionHistory(history_length=8, score_threshold=.6)
    right = TrackerDetectionHistory(history_length=8, score_threshold=.6)
    left._histories[1] = [DetectionObservation(1, np.array([0., 0., 1., 1.]), .9)]
    right._histories[1] = [DetectionObservation(1, np.array([1., 0., 2., 1.]), .9)]
    assert left.snapshot() == right.snapshot()
    assert not state_equal(left, right)
