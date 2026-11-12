from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mot_research.tracking.first_round_adapter import FirstRoundAssociationAdapter


def upstream_associate(detections, trackers, threshold, velocities, observations, weight):
    del threshold, velocities, observations, weight
    count = min(len(detections), len(trackers))
    matches = np.asarray([[index, index] for index in range(count)], dtype=int).reshape(-1, 2)
    return matches, np.arange(count, len(detections)), np.arange(count, len(trackers))


class FakeTracker:
    def __init__(self, module) -> None:
        self.module = module
        self.frame_count = 0
        self.trackers = [SimpleNamespace(id=4)]

    def update(self, output_results, img_info, img_size):
        del img_info, img_size
        self.frame_count += 1
        detections = np.asarray(output_results, dtype=float)
        trackers = np.asarray([[0, 0, 1, 1, 0]], dtype=float)
        velocities = np.zeros((1, 2), dtype=float)
        observations = np.asarray([[0, 0, 1, 1, 1]], dtype=float)
        return self.module.associate(
            detections, trackers, 0.3, velocities, observations, 0.2
        )[0]


def test_upstream_policy_is_exact_and_restores_symbol() -> None:
    module = SimpleNamespace(associate=upstream_associate)
    tracker = FakeTracker(module)
    adapter = FirstRoundAssociationAdapter(tracker, module)
    detections = np.asarray([[0, 0, 1, 1, 0.9]], dtype=float)
    result = adapter.update(detections, (10, 10), (10, 10))
    assert np.array_equal(result, np.asarray([[0, 0]]))
    assert adapter.association_calls == 1
    assert module.associate is upstream_associate


def test_policy_inputs_are_read_only_and_symbol_restores_after_failure() -> None:
    module = SimpleNamespace(associate=upstream_associate)
    tracker = FakeTracker(module)

    def bad_policy(request, original):
        del original
        request.detections[0, 0] = 99
        raise AssertionError("unreachable")

    adapter = FirstRoundAssociationAdapter(tracker, module, bad_policy)
    with pytest.raises(ValueError, match="read-only"):
        adapter.update(np.asarray([[0, 0, 1, 1, 0.9]]), (10, 10), (10, 10))
    assert module.associate is upstream_associate

