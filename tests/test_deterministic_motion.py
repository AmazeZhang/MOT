from __future__ import annotations

import numpy as np
import torch

from mot_research.models.deterministic_motion import (
    DeterministicMotionGRU,
    MotionNormalization,
    deterministic_sample_indices,
)


def normalization() -> MotionNormalization:
    return MotionNormalization(
        history_mean=(0.0, 0.0, 0.0, 0.0),
        history_std=(1.0, 1.0, 1.0, 1.0),
        delta_mean=(0.1, -0.2, 0.3, -0.4),
        delta_std=(0.5, 0.5, 0.25, 0.25),
    )


def test_subset_selection_is_deterministic_and_key_sensitive() -> None:
    sequence = np.asarray([0, 0, 1, 1, 2], dtype=np.int32)
    gt_id = np.asarray([1, 2, 1, 2, 1], dtype=np.int32)
    frame = np.asarray([10, 10, 20, 20, 30], dtype=np.int32)
    first, first_hash = deterministic_sample_indices(
        sequence, gt_id, frame, count=3, seed=7
    )
    second, second_hash = deterministic_sample_indices(
        sequence, gt_id, frame, count=3, seed=7
    )
    assert np.array_equal(first, second)
    assert first_hash == second_hash
    changed, changed_hash = deterministic_sample_indices(
        sequence, gt_id, frame, count=3, seed=8
    )
    assert not (np.array_equal(first, changed) and first_hash == changed_hash)


def test_left_padding_values_do_not_affect_prediction() -> None:
    torch.manual_seed(3)
    model = DeterministicMotionGRU(normalization(), hidden_size=8).eval()
    states = torch.tensor([[[99.0] * 4, [1.0] * 4, [2.0] * 4]])
    frames = torch.tensor([[0, 3, 4]])
    valid = torch.tensor([[False, True, True]])
    target = torch.tensor([5])
    fps = torch.tensor([10.0])
    with torch.no_grad():
        first = model(states, frames, valid, target, fps)
        states[:, 0] = -1234.0
        frames[:, 0] = 999
        second = model(states, frames, valid, target, fps)
    assert torch.equal(first, second)


def test_delta_normalization_roundtrip_and_target_state() -> None:
    torch.manual_seed(4)
    model = DeterministicMotionGRU(normalization(), hidden_size=8).eval()
    delta = torch.tensor([[0.2, -0.1, 0.4, -0.3]])
    assert torch.allclose(model.denormalize_delta(model.normalize_delta(delta)), delta)
    states = torch.tensor([[[0.0] * 4, [1.0, 2.0, 3.0, 4.0]]])
    frames = torch.tensor([[0, 4]])
    valid = torch.tensor([[False, True]])
    with torch.no_grad():
        normalized_delta = model(states, frames, valid, torch.tensor([5]), torch.tensor([10.0]))
        target = model.predict_target_state(
            states, frames, valid, torch.tensor([5]), torch.tensor([10.0])
        )
    assert torch.allclose(target, states[:, -1] + model.denormalize_delta(normalized_delta))
