"""Minimal deterministic GRU for the P1 motion feasibility gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class MotionNormalization:
    history_mean: tuple[float, float, float, float]
    history_std: tuple[float, float, float, float]
    delta_mean: tuple[float, float, float, float]
    delta_std: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "MotionNormalization":
        fields = {
            "history_mean": value["history_state_mean"],
            "history_std": value["history_state_population_std"],
            "delta_mean": value["target_delta_mean"],
            "delta_std": value["target_delta_population_std"],
        }
        converted: dict[str, tuple[float, float, float, float]] = {}
        for name, raw in fields.items():
            values = tuple(float(item) for item in raw)  # type: ignore[arg-type]
            if len(values) != 4 or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain four finite values")
            converted[name] = values  # type: ignore[assignment]
        if min(converted["history_std"] + converted["delta_std"]) <= 0:
            raise ValueError("normalization standard deviations must be positive")
        return cls(**converted)


def deterministic_sample_indices(
    sequence_index: np.ndarray,
    gt_id: np.ndarray,
    target_frame: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Select stable sample keys by SHA256 rank rather than array prefix."""
    keys = np.stack([sequence_index, gt_id, target_frame], axis=1).astype(np.int64)
    if count < 1 or count > len(keys):
        raise ValueError("sample count must be between one and the population size")
    ranks = []
    for index, key in enumerate(keys):
        text = f"{seed}:{int(key[0])}:{int(key[1])}:{int(key[2])}".encode("ascii")
        ranks.append((hashlib.sha256(text).digest(), index))
    selected = np.asarray([index for _, index in sorted(ranks)[:count]], dtype=np.int64)
    selected.sort()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(keys[selected]).tobytes())
    return selected, digest.hexdigest()


class DeterministicMotionGRU(nn.Module):
    """Predict a normalized four-dimensional delta from prior detector states."""

    def __init__(
        self,
        normalization: MotionNormalization,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(5, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 4),
        )
        self.register_buffer("history_mean", torch.tensor(normalization.history_mean))
        self.register_buffer("history_std", torch.tensor(normalization.history_std))
        self.register_buffer("delta_mean", torch.tensor(normalization.delta_mean))
        self.register_buffer("delta_std", torch.tensor(normalization.delta_std))

    def forward(
        self,
        history_states: torch.Tensor,
        history_frames: torch.Tensor,
        history_valid: torch.Tensor,
        target_frames: torch.Tensor,
        fps: torch.Tensor,
    ) -> torch.Tensor:
        if history_states.ndim != 3 or history_states.shape[-1] != 4:
            raise ValueError("history_states must have shape [batch, history, 4]")
        if history_valid.shape != history_states.shape[:2]:
            raise ValueError("history_valid shape differs from history_states")
        lengths = history_valid.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("each sample requires at least one valid observation")
        normalized = (history_states - self.history_mean) / self.history_std
        delta_seconds = (
            (target_frames[:, None] - history_frames).clamp_min(0).to(history_states.dtype)
            / fps[:, None]
        )
        time_feature = torch.log1p(delta_seconds).unsqueeze(-1)
        inputs = torch.cat([normalized, time_feature], dim=-1)

        # Construction stores histories left-padded. Compact valid rows to the left
        # before packing so padding cannot affect the recurrent state.
        compact = torch.zeros_like(inputs)
        ranks = history_valid.to(torch.long).cumsum(dim=1) - 1
        batch = torch.arange(inputs.shape[0], device=inputs.device)[:, None]
        expanded_batch = batch.expand_as(history_valid)
        compact[expanded_batch[history_valid], ranks[history_valid]] = inputs[history_valid]
        packed = pack_padded_sequence(
            compact,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self.delta_mean) / self.delta_std

    def denormalize_delta(self, normalized_delta: torch.Tensor) -> torch.Tensor:
        return normalized_delta * self.delta_std + self.delta_mean

    def predict_target_state(
        self,
        history_states: torch.Tensor,
        history_frames: torch.Tensor,
        history_valid: torch.Tensor,
        target_frames: torch.Tensor,
        fps: torch.Tensor,
    ) -> torch.Tensor:
        predicted_delta = self.denormalize_delta(
            self(history_states, history_frames, history_valid, target_frames, fps)
        )
        batch = torch.arange(history_states.shape[0], device=history_states.device)
        last_indices = history_valid.sum(dim=1) - 1 + (
            history_valid.shape[1] - history_valid.sum(dim=1)
        )
        return history_states[batch, last_indices] + predicted_delta
