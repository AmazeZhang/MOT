"""Small Local/Context action heads for the frozen S06 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


CONTINUOUS_FEATURES = (
    "iou", "direction_score", "detection_score", "ema_detection_cosine",
    "observation_gap", "valid_history_length", "row_best_margin",
    "column_best_margin",
)


@dataclass(frozen=True)
class FeatureNormalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def as_dict(self) -> dict:
        return {"features": list(CONTINUOUS_FEATURES), "mean": list(self.mean),
                "std": list(self.std)}

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureNormalization":
        if tuple(value["features"]) != CONTINUOUS_FEATURES:
            raise ValueError("feature normalization order changed")
        return cls(tuple(value["mean"]), tuple(value["std"]))


class ComponentActionSelector(nn.Module):
    """Parameter-matched heads; Context alone receives row/column neighbors."""

    def __init__(self, mode: str, normalization: FeatureNormalization, width: int = 64):
        super().__init__()
        if mode not in {"Local", "Context"}:
            raise ValueError("mode must be Local or Context")
        self.mode, self.width = mode, int(width)
        self.register_buffer("feature_mean", torch.tensor(normalization.mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(normalization.std, dtype=torch.float32))
        self.edge_encoder = nn.Sequential(nn.Linear(10, width), nn.ReLU(),
                                          nn.Linear(width, width), nn.ReLU())
        # Identical parameter count. Local feeds zero neighbor tensors.
        self.context_layer = nn.Sequential(nn.Linear(width * 3, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(width * 2 + 3, width), nn.ReLU(),
                                  nn.Linear(width, 2), nn.Sigmoid())

    def forward(self, continuous: torch.Tensor, binary: torch.Tensor,
                detection_rows: torch.Tensor, tracker_rows: torch.Tensor,
                selected: torch.Tensor, component_features: torch.Tensor) -> torch.Tensor:
        if continuous.ndim != 2 or continuous.shape[1] != 8 or binary.shape != (len(continuous), 2):
            raise ValueError("edge feature shape changed")
        normalized = (continuous - self.feature_mean) / self.feature_std
        encoded = self.edge_encoder(torch.cat([normalized, binary], dim=1))
        zeros = torch.zeros_like(encoded)
        row_context, column_context = zeros.clone(), zeros.clone()
        if self.mode == "Context":
            for i in range(len(encoded)):
                row_mask = (detection_rows == detection_rows[i])
                row_mask[i] = False
                col_mask = (tracker_rows == tracker_rows[i])
                col_mask[i] = False
                if row_mask.any():
                    row_context[i] = encoded[row_mask].mean(dim=0)
                if col_mask.any():
                    column_context[i] = encoded[col_mask].mean(dim=0)
        contextual = self.context_layer(torch.cat([encoded, row_context, column_context], dim=1))
        if selected.any():
            chosen = contextual[selected]
            pooled = torch.cat([chosen.mean(dim=0), chosen.max(dim=0).values])
        else:
            pooled = torch.zeros(self.width * 2, dtype=contextual.dtype,
                                 device=contextual.device)
        return self.head(torch.cat([pooled, component_features]))


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)
