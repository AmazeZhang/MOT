"""A narrow, reversible adapter for OC-SORT's first association round."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


AssociationResult = tuple[np.ndarray, np.ndarray, np.ndarray]


class UpstreamAssociate(Protocol):
    def __call__(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
        iou_threshold: float,
        velocities: np.ndarray,
        previous_observations: np.ndarray,
        velocity_direction_weight: float,
    ) -> AssociationResult: ...


@dataclass(frozen=True)
class FirstRoundRequest:
    """Read-only association inputs exposed to a research policy."""

    frame_id: int
    call_index: int
    tracker_ids: tuple[int, ...]
    detections: np.ndarray
    trackers: np.ndarray
    iou_threshold: float
    velocities: np.ndarray
    previous_observations: np.ndarray
    velocity_direction_weight: float


AssociationPolicy = Callable[[FirstRoundRequest, UpstreamAssociate], AssociationResult]


def upstream_first_round_policy(
    request: FirstRoundRequest, upstream_associate: UpstreamAssociate
) -> AssociationResult:
    """B1-off policy: delegate the exact original inputs to upstream OC-SORT."""
    return upstream_associate(
        request.detections,
        request.trackers,
        request.iou_threshold,
        request.velocities,
        request.previous_observations,
        request.velocity_direction_weight,
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


class FirstRoundAssociationAdapter:
    """Patch only the module-level first-round call during one tracker update.

    The fixed upstream OC-SORT implementation imports ``associate`` into its
    tracker module. The adapter replaces that symbol only while ``update`` is
    executing, restores it in ``finally``, and serializes in-process access.
    Independent experiment jobs use separate processes.
    """

    _patch_lock = threading.Lock()

    def __init__(
        self,
        tracker: Any,
        ocsort_module: Any,
        policy: AssociationPolicy = upstream_first_round_policy,
    ) -> None:
        self.tracker = tracker
        self.ocsort_module = ocsort_module
        self.policy = policy
        self.upstream_associate: UpstreamAssociate = ocsort_module.associate
        self.association_calls = 0

    def update(self, output_results: Any, img_info: Any, img_size: Any) -> np.ndarray:
        """Run one upstream update with the selected first-round policy."""
        with self._patch_lock:
            if self.ocsort_module.associate is not self.upstream_associate:
                raise RuntimeError("OC-SORT associate symbol changed outside this adapter")

            def adapted_associate(
                detections: np.ndarray,
                trackers: np.ndarray,
                iou_threshold: float,
                velocities: np.ndarray,
                previous_observations: np.ndarray,
                velocity_direction_weight: float,
            ) -> AssociationResult:
                self.association_calls += 1
                request = FirstRoundRequest(
                    frame_id=int(self.tracker.frame_count),
                    call_index=self.association_calls,
                    tracker_ids=tuple(item.id + 1 for item in self.tracker.trackers),
                    detections=_readonly(detections),
                    trackers=_readonly(trackers),
                    iou_threshold=float(iou_threshold),
                    velocities=_readonly(velocities),
                    previous_observations=_readonly(previous_observations),
                    velocity_direction_weight=float(velocity_direction_weight),
                )
                result = self.policy(request, self.upstream_associate)
                if not isinstance(result, tuple) or len(result) != 3:
                    raise TypeError("association policy must return the upstream three-array tuple")
                if not all(isinstance(item, np.ndarray) for item in result):
                    raise TypeError("association policy outputs must be NumPy arrays")
                return result

            self.ocsort_module.associate = adapted_associate
            try:
                return self.tracker.update(output_results, img_info, img_size)
            finally:
                self.ocsort_module.associate = self.upstream_associate

