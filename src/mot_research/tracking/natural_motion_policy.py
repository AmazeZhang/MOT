"""Natural tracker-history policies for first-round OC-SORT association.

The policy never reads GT. Histories are updated only after the complete
upstream tracker update returns, using detections that actually updated each
live tracker. Alternative boxes affect only the first-round association call;
the upstream Kalman state and lifecycle remain untouched.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from mot_research.data.motion_samples import box_xyxy_to_state, state_to_box_xyxy
from mot_research.tracking.first_round_adapter import (
    AssociationResult,
    FirstRoundAssociationAdapter,
    FirstRoundRequest,
    UpstreamAssociate,
)


class StatePredictor(Protocol):
    def predict_target_state(
        self,
        history_states: np.ndarray,
        history_frames: np.ndarray,
        history_valid: np.ndarray,
        target_frames: np.ndarray,
        fps: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class DetectionObservation:
    frame_id: int
    box_xyxy: np.ndarray
    score: float


class TrackerDetectionHistory:
    """Keep bounded, post-association detector observations by stable track ID."""

    def __init__(self, *, history_length: int = 8, score_threshold: float = 0.6) -> None:
        if history_length < 2:
            raise ValueError("history_length must be at least two")
        self.history_length = history_length
        self.score_threshold = score_threshold
        self._histories: dict[int, list[DetectionObservation]] = {}
        self.counters: Counter[str] = Counter()

    def observations(self, track_id: int) -> tuple[DetectionObservation, ...]:
        return tuple(self._histories.get(int(track_id), ()))

    def record_after_update(
        self,
        frame_id: int,
        trackers: list[Any],
        birth_observations: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Append only current-frame real detections after full tracker update."""
        births = birth_observations or {}
        live_ids = {int(tracker.id) + 1 for tracker in trackers}
        removed = set(self._histories) - live_ids
        for track_id in removed:
            del self._histories[track_id]
            self.counters["purged_deleted_track"] += 1

        for tracker in trackers:
            track_id = int(tracker.id) + 1
            if int(tracker.time_since_update) != 0:
                self.counters["live_not_updated"] += 1
                continue
            value = np.asarray(tracker.last_observation, dtype=np.float64).reshape(-1)
            if track_id in births:
                placeholder = np.full(5, -1.0, dtype=np.float64)
                if value.size >= 5 and not np.array_equal(value[:5], placeholder):
                    raise RuntimeError(
                        f"track {track_id}: birth mapping supplied for a regular observation"
                    )
                value = np.asarray(births[track_id], dtype=np.float64).reshape(-1)
                self.counters["recorded_birth_observation"] += 1
            if value.size < 5:
                raise ValueError(f"track {track_id}: last observation lacks score")
            box = value[:4]
            score = float(value[4])
            if not np.isfinite(value[:5]).all():
                self.counters["rejected_nonfinite_observation"] += 1
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                self.counters["rejected_invalid_observation_box"] += 1
                continue
            if score <= self.score_threshold:
                self.counters["rejected_below_or_equal_score_threshold"] += 1
                continue
            history = self._histories.setdefault(track_id, [])
            if history and history[-1].frame_id >= frame_id:
                raise RuntimeError(
                    f"track {track_id}: non-increasing observation frame "
                    f"{history[-1].frame_id} -> {frame_id}"
                )
            history.append(
                DetectionObservation(frame_id, box.copy(), score)
            )
            if len(history) > self.history_length:
                del history[:-self.history_length]
            self.counters["recorded_observation"] += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "live_history_count": len(self._histories),
            "stored_observation_count": sum(len(value) for value in self._histories.values()),
            "counters": dict(sorted(self.counters.items())),
        }


class TorchMotionPredictor:
    """NumPy-facing inference wrapper around a frozen deterministic checkpoint."""

    def __init__(self, checkpoint: dict[str, Any], device: str) -> None:
        import torch

        from mot_research.models.deterministic_motion import (
            DeterministicMotionGRU,
            MotionNormalization,
        )

        normalization = MotionNormalization.from_mapping(checkpoint["normalization"])
        hidden_size = int(checkpoint["model"]["hidden_size"])
        self.model = DeterministicMotionGRU(
            normalization, hidden_size=hidden_size
        ).to(torch.device(device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.torch = torch

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict_target_state(
        self,
        history_states: np.ndarray,
        history_frames: np.ndarray,
        history_valid: np.ndarray,
        target_frames: np.ndarray,
        fps: np.ndarray,
    ) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            states = torch.from_numpy(history_states).to(self.device, dtype=torch.float32)
            frames = torch.from_numpy(history_frames).to(self.device)
            valid = torch.from_numpy(history_valid).to(self.device)
            targets = torch.from_numpy(target_frames).to(self.device)
            frame_rates = torch.from_numpy(fps).to(self.device, dtype=torch.float32)
            result = self.model.predict_target_state(
                states, frames, valid, targets, frame_rates
            )
        return result.detach().cpu().numpy().astype(np.float64)


class NaturalMotionFirstRoundPolicy:
    """Replace eligible first-round tracker boxes using natural histories."""

    MODES = {"upstream", "zero_delta", "constant_velocity", "learned"}

    def __init__(
        self,
        *,
        mode: str,
        history: TrackerDetectionHistory,
        image_width: int,
        image_height: int,
        fps: float,
        predictor: StatePredictor | None = None,
        minimum_observations: int = 2,
        maximum_frame_gap: int = 31,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown policy mode: {mode}")
        if mode == "learned" and predictor is None:
            raise ValueError("learned mode requires a predictor")
        if mode != "learned" and predictor is not None:
            raise ValueError("predictor is only valid in learned mode")
        if image_width <= 0 or image_height <= 0 or fps <= 0:
            raise ValueError("image dimensions and fps must be positive")
        self.mode = mode
        self.history = history
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fps = float(fps)
        self.predictor = predictor
        self.minimum_observations = int(minimum_observations)
        self.maximum_frame_gap = int(maximum_frame_gap)
        self.counters: Counter[str] = Counter()
        self.call_records: list[dict[str, object]] = []

    def _eligible_arrays(
        self, request: FirstRoundRequest
    ) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        eligible_rows: list[int] = []
        states: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        length = self.history.history_length
        for row, track_id in enumerate(request.tracker_ids):
            observations = self.history.observations(track_id)
            if len(observations) == 0:
                self.counters["fallback_zero_history"] += 1
                continue
            if len(observations) < self.minimum_observations:
                self.counters["fallback_insufficient_history"] += 1
                continue
            last_frame = observations[-1].frame_id
            if last_frame >= request.frame_id:
                raise RuntimeError(
                    f"track {track_id}: current/future frame leaked into history"
                )
            frame_gap = request.frame_id - last_frame
            if frame_gap > self.maximum_frame_gap:
                self.counters["fallback_gap_above_maximum"] += 1
                continue
            selected = observations[-length:]
            history_states = np.zeros((length, 4), dtype=np.float64)
            history_frames = np.zeros(length, dtype=np.int64)
            history_valid = np.zeros(length, dtype=np.bool_)
            offset = length - len(selected)
            boxes = np.stack([item.box_xyxy for item in selected], axis=0)
            encoded = box_xyxy_to_state(
                boxes, self.image_width, self.image_height
            )
            history_states[offset:] = encoded
            history_frames[offset:] = [item.frame_id for item in selected]
            history_valid[offset:] = True
            eligible_rows.append(row)
            states.append(history_states)
            frames.append(history_frames)
            valid_masks.append(history_valid)
            self.counters[f"eligible_gap_{min(frame_gap, 31)}"] += 1
        if not eligible_rows:
            return (
                eligible_rows,
                np.empty((0, length, 4), dtype=np.float64),
                np.empty((0, length), dtype=np.int64),
                np.empty((0, length), dtype=np.bool_),
            )
        return (
            eligible_rows,
            np.stack(states),
            np.stack(frames),
            np.stack(valid_masks),
        )

    def _predict_states(
        self,
        request: FirstRoundRequest,
        states: np.ndarray,
        frames: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        if self.mode == "zero_delta":
            return states[:, -1].copy()
        if self.mode == "constant_velocity":
            previous = states[:, -2]
            last = states[:, -1]
            observation_gap = frames[:, -1] - frames[:, -2]
            target_gap = request.frame_id - frames[:, -1]
            if np.any(observation_gap <= 0) or np.any(target_gap <= 0):
                raise RuntimeError("invalid frame order in constant-velocity history")
            velocity = (last - previous) / observation_gap[:, None]
            return last + velocity * target_gap[:, None]
        if self.mode == "learned":
            assert self.predictor is not None
            return self.predictor.predict_target_state(
                states,
                frames,
                valid,
                np.full(len(states), request.frame_id, dtype=np.int64),
                np.full(len(states), self.fps, dtype=np.float64),
            )
        raise AssertionError(f"prediction requested for mode {self.mode}")

    def __call__(
        self, request: FirstRoundRequest, upstream_associate: UpstreamAssociate
    ) -> AssociationResult:
        call_start = time.perf_counter_ns()
        self.counters["association_calls"] += 1
        self.counters["tracker_rows_seen"] += len(request.tracker_ids)
        if self.mode == "upstream":
            result = upstream_associate(
                request.detections,
                request.trackers,
                request.iou_threshold,
                request.velocities,
                request.previous_observations,
                request.velocity_direction_weight,
            )
            self.call_records.append({
                "frame_id": request.frame_id,
                "tracker_rows": len(request.tracker_ids),
                "eligible_rows": 0,
                "replaced_rows": 0,
                "fallback_rows": len(request.tracker_ids),
                "policy_nanoseconds": time.perf_counter_ns() - call_start,
            })
            return result

        eligible_rows, states, frames, valid = self._eligible_arrays(request)
        adapted = np.asarray(request.trackers).copy()
        replaced = 0
        if eligible_rows:
            predicted_states = self._predict_states(request, states, frames, valid)
            predicted_boxes = state_to_box_xyxy(
                predicted_states, self.image_width, self.image_height
            )
            for prediction_index, tracker_row in enumerate(eligible_rows):
                box = predicted_boxes[prediction_index]
                if not np.isfinite(box).all():
                    self.counters["fallback_nonfinite_prediction"] += 1
                    continue
                if box[2] <= box[0] or box[3] <= box[1]:
                    self.counters["fallback_invalid_prediction_box"] += 1
                    continue
                adapted[tracker_row, :4] = box
                replaced += 1
        eligible_count = len(eligible_rows)
        fallback_count = len(request.tracker_ids) - replaced
        self.counters["eligible_tracker_rows"] += eligible_count
        self.counters["replaced_tracker_rows"] += replaced
        self.counters["fallback_tracker_rows"] += fallback_count
        result = upstream_associate(
            request.detections,
            adapted,
            request.iou_threshold,
            request.velocities,
            request.previous_observations,
            request.velocity_direction_weight,
        )
        self.call_records.append({
            "frame_id": request.frame_id,
            "tracker_rows": len(request.tracker_ids),
            "eligible_rows": eligible_count,
            "replaced_rows": replaced,
            "fallback_rows": fallback_count,
            "policy_nanoseconds": time.perf_counter_ns() - call_start,
        })
        return result

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "minimum_observations": self.minimum_observations,
            "maximum_frame_gap": self.maximum_frame_gap,
            "counters": dict(sorted(self.counters.items())),
        }


class NaturalHistoryTrackerRunner:
    """Run the adapter and append tracker observations only after the frame."""

    def __init__(
        self,
        tracker: Any,
        ocsort_module: Any,
        policy: NaturalMotionFirstRoundPolicy,
    ) -> None:
        if policy.history is None:
            raise ValueError("policy requires a history store")
        self.tracker = tracker
        self.policy = policy
        self.adapter = FirstRoundAssociationAdapter(tracker, ocsort_module, policy)
        self.update_nanoseconds: list[int] = []
        self.birth_mapping_counters: Counter[str] = Counter()
        self.source_frame_id = 0

    @property
    def association_calls(self) -> int:
        return self.adapter.association_calls

    def _high_detections_in_original_coordinates(
        self, output_results: Any, img_info: Any, img_size: Any
    ) -> np.ndarray:
        if output_results is None:
            return np.empty((0, 5), dtype=np.float64)
        values = np.asarray(output_results, dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 5), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 5:
            raise ValueError("natural-history runner requires NumPy Nx5 cached detections")
        result = values.copy()
        image_height, image_width = float(img_info[0]), float(img_info[1])
        scale = min(float(img_size[0]) / image_height, float(img_size[1]) / image_width)
        result[:, :4] /= scale
        return result[result[:, 4] > float(self.tracker.det_thresh)]

    def _map_birth_observations(
        self, before_ids: set[int], high_detections: np.ndarray
    ) -> dict[int, np.ndarray]:
        new_trackers = [
            tracker for tracker in self.tracker.trackers
            if int(tracker.id) + 1 not in before_ids
        ]
        if not new_trackers:
            return {}
        unused = set(range(len(high_detections)))
        result: dict[int, np.ndarray] = {}
        for tracker in sorted(new_trackers, key=lambda item: int(item.id)):
            track_id = int(tracker.id) + 1
            state_box = np.asarray(tracker.get_state()[0], dtype=np.float64)
            candidates = []
            for detection_index in sorted(unused):
                error = float(np.max(np.abs(
                    state_box - high_detections[detection_index, :4]
                )))
                if error <= 1e-4:
                    candidates.append((error, detection_index))
            if not candidates:
                self.birth_mapping_counters["unresolved_birth"] += 1
                continue
            candidates.sort()
            best_error = candidates[0][0]
            tied = [item for item in candidates if abs(item[0] - best_error) <= 1e-12]
            if len(tied) != 1:
                self.birth_mapping_counters["ambiguous_birth"] += 1
                continue
            detection_index = candidates[0][1]
            result[track_id] = high_detections[detection_index].copy()
            unused.remove(detection_index)
            self.birth_mapping_counters["mapped_birth"] += 1
        return result

    def update(
        self,
        output_results: Any,
        img_info: Any,
        img_size: Any,
        *,
        frame_id: int | None = None,
    ) -> np.ndarray:
        source_frame_id = self.source_frame_id + 1 if frame_id is None else int(frame_id)
        if source_frame_id <= self.source_frame_id:
            raise ValueError(
                f"source frame IDs must increase: {self.source_frame_id} -> {source_frame_id}"
            )
        start = time.perf_counter_ns()
        before_ids = {int(item.id) + 1 for item in self.tracker.trackers}
        high_detections = self._high_detections_in_original_coordinates(
            output_results, img_info, img_size
        )
        result = self.adapter.update(
            output_results, img_info, img_size, frame_id=source_frame_id
        )
        if output_results is not None:
            births = self._map_birth_observations(before_ids, high_detections)
            self.policy.history.record_after_update(
                source_frame_id, self.tracker.trackers, births
            )
        else:
            self.birth_mapping_counters["upstream_none_frame_no_history_append"] += 1
        self.source_frame_id = source_frame_id
        self.update_nanoseconds.append(time.perf_counter_ns() - start)
        return result

    def snapshot(self, warmup_frames: int = 20) -> dict[str, object]:
        values = np.asarray(self.update_nanoseconds[warmup_frames:], dtype=np.float64) / 1e6
        latency = {
            "warmup_frames": warmup_frames,
            "measured_frames": int(len(values)),
            "p50_ms": float(np.percentile(values, 50)) if len(values) else None,
            "p95_ms": float(np.percentile(values, 95)) if len(values) else None,
            "mean_ms": float(np.mean(values)) if len(values) else None,
        }
        return {
            "association_calls": self.association_calls,
            "last_source_frame_id": self.source_frame_id,
            "policy": self.policy.snapshot(),
            "history": self.policy.history.snapshot(),
            "birth_mapping_counters": dict(sorted(self.birth_mapping_counters.items())),
            "tracker_update_latency": latency,
        }
