#!/usr/bin/env python3
"""Diagnose P1 predictors on identical B0 histories and association states.

All alternatives are evaluated only as first-round, one-step counterfactuals
at the fixed event reappearance frames.  B0 alone updates the tracker and the
natural detection history; no counterfactual result is written back.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from mot_research.analysis.d2_first_association import (
    association_maps,
    compare_tracker_replay,
)
from mot_research.analysis.failure_events import read_gt, sha256_file
from mot_research.analysis.p1_mechanism import (
    box_geometry_metrics,
    diagnose_learned_vs_reference,
    event_assignment_metrics,
    grouped_method_summaries,
    summarize_method_rows,
)
from mot_research.tracking.first_round_adapter import FirstRoundRequest
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner,
    NaturalMotionFirstRoundPolicy,
    TorchMotionPredictor,
    TrackerDetectionHistory,
)


METHODS = (
    "b0_upstream",
    "zero_delta",
    "constant_velocity",
    "learned_seed20261112",
    "learned_seed20261113",
    "learned_seed20261114",
)
LEARNED_METHODS = list(METHODS[3:])
OUTPUT_FIELDS = (
    "event_id", "sequence", "after_frame", "gt_id", "gap_bin",
    "diagnostic_anchor", "baseline_stratum", "method_id", "status",
    "non_applicable_reason", "positive_detection_id", "intended_track_id",
    "live_track_count", "first_round_detection_count", "history_observations",
    "history_last_frame", "history_target_gap", "history_eligible",
    "prediction_applied", "prediction_xyxy", "target_iou",
    "center_error_normalized", "log_width_abs_error",
    "log_height_abs_error", "intended_positive_iou",
    "intended_positive_total_score", "intended_gate_eligible",
    "eligible_other_track_count", "best_eligible_other_track_id",
    "best_eligible_other_score", "intended_minus_best_eligible_other",
    "positive_detection_assigned_track_id", "intended_assigned_detection_id",
    "immediate_positive_match", "association_pairs_changed_vs_b0",
    "closed_loop_persistence_censored", "closed_loop_persistence_success",
    "closed_loop_effect",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--baseline-tracker-root", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--events-sha256", required=True)
    parser.add_argument("--closed-loop-evaluation-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    for seed in (20261112, 20261113, 20261114):
        parser.add_argument(f"--checkpoint-{seed}", type=Path, required=True)
        parser.add_argument(f"--checkpoint-{seed}-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--maximum-frame-gap", type=int, default=31)
    parser.add_argument("--replay-atol", type=float, default=1e-4)
    return parser.parse_args()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def parse_optional_bool(value: str) -> bool | None:
    return None if value == "" else parse_bool(value)


def read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                **row,
                "gt_id": int(row["gt_id"]),
                "after_frame": int(row["after_frame"]),
                "positive_detection_id": int(row["reappearance_detection_id"]),
                "intended_track_id": int(row["baseline_before_tracker_id"]),
                "diagnostic_anchor": parse_bool(row["diagnostic_anchor"]),
            })
    if len(rows) != 831 or len({row["event_id"] for row in rows}) != 831:
        raise RuntimeError("expected 831 unique fixed events")
    return rows


def read_detection_cache(path: Path) -> tuple[dict[int, list[list[float]]], dict[int, list[int]]]:
    values: dict[int, list[list[float]]] = defaultdict(list)
    ids: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_id = int(row["frame"])
            detection_id = int(row["det_id"])
            key = (frame_id, detection_id)
            if key in seen:
                raise ValueError(f"duplicate detection key {key} in {path}")
            seen.add(key)
            score = float(row["score"])
            box = [float(row[name]) for name in ("x1", "y1", "x2", "y2")]
            values[frame_id].append([*box, score])
            if score > 0.6:
                ids[frame_id].append(detection_id)
    return dict(values), dict(ids)


def read_outcomes(root: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    all_outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for method in METHODS:
        path = root / method / "recovery" / "outcomes.csv"
        rows: dict[str, dict[str, Any]] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["event_id"]] = {
                    "persistence_censored": parse_bool(row["persistence_censored"]),
                    "persistence_success": parse_optional_bool(row["persistence_success"]),
                }
        if len(rows) != 831:
            raise RuntimeError(f"{method}: expected 831 closed-loop outcomes")
        all_outcomes[method] = rows
        hashes[method] = sha256_file(path)
    return all_outcomes, hashes


def association_scores(
    association_module: Any,
    detections: np.ndarray,
    trackers: np.ndarray,
    velocities: np.ndarray,
    previous_observations: np.ndarray,
    velocity_direction_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    iou = association_module.iou_batch(detections, trackers)
    if len(detections) == 0 or len(trackers) == 0:
        return iou, iou.copy()
    direction_y, direction_x = association_module.speed_direction_batch(
        detections, previous_observations
    )
    inertia_y = np.repeat(velocities[:, 0, None], direction_y.shape[1], axis=1)
    inertia_x = np.repeat(velocities[:, 1, None], direction_x.shape[1], axis=1)
    cosine = np.clip(inertia_x * direction_x + inertia_y * direction_y, -1, 1)
    angle = (np.pi / 2.0 - np.abs(np.arccos(cosine))) / np.pi
    valid = np.repeat(
        (previous_observations[:, 4] >= 0).astype(float)[:, None],
        direction_x.shape[1], axis=1,
    )
    scores = np.repeat(detections[:, -1, None], trackers.shape[0], axis=1)
    direction_cost = (valid * angle).T * velocity_direction_weight * scores
    return iou, iou + direction_cost


def write_tracker_row(writer: Any, frame_id: int, track: np.ndarray) -> None:
    x1, y1, x2, y2, track_id = track
    writer.writerow([
        frame_id, int(track_id), f"{x1:.8f}", f"{y1:.8f}",
        f"{x2 - x1:.8f}", f"{y2 - y1:.8f}", 1, -1, -1, -1,
    ])


def read_tracker_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    keys: list[list[int]] = []
    boxes: list[list[float]] = []
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 columns")
            keys.append([int(row[0]), int(row[1])])
            x, y, width, height = map(float, row[2:6])
            boxes.append([x, y, x + width, y + height])
    return np.asarray(keys, dtype=np.int64), np.asarray(boxes, dtype=np.float64)


def closed_loop_effect(stratum: str, success: bool | None) -> str:
    if success is None:
        return "censored"
    if stratum == "success_control":
        return "retained_control" if success else "harmed_control"
    if stratum == "recovery_failure":
        return "recovered_failure" if success else "unrecovered_failure"
    return "censored"


class CommonStateCapturePolicy:
    """Return B0 association while recording all one-step alternatives."""

    def __init__(
        self,
        *,
        history: TrackerDetectionHistory,
        alternatives: dict[str, NaturalMotionFirstRoundPolicy],
        frame_events: dict[int, list[dict[str, Any]]],
        detection_ids: dict[int, list[int]],
        gt: dict[int, dict[int, np.ndarray]],
        image_width: int,
        image_height: int,
        association_module: Any,
        outcomes: dict[str, dict[str, dict[str, Any]]],
        sink: list[dict[str, Any]],
    ) -> None:
        self.history = history
        self.alternatives = alternatives
        self.frame_events = frame_events
        self.detection_ids = detection_ids
        self.gt = gt
        self.image_width = image_width
        self.image_height = image_height
        self.association_module = association_module
        self.outcomes = outcomes
        self.sink = sink
        self.calls = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "b0_with_read_only_common_state_capture",
            "association_calls": self.calls,
            "alternative_snapshots": {
                method: policy.snapshot()
                for method, policy in sorted(self.alternatives.items())
            },
        }

    def __call__(self, request: FirstRoundRequest, upstream_associate: Callable[..., Any]):
        self.calls += 1
        baseline_result = upstream_associate(
            request.detections, request.trackers, request.iou_threshold,
            request.velocities, request.previous_observations,
            request.velocity_direction_weight,
        )
        events = self.frame_events.get(request.frame_id, [])
        if not events:
            return baseline_result

        method_payloads: dict[str, tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            "b0_upstream": (np.asarray(request.trackers).copy(), baseline_result)
        }
        for method, policy in self.alternatives.items():
            captured: list[np.ndarray] = []

            def capture(
                detections: np.ndarray,
                trackers: np.ndarray,
                iou_threshold: float,
                velocities: np.ndarray,
                previous_observations: np.ndarray,
                velocity_direction_weight: float,
            ):
                captured.append(np.asarray(trackers).copy())
                return upstream_associate(
                    detections, trackers, iou_threshold, velocities,
                    previous_observations, velocity_direction_weight,
                )

            result = policy(request, capture)
            if len(captured) != 1:
                raise RuntimeError(f"{method}: adapted tracker capture count != 1")
            method_payloads[method] = (captured[0], result)

        tracker_ids = list(request.tracker_ids)
        frame_detection_ids = self.detection_ids.get(request.frame_id, [])
        if len(frame_detection_ids) != len(request.detections):
            raise RuntimeError(
                f"frame {request.frame_id}: detection ID/value count mismatch"
            )
        baseline_pairs = set(association_maps(
            baseline_result[0], tracker_ids, frame_detection_ids
        )[0].items())
        for event in events:
            intended_track_id = int(event["intended_track_id"])
            positive_detection_id = int(event["positive_detection_id"])
            if positive_detection_id not in frame_detection_ids:
                raise RuntimeError(
                    f"{event['event_id']}: registered positive detection missing"
                )
            target_box = self.gt[request.frame_id][int(event["gt_id"])]
            observations = self.history.observations(intended_track_id)
            history_last_frame = observations[-1].frame_id if observations else None
            history_gap = (
                request.frame_id - history_last_frame
                if history_last_frame is not None else None
            )
            history_eligible = (
                len(observations) >= 2
                and history_gap is not None
                and 0 < history_gap <= 31
            )
            for method in METHODS:
                adapted, result = method_payloads[method]
                iou, total = association_scores(
                    self.association_module, request.detections, adapted,
                    request.velocities, request.previous_observations,
                    request.velocity_direction_weight,
                )
                assignment = event_assignment_metrics(
                    intended_track_id=intended_track_id,
                    positive_detection_id=positive_detection_id,
                    tracker_ids=tracker_ids,
                    detection_ids=frame_detection_ids,
                    matches=result[0],
                    iou_matrix=iou,
                    total_score_matrix=total,
                    iou_threshold=request.iou_threshold,
                )
                row: dict[str, Any] = {
                    "event_id": event["event_id"],
                    "sequence": event["sequence"],
                    "after_frame": request.frame_id,
                    "gt_id": event["gt_id"],
                    "gap_bin": event["gap_bin"],
                    "diagnostic_anchor": event["diagnostic_anchor"],
                    "baseline_stratum": event["baseline_stratum"],
                    "method_id": method,
                    "positive_detection_id": positive_detection_id,
                    "intended_track_id": intended_track_id,
                    "live_track_count": len(tracker_ids),
                    "first_round_detection_count": len(frame_detection_ids),
                    "history_observations": len(observations),
                    "history_last_frame": history_last_frame,
                    "history_target_gap": history_gap,
                    "history_eligible": history_eligible,
                    "prediction_applied": method != "b0_upstream" and history_eligible,
                    **assignment,
                }
                if assignment["status"] == "applicable":
                    tracker_index = tracker_ids.index(intended_track_id)
                    prediction = adapted[tracker_index, :4]
                    row["prediction_xyxy"] = ",".join(f"{value:.12g}" for value in prediction)
                    row.update(box_geometry_metrics(
                        prediction, target_box, self.image_width, self.image_height
                    ))
                    method_pairs = set(association_maps(
                        result[0], tracker_ids, frame_detection_ids
                    )[0].items())
                    row["association_pairs_changed_vs_b0"] = len(
                        baseline_pairs.symmetric_difference(method_pairs)
                    )
                else:
                    for field in (
                        "prediction_xyxy", "target_iou", "center_error_normalized",
                        "log_width_abs_error", "log_height_abs_error",
                        "intended_positive_iou", "intended_positive_total_score",
                        "intended_gate_eligible", "eligible_other_track_count",
                        "best_eligible_other_track_id", "best_eligible_other_score",
                        "intended_minus_best_eligible_other",
                        "positive_detection_assigned_track_id",
                        "intended_assigned_detection_id", "immediate_positive_match",
                        "association_pairs_changed_vs_b0",
                    ):
                        row.setdefault(field, None)
                outcome = self.outcomes[method][event["event_id"]]
                row["closed_loop_persistence_censored"] = outcome["persistence_censored"]
                row["closed_loop_persistence_success"] = outcome["persistence_success"]
                row["closed_loop_effect"] = closed_loop_effect(
                    str(event["baseline_stratum"]), outcome["persistence_success"]
                )
                self.sink.append(row)
        return baseline_result


def method_linkage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["closed_loop_effect"])].append(row)
    result: dict[str, Any] = {}
    for name, values in sorted(groups.items()):
        applicable = [row for row in values if row["status"] == "applicable"]
        result[name] = {
            "events": len(values),
            "applicable": len(applicable),
            "immediate_positive_match": sum(
                bool(row["immediate_positive_match"]) for row in applicable
            ),
            "intended_gate_eligible": sum(
                bool(row["intended_gate_eligible"]) for row in applicable
            ),
        }
    return result


def assignment_runtime() -> dict[str, str]:
    try:
        import lap  # noqa: F401
        return {"backend": "lap.lapjv", "lap_version": importlib.metadata.version("lap")}
    except ImportError:
        return {
            "backend": "scipy.optimize.linear_sum_assignment",
            "scipy_version": importlib.metadata.version("scipy"),
        }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("common-state diagnosis is CPU-only and requires CUDA_VISIBLE_DEVICES empty")
    if min(args.input_height, args.input_width, args.history_length) <= 0:
        raise ValueError("dimensions and history length must be positive")
    for path in (
        args.oc_sort_root, args.train_root, args.detection_root,
        args.baseline_tracker_root, args.closed_loop_evaluation_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    verify_hash(args.events, args.events_sha256)
    verify_hash(args.protocol, args.protocol_sha256)
    checkpoint_specs = {
        "learned_seed20261112": (
            args.checkpoint_20261112, args.checkpoint_20261112_sha256
        ),
        "learned_seed20261113": (
            args.checkpoint_20261113, args.checkpoint_20261113_sha256
        ),
        "learned_seed20261114": (
            args.checkpoint_20261114, args.checkpoint_20261114_sha256
        ),
    }
    for path, expected in checkpoint_specs.values():
        verify_hash(path, expected)

    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    failed = args.run_dir.with_name(args.run_dir.name + ".failed")
    if any(path.exists() for path in (args.run_dir, partial, failed)):
        raise FileExistsError(f"refusing to overwrite output near {args.run_dir}")
    partial.mkdir(parents=True)
    try:
        protocol = yaml.safe_load(args.protocol.read_text())
        events = read_events(args.events)
        event_ids = {row["event_id"] for row in events}
        by_sequence_frame: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in events:
            by_sequence_frame[str(event["sequence"])][int(event["after_frame"])].append(event)
        expected_sequences = list(protocol["scope"]["sequences"])
        if sorted(by_sequence_frame) != sorted(expected_sequences):
            raise RuntimeError("event sequences differ from frozen protocol")
        outcomes, outcome_hashes = read_outcomes(args.closed_loop_evaluation_root)
        for method in METHODS:
            if set(outcomes[method]) != event_ids:
                raise RuntimeError(f"{method}: closed-loop outcome event IDs differ")
            frozen_hash = protocol["fixed_inputs"]["closed_loop_outcomes_sha256"][method]
            if outcome_hashes[method] != frozen_hash:
                raise RuntimeError(f"{method}: outcome SHA differs from frozen protocol")

        import torch

        checkpoints: dict[str, dict[str, Any]] = {}
        predictors: dict[str, TorchMotionPredictor] = {}
        for method, (path, _) in checkpoint_specs.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoints[method] = checkpoint
            predictors[method] = TorchMotionPredictor(checkpoint, "cpu")

        sys.path.insert(0, str(args.oc_sort_root.resolve()))
        import trackers.ocsort_tracker.association as association_module
        import trackers.ocsort_tracker.ocsort as ocsort_module

        tracker_args = {
            "det_thresh": 0.6,
            "iou_threshold": 0.3,
            "asso_func": "iou",
            "delta_t": 3,
            "inertia": 0.2,
            "use_byte": False,
        }
        rows: list[dict[str, Any]] = []
        sequence_summaries: list[dict[str, Any]] = []
        input_hashes: dict[str, Any] = {"sequences": {}, "closed_loop_outcomes": outcome_hashes}
        replay_root = partial / "b0_replay"
        replay_root.mkdir()

        for sequence in expected_sequences:
            sequence_dir = args.train_root / sequence
            seqinfo_path = sequence_dir / "seqinfo.ini"
            config = configparser.ConfigParser()
            if not config.read(seqinfo_path):
                raise FileNotFoundError(seqinfo_path)
            info = config["Sequence"]
            sequence_length = info.getint("seqLength")
            image_width = info.getint("imWidth")
            image_height = info.getint("imHeight")
            fps = info.getfloat("frameRate")
            scale = min(
                args.input_height / image_height,
                args.input_width / image_width,
            )
            cache_path = args.detection_root / f"{sequence}.csv"
            baseline_path = args.baseline_tracker_root / f"{sequence}.txt"
            gt_path = sequence_dir / "gt" / "gt.txt"
            cached_values, detection_ids = read_detection_cache(cache_path)
            gt = read_gt(gt_path)
            history = TrackerDetectionHistory(
                history_length=args.history_length, score_threshold=0.6
            )
            alternatives = {
                "zero_delta": NaturalMotionFirstRoundPolicy(
                    mode="zero_delta", history=history, image_width=image_width,
                    image_height=image_height, fps=fps,
                    minimum_observations=args.minimum_observations,
                    maximum_frame_gap=args.maximum_frame_gap,
                ),
                "constant_velocity": NaturalMotionFirstRoundPolicy(
                    mode="constant_velocity", history=history, image_width=image_width,
                    image_height=image_height, fps=fps,
                    minimum_observations=args.minimum_observations,
                    maximum_frame_gap=args.maximum_frame_gap,
                ),
            }
            for method in LEARNED_METHODS:
                alternatives[method] = NaturalMotionFirstRoundPolicy(
                    mode="learned", history=history, image_width=image_width,
                    image_height=image_height, fps=fps, predictor=predictors[method],
                    minimum_observations=args.minimum_observations,
                    maximum_frame_gap=args.maximum_frame_gap,
                )
            policy = CommonStateCapturePolicy(
                history=history,
                alternatives=alternatives,
                frame_events=by_sequence_frame[sequence],
                detection_ids=detection_ids,
                gt=gt,
                image_width=image_width,
                image_height=image_height,
                association_module=association_module,
                outcomes=outcomes,
                sink=rows,
            )
            tracker = ocsort_module.OCSort(**tracker_args)
            runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
            replay_path = replay_root / f"{sequence}.txt"
            with replay_path.open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                for frame_id in range(1, sequence_length + 1):
                    values = np.asarray(cached_values.get(frame_id, []), dtype=np.float64)
                    tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                    if tracker_input is not None:
                        tracker_input[:, :4] *= scale
                    tracks = runner.update(
                        tracker_input,
                        (image_height, image_width),
                        (args.input_height, args.input_width),
                        frame_id=frame_id,
                    )
                    for track in tracks:
                        write_tracker_row(writer, frame_id, track)
            reference_keys, reference_boxes = read_tracker_rows(baseline_path)
            replay_keys, replay_boxes = read_tracker_rows(replay_path)
            replay = compare_tracker_replay(
                reference_keys, reference_boxes, replay_keys, replay_boxes,
                args.replay_atol,
            )
            if replay["status"] != "PASS":
                raise RuntimeError(f"{sequence}: B0 replay mismatch: {replay}")
            sequence_rows = [row for row in rows if row["sequence"] == sequence]
            expected_row_count = sum(
                len(frame_events)
                for frame_events in by_sequence_frame[sequence].values()
            ) * len(METHODS)
            if len(sequence_rows) != expected_row_count:
                raise RuntimeError(f"{sequence}: diagnostic row coverage mismatch")
            sequence_summaries.append({
                "sequence": sequence,
                "events": expected_row_count // len(METHODS),
                "rows": len(sequence_rows),
                "b0_replay": replay,
                "runner": runner.snapshot(),
            })
            input_hashes["sequences"][sequence] = {
                "seqinfo": sha256_file(seqinfo_path),
                "gt": sha256_file(gt_path),
                "detections": sha256_file(cache_path),
                "b0_tracker": sha256_file(baseline_path),
                "b0_replay": sha256_file(replay_path),
            }

        expected_keys = {(event_id, method) for event_id in event_ids for method in METHODS}
        actual_keys = {(str(row["event_id"]), str(row["method_id"])) for row in rows}
        if len(rows) != 4986 or actual_keys != expected_keys:
            raise RuntimeError("expected exactly 831 x 6 unique diagnostic rows")
        rows.sort(key=lambda row: (str(row["event_id"]), str(row["method_id"])))
        result_path = partial / "event_method_mechanism.csv"
        with result_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in OUTPUT_FIELDS} for row in rows)

        method_summaries = {
            method: summarize_method_rows([
                row for row in rows if row["method_id"] == method
            ])
            for method in METHODS
        }
        thresholds = protocol["descriptive_decision_rule"]
        diagnosis = diagnose_learned_vs_reference(
            method_summaries,
            learned_methods=LEARNED_METHODS,
            reference_method="constant_velocity",
            thresholds=thresholds,
        )
        summary = {
            "execution_status": "PASS",
            "scope": "831 fixed event reappearance frames; common B0 state; no writeback",
            "protocol_sha256": args.protocol_sha256,
            "events_sha256": args.events_sha256,
            "event_count": 831,
            "method_count": 6,
            "row_count": len(rows),
            "methods": method_summaries,
            "by_method_gap": {
                method: grouped_method_summaries(
                    [row for row in rows if row["method_id"] == method], "gap_bin"
                ) for method in METHODS
            },
            "by_method_baseline_stratum": {
                method: grouped_method_summaries(
                    [row for row in rows if row["method_id"] == method],
                    "baseline_stratum",
                ) for method in METHODS
            },
            "closed_loop_linkage": {
                method: method_linkage([
                    row for row in rows if row["method_id"] == method
                ]) for method in METHODS
            },
            "diagnosis_vs_constant_velocity": diagnosis,
            "sequence_summaries": sequence_summaries,
            "event_method_csv_sha256": sha256_file(result_path),
            "assignment_runtime": assignment_runtime(),
            "limitations": [
                "This is a one-step, common-B0-state counterfactual and is not a new closed-loop method result.",
                "GT is used only to score geometry and identify the registered positive detection after tracking inputs are fixed.",
                "Joined closed-loop outcomes came from method-specific histories, so linkage is descriptive rather than causal.",
                "The selected motion-dev split and 17 long-gap events do not support validation generalization.",
                "A triggered descriptive stage authorizes event-level review, not automatic retraining or a model claim.",
            ],
        }
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        input_hashes.update({
            "events": sha256_file(args.events),
            "protocol": sha256_file(args.protocol),
            "script": sha256_file(Path(__file__)),
            "helper": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/mot_research/analysis/p1_mechanism.py"
            ),
            "natural_policy": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/mot_research/tracking/natural_motion_policy.py"
            ),
            "checkpoints": {
                method: sha256_file(path)
                for method, (path, _) in checkpoint_specs.items()
            },
        })
        (partial / "input_hashes.json").write_text(
            json.dumps(input_hashes, indent=2, sort_keys=True) + "\n"
        )
        (partial / "runtime.json").write_text(json.dumps({
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": "cpu",
            "physical_gpu": None,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": importlib.metadata.version("torch"),
            "numpy_version": np.__version__,
        }, indent=2) + "\n")
        os.replace(partial, args.run_dir)
        print(json.dumps({
            "execution_status": "PASS",
            "row_count": len(rows),
            "diagnosis": diagnosis,
            "output": str(args.run_dir),
        }, indent=2))
    except Exception as error:
        if partial.exists():
            (partial / "failure.json").write_text(json.dumps({
                "execution_status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }, indent=2) + "\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
