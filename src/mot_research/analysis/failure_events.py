"""Train-only bounded detection-gap recovery diagnostics.

The labels produced here are diagnostic proxies. They do not identify true
occlusion causes and they are not official TrackEval event annotations.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    box_xyxy: np.ndarray
    score: float = 1.0


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    length: int
    width: int
    height: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(x: float, y: float, width: float, height: float) -> np.ndarray:
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def pairwise_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return continuous-coordinate IoU for arrays shaped [N,4] and [M,4]."""
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_wh = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    first_area = np.maximum(0.0, first[:, 2] - first[:, 0]) * np.maximum(
        0.0, first[:, 3] - first[:, 1]
    )
    second_area = np.maximum(0.0, second[:, 2] - second[:, 0]) * np.maximum(
        0.0, second[:, 3] - second[:, 1]
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def hungarian_iou_matches(
    gt: dict[int, np.ndarray], candidates: list[Candidate], threshold: float
) -> tuple[dict[int, tuple[int, float]], dict[int, float]]:
    """Match candidates one-to-one and also return each GT's best raw IoU."""
    if not gt:
        return {}, {}
    gt_ids = sorted(gt)
    if not candidates:
        return {}, {gt_id: 0.0 for gt_id in gt_ids}
    gt_boxes = np.stack([gt[gt_id] for gt_id in gt_ids])
    candidate_boxes = np.stack([item.box_xyxy for item in candidates])
    similarities = pairwise_iou(gt_boxes, candidate_boxes)
    best = {
        gt_id: float(similarities[index].max(initial=0.0))
        for index, gt_id in enumerate(gt_ids)
    }
    eligible = np.where(similarities >= threshold, similarities, 0.0)
    row_indices, column_indices = linear_sum_assignment(eligible, maximize=True)
    matches: dict[int, tuple[int, float]] = {}
    for row, column in zip(row_indices, column_indices):
        similarity = float(similarities[row, column])
        if similarity >= threshold:
            matches[gt_ids[row]] = (candidates[column].candidate_id, similarity)
    return matches, best


def read_sequence_info(sequence_dir: Path) -> SequenceInfo:
    parser = configparser.ConfigParser()
    path = sequence_dir / "seqinfo.ini"
    if not parser.read(path):
        raise FileNotFoundError(path)
    section = parser["Sequence"]
    return SequenceInfo(
        name=sequence_dir.name,
        length=section.getint("seqLength"),
        width=section.getint("imWidth"),
        height=section.getint("imHeight"),
    )


def read_gt(path: Path) -> dict[int, dict[int, np.ndarray]]:
    frames: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 6:
                raise ValueError(f"{path}:{line_number}: expected at least 6 columns")
            frame_id, gt_id = int(float(row[0])), int(float(row[1]))
            if gt_id in frames[frame_id]:
                raise ValueError(f"{path}:{line_number}: duplicate frame/GT ID")
            frames[frame_id][gt_id] = xywh_to_xyxy(*map(float, row[2:6]))
    return dict(frames)


def read_detections(path: Path, score_threshold: float) -> dict[int, list[Candidate]]:
    frames: dict[int, list[Candidate]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"frame", "det_id", "x1", "y1", "x2", "y2", "score"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path}: unexpected detection schema {reader.fieldnames}")
        for row in reader:
            score = float(row["score"])
            if score <= score_threshold:
                continue
            frames[int(row["frame"])].append(
                Candidate(
                    candidate_id=int(row["det_id"]),
                    box_xyxy=np.asarray(
                        [row["x1"], row["y1"], row["x2"], row["y2"]],
                        dtype=np.float64,
                    ),
                    score=score,
                )
            )
    return dict(frames)


def read_tracker(path: Path) -> dict[int, list[Candidate]]:
    frames: dict[int, list[Candidate]] = defaultdict(list)
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 columns")
            frames[int(row[0])].append(
                Candidate(
                    candidate_id=int(row[1]),
                    box_xyxy=xywh_to_xyxy(*map(float, row[2:6])),
                )
            )
    return dict(frames)


def _target_neighbor_iou(
    gt_by_frame: dict[int, dict[int, np.ndarray]], gt_id: int, frames: Iterable[int]
) -> float:
    maximum = 0.0
    for frame_id in frames:
        frame_gt = gt_by_frame[frame_id]
        if gt_id not in frame_gt:
            continue
        other = [box for other_id, box in frame_gt.items() if other_id != gt_id]
        if other:
            value = pairwise_iou(
                frame_gt[gt_id][None, :], np.stack(other)
            ).max(initial=0.0)
            maximum = max(maximum, float(value))
    return maximum


def _boundary_distance(
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    gt_id: int,
    frames: Iterable[int],
    width: int,
    height: int,
) -> float:
    values: list[float] = []
    for frame_id in frames:
        x1, y1, x2, y2 = gt_by_frame[frame_id][gt_id]
        values.append(min(x1 / width, y1 / height, (width - x2) / width, (height - y2) / height))
    return float(min(values))


def _max_motion_change(
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    gt_id: int,
    frames: list[int],
    width: int,
    height: int,
) -> float:
    if len(frames) < 3:
        return 0.0
    centers = []
    for frame_id in frames:
        box = gt_by_frame[frame_id][gt_id]
        centers.append((box[:2] + box[2:]) / 2.0)
    center_array = np.stack(centers)
    acceleration = center_array[2:] - 2.0 * center_array[1:-1] + center_array[:-2]
    return float(np.linalg.norm(acceleration, axis=1).max(initial=0.0) / math.hypot(width, height))


def extract_recovery_events(
    sequence_info: SequenceInfo,
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    detections_by_frame: dict[int, list[Candidate]],
    tracker_by_frame: dict[int, list[Candidate]],
    matching_iou: float = 0.5,
    max_gap: int = 30,
    history_window: int = 30,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    det_matches: dict[int, dict[int, tuple[int, float]]] = {}
    best_detection_iou: dict[int, dict[int, float]] = {}
    tracker_matches: dict[int, dict[int, tuple[int, float]]] = {}
    for frame_id, frame_gt in gt_by_frame.items():
        det_matches[frame_id], best_detection_iou[frame_id] = hungarian_iou_matches(
            frame_gt, detections_by_frame.get(frame_id, []), matching_iou
        )
        tracker_matches[frame_id], _ = hungarian_iou_matches(
            frame_gt, tracker_by_frame.get(frame_id, []), matching_iou
        )

    gt_frames: dict[int, list[int]] = defaultdict(list)
    for frame_id, frame_gt in gt_by_frame.items():
        for gt_id in frame_gt:
            gt_frames[gt_id].append(frame_id)

    consecutive_id_changes = 0
    for gt_id, frames in gt_frames.items():
        frames.sort()
        for before, after in zip(frames, frames[1:]):
            if after != before + 1:
                continue
            before_match = tracker_matches[before].get(gt_id)
            after_match = tracker_matches[after].get(gt_id)
            if before_match and after_match and before_match[0] != after_match[0]:
                consecutive_id_changes += 1

    events: list[dict[str, object]] = []
    for gt_id, frames in sorted(gt_frames.items()):
        frames.sort()
        for segment_start in range(len(frames)):
            if segment_start and frames[segment_start] == frames[segment_start - 1] + 1:
                continue
            segment_end = segment_start
            while segment_end + 1 < len(frames) and frames[segment_end + 1] == frames[segment_end] + 1:
                segment_end += 1
            segment = frames[segment_start : segment_end + 1]
            available = [gt_id in det_matches[frame_id] for frame_id in segment]
            index = 1
            while index < len(segment) - 1:
                if available[index] or not available[index - 1]:
                    index += 1
                    continue
                gap_start_index = index
                while index < len(segment) and not available[index]:
                    index += 1
                if index >= len(segment) or not available[index]:
                    continue
                gap_length = index - gap_start_index
                if gap_length > max_gap:
                    continue
                before_frame = segment[gap_start_index - 1]
                after_frame = segment[index]
                window_frames = list(range(before_frame, after_frame + 1))
                before_tracker = tracker_matches[before_frame].get(gt_id)
                after_tracker = tracker_matches[after_frame].get(gt_id)
                if before_tracker is None or after_tracker is None:
                    outcome = "unresolved"
                elif before_tracker[0] == after_tracker[0]:
                    outcome = "success_control"
                else:
                    outcome = "identity_change_failure"
                post_tracker_id = after_tracker[0] if after_tracker else None
                history_pollution = False
                if post_tracker_id is not None:
                    for history_frame in range(max(1, after_frame - history_window), after_frame):
                        for other_gt_id, match in tracker_matches.get(history_frame, {}).items():
                            if match[0] == post_tracker_id and other_gt_id != gt_id:
                                history_pollution = True
                                break
                        if history_pollution:
                            break
                gap_best_ious = [best_detection_iou[frame_id][gt_id] for frame_id in segment[gap_start_index:index]]
                neighbor_iou = _target_neighbor_iou(gt_by_frame, gt_id, window_frames)
                boundary_distance = _boundary_distance(
                    gt_by_frame,
                    gt_id,
                    window_frames,
                    sequence_info.width,
                    sequence_info.height,
                )
                event = {
                    "event_id": f"{sequence_info.name}:gt{gt_id}:f{before_frame}-{after_frame}",
                    "sequence": sequence_info.name,
                    "gt_id": gt_id,
                    "before_frame": before_frame,
                    "gap_start_frame": segment[gap_start_index],
                    "gap_end_frame": segment[index - 1],
                    "after_frame": after_frame,
                    "missing_length": gap_length,
                    "outcome": outcome,
                    "before_tracker_id": before_tracker[0] if before_tracker else None,
                    "after_tracker_id": post_tracker_id,
                    "before_tracker_iou": before_tracker[1] if before_tracker else None,
                    "after_tracker_iou": after_tracker[1] if after_tracker else None,
                    "max_raw_detection_iou_during_gap": max(gap_best_ious, default=0.0),
                    "localization_ambiguity_proxy": any(0.1 <= value < matching_iou for value in gap_best_ious),
                    "max_neighbor_gt_iou": neighbor_iou,
                    "neighbor_overlap_proxy": bool(neighbor_iou >= 0.1),
                    "min_normalized_boundary_distance": boundary_distance,
                    "boundary_proximity_proxy": bool(boundary_distance <= 0.05),
                    "max_normalized_center_second_difference": _max_motion_change(
                        gt_by_frame,
                        gt_id,
                        window_frames,
                        sequence_info.width,
                        sequence_info.height,
                    ),
                    "history_id_pollution_proxy": history_pollution,
                }
                events.append(event)
            if segment_end == len(frames) - 1:
                break
    counters = {
        "gt_identities": len(gt_frames),
        "diagnostic_consecutive_match_id_changes": consecutive_id_changes,
    }
    return sorted(events, key=lambda item: str(item["event_id"])), counters


def sample_events(
    events: list[dict[str, object]], seed: int, per_stratum: int
) -> list[dict[str, object]]:
    random_generator = random.Random(seed)
    selected: list[dict[str, object]] = []
    for outcome in ("identity_change_failure", "success_control"):
        candidates = sorted(
            (event for event in events if event["outcome"] == outcome),
            key=lambda item: str(item["event_id"]),
        )
        count = min(per_stratum, len(candidates))
        chosen = random_generator.sample(candidates, count)
        for event in chosen:
            copied = dict(event)
            copied["sample_stratum"] = outcome
            selected.append(copied)
    return sorted(selected, key=lambda item: str(item["event_id"]))


def review_frames(event: dict[str, object]) -> list[int]:
    """Select unique before/gap-boundary/after frames in temporal order."""
    return list(
        dict.fromkeys(
            int(event[key])
            for key in ("before_frame", "gap_start_frame", "gap_end_frame", "after_frame")
        )
    )


def review_crop_bounds(
    target_boxes: list[np.ndarray], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Return a context crop around all target boxes, clipped to the image."""
    boxes = np.stack(target_boxes)
    center = np.asarray(
        [
            (boxes[:, 0].min() + boxes[:, 2].max()) / 2,
            (boxes[:, 1].min() + boxes[:, 3].max()) / 2,
        ]
    )
    target_width = float(np.max(boxes[:, 2] - boxes[:, 0]))
    target_height = float(np.max(boxes[:, 3] - boxes[:, 1]))
    width = min(image_width, max(480, round(target_width * 4.0)))
    height = min(image_height, max(360, round(target_height * 3.0)))
    x1 = max(0, min(image_width - width, round(center[0] - width / 2)))
    y1 = max(0, min(image_height - height, round(center[1] - height / 2)))
    return x1, y1, x1 + width, y1 + height


def classify_event_candidate_contention(
    event: dict[str, object],
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    detections_by_frame: dict[int, list[Candidate]],
    matching_iou: float = 0.5,
) -> dict[str, object]:
    """Split parent gap frames into raw-candidate absence and contention.

    A contention frame has at least one raw eligible candidate at the target
    IoU threshold, while the target remains unmatched after the same global
    one-to-one assignment used to define the parent event.
    """
    gt_id = int(event["gt_id"])
    start = int(event["gap_start_frame"])
    end = int(event["gap_end_frame"])
    no_candidate_frames = 0
    contention_frames = 0
    raw_candidate_counts: list[int] = []
    best_ious: list[float] = []
    for frame_id in range(start, end + 1):
        frame_gt = gt_by_frame.get(frame_id, {})
        if gt_id not in frame_gt:
            raise ValueError(f"{event['event_id']}: target GT missing on frame {frame_id}")
        candidates = detections_by_frame.get(frame_id, [])
        matches, best = hungarian_iou_matches(frame_gt, candidates, matching_iou)
        if gt_id in matches:
            raise ValueError(
                f"{event['event_id']}: parent gap target unexpectedly matched on frame {frame_id}"
            )
        target = frame_gt[gt_id][None, :]
        if candidates:
            ious = pairwise_iou(target, np.stack([candidate.box_xyxy for candidate in candidates]))[0]
            raw_count = int(np.count_nonzero(ious >= matching_iou))
        else:
            raw_count = 0
        raw_candidate_counts.append(raw_count)
        best_ious.append(best[gt_id])
        if raw_count:
            contention_frames += 1
        else:
            no_candidate_frames += 1
    if contention_frames == 0:
        event_class = "absence_only"
    elif no_candidate_frames == 0:
        event_class = "contention_only"
    else:
        event_class = "mixed"
    return {
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "gt_id": gt_id,
        "outcome": event["outcome"],
        "missing_length": int(event["missing_length"]),
        "no_raw_candidate_frames": no_candidate_frames,
        "candidate_contention_frames": contention_frames,
        "event_candidate_class": event_class,
        "max_raw_candidates_per_gap_frame": max(raw_candidate_counts),
        "mean_best_raw_detection_iou": float(np.mean(best_ious)),
        "max_best_raw_detection_iou": max(best_ious),
    }
