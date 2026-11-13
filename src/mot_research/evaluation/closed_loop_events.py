"""Common-denominator recovery events for natural closed-loop evaluation.

Ground truth and fixed detections are used only after tracker runs finish.  The
event population is defined once from GT, detections and the registered B0
tracker, never from a candidate method's own failures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from mot_research.analysis.failure_events import Candidate, hungarian_iou_matches


GAP_BINS = (
    (1, 1, "gap1"),
    (2, 3, "gap2_3"),
    (4, 7, "gap4_7"),
    (8, 15, "gap8_15"),
    (16, 31, "gap16_31"),
)


def gap_bin(length: int) -> str:
    for lower, upper, name in GAP_BINS:
        if lower <= length <= upper:
            return name
    raise ValueError(f"gap length outside registered range: {length}")


def _matches_by_frame(
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    candidates_by_frame: dict[int, list[Candidate]],
    matching_iou: float,
) -> dict[int, dict[int, tuple[int, float]]]:
    return {
        frame_id: hungarian_iou_matches(
            frame_gt, candidates_by_frame.get(frame_id, []), matching_iou
        )[0]
        for frame_id, frame_gt in gt_by_frame.items()
    }


def evaluate_event(
    event: dict[str, Any],
    tracker_matches: dict[int, dict[int, tuple[int, float]]],
    persistence_frames: int,
) -> dict[str, Any]:
    gt_id = int(event["gt_id"])
    before_frame = int(event["before_frame"])
    after_frame = int(event["after_frame"])
    before = tracker_matches.get(before_frame, {}).get(gt_id)
    after = tracker_matches.get(after_frame, {}).get(gt_id)
    before_id = None if before is None else int(before[0])
    after_id = None if after is None else int(after[0])
    reconnect_success = before_id is not None and after_id == before_id
    persistence_censored = bool(event["persistence_censored"])
    persistence_success: bool | None
    if persistence_censored:
        persistence_success = None
    elif not reconnect_success:
        persistence_success = False
    else:
        persistence_success = all(
            (
                tracker_matches.get(frame_id, {}).get(gt_id) is not None
                and int(tracker_matches[frame_id][gt_id][0]) == before_id
            )
            for frame_id in range(after_frame, after_frame + persistence_frames)
        )
    return {
        "event_id": event["event_id"],
        "before_tracker_id": before_id,
        "after_tracker_id": after_id,
        "precondition_match_present": before_id is not None,
        "reconnect_success": reconnect_success,
        "persistence_censored": persistence_censored,
        "persistence_success": persistence_success,
    }


def build_common_recovery_events(
    *,
    sequence: str,
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    detections_by_frame: dict[int, list[Candidate]],
    baseline_tracker_by_frame: dict[int, list[Candidate]],
    diagnostic_anchor: bool,
    matching_iou: float = 0.5,
    maximum_gap: int = 31,
    persistence_frames: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build a fixed event set from detection gaps and registered B0 only."""
    if not 0 < matching_iou <= 1:
        raise ValueError("matching_iou must be in (0, 1]")
    if maximum_gap < 1 or persistence_frames < 1:
        raise ValueError("gap and persistence lengths must be positive")
    detection_matches = _matches_by_frame(gt_by_frame, detections_by_frame, matching_iou)
    baseline_matches = _matches_by_frame(
        gt_by_frame, baseline_tracker_by_frame, matching_iou
    )
    gt_frames: dict[int, list[int]] = defaultdict(list)
    for frame_id, frame_gt in gt_by_frame.items():
        for gt_id in frame_gt:
            gt_frames[gt_id].append(frame_id)

    events: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for gt_id, all_frames in sorted(gt_frames.items()):
        all_frames.sort()
        segment_start = 0
        while segment_start < len(all_frames):
            segment_end = segment_start
            while (
                segment_end + 1 < len(all_frames)
                and all_frames[segment_end + 1] == all_frames[segment_end] + 1
            ):
                segment_end += 1
            segment = all_frames[segment_start:segment_end + 1]
            available = [gt_id in detection_matches[frame] for frame in segment]
            index = 1
            while index < len(segment) - 1:
                if available[index] or not available[index - 1]:
                    index += 1
                    continue
                gap_start_index = index
                while index < len(segment) and not available[index]:
                    index += 1
                if index >= len(segment):
                    break
                missing_length = index - gap_start_index
                if missing_length > maximum_gap:
                    counters["excluded_gap_above_maximum"] += 1
                    continue
                before_frame = segment[gap_start_index - 1]
                after_frame = segment[index]
                if gt_id not in baseline_matches.get(before_frame, {}):
                    counters["excluded_baseline_precondition_absent"] += 1
                    continue
                persistence_censored = any(
                    gt_id not in gt_by_frame.get(frame_id, {})
                    for frame_id in range(after_frame, after_frame + persistence_frames)
                )
                event: dict[str, Any] = {
                    "event_id": f"{sequence}:gt{gt_id}:f{before_frame}-{after_frame}",
                    "sequence": sequence,
                    "gt_id": gt_id,
                    "before_frame": before_frame,
                    "gap_start_frame": segment[gap_start_index],
                    "gap_end_frame": segment[index - 1],
                    "after_frame": after_frame,
                    "missing_length": missing_length,
                    "gap_bin": gap_bin(missing_length),
                    "diagnostic_anchor": bool(diagnostic_anchor),
                    "reappearance_detection_id": int(detection_matches[after_frame][gt_id][0]),
                    "persistence_censored": persistence_censored,
                }
                baseline = evaluate_event(event, baseline_matches, persistence_frames)
                event.update({
                    "baseline_before_tracker_id": baseline["before_tracker_id"],
                    "baseline_after_tracker_id": baseline["after_tracker_id"],
                    "baseline_reconnect_success": baseline["reconnect_success"],
                    "baseline_persistence_success": baseline["persistence_success"],
                    "baseline_stratum": (
                        "persistence_censored"
                        if persistence_censored
                        else "success_control"
                        if baseline["persistence_success"]
                        else "recovery_failure"
                    ),
                })
                events.append(event)
                counters["included_events"] += 1
                counters[f"included_{event['gap_bin']}"] += 1
                counters[f"included_{event['baseline_stratum']}"] += 1
            segment_start = segment_end + 1
    return sorted(events, key=lambda row: str(row["event_id"])), dict(sorted(counters.items()))


def evaluate_method_recovery_events(
    *,
    events: list[dict[str, Any]],
    gt_by_frame: dict[int, dict[int, np.ndarray]],
    tracker_by_frame: dict[int, list[Candidate]],
    matching_iou: float = 0.5,
    persistence_frames: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Evaluate one method on exactly the supplied common event IDs."""
    tracker_matches = _matches_by_frame(gt_by_frame, tracker_by_frame, matching_iou)
    outcomes = []
    counters: Counter[str] = Counter()
    for event in events:
        result = evaluate_event(event, tracker_matches, persistence_frames)
        row = {
            "event_id": event["event_id"],
            "sequence": event["sequence"],
            "gt_id": event["gt_id"],
            "gap_bin": event["gap_bin"],
            "diagnostic_anchor": event["diagnostic_anchor"],
            "baseline_stratum": event["baseline_stratum"],
            **result,
        }
        outcomes.append(row)
        counters["events"] += 1
        counters["precondition_match_present"] += int(result["precondition_match_present"])
        counters["reconnect_success"] += int(result["reconnect_success"])
        if result["persistence_censored"]:
            counters["persistence_censored"] += 1
        else:
            counters["persistence_evaluable"] += 1
            counters["persistence_success"] += int(bool(result["persistence_success"]))
            if event["baseline_stratum"] == "success_control":
                counters["success_controls"] += 1
                counters["success_control_harms"] += int(
                    not bool(result["persistence_success"])
                )
            elif event["baseline_stratum"] == "recovery_failure":
                counters["baseline_failures"] += 1
                counters["baseline_failure_recoveries"] += int(
                    bool(result["persistence_success"])
                )
    return outcomes, dict(sorted(counters.items()))
