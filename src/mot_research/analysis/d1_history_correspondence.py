"""Pure helpers for D1 privileged history-correspondence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from mot_research.analysis.d2_first_association import association_maps


def summarize_history_owner(
    observations: list[dict[str, int | None]],
    evaluation_frame: int,
    window_frames: int,
) -> dict[str, int | None]:
    """Resolve a track's latest mapped GT owner before an evaluation frame."""
    start = max(1, evaluation_frame - window_frames)
    eligible = [
        item
        for item in observations
        if start <= int(item["frame"]) < evaluation_frame
    ]
    mapped = [item for item in eligible if item.get("gt_id") is not None]
    latest = max(mapped, key=lambda item: int(item["frame"])) if mapped else None
    return {
        "owner_gt_id": None if latest is None else int(latest["gt_id"]),
        "owner_observation_frame": None if latest is None else int(latest["frame"]),
        "owner_age_frames": None
        if latest is None
        else evaluation_frame - int(latest["frame"]),
        "mapped_observations": len(mapped),
        "unmapped_observations": len(eligible) - len(mapped),
        "unique_mapped_gt_ids": len({int(item["gt_id"]) for item in mapped}),
    }


def evaluate_history_owner_relabel(
    intended_track_id: int,
    target_gt_id: int,
    positive_detection_id: int,
    tracker_ids: list[int],
    detection_ids: list[int],
    baseline_matches: np.ndarray,
    positive_track_owner: dict[str, int | None] | None,
) -> dict[str, Any]:
    """Evaluate semantic owner relabeling without changing association pairs."""
    track_to_detection, detection_to_track = association_maps(
        baseline_matches, tracker_ids, detection_ids
    )
    assigned_track_id = detection_to_track.get(positive_detection_id)
    baseline_positive = track_to_detection.get(intended_track_id) == positive_detection_id
    common: dict[str, Any] = {
        "baseline_target_detection_id": track_to_detection.get(intended_track_id),
        "baseline_positive_detection_track_id": assigned_track_id,
        "baseline_positive_match": baseline_positive,
    }
    if positive_detection_id not in detection_ids:
        return {
            **common,
            "status": "not_applicable",
            "non_applicable_reason": "positive_detection_not_in_first_round_pool",
        }
    if assigned_track_id is None:
        return {
            **common,
            "status": "not_applicable",
            "non_applicable_reason": "positive_detection_unassigned_in_first_round",
        }
    owner_gt_id = None if positive_track_owner is None else positive_track_owner.get("owner_gt_id")
    if owner_gt_id is None:
        return {
            **common,
            "status": "not_applicable",
            "non_applicable_reason": "assigned_track_history_owner_unmapped",
        }
    corrected = int(owner_gt_id) == target_gt_id
    return {
        **common,
        "status": "applicable",
        "non_applicable_reason": "",
        "D1_history_owner_positive_match": corrected,
        "recoverable": (not baseline_positive) and corrected,
        "harmed": baseline_positive and (not corrected),
    }


def summarize_gap_updates(
    observations: list[dict[str, int | None]],
    gap_start_frame: int,
    gap_end_frame: int,
    target_gt_id: int,
) -> dict[str, int]:
    """Count baseline target-track updates by privileged observation owner."""
    gap = [
        item
        for item in observations
        if gap_start_frame <= int(item["frame"]) <= gap_end_frame
    ]
    return {
        "baseline_gap_updates": len(gap),
        "baseline_gap_updates_target_owner": sum(
            item.get("gt_id") == target_gt_id for item in gap
        ),
        "baseline_gap_updates_other_owner": sum(
            item.get("gt_id") is not None and item.get("gt_id") != target_gt_id
            for item in gap
        ),
        "baseline_gap_updates_unmapped_owner": sum(
            item.get("gt_id") is None for item in gap
        ),
    }
