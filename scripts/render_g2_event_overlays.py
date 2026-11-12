#!/usr/bin/env python3
"""Render compact review sheets for sampled G2 train failure events."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

from mot_research.analysis.failure_events import (
    Candidate,
    read_detections,
    read_gt,
    read_tracker,
    review_crop_bounds,
    review_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampled-events", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    return parser.parse_args()


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    crop: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
    label: str | None = None,
) -> None:
    x1, y1, _, _ = crop
    first = (round(float(box[0]) - x1), round(float(box[1]) - y1))
    second = (round(float(box[2]) - x1), round(float(box[3]) - y1))
    cv2.rectangle(image, first, second, color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (first[0], max(18, first[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def render_panel(
    image_path: Path,
    frame_id: int,
    gt: dict[int, np.ndarray],
    detections: list[Candidate],
    trackers: list[Candidate],
    target_gt_id: int,
    endpoint_tracker_ids: set[int],
    crop: tuple[int, int, int, int],
) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read {image_path}")
    x1, y1, x2, y2 = crop
    panel = image[y1:y2, x1:x2].copy()
    for gt_id, box in gt.items():
        target = gt_id == target_gt_id
        draw_box(panel, box, crop, (0, 255, 0) if target else (0, 100, 0), 4 if target else 1, f"GT {gt_id}")
    for detection in detections:
        draw_box(panel, detection.box_xyxy, crop, (0, 220, 220), 1)
    for tracker in trackers:
        endpoint = tracker.candidate_id in endpoint_tracker_ids
        draw_box(
            panel,
            tracker.box_xyxy,
            crop,
            (0, 0, 255) if endpoint else (100, 100, 255),
            3 if endpoint else 1,
            f"T {tracker.candidate_id}",
        )
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, f"frame {frame_id}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    scale = 480 / panel.shape[0]
    return cv2.resize(panel, (round(panel.shape[1] * scale), 480), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    events = list(csv.DictReader(args.sampled_events.open(newline="")))
    if not events:
        raise RuntimeError("sampled event list is empty")
    args.output_dir.mkdir(parents=True)
    cache: dict[str, tuple[dict, dict, dict]] = {}
    index_rows: list[dict[str, object]] = []
    for event in events:
        sequence = event["sequence"]
        if sequence not in cache:
            cache[sequence] = (
                read_gt(args.train_root / sequence / "gt" / "gt.txt"),
                read_detections(args.detection_root / f"{sequence}.csv", args.score_threshold),
                read_tracker(args.tracker_root / f"{sequence}.txt"),
            )
        gt_by_frame, detection_by_frame, tracker_by_frame = cache[sequence]
        frames = review_frames(event)
        target_gt_id = int(event["gt_id"])
        target_boxes = [gt_by_frame[frame_id][target_gt_id] for frame_id in frames]
        probe = cv2.imread(str(args.train_root / sequence / "img1" / f"{frames[0]:08d}.jpg"))
        if probe is None:
            raise RuntimeError(f"failed to read first event frame for {event['event_id']}")
        crop = review_crop_bounds(target_boxes, probe.shape[1], probe.shape[0])
        endpoint_ids = {
            int(event[key])
            for key in ("before_tracker_id", "after_tracker_id")
            if event[key]
        }
        panels = [
            render_panel(
                args.train_root / sequence / "img1" / f"{frame_id:08d}.jpg",
                frame_id,
                gt_by_frame[frame_id],
                detection_by_frame.get(frame_id, []),
                tracker_by_frame.get(frame_id, []),
                target_gt_id,
                endpoint_ids,
                crop,
            )
            for frame_id in frames
        ]
        sheet = cv2.hconcat(panels)
        header = np.zeros((50, sheet.shape[1], 3), dtype=np.uint8)
        title = f"{event['event_id']} | {event['outcome']} | green=GT yellow=det>0.6 red=endpoint track"
        cv2.putText(header, title, (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        sheet = cv2.vconcat([header, sheet])
        filename = event["event_id"].replace(":", "__") + ".jpg"
        output_path = args.output_dir / filename
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"failed to write {output_path}")
        index_rows.append(
            {
                "event_id": event["event_id"],
                "outcome": event["outcome"],
                "frames": ";".join(map(str, frames)),
                "file": filename,
                "bytes": output_path.stat().st_size,
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
        )
        print(output_path)
    with (args.output_dir / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_rows)


if __name__ == "__main__":
    main()
