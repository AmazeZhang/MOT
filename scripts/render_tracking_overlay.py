#!/usr/bin/env python3
"""Render GT, fixed detections, and tracker output in original image coordinates."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--tracker-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.sequence_dir, args.cache_file, args.tracker_file):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.frames)
    ground_truth: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with (args.sequence_dir / "gt" / "gt.txt").open() as handle:
        for row in csv.reader(handle):
            frame_id = int(float(row[0]))
            if frame_id in selected:
                ground_truth[frame_id].append(
                    (int(float(row[1])), float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                )

    detections: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    with args.cache_file.open() as handle:
        for row in csv.DictReader(handle):
            frame_id = int(row["frame"])
            if frame_id in selected:
                detections[frame_id].append(
                    tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2", "score"))
                )

    tracks: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with args.tracker_file.open() as handle:
        for row in csv.reader(handle):
            frame_id = int(row[0])
            if frame_id in selected:
                tracks[frame_id].append(
                    (int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))
                )

    for frame_id in args.frames:
        image_path = args.sequence_dir / "img1" / f"{frame_id:08d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        output_path = args.output_dir / f"frame_{frame_id:08d}.jpg"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")

        for gt_id, x, y, width, height in ground_truth.get(frame_id, []):
            p1, p2 = (round(x), round(y)), (round(x + width), round(y + height))
            cv2.rectangle(image, p1, p2, (0, 200, 0), 2)
            cv2.putText(image, f"GT {gt_id}", (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        for x1, y1, x2, y2, score in detections.get(frame_id, []):
            cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 220), 1)
        for track_id, x, y, width, height in tracks.get(frame_id, []):
            p1, p2 = (round(x), round(y)), (round(x + width), round(y + height))
            cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
            cv2.putText(image, f"T {track_id}", (p1[0], min(image.shape[0] - 5, p2[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
        label = (
            f"frame={frame_id}  GT(green)={len(ground_truth.get(frame_id, []))}  "
            f"detections(yellow)={len(detections.get(frame_id, []))}  tracks(red)={len(tracks.get(frame_id, []))}"
        )
        cv2.putText(image, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"failed to write {output_path}")
        print(output_path)


if __name__ == "__main__":
    main()
