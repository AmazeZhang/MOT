#!/usr/bin/env python3
"""Replay an original-image-coordinate detection cache through pinned OC-SORT."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument(
        "--empty-frame-policy",
        choices=["upstream_none", "advance_empty"],
        default="upstream_none",
        help="Match upstream None behavior or advance tracker time with an empty [0,5] array",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.oc_sort_root, args.sequence_dir, args.cache_file):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.output_file, args.summary_file):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("model input dimensions must be positive")

    config = configparser.ConfigParser()
    config.read(args.sequence_dir / "seqinfo.ini")
    sequence = config["Sequence"]
    sequence_length = sequence.getint("seqLength")
    image_height = sequence.getint("imHeight")
    image_width = sequence.getint("imWidth")
    scale = min(args.input_height / image_height, args.input_width / image_width)

    cache: dict[int, list[list[float]]] = defaultdict(list)
    seen_det_ids: set[tuple[int, int]] = set()
    with args.cache_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = ["frame", "det_id", "x1", "y1", "x2", "y2", "score", "class_id"]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"unexpected cache schema: {reader.fieldnames}")
        for row in reader:
            frame_id = int(row["frame"])
            det_id = int(row["det_id"])
            if not 1 <= frame_id <= sequence_length:
                raise ValueError(f"cache frame outside sequence: {frame_id}")
            key = (frame_id, det_id)
            if key in seen_det_ids:
                raise ValueError(f"duplicate frame/det_id: {key}")
            seen_det_ids.add(key)
            x1, y1, x2, y2, score = [float(row[name]) for name in ("x1", "y1", "x2", "y2", "score")]
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"non-positive detection box at {key}")
            if not 0 <= score <= 1:
                raise ValueError(f"invalid detection score at {key}: {score}")
            cache[frame_id].append([x1, y1, x2, y2, score])

    sys.path.insert(0, str(args.oc_sort_root.resolve()))
    from trackers.ocsort_tracker.ocsort import OCSort

    tracker_args = {
        "det_thresh": 0.6,
        "iou_threshold": 0.3,
        "asso_func": "iou",
        "delta_t": 3,
        "inertia": 0.2,
        "use_byte": False,
    }
    tracker = OCSort(**tracker_args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    tracker_rows = 0
    empty_detection_frames: list[int] = []
    with args.output_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for frame_id in range(1, sequence_length + 1):
            rows = np.asarray(cache.get(frame_id, []), dtype=float)
            if rows.size == 0:
                tracker_input = None if args.empty_frame_policy == "upstream_none" else np.empty((0, 5))
                empty_detection_frames.append(frame_id)
            else:
                tracker_input = rows.reshape(-1, 5).copy()
                tracker_input[:, :4] *= scale
            tracks = tracker.update(tracker_input, (image_height, image_width), (args.input_height, args.input_width))
            for x1, y1, x2, y2, track_id in tracks:
                writer.writerow(
                    [
                        frame_id,
                        int(track_id),
                        f"{x1:.6f}",
                        f"{y1:.6f}",
                        f"{x2 - x1:.6f}",
                        f"{y2 - y1:.6f}",
                        1,
                        -1,
                        -1,
                        -1,
                    ]
                )
                tracker_rows += 1

    summary = {
        "status": "PASS",
        "sequence": args.sequence_dir.name,
        "sequence_length": sequence_length,
        "cache_file": str(args.cache_file.resolve()),
        "cache_sha256": sha256(args.cache_file),
        "detections": len(seen_det_ids),
        "empty_detection_frames": empty_detection_frames,
        "empty_frame_policy": args.empty_frame_policy,
        "tracker": tracker_args,
        "tracker_rows": tracker_rows,
        "output_file": str(args.output_file.resolve()),
        "output_sha256": sha256(args.output_file),
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    args.summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
