#!/usr/bin/env python3
"""Replay every fixed-detection cache and compare against direct OC-SORT outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--direct-tracker-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute XYXY tolerance in original-image pixels (default: 1e-4)",
    )
    return parser.parse_args()


def read_tracker(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame_and_id: list[list[int]] = []
    boxes: list[list[float]] = []
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 columns, got {len(row)}")
            frame_and_id.append([int(row[0]), int(row[1])])
            x, y, width, height = [float(value) for value in row[2:6]]
            boxes.append([x, y, x + width, y + height])
    return np.asarray(frame_and_id, dtype=np.int64), np.asarray(boxes, dtype=np.float64)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {path} or {partial}")
    with partial.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    if args.atol <= 0:
        raise ValueError("--atol must be positive")
    for path in (
        args.oc_sort_root,
        args.split_dir,
        args.cache_root,
        args.direct_tracker_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.run_dir}")

    sequence_dirs = sorted(path for path in args.split_dir.iterdir() if path.is_dir())
    if not sequence_dirs:
        raise RuntimeError(f"no sequences found under {args.split_dir}")
    replay_root = args.run_dir / "trackers" / "replay" / "data"
    sequence_summary_root = args.run_dir / "sequences"
    replay_root.mkdir(parents=True)
    sequence_summary_root.mkdir(parents=True)
    replay_script = Path(__file__).with_name("replay_ocsort_cache.py")
    started = time.time()
    comparisons: list[dict[str, object]] = []

    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        name = sequence_dir.name
        cache_file = args.cache_root / f"{name}.csv"
        direct_file = args.direct_tracker_root / f"{name}.txt"
        replay_file = replay_root / f"{name}.txt"
        replay_summary_file = sequence_summary_root / f"{name}.json"
        for path in (cache_file, direct_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        command = [
            sys.executable,
            str(replay_script),
            "--oc-sort-root",
            str(args.oc_sort_root),
            "--sequence-dir",
            str(sequence_dir),
            "--cache-file",
            str(cache_file),
            "--output-file",
            str(replay_file),
            "--summary-file",
            str(replay_summary_file),
            "--empty-frame-policy",
            "upstream_none",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            print(result.stdout, end="", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
            raise RuntimeError(f"cache replay failed for {name} with exit {result.returncode}")

        direct_keys, direct_boxes = read_tracker(direct_file)
        replay_keys, replay_boxes = read_tracker(replay_file)
        same_shape = direct_keys.shape == replay_keys.shape and direct_boxes.shape == replay_boxes.shape
        frame_and_id_equal = same_shape and np.array_equal(direct_keys, replay_keys)
        if same_shape and direct_boxes.size:
            max_abs_box_error = float(np.max(np.abs(direct_boxes - replay_boxes)))
        elif same_shape:
            max_abs_box_error = 0.0
        else:
            max_abs_box_error = None
        boxes_close = (
            same_shape
            and max_abs_box_error is not None
            and max_abs_box_error <= args.atol
        )
        comparison: dict[str, object] = {
            "sequence": name,
            "status": "PASS" if frame_and_id_equal and boxes_close else "FAIL",
            "rows": int(len(direct_keys)),
            "frame_and_id_equal": frame_and_id_equal,
            "max_abs_xyxy_error": max_abs_box_error,
            "atol": args.atol,
            "direct_output": str(direct_file.resolve()),
            "replay_output": str(replay_file.resolve()),
            "replay_summary": str(replay_summary_file.resolve()),
        }
        comparisons.append(comparison)
        print(
            f"[{index}/{len(sequence_dirs)}] {name}: {comparison['status']}, "
            f"rows={comparison['rows']}, max_error={max_abs_box_error}",
            flush=True,
        )
        if comparison["status"] != "PASS":
            raise RuntimeError(f"direct/replay comparison failed for {name}: {comparison}")

    summary: dict[str, object] = {
        "status": "PASS",
        "split_dir": str(args.split_dir.resolve()),
        "sequences": len(comparisons),
        "rows": sum(int(item["rows"]) for item in comparisons),
        "empty_frame_policy": "upstream_none",
        "frame_and_id_equal_for_all_sequences": all(
            bool(item["frame_and_id_equal"]) for item in comparisons
        ),
        "coordinate_atol": args.atol,
        "max_abs_xyxy_error": max(
            float(item["max_abs_xyxy_error"] or 0.0) for item in comparisons
        ),
        "wall_seconds": time.time() - started,
        "comparisons": comparisons,
    }
    write_json_atomic(args.run_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
