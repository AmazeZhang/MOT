#!/usr/bin/env python3
"""Validate direct OC-SORT B0 against the research-wrapper B1 off state."""

from __future__ import annotations

import argparse
import configparser
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import yaml

from mot_research.analysis.d2_first_association import (
    compare_tracker_replay,
)
from mot_research.analysis.failure_events import sha256_file
from mot_research.data.motion_split import validate_motion_split
from mot_research.tracking.first_round_adapter import FirstRoundAssociationAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--reference-tracker-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument("--reference-atol", type=float, default=1e-4)
    return parser.parse_args()


def read_cache(path: Path) -> dict[int, list[list[float]]]:
    cache: dict[int, list[list[float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cache[int(row["frame"])].append([
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]), float(row["score"]),
            ])
    return cache


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


def run_tracker(
    ocsort_module: object,
    tracker_args: dict[str, object],
    cache: dict[int, list[list[float]]],
    sequence_length: int,
    image_height: int,
    image_width: int,
    input_height: int,
    input_width: int,
    wrapped: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    tracker = ocsort_module.OCSort(**tracker_args)
    runner = FirstRoundAssociationAdapter(tracker, ocsort_module) if wrapped else tracker
    scale = min(input_height / image_height, input_width / image_width)
    keys: list[list[int]] = []
    boxes: list[list[float]] = []
    for frame_id in range(1, sequence_length + 1):
        values = np.asarray(cache.get(frame_id, []), dtype=np.float64)
        tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
        if tracker_input is not None:
            tracker_input[:, :4] *= scale
        tracks = runner.update(
            tracker_input,
            (image_height, image_width),
            (input_height, input_width),
        )
        for x1, y1, x2, y2, track_id in tracks:
            keys.append([frame_id, int(track_id)])
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
    calls = runner.association_calls if wrapped else sequence_length
    return (
        np.asarray(keys, dtype=np.int64),
        np.asarray(boxes, dtype=np.float64),
        int(calls),
    )


def write_tracker(path: Path, keys: np.ndarray, boxes: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for (frame_id, track_id), (x1, y1, x2, y2) in zip(keys, boxes):
            writer.writerow([
                int(frame_id), int(track_id), f"{x1:.8f}", f"{y1:.8f}",
                f"{x2 - x1:.8f}", f"{y2 - y1:.8f}", 1, -1, -1, -1,
            ])


def assignment_runtime() -> dict[str, str]:
    try:
        import lap  # noqa: F401

        return {"backend": "lap.lapjv", "lap_version": importlib.metadata.version("lap")}
    except ImportError:
        return {
            "backend": "scipy.optimize.linear_sum_assignment",
            "scipy_version": scipy.__version__,
        }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("B0/B1 validation requires CUDA_VISIBLE_DEVICES to be the empty string")
    for path, expected in {
        args.split_manifest: args.split_manifest_sha256,
        args.protocol: args.protocol_sha256,
    }.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")

    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    failed = args.run_dir.with_name(args.run_dir.name + ".failed")
    if args.run_dir.exists() or partial.exists() or failed.exists():
        raise FileExistsError(f"refusing to overwrite output near {args.run_dir}")
    partial.mkdir(parents=True)
    try:
        split = yaml.safe_load(args.split_manifest.read_text())
        split_validation = validate_motion_split(split, args.train_root)
        sequences = list(split["motion_dev"]["sequences"])

        sys.path.insert(0, str(args.oc_sort_root.resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module

        original_associate = ocsort_module.associate
        tracker_args = {
            "det_thresh": 0.6,
            "iou_threshold": 0.3,
            "asso_func": "iou",
            "delta_t": 3,
            "inertia": 0.2,
            "use_byte": False,
        }
        comparisons: list[dict[str, object]] = []
        input_hashes: dict[str, object] = {"sequences": {}}
        b1_output_dir = partial / "b1_trackers"
        b1_output_dir.mkdir()

        for sequence in sequences:
            sequence_dir = args.train_root / sequence
            seqinfo_path = sequence_dir / "seqinfo.ini"
            gt_path = sequence_dir / "gt" / "gt.txt"
            cache_path = args.detection_root / f"{sequence}.csv"
            reference_path = args.reference_tracker_root / f"{sequence}.txt"
            config = configparser.ConfigParser()
            if not config.read(seqinfo_path):
                raise FileNotFoundError(seqinfo_path)
            info = config["Sequence"]
            sequence_length = info.getint("seqLength")
            image_height = info.getint("imHeight")
            image_width = info.getint("imWidth")
            cache = read_cache(cache_path)

            b0_keys, b0_boxes, _ = run_tracker(
                ocsort_module, tracker_args, cache, sequence_length, image_height, image_width,
                args.input_height, args.input_width, wrapped=False,
            )
            if ocsort_module.associate is not original_associate:
                raise RuntimeError(f"{sequence}: upstream symbol changed after B0")
            b1_keys, b1_boxes, association_calls = run_tracker(
                ocsort_module, tracker_args, cache, sequence_length, image_height, image_width,
                args.input_height, args.input_width, wrapped=True,
            )
            if ocsort_module.associate is not original_associate:
                raise RuntimeError(f"{sequence}: adapter did not restore upstream symbol")
            if association_calls != sequence_length:
                raise RuntimeError(
                    f"{sequence}: association calls {association_calls} != frames {sequence_length}"
                )
            exact_keys = np.array_equal(b0_keys, b1_keys)
            exact_boxes = np.array_equal(b0_boxes, b1_boxes)
            if not exact_keys or not exact_boxes:
                raise RuntimeError(f"{sequence}: B0/B1 in-memory outputs are not exact")

            reference_keys, reference_boxes = read_tracker_rows(reference_path)
            b0_reference = compare_tracker_replay(
                reference_keys, reference_boxes, b0_keys, b0_boxes, args.reference_atol
            )
            b1_reference = compare_tracker_replay(
                reference_keys, reference_boxes, b1_keys, b1_boxes, args.reference_atol
            )
            if b0_reference["status"] != "PASS" or b1_reference["status"] != "PASS":
                raise RuntimeError(f"{sequence}: registered G1 comparison failed")
            write_tracker(b1_output_dir / f"{sequence}.txt", b1_keys, b1_boxes)
            comparisons.append({
                "sequence": sequence,
                "frames": sequence_length,
                "tracker_rows": len(b1_keys),
                "association_calls": association_calls,
                "b0_b1_frame_track_id_exact": exact_keys,
                "b0_b1_xyxy_exact_float64": exact_boxes,
                "b0_reference": b0_reference,
                "b1_reference": b1_reference,
            })
            input_hashes["sequences"][sequence] = {
                "seqinfo": sha256_file(seqinfo_path),
                "gt": sha256_file(gt_path),
                "detections": sha256_file(cache_path),
                "registered_G1_tracker": sha256_file(reference_path),
                "B1_tracker": sha256_file(b1_output_dir / f"{sequence}.txt"),
            }

        summary = {
            "status": "PASS",
            "scope": "DanceTrack motion-dev B0/B1 off-state equivalence",
            "sequences": len(comparisons),
            "frames": sum(int(row["frames"]) for row in comparisons),
            "tracker_rows": sum(int(row["tracker_rows"]) for row in comparisons),
            "association_calls": sum(int(row["association_calls"]) for row in comparisons),
            "b0_b1_frame_track_id_exact_all": all(
                bool(row["b0_b1_frame_track_id_exact"]) for row in comparisons
            ),
            "b0_b1_xyxy_exact_float64_all": all(
                bool(row["b0_b1_xyxy_exact_float64"]) for row in comparisons
            ),
            "max_registered_G1_xyxy_error": max(
                float(row["b1_reference"]["max_abs_xyxy_error"]) for row in comparisons
            ),
            "registered_G1_atol": args.reference_atol,
            "split_validation": split_validation,
            "per_sequence": comparisons,
            "assignment": assignment_runtime(),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "split_manifest": str(args.split_manifest.resolve()),
            "split_manifest_sha256": args.split_manifest_sha256,
            "limitations": [
                "This validates only the B1 research-wrapper off state.",
                "No learned score, P1 model, state writeback change, or metric improvement is tested.",
                "Motion-dev is development evidence and includes four selected G2 anchors.",
            ],
        }
        with (partial / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

        repo_root = Path(__file__).resolve().parents[1]
        input_hashes.update({
            "split_manifest": sha256_file(args.split_manifest),
            "protocol": sha256_file(args.protocol),
            "analysis_script": sha256_file(Path(__file__)),
            "adapter": sha256_file(
                repo_root / "src/mot_research/tracking/first_round_adapter.py"
            ),
            "split_helper": sha256_file(repo_root / "src/mot_research/data/motion_split.py"),
            "comparison_helper": sha256_file(
                repo_root / "src/mot_research/analysis/d2_first_association.py"
            ),
            "upstream_ocsort": sha256_file(args.oc_sort_root / "trackers/ocsort_tracker/ocsort.py"),
            "upstream_association": sha256_file(
                args.oc_sort_root / "trackers/ocsort_tracker/association.py"
            ),
        })
        with (partial / "input_hashes.json").open("w") as handle:
            json.dump(input_hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")

        upstream_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.oc_sort_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        runtime = {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "assignment": assignment_runtime(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_policy": "CPU-only B0/B1 validation; physical GPU 0 forbidden",
            "upstream_oc_sort_commit": upstream_commit,
            "cwd": str(Path.cwd()),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (partial / "runtime.json").open("w") as handle:
            json.dump(runtime, handle, indent=2)
            handle.write("\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        if partial.exists():
            failure = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            with (partial / "failure.json").open("w") as handle:
                json.dump(failure, handle, indent=2)
                handle.write("\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
