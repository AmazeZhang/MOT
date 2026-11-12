#!/usr/bin/env python3
"""Build frozen P1 motion-train/dev samples with train-only normalization."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import yaml

from mot_research.analysis.failure_events import (
    read_detections,
    read_gt,
    read_sequence_info,
    sha256_file,
)
from mot_research.data.motion_samples import (
    array_content_sha256,
    build_sequence_motion_samples,
    concatenate_motion_samples,
    fit_motion_normalization,
    state_to_box_xyxy,
)
from mot_research.data.motion_split import validate_motion_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--detection-manifest", type=Path, required=True)
    parser.add_argument("--detection-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def fps_from_seqinfo(path: Path) -> float:
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(path)
    fps = config["Sequence"].getfloat("frameRate")
    if fps <= 0:
        raise ValueError(f"invalid frame rate in {path}")
    return fps


def summarize_sequence(sequence: str, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    positive = arrays["positive_detection_id"] >= 0
    gaps = Counter(map(int, arrays["frame_gap"]))
    return {
        "sequence": sequence,
        "samples": int(len(positive)),
        "target_with_positive_detection": int(positive.sum()),
        "target_without_positive_detection": int((~positive).sum()),
        "frame_gap_min": int(arrays["frame_gap"].min()),
        "frame_gap_max": int(arrays["frame_gap"].max()),
        "frame_gap_histogram": {str(key): value for key, value in sorted(gaps.items())},
        "array_content_sha256": array_content_sha256(arrays),
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("sample construction requires CUDA_VISIBLE_DEVICES to be empty")
    for path, expected in {
        args.split_manifest: args.split_manifest_sha256,
        args.detection_manifest: args.detection_manifest_sha256,
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
        all_sequences = sorted(
            split["motion_train"]["sequences"] + split["motion_dev"]["sequences"]
        )
        sequence_index = {sequence: index for index, sequence in enumerate(all_sequences)}
        input_hashes: dict[str, object] = {"sequences": {}}
        group_arrays: dict[str, dict[str, np.ndarray]] = {}
        group_sequence_summaries: dict[str, list[dict[str, object]]] = {}

        for group_name in ("motion_train", "motion_dev"):
            parts: list[dict[str, np.ndarray]] = []
            summaries: list[dict[str, object]] = []
            for sequence in split[group_name]["sequences"]:
                sequence_dir = args.train_root / sequence
                seqinfo_path = sequence_dir / "seqinfo.ini"
                gt_path = sequence_dir / "gt" / "gt.txt"
                detection_path = args.detection_root / f"{sequence}.csv"
                info = read_sequence_info(sequence_dir)
                fps = fps_from_seqinfo(seqinfo_path)
                arrays = build_sequence_motion_samples(
                    sequence_index=sequence_index[sequence],
                    sequence_length=info.length,
                    image_width=info.width,
                    image_height=info.height,
                    fps=fps,
                    gt_by_frame=read_gt(gt_path),
                    detections_by_frame=read_detections(detection_path, 0.6),
                    history_length=8,
                    minimum_history_observations=2,
                    maximum_frame_gap=31,
                    matching_iou_threshold=0.5,
                )
                if len(arrays["target_frame"]) == 0:
                    raise RuntimeError(f"{sequence}: zero samples")
                parts.append(arrays)
                summaries.append(summarize_sequence(sequence, arrays))
                input_hashes["sequences"][sequence] = {
                    "group": group_name,
                    "seqinfo": sha256_file(seqinfo_path),
                    "gt": sha256_file(gt_path),
                    "detections": sha256_file(detection_path),
                }
            group_arrays[group_name] = concatenate_motion_samples(parts)
            group_sequence_summaries[group_name] = summaries

        train_keys = set(map(tuple, np.stack([
            group_arrays["motion_train"]["sequence_index"],
            group_arrays["motion_train"]["gt_id"],
            group_arrays["motion_train"]["target_frame"],
        ], axis=1)))
        dev_keys = set(map(tuple, np.stack([
            group_arrays["motion_dev"]["sequence_index"],
            group_arrays["motion_dev"]["gt_id"],
            group_arrays["motion_dev"]["target_frame"],
        ], axis=1)))
        if train_keys & dev_keys:
            raise RuntimeError("motion train/dev sample keys overlap")

        normalization = fit_motion_normalization(group_arrays["motion_train"], 1e-6)
        normalization.update({
            "fit_group": "motion_train",
            "split_manifest_sha256": args.split_manifest_sha256,
            "motion_dev_refit": False,
        })
        with (partial / "normalization.json").open("w") as handle:
            json.dump(normalization, handle, indent=2)
            handle.write("\n")

        group_summaries: dict[str, object] = {}
        for group_name, arrays in group_arrays.items():
            npz_path = partial / f"{group_name}.npz"
            np.savez_compressed(npz_path, **arrays)
            decoded = state_to_box_xyxy(
                arrays["target_state"], arrays["image_width"], arrays["image_height"]
            )
            reencoded = np.stack([
                (decoded[:, 0] + decoded[:, 2]) / (2.0 * arrays["image_width"]),
                (decoded[:, 1] + decoded[:, 3]) / (2.0 * arrays["image_height"]),
                np.log((decoded[:, 2] - decoded[:, 0]) / arrays["image_width"]),
                np.log((decoded[:, 3] - decoded[:, 1]) / arrays["image_height"]),
            ], axis=1)
            roundtrip_error = float(np.max(np.abs(reencoded - arrays["target_state"])))
            if roundtrip_error > 1e-10:
                raise RuntimeError(f"{group_name}: state roundtrip error {roundtrip_error}")
            positive = arrays["positive_detection_id"] >= 0
            group_summaries[group_name] = {
                "samples": int(len(arrays["target_frame"])),
                "target_with_positive_detection": int(positive.sum()),
                "target_without_positive_detection": int((~positive).sum()),
                "frame_gap_min": int(arrays["frame_gap"].min()),
                "frame_gap_max": int(arrays["frame_gap"].max()),
                "sequence_count": len(group_sequence_summaries[group_name]),
                "array_content_sha256": array_content_sha256(arrays),
                "npz_sha256": sha256_file(npz_path),
                "npz_bytes": npz_path.stat().st_size,
                "state_roundtrip_max_abs_error": roundtrip_error,
                "per_sequence": group_sequence_summaries[group_name],
            }

        sequence_index_path = partial / "sequence_index.json"
        with sequence_index_path.open("w") as handle:
            json.dump({str(index): sequence for sequence, index in sequence_index.items()}, handle, indent=2)
            handle.write("\n")
        repo_root = Path(__file__).resolve().parents[1]
        input_hashes.update({
            "split_manifest": sha256_file(args.split_manifest),
            "detection_manifest": sha256_file(args.detection_manifest),
            "protocol": sha256_file(args.protocol),
            "analysis_script": sha256_file(Path(__file__)),
            "sample_helper": sha256_file(repo_root / "src/mot_research/data/motion_samples.py"),
            "split_helper": sha256_file(repo_root / "src/mot_research/data/motion_split.py"),
            "failure_event_helper": sha256_file(
                repo_root / "src/mot_research/analysis/failure_events.py"
            ),
        })
        with (partial / "input_hashes.json").open("w") as handle:
            json.dump(input_hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")

        summary = {
            "status": "PASS",
            "scope": "P1 strictly-online motion samples from DanceTrack motion train/dev",
            "split_validation": split_validation,
            "groups": group_summaries,
            "train_dev_sample_key_overlap": 0,
            "normalization_fit_group": "motion_train",
            "motion_dev_normalization_refit": False,
            "history_length": 8,
            "minimum_history_observations": 2,
            "maximum_frame_gap": 31,
            "detection_threshold_strictly_greater_than": 0.6,
            "matching_iou_threshold": 0.5,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "split_manifest_sha256": args.split_manifest_sha256,
            "detection_manifest_sha256": args.detection_manifest_sha256,
            "limitations": [
                "GT is used for training-only correspondence and supervised targets.",
                "Histories contain prior matched detector observations, not closed-loop tracker histories.",
                "Sample counts and offline targets are not tracking improvements.",
                "Motion-dev includes four deliberately selected G2 anchors.",
            ],
        }
        with (partial / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        runtime = {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_policy": "CPU-only sample construction; physical GPU 0 forbidden",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "git_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root, check=True,
                capture_output=True, text=True,
            ).stdout),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (partial / "runtime.json").open("w") as handle:
            json.dump(runtime, handle, indent=2)
            handle.write("\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        if partial.exists():
            with (partial / "failure.json").open("w") as handle:
                json.dump({
                    "status": "FAIL", "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                }, handle, indent=2)
                handle.write("\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
