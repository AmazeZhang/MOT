#!/usr/bin/env python3
"""Run official TrackEval aggregation for one P1 tracker over fixed sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mot_research.data.motion_split import sequence_names_sha256


METRICS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", dest="sequences", required=True)
    parser.add_argument("--sequence-names-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rounded-csv", type=Path)
    return parser.parse_args()


def metric_projection(values: dict[str, Any]) -> dict[str, float | int]:
    return {
        "HOTA": float(np.mean(values["HOTA"]["HOTA"])) * 100.0,
        "DetA": float(np.mean(values["HOTA"]["DetA"])) * 100.0,
        "AssA": float(np.mean(values["HOTA"]["AssA"])) * 100.0,
        "MOTA": float(values["CLEAR"]["MOTA"]) * 100.0,
        "IDF1": float(values["Identity"]["IDF1"]) * 100.0,
        "IDSW": int(values["CLEAR"]["IDSW"]),
        "Frag": int(values["CLEAR"]["Frag"]),
        "Dets": int(values["Count"]["Dets"]),
        "GT_Dets": int(values["Count"]["GT_Dets"]),
        "IDs": int(values["Count"]["IDs"]),
        "GT_IDs": int(values["Count"]["GT_IDs"]),
    }


def validate_rounded_reference(
    per_sequence: dict[str, dict[str, float | int]], path: Path
) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = {row["sequence"]: row for row in csv.DictReader(handle)}
    differences: dict[str, dict[str, float]] = {}
    for sequence, actual in per_sequence.items():
        if sequence not in rows:
            raise RuntimeError(f"rounded reference lacks {sequence}")
        row_differences = {}
        for metric in METRICS:
            expected = int(rows[sequence][metric]) if metric == "IDSW" else float(
                rows[sequence][metric]
            )
            difference = float(actual[metric]) - expected
            if metric == "IDSW":
                if difference != 0:
                    raise RuntimeError(f"{sequence}.{metric} differs by {difference}")
            elif abs(difference) > 0.0005000001:
                raise RuntimeError(
                    f"{sequence}.{metric} differs from 3-decimal reference by {difference}"
                )
            row_differences[metric] = difference
        differences[sequence] = row_differences
    return {
        "status": "PASS",
        "reference": str(path.resolve()),
        "reference_sha256": sha256_file(path),
        "maximum_absolute_difference": max(
            abs(value)
            for row in differences.values()
            for value in row.values()
        ),
        "differences": differences,
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("TrackEval requires CUDA_VISIBLE_DEVICES empty")
    sequences = list(args.sequences)
    if len(sequences) != len(set(sequences)) or sequences != sorted(sequences):
        raise ValueError("sequences must be unique and sorted")
    digest = sequence_names_sha256(sequences)
    if digest != args.sequence_names_sha256:
        raise RuntimeError(f"sequence-name hash mismatch: {digest}")
    for sequence in sequences:
        for path in (
            args.gt_root / sequence / "seqinfo.ini",
            args.gt_root / sequence / "gt" / "gt.txt",
            args.tracker_root / f"{sequence}.txt",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

    partial = args.output_dir.with_name(args.output_dir.name + ".partial")
    failed = args.output_dir.with_name(args.output_dir.name + ".failed")
    if any(path.exists() for path in (args.output_dir, partial, failed)):
        raise FileExistsError(f"refusing to overwrite output near {args.output_dir}")
    partial.mkdir(parents=True)
    try:
        seqmap = partial / "seqmap.txt"
        seqmap.write_text("name\n" + "".join(f"{name}\n" for name in sequences))
        sys.path.insert(0, str(args.trackeval_root.resolve()))
        import trackeval

        evaluator_config = trackeval.Evaluator.get_default_eval_config()
        evaluator_config.update({
            "USE_PARALLEL": False,
            "PRINT_RESULTS": False,
            "PRINT_ONLY_COMBINED": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "DISPLAY_LESS_PROGRESS": True,
        })
        official_output = partial / "official"
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update({
            "GT_FOLDER": str(args.gt_root.resolve()),
            "TRACKERS_FOLDER": str(args.tracker_root.resolve().parent),
            "OUTPUT_FOLDER": str(official_output),
            "TRACKERS_TO_EVAL": [args.tracker_root.name],
            "TRACKER_DISPLAY_NAMES": [args.method_id],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQMAP_FILE": str(seqmap),
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        })
        metrics = [
            trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
            trackeval.metrics.Identity({"PRINT_CONFIG": False}),
        ]
        raw, messages = trackeval.Evaluator(evaluator_config).evaluate(
            [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics
        )
        tracker_key = args.tracker_root.name
        message = messages["MotChallenge2DBox"][tracker_key]
        if message != "Success":
            raise RuntimeError(str(message))
        tracker_results = raw["MotChallenge2DBox"][tracker_key]
        per_sequence = {
            sequence: metric_projection(tracker_results[sequence]["pedestrian"])
            for sequence in sequences
        }
        combined = metric_projection(tracker_results["COMBINED_SEQ"]["pedestrian"])
        rounded_validation = None
        if args.expected_rounded_csv is not None:
            rounded_validation = validate_rounded_reference(
                per_sequence, args.expected_rounded_csv
            )
        summary = {
            "execution_status": "PASS",
            "method_id": args.method_id,
            "official_trackeval_combined": combined,
            "per_sequence": per_sequence,
            "sequence_count": len(sequences),
            "sequence_names_sha256": digest,
            "rounded_registered_reference_validation": rounded_validation,
            "aggregation_note": "COMBINED_SEQ from official TrackEval; not a mean of sequence HOTA",
        }
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        input_hashes = {
            "sequences": {
                sequence: {
                    "seqinfo": sha256_file(args.gt_root / sequence / "seqinfo.ini"),
                    "gt": sha256_file(args.gt_root / sequence / "gt" / "gt.txt"),
                    "tracker": sha256_file(args.tracker_root / f"{sequence}.txt"),
                }
                for sequence in sequences
            },
            "evaluator": sha256_file(Path(__file__)),
            "trackeval_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=args.trackeval_root,
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "rounded_reference": (
                None if args.expected_rounded_csv is None
                else sha256_file(args.expected_rounded_csv)
            ),
        }
        (partial / "input_hashes.json").write_text(
            json.dumps(input_hashes, indent=2, sort_keys=True) + "\n"
        )
        (partial / "runtime.json").write_text(json.dumps({
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "trackeval_commit": input_hashes["trackeval_commit"],
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        os.replace(partial, args.output_dir)
        print(json.dumps(summary, indent=2))
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
