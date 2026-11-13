#!/usr/bin/env python3
"""Replay fixed detections through OC-SORT with a natural-history P1 policy.

This entry point does not read ground truth or compute research metrics.  It
produces tracker outputs and engineering counters for a separately frozen
evaluation step.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from mot_research.data.motion_split import validate_motion_split
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner,
    NaturalMotionFirstRoundPolicy,
    TorchMotionPredictor,
    TrackerDetectionHistory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-run-id", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--run-dir", type=Path)
    output.add_argument("--run-dir-from-job-artifact", action="store_true")
    parser.add_argument(
        "--mode",
        choices=sorted(NaturalMotionFirstRoundPolicy.MODES),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--input-height", type=int, default=800)
    parser.add_argument("--input-width", type=int, default=1440)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--maximum-frame-gap", type=int, default=31)
    parser.add_argument("--warmup-frames", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}: {actual} != {expected}")


def load_cache(path: Path, sequence_length: int) -> dict[int, list[list[float]]]:
    result: dict[int, list[list[float]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["frame", "det_id", "x1", "y1", "x2", "y2", "score", "class_id"]
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: unexpected cache schema {reader.fieldnames}")
        for row in reader:
            frame_id = int(row["frame"])
            detection_id = int(row["det_id"])
            key = (frame_id, detection_id)
            if not 1 <= frame_id <= sequence_length:
                raise ValueError(f"{path}: frame outside sequence: {frame_id}")
            if key in seen:
                raise ValueError(f"{path}: duplicate frame/det_id: {key}")
            seen.add(key)
            values = [float(row[name]) for name in ("x1", "y1", "x2", "y2", "score")]
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite detection at {key}")
            if values[2] <= values[0] or values[3] <= values[1]:
                raise ValueError(f"{path}: invalid detection box at {key}")
            if not 0 <= values[4] <= 1:
                raise ValueError(f"{path}: invalid score at {key}")
            result[frame_id].append(values)
    return result


def assignment_runtime() -> dict[str, str]:
    try:
        import lap  # noqa: F401

        return {"backend": "lap.lapjv", "lap_version": importlib.metadata.version("lap")}
    except ImportError:
        return {
            "backend": "scipy.optimize.linear_sum_assignment",
            "scipy_version": scipy.__version__,
        }


def validate_device(mode: str, device: str) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = os.environ.get("MOT_PHYSICAL_GPU")
    process_device = os.environ.get("MOT_PROCESS_DEVICE")
    if mode == "learned":
        if device != "cuda:0":
            raise RuntimeError("learned closed-loop runs require --device cuda:0")
        if physical is None or int(physical) not in range(1, 8):
            raise RuntimeError("physical GPU must be one of 1..7; physical GPU0 is forbidden")
        if visible != physical or process_device != "cuda:0":
            raise RuntimeError("physical/logical CUDA mapping is inconsistent")
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("learned run must see exactly one CUDA device")
        return {
            "physical_gpu": int(physical),
            "CUDA_VISIBLE_DEVICES": visible,
            "process_device": "cuda:0",
            "torch_device_name": torch.cuda.get_device_name(0),
        }
    if device != "cpu":
        raise RuntimeError(f"{mode} requires --device cpu")
    if visible != "":
        raise RuntimeError("CPU policy runs require CUDA_VISIBLE_DEVICES to be empty")
    return {
        "physical_gpu": None,
        "CUDA_VISIBLE_DEVICES": visible,
        "process_device": "cpu",
    }


def load_predictor(args: argparse.Namespace) -> tuple[TorchMotionPredictor | None, dict[str, object]]:
    if args.mode != "learned":
        if args.checkpoint is not None or args.checkpoint_sha256 is not None:
            raise ValueError("checkpoints are valid only for learned mode")
        return None, {}
    if args.checkpoint is None or args.checkpoint_sha256 is None:
        raise ValueError("learned mode requires checkpoint and checkpoint SHA256")
    verify_hash(args.checkpoint, args.checkpoint_sha256)
    import torch

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    predictor = TorchMotionPredictor(checkpoint, args.device)
    return predictor, {
        "path": str(args.checkpoint.resolve()),
        "sha256": args.checkpoint_sha256,
        "seed": int(checkpoint["seed"]),
        "training_protocol_sha256": str(checkpoint["protocol_sha256"]),
        "train_samples_sha256": str(checkpoint["train_samples_sha256"]),
        "dev_samples_sha256": str(checkpoint["dev_samples_sha256"]),
    }


def validate_run_contract(
    protocol: dict[str, Any], args: argparse.Namespace, sequences: list[str]
) -> None:
    runs = protocol.get("runs")
    if not isinstance(runs, dict) or args.protocol_run_id not in runs:
        raise ValueError(f"protocol has no run contract {args.protocol_run_id!r}")
    contract = runs[args.protocol_run_id]
    expected = {
        "mode": args.mode,
        "sequences": sequences,
        "frame_range": [args.start_frame, args.end_frame],
        "device": args.device,
        "checkpoint_sha256": args.checkpoint_sha256,
        "history_length": args.history_length,
        "minimum_observations": args.minimum_observations,
        "maximum_frame_gap": args.maximum_frame_gap,
    }
    actual = {key: contract.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "CLI differs from frozen protocol run contract: "
            + json.dumps({"expected_from_cli": expected, "protocol": actual}, sort_keys=True)
        )


def apply_engineering_acceptance(
    protocol: dict[str, Any], run_id: str, summaries: list[dict[str, Any]]
) -> dict[str, object] | None:
    contract = protocol["runs"][run_id]
    acceptance = contract.get("engineering_acceptance")
    if acceptance is None:
        return None
    totals = {
        "frames_processed": sum(int(row["frames_processed"]) for row in summaries),
        "association_calls": sum(
            int(row["runner"]["association_calls"]) for row in summaries
        ),
        "replaced_tracker_rows": sum(
            int(row["runner"]["policy"]["counters"].get("replaced_tracker_rows", 0))
            for row in summaries
        ),
        "recorded_birth_observation": sum(
            int(row["runner"]["history"]["counters"].get("recorded_birth_observation", 0))
            for row in summaries
        ),
        "unresolved_birth": sum(
            int(row["runner"]["birth_mapping_counters"].get("unresolved_birth", 0))
            for row in summaries
        ),
        "ambiguous_birth": sum(
            int(row["runner"]["birth_mapping_counters"].get("ambiguous_birth", 0))
            for row in summaries
        ),
        "fallback_nonfinite_prediction": sum(
            int(row["runner"]["policy"]["counters"].get("fallback_nonfinite_prediction", 0))
            for row in summaries
        ),
        "fallback_invalid_prediction_box": sum(
            int(row["runner"]["policy"]["counters"].get("fallback_invalid_prediction_box", 0))
            for row in summaries
        ),
    }
    checks = {
        "frames_processed_exact": totals["frames_processed"]
        == int(acceptance["expected_frames_processed"]),
        "association_calls_exact": totals["association_calls"]
        == int(acceptance["expected_association_calls"]),
        "replaced_rows_minimum": totals["replaced_tracker_rows"]
        >= int(acceptance["minimum_replaced_tracker_rows"]),
        "birth_observations_minimum": totals["recorded_birth_observation"]
        >= int(acceptance["minimum_recorded_birth_observations"]),
        "unresolved_births_maximum": totals["unresolved_birth"]
        <= int(acceptance["maximum_unresolved_births"]),
        "ambiguous_births_maximum": totals["ambiguous_birth"]
        <= int(acceptance["maximum_ambiguous_births"]),
        "nonfinite_prediction_fallbacks_maximum": totals["fallback_nonfinite_prediction"]
        <= int(acceptance["maximum_nonfinite_prediction_fallbacks"]),
        "invalid_prediction_fallbacks_maximum": totals["fallback_invalid_prediction_box"]
        <= int(acceptance["maximum_invalid_prediction_fallbacks"]),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "totals": totals,
        "checks": checks,
    }
    if result["status"] != "PASS":
        raise RuntimeError("engineering acceptance failed: " + json.dumps(result, sort_keys=True))
    return result


def write_tracker_row(writer: Any, frame_id: int, track: np.ndarray) -> None:
    x1, y1, x2, y2, track_id = track
    writer.writerow([
        frame_id,
        int(track_id),
        f"{x1:.8f}",
        f"{y1:.8f}",
        f"{x2 - x1:.8f}",
        f"{y2 - y1:.8f}",
        1,
        -1,
        -1,
        -1,
    ])


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if args.run_dir_from_job_artifact:
        artifact_root = os.environ.get("MOT_JOB_ARTIFACT_DIR")
        if not artifact_root:
            raise RuntimeError("--run-dir-from-job-artifact requires MOT_JOB_ARTIFACT_DIR")
        args.run_dir = Path(artifact_root) / "closed_loop"
    assert args.run_dir is not None
    if args.start_frame != 1:
        raise ValueError("tracker fragments must start at frame 1 to preserve natural state")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError("end frame precedes start frame")
    if min(args.input_height, args.input_width, args.warmup_frames + 1) <= 0:
        raise ValueError("input dimensions and warmup must be positive")
    for path in (args.oc_sort_root, args.train_root, args.detection_root):
        if not path.exists():
            raise FileNotFoundError(path)
    verify_hash(args.split_manifest, args.split_manifest_sha256)
    verify_hash(args.protocol, args.protocol_sha256)
    device_record = validate_device(args.mode, args.device)

    partial = args.run_dir.with_name(args.run_dir.name + ".partial")
    failed = args.run_dir.with_name(args.run_dir.name + ".failed")
    if any(path.exists() for path in (args.run_dir, partial, failed)):
        raise FileExistsError(f"refusing to overwrite output near {args.run_dir}")
    partial.mkdir(parents=True)
    try:
        split = yaml.safe_load(args.split_manifest.read_text())
        protocol = yaml.safe_load(args.protocol.read_text())
        split_validation = validate_motion_split(split, args.train_root)
        allowed = list(split["motion_dev"]["sequences"])
        sequences = allowed if args.sequences is None else list(args.sequences)
        if not sequences or len(sequences) != len(set(sequences)):
            raise ValueError("sequence selection must be non-empty and unique")
        unexpected = sorted(set(sequences) - set(allowed))
        if unexpected:
            raise ValueError(f"sequences outside frozen motion-dev: {unexpected}")
        validate_run_contract(protocol, args, sequences)

        sys.path.insert(0, str(args.oc_sort_root.resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module

        predictor, checkpoint_record = load_predictor(args)
        tracker_args = {
            "det_thresh": 0.6,
            "iou_threshold": 0.3,
            "asso_func": "iou",
            "delta_t": 3,
            "inertia": 0.2,
            "use_byte": False,
        }
        tracker_root = partial / "trackers"
        diagnostic_root = partial / "diagnostics"
        tracker_root.mkdir()
        diagnostic_root.mkdir()
        sequence_summaries = []
        input_hashes: dict[str, Any] = {"sequences": {}}

        for sequence in sequences:
            sequence_dir = args.train_root / sequence
            seqinfo_path = sequence_dir / "seqinfo.ini"
            cache_path = args.detection_root / f"{sequence}.csv"
            config = configparser.ConfigParser()
            if not config.read(seqinfo_path):
                raise FileNotFoundError(seqinfo_path)
            info = config["Sequence"]
            sequence_length = info.getint("seqLength")
            image_height = info.getint("imHeight")
            image_width = info.getint("imWidth")
            fps = info.getfloat("frameRate")
            end_frame = sequence_length if args.end_frame is None else args.end_frame
            if end_frame > sequence_length:
                raise ValueError(f"{sequence}: end frame {end_frame} exceeds {sequence_length}")
            cache = load_cache(cache_path, sequence_length)
            scale = min(args.input_height / image_height, args.input_width / image_width)

            tracker = ocsort_module.OCSort(**tracker_args)
            history = TrackerDetectionHistory(
                history_length=args.history_length,
                score_threshold=float(tracker_args["det_thresh"]),
            )
            policy = NaturalMotionFirstRoundPolicy(
                mode=args.mode,
                history=history,
                image_width=image_width,
                image_height=image_height,
                fps=fps,
                predictor=predictor,
                minimum_observations=args.minimum_observations,
                maximum_frame_gap=args.maximum_frame_gap,
            )
            runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, policy)
            output_path = tracker_root / f"{sequence}.txt"
            tracker_rows = 0
            none_frames = []
            with output_path.open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                for frame_id in range(1, end_frame + 1):
                    rows = np.asarray(cache.get(frame_id, []), dtype=np.float64)
                    if rows.size == 0:
                        tracker_input = None
                        none_frames.append(frame_id)
                    else:
                        tracker_input = rows.reshape(-1, 5).copy()
                        tracker_input[:, :4] *= scale
                    tracks = runner.update(
                        tracker_input,
                        (image_height, image_width),
                        (args.input_height, args.input_width),
                        frame_id=frame_id,
                    )
                    tracks = np.asarray(tracks, dtype=np.float64)
                    if tracks.ndim != 2 or tracks.shape[1] != 5:
                        raise RuntimeError(f"{sequence}:{frame_id}: invalid tracker output shape")
                    if not np.isfinite(tracks).all():
                        raise RuntimeError(f"{sequence}:{frame_id}: non-finite tracker output")
                    if len(tracks) and np.any(
                        (tracks[:, 2] <= tracks[:, 0]) | (tracks[:, 3] <= tracks[:, 1])
                    ):
                        raise RuntimeError(f"{sequence}:{frame_id}: invalid tracker box")
                    if ocsort_module.associate is not runner.adapter.upstream_associate:
                        raise RuntimeError(f"{sequence}:{frame_id}: upstream symbol not restored")
                    for track in tracks:
                        write_tracker_row(writer, frame_id, track)
                        tracker_rows += 1

            calls_path = diagnostic_root / f"{sequence}_association_calls.jsonl"
            with calls_path.open("w") as handle:
                for record in policy.call_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            runner_summary = runner.snapshot(warmup_frames=args.warmup_frames)
            summary = {
                "sequence": sequence,
                "source_sequence_length": sequence_length,
                "frame_range": [1, end_frame],
                "frames_processed": end_frame,
                "cache_none_frames": none_frames,
                "tracker_rows": tracker_rows,
                "tracker_output_sha256": sha256_file(output_path),
                "association_calls_log_sha256": sha256_file(calls_path),
                "runner": runner_summary,
            }
            (diagnostic_root / f"{sequence}_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            sequence_summaries.append(summary)
            input_hashes["sequences"][sequence] = {
                "seqinfo": sha256_file(seqinfo_path),
                "detections": sha256_file(cache_path),
            }

        engineering_acceptance = apply_engineering_acceptance(
            protocol, args.protocol_run_id, sequence_summaries
        )
        aggregate = {
            "execution_status": "PASS",
            "scope": "natural tracker-history P1 engineering replay",
            "mode": args.mode,
            "research_metrics_computed": False,
            "ground_truth_read": False,
            "sequence_count": len(sequence_summaries),
            "frames_processed": sum(row["frames_processed"] for row in sequence_summaries),
            "tracker_rows": sum(row["tracker_rows"] for row in sequence_summaries),
            "sequences": sequence_summaries,
            "tracker": tracker_args,
            "history_contract": {
                "source": "real detections that update a live tracker, recorded after full update",
                "score_rule": "strictly greater than det_thresh=0.6",
                "history_length": args.history_length,
                "minimum_observations": args.minimum_observations,
                "maximum_frame_gap": args.maximum_frame_gap,
                "ineligible_or_invalid_action": "retain upstream KF first-round box and count fallback",
                "writeback": "none; alternative box exists only in first-round association input",
            },
            "checkpoint": checkpoint_record or None,
            "engineering_acceptance": engineering_acceptance,
            "device": device_record,
            "assignment": assignment_runtime(),
            "split_validation": split_validation,
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": args.protocol_sha256,
            "protocol_run_id": args.protocol_run_id,
            "limitations": [
                "This output contains engineering replay counters, not a research metric or decision.",
                "No checkpoint, threshold, or method is selected from this output.",
                "Motion-dev is development evidence and includes four selected diagnostic anchors.",
            ],
        }
        (partial / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")

        repo_root = Path(__file__).resolve().parents[1]
        input_hashes.update({
            "split_manifest": sha256_file(args.split_manifest),
            "protocol": sha256_file(args.protocol),
            "checkpoint": checkpoint_record or None,
            "runner_script": sha256_file(Path(__file__)),
            "natural_policy": sha256_file(
                repo_root / "src/mot_research/tracking/natural_motion_policy.py"
            ),
            "first_round_adapter": sha256_file(
                repo_root / "src/mot_research/tracking/first_round_adapter.py"
            ),
            "motion_model": sha256_file(
                repo_root / "src/mot_research/models/deterministic_motion.py"
            ),
            "upstream_ocsort": sha256_file(
                args.oc_sort_root / "trackers/ocsort_tracker/ocsort.py"
            ),
            "upstream_association": sha256_file(
                args.oc_sort_root / "trackers/ocsort_tracker/association.py"
            ),
        })
        (partial / "input_hashes.json").write_text(
            json.dumps(input_hashes, indent=2, sort_keys=True) + "\n"
        )

        runtime = {
            "argv": sys.argv,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "device": device_record,
            "gpu_policy": "physical GPU0 forbidden; learned job explicitly maps GPU1-7 to cuda:0",
            "cwd": str(Path.cwd()),
            "git_commit": git_commit(repo_root),
            "upstream_oc_sort_commit": git_commit(args.oc_sort_root),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.mode == "learned":
            import torch

            runtime.update({
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            })
        (partial / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        os.replace(partial, args.run_dir)
        print(json.dumps(aggregate, indent=2))
    except Exception as error:
        if partial.exists():
            failure = {
                "execution_status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {
                    key: os.environ.get(key) for key in (
                        "CUDA_VISIBLE_DEVICES", "MOT_PHYSICAL_GPU", "MOT_PROCESS_DEVICE",
                        "MOT_QUEUE_ID", "MOT_JOB_ID", "MOT_ATTEMPT",
                    )
                },
            }
            (partial / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
