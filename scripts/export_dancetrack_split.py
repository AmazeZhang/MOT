#!/usr/bin/env python3
"""Export fixed detections and direct OC-SORT outputs for a DanceTrack split."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

import numpy as np


CACHE_FIELDS = ["frame", "det_id", "x1", "y1", "x2", "y2", "score", "class_id"]
TRACKER_ARGS = {
    "det_thresh": 0.6,
    "iou_threshold": 0.3,
    "asso_func": "iou",
    "delta_t": 3,
    "inertia": 0.2,
    "use_byte": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Process only this sequence (repeatable); default is every sequence in the split",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed sequences only after validating their recorded hashes",
    )
    return parser.parse_args()


def require_safe_single_gpu() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select one physical GPU from 1-7")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) != 1 or not devices[0].isdigit():
        raise RuntimeError(f"expected one numeric physical GPU, got CUDA_VISIBLE_DEVICES={visible!r}")
    physical_gpu = int(devices[0])
    if physical_gpu == 0:
        raise RuntimeError("physical GPU 0 is reserved for the Ubuntu graphical desktop")
    if physical_gpu not in range(1, 8):
        raise RuntimeError(f"physical GPU must be in 1-7, got {physical_gpu}")
    return physical_gpu


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def sync_file(handle: IO[str]) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = partial_path(path)
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path} or {temporary}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        sync_file(handle)
    os.replace(temporary, path)


def validate_completed(
    sequence_name: str,
    cache_file: Path,
    tracker_file: Path,
    summary_file: Path,
) -> dict[str, object]:
    summary = json.loads(summary_file.read_text())
    if summary.get("status") != "PASS" or summary.get("sequence") != sequence_name:
        raise RuntimeError(f"invalid completion summary for {sequence_name}: {summary_file}")
    expected = {
        "cache_sha256": sha256(cache_file),
        "tracker_output_sha256": sha256(tracker_file),
    }
    for key, actual in expected.items():
        if summary.get(key) != actual:
            raise RuntimeError(
                f"completed artifact hash mismatch for {sequence_name}: "
                f"{key} recorded={summary.get(key)!r} actual={actual!r}"
            )
    return summary


def discover_sequences(split_dir: Path, requested: list[str]) -> list[Path]:
    available = {path.name: path for path in split_dir.iterdir() if path.is_dir()}
    names = sorted(requested or available)
    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(f"unknown sequences under {split_dir}: {missing}")
    if len(names) != len(set(names)):
        raise ValueError("--sequence values must be unique")
    if not names:
        raise RuntimeError(f"no sequence directories found under {split_dir}")
    return [available[name] for name in names]


def sequence_metadata(sequence_dir: Path) -> tuple[list[Path], int, int]:
    config_file = sequence_dir / "seqinfo.ini"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    sequence = config["Sequence"]
    sequence_length = sequence.getint("seqLength")
    image_height = sequence.getint("imHeight")
    image_width = sequence.getint("imWidth")
    image_paths = sorted((sequence_dir / "img1").glob("*.jpg"))
    frame_ids = [int(path.stem) for path in image_paths]
    if len(image_paths) != sequence_length:
        raise RuntimeError(
            f"{sequence_dir.name}: seqLength={sequence_length}, found {len(image_paths)} images"
        )
    if frame_ids != list(range(1, sequence_length + 1)):
        raise RuntimeError(f"{sequence_dir.name}: image frame names are not exactly 1..seqLength")
    return image_paths, image_height, image_width


def main() -> None:
    args = parse_args()
    physical_gpu = require_safe_single_gpu()
    for path in (args.oc_sort_root, args.split_dir, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.run_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse run directory without --resume: {args.run_dir}")

    sequence_dirs = discover_sequences(args.split_dir, args.sequence)
    tracker_root = args.run_dir / "trackers" / "baseline" / "data"
    sequence_summary_root = args.run_dir / "sequences"
    split_summary_file = args.run_dir / "summary.json"
    if split_summary_file.exists():
        raise FileExistsError(f"split is already finalized: {split_summary_file}")

    args.cache_root.mkdir(parents=True, exist_ok=True)
    tracker_root.mkdir(parents=True, exist_ok=True)
    sequence_summary_root.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, object]] = []
    pending: list[Path] = []
    for sequence_dir in sequence_dirs:
        name = sequence_dir.name
        cache_file = args.cache_root / f"{name}.csv"
        tracker_file = tracker_root / f"{name}.txt"
        summary_file = sequence_summary_root / f"{name}.json"
        finals = [cache_file, tracker_file, summary_file]
        partials = [partial_path(path) for path in finals]
        existing_partials = [str(path) for path in partials if path.exists()]
        if existing_partials:
            raise RuntimeError(
                f"partial artifacts require manual inspection before resuming {name}: {existing_partials}"
            )
        existing_finals = [path.exists() for path in finals]
        if all(existing_finals):
            if not args.resume:
                raise FileExistsError(f"completed artifacts already exist for {name}")
            summary = validate_completed(name, cache_file, tracker_file, summary_file)
            completed.append(summary)
            print(f"[skip] {name}: completed hashes verified", flush=True)
        elif any(existing_finals):
            raise RuntimeError(f"incomplete final artifact set for {name}: {finals}")
        else:
            pending.append(sequence_dir)

    sys.path.insert(0, str(args.oc_sort_root.resolve()))
    import cv2
    import torch
    from trackers.ocsort_tracker.ocsort import OCSort
    from yolox.data import ValTransform
    from yolox.exp import get_exp
    from yolox.utils import postprocess

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the exporter must see exactly one explicitly selected safe CUDA device")

    exp_file = args.oc_sort_root / "exps/example/mot/yolox_dancetrack_val.py"
    exp = get_exp(str(exp_file), None)
    model = exp.get_model()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to("cuda:0")
    transform = ValTransform(
        rgb_means=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    oc_sort_commit = subprocess.check_output(
        ["git", "-C", str(args.oc_sort_root), "rev-parse", "HEAD"], text=True
    ).strip()
    weight_sha256 = sha256(args.weights)
    run_started = time.time()

    for sequence_index, sequence_dir in enumerate(pending, start=1):
        name = sequence_dir.name
        image_paths, expected_height, expected_width = sequence_metadata(sequence_dir)
        cache_file = args.cache_root / f"{name}.csv"
        tracker_file = tracker_root / f"{name}.txt"
        summary_file = sequence_summary_root / f"{name}.json"
        cache_partial = partial_path(cache_file)
        tracker_partial = partial_path(tracker_file)
        tracker = OCSort(**TRACKER_ARGS)
        detections = 0
        tracker_rows = 0
        empty_detection_frames: list[int] = []
        inference_seconds = 0.0
        sequence_started = time.time()
        print(
            f"[run {sequence_index}/{len(pending)}] {name}: {len(image_paths)} frames",
            flush=True,
        )

        with cache_partial.open("w", newline="") as cache_handle, tracker_partial.open(
            "w", newline=""
        ) as tracker_handle:
            cache_writer = csv.writer(cache_handle)
            tracker_writer = csv.writer(tracker_handle)
            cache_writer.writerow(CACHE_FIELDS)
            for frame_id, image_path in enumerate(image_paths, start=1):
                raw = cv2.imread(str(image_path))
                if raw is None:
                    raise RuntimeError(f"failed to read {image_path}")
                height, width = raw.shape[:2]
                if (height, width) != (expected_height, expected_width):
                    raise RuntimeError(
                        f"{image_path}: image is {width}x{height}, expected "
                        f"{expected_width}x{expected_height}"
                    )
                image, _ = transform(raw, None, exp.test_size)
                tensor = torch.from_numpy(image).unsqueeze(0).float().to("cuda:0")
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    prediction = model(tensor)
                    output = postprocess(
                        prediction,
                        exp.num_classes,
                        exp.test_conf,
                        exp.nmsthre,
                    )[0]
                torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - start

                scale = min(exp.test_size[0] / height, exp.test_size[1] / width)
                if output is None:
                    tracker_input = None
                    empty_detection_frames.append(frame_id)
                else:
                    output_cpu = output.detach().cpu().numpy()
                    boxes = output_cpu[:, :4] / scale
                    scores = output_cpu[:, 4] * output_cpu[:, 5]
                    classes = output_cpu[:, 6].astype(int)
                    for det_id, (box, score, class_id) in enumerate(
                        zip(boxes, scores, classes)
                    ):
                        cache_writer.writerow(
                            [
                                frame_id,
                                det_id,
                                *[f"{value:.10g}" for value in box],
                                f"{score:.10g}",
                                int(class_id),
                            ]
                        )
                    detections += len(boxes)
                    tracker_input = output

                tracks = tracker.update(tracker_input, (height, width), exp.test_size)
                for x1, y1, x2, y2, track_id in tracks:
                    tracker_writer.writerow(
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
            sync_file(cache_handle)
            sync_file(tracker_handle)

        cache_hash = sha256(cache_partial)
        tracker_hash = sha256(tracker_partial)
        os.replace(cache_partial, cache_file)
        os.replace(tracker_partial, tracker_file)
        summary: dict[str, object] = {
            "status": "PASS",
            "sequence": name,
            "sequence_length": len(image_paths),
            "image_height": expected_height,
            "image_width": expected_width,
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "process_device": "cuda:0",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "oc_sort_commit": oc_sort_commit,
            "weight": str(args.weights.resolve()),
            "weight_sha256": weight_sha256,
            "input_size": list(exp.test_size),
            "detector_confidence": exp.test_conf,
            "detector_nms": exp.nmsthre,
            "tracker": TRACKER_ARGS,
            "empty_frame_policy": "upstream_none",
            "cache_file": str(cache_file.resolve()),
            "cache_sha256": cache_hash,
            "detections": detections,
            "empty_detection_frames": empty_detection_frames,
            "tracker_rows": tracker_rows,
            "tracker_output": str(tracker_file.resolve()),
            "tracker_output_sha256": tracker_hash,
            "inference_seconds": inference_seconds,
            "wall_seconds": time.time() - sequence_started,
        }
        write_json_atomic(summary_file, summary)
        completed.append(summary)
        print(
            f"[done] {name}: detections={detections}, tracks={tracker_rows}, "
            f"empty_frames={len(empty_detection_frames)}, inference={inference_seconds:.3f}s",
            flush=True,
        )

    completed.sort(key=lambda item: str(item["sequence"]))
    split_summary: dict[str, object] = {
        "status": "PASS",
        "split_dir": str(args.split_dir.resolve()),
        "sequences": len(completed),
        "frames": sum(int(item["sequence_length"]) for item in completed),
        "detections": sum(int(item["detections"]) for item in completed),
        "tracker_rows": sum(int(item["tracker_rows"]) for item in completed),
        "empty_detection_frames": sum(
            len(item["empty_detection_frames"]) for item in completed
        ),
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "process_device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "oc_sort_commit": oc_sort_commit,
        "weight": str(args.weights.resolve()),
        "weight_sha256": weight_sha256,
        "input_size": list(exp.test_size),
        "detector_confidence": exp.test_conf,
        "detector_nms": exp.nmsthre,
        "tracker": TRACKER_ARGS,
        "empty_frame_policy": "upstream_none",
        "cache_root": str(args.cache_root.resolve()),
        "tracker_root": str(tracker_root.resolve()),
        "sequence_summary_root": str(sequence_summary_root.resolve()),
        "wall_seconds_this_invocation": time.time() - run_started,
        "sequence_summaries": completed,
    }
    write_json_atomic(split_summary_file, split_summary)
    print(json.dumps({key: value for key, value in split_summary.items() if key != "sequence_summaries"}, indent=2))


if __name__ == "__main__":
    main()
