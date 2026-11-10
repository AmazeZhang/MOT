#!/usr/bin/env python3
"""Run a short DanceTrack detector -> cache -> OC-SORT replay smoke test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oc-sort-root", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=50)
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


def write_tracker_output(path: Path, outputs: list[tuple[int, np.ndarray]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for frame_id, tracks in outputs:
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


def main() -> None:
    args = parse_args()
    physical_gpu = require_safe_single_gpu()
    for path in (args.oc_sort_root, args.sequence_dir, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.cache_file.exists():
        raise FileExistsError(f"refusing to overwrite detection cache: {args.cache_file}")
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.run_dir}")

    sys.path.insert(0, str(args.oc_sort_root.resolve()))
    import cv2
    import torch
    from trackers.ocsort_tracker.ocsort import OCSort
    from yolox.data import ValTransform
    from yolox.exp import get_exp
    from yolox.utils import postprocess

    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.device_count() == 1, "the smoke test must expose exactly one safe physical GPU"

    image_paths = sorted((args.sequence_dir / "img1").glob("*.jpg"))[: args.max_frames]
    if not image_paths:
        raise RuntimeError(f"no JPG frames found under {args.sequence_dir / 'img1'}")
    frame_ids = [int(path.stem) for path in image_paths]
    if frame_ids != list(range(frame_ids[0], frame_ids[0] + len(frame_ids))):
        raise RuntimeError("selected frame names are not contiguous")

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
    tracker_args = {
        "det_thresh": 0.6,
        "iou_threshold": 0.3,
        "asso_func": "iou",
        "delta_t": 3,
        "inertia": 0.2,
        "use_byte": False,
    }
    direct_tracker = OCSort(**tracker_args)

    args.cache_file.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    direct_outputs: list[tuple[int, np.ndarray]] = []
    frame_meta: dict[int, dict[str, object]] = {}
    inference_seconds = 0.0

    with args.cache_file.open("w", newline="") as cache_handle:
        cache_writer = csv.writer(cache_handle)
        cache_writer.writerow(["frame", "det_id", "x1", "y1", "x2", "y2", "score", "class_id"])
        for frame_id, image_path in zip(frame_ids, image_paths):
            raw = cv2.imread(str(image_path))
            if raw is None:
                raise RuntimeError(f"failed to read {image_path}")
            height, width = raw.shape[:2]
            image, _ = transform(raw, None, exp.test_size)
            tensor = torch.from_numpy(image).unsqueeze(0).float().to("cuda:0")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model(tensor)
                output = postprocess(prediction, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start

            scale = min(exp.test_size[0] / height, exp.test_size[1] / width)
            if output is None:
                cache_rows = np.empty((0, 8), dtype=float)
                tracker_input = None
            else:
                output_cpu = output.detach().cpu().numpy()
                boxes = output_cpu[:, :4] / scale
                scores = output_cpu[:, 4] * output_cpu[:, 5]
                classes = output_cpu[:, 6].astype(int)
                cache_rows = np.column_stack(
                    [
                        np.full(len(boxes), frame_id),
                        np.arange(len(boxes)),
                        boxes,
                        scores,
                        classes,
                    ]
                )
                tracker_input = output
            for row in cache_rows:
                cache_writer.writerow(
                    [int(row[0]), int(row[1]), *[f"{value:.10g}" for value in row[2:7]], int(row[7])]
                )

            tracks = direct_tracker.update(tracker_input, (height, width), exp.test_size)
            direct_outputs.append((frame_id, tracks.copy()))
            frame_meta[frame_id] = {
                "image": image_path.name,
                "height": height,
                "width": width,
                "scale": scale,
                "detections": int(len(cache_rows)),
                "tracks": int(len(tracks)),
            }

    cached_rows: dict[int, list[list[float]]] = defaultdict(list)
    with args.cache_file.open(newline="") as cache_handle:
        for row in csv.DictReader(cache_handle):
            cached_rows[int(row["frame"])].append(
                [float(row[key]) for key in ("x1", "y1", "x2", "y2", "score")]
            )

    replay_tracker = OCSort(**tracker_args)
    replay_outputs: list[tuple[int, np.ndarray]] = []
    mismatches: list[dict[str, object]] = []
    max_abs_replay_error = 0.0
    replay_frame_and_id_equal = True
    for (frame_id, direct_tracks), image_path in zip(direct_outputs, image_paths):
        meta = frame_meta[frame_id]
        scale = float(meta["scale"])
        rows = np.asarray(cached_rows.get(frame_id, []), dtype=float)
        if rows.size == 0:
            replay_input = None
        else:
            rows = rows.reshape(-1, 5)
            replay_input = rows.copy()
            replay_input[:, :4] *= scale
        replay_tracks = replay_tracker.update(
            replay_input,
            (int(meta["height"]), int(meta["width"])),
            exp.test_size,
        )
        replay_outputs.append((frame_id, replay_tracks.copy()))
        same_shape = direct_tracks.shape == replay_tracks.shape
        if same_shape and direct_tracks.size:
            max_abs_replay_error = max(
                max_abs_replay_error,
                float(np.max(np.abs(direct_tracks[:, :4] - replay_tracks[:, :4]))),
            )
            replay_frame_and_id_equal = replay_frame_and_id_equal and np.array_equal(
                direct_tracks[:, 4], replay_tracks[:, 4]
            )
        elif not same_shape:
            replay_frame_and_id_equal = False
        same_values = same_shape and np.allclose(direct_tracks, replay_tracks, rtol=0, atol=1e-5)
        if not same_values:
            mismatches.append(
                {
                    "frame": frame_id,
                    "direct_shape": list(direct_tracks.shape),
                    "replay_shape": list(replay_tracks.shape),
                    "max_abs_error": (
                        float(np.max(np.abs(direct_tracks - replay_tracks))) if same_shape and direct_tracks.size else None
                    ),
                }
            )

    direct_path = args.run_dir / "tracker_direct.txt"
    replay_path = args.run_dir / "tracker_replay.txt"
    write_tracker_output(direct_path, direct_outputs)
    write_tracker_output(replay_path, replay_outputs)
    summary = {
        "status": "PASS" if not mismatches else "FAIL",
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "process_device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "oc_sort_commit": subprocess.check_output(
            ["git", "-C", str(args.oc_sort_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "weight": str(args.weights.resolve()),
        "weight_sha256": sha256(args.weights),
        "sequence": args.sequence_dir.name,
        "frame_start": frame_ids[0],
        "frame_end": frame_ids[-1],
        "frames": len(frame_ids),
        "input_size": list(exp.test_size),
        "detector_confidence": exp.test_conf,
        "detector_nms": exp.nmsthre,
        "tracker": tracker_args,
        "cache_file": str(args.cache_file.resolve()),
        "cache_sha256": sha256(args.cache_file),
        "total_detections": sum(int(meta["detections"]) for meta in frame_meta.values()),
        "total_direct_track_rows": sum(len(tracks) for _, tracks in direct_outputs),
        "inference_seconds": inference_seconds,
        "cache_replay_mismatches": mismatches,
        "cache_replay_frame_and_id_equal": replay_frame_and_id_equal,
        "cache_replay_max_abs_xyxy_error": max_abs_replay_error,
        "empty_frame_policy": "upstream_none",
        "direct_output": str(direct_path.resolve()),
        "direct_output_sha256": sha256(direct_path),
        "replay_output": str(replay_path.resolve()),
        "replay_output_sha256": sha256(replay_path),
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
