#!/usr/bin/env python3
"""Extract fixed Deep-OC-SORT DanceTrack ReID features from cached detections.

The official model/config/crop path is used with FP32 inference as required by
the S01 protocol. Features and metadata are written outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_detections(path: Path, last_frame: int) -> dict[int, list[tuple[int, np.ndarray, float]]]:
    result: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    seen: set[tuple[int, int]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["frame", "det_id", "x1", "y1", "x2", "y2", "score", "class_id"]:
            raise ValueError("unexpected detection cache schema")
        for row in reader:
            frame, det_id = int(row["frame"]), int(row["det_id"])
            if frame < 1 or det_id < 0 or (frame, det_id) in seen:
                raise ValueError("invalid or duplicate detection key")
            seen.add((frame, det_id))
            if frame > last_frame:
                continue
            box = np.asarray([float(row[k]) for k in ("x1", "y1", "x2", "y2")], dtype=np.float64)
            score = float(row["score"])
            if not np.isfinite(box).all() or not np.isfinite(score) or np.any(box[2:] <= box[:2]):
                raise ValueError("invalid source detection")
            if score > 0.6:
                result.setdefault(frame, []).append((det_id, box, score))
    return result


def crop_image(image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, str]:
    height, width = image.shape[:2]
    coords = np.round(box).astype(np.int32)
    coords[[0, 2]] = np.clip(coords[[0, 2]], 0, width)
    coords[[1, 3]] = np.clip(coords[[1, 3]], 0, height)
    crop_box = coords.astype(np.int32)
    x1, y1, x2, y2 = crop_box
    if x2 <= x1 or y2 <= y1:
        return None, crop_box, "empty_crop_after_round_clip"
    patch = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    patch = cv2.resize(patch, (128, 384), interpolation=cv2.INTER_LINEAR)
    # Official EmbeddingComputer.grid_off branch uses RGB float32 in 0..255.
    return patch.transpose(2, 0, 1).astype(np.float32), crop_box, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.runtime_lock.read_text())
    source = Path(lock["feature_source_root"])
    weight = Path(lock["feature_weight_path"])
    if sha256(weight) != lock["feature_weight_sha256"]:
        raise RuntimeError("feature weight SHA256 differs from runtime lock")
    if sha256(Path(__file__)) != lock["code_hashes"]["scripts/extract_s01_appearance.py"]:
        raise RuntimeError("extractor code SHA256 differs from runtime lock")
    if args.sequence not in lock["selected_sequences"] or args.last_frame < 1:
        raise ValueError("sequence/frame range outside runtime lock")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(lock["cuda_visible_devices"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from runtime lock")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible nonzero physical GPU")
    sys.path.insert(0, str(source / "external"))
    from fast_reid.fastreid.config import get_cfg
    from fast_reid.fastreid.modeling.meta_arch import build_model
    from fast_reid.fastreid.utils.checkpoint import Checkpointer

    config_file = source / "external/fast_reid/configs/MOT17/sbs_S50.yml"
    if sha256(config_file) != lock["extractor_config_sha256"]:
        raise RuntimeError("extractor config SHA256 differs from runtime lock")
    cfg = get_cfg()
    cfg.merge_from_file(str(config_file))
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = str(weight)
    cfg.freeze()
    model = build_model(cfg).eval().float().cuda()
    Checkpointer(model).load(str(weight))

    cache = Path(lock["detection_root"]) / f"{args.sequence}.csv"
    if sha256(cache) != lock["input_detection_sha256"][args.sequence]:
        raise RuntimeError("detection cache SHA256 differs from runtime lock")
    data = load_detections(cache, args.last_frame)
    image_dir = Path(lock["train_root"]) / args.sequence / "img1"
    keys: list[tuple[int, int]] = []
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    crop_boxes: list[np.ndarray] = []
    features: list[np.ndarray] = []
    valid: list[bool] = []
    reasons: list[str] = []
    extract_seconds = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for frame in range(1, args.last_frame + 1):
            rows = data.get(frame, [])
            if not rows:
                continue
            image = cv2.imread(str(image_dir / f"{frame:08d}.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"missing or unreadable image: {args.sequence} frame {frame}")
            batch: list[np.ndarray] = []
            batch_indices: list[int] = []
            for det_id, box, score in rows:
                patch, crop_box, reason = crop_image(image, box)
                index = len(keys)
                keys.append((frame, det_id))
                boxes.append(box)
                scores.append(score)
                crop_boxes.append(crop_box)
                features.append(np.zeros(2048, dtype=np.float32))
                valid.append(False)
                reasons.append(reason)
                if patch is not None:
                    batch.append(patch)
                    batch_indices.append(index)
            if batch:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                raw = model(torch.from_numpy(np.stack(batch)).cuda()).float()
                normalized = torch.nn.functional.normalize(raw, dim=-1).cpu().numpy()
                torch.cuda.synchronize()
                extract_seconds += time.perf_counter() - t0
                if normalized.shape != (len(batch), 2048):
                    raise RuntimeError("unexpected feature dimension")
                for index, value in zip(batch_indices, normalized):
                    if np.isfinite(value).all() and np.linalg.norm(value) > 0:
                        features[index] = value
                        valid[index] = True
                    else:
                        reasons[index] = "nonfinite_or_zero_feature"
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate feature cache keys")
    feature_array = np.stack(features) if features else np.empty((0, 2048), np.float32)
    if len(keys) == 0 or sum(not item for item in valid) / len(keys) > 0.01:
        raise RuntimeError("empty feature table or invalid feature rate above 1%")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite feature cache")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keys=np.asarray(keys, dtype=np.int32),
                        xyxy=np.asarray(boxes, dtype=np.float64),
                        scores=np.asarray(scores, dtype=np.float64),
                        crop_xyxy=np.asarray(crop_boxes, dtype=np.int32),
                        features=feature_array, valid=np.asarray(valid, dtype=np.bool_),
                        invalid_reason=np.asarray(reasons, dtype="U40"))
    with np.load(args.output, allow_pickle=False) as saved:
        assert np.array_equal(saved["keys"], np.asarray(keys, dtype=np.int32))
        assert np.array_equal(saved["valid"], np.asarray(valid, dtype=np.bool_))
        assert np.array_equal(saved["features"], feature_array)
    t0 = time.perf_counter()
    check_ok = 0
    wanted = keys[:100]
    wanted_frames = set(frame for frame, _ in wanted)
    direct: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, bool, str]] = {}
    with torch.inference_mode():
        for frame in sorted(wanted_frames):
            image = cv2.imread(str(image_dir / f"{frame:08d}.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"direct check image unreadable: {frame}")
            patches: list[np.ndarray] = []
            patch_keys: list[tuple[int, int]] = []
            patch_crops: list[np.ndarray] = []
            for det_id, box, _ in data[frame]:
                patch, crop_box, reason = crop_image(image, box)
                key = (frame, det_id)
                if patch is None:
                    direct[key] = (np.zeros(2048, np.float32), crop_box, False, reason)
                else:
                    patches.append(patch)
                    patch_keys.append(key)
                    patch_crops.append(crop_box)
            if patches:
                values = model(torch.from_numpy(np.stack(patches)).cuda()).float()
                values = torch.nn.functional.normalize(values, dim=-1).cpu().numpy()
                for key, crop_box, value in zip(patch_keys, patch_crops, values):
                    ok = bool(np.isfinite(value).all() and np.linalg.norm(value) > 0)
                    direct[key] = (value if ok else np.zeros(2048, np.float32),
                                   crop_box, ok,
                                   "" if ok else "nonfinite_or_zero_feature")
        for index, key in enumerate(wanted):
            value, direct_crop, direct_valid, direct_reason = direct[key]
            if (direct_valid != valid[index] or direct_reason != reasons[index] or
                    not np.array_equal(direct_crop, crop_boxes[index]) or
                    not np.allclose(value, feature_array[index], atol=1e-5, rtol=1e-4)):
                raise RuntimeError(f"direct/cache feature mismatch at {key}")
            check_ok += 1
    check_seconds = time.perf_counter() - t0
    metadata = {
        "status": "ENGINEERING_PASS", "sequence": args.sequence,
        "frames_inclusive": [1, args.last_frame], "records": len(keys),
        "invalid_features": int(sum(not item for item in valid)),
        "duplicate_keys": 0, "direct_cache_checks": check_ok,
        "dimension": 2048, "dtype": "float32", "normalization": "L2",
        "feature_weight_sha256": sha256(weight), "detection_cache_sha256": sha256(cache),
        "runtime_lock_sha256": sha256(args.runtime_lock), "feature_cache_sha256": sha256(args.output),
        "feature_cache_seconds": extract_seconds, "direct_check_seconds": check_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
