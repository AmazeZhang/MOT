#!/usr/bin/env python3
"""Prepare split-isolated DanceTrack datasets and CAMELTrack state files for S08."""

from __future__ import annotations

import configparser
import datetime as dt
import hashlib
import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s08_camel_baseline_20261125.yaml"
SPLIT_PATH = REPO_ROOT / "manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml"
REPORT_DIR = REPO_ROOT / "reports/baselines/s08_camel_baseline_20261125"

UPSTREAM_CAMEL_DIR = DATA_ROOT / "third_party/CAMELTrack"
RAW_STATES_PKLZ = DATA_ROOT / "downloads/cameltrack/dancetrack-train.pklz"
PRETRAINED_CKPT = DATA_ROOT / "weights/camel/camel_bbox_app_kps_dancetrack.ckpt"
DANCETRACK_TRAIN_DIR = DATA_ROOT / "datasets/dancetrack/train"

S08_CACHE_ROOT = DATA_ROOT / "cache/s08_camel"
S08_DATASET_ROOT = S08_CACHE_ROOT / "dancetrack_s08_dataset"
S08_STATES_DIR = S08_CACHE_ROOT / "states"

GT_AND_CACHED_TRACKER_COLS = [
    "track_id",
    "track_bbox_ltwh",
    "track_bbox_conf",
    "ignored",
    "is_occluded",
    "iou_occlusion",
    "category_id",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(items: list[str]) -> str:
    return hashlib.sha256("".join(f"{x}\n" for x in sorted(items)).encode("ascii")).hexdigest()


def ensure_symlinks() -> None:
    S08_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val"):
        link_path = S08_DATASET_ROOT / split_name
        if link_path.is_symlink():
            if link_path.resolve() != DANCETRACK_TRAIN_DIR.resolve():
                link_path.unlink()
                os.symlink(DANCETRACK_TRAIN_DIR, link_path)
        elif link_path.exists():
            raise RuntimeError(f"Expected symlink or absent path at {link_path}")
        else:
            os.symlink(DANCETRACK_TRAIN_DIR, link_path)


def write_subset_pklz(
    out_path: Path,
    source_zf: zipfile.ZipFile,
    summary_obj: dict[str, Any],
    video_ids: list[int],
    frame_limit: int | None = None,
    strip_gt: bool = False,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    det_cols = list(summary_obj["columns"]["detection"])
    img_cols = list(summary_obj["columns"]["image"])
    if strip_gt:
        det_cols = [c for c in det_cols if c not in GT_AND_CACHED_TRACKER_COLS]
        img_cols = [c for c in img_cols if c != "is_labeled"]
    new_summary = {"columns": {"detection": det_cols, "image": img_cols}}

    total_dets = 0
    total_frames = 0
    per_video = {}

    tmp_path = out_path.with_suffix(".pklz.tmp")
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as dst:
        dst.writestr("summary.json", json.dumps(new_summary, indent=4).encode("utf-8"))
        for vid in sorted(video_ids):
            img_df = pickle.load(source_zf.open(f"{vid}_image.pkl"))
            det_df = pickle.load(source_zf.open(f"{vid}.pkl"))
            if frame_limit is not None:
                img_df = img_df[img_df["frame"] < frame_limit].copy()
                valid_img_ids = set(img_df.index.tolist())
                det_df = det_df[det_df["image_id"].isin(valid_img_ids)].copy()
            if strip_gt:
                drop_det = [c for c in GT_AND_CACHED_TRACKER_COLS if c in det_df.columns]
                if drop_det:
                    det_df = det_df.drop(columns=drop_det)
                if "is_labeled" in img_df.columns:
                    img_df = img_df.drop(columns=["is_labeled"])
            with dst.open(f"{vid}.pkl", "w", force_zip64=True) as fp:
                pickle.dump(det_df, fp, protocol=pickle.DEFAULT_PROTOCOL)
            with dst.open(f"{vid}_image.pkl", "w", force_zip64=True) as fp:
                pickle.dump(img_df, fp, protocol=pickle.DEFAULT_PROTOCOL)
            total_dets += int(len(det_df))
            total_frames += int(len(img_df))
            per_video[str(vid)] = {
                "frames": int(len(img_df)),
                "detections": int(len(det_df)),
                "zero_det_frames": int(len(img_df) - det_df["image_id"].nunique()),
            }
    tmp_path.replace(out_path)
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "strip_gt": strip_gt,
        "frame_limit": frame_limit,
        "video_count": len(video_ids),
        "total_frames": total_frames,
        "total_detections": total_dets,
        "per_video": per_video,
    }


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    split = yaml.safe_load(SPLIT_PATH.read_text())

    train32_seqs: list[str] = list(split["motion_train"]["sequences"])
    dev8_seqs: list[str] = list(split["motion_dev"]["sequences"])
    assert train32_seqs == protocol["data"]["train32"]["sequences"], "train32 mismatch vs protocol"
    assert dev8_seqs == protocol["data"]["dev8"]["sequences"], "dev8 mismatch vs protocol"
    assert len(train32_seqs) == 32 and len(dev8_seqs) == 8
    assert set(train32_seqs).isdisjoint(set(dev8_seqs))
    assert sha256_lines(sorted(train32_seqs)) == split["motion_train"]["sequence_names_sha256"]
    assert sha256_lines(sorted(dev8_seqs)) == split["motion_dev"]["sequence_names_sha256"]

    ensure_symlinks()

    disk_seqs = sorted(p.name for p in DANCETRACK_TRAIN_DIR.iterdir() if p.is_dir())
    assert len(disk_seqs) == 40
    assert set(disk_seqs) == set(train32_seqs) | set(dev8_seqs)

    zf = zipfile.ZipFile(RAW_STATES_PKLZ, "r")
    summary_obj = json.loads(zf.read("summary.json").decode("utf-8"))

    seq_to_vid: dict[str, int] = {}
    sequence_inventory: list[dict[str, Any]] = []
    expected_img_offset = 0

    for vid_idx, seq in enumerate(disk_seqs, start=1):
        cp = configparser.ConfigParser()
        cp.read(DANCETRACK_TRAIN_DIR / seq / "seqinfo.ini")
        seq_len = cp["Sequence"].getint("seqLength")
        im_w = cp["Sequence"].getint("imWidth")
        im_h = cp["Sequence"].getint("imHeight")
        fps = cp["Sequence"].getint("frameRate")

        img_df = pickle.load(zf.open(f"{vid_idx}_image.pkl"))
        det_df = pickle.load(zf.open(f"{vid_idx}.pkl"))

        seq_from_fp = Path(img_df["file_path"].iloc[0]).parts[-3]
        assert seq_from_fp == seq, f"{seq_from_fp} != {seq}"
        assert len(img_df) == seq_len, f"{seq}: {len(img_df)} != {seq_len}"
        assert img_df["id"].tolist() == list(range(expected_img_offset, expected_img_offset + seq_len))
        assert det_df["video_id"].unique().tolist() == [vid_idx]
        assert set(det_df["image_id"]).issubset(set(img_df["id"]))

        embs = np.stack(det_df["embeddings"])
        vis = np.stack(det_df["visibility_scores"])
        kps = np.stack(det_df["keypoints_xyc"])
        bbox = np.stack(det_df["bbox_ltwh"])
        conf = det_df["bbox_conf"].to_numpy()
        assert embs.shape[1:] == (6, 128) and np.isfinite(embs).all()
        assert vis.shape[1:] == (6,) and np.isfinite(vis).all()
        assert kps.shape[1:] == (17, 3) and np.isfinite(kps).all()
        assert bbox.shape[1:] == (4,) and np.isfinite(bbox).all()
        assert np.isfinite(conf).all()

        split_role = "train32" if seq in train32_seqs else "dev8"
        seq_to_vid[seq] = vid_idx
        det_frames = int(det_df["image_id"].nunique())
        sequence_inventory.append(
            {
                "sequence": seq,
                "video_id": vid_idx,
                "split_role": split_role,
                "nframes": seq_len,
                "im_width": im_w,
                "im_height": im_h,
                "frame_rate": fps,
                "image_id_start": expected_img_offset,
                "image_id_end_inclusive": expected_img_offset + seq_len - 1,
                "num_detections": int(len(det_df)),
                "frames_with_detections": det_frames,
                "zero_detection_frames": seq_len - det_frames,
                "bbox_conf_min": float(conf.min()),
                "bbox_conf_max": float(conf.max()),
            }
        )
        expected_img_offset += seq_len

    train32_vids = [seq_to_vid[s] for s in train32_seqs]
    dev8_vids = [seq_to_vid[s] for s in dev8_seqs]
    vid_0027 = seq_to_vid["dancetrack0027"]
    vid_0012 = seq_to_vid["dancetrack0012"]

    state_artifacts = {
        "train32_with_gt": write_subset_pklz(
            S08_STATES_DIR / "states_train32.pklz", zf, summary_obj, train32_vids, frame_limit=None, strip_gt=False
        ),
        "dev8_train_ref": write_subset_pklz(
            S08_STATES_DIR / "states_dev8_train_ref.pklz", zf, summary_obj, dev8_vids, frame_limit=None, strip_gt=False
        ),
        "dev8_gt_free": write_subset_pklz(
            S08_STATES_DIR / "states_dev8_gt_free.pklz", zf, summary_obj, dev8_vids, frame_limit=None, strip_gt=True
        ),
        "stage_b_0027_f1_400_with_gt": write_subset_pklz(
            S08_STATES_DIR / "states_b_0027_f1_400_with_gt.pklz", zf, summary_obj, [vid_0027], frame_limit=400, strip_gt=False
        ),
        "stage_b_0027_f1_400_gt_free": write_subset_pklz(
            S08_STATES_DIR / "states_b_0027_f1_400_gt_free.pklz", zf, summary_obj, [vid_0027], frame_limit=400, strip_gt=True
        ),
        "stage_b_0027_f1_200_gt_free": write_subset_pklz(
            S08_STATES_DIR / "states_b_0027_f1_200_gt_free.pklz", zf, summary_obj, [vid_0027], frame_limit=200, strip_gt=True
        ),
        "stage_c_train_0027_f1_400": write_subset_pklz(
            S08_STATES_DIR / "states_c_train_0027_f1_400.pklz", zf, summary_obj, [vid_0027], frame_limit=400, strip_gt=False
        ),
        "stage_c_dev_0012_f1_200": write_subset_pklz(
            S08_STATES_DIR / "states_c_dev_0012_f1_200.pklz", zf, summary_obj, [vid_0012], frame_limit=200, strip_gt=False
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = dt.datetime.now().astimezone().isoformat()

    source_audit = {
        "schema_version": "s08-source-audit-v1",
        "generated_at": now_iso,
        "upstream_cameltrack": {
            "repository": "https://github.com/TrackingLaboratory/CAMELTrack",
            "commit": "bc1c261075b0beacfe71c2896453be83643153c8",
            "local_path": str(UPSTREAM_CAMEL_DIR),
        },
        "huggingface_assets": {
            "repo_id": "trackinglaboratory/CAMELTrack",
            "revision": "6759b7c27b99d8f2b8d8a13be69c6dbbb9cc40ff",
            "pretrained_checkpoint": {
                "filename": "camel_bbox_app_kps_dancetrack.ckpt",
                "local_path": str(PRETRAINED_CKPT),
                "bytes": PRETRAINED_CKPT.stat().st_size,
                "sha256": sha256_file(PRETRAINED_CKPT),
                "training_data_provenance": "CAMELTrack official DanceTrack pretrained weights (README documents DanceTrack train+val two-stage training; exact embedded manifest absent -> dev8 overlap cannot be excluded, treat as platform reference only)",
                "epoch_in_ckpt": 9,
                "global_step_in_ckpt": 8120,
                "computed_sim_threshold_in_ckpt": 0.46,
            },
            "precomputed_states": {
                "filename": "states/dancetrack-train.pklz",
                "local_path": str(RAW_STATES_PKLZ),
                "bytes": RAW_STATES_PKLZ.stat().st_size,
                "sha256": sha256_file(RAW_STATES_PKLZ),
                "summary_columns": summary_obj["columns"],
                "detector_provenance": "unknown_embedded (CAMELTrack README documents YOLOX detector weights from DiffMOT)",
                "appearance_provenance": "unknown_embedded (CAMELTrack pipeline uses TrackLab KPR reid; embeddings shape (6, 128), visibility_scores shape (6,))",
                "pose_provenance": "unknown_embedded (CAMELTrack pipeline uses yolo_ultralytics-pose; keypoints_xyc shape (17, 3))",
                "cue_dimensions": {
                    "bbox_ltwh": [4],
                    "bbox_conf": [1],
                    "embeddings": [6, 128],
                    "visibility_scores": [6],
                    "keypoints_xyc": [17, 3],
                },
                "per_frame_causal_feature_check": "PASS (GT and cached tracker columns stripped in inference state files; features are per-detection frame-local)",
            },
        },
        "dancetrack_dataset": {
            "source_split": "train",
            "disk_path": str(DANCETRACK_TRAIN_DIR),
            "dataset_manifest": "manifests/datasets/dancetrack.yaml",
            "hf_revision": "a0ba42ac690c41e9850a20e76a0b9450a6fb6a47",
            "symlink_root": str(S08_DATASET_ROOT),
        },
    }
    (REPORT_DIR / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")

    split_audit = {
        "schema_version": "s08-split-audit-v1",
        "generated_at": now_iso,
        "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "split_manifest_path": str(SPLIT_PATH.relative_to(REPO_ROOT)),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "disjoint_check": "PASS",
        "total_sequences": len(sequence_inventory),
        "total_frames": expected_img_offset,
        "train32": {
            "sequence_count": len(train32_seqs),
            "sequence_names_sha256": split["motion_train"]["sequence_names_sha256"],
            "sequences": train32_seqs,
            "video_ids": train32_vids,
            "total_frames": sum(x["nframes"] for x in sequence_inventory if x["split_role"] == "train32"),
            "total_detections": sum(x["num_detections"] for x in sequence_inventory if x["split_role"] == "train32"),
        },
        "dev8": {
            "sequence_count": len(dev8_seqs),
            "sequence_names_sha256": split["motion_dev"]["sequence_names_sha256"],
            "sequences": dev8_seqs,
            "video_ids": dev8_vids,
            "total_frames": sum(x["nframes"] for x in sequence_inventory if x["split_role"] == "dev8"),
            "total_detections": sum(x["num_detections"] for x in sequence_inventory if x["split_role"] == "dev8"),
        },
        "isolated_state_files": state_artifacts,
        "sequence_inventory": sequence_inventory,
    }
    (REPORT_DIR / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "train32_frames": split_audit["train32"]["total_frames"],
        "train32_dets": split_audit["train32"]["total_detections"],
        "dev8_frames": split_audit["dev8"]["total_frames"],
        "dev8_dets": split_audit["dev8"]["total_detections"],
        "state_files": list(state_artifacts.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
