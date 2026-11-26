#!/usr/bin/env python3
"""S09 CAMELTrack threshold calibration, two-stage baseline, and closed-loop attribution runner."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import trackeval
import yaml
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import linear_sum_assignment

import tracklab.main  # noqa: F401
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb as tl_wandb

from cameltrack.architecture.base_module import Module as CAMELBaseModule
from cameltrack.cameltrack import Tracklet
from cameltrack.train.callbacks import SimMetrics
from cameltrack.train.dataset import CAMELDataset

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml"
SPLIT_PATH = REPO_ROOT / "manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml"
REPORT_DIR = REPO_ROOT / "reports/diagnostics/s09_camel_calibration_and_attribution_20261126"
RUNS_ROOT = DATA_ROOT / "runs/s09_camel_calibration_and_attribution_20261126"

S08_DATASET_ROOT = DATA_ROOT / "cache/s08_camel/dancetrack_s08_dataset"
S08_STATES_DIR = DATA_ROOT / "cache/s08_camel/states"
PRETRAINED_CKPT = DATA_ROOT / "weights/camel/camel_bbox_app_kps_dancetrack.ckpt"
S08_T_SCRATCH_CKPT = DATA_ROOT / "runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt"

ALLOWED_PHYSICAL_GPUS = {"1", "2", "3", "4", "5", "6", "7"}
TE_CUES = ("app_encoder", "kp_encoder", "bbox_encoder")


def apply_s09_runtime_compat_patches() -> None:
    """Apply zero-semantic-change compatibility and pickle-cache patches at runtime."""
    # 1. PyTorch 2.6 weights_only=False compatibility for partial TE checkpoint loading
    def _patched_init_weights(self: CAMELBaseModule, checkpoint_path: str = None, module_name: str = None) -> None:
        if checkpoint_path and module_name:
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
            ckpt_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            raw_sd = ckpt_obj["state_dict"]
            prefix = module_name + "."
            state_dict = OrderedDict(
                (k[len(prefix):], v) for k, v in raw_sd.items() if k.startswith(prefix)
            )
            if not state_dict:
                raise RuntimeError(f"No keys matching prefix {prefix!r} in {checkpoint_path}")
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Partial checkpoint key mismatch for {module_name} from {checkpoint_path}: "
                    f"missing={missing}, unexpected={unexpected}"
                )

    CAMELBaseModule.init_weights = _patched_init_weights

    # 2. Instance-local DataFrame cache in CAMELDataset (avoids zip re-decompression thrashing; preserves exact DataFrames)
    def _cached_load_pickle(self: CAMELDataset, sample_video_id: str) -> pd.DataFrame:
        cache = getattr(self, "_s09_pkl_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_s09_pkl_cache", cache)
        if sample_video_id not in cache:
            with self.zf.open(sample_video_id, "r") as fp:
                cache[sample_video_id] = pickle.load(fp)
        return cache[sample_video_id]

    CAMELDataset._load_pickle = _cached_load_pickle


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def enforce_gpu_safety(require_gpu: bool) -> dict[str, Any]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if require_gpu:
        parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
        if len(parts) != 1 or parts[0] not in ALLOWED_PHYSICAL_GPUS:
            raise RuntimeError(
                f"GPU safety violation: CUDA_VISIBLE_DEVICES={cvd!r}. "
                f"Must be a single physical GPU in {sorted(ALLOWED_PHYSICAL_GPUS)} (GPU 0 is forbidden)."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")
        torch.cuda.init()
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        return {
            "CUDA_VISIBLE_DEVICES": cvd,
            "physical_gpu_ids": [int(parts[0])],
            "logical_to_physical": {"cuda:0": f"physical_gpu_{parts[0]}"},
            "gpu_name": torch.cuda.get_device_name(0),
        }
    return {
        "CUDA_VISIBLE_DEVICES": cvd,
        "physical_gpu_ids": [],
        "logical_to_physical": {},
        "gpu_name": "CPU",
    }


def setup_offline_wandb(run_dir: Path) -> None:
    for sub in ("wandb", "wandb_cache", "wandb_data", "wandb_config"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DIR"] = str(run_dir / "wandb")
    os.environ["WANDB_CACHE_DIR"] = str(run_dir / "wandb_cache")
    os.environ["WANDB_DATA_DIR"] = str(run_dir / "wandb_data")
    os.environ["WANDB_CONFIG_DIR"] = str(run_dir / "wandb_config")


def load_split_lists() -> tuple[list[str], list[str]]:
    split = yaml.safe_load(SPLIT_PATH.read_text())
    return list(split["motion_train"]["sequences"]), list(split["motion_dev"]["sequences"])


def compose_s09_infer_config(
    run_dir: Path,
    train_vids: list[str],
    val_vids: list[str],
    val_state: Path,
    checkpoint_path: Path,
    sim_threshold: float,
    num_workers: int = 4,
) -> DictConfig:
    overrides = [
        "dataset=dancetrack",
        "visualization=none",
        "pipeline=[track]",
        f"dataset.dataset_path={S08_DATASET_ROOT}",
        "dataset.eval_set=val",
        "dataset.nframes=-1",
        f"dataset.vids_dict.train={json.dumps(train_vids)}",
        f"dataset.vids_dict.val={json.dumps(val_vids)}",
        "dataset.vids_dict.test=[]",
        f"state.load_file={val_state}",
        "state.save_file=null",
        "state.load_from_groundtruth=false",
        "state.load_from_public_dets=false",
        "modules.track.training_enabled=false",
        f"modules.track.checkpoint_path={checkpoint_path}",
        "modules.track.min_det_conf=0.4",
        "modules.track.min_init_det_conf=0.9",
        "modules.track.min_num_hits=0",
        "modules.track.max_wo_hits=150",
        "modules.track.max_track_gallery_size=50",
        f"modules.track.override_camel_cfg.sim_threshold={repr(float(sim_threshold))}",
        "+modules.track.override_camel_cfg.use_computed_sim_threshold=false",
        f"modules.track.CAMEL.sim_threshold={repr(float(sim_threshold))}",
        "modules.track.CAMEL.use_computed_sim_threshold=false",
        "eval_tracking=true",
        "test_tracking=true",
        "use_wandb=false",
        "use_rich=false",
        f"num_cores={num_workers}",
        f"project_dir={run_dir}",
    ]
    with initialize_config_module(version_base=None, config_module="tracklab.configs"):
        return compose(config_name="cameltrack", overrides=overrides)


def compose_s09_train_config(
    run_dir: Path,
    train_vids: list[str],
    val_vids: list[str],
    train_state: Path,
    val_state: Path,
    dm_cache_dir: Path,
    stage1_cue: str | None = None,
    te_checkpoints: dict[str, Path] | None = None,
    max_epochs: int = 20,
    num_workers: int = 4,
) -> DictConfig:
    overrides = [
        "dataset=dancetrack",
        "visualization=none",
        f"dataset.dataset_path={S08_DATASET_ROOT}",
        "dataset.eval_set=val",
        "dataset.nframes=-1",
        f"dataset.vids_dict.train={json.dumps(train_vids)}",
        f"dataset.vids_dict.val={json.dumps(val_vids)}",
        "dataset.vids_dict.test=[]",
        f"state.load_file={val_state}",
        "state.save_file=null",
        "modules.track.training_enabled=true",
        "modules.track.checkpoint_path=null",
        "modules.track.CAMEL.gaffe.checkpoint_path=null",
        "modules.track.CAMEL.sim_threshold=0.1",
        "modules.track.CAMEL.use_computed_sim_threshold=false",
        "modules.track.CAMEL.optimizer.init_lr=0.0001",
        "modules.track.CAMEL.optimizer.weight_decay=0.0001",
        f"modules.track.train_cfg.pl_trainer.max_epochs={max_epochs}",
        "modules.track.train_cfg.pl_trainer.precision=32",
        "modules.track.train_cfg.pl_trainer.accumulate_grad_batches=1",
        "modules.track.train_cfg.pl_trainer.enable_progress_bar=false",
        "modules.track.train_cfg.pl_trainer.log_every_n_steps=50",
        "modules.track.train_cfg.pl_trainer.check_val_every_n_epochs=1",
        "modules.track.train_cfg.model_selection_criteria=last",
        "modules.track.datamodule_cfg.batch_size=8",
        "modules.track.datamodule_cfg.num_samples=32",
        "modules.track.datamodule_cfg.samples_per_video=100",
        "modules.track.datamodule_cfg.max_length=50",
        "modules.track.datamodule_cfg.train_add_val=false",
        "modules.track.datamodule_cfg.multi_dataset_training=false",
        f"modules.track.datamodule_cfg.num_workers={num_workers}",
        f"modules.track.datamodule_cfg.path={dm_cache_dir}",
        f"modules.track.datamodule_cfg.tracker_states.train={train_state}",
        f"modules.track.datamodule_cfg.tracker_states.val={val_state}",
        "test_tracking=false",
        "eval_tracking=false",
        "use_wandb=false",
        "use_rich=false",
        f"num_cores={num_workers}",
        f"project_dir={run_dir}",
    ]
    if stage1_cue is not None:
        if stage1_cue not in TE_CUES:
            raise ValueError(f"Unknown cue: {stage1_cue}")
        overrides.append("modules.track.CAMEL.gaffe._target_=cameltrack.architecture.Identity")
        for cue in TE_CUES:
            if cue == stage1_cue:
                overrides.append(f"modules.track.CAMEL.temporal_encoders.{cue}.checkpoint_path=null")
            else:
                overrides.append(f"~modules.track.CAMEL.temporal_encoders.{cue}")
    else:
        for cue in TE_CUES:
            ckpt_p = te_checkpoints[cue] if te_checkpoints and cue in te_checkpoints else None
            val_s = str(ckpt_p) if ckpt_p is not None else "null"
            overrides.append(f"modules.track.CAMEL.temporal_encoders.{cue}.checkpoint_path={val_s}")

    with initialize_config_module(version_base=None, config_module="tracklab.configs"):
        return compose(config_name="cameltrack_train", overrides=overrides)


def param_summary(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, torch.nn.Module] = {"total": model, "gaffe": model.gaffe}
    for k, v in model.temp_encs.items():
        groups[f"temp_encs.{k}"] = v
    return {
        k: {
            "total_params": sum(p.numel() for p in mod.parameters()),
            "trainable_params": sum(p.numel() for p in mod.parameters() if p.requires_grad),
        }
        for k, mod in groups.items()
    }


def clone_module_state(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    res: dict[str, dict[str, torch.Tensor]] = {}
    prefixes = ["gaffe"] + [f"temp_encs.{k}" for k in model.temp_encs.keys()]
    for prefix in prefixes:
        res[prefix] = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if name.startswith(prefix + ".")
        }
    return res


def module_update_deltas(
    before: dict[str, dict[str, torch.Tensor]], model: torch.nn.Module
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    current = dict(model.named_parameters())
    for prefix, ref_dict in before.items():
        max_abs = 0.0
        l2_sq = 0.0
        n_changed = 0
        for name, ref_t in ref_dict.items():
            cur_t = current[name].detach().cpu()
            diff = (cur_t - ref_t).abs()
            m = float(diff.max().item())
            if m > 0.0:
                n_changed += 1
            max_abs = max(max_abs, m)
            l2_sq += float((diff.double() ** 2).sum().item())
        deltas[prefix] = {
            "num_tensors": len(ref_dict),
            "num_changed_tensors": n_changed,
            "max_abs_delta": max_abs,
            "l2_delta": float(np.sqrt(l2_sq)),
        }
    return deltas


def cmd_stage_a_lock(_args: argparse.Namespace) -> None:
    apply_s09_runtime_compat_patches()
    gpu_info = enforce_gpu_safety(require_gpu=False)
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    train32, dev8 = load_split_lists()

    # Verify S08 reused assets SHA256
    expected_hashes = {
        "p_official_ckpt": (PRETRAINED_CKPT, protocol["reused_s08_assets"]["p_official_ckpt"]["sha256"]),
        "t_scratch_ckpt": (S08_T_SCRATCH_CKPT, protocol["reused_s08_assets"]["t_scratch_ckpt"]["sha256"]),
        "split_manifest": (SPLIT_PATH, protocol["data"]["membership_sha256"]),
        "states_train32": (S08_STATES_DIR / "states_train32.pklz", None),
        "states_dev8_train_ref": (S08_STATES_DIR / "states_dev8_train_ref.pklz", None),
        "states_dev8_gt_free": (S08_STATES_DIR / "states_dev8_gt_free.pklz", None),
    }
    s08_split_audit = json.loads((REPO_ROOT / "reports/baselines/s08_camel_baseline_20261125/split_audit.json").read_text())
    expected_hashes["states_train32"] = (S08_STATES_DIR / "states_train32.pklz", s08_split_audit["isolated_state_files"]["train32_with_gt"]["sha256"])
    expected_hashes["states_dev8_train_ref"] = (S08_STATES_DIR / "states_dev8_train_ref.pklz", s08_split_audit["isolated_state_files"]["dev8_train_ref"]["sha256"])
    expected_hashes["states_dev8_gt_free"] = (S08_STATES_DIR / "states_dev8_gt_free.pklz", s08_split_audit["isolated_state_files"]["dev8_gt_free"]["sha256"])

    verified_assets = {}
    for k, (path, exp_sha) in expected_hashes.items():
        actual_sha = sha256_file(path)
        if actual_sha != exp_sha:
            raise RuntimeError(f"Asset SHA256 mismatch for {k} ({path}): {actual_sha} != {exp_sha}")
        verified_assets[k] = {"path": str(path), "sha256": actual_sha, "bytes": path.stat().st_size, "verified": True}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dir = REPORT_DIR / "resolved_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    dummy_run = RUNS_ROOT / "_resolved_placeholder"

    # Resolve Stage B config
    cfg_b = compose_s09_infer_config(
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=S08_T_SCRATCH_CKPT,
        sim_threshold=0.50,
    )
    (cfg_dir / "stage_b_t_scratch_th050.yaml").write_text(OmegaConf.to_yaml(cfg_b, resolve=True))

    # Resolve Stage C1 configs (3 single-cue TEs)
    c1_params = {}
    for cue in TE_CUES:
        cfg_c1 = compose_s09_train_config(
            run_dir=dummy_run,
            train_vids=train32,
            val_vids=dev8,
            train_state=S08_STATES_DIR / "states_train32.pklz",
            val_state=S08_STATES_DIR / "states_dev8_train_ref.pklz",
            dm_cache_dir=DATA_ROOT / f"cache/s09_camel/dm_c1_{cue}",
            stage1_cue=cue,
            max_epochs=20,
        )
        (cfg_dir / f"stage_c1_{cue}.yaml").write_text(OmegaConf.to_yaml(cfg_c1, resolve=True))
        m_c1 = instantiate(cfg_c1.modules.track.CAMEL, _recursive_=False)
        c1_params[cue] = param_summary(m_c1)

    # Resolve Stage C2 config
    placeholder_te_ckpts = {cue: RUNS_ROOT / f"c1_{cue}_seed13_e20_v1/CAMEL/last.ckpt" for cue in TE_CUES}
    cfg_c2 = compose_s09_train_config(
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=dev8,
        train_state=S08_STATES_DIR / "states_train32.pklz",
        val_state=S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=DATA_ROOT / "cache/s09_camel/dm_c2_joint",
        stage1_cue=None,
        te_checkpoints=placeholder_te_ckpts,
        max_epochs=20,
    )
    (cfg_dir / "stage_c2_twostage_joint.yaml").write_text(OmegaConf.to_yaml(cfg_c2, resolve=True))

    cfg_c3 = compose_s09_infer_config(
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=RUNS_ROOT / "c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt",
        sim_threshold=0.50,
    )
    (cfg_dir / "stage_c3_twostage_th050.yaml").write_text(OmegaConf.to_yaml(cfg_c3, resolve=True))

    source_audit = {
        "schema_version": "s09-source-audit-v1",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "upstream_cameltrack": {
            "repository": "https://github.com/TrackingLaboratory/CAMELTrack",
            "commit": "bc1c261075b0beacfe71c2896453be83643153c8",
            "stage1_te_pretraining_resolution": (
                "No separate Stage 1 YAML exists in upstream CAMELTrack; README 'Reproducing the paper results' "
                "specifies keeping camel.yaml defaults with max_epochs=20, gaffe._target_=cameltrack.architecture.Identity, "
                "and a single encoder under temporal_encoders (train_add_val=false for S09 motion_train32)."
            ),
            "partial_te_loading_resolution": (
                "Upstream cameltrack/architecture/base_module.py:22 calls torch.load(checkpoint_path) without weights_only=False, "
                "which fails under PyTorch 2.6.0. S09 patches Module.init_weights at runtime with weights_only=False and "
                "strict missing==[] & unexpected==[] key verification without editing upstream repo files."
            ),
            "dataloader_cache_resolution": (
                "Upstream CAMELDataset._load_pickle uses lru_cache(maxsize=1), thrashing zip decompression across 32 videos. "
                "S09 patches _load_pickle to cache per-video DataFrames on the dataset instance (zero sampling/math/RNG change)."
            ),
        },
        "reused_s08_assets_verified": verified_assets,
        "c1_model_parameters": c1_params,
    }
    (REPORT_DIR / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")

    pip_freeze = subprocess.check_output(
        ["/home/imc/yzy/agent/.tools/bin/uv", "pip", "freeze", "--python", "/media/imc/data/mot-data/envs/s08-camel/bin/python"],
        text=True,
    ).splitlines()
    df_out = subprocess.check_output(["df", "-B1", str(DATA_ROOT)], text=True).splitlines()[1].split()
    disk_free_gib = int(df_out[3]) / (1024 ** 3)
    nvidia_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()

    runtime_lock = {
        "schema_version": "s09-runtime-lock-v1",
        "frozen_at": dt.datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "uv_environment": "/media/imc/data/mot-data/envs/s08-camel",
        "repo_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_cameltrack_commit": subprocess.check_output(["git", "-C", str(DATA_ROOT / "third_party/CAMELTrack"), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_cameltrack_dirty": bool(subprocess.check_output(["git", "-C", str(DATA_ROOT / "third_party/CAMELTrack"), "status", "--porcelain"], text=True).strip()),
        "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "scripts_sha256": {
            "scripts/run_s09_camel.py": sha256_file(REPO_ROOT / "scripts/run_s09_camel.py"),
        },
        "all_packages_count": len(pip_freeze),
        "all_packages": pip_freeze,
        "gpu_inventory": nvidia_out,
        "gpu_safety_policy": {
            "forbidden_physical_gpus": [0],
            "allowed_physical_gpus": [1, 2, 3, 4, 5, 6, 7],
            "check_during_lock": gpu_info,
        },
        "storage": {"data_root": str(DATA_ROOT), "free_gib_before_runs": round(disk_free_gib, 3)},
    }
    (REPORT_DIR / "runtime_lock.json").write_text(json.dumps(runtime_lock, indent=2) + "\n")

    commands_sh = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
SCRIPT=/home/imc/yzy/MOT/scripts/run_s09_camel.py

# Stage A: asset verification, runtime lock & unit tests
CUDA_VISIBLE_DEVICES='' $PY $SCRIPT stage-a-lock
CUDA_VISIBLE_DEVICES='' $PY -m pytest /home/imc/yzy/MOT/tests/test_s09_camel.py -q

# Stage B: Zero-retrain threshold calibration on dev8 (4 runs)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_SCRATCH_TH046 --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.46 --seed 13 --run-id b_t_scratch_th046_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_SCRATCH_TH050 --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --seed 13 --run-id b_t_scratch_th050_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_SCRATCH_TH_COMP --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.5197290778160095 --seed 13 --run-id b_t_scratch_th_comp_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition P_OFFICIAL_TH050 --ckpt /media/imc/data/mot-data/weights/camel/camel_bbox_app_kps_dancetrack.ckpt --sim-threshold 0.50 --seed NA --run-id b_p_official_th050_v1

# Stage C1: Single-cue TE pretraining on motion_train32 (3 runs)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c1 --cue app_encoder --run-id c1_app_encoder_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c1 --cue kp_encoder --run-id c1_kp_encoder_seed13_e20_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c1 --cue bbox_encoder --run-id c1_bbox_encoder_seed13_e20_v1

# Stage C2: Two-stage joint GAFFE training on motion_train32
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT train-c2 --run-id c2_twostage_joint_seed13_e20_v1 --app-ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c1_app_encoder_seed13_e20_v1/CAMEL/last.ckpt --kp-ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c1_kp_encoder_seed13_e20_v1/CAMEL/last.ckpt --bbox-ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c1_bbox_encoder_seed13_e20_v1/CAMEL/last.ckpt

# Stage C3: T_TWOSTAGE dev8 closed-loop eval (0.50 and epoch20 computed_sim_threshold)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_TWOSTAGE_TH050 --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --seed 13 --run-id c3_t_twostage_th050_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT infer-eval --condition T_TWOSTAGE_TH_COMP --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold-from-summary /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/stage_c2_summary.json --seed 13 --run-id c3_t_twostage_th_comp_v1

# Stage D1: Similarity margin trace on T_TWOSTAGE_TH050 and T_SCRATCH_TH_COMP
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d1-trace --condition T_TWOSTAGE_TH050 --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --ref-run-id c3_t_twostage_th050_v1 --run-id d1_t_twostage_th050_trace_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d1-trace --condition T_SCRATCH_TH_COMP --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.5197290778160095 --ref-run-id b_t_scratch_th_comp_v1 --run-id d1_t_scratch_th_comp_trace_v1

# Stage D2: Privileged GT-routed clean history probe on T_TWOSTAGE_TH050
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d2-clean-history --ckpt /media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt --sim-threshold 0.50 --run-id d2_t_twostage_th050_clean_history_v3

# Stage E: Standard report export
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s09_report.py
"""
    (REPORT_DIR / "commands.sh").write_text(commands_sh)
    os.chmod(REPORT_DIR / "commands.sh", 0o755)
    print(json.dumps({"status": "PASS", "verified_assets": list(verified_assets.keys()), "c1_params": c1_params}, indent=2))


class S09EpochLogger(pl.Callback):
    def __init__(self, run_dir: Path, stage_tag: str, t_start: float) -> None:
        self.run_dir = run_dir
        self.stage_tag = stage_tag
        self.t_start = t_start
        self.rows: list[dict[str, Any]] = []
        self.nonfinite_grad_steps = 0

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        for p in pl_module.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self.nonfinite_grad_steps += 1
                break

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_num = int(trainer.current_epoch) + 1
        elapsed = time.perf_counter() - self.t_start
        metrics = trainer.callback_metrics
        lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        step = int(trainer.global_step)
        samples_seen = step * 8
        ckpt_id = "epoch20_last" if epoch_num == trainer.max_epochs else ""
        for metric_key, unit in (
            ("train/loss_epoch", "loss"),
            ("val/loss", "loss"),
            ("val/sim_acc", "ratio_0_1"),
            ("val/sim_auroc", "ratio_0_1"),
            ("val/computed_sim_threshold", "similarity"),
        ):
            if metric_key in metrics:
                val = float(metrics[metric_key].item())
                split_name = "train" if metric_key.startswith("train/") else "val"
                self.rows.append({
                    "run_id": self.run_dir.name,
                    "stage_tag": self.stage_tag,
                    "seed": 13,
                    "epoch": epoch_num,
                    "step": step,
                    "split": split_name,
                    "metric": metric_key,
                    "value": repr(val),
                    "unit": unit,
                    "samples_seen": samples_seen,
                    "elapsed_seconds": round(elapsed, 3),
                    "checkpoint_id": ckpt_id,
                })
        self.rows.append({
            "run_id": self.run_dir.name,
            "stage_tag": self.stage_tag,
            "seed": 13,
            "epoch": epoch_num,
            "step": step,
            "split": "train",
            "metric": "lr",
            "value": repr(lr),
            "unit": "lr",
            "samples_seen": samples_seen,
            "elapsed_seconds": round(elapsed, 3),
            "checkpoint_id": ckpt_id,
        })
        self.write_curves()

    def write_curves(self) -> None:
        out_csv = self.run_dir / "learning_curves.csv"
        fields = [
            "run_id", "stage_tag", "seed", "epoch", "step", "split",
            "metric", "value", "unit", "samples_seen", "elapsed_seconds", "checkpoint_id",
        ]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)


def run_training_job(
    cfg: DictConfig,
    run_dir: Path,
    stage_tag: str,
    gpu_info: dict[str, Any],
    summary_filename: str,
    extra_meta: dict[str, Any],
) -> None:
    t0 = time.perf_counter()
    pl.seed_everything(13, workers=True)
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL
    init_snapshot = clone_module_state(model)
    params_info = param_summary(model)

    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})

    epoch_cb = S09EpochLogger(run_dir, stage_tag, t0)
    save_best_loss = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        dirpath="CAMEL",
        mode="min",
        filename="epoch={epoch}-loss={val/loss:.4f}",
        auto_insert_metric_name=False,
        save_last=True,
    )
    logger = pl.loggers.WandbLogger(project="CAMEL", offline=True, save_dir=str(run_dir))
    tr_cfg = cfg.modules.track.train_cfg.pl_trainer
    trainer = pl.Trainer(
        max_epochs=int(tr_cfg.max_epochs),
        logger=logger,
        callbacks=[save_best_loss, pl.callbacks.LearningRateMonitor(), SimMetrics(), epoch_cb],
        accelerator="cuda",
        devices=1,
        precision=int(tr_cfg.precision),
        gradient_clip_val=float(tr_cfg.gradient_clip_val),
        accumulate_grad_batches=int(tr_cfg.accumulate_grad_batches),
        num_sanity_val_steps=0,
        log_every_n_steps=int(tr_cfg.log_every_n_steps),
        check_val_every_n_epoch=int(tr_cfg.check_val_every_n_epochs),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    last_ckpt = run_dir / "CAMEL/last.ckpt"
    assert last_ckpt.is_file(), f"Missing {last_ckpt}"
    last_sha = sha256_file(last_ckpt)

    deltas = module_update_deltas(init_snapshot, model)
    for mod_name, d in deltas.items():
        if d["num_tensors"] > 0 and d["num_changed_tensors"] == 0:
            raise RuntimeError(f"Module {mod_name} had zero parameter updates!")

    # Extract epoch20 val/computed_sim_threshold from learning curves
    epoch20_comp_th = None
    for row in reversed(epoch_cb.rows):
        if row["metric"] == "val/computed_sim_threshold":
            epoch20_comp_th = float(row["value"])
            break

    # New-process style explicit reload check on CPU/CUDA in-memory
    reloaded = type(model).load_from_checkpoint(last_ckpt, map_location="cuda")
    sd_cur = model.state_dict()
    sd_rel = reloaded.state_dict()
    max_reload_diff = max(float((sd_cur[k].cpu() - sd_rel[k].cpu()).abs().max().item()) for k in sd_cur)
    assert max_reload_diff == 0.0, f"Reload state_dict diff {max_reload_diff} != 0"

    summary = {
        "schema_version": "s09-train-v1",
        "status": "COMPLETED",
        "stage_tag": stage_tag,
        "run_id": run_dir.name,
        "seed": 13,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "gpu_device_hours": (t1 - t0) / 3600.0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "completed_epochs": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "nonfinite_grad_steps": epoch_cb.nonfinite_grad_steps,
        "parameters": params_info,
        "module_deltas_init_to_end": deltas,
        "last_checkpoint": {
            "path": str(last_ckpt),
            "sha256": last_sha,
            "bytes": last_ckpt.stat().st_size,
            "reload_state_dict_max_abs_diff": max_reload_diff,
        },
        "epoch20_val_computed_sim_threshold": epoch20_comp_th,
        "learning_curves_csv": str(run_dir / "learning_curves.csv"),
        **extra_meta,
    }
    (run_dir / summary_filename).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "stage_tag": stage_tag,
        "epochs": summary["completed_epochs"],
        "global_step": summary["global_step"],
        "wall_seconds": round(summary["wall_seconds"], 2),
        "peak_vram_mib": round(summary["peak_vram_mib"], 2),
        "last_ckpt_sha256": last_sha,
        "epoch20_val_computed_sim_threshold": epoch20_comp_th,
    }, indent=2))


def cmd_train_c1(args: argparse.Namespace) -> None:
    apply_s09_runtime_compat_patches()
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    train32, dev8 = load_split_lists()
    dm_cache = DATA_ROOT / f"cache/s09_camel/dm_{args.run_id}_isolated"
    cfg = compose_s09_train_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        train_state=S08_STATES_DIR / "states_train32.pklz",
        val_state=S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=dm_cache,
        stage1_cue=args.cue,
        max_epochs=20,
        num_workers=4,
    )
    run_training_job(
        cfg=cfg,
        run_dir=run_dir,
        stage_tag=f"C1_{args.cue}",
        gpu_info=gpu_info,
        summary_filename="stage_c1_summary.json",
        extra_meta={"cue": args.cue, "datamodule_cache_dir": str(dm_cache)},
    )


def cmd_train_c2(args: argparse.Namespace) -> None:
    apply_s09_runtime_compat_patches()
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    te_ckpts = {
        "app_encoder": Path(args.app_ckpt),
        "kp_encoder": Path(args.kp_ckpt),
        "bbox_encoder": Path(args.bbox_ckpt),
    }
    te_meta = {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in te_ckpts.items()}

    train32, dev8 = load_split_lists()
    dm_cache = DATA_ROOT / f"cache/s09_camel/dm_{args.run_id}_isolated"
    cfg = compose_s09_train_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        train_state=S08_STATES_DIR / "states_train32.pklz",
        val_state=S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=dm_cache,
        stage1_cue=None,
        te_checkpoints=te_ckpts,
        max_epochs=20,
        num_workers=4,
    )
    run_training_job(
        cfg=cfg,
        run_dir=run_dir,
        stage_tag="C2_joint",
        gpu_info=gpu_info,
        summary_filename="stage_c2_summary.json",
        extra_meta={"te_pretrained_checkpoints": te_meta, "datamodule_cache_dir": str(dm_cache)},
    )


def load_camel_model_explicit(cfg: DictConfig, ckpt_path: Path, sim_threshold: float) -> Any:
    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=instantiate(cfg.dataset))
    override_cfg = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    override_cfg["sim_threshold"] = float(sim_threshold)
    override_cfg["use_computed_sim_threshold"] = False
    track_mod.CAMEL = type(track_mod.CAMEL).load_from_checkpoint(
        ckpt_path, map_location=track_mod.device, **override_cfg
    )
    assert abs(float(track_mod.CAMEL.sim_threshold) - float(sim_threshold)) < 1e-12
    assert bool(track_mod.CAMEL.use_computed_sim_threshold) is False
    return track_mod


def cmd_infer_eval(args: argparse.Namespace) -> None:
    apply_s09_runtime_compat_patches()
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    if args.sim_threshold_from_summary is not None:
        s_obj = json.loads(Path(args.sim_threshold_from_summary).read_text())
        sim_threshold = float(s_obj["epoch20_val_computed_sim_threshold"])
    elif args.sim_threshold is not None:
        sim_threshold = float(args.sim_threshold)
    else:
        raise ValueError("Must specify --sim-threshold or --sim-threshold-from-summary")

    ckpt_path = Path(args.ckpt)
    ckpt_sha = sha256_file(ckpt_path)
    train32, dev8 = load_split_lists()

    t0 = time.perf_counter()
    cfg = compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        sim_threshold=sim_threshold,
    )
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]

    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    override_cfg = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
    override_cfg["sim_threshold"] = sim_threshold
    override_cfg["use_computed_sim_threshold"] = False
    track_mod.CAMEL = type(track_mod.CAMEL).load_from_checkpoint(
        ckpt_path, map_location=track_mod.device, **override_cfg
    )
    assert abs(float(track_mod.CAMEL.sim_threshold) - sim_threshold) < 1e-12
    assert bool(track_mod.CAMEL.use_computed_sim_threshold) is False

    Tracklet.count = 1
    pipeline = Pipeline(models=[track_mod])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    torch.cuda.synchronize()
    t_track0 = time.perf_counter()
    tracking_engine.track_dataset()
    torch.cuda.synchronize()
    t_track1 = time.perf_counter()
    peak_vram_mib = float(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    mot_dir = run_dir / "mot_pred"
    tracking_dataset.save_for_eval(
        tracker_state.detections_pred,
        tracker_state.image_metadatas,
        tracker_state.video_metadatas,
        str(mot_dir),
        "bbox_ltwh",
        False,
        is_ground_truth=False,
    )
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
    t_eval0 = time.perf_counter()
    evaluator.run(tracker_state)
    t_eval1 = time.perf_counter()
    t1 = time.perf_counter()

    detailed_csv = run_dir / "eval/results/tracklab/pedestrian_detailed.csv"
    assert detailed_csv.is_file(), f"Missing {detailed_csv}"
    with detailed_csv.open(newline="") as f:
        rows = {r["seq"]: r for r in csv.DictReader(f)}
    assert set(rows.keys()) == (set(dev8) | {"COMBINED"})

    mot_hashes = {s: sha256_file(mot_dir / f"{s}.txt") for s in dev8}
    summary = {
        "schema_version": "s09-eval-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "condition": args.condition,
        "seed": args.seed,
        "scope_id": "dev8",
        "sim_threshold": sim_threshold,
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "state_file": str(S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "state_file_sha256": sha256_file(S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "wall_seconds": t1 - t0,
        "tracking_seconds": t_track1 - t_track0,
        "eval_seconds": t_eval1 - t_eval0,
        "peak_vram_mib": peak_vram_mib,
        "mot_dir": str(mot_dir),
        "mot_sha256_by_seq": mot_hashes,
        "eval_summary": {
            "detailed_csv": str(detailed_csv),
            "summary_txt": str(run_dir / "eval/results/tracklab/pedestrian_summary.txt"),
            "gt_dir": str(run_dir / "eval/gt/DanceTrack-val"),
            "pred_dir": str(run_dir / "eval/pred/DanceTrack-val/tracklab"),
        },
        "combined_metrics_preview": {
            "HOTA": float(rows["COMBINED"]["HOTA___AUC"]) * 100.0,
            "DetA": float(rows["COMBINED"]["DetA___AUC"]) * 100.0,
            "AssA": float(rows["COMBINED"]["AssA___AUC"]) * 100.0,
            "IDF1": float(rows["COMBINED"]["IDF1"]) * 100.0,
            "MOTA": float(rows["COMBINED"]["MOTA"]) * 100.0,
            "IDSW": int(float(rows["COMBINED"]["IDSW"])),
            "FP": int(float(rows["COMBINED"]["CLR_FP"])),
            "FN": int(float(rows["COMBINED"]["CLR_FN"])),
            "Frag": int(float(rows["COMBINED"]["Frag"])),
        },
    }
    (run_dir / "stage_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "condition": args.condition,
        "sim_threshold": sim_threshold,
        "wall_seconds": round(summary["wall_seconds"], 2),
        "combined": summary["combined_metrics_preview"],
    }, indent=2))


def box_iou_ltwh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


def match_detections_to_gt_frame(
    det_bboxes: list[np.ndarray], gt_ids: np.ndarray, gt_bboxes: np.ndarray, iou_thresh: float = 0.5
) -> tuple[dict[int, int], dict[int, int]]:
    """Return (det_idx -> gt_id, gt_id -> det_idx) via 1-to-1 Hungarian matching >= iou_thresh."""
    if len(det_bboxes) == 0 or len(gt_ids) == 0:
        return {}, {}
    sim = np.zeros((len(gt_ids), len(det_bboxes)), dtype=float)
    for r, gbox in enumerate(gt_bboxes):
        for c, dbox in enumerate(det_bboxes):
            sim[r, c] = box_iou_ltwh(gbox, dbox)
    score = sim.copy()
    score[score < iou_thresh - np.finfo("float").eps] = 0.0
    rows, cols = linear_sum_assignment(-score)
    det_to_gt: dict[int, int] = {}
    gt_to_det: dict[int, int] = {}
    for r, c in zip(rows, cols):
        if score[r, c] > 0.0 + np.finfo("float").eps:
            gid = int(gt_ids[r])
            det_to_gt[int(c)] = gid
            gt_to_det[gid] = int(c)
    return det_to_gt, gt_to_det


class TraceRecordingCAMELTrack:
    """Wrapper around CAMELTrack for Stage D1 frame-level similarity matrix tracing (zero output change)."""

    def __init__(self, base_mod: Any, image_id_to_seq_frame: dict[int, tuple[str, int]]) -> None:
        self.base = base_mod
        self.image_id_to_seq_frame = image_id_to_seq_frame
        self.traces: dict[tuple[str, int], dict[str, Any]] = {}
        # Delegate TrackLab module metadata
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        for track in self.base.tracklets:
            track.forward()

        if len(self.base.tracklets) == 0 or len(detections) == 0:
            matched = np.empty((0, 2), dtype=int)
            unmatched_trks = list(range(len(self.base.tracklets)))
            unmatched_dets = list(range(len(detections)))
            td_sim_np = np.empty((len(self.base.tracklets), len(detections)), dtype=np.float32)
        else:
            batch = self.base.build_camel_batch(self.base.tracklets, detections)
            _, association_result, td_sim_matrix = self.base.CAMEL.predict_step(batch, self.base.frame_count)
            matched = np.asarray(association_result[0]["matched_td_indices"], dtype=int)
            unmatched_trks = list(association_result[0]["unmatched_trackers"])
            unmatched_dets = list(association_result[0]["unmatched_detections"])
            td_sim_np = td_sim_matrix.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Record frame state BEFORE updating tracklets / creating new ones
        img_id = int(metadatas.index[0]) if hasattr(metadatas, "index") else int(detections[0].image_id.item())
        seq_frame = self.image_id_to_seq_frame[img_id]
        trk_snapshot = [
            {
                "track_id": int(t.id),
                "history_len": int(len(t.detections)),
                "time_wo_hits": int(t.time_wo_hits),
                "state": str(t.state),
            }
            for t in self.base.tracklets
        ]
        det_snapshot = [
            {
                "index": int(d.index.item()),
                "bbox_ltwh": [float(x) for x in d.bbox_ltwh.detach().cpu().tolist()],
                "bbox_conf": float(d.bbox_conf.detach().cpu().item()),
            }
            for d in detections
        ]

        # Apply exact same Tracklet updates and output logic as CAMELTrack.process()
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.base.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        sim_tensor = torch.from_numpy(td_sim_np) if td_sim_np.size > 0 else torch.empty((0,))
        for m in matched:
            tracklet = self.base.tracklets[m[0]]
            detection = detections[m[1]]
            tracklet.update(detection)
            detection.similarity_with_tracklet = sim_tensor[m[0], m[1]]
            detection.similarities = sim_tensor[: len(self.base.tracklets), m[1]]

        new_track_ids_for_det: dict[int, int] = {}
        for i in unmatched_dets:
            if detections[i].bbox_conf >= self.base.min_init_det_conf:
                new_trk = Tracklet(detections[i], self.base.max_track_gallery_size)
                self.base.tracklets.append(new_trk)
                new_track_ids_for_det[int(i)] = int(new_trk.id)

        self.traces[seq_frame] = {
            "tracklets": trk_snapshot,
            "detections": det_snapshot,
            "sim_matrix": td_sim_np,
            "matched": matched.tolist(),
            "unmatched_trks": unmatched_trks,
            "unmatched_dets": unmatched_dets,
            "new_track_ids_for_det": new_track_ids_for_det,
        }

        actives = []
        for trk in self.base.tracklets:
            self.base.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                actives.append(
                    {
                        "index": trk.last_detection.index.item(),
                        "track_id": trk.id,
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.item())
                        if trk.last_detection.similarity_with_tracklet is not None
                        else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.base.tracklets[j].id: float(sim) for j, sim in enumerate(trk.last_detection.similarities.numpy())}
                            if trk.last_detection.similarities is not None
                            else None,
                            "St": self.base.CAMEL.sim_threshold,
                        },
                    }
                )

        self.base.tracklets = [trk for trk in self.base.tracklets if trk.state != "dead"]
        self.base.frame_count += 1

        if actives:
            results = pd.DataFrame(actives).set_index("index", drop=True)
            assert set(results.index).issubset(detections_df.index)
            return results
        return pd.DataFrame(columns=["track_id", "hits", "age", "matched_with", "time_since_update", "state", "costs"])


class CleanHistoryCAMELTrack:
    """Stage D2 privileged diagnostic: causal CAMEL prediction + GT-routed clean tracklet galleries."""

    def __init__(self, base_mod: Any, gt_by_image_id: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self.base = base_mod
        self.gt_by_image_id = gt_by_image_id
        self.input_columns = base_mod.input_columns
        self.output_columns = base_mod.output_columns
        self.level = base_mod.level
        self.batch_size = base_mod.batch_size
        self.collate_fn = base_mod.collate_fn
        self.routing_stats = {
            "total_frames": 0,
            "total_predicted_matches": 0,
            "cross_id_matches_prevented_from_gallery": 0,
            "fp_matches_prevented_from_gallery": 0,
            "gt_routed_gallery_updates": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset(self) -> None:
        self.base.reset()

    @torch.no_grad()
    def preprocess(self, image: Any, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return self.base.preprocess(image, detections, metadata)

    @torch.no_grad()
    def process(self, detections: list[Any], detections_df: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        self.routing_stats["total_frames"] += 1
        for track in self.base.tracklets:
            track.forward()

        matched, unmatched_trks, unmatched_dets, td_sim_matrix = self.base.associate_dets_to_trks(
            self.base.tracklets, detections
        )
        assert len(set([m[0] for m in matched.tolist()] + unmatched_trks)) == len(self.base.tracklets)
        assert len(set([m[1] for m in matched.tolist()] + unmatched_dets)) == len(detections)

        # Current-frame GT matching ONLY for clean gallery routing (never changes current-frame output!)
        img_id = int(metadatas.index[0]) if hasattr(metadatas, "index") else int(detections[0].image_id.item())
        gt_ids_f, gt_boxes_f = self.gt_by_image_id.get(img_id, (np.empty((0,), dtype=int), np.empty((0, 4), dtype=float)))
        det_boxes = [d.bbox_ltwh.detach().cpu().numpy().astype(float) for d in detections]
        det_to_gt, gt_to_det = match_detections_to_gt_frame(det_boxes, gt_ids_f, gt_boxes_f, iou_thresh=0.5)

        # Build current-frame emitted predictions strictly from CAMEL matching (before clean-gallery overwrite)
        emitted_rows: list[dict[str, Any]] = []
        matched_det_set = set()
        for m in matched:
            trk_idx, det_idx = int(m[0]), int(m[1])
            matched_det_set.add(det_idx)
            trk = self.base.tracklets[trk_idx]
            det = detections[det_idx]
            sim_val = float(td_sim_matrix[trk_idx, det_idx].cpu().item())
            all_sims = {self.base.tracklets[j].id: float(s) for j, s in enumerate(td_sim_matrix[: len(self.base.tracklets), det_idx].cpu().numpy())}
            self.routing_stats["total_predicted_matches"] += 1
            anchor_gt = getattr(trk, "s09_gt_anchor", None)
            det_gt = det_to_gt.get(det_idx)
            if anchor_gt is not None and det_gt is None:
                self.routing_stats["fp_matches_prevented_from_gallery"] += 1
            elif anchor_gt is not None and det_gt is not None and det_gt != anchor_gt:
                self.routing_stats["cross_id_matches_prevented_from_gallery"] += 1

            # Update emission lifecycle counters (CAMEL predicted match)
            trk.hits += 1
            trk.hit_streak += 1
            trk.time_wo_hits = 0
            trk.last_detection = det
            self.base.update_state(trk)
            if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                emitted_rows.append(
                    {
                        "index": det.index.item(),
                        "track_id": trk.id,
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", sim_val),
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {"S": all_sims, "St": self.base.CAMEL.sim_threshold},
                    }
                )

        # Spawn new tracklets for unmatched high-conf detections (strictly per CAMEL policy)
        new_tracklets: list[Any] = []
        for i in unmatched_dets:
            det = detections[int(i)]
            if det.bbox_conf >= self.base.min_init_det_conf:
                trk = Tracklet(det, self.base.max_track_gallery_size)
                trk.s09_gt_anchor = det_to_gt.get(int(i))
                new_tracklets.append(trk)
                self.base.update_state(trk)
                if (trk.state == "active") or (trk.state == "init" and self.base.frame_count < self.base.min_num_hits):
                    emitted_rows.append(
                        {
                            "index": det.index.item(),
                            "track_id": trk.id,
                            "hits": trk.hits,
                            "age": trk.age,
                            "matched_with": None,
                            "time_since_update": trk.time_wo_hits,
                            "state": trk.state,
                            "costs": {"S": None, "St": self.base.CAMEL.sim_threshold},
                        }
                    )

        # Update state for unmatched existing tracklets
        for u_idx in unmatched_trks:
            self.base.update_state(self.base.tracklets[int(u_idx)])

        # PRIVILEGED 1-TO-1 CLEAN GALLERY ROUTING:
        # For each GT ID present in this frame, route its detection (as an independent shallow copy so `.age`
        # is never shared/double-incremented across tracklets) to AT MOST ONE canonical existing tracklet anchored
        # to that GT ID. Unanchored tracklets update normally only on unanchored detections.
        matched_by_trk = {int(m[0]): int(m[1]) for m in matched}
        anchored_by_gt: dict[int, list[tuple[int, Any]]] = {}
        for idx_t, trk in enumerate(self.base.tracklets):
            anchor_gt = getattr(trk, "s09_gt_anchor", None)
            if anchor_gt is not None:
                anchored_by_gt.setdefault(int(anchor_gt), []).append((idx_t, trk))
            elif idx_t in matched_by_trk:
                det_i = matched_by_trk[idx_t]
                if det_i not in det_to_gt:
                    det_copy = copy.copy(detections[det_i])
                    if torch.is_tensor(detections[det_i].age):
                        det_copy.age = detections[det_i].age.clone().zero_()
                    else:
                        det_copy.age = 0
                    trk.detections.append(det_copy)
                    trk.detections = trk.detections[-trk.max_gallery_size:]

        for anchor_gt, candidates in anchored_by_gt.items():
            if anchor_gt not in gt_to_det:
                continue
            target_det_idx = gt_to_det[anchor_gt]
            clean_matched = [
                (idx_t, trk) for idx_t, trk in candidates if matched_by_trk.get(idx_t) == target_det_idx
            ]
            if clean_matched:
                _, chosen_trk = max(clean_matched, key=lambda x: (int(x[1].hits), len(x[1].detections), -int(x[1].id)))
            else:
                _, chosen_trk = max(candidates, key=lambda x: (int(x[1].hits), len(x[1].detections), -int(x[1].id)))
            det_copy = copy.copy(detections[target_det_idx])
            if torch.is_tensor(detections[target_det_idx].age):
                det_copy.age = detections[target_det_idx].age.clone().zero_()
            else:
                det_copy.age = 0
            chosen_trk.detections.append(det_copy)
            chosen_trk.detections = chosen_trk.detections[-chosen_trk.max_gallery_size:]
            self.routing_stats["gt_routed_gallery_updates"] += 1

        self.base.tracklets.extend(new_tracklets)
        self.base.tracklets = [trk for trk in self.base.tracklets if trk.state != "dead"]
        self.base.frame_count += 1

        if emitted_rows:
            results = pd.DataFrame(emitted_rows).set_index("index", drop=True)
            assert set(results.index).issubset(detections_df.index)
            return results
        return pd.DataFrame(columns=["track_id", "hits", "age", "matched_with", "time_since_update", "state", "costs"])


def analyze_idsw_with_traces(
    traces: dict[tuple[str, int], dict[str, Any]],
    gt_dir: Path,
    pred_dir: Path,
    detailed_csv: Path,
    dev8: list[str],
    seq_nframes: dict[str, int],
    sim_threshold: float,
    condition: str,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with detailed_csv.open(newline="") as f:
        official_rows = {r["seq"]: r for r in csv.DictReader(f)}

    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "BENCHMARK": "DanceTrack",
        "SPLIT_TO_EVAL": "val",
        "GT_FOLDER": str(gt_dir.parent),
        "TRACKERS_FOLDER": str(pred_dir.parent.parent),
        "TRACKERS_TO_EVAL": ["tracklab"],
        "TRACKER_SUB_FOLDER": "",
        "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt",
        "DO_PREPROC": False,
        "PRINT_CONFIG": False,
        "SEQ_INFO": seq_nframes,
    })
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)

    all_events: list[dict[str, Any]] = []
    for seq in dev8:
        expected_idsw = int(float(official_rows[seq]["IDSW"]))
        raw_data = dataset.get_raw_seq_data("tracklab", seq)
        data = dataset.get_preprocessed_seq_data(raw_data, "pedestrian")

        unique_gt_ids = []
        unique_tracker_ids = []
        for t in range(raw_data["num_timesteps"]):
            gt_keep = np.not_equal(raw_data["gt_extras"][t]["zero_marked"], 0)
            unique_gt_ids.extend(np.unique(raw_data["gt_ids"][t][gt_keep]).tolist())
            unique_tracker_ids.extend(np.unique(raw_data["tracker_ids"][t]).tolist())
        unique_gt_ids = np.unique(unique_gt_ids) if unique_gt_ids else np.array([], dtype=int)
        unique_tracker_ids = np.unique(unique_tracker_ids) if unique_tracker_ids else np.array([], dtype=int)

        num_gt_ids = data["num_gt_ids"]
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)
        last_matched_t = np.full(num_gt_ids, -1, dtype=int)
        seq_idsw = 0

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"])):
            frame_1 = t + 1
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                continue
            gt_dets_t = data["gt_dets"][t]
            trk_dets_t = data["tracker_dets"][t]
            similarity = data["similarity_scores"][t]
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < 0.5 - np.finfo("float").eps] = 0
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched = score_mat[match_rows, match_cols] > 0 + np.finfo("float").eps
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]
            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            for r, c, g_idx, trk_idx in zip(match_rows, match_cols, matched_gt_ids, matched_tracker_ids):
                prev_trk_idx = prev_tracker_id[g_idx]
                if not np.isnan(prev_trk_idx) and int(trk_idx) != int(prev_trk_idx):
                    seq_idsw += 1
                    orig_gt = int(unique_gt_ids[g_idx])
                    orig_prev = int(unique_tracker_ids[int(prev_trk_idx)])
                    orig_new = int(unique_tracker_ids[int(trk_idx)])
                    prev_frame_1 = int(last_matched_t[g_idx] + 1)
                    gap = int(frame_1 - prev_frame_1)
                    gt_box = np.asarray(gt_dets_t[r], dtype=float)
                    pred_box = np.asarray(trk_dets_t[c], dtype=float)

                    # Competing GT IoU >= 0.3
                    has_competing_gt = any(
                        j != r and box_iou_ltwh(gt_box, np.asarray(gt_dets_t[j], dtype=float)) >= 0.3
                        for j in range(len(gt_ids_t))
                    )

                    # Map to internal similarity trace at (seq, frame_1)
                    tr = traces[(seq, frame_1)]
                    trk_id_to_row = {t_info["track_id"]: row_i for row_i, t_info in enumerate(tr["tracklets"])}
                    # Locate assigned detection by closest bbox match (same saved float coordinates)
                    c_det = int(np.argmin([
                        np.max(np.abs(np.asarray(d["bbox_ltwh"], dtype=float) - pred_box))
                        for d in tr["detections"]
                    ]))
                    r_prev = trk_id_to_row.get(orig_prev)
                    r_new = trk_id_to_row.get(orig_new)
                    sim_mat = tr["sim_matrix"]

                    s_pos = float(sim_mat[r_prev, c_det]) if (r_prev is not None and sim_mat.size > 0) else None
                    s_assigned = float(sim_mat[r_new, c_det]) if (r_new is not None and sim_mat.size > 0) else None
                    prev_hist_len = tr["tracklets"][r_prev]["history_len"] if r_prev is not None else None
                    new_hist_len = tr["tracklets"][r_new]["history_len"] if r_new is not None else 0

                    if r_prev is not None and sim_mat.shape[0] > 1:
                        comp_scores = [float(sim_mat[ri, c_det]) for ri in range(sim_mat.shape[0]) if ri != r_prev]
                        s_best_comp = max(comp_scores) if comp_scores else None
                    elif sim_mat.shape[0] > 0:
                        s_best_comp = float(np.max(sim_mat[:, c_det]))
                    else:
                        s_best_comp = None

                    margin = (s_pos - s_best_comp) if (s_pos is not None and s_best_comp is not None) else None
                    spawned_new = orig_new in tr["new_track_ids_for_det"].values()

                    # Classify attribution bucket cleanly
                    if r_prev is None:
                        bucket = "prev_tracklet_not_active"
                    elif spawned_new:
                        bucket = "unmatched_spawned_new_track"
                    elif margin is not None and abs(margin) < 0.05:
                        bucket = "high_score_tie_abs_margin_lt_0_05"
                    elif margin is not None and margin < -0.05:
                        bucket = "misassociation_reversal_margin_lt_neg_0_05"
                    else:
                        bucket = "pos_higher_lost_in_hungarian"

                    sort_key = hashlib.sha256(f"13:{seq}:{frame_1}:{orig_gt}".encode("ascii")).hexdigest()
                    all_events.append({
                        "condition": condition,
                        "run_id": run_id,
                        "sequence": seq,
                        "frame": frame_1,
                        "gt_id": orig_gt,
                        "sort_key_sha256": sort_key,
                        "prev_pred_id": orig_prev,
                        "new_pred_id": orig_new,
                        "prev_matched_frame": prev_frame_1,
                        "gap_since_prev_match_frames": gap,
                        "has_competing_gt_iou_ge_0_3": has_competing_gt,
                        "prev_tracklet_active": r_prev is not None,
                        "prev_tracklet_history_len": prev_hist_len,
                        "new_tracklet_history_len": new_hist_len,
                        "spawned_new_track": spawned_new,
                        "sim_threshold": sim_threshold,
                        "s_pos": s_pos,
                        "s_assigned": s_assigned,
                        "s_best_competing": s_best_comp,
                        "margin_pos_minus_best_competing": margin,
                        "bucket": bucket,
                    })

            for g_idx, trk_idx in zip(matched_gt_ids, matched_tracker_ids):
                last_matched_t[g_idx] = t
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids

        assert seq_idsw == expected_idsw, f"{seq}: {seq_idsw} != {expected_idsw}"

    total = len(all_events)
    bucket_counts: dict[str, int] = {}
    for ev in all_events:
        bucket_counts[ev["bucket"]] = bucket_counts.get(ev["bucket"], 0) + 1
    margins = [ev["margin_pos_minus_best_competing"] for ev in all_events if ev["margin_pos_minus_best_competing"] is not None]
    s_pos_vals = [ev["s_pos"] for ev in all_events if ev["s_pos"] is not None]
    summary = {
        "condition": condition,
        "run_id": run_id,
        "sim_threshold": sim_threshold,
        "total_official_idsw": total,
        "bucket_counts": bucket_counts,
        "bucket_fractions": {k: v / total for k, v in bucket_counts.items()} if total else {},
        "gap_gt_1_count": sum(1 for e in all_events if e["gap_since_prev_match_frames"] > 1),
        "competing_gt_count": sum(1 for e in all_events if e["has_competing_gt_iou_ge_0_3"]),
        "margin_stats": {
            "count_defined": len(margins),
            "median": float(np.median(margins)) if margins else None,
            "mean": float(np.mean(margins)) if margins else None,
            "p10": float(np.percentile(margins, 10)) if margins else None,
            "p90": float(np.percentile(margins, 90)) if margins else None,
        },
        "s_pos_stats": {
            "count_defined": len(s_pos_vals),
            "median": float(np.median(s_pos_vals)) if s_pos_vals else None,
            "mean": float(np.mean(s_pos_vals)) if s_pos_vals else None,
        },
    }
    return all_events, summary


def cmd_stage_d1_trace(args: argparse.Namespace) -> None:
    apply_s09_runtime_compat_patches()
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    ckpt_path = Path(args.ckpt)
    sim_threshold = float(args.sim_threshold)
    train32, dev8 = load_split_lists()
    ref_summary = json.loads((RUNS_ROOT / args.ref_run_id / "stage_eval_summary.json").read_text())

    t0 = time.perf_counter()
    cfg = compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        sim_threshold=sim_threshold,
    )
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]

    vid_to_name = tracking_set.video_metadatas["name"].to_dict()
    image_id_to_seq_frame = {
        int(idx): (str(vid_to_name[int(row["video_id"])]), int(row["frame"]) + 1)
        for idx, row in tracking_set.image_metadatas.iterrows()
    }
    seq_nframes = tracking_set.video_metadatas.set_index("name")["nframes"].to_dict()

    base_mod = load_camel_model_explicit(cfg, ckpt_path, sim_threshold)
    wrapped_mod = TraceRecordingCAMELTrack(base_mod, image_id_to_seq_frame)

    Tracklet.count = 1
    pipeline = Pipeline(models=[wrapped_mod])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    torch.cuda.synchronize()
    tracking_engine.track_dataset()
    torch.cuda.synchronize()

    mot_dir = run_dir / "mot_pred"
    tracking_dataset.save_for_eval(
        tracker_state.detections_pred,
        tracker_state.image_metadatas,
        tracker_state.video_metadatas,
        str(mot_dir),
        "bbox_ltwh",
        False,
        is_ground_truth=False,
    )
    # Verify D1 traced MOT outputs match reference non-instrumented Stage B/C3 run byte-for-byte
    for seq in dev8:
        cur_sha = sha256_file(mot_dir / f"{seq}.txt")
        ref_sha = ref_summary["mot_sha256_by_seq"][seq]
        if cur_sha != ref_sha:
            raise RuntimeError(f"D1 trace MOT mismatch vs {args.ref_run_id} on {seq}: {cur_sha} != {ref_sha}")

    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
    evaluator.run(tracker_state)

    events, summary = analyze_idsw_with_traces(
        traces=wrapped_mod.traces,
        gt_dir=run_dir / "eval/gt/DanceTrack-val",
        pred_dir=run_dir / "eval/pred/DanceTrack-val/tracklab",
        detailed_csv=run_dir / "eval/results/tracklab/pedestrian_detailed.csv",
        dev8=dev8,
        seq_nframes=seq_nframes,
        sim_threshold=sim_threshold,
        condition=args.condition,
        run_id=args.run_id,
    )
    t1 = time.perf_counter()

    events_csv = run_dir / "d1_idsw_margins.csv"
    if events:
        with events_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(events)
    else:
        events_csv.write_text("")

    out_summary = {
        **summary,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "mot_sha256_exact_match_ref_run": args.ref_run_id,
        "events_csv": str(events_csv),
    }
    (run_dir / "stage_d1_summary.json").write_text(json.dumps(out_summary, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "condition": args.condition, "total_idsw": summary["total_official_idsw"], "buckets": summary["bucket_counts"]}, indent=2))


def cmd_stage_d2_clean_history(args: argparse.Namespace) -> None:
    apply_s09_runtime_compat_patches()
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    ckpt_path = Path(args.ckpt)
    ckpt_sha = sha256_file(ckpt_path)
    sim_threshold = float(args.sim_threshold)
    train32, dev8 = load_split_lists()

    t0 = time.perf_counter()
    cfg = compose_s09_infer_config(
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        sim_threshold=sim_threshold,
    )
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]

    # Build GT lookup strictly by image_id for privileged clean gallery routing
    gt_by_image_id: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for img_id, grp in tracking_set.detections_gt.groupby("image_id"):
        gids = grp["person_id"].to_numpy(dtype=int)
        gboxes = np.stack(grp["bbox_ltwh"].to_list()).astype(float)
        gt_by_image_id[int(img_id)] = (gids, gboxes)

    base_mod = load_camel_model_explicit(cfg, ckpt_path, sim_threshold)
    wrapped_mod = CleanHistoryCAMELTrack(base_mod, gt_by_image_id)

    Tracklet.count = 1
    pipeline = Pipeline(models=[wrapped_mod])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    torch.cuda.synchronize()
    t_track0 = time.perf_counter()
    tracking_engine.track_dataset()
    torch.cuda.synchronize()
    t_track1 = time.perf_counter()
    peak_vram_mib = float(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    mot_dir = run_dir / "mot_pred"
    tracking_dataset.save_for_eval(
        tracker_state.detections_pred,
        tracker_state.image_metadatas,
        tracker_state.video_metadatas,
        str(mot_dir),
        "bbox_ltwh",
        False,
        is_ground_truth=False,
    )
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
    evaluator.run(tracker_state)
    t1 = time.perf_counter()

    detailed_csv = run_dir / "eval/results/tracklab/pedestrian_detailed.csv"
    with detailed_csv.open(newline="") as f:
        rows = {r["seq"]: r for r in csv.DictReader(f)}
    assert set(rows.keys()) == (set(dev8) | {"COMBINED"})

    summary = {
        "schema_version": "s09-d2-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "condition": "T_TWOSTAGE_TH050_PRIVILEGED_CLEAN_HISTORY",
        "gt_role": "privileged_clean_history_counterfactual_not_deployable",
        "seed": "13",
        "scope_id": "dev8",
        "sim_threshold": sim_threshold,
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "wall_seconds": t1 - t0,
        "tracking_seconds": t_track1 - t_track0,
        "peak_vram_mib": peak_vram_mib,
        "routing_stats": wrapped_mod.routing_stats,
        "eval_summary": {
            "detailed_csv": str(detailed_csv),
            "summary_txt": str(run_dir / "eval/results/tracklab/pedestrian_summary.txt"),
            "gt_dir": str(run_dir / "eval/gt/DanceTrack-val"),
            "pred_dir": str(run_dir / "eval/pred/DanceTrack-val/tracklab"),
        },
        "combined_metrics_preview": {
            "HOTA": float(rows["COMBINED"]["HOTA___AUC"]) * 100.0,
            "DetA": float(rows["COMBINED"]["DetA___AUC"]) * 100.0,
            "AssA": float(rows["COMBINED"]["AssA___AUC"]) * 100.0,
            "IDF1": float(rows["COMBINED"]["IDF1"]) * 100.0,
            "MOTA": float(rows["COMBINED"]["MOTA"]) * 100.0,
            "IDSW": int(float(rows["COMBINED"]["IDSW"])),
            "FP": int(float(rows["COMBINED"]["CLR_FP"])),
            "FN": int(float(rows["COMBINED"]["CLR_FN"])),
            "Frag": int(float(rows["COMBINED"]["Frag"])),
        },
    }
    (run_dir / "stage_d2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "routing_stats": wrapped_mod.routing_stats,
        "combined": summary["combined_metrics_preview"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stage-a-lock")

    p_c1 = sub.add_parser("train-c1")
    p_c1.add_argument("--cue", choices=list(TE_CUES), required=True)
    p_c1.add_argument("--run-id", required=True)

    p_c2 = sub.add_parser("train-c2")
    p_c2.add_argument("--run-id", required=True)
    p_c2.add_argument("--app-ckpt", required=True)
    p_c2.add_argument("--kp-ckpt", required=True)
    p_c2.add_argument("--bbox-ckpt", required=True)

    p_ie = sub.add_parser("infer-eval")
    p_ie.add_argument("--condition", required=True)
    p_ie.add_argument("--ckpt", required=True)
    p_ie.add_argument("--sim-threshold", type=float, default=None)
    p_ie.add_argument("--sim-threshold-from-summary", type=Path, default=None)
    p_ie.add_argument("--seed", required=True)
    p_ie.add_argument("--run-id", required=True)

    p_d1 = sub.add_parser("stage-d1-trace")
    p_d1.add_argument("--condition", required=True)
    p_d1.add_argument("--ckpt", required=True)
    p_d1.add_argument("--sim-threshold", type=float, required=True)
    p_d1.add_argument("--ref-run-id", required=True)
    p_d1.add_argument("--run-id", required=True)

    p_d2 = sub.add_parser("stage-d2-clean-history")
    p_d2.add_argument("--ckpt", required=True)
    p_d2.add_argument("--sim-threshold", type=float, required=True)
    p_d2.add_argument("--run-id", required=True)

    args = parser.parse_args()
    dispatch = {
        "stage-a-lock": cmd_stage_a_lock,
        "train-c1": cmd_train_c1,
        "train-c2": cmd_train_c2,
        "infer-eval": cmd_infer_eval,
        "stage-d1-trace": cmd_stage_d1_trace,
        "stage-d2-clean-history": cmd_stage_d2_clean_history,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
