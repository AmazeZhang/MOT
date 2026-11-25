#!/usr/bin/env python3
"""S08 CAMELTrack bounded baseline runner (Stages A-E)."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Register TrackLab plugins (including cameltrack SearchPathPlugin)
import tracklab.main  # noqa: F401
from tracklab.datastruct import TrackerState
from tracklab.engine import OfflineTrackingEngine
from tracklab.pipeline import Pipeline

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s08_camel_baseline_20261125.yaml"
SPLIT_PATH = REPO_ROOT / "manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml"
REPORT_DIR = REPO_ROOT / "reports/baselines/s08_camel_baseline_20261125"
RUNS_ROOT = DATA_ROOT / "runs/s08_camel_baseline_20261125"

S08_DATASET_ROOT = DATA_ROOT / "cache/s08_camel/dancetrack_s08_dataset"
S08_STATES_DIR = DATA_ROOT / "cache/s08_camel/states"
PRETRAINED_CKPT = DATA_ROOT / "weights/camel/camel_bbox_app_kps_dancetrack.ckpt"

ALLOWED_PHYSICAL_GPUS = {"1", "2", "3", "4", "5", "6", "7"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            raise RuntimeError("CUDA is required for this stage but torch.cuda.is_available() is False.")
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


def compose_s08_config(
    mode: str,
    run_dir: Path,
    train_vids: list[str],
    val_vids: list[str],
    train_state: Path | None = None,
    val_state: Path | None = None,
    dm_cache_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    nframes: int = -1,
    max_epochs: int = 20,
    samples_per_video: int = 100,
    num_workers: int = 4,
    eval_tracking: bool = False,
) -> DictConfig:
    if mode == "infer":
        overrides = [
            "dataset=dancetrack",
            "visualization=none",
            "pipeline=[track]",
            f"dataset.dataset_path={S08_DATASET_ROOT}",
            "dataset.eval_set=val",
            f"dataset.nframes={nframes}",
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
            "modules.track.override_camel_cfg.sim_threshold=0.1",
            "+modules.track.override_camel_cfg.use_computed_sim_threshold=false",
            "modules.track.CAMEL.sim_threshold=0.1",
            "modules.track.CAMEL.use_computed_sim_threshold=false",
            f"eval_tracking={str(eval_tracking).lower()}",
            "test_tracking=true",
            "use_wandb=false",
            "use_rich=false",
            f"num_cores={num_workers}",
            f"project_dir={run_dir}",
        ]
        with initialize_config_module(version_base=None, config_module="tracklab.configs"):
            return compose(config_name="cameltrack", overrides=overrides)

    if mode == "train":
        overrides = [
            "dataset=dancetrack",
            f"dataset.dataset_path={S08_DATASET_ROOT}",
            "dataset.eval_set=val",
            f"dataset.nframes={nframes}",
            f"dataset.vids_dict.train={json.dumps(train_vids)}",
            f"dataset.vids_dict.val={json.dumps(val_vids)}",
            "dataset.vids_dict.test=[]",
            f"state.load_file={val_state}",
            "state.save_file=null",
            "modules.track.training_enabled=true",
            "modules.track.checkpoint_path=null",
            "modules.track.CAMEL.gaffe.checkpoint_path=null",
            "modules.track.CAMEL.temporal_encoders.app_encoder.checkpoint_path=null",
            "modules.track.CAMEL.temporal_encoders.kp_encoder.checkpoint_path=null",
            "modules.track.CAMEL.temporal_encoders.bbox_encoder.checkpoint_path=null",
            "modules.track.CAMEL.sim_threshold=0.1",
            "modules.track.CAMEL.use_computed_sim_threshold=false",
            "modules.track.CAMEL.optimizer.init_lr=0.0001",
            "modules.track.CAMEL.optimizer.weight_decay=0.0001",
            f"modules.track.train_cfg.pl_trainer.max_epochs={max_epochs}",
            "modules.track.train_cfg.pl_trainer.precision=32",
            "modules.track.train_cfg.pl_trainer.accumulate_grad_batches=1",
            "modules.track.train_cfg.pl_trainer.enable_progress_bar=false",
            "modules.track.train_cfg.pl_trainer.log_every_n_steps=10",
            "modules.track.train_cfg.pl_trainer.check_val_every_n_epochs=1",
            "modules.track.train_cfg.model_selection_criteria=last",
            "modules.track.datamodule_cfg.batch_size=8",
            "modules.track.datamodule_cfg.num_samples=32",
            f"modules.track.datamodule_cfg.samples_per_video={samples_per_video}",
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
        with initialize_config_module(version_base=None, config_module="tracklab.configs"):
            return compose(config_name="cameltrack_train", overrides=overrides)
    raise ValueError(f"Unknown mode: {mode}")


def param_summary(model: torch.nn.Module) -> dict[str, Any]:
    groups = {
        "total": model,
        "gaffe": model.gaffe,
        "temp_encs.app_encoder": model.temp_encs["app_encoder"],
        "temp_encs.kp_encoder": model.temp_encs["kp_encoder"],
        "temp_encs.bbox_encoder": model.temp_encs["bbox_encoder"],
    }
    out = {}
    for k, mod in groups.items():
        tot = sum(p.numel() for p in mod.parameters())
        trn = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        out[k] = {"total_params": tot, "trainable_params": trn}
    return out


def clone_module_state(model: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    res: dict[str, dict[str, torch.Tensor]] = {}
    for prefix in ("gaffe", "temp_encs.app_encoder", "temp_encs.kp_encoder", "temp_encs.bbox_encoder"):
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


def cmd_resolve_configs(_args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=False)
    train32, dev8 = load_split_lists()
    out_cfg_dir = REPORT_DIR / "resolved_configs"
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    dummy_run = RUNS_ROOT / "_resolved_placeholder"
    cfg_p = compose_s08_config(
        mode="infer",
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=PRETRAINED_CKPT,
        eval_tracking=True,
    )
    cfg_t = compose_s08_config(
        mode="train",
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=dev8,
        train_state=S08_STATES_DIR / "states_train32.pklz",
        val_state=S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=DATA_ROOT / "cache/s08_camel/dm_d_train32_dev8_v1",
        max_epochs=20,
        samples_per_video=100,
    )
    cfg_c = compose_s08_config(
        mode="train",
        run_dir=dummy_run,
        train_vids=["dancetrack0027"],
        val_vids=["dancetrack0012"],
        train_state=S08_STATES_DIR / "states_c_train_0027_f1_400.pklz",
        val_state=S08_STATES_DIR / "states_c_dev_0012_f1_200.pklz",
        dm_cache_dir=DATA_ROOT / "cache/s08_camel/dm_c_smoke_v1",
        nframes=-1,
        max_epochs=10,
        samples_per_video=100,
    )
    cfg_b = compose_s08_config(
        mode="infer",
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=["dancetrack0027"],
        val_state=S08_STATES_DIR / "states_b_0027_f1_400_gt_free.pklz",
        checkpoint_path=PRETRAINED_CKPT,
        nframes=400,
        eval_tracking=False,
    )

    (out_cfg_dir / "stage_b_p_official_infer.yaml").write_text(OmegaConf.to_yaml(cfg_b, resolve=True))
    (out_cfg_dir / "stage_c_t_scratch_smoke.yaml").write_text(OmegaConf.to_yaml(cfg_c, resolve=True))
    (out_cfg_dir / "stage_d_t_scratch_train.yaml").write_text(OmegaConf.to_yaml(cfg_t, resolve=True))
    (out_cfg_dir / "stage_e_p_official_dev8.yaml").write_text(OmegaConf.to_yaml(cfg_p, resolve=True))

    # Also resolve T_SCRATCH inference config (with checkpoint placeholder path)
    cfg_te = compose_s08_config(
        mode="infer",
        run_dir=dummy_run,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=RUNS_ROOT / "d_t_scratch_seed13_e20/CAMEL/last.ckpt",
        eval_tracking=True,
    )
    (out_cfg_dir / "stage_e_t_scratch_dev8.yaml").write_text(OmegaConf.to_yaml(cfg_te, resolve=True))

    # Instantiate T_SCRATCH model on CPU to inspect optimizer, scheduler, and parameter counts
    model = instantiate(cfg_t.modules.track.CAMEL, _recursive_=False)
    params_info = param_summary(model)
    opt_cls = torch.optim.AdamW
    opt_inst = opt_cls(model.parameters(), lr=cfg_t.modules.track.CAMEL.optimizer.init_lr, weight_decay=cfg_t.modules.track.CAMEL.optimizer.weight_decay)
    sched_info = {
        "scheduler": "transformers.get_cosine_schedule_with_warmup",
        "interval": "step",
        "num_warmup_steps_formula": "estimated_stepping_batches // 20",
        "expected_stepping_batches_train32_e20": 8000,
        "expected_warmup_steps_train32_e20": 400,
    }

    # Lock packages via uv pip freeze in s08-camel
    pip_freeze = subprocess.check_output(
        ["/home/imc/yzy/agent/.tools/bin/uv", "pip", "freeze", "--python", "/media/imc/data/mot-data/envs/s08-camel/bin/python"],
        text=True,
    ).splitlines()
    tracked_pkgs = {}
    for line in pip_freeze:
        for pkg in ("torch==", "torchvision==", "pytorch-lightning==", "lightning==", "tracklab==", "sn-trackeval==", "torchreid", "transformers==", "pytorch-metric-learning==", "hydra-core==", "omegaconf==", "numpy==", "pandas==", "scipy==", "wandb==", "cameltrack"):
            if line.lower().startswith(pkg.lower()):
                tracked_pkgs[line.split("==")[0].split(" @ ")[0]] = line

    df_out = subprocess.check_output(["df", "-B1", str(DATA_ROOT)], text=True).splitlines()[1].split()
    disk_free_gib = int(df_out[3]) / (1024 ** 3)

    nvidia_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()

    runtime_lock = {
        "schema_version": "s08-runtime-lock-v1",
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
            "scripts/prepare_s08_camel_assets.py": sha256_file(REPO_ROOT / "scripts/prepare_s08_camel_assets.py"),
            "scripts/run_s08_camel.py": sha256_file(REPO_ROOT / "scripts/run_s08_camel.py"),
        },
        "model_architecture_and_optimizer": {
            "parameters": params_info,
            "optimizer_class": f"{opt_inst.__class__.__module__}.{opt_inst.__class__.__name__}",
            "optimizer_defaults": {
                "lr": cfg_t.modules.track.CAMEL.optimizer.init_lr,
                "weight_decay": cfg_t.modules.track.CAMEL.optimizer.weight_decay,
            },
            "scheduler": sched_info,
        },
        "key_packages": tracked_pkgs,
        "all_packages_count": len(pip_freeze),
        "all_packages": pip_freeze,
        "gpu_inventory": nvidia_out,
        "gpu_safety_policy": {
            "forbidden_physical_gpus": [0],
            "allowed_physical_gpus": [1, 2, 3, 4, 5, 6, 7],
            "check_during_resolve": gpu_info,
        },
        "storage": {
            "data_root": str(DATA_ROOT),
            "free_gib_before_runs": round(disk_free_gib, 3),
        },
    }
    (REPORT_DIR / "runtime_lock.json").write_text(json.dumps(runtime_lock, indent=2) + "\n")

    commands_sh = """#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=/media/imc/data/mot-data/envs/s08-camel/bin/python
SCRIPT=/home/imc/yzy/MOT/scripts/run_s08_camel.py

# Stage A: asset audit & runtime lock (CPU)
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/prepare_s08_camel_assets.py
CUDA_VISIBLE_DEVICES='' $PY $SCRIPT resolve-configs

# Stage B: P_OFFICIAL online boundary verification on dancetrack0027 (single physical GPU 1..7, GPU 0 forbidden)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-b --run-id b_p_official_online_check_v1

# Stage C: T_SCRATCH_SMOKE 100 steps with step-50 save & new-process resume + reload check
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-c-part1 --run-id c_t_scratch_smoke_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-c-part2 --run-id c_t_scratch_smoke_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-c-reload-check --run-id c_t_scratch_smoke_v1

# Stage D: T_SCRATCH full 20-epoch training on motion_train32 (new dir, isolated cache)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-d --run-id d_t_scratch_seed13_e20_v1

# Stage E: Independent closed-loop evaluation on motion_dev8 (P_OFFICIAL and T_SCRATCH)
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-e --condition P_OFFICIAL --run-id e_p_official_dev8_v1
CUDA_VISIBLE_DEVICES=1 $PY $SCRIPT stage-e --condition T_SCRATCH --ckpt /media/imc/data/mot-data/runs/s08_camel_baseline_20261125/d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --run-id e_t_scratch_dev8_v1

# Stage F: Bounded natural failure summary on T_SCRATCH dev8 predictions
CUDA_VISIBLE_DEVICES='' $PY /home/imc/yzy/MOT/scripts/export_s08_failures.py
"""
    (REPORT_DIR / "commands.sh").write_text(commands_sh)
    os.chmod(REPORT_DIR / "commands.sh", 0o755)
    print(json.dumps({"status": "PASS", "resolved_configs": sorted(p.name for p in out_cfg_dir.iterdir()), "total_params": params_info["total"]["total_params"]}, indent=2))


def run_inference_tracking(
    cfg: DictConfig,
    run_dir: Path,
    expected_ckpt_sha256: str | None = None,
    explicit_ckpt_path: Path | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    t0 = time.perf_counter()
    from tracklab.utils import wandb as tl_wandb
    tl_wandb.init(cfg)
    tracking_dataset = instantiate(cfg.dataset)
    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]

    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    if explicit_ckpt_path is not None:
        actual_sha = sha256_file(explicit_ckpt_path)
        if expected_ckpt_sha256 is not None and actual_sha != expected_ckpt_sha256:
            raise RuntimeError(f"Checkpoint SHA256 mismatch: {actual_sha} != {expected_ckpt_sha256}")
        override_cfg = OmegaConf.to_container(cfg.modules.track.override_camel_cfg, resolve=True) or {}
        track_mod.CAMEL = type(track_mod.CAMEL).load_from_checkpoint(
            explicit_ckpt_path, map_location=track_mod.device, **override_cfg
        )
        # Re-sync sim_threshold explicitly
        assert float(track_mod.CAMEL.sim_threshold) == 0.1
        assert bool(track_mod.CAMEL.use_computed_sim_threshold) is False

    from cameltrack.cameltrack import Tracklet
    Tracklet.count = 1
    pipeline = Pipeline(models=[track_mod])
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(cfg.engine, modules=pipeline, tracker_state=tracker_state)

    torch.cuda.synchronize()
    t_track_start = time.perf_counter()
    tracking_engine.track_dataset()
    torch.cuda.synchronize()
    t_track_end = time.perf_counter()

    peak_vram_mib = float(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    # Save MOT format outputs deterministically
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

    eval_summary = None
    if cfg.eval_tracking:
        evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
        t_eval_start = time.perf_counter()
        evaluator.run(tracker_state)
        t_eval_end = time.perf_counter()
        eval_summary = {
            "eval_seconds": t_eval_end - t_eval_start,
            "detailed_csv": str(run_dir / "eval/results/tracklab/pedestrian_detailed.csv"),
            "summary_txt": str(run_dir / "eval/results/tracklab/pedestrian_summary.txt"),
            "gt_dir": str(run_dir / "eval/gt/DanceTrack-val"),
            "pred_dir": str(run_dir / "eval/pred/DanceTrack-val/tracklab"),
        }

    t1 = time.perf_counter()
    return {
        "wall_seconds": t1 - t0,
        "tracking_seconds": t_track_end - t_track_start,
        "peak_vram_mib": peak_vram_mib,
        "mot_dir": str(mot_dir),
        "detections_pred_rows": int(len(tracker_state.detections_pred)) if tracker_state.detections_pred is not None else 0,
        "eval_summary": eval_summary,
        "video_metadatas": tracking_set.video_metadatas[["name", "nframes"]].to_dict(orient="records"),
    }


def parse_mot_file(path: Path) -> pd.DataFrame:
    cols = ["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, header=None, names=cols)


def cmd_stage_b(args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=True)
    train32, _ = load_split_lists()
    base_dir = RUNS_ROOT / args.run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sha = sha256_file(PRETRAINED_CKPT)

    specs = [
        ("normal_f1_400", S08_STATES_DIR / "states_b_0027_f1_400_with_gt.pklz", 400),
        ("future_truncated_f1_200", S08_STATES_DIR / "states_b_0027_f1_200_gt_free.pklz", 200),
        ("gt_removed_f1_400", S08_STATES_DIR / "states_b_0027_f1_400_gt_free.pklz", 400),
        ("repeat_normal_f1_400", S08_STATES_DIR / "states_b_0027_f1_400_with_gt.pklz", 400),
    ]

    results: dict[str, Any] = {}
    for name, state_file, nframes in specs:
        torch.cuda.reset_peak_memory_stats(0)
        sub_dir = base_dir / name
        cfg = compose_s08_config(
            mode="infer",
            run_dir=sub_dir,
            train_vids=train32,
            val_vids=["dancetrack0027"],
            val_state=state_file,
            checkpoint_path=PRETRAINED_CKPT,
            nframes=nframes,
            eval_tracking=False,
        )
        res = run_inference_tracking(cfg, sub_dir, expected_ckpt_sha256=ckpt_sha, explicit_ckpt_path=PRETRAINED_CKPT)
        mot_path = Path(res["mot_dir"]) / "dancetrack0027.txt"
        df = parse_mot_file(mot_path)
        assert not df.empty, f"{name}: MOT output is empty"
        assert np.isfinite(df[["frame", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]].to_numpy()).all()
        assert (df["frame"] >= 1).all() and (df["frame"] <= nframes).all()
        # Unique track_id per frame
        dup_frame_id = int(df.duplicated(subset=["frame", "track_id"]).sum())
        assert dup_frame_id == 0, f"{name}: duplicate frame/track_id count={dup_frame_id}"
        covered_frames = int(df["frame"].nunique())
        results[name] = {
            **res,
            "mot_file": str(mot_path),
            "mot_sha256": sha256_file(mot_path),
            "mot_rows": int(len(df)),
            "unique_track_ids": int(df["track_id"].nunique()),
            "first_frame": int(df["frame"].min()),
            "last_frame": int(df["frame"].max()),
            "covered_frames": covered_frames,
            "empty_output_frames": int(nframes - covered_frames),
            "duplicate_frame_track_id": dup_frame_id,
        }

    # Verify invariances
    normal_df = parse_mot_file(Path(results["normal_f1_400"]["mot_file"]))
    gt_free_df = parse_mot_file(Path(results["gt_removed_f1_400"]["mot_file"]))
    repeat_df = parse_mot_file(Path(results["repeat_normal_f1_400"]["mot_file"]))
    prefix_df = parse_mot_file(Path(results["future_truncated_f1_200"]["mot_file"]))

    repeat_exact = results["normal_f1_400"]["mot_sha256"] == results["repeat_normal_f1_400"]["mot_sha256"]
    gt_free_exact = results["normal_f1_400"]["mot_sha256"] == results["gt_removed_f1_400"]["mot_sha256"]
    prefix_slice_norm = normal_df[normal_df["frame"] <= 200].sort_values(["frame", "track_id"]).reset_index(drop=True)
    prefix_sorted = prefix_df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    prefix_exact = prefix_sorted.equals(prefix_slice_norm)
    repeat_df_exact = repeat_df.sort_values(["frame", "track_id"]).reset_index(drop=True).equals(
        normal_df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    )
    gt_free_df_exact = gt_free_df.sort_values(["frame", "track_id"]).reset_index(drop=True).equals(
        normal_df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    )
    if not (repeat_df_exact and gt_free_df_exact and prefix_exact):
        raise RuntimeError(
            f"Stage B invariance check failed: repeat={repeat_df_exact}, gt_free={gt_free_df_exact}, prefix={prefix_exact}"
        )

    summary = {
        "schema_version": "s08-stage-b-v1",
        "status": "PASS",
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "gpu": gpu_info,
        "checkpoint_sha256": ckpt_sha,
        "invariances": {
            "repeat_sha256_identical": repeat_exact,
            "repeat_rows_identical": repeat_df_exact,
            "gt_removed_sha256_identical": gt_free_exact,
            "gt_removed_rows_identical": gt_free_df_exact,
            "future_truncated_f1_200_exact_match": prefix_exact,
        },
        "runs": results,
    }
    (base_dir / "stage_b_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "invariances": summary["invariances"],
        "rows_400": results["normal_f1_400"]["mot_rows"],
        "rows_200": results["future_truncated_f1_200"]["mot_rows"],
    }, indent=2))


def prepare_stage_c_dataset(cfg: DictConfig) -> tuple[Any, Any]:
    tracking_dataset = instantiate(cfg.dataset)
    # TrackingDataset.__init__ keeps full-video training_sets before nframes subsampling; restrict explicitly for Stage C
    for split_name in ("train", "val"):
        tracking_dataset.training_sets[split_name] = tracking_dataset.sets[split_name]
    return tracking_dataset, Pipeline(models=[])


def compute_probe_loss(model: torch.nn.Module, batch: dict[str, Any], device: str = "cuda") -> float:
    was_training = model.training
    orig_dev = next(model.parameters()).device
    model.to(device)
    model.eval()
    with torch.no_grad():
        b = _move_batch_to_device(batch, device)
        tracks, dets = model.train_val_preprocess(b)
        tracks, dets, td_sim_matrix = model.forward(tracks, dets)
        loss = model.compute_loss(tracks, dets, td_sim_matrix)
        val = float(loss.item())
    model.to(orig_dev)
    if was_training:
        model.train()
    return val


def compute_probe_eval_outputs(model: torch.nn.Module, batch: dict[str, Any], device: str = "cuda") -> dict[str, torch.Tensor]:
    was_training = model.training
    orig_dev = next(model.parameters()).device
    model.to(device)
    model.eval()
    with torch.no_grad():
        b = _move_batch_to_device(batch, device)
        tracks, dets = model.train_val_preprocess(b)
        tracks, dets, td_sim_matrix = model.forward(tracks, dets)
        out = {
            "td_sim_matrix": td_sim_matrix.detach().cpu(),
        }
    model.to(orig_dev)
    if was_training:
        model.train()
    return out


def _move_batch_to_device(obj: Any, device: str) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone().to(device)
    if isinstance(obj, dict):
        return {k: _move_batch_to_device(v, device) for k, v in obj.items()}
    return obj


def compare_sim_matrices(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    exact_equal = bool(torch.equal(a, b))
    fin_a = torch.isfinite(a)
    fin_b = torch.isfinite(b)
    finite_mask_equal = bool(torch.equal(fin_a, fin_b))
    max_finite_diff = float((a[fin_a] - b[fin_b]).abs().max().item()) if (finite_mask_equal and fin_a.any()) else float("nan")
    return {
        "exact_equal": exact_equal,
        "finite_mask_equal": finite_mask_equal,
        "max_finite_abs_diff": max_finite_diff,
        "num_finite": int(fin_a.sum().item()),
        "num_neginf": int((a == -float("inf")).sum().item()),
    }


class StageCGradProbeCallback(pl.Callback):
    def __init__(self) -> None:
        self.grad_norms: list[dict[str, Any]] = []
        self.restored_check: dict[str, Any] | None = None

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        total_sq = 0.0
        finite = True
        non_zero_grads = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    finite = False
                gn = float(p.grad.detach().norm(2).item())
                if gn > 0:
                    non_zero_grads += 1
                total_sq += gn ** 2
        self.grad_norms.append({
            "step_before_opt": int(trainer.global_step),
            "grad_l2": float(np.sqrt(total_sq)),
            "finite": finite,
            "non_zero_grad_tensors": non_zero_grads,
        })


def build_smoke_probes(datamodule: Any, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train_probe_path = run_dir / "probe_train_noaug_batch.pt"
    dev_probe_path = run_dir / "probe_dev_batch.pt"
    datamodule.prepare_data()
    datamodule.setup("fit")
    # Save a fixed no-augmentation training probe batch
    orig_tt = datamodule.tracklet_transforms
    orig_bt = datamodule.batch_transforms
    datamodule.tracklet_transforms = None
    datamodule.batch_transforms = {}
    datamodule.setup("fit")
    tr_dl = datamodule.train_dataloader()
    tr_batch = next(iter(tr_dl))
    val_dl = datamodule.val_dataloader()
    val_batch = next(iter(val_dl))
    torch.save(tr_batch, train_probe_path)
    torch.save(val_batch, dev_probe_path)
    # Restore transforms
    datamodule.tracklet_transforms = orig_tt
    datamodule.batch_transforms = orig_bt
    datamodule.setup("fit")
    return tr_batch, val_batch


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(0),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], 0)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


class S08RngCheckpointCallback(pl.Callback):
    def __init__(self) -> None:
        self._pending_rng: dict[str, Any] | None = None

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        checkpoint["s08_rng_states"] = capture_rng_state()

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        if "s08_rng_states" in checkpoint:
            self._pending_rng = checkpoint["s08_rng_states"]
            restore_rng_state(self._pending_rng)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._pending_rng is not None:
            restore_rng_state(self._pending_rng)
            self._pending_rng = None


def cmd_stage_c_part1(args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists (immutable attempt rule): {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    pl.seed_everything(13, workers=True)
    t0 = time.perf_counter()

    dm_cache = DATA_ROOT / f"cache/s08_camel/dm_{args.run_id}"
    cfg = compose_s08_config(
        mode="train",
        run_dir=run_dir,
        train_vids=["dancetrack0027"],
        val_vids=["dancetrack0012"],
        train_state=S08_STATES_DIR / "states_c_train_0027_f1_400.pklz",
        val_state=S08_STATES_DIR / "states_c_dev_0012_f1_200.pklz",
        dm_cache_dir=dm_cache,
        nframes=400,
        max_epochs=10,
        samples_per_video=100,
        num_workers=0,
    )
    # Explicitly cap dev to 200 frames as well
    tracking_dataset = instantiate(cfg.dataset)
    # Subsample val to 200 frames, train to 400 frames
    tracking_dataset.sets["train"] = tracking_dataset._subsample(tracking_dataset.sets["train"], -1, 400, ["dancetrack0027"])
    tracking_dataset.sets["val"] = tracking_dataset._subsample(tracking_dataset.sets["val"], -1, 200, ["dancetrack0012"])
    tracking_dataset.training_sets["train"] = tracking_dataset.sets["train"]
    tracking_dataset.training_sets["val"] = tracking_dataset.sets["val"]

    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL
    init_snapshot = clone_module_state(model)
    torch.save(init_snapshot, run_dir / "step0_init_params.pt")

    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})
    tr_probe, dev_probe = build_smoke_probes(datamodule, run_dir)
    step0_probe_loss = compute_probe_loss(model, tr_probe, device="cuda")

    grad_cb = StageCGradProbeCallback()
    rng_cb = S08RngCheckpointCallback()
    logger = pl.loggers.WandbLogger(project="CAMEL", offline=True, save_dir=str(run_dir))
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", dirpath="CAMEL", mode="min", filename="best", save_last=True
    )
    trainer = pl.Trainer(
        max_steps=50,
        max_epochs=10,
        logger=logger,
        callbacks=[ckpt_cb, pl.callbacks.LearningRateMonitor(), grad_cb, rng_cb],
        accelerator="cuda",
        devices=1,
        precision=32,
        gradient_clip_val=0,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        check_val_every_n_epoch=10,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule)
    assert trainer.global_step == 50, f"Expected global_step=50, got {trainer.global_step}"

    # Explicitly save step50 checkpoint with full trainer/optimizer/scheduler/RNG state
    step50_ckpt = run_dir / "CAMEL/last.ckpt"
    trainer.save_checkpoint(step50_ckpt)
    shutil.copy2(step50_ckpt, run_dir / "CAMEL/step50.ckpt")

    deltas_50 = module_update_deltas(init_snapshot, model)
    for mod_name, d in deltas_50.items():
        assert d["num_changed_tensors"] > 0 and d["max_abs_delta"] > 0.0, f"Module {mod_name} did not update at step 50"

    dev_out_50 = compute_probe_eval_outputs(model, dev_probe, device="cuda")
    torch.save(dev_out_50, run_dir / "probe_dev_out_step50.pt")

    # Save exact step-50 state for strict resume verification in part2
    opt = trainer.optimizers[0]
    sched = trainer.lr_scheduler_configs[0].scheduler
    saved_ckpt_raw = torch.load(step50_ckpt, map_location="cpu", weights_only=False)
    step50_ref = {
        "global_step": int(trainer.global_step),
        "lr": float(opt.param_groups[0]["lr"]),
        "sched_last_epoch": int(sched.last_epoch),
        "opt_state_dict": opt.state_dict(),
        "rng_states": saved_ckpt_raw["s08_rng_states"],
    }
    torch.save(step50_ref, run_dir / "step50_ref_state.pt")

    t1 = time.perf_counter()
    summary = {
        "part": 1,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "global_step": int(trainer.global_step),
        "step0_noaug_probe_loss": step0_probe_loss,
        "step50_ckpt": str(step50_ckpt),
        "step50_ckpt_sha256": sha256_file(step50_ckpt),
        "grad_checks": {
            "num_steps_checked": len(grad_cb.grad_norms),
            "all_finite": all(g["finite"] for g in grad_cb.grad_norms),
            "min_grad_l2": min(g["grad_l2"] for g in grad_cb.grad_norms),
            "max_grad_l2": max(g["grad_l2"] for g in grad_cb.grad_norms),
        },
        "module_deltas_step0_to_50": deltas_50,
    }
    (run_dir / "stage_c_part1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


class ResumeVerifyCallback(pl.Callback):
    def __init__(self, ref_state_path: Path, dev_probe: dict[str, Any], dev_out_50: dict[str, Any]) -> None:
        self.ref_state_path = ref_state_path
        self.dev_probe = dev_probe
        self.dev_out_50 = dev_out_50
        self.report: dict[str, Any] = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ref = torch.load(self.ref_state_path, map_location="cpu", weights_only=False)
        opt = trainer.optimizers[0]
        sched = trainer.lr_scheduler_configs[0].scheduler
        cur_opt = opt.state_dict()
        ref_opt = ref["opt_state_dict"]
        # Compare optimizer state tensors
        max_opt_diff = 0.0
        for pid in ref_opt["state"]:
            for k, v in ref_opt["state"][pid].items():
                cv = cur_opt["state"][pid][k]
                if isinstance(v, torch.Tensor):
                    diff = float((v.cpu() - cv.cpu()).abs().max().item())
                    max_opt_diff = max(max_opt_diff, diff)
        cur_rng = capture_rng_state()
        ref_rng = ref["rng_states"]
        rng_match = {
            "torch": bool(torch.equal(cur_rng["torch"], ref_rng["torch"])),
            "cuda": bool(torch.equal(cur_rng["cuda"], ref_rng["cuda"])),
            "numpy": bool(np.array_equal(cur_rng["numpy"][1], ref_rng["numpy"][1]) and cur_rng["numpy"][2] == ref_rng["numpy"][2]),
            "python": bool(cur_rng["python"] == ref_rng["python"]),
        }
        eval_now = compute_probe_eval_outputs(pl_module, self.dev_probe, device="cuda")
        cmp_eval = compare_sim_matrices(eval_now["td_sim_matrix"], self.dev_out_50["td_sim_matrix"])
        self.report = {
            "restored_global_step": int(trainer.global_step),
            "expected_global_step": int(ref["global_step"]),
            "restored_lr": float(opt.param_groups[0]["lr"]),
            "expected_lr": float(ref["lr"]),
            "restored_sched_last_epoch": int(sched.last_epoch),
            "expected_sched_last_epoch": int(ref["sched_last_epoch"]),
            "max_optimizer_state_abs_diff": max_opt_diff,
            "step50_eval_comparison": cmp_eval,
            "rng_restored_match": rng_match,
        }
        assert trainer.global_step == ref["global_step"]
        assert abs(opt.param_groups[0]["lr"] - ref["lr"]) < 1e-12
        assert sched.last_epoch == ref["sched_last_epoch"]
        assert max_opt_diff == 0.0
        assert cmp_eval["exact_equal"] and cmp_eval["max_finite_abs_diff"] == 0.0
        assert all(rng_match.values()), f"RNG mismatch: {rng_match}"


def cmd_stage_c_part2(args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)
    pl.seed_everything(13, workers=True)
    t0 = time.perf_counter()

    dm_cache = DATA_ROOT / f"cache/s08_camel/dm_{args.run_id}"
    cfg = compose_s08_config(
        mode="train",
        run_dir=run_dir,
        train_vids=["dancetrack0027"],
        val_vids=["dancetrack0012"],
        train_state=S08_STATES_DIR / "states_c_train_0027_f1_400.pklz",
        val_state=S08_STATES_DIR / "states_c_dev_0012_f1_200.pklz",
        dm_cache_dir=dm_cache,
        nframes=400,
        max_epochs=10,
        samples_per_video=100,
        num_workers=0,
    )
    tracking_dataset = instantiate(cfg.dataset)
    tracking_dataset.sets["train"] = tracking_dataset._subsample(tracking_dataset.sets["train"], -1, 400, ["dancetrack0027"])
    tracking_dataset.sets["val"] = tracking_dataset._subsample(tracking_dataset.sets["val"], -1, 200, ["dancetrack0012"])
    tracking_dataset.training_sets["train"] = tracking_dataset.sets["train"]
    tracking_dataset.training_sets["val"] = tracking_dataset.sets["val"]

    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL
    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})

    tr_probe = torch.load(run_dir / "probe_train_noaug_batch.pt", map_location="cpu", weights_only=False)
    dev_probe = torch.load(run_dir / "probe_dev_batch.pt", map_location="cpu", weights_only=False)
    dev_out_50 = torch.load(run_dir / "probe_dev_out_step50.pt", map_location="cpu", weights_only=False)
    init_snapshot = torch.load(run_dir / "step0_init_params.pt", map_location="cpu", weights_only=False)

    resume_cb = ResumeVerifyCallback(run_dir / "step50_ref_state.pt", dev_probe, dev_out_50)
    grad_cb = StageCGradProbeCallback()
    rng_cb = S08RngCheckpointCallback()
    logger = pl.loggers.WandbLogger(project="CAMEL", offline=True, save_dir=str(run_dir))
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", dirpath="CAMEL", mode="min", filename="best", save_last=True
    )
    trainer = pl.Trainer(
        max_steps=100,
        max_epochs=20,
        logger=logger,
        callbacks=[ckpt_cb, pl.callbacks.LearningRateMonitor(), rng_cb, resume_cb, grad_cb],
        accelerator="cuda",
        devices=1,
        precision=32,
        gradient_clip_val=0,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        check_val_every_n_epoch=20,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    ckpt_path = run_dir / "CAMEL/last.ckpt"
    trainer.fit(model, datamodule, ckpt_path=str(ckpt_path))
    assert trainer.global_step == 100, f"Expected global_step=100, got {trainer.global_step}"

    step100_ckpt = run_dir / "CAMEL/step100_last.ckpt"
    trainer.save_checkpoint(step100_ckpt)

    step100_probe_loss = compute_probe_loss(model, tr_probe, device="cuda")
    dev_out_100 = compute_probe_eval_outputs(model, dev_probe, device="cuda")
    torch.save(dev_out_100, run_dir / "probe_dev_out_step100.pt")
    deltas_100 = module_update_deltas(init_snapshot, model)
    for mod_name, d in deltas_100.items():
        assert d["num_changed_tensors"] > 0 and d["max_abs_delta"] > 0.0

    # Check RNG states were saved in checkpoint
    raw_ckpt = torch.load(step100_ckpt, map_location="cpu", weights_only=False)
    has_rng_states = "loops" in raw_ckpt and "state_dict" in raw_ckpt and "optimizer_states" in raw_ckpt and "lr_schedulers" in raw_ckpt

    t1 = time.perf_counter()
    summary = {
        "part": 2,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "global_step": int(trainer.global_step),
        "resume_verification": resume_cb.report,
        "step100_noaug_probe_loss": step100_probe_loss,
        "step100_ckpt": str(step100_ckpt),
        "step100_ckpt_sha256": sha256_file(step100_ckpt),
        "has_full_lightning_state_in_ckpt": has_rng_states,
        "grad_checks": {
            "num_steps_checked": len(grad_cb.grad_norms),
            "all_finite": all(g["finite"] for g in grad_cb.grad_norms),
            "min_grad_l2": min(g["grad_l2"] for g in grad_cb.grad_norms),
            "max_grad_l2": max(g["grad_l2"] for g in grad_cb.grad_norms),
        },
        "module_deltas_step0_to_100": deltas_100,
    }
    (run_dir / "stage_c_part2.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def cmd_stage_c_reload_check(args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    step100_ckpt = run_dir / "CAMEL/step100_last.ckpt"
    p2 = json.loads((run_dir / "stage_c_part2.json").read_text())
    actual_sha = sha256_file(step100_ckpt)
    assert actual_sha == p2["step100_ckpt_sha256"]

    cfg = compose_s08_config(
        mode="infer",
        run_dir=run_dir,
        train_vids=["dancetrack0027"],
        val_vids=["dancetrack0012"],
        val_state=S08_STATES_DIR / "states_b_0027_f1_400_gt_free.pklz",
        checkpoint_path=step100_ckpt,
        eval_tracking=False,
    )
    model_stub = instantiate(cfg.modules.track.CAMEL, _recursive_=False)
    reloaded = type(model_stub).load_from_checkpoint(step100_ckpt, map_location="cuda", sim_threshold=0.1, use_computed_sim_threshold=False)
    dev_probe = torch.load(run_dir / "probe_dev_batch.pt", map_location="cpu", weights_only=False)
    dev_out_100 = torch.load(run_dir / "probe_dev_out_step100.pt", map_location="cpu", weights_only=False)
    reloaded_out = compute_probe_eval_outputs(reloaded, dev_probe, device="cuda")
    cmp_reload = compare_sim_matrices(reloaded_out["td_sim_matrix"], dev_out_100["td_sim_matrix"])
    assert cmp_reload["exact_equal"] and cmp_reload["max_finite_abs_diff"] == 0.0, f"Reload eval mismatch: {cmp_reload}"

    p1 = json.loads((run_dir / "stage_c_part1.json").read_text())
    merged = {
        "schema_version": "s08-stage-c-v1",
        "status": "PASS",
        "gpu": gpu_info,
        "reload_sha256": actual_sha,
        "reload_eval_comparison": cmp_reload,
        "step0_noaug_probe_loss": p1["step0_noaug_probe_loss"],
        "step100_noaug_probe_loss": p2["step100_noaug_probe_loss"],
        "part1": p1,
        "part2": p2,
    }
    (run_dir / "stage_c_summary.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS",
        "reload_eval_comparison": cmp_reload,
        "step0_noaug_probe_loss": p1["step0_noaug_probe_loss"],
        "step100_noaug_probe_loss": p2["step100_noaug_probe_loss"],
    }, indent=2))


class StageDEpochLogger(pl.Callback):
    def __init__(self, run_dir: Path, init_snapshot: dict[str, dict[str, torch.Tensor]], t_start: float) -> None:
        self.run_dir = run_dir
        self.init_snapshot = init_snapshot
        self.t_start = t_start
        self.rows: list[dict[str, Any]] = []
        self.epoch_grad_nonfinite = 0
        self.epoch2_ckpt_info: dict[str, Any] | None = None

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        for p in pl_module.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self.epoch_grad_nonfinite += 1
                break

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_num = int(trainer.current_epoch) + 1
        elapsed = time.perf_counter() - self.t_start
        metrics = trainer.callback_metrics
        lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        step = int(trainer.global_step)
        # Each step has batch_size=8 videos/items (each item has num_samples=32 tracklet-det queries)
        samples_seen = step * 8

        # Snapshot epoch 2 engineering checkpoint cleanly without disturbing schedule
        if epoch_num == 2:
            ep2_path = self.run_dir / "CAMEL/epoch2_engineering.ckpt"
            trainer.save_checkpoint(ep2_path)
            self.epoch2_ckpt_info = {
                "epoch": 2,
                "global_step": step,
                "path": str(ep2_path),
                "sha256": sha256_file(ep2_path),
                "bytes": ep2_path.stat().st_size,
                "nonfinite_grad_steps_up_to_e2": self.epoch_grad_nonfinite,
            }

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
                    "seed": 13,
                    "epoch": epoch_num,
                    "step": step,
                    "split": split_name,
                    "metric": metric_key,
                    "value": repr(val),
                    "unit": unit,
                    "samples_seen": samples_seen,
                    "elapsed_seconds": round(elapsed, 3),
                    "checkpoint_id": "epoch2_engineering" if epoch_num == 2 else ("epoch20_last" if epoch_num == 20 else ""),
                })
        self.rows.append({
            "run_id": self.run_dir.name,
            "seed": 13,
            "epoch": epoch_num,
            "step": step,
            "split": "train",
            "metric": "lr",
            "value": repr(lr),
            "unit": "lr",
            "samples_seen": samples_seen,
            "elapsed_seconds": round(elapsed, 3),
            "checkpoint_id": "epoch2_engineering" if epoch_num == 2 else ("epoch20_last" if epoch_num == 20 else ""),
        })
        # Flush curve incrementally
        self.write_curves()

    def write_curves(self) -> None:
        out_csv = self.run_dir / "learning_curves.csv"
        fields = ["run_id", "seed", "epoch", "step", "split", "metric", "value", "unit", "samples_seen", "elapsed_seconds", "checkpoint_id"]
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)


def cmd_stage_d(args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists (immutable attempt rule): {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_offline_wandb(run_dir)
    os.chdir(run_dir)

    pl.seed_everything(13, workers=True)
    t0 = time.perf_counter()

    train32, dev8 = load_split_lists()
    dm_cache = DATA_ROOT / f"cache/s08_camel/dm_{args.run_id}_isolated"
    cfg = compose_s08_config(
        mode="train",
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        train_state=S08_STATES_DIR / "states_train32.pklz",
        val_state=S08_STATES_DIR / "states_dev8_train_ref.pklz",
        dm_cache_dir=dm_cache,
        nframes=-1,
        max_epochs=20,
        samples_per_video=100,
        num_workers=4,
    )

    tracking_dataset = instantiate(cfg.dataset)
    track_mod = instantiate(cfg.modules.track, device="cuda", tracking_dataset=tracking_dataset)
    model = track_mod.CAMEL
    init_snapshot = clone_module_state(model)
    params_info = param_summary(model)

    datamodule = instantiate(cfg.modules.track.datamodule_cfg, tracking_dataset=tracking_dataset, pipeline=Pipeline([]), datasets={})

    from cameltrack.train.callbacks import SimMetrics
    epoch_cb = StageDEpochLogger(run_dir, init_snapshot, t0)
    save_best_loss = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        dirpath="CAMEL",
        mode="min",
        filename="epoch={epoch}-loss={val/loss:.4f}",
        auto_insert_metric_name=False,
        save_last=True,
    )
    logger = pl.loggers.WandbLogger(project="CAMEL", offline=True, save_dir=str(run_dir))
    trainer = pl.Trainer(
        max_epochs=20,
        logger=logger,
        callbacks=[save_best_loss, pl.callbacks.LearningRateMonitor(), SimMetrics(), epoch_cb],
        accelerator="cuda",
        devices=1,
        precision=32,
        gradient_clip_val=0,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    last_ckpt = run_dir / "CAMEL/last.ckpt"
    epoch20_copy = run_dir / "CAMEL/epoch20_last.ckpt"
    shutil.copy2(last_ckpt, epoch20_copy)

    deltas_20 = module_update_deltas(init_snapshot, model)
    summary = {
        "schema_version": "s08-stage-d-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "seed": 13,
        "gpu": gpu_info,
        "wall_seconds": t1 - t0,
        "gpu_device_hours": (t1 - t0) / 3600.0,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(0) / (1024 ** 2)),
        "completed_epochs": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "nonfinite_grad_steps": epoch_cb.epoch_grad_nonfinite,
        "parameters": params_info,
        "module_deltas_init_to_epoch20": deltas_20,
        "epoch2_checkpoint": epoch_cb.epoch2_ckpt_info,
        "epoch20_checkpoint": {
            "path": str(last_ckpt),
            "copy_path": str(epoch20_copy),
            "sha256": sha256_file(last_ckpt),
            "bytes": last_ckpt.stat().st_size,
        },
        "best_loss_checkpoint_observed_not_selected": {
            "path": str(save_best_loss.best_model_path),
            "best_val_loss": float(save_best_loss.best_model_score.item()) if save_best_loss.best_model_score is not None else None,
        },
        "datamodule_cache_dir": str(dm_cache),
        "learning_curves_csv": str(run_dir / "learning_curves.csv"),
    }
    (run_dir / "stage_d_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "completed_epochs": summary["completed_epochs"],
        "global_step": summary["global_step"],
        "wall_seconds": round(summary["wall_seconds"], 2),
        "peak_vram_mib": round(summary["peak_vram_mib"], 2),
        "last_ckpt_sha256": summary["epoch20_checkpoint"]["sha256"],
    }, indent=2))


def cmd_stage_e(args: argparse.Namespace) -> None:
    gpu_info = enforce_gpu_safety(require_gpu=True)
    run_dir = RUNS_ROOT / args.run_id
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists (immutable attempt rule): {run_dir}")

    train32, dev8 = load_split_lists()
    if args.condition == "P_OFFICIAL":
        ckpt_path = PRETRAINED_CKPT
        expected_sha = sha256_file(PRETRAINED_CKPT)
        seed_val = "NA"
    elif args.condition == "T_SCRATCH":
        if args.ckpt is None:
            raise ValueError("--ckpt is required for T_SCRATCH")
        ckpt_path = Path(args.ckpt)
        expected_sha = sha256_file(ckpt_path)
        seed_val = "13"
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    torch.cuda.reset_peak_memory_stats(0)
    cfg = compose_s08_config(
        mode="infer",
        run_dir=run_dir,
        train_vids=train32,
        val_vids=dev8,
        val_state=S08_STATES_DIR / "states_dev8_gt_free.pklz",
        checkpoint_path=ckpt_path,
        nframes=-1,
        eval_tracking=True,
    )
    res = run_inference_tracking(cfg, run_dir, expected_ckpt_sha256=expected_sha, explicit_ckpt_path=ckpt_path)

    # Verify TrackEval detailed CSV exists and covers all 8 sequences + COMBINED
    detailed_csv = Path(res["eval_summary"]["detailed_csv"])
    assert detailed_csv.is_file(), f"Missing {detailed_csv}"
    with detailed_csv.open(newline="") as f:
        rows = {r["seq"]: r for r in csv.DictReader(f)}
    assert set(rows.keys()) == (set(dev8) | {"COMBINED"}), f"Unexpected seq keys: {sorted(rows.keys())}"

    summary = {
        "schema_version": "s08-stage-e-v1",
        "status": "COMPLETED",
        "run_id": args.run_id,
        "condition": args.condition,
        "seed": seed_val,
        "scope_id": "dev8",
        "gpu": gpu_info,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": expected_sha,
        "state_file": str(S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        "state_file_sha256": sha256_file(S08_STATES_DIR / "states_dev8_gt_free.pklz"),
        **res,
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
    (run_dir / "stage_e_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "condition": args.condition,
        "wall_seconds": round(summary["wall_seconds"], 2),
        "peak_vram_mib": round(summary["peak_vram_mib"], 2),
        "combined": summary["combined_metrics_preview"],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("resolve-configs")

    p_b = sub.add_parser("stage-b")
    p_b.add_argument("--run-id", required=True)

    p_c1 = sub.add_parser("stage-c-part1")
    p_c1.add_argument("--run-id", required=True)

    p_c2 = sub.add_parser("stage-c-part2")
    p_c2.add_argument("--run-id", required=True)

    p_cr = sub.add_parser("stage-c-reload-check")
    p_cr.add_argument("--run-id", required=True)

    p_d = sub.add_parser("stage-d")
    p_d.add_argument("--run-id", required=True)

    p_e = sub.add_parser("stage-e")
    p_e.add_argument("--condition", choices=["P_OFFICIAL", "T_SCRATCH"], required=True)
    p_e.add_argument("--run-id", required=True)
    p_e.add_argument("--ckpt", type=Path, default=None)

    args = parser.parse_args()
    dispatch = {
        "resolve-configs": cmd_resolve_configs,
        "stage-b": cmd_stage_b,
        "stage-c-part1": cmd_stage_c_part1,
        "stage-c-part2": cmd_stage_c_part2,
        "stage-c-reload-check": cmd_stage_c_reload_check,
        "stage-d": cmd_stage_d,
        "stage-e": cmd_stage_e,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
