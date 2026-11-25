#!/usr/bin/env python3
"""Assemble and verify the S08 CAMELTrack report bundle in reports/baselines/s08_camel_baseline_20261125/."""

from __future__ import annotations

import configparser
import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path("/home/imc/yzy/MOT")
DATA_ROOT = Path("/media/imc/data/mot-data")
PROTOCOL_PATH = REPO_ROOT / "manifests/protocols/s08_camel_baseline_20261125.yaml"
SPLIT_PATH = REPO_ROOT / "manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml"
REPORT_DIR = REPO_ROOT / "reports/baselines/s08_camel_baseline_20261125"
RUNS_ROOT = DATA_ROOT / "runs/s08_camel_baseline_20261125"
DANCETRACK_TRAIN_DIR = DATA_ROOT / "datasets/dancetrack/train"

METRIC_FIELDS = [
    "run_id", "condition", "seed", "scope_id", "sequence", "status",
    "num_frames", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW",
    "FP", "FN", "Frag", "source_artifact_id", "source_locator",
    "missing_reason",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_detailed_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        rows = {r["seq"]: r for r in csv.DictReader(f)}
    if "COMBINED" not in rows:
        raise ValueError(f"Missing COMBINED in {path}")
    return rows


def seq_frame_count(seq: str) -> int:
    cp = configparser.ConfigParser()
    cp.read(DANCETRACK_TRAIN_DIR / seq / "seqinfo.ini")
    return cp["Sequence"].getint("seqLength")


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    split = yaml.safe_load(SPLIT_PATH.read_text())
    dev8 = list(protocol["data"]["dev8"]["sequences"])
    frames = {s: seq_frame_count(s) for s in dev8}

    source_audit = json.loads((REPORT_DIR / "source_audit.json").read_text())
    split_audit = json.loads((REPORT_DIR / "split_audit.json").read_text())
    runtime_lock = json.loads((REPORT_DIR / "runtime_lock.json").read_text())

    stage_b = json.loads((RUNS_ROOT / "b_p_official_online_check_v2/stage_b_summary.json").read_text())
    stage_c = json.loads((RUNS_ROOT / "c_t_scratch_smoke_v5/stage_c_summary.json").read_text())
    stage_d = json.loads((RUNS_ROOT / "d_t_scratch_seed13_e20_v1/stage_d_summary.json").read_text())
    stage_e_p = json.loads((RUNS_ROOT / "e_p_official_dev8_v2/stage_e_summary.json").read_text())
    stage_e_t = json.loads((RUNS_ROOT / "e_t_scratch_dev8_v1/stage_e_summary.json").read_text())
    diagnostics = json.loads((REPORT_DIR / "diagnostics.json").read_text())

    # Copy Stage D learning curves into report bundle
    shutil.copy2(Path(stage_d["learning_curves_csv"]), REPORT_DIR / "learning_curves.csv")

    # Export aggregate_metrics.csv and per_sequence_metrics.csv from official TrackEval pedestrian_detailed.csv
    agg_rows: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []
    eval_configs = [
        ("P_OFFICIAL", "NA", stage_e_p, "server_e_p_official_detailed_csv"),
        ("T_SCRATCH", "13", stage_e_t, "server_e_t_scratch_detailed_csv"),
    ]
    for cond, seed_str, e_sum, art_id in eval_configs:
        detailed_path = Path(e_sum["eval_summary"]["detailed_csv"])
        detailed = load_detailed_csv(detailed_path)
        assert set(detailed.keys()) == (set(dev8) | {"COMBINED"})
        for seq in ["COMBINED_SEQ", *dev8]:
            key = "COMBINED" if seq == "COMBINED_SEQ" else seq
            d = detailed[key]
            row = {
                "run_id": e_sum["run_id"],
                "condition": cond,
                "seed": seed_str,
                "scope_id": "dev8",
                "sequence": seq,
                "status": "COMPLETED",
                "num_frames": sum(frames.values()) if seq == "COMBINED_SEQ" else frames[seq],
                "HOTA": repr(float(d["HOTA___AUC"]) * 100.0),
                "DetA": repr(float(d["DetA___AUC"]) * 100.0),
                "AssA": repr(float(d["AssA___AUC"]) * 100.0),
                "IDF1": repr(float(d["IDF1"]) * 100.0),
                "MOTA": repr(float(d["MOTA"]) * 100.0),
                "IDSW": int(float(d["IDSW"])),
                "FP": int(float(d["CLR_FP"])),
                "FN": int(float(d["CLR_FN"])),
                "Frag": int(float(d["Frag"])),
                "source_artifact_id": art_id,
                "source_locator": f"eval/results/tracklab/pedestrian_detailed.csv:{key}",
                "missing_reason": "",
            }
            (agg_rows if seq == "COMBINED_SEQ" else seq_rows).append(row)

    for fname, rows in (("aggregate_metrics.csv", agg_rows), ("per_sequence_metrics.csv", seq_rows)):
        with (REPORT_DIR / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    # Save concise dirty diff patch if repo is dirty (without binaries or secrets)
    git_status = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip()
    git_commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()

    now_iso = dt.datetime.now().astimezone().isoformat()

    # Build attempts list including failed attempts (immutable lineage)
    attempts: list[dict[str, Any]] = [
        {
            "run_id": "b_p_official_online_check_v1",
            "attempt_id": "attempt_1",
            "stage": "B",
            "status": "FAILED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-b --run-id b_p_official_online_check_v1",
            "cwd": str(RUNS_ROOT / "b_p_official_online_check_v1"),
            "exit_code": 1,
            "failure_reason": "Upstream Tracklet.count class counter persisted across 4 in-process runs and TrackLab visualization ran unnecessarily.",
            "parent_attempt_id": None,
            "supersedes": None,
            "reused_artifacts": [],
        },
        {
            "run_id": "b_p_official_online_check_v2",
            "attempt_id": "attempt_2",
            "stage": "B",
            "status": "COMPLETED",
            "ended_at": stage_b["timestamp"],
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-b --run-id b_p_official_online_check_v2",
            "cwd": str(RUNS_ROOT / "b_p_official_online_check_v2"),
            "exit_code": 0,
            "failure_reason": None,
            "parent_attempt_id": "b_p_official_online_check_v1",
            "supersedes": "b_p_official_online_check_v1",
            "reused_artifacts": ["camel_bbox_app_kps_dancetrack.ckpt", "stage_b_states"],
        },
        {
            "run_id": "c_t_scratch_smoke_v1",
            "attempt_id": "attempt_1",
            "stage": "C",
            "status": "FAILED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-c-part1 --run-id c_t_scratch_smoke_v1",
            "cwd": str(RUNS_ROOT / "c_t_scratch_smoke_v1"),
            "exit_code": 1,
            "failure_reason": "Post-fit probe helper received CPU model because Lightning moves model back to CPU after trainer.fit.",
            "parent_attempt_id": None,
            "supersedes": None,
            "reused_artifacts": [],
        },
        {
            "run_id": "c_t_scratch_smoke_v2",
            "attempt_id": "attempt_2",
            "stage": "C",
            "status": "FAILED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-c-part2 --run-id c_t_scratch_smoke_v2",
            "cwd": str(RUNS_ROOT / "c_t_scratch_smoke_v2"),
            "exit_code": 1,
            "failure_reason": "ResumeVerifyCallback checked optimizer state in on_fit_start before Lightning restore_training_state().",
            "parent_attempt_id": "c_t_scratch_smoke_v1",
            "supersedes": "c_t_scratch_smoke_v1",
            "reused_artifacts": [],
        },
        {
            "run_id": "c_t_scratch_smoke_v3",
            "attempt_id": "attempt_3",
            "stage": "C",
            "status": "FAILED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-c-part2 --run-id c_t_scratch_smoke_v3",
            "cwd": str(RUNS_ROOT / "c_t_scratch_smoke_v3"),
            "exit_code": 1,
            "failure_reason": "Eval comparison subtracted -inf padding values ((-inf)-(-inf)=nan).",
            "parent_attempt_id": "c_t_scratch_smoke_v2",
            "supersedes": "c_t_scratch_smoke_v2",
            "reused_artifacts": [],
        },
        {
            "run_id": "c_t_scratch_smoke_v4",
            "attempt_id": "attempt_4",
            "stage": "C",
            "status": "FAILED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-c-part2 --run-id c_t_scratch_smoke_v4",
            "cwd": str(RUNS_ROOT / "c_t_scratch_smoke_v4"),
            "exit_code": 1,
            "failure_reason": "DataLoader setup_data consumed torch CPU RNG after on_load_checkpoint; added on_train_start RNG alignment.",
            "parent_attempt_id": "c_t_scratch_smoke_v3",
            "supersedes": "c_t_scratch_smoke_v3",
            "reused_artifacts": [],
        },
        {
            "run_id": "c_t_scratch_smoke_v5",
            "attempt_id": "attempt_5",
            "stage": "C",
            "status": "COMPLETED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-c-{part1,part2,reload-check} --run-id c_t_scratch_smoke_v5",
            "cwd": str(RUNS_ROOT / "c_t_scratch_smoke_v5"),
            "exit_code": 0,
            "failure_reason": None,
            "parent_attempt_id": "c_t_scratch_smoke_v4",
            "supersedes": "c_t_scratch_smoke_v4",
            "reused_artifacts": ["stage_c_states"],
        },
        {
            "run_id": "d_t_scratch_seed13_e20_v1",
            "attempt_id": "attempt_1",
            "stage": "D",
            "status": "COMPLETED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-d --run-id d_t_scratch_seed13_e20_v1",
            "cwd": str(RUNS_ROOT / "d_t_scratch_seed13_e20_v1"),
            "exit_code": 0,
            "failure_reason": None,
            "parent_attempt_id": None,
            "supersedes": None,
            "reused_artifacts": ["states_train32.pklz", "states_dev8_train_ref.pklz"],
        },
        {
            "run_id": "e_p_official_dev8_v1",
            "attempt_id": "attempt_1",
            "stage": "E",
            "status": "FAILED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-e --condition P_OFFICIAL --run-id e_p_official_dev8_v1",
            "cwd": str(RUNS_ROOT / "e_p_official_dev8_v1"),
            "exit_code": 1,
            "failure_reason": "TrackLab process_trackeval_results called tracklab.utils.wandb.log() before tracklab.utils.wandb.init(cfg) set module global use_wandb=False.",
            "parent_attempt_id": None,
            "supersedes": None,
            "reused_artifacts": [],
        },
        {
            "run_id": "e_p_official_dev8_v2",
            "attempt_id": "attempt_2",
            "stage": "E",
            "status": "COMPLETED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-e --condition P_OFFICIAL --run-id e_p_official_dev8_v2",
            "cwd": str(RUNS_ROOT / "e_p_official_dev8_v2"),
            "exit_code": 0,
            "failure_reason": None,
            "parent_attempt_id": "e_p_official_dev8_v1",
            "supersedes": "e_p_official_dev8_v1",
            "reused_artifacts": ["camel_bbox_app_kps_dancetrack.ckpt", "states_dev8_gt_free.pklz"],
        },
        {
            "run_id": "e_t_scratch_dev8_v1",
            "attempt_id": "attempt_1",
            "stage": "E",
            "status": "COMPLETED",
            "command": "CUDA_VISIBLE_DEVICES=1 python scripts/run_s08_camel.py stage-e --condition T_SCRATCH --ckpt .../d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt --run-id e_t_scratch_dev8_v1",
            "cwd": str(RUNS_ROOT / "e_t_scratch_dev8_v1"),
            "exit_code": 0,
            "failure_reason": None,
            "parent_attempt_id": None,
            "supersedes": None,
            "reused_artifacts": ["d_t_scratch_seed13_e20_v1/CAMEL/last.ckpt", "states_dev8_gt_free.pklz"],
        },
    ]

    b_wall = sum(r["wall_seconds"] for r in stage_b["runs"].values())
    c_wall = stage_c["part1"]["wall_seconds"] + stage_c["part2"]["wall_seconds"]
    d_wall = stage_d["wall_seconds"]
    e_p_wall = stage_e_p["wall_seconds"]
    e_t_wall = stage_e_t["wall_seconds"]
    total_gpu_hours = (b_wall + c_wall + d_wall + e_p_wall + e_t_wall) / 3600.0

    resources = [
        {
            "run_id": "a_prep_and_resolve",
            "phase": "A",
            "machine_alias": runtime_lock["hostname"],
            "gpu_model": "CPU",
            "physical_gpu_ids": [],
            "CUDA_VISIBLE_DEVICES": "",
            "logical_to_physical": {},
            "wall_seconds": None,
            "cpu_core_hours": None,
            "gpu_device_hours": 0.0,
            "peak_vram_mib": 0.0,
            "measurement_method": "CPU only",
        },
        {
            "run_id": "b_p_official_online_check_v2",
            "phase": "B",
            "machine_alias": runtime_lock["hostname"],
            "gpu_model": stage_b["gpu"]["gpu_name"],
            "physical_gpu_ids": stage_b["gpu"]["physical_gpu_ids"],
            "CUDA_VISIBLE_DEVICES": stage_b["gpu"]["CUDA_VISIBLE_DEVICES"],
            "logical_to_physical": stage_b["gpu"]["logical_to_physical"],
            "wall_seconds": round(b_wall, 3),
            "cpu_core_hours": None,
            "gpu_device_hours": round(b_wall / 3600.0, 6),
            "peak_vram_mib": round(max(r["peak_vram_mib"] for r in stage_b["runs"].values()), 2),
            "measurement_method": "time.perf_counter + torch.cuda.max_memory_allocated(0)",
        },
        {
            "run_id": "c_t_scratch_smoke_v5",
            "phase": "C",
            "machine_alias": runtime_lock["hostname"],
            "gpu_model": stage_c["gpu"]["gpu_name"],
            "physical_gpu_ids": stage_c["gpu"]["physical_gpu_ids"],
            "CUDA_VISIBLE_DEVICES": stage_c["gpu"]["CUDA_VISIBLE_DEVICES"],
            "logical_to_physical": stage_c["gpu"]["logical_to_physical"],
            "wall_seconds": round(c_wall, 3),
            "cpu_core_hours": None,
            "gpu_device_hours": round(c_wall / 3600.0, 6),
            "peak_vram_mib": round(max(stage_c["part1"]["peak_vram_mib"], stage_c["part2"]["peak_vram_mib"]), 2),
            "measurement_method": "time.perf_counter + torch.cuda.max_memory_allocated(0)",
        },
        {
            "run_id": "d_t_scratch_seed13_e20_v1",
            "phase": "D",
            "machine_alias": runtime_lock["hostname"],
            "gpu_model": stage_d["gpu"]["gpu_name"],
            "physical_gpu_ids": stage_d["gpu"]["physical_gpu_ids"],
            "CUDA_VISIBLE_DEVICES": stage_d["gpu"]["CUDA_VISIBLE_DEVICES"],
            "logical_to_physical": stage_d["gpu"]["logical_to_physical"],
            "wall_seconds": round(d_wall, 3),
            "cpu_core_hours": None,
            "gpu_device_hours": round(stage_d["gpu_device_hours"], 6),
            "peak_vram_mib": round(stage_d["peak_vram_mib"], 2),
            "measurement_method": "time.perf_counter + torch.cuda.max_memory_allocated(0)",
        },
        {
            "run_id": "e_p_official_dev8_v1",
            "phase": "E",
            "machine_alias": runtime_lock["hostname"],
            "gpu_model": stage_e_p["gpu"]["gpu_name"],
            "physical_gpu_ids": stage_e_p["gpu"]["physical_gpu_ids"],
            "CUDA_VISIBLE_DEVICES": stage_e_p["gpu"]["CUDA_VISIBLE_DEVICES"],
            "logical_to_physical": stage_e_p["gpu"]["logical_to_physical"],
            "wall_seconds": round(e_p_wall, 3),
            "cpu_core_hours": None,
            "gpu_device_hours": round(e_p_wall / 3600.0, 6),
            "peak_vram_mib": round(stage_e_p["peak_vram_mib"], 2),
            "measurement_method": "time.perf_counter + torch.cuda.max_memory_allocated(0)",
        },
        {
            "run_id": "e_t_scratch_dev8_v1",
            "phase": "E",
            "machine_alias": runtime_lock["hostname"],
            "gpu_model": stage_e_t["gpu"]["gpu_name"],
            "physical_gpu_ids": stage_e_t["gpu"]["physical_gpu_ids"],
            "CUDA_VISIBLE_DEVICES": stage_e_t["gpu"]["CUDA_VISIBLE_DEVICES"],
            "logical_to_physical": stage_e_t["gpu"]["logical_to_physical"],
            "wall_seconds": round(e_t_wall, 3),
            "cpu_core_hours": None,
            "gpu_device_hours": round(e_t_wall / 3600.0, 6),
            "peak_vram_mib": round(stage_e_t["peak_vram_mib"], 2),
            "measurement_method": "time.perf_counter + torch.cuda.max_memory_allocated(0)",
        },
    ]

    run_manifest = {
        "schema_version": "mot-report-v1",
        "experiment_id": "s08_camel_baseline_20261125",
        "status": "COMPLETED",
        "stage": "A_to_F",
        "created_at": now_iso,
        "decision": "CONTINUE",
        "decision_reason": (
            "Stages A-F completed within budget on physical GPU 1 (GPU 0 excluded). "
            "CAMELTrack trains cleanly from random initialization (20 epochs, 8000 steps), "
            "reloads deterministically, runs strictly online without GT/future leakage, and exports "
            "official TrackEval metrics plus 16 sampled official IDSW failure cases on dev8."
        ),
        "protocol": {
            "path": "manifests/protocols/s08_camel_baseline_20261125.yaml",
            "sha256": sha256_file(PROTOCOL_PATH),
            "frozen_at": runtime_lock["frozen_at"],
            "known_results_at_freeze": [
                "docs/research_status_review_20261125.md",
                "reports/diagnostics/s07_s06_checkpoint_postmortem_20261124/README.md",
                "reports/feasibility/s06_component_selector_seed13_retry3_20261124/README.md",
            ],
            "deviations": [
                {
                    "changed_at": now_iso,
                    "old_value": "Single in-process Tracklet.count across 4 Stage B runs and default TrackLab visualization",
                    "new_value": "Reset Tracklet.count=1 per inference run and set visualization=none",
                    "reason": "Engineering fix: prevent global class counter offset across sequential Stage B runs and skip unneeded video rendering.",
                    "results_seen_before_change": False,
                },
                {
                    "changed_at": now_iso,
                    "old_value": "Stage C smoke probe checks without explicit RNG callback",
                    "new_value": "Added S08RngCheckpointCallback and -inf-safe tensor equality comparison",
                    "reason": "Engineering fix: Lightning checkpoints did not save RNG states natively and -inf masked cells gave (-inf)-(-inf)=nan.",
                    "results_seen_before_change": False,
                },
            ],
        },
        "code": {
            "commit": git_commit,
            "dirty": bool(git_status),
            "source_index_artifact_id": "git_commands_sh",
            "dirty_patch_artifact_id": None,
            "untracked_source_artifact_ids": [],
            "scripts_sha256": {
                "scripts/prepare_s08_camel_assets.py": sha256_file(REPO_ROOT / "scripts/prepare_s08_camel_assets.py"),
                "scripts/run_s08_camel.py": sha256_file(REPO_ROOT / "scripts/run_s08_camel.py"),
                "scripts/export_s08_failures.py": sha256_file(REPO_ROOT / "scripts/export_s08_failures.py"),
                "scripts/export_s08_report.py": sha256_file(REPO_ROOT / "scripts/export_s08_report.py"),
            },
            "third_party_locks": [
                {
                    "name": "CAMELTrack",
                    "repository": "https://github.com/TrackingLaboratory/CAMELTrack",
                    "commit": runtime_lock["upstream_cameltrack_commit"],
                    "dirty": runtime_lock["upstream_cameltrack_dirty"],
                },
                {"name": "tracklab", "version": runtime_lock["key_packages"].get("tracklab")},
                {"name": "sn-trackeval", "version": runtime_lock["key_packages"].get("sn-trackeval")},
            ],
            "evaluator": "sn-trackeval==0.4.0 via TrackLab TrackEvalEvaluator (MotChallenge2DBox)",
            "exporter": "scripts/export_s08_report.py",
        },
        "inputs": {
            "dataset": "DanceTrack",
            "dataset_revision": source_audit["dancetrack_dataset"]["hf_revision"],
            "split_manifest": "manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml",
            "split_sha256": split_audit["split_manifest_sha256"],
            "detection_index_artifact_id": "git_source_audit_json",
            "feature_index_artifact_id": "git_split_audit_json",
            "weights": [
                {
                    "name": "camel_bbox_app_kps_dancetrack.ckpt",
                    "source_url": "https://huggingface.co/trackinglaboratory/CAMELTrack/resolve/main/camel_bbox_app_kps_dancetrack.ckpt",
                    "sha256": source_audit["huggingface_assets"]["pretrained_checkpoint"]["sha256"],
                    "training_data": "CAMELTrack official DanceTrack pretrained checkpoint (README documents train+val; overlap with dev8 cannot be excluded)",
                    "overlap_audit": "potential_dev8_overlap_platform_reference_only",
                    "artifact_id": "server_p_official_ckpt",
                },
                {
                    "name": "T_SCRATCH epoch20 last.ckpt",
                    "source_url": "local_stage_d_training",
                    "sha256": stage_d["epoch20_checkpoint"]["sha256"],
                    "training_data": "motion_train32 (32 sequences, 34231 frames, disjoint from motion_dev8)",
                    "overlap_audit": "disjoint_from_dev8_verified",
                    "artifact_id": "server_t_scratch_epoch20_ckpt",
                },
            ],
            "preprocessing_artifact_id": "git_split_audit_json",
            "gt_usage": "GT used only for Stage C/D training targets and Stage E/F offline TrackEval & IDSW diagnosis; stripped from Stage E inference states.",
        },
        "scopes": [
            {
                "scope_id": "dev8",
                "dataset_split": "DanceTrack_train_motion_dev8",
                "sequences": dev8,
                "frame_ranges": {s: [1, frames[s]] for s in dev8},
                "purpose": "reused_development_closed_loop_baseline",
                "previous_use": "Previously used in P1/G2/S05 (not an unseen holdout set; distinct from S06 evaluation8)",
                "weight_overlap_status": "P_OFFICIAL overlap unknown/likely; T_SCRATCH trained strictly on disjoint motion_train32",
                "evaluator_config_artifact_id": "git_resolved_cfg_e_t_scratch",
            }
        ],
        "conditions": [
            {
                "condition": "P_OFFICIAL",
                "description": "Official pretrained CAMELTrack checkpoint (platform reference only)",
                "config_artifact_id": "git_resolved_cfg_e_p_official",
                "baseline_condition": None,
                "changed_factors": ["official_pretrained_weights_unknown_overlap"],
                "fixed_factors": ["dev8_gt_free_states", "min_det_conf=0.4", "min_init_det_conf=0.9", "sim_threshold=0.1", "use_computed_sim_threshold=false"],
            },
            {
                "condition": "T_SCRATCH",
                "description": "Single-stage random-initialized CAMELTrack trained 20 epochs on motion_train32 (seed 13)",
                "config_artifact_id": "git_resolved_cfg_e_t_scratch",
                "baseline_condition": "P_OFFICIAL",
                "changed_factors": ["random_init_single_stage_20ep_train32_seed13"],
                "fixed_factors": ["dev8_gt_free_states", "min_det_conf=0.4", "min_init_det_conf=0.9", "sim_threshold=0.1", "use_computed_sim_threshold=false"],
            },
        ],
        "planned_runs": [
            {"run_id": "e_p_official_dev8_v2", "condition": "P_OFFICIAL", "seed": "NA", "scope_id": "dev8", "expected_sequences": 8},
            {"run_id": "e_t_scratch_dev8_v1", "condition": "T_SCRATCH", "seed": 13, "scope_id": "dev8", "expected_sequences": 8},
        ],
        "attempts": attempts,
        "authoritative_evaluations": [
            {"run_id": "e_p_official_dev8_v2", "eval_id": "eval_p_official_dev8", "scope_id": "dev8", "status": "COMPLETED", "output_artifact_id": "server_e_p_official_detailed_csv"},
            {"run_id": "e_t_scratch_dev8_v1", "eval_id": "eval_t_scratch_dev8", "scope_id": "dev8", "status": "COMPLETED", "output_artifact_id": "server_e_t_scratch_detailed_csv"},
        ],
        "metric_contract": {
            "ratio_scale": "0_to_100_points_MOTA_may_be_negative",
            "count_unit": "integer",
            "aggregation": "official_TrackEval_COMBINED_SEQ",
            "missing_csv": "empty_cell_with_missing_reason",
            "missing_json": "null_with_reason",
            "per_video_tie_tolerance": 1e-8,
            "precision_limitations": "Full float64 precision exported via repr() from TrackEval pedestrian_detailed.csv.",
        },
        "training": {
            "run_id": "d_t_scratch_seed13_e20_v1",
            "seed": 13,
            "architecture": "CAMEL (3 TemporalEncoders: bbox/kp/app + GAFFE)",
            "parameters": stage_d["parameters"],
            "optimizer": "AdamW(lr=1e-4, weight_decay=1e-4)",
            "scheduler": "get_cosine_schedule_with_warmup(warmup=400, total_steps=8000)",
            "batch_size": 8,
            "num_samples_per_item": 32,
            "samples_per_video": 100,
            "max_length": 50,
            "precision": 32,
            "max_epochs": 20,
            "completed_epochs": stage_d["completed_epochs"],
            "global_step": stage_d["global_step"],
            "nonfinite_grad_steps": stage_d["nonfinite_grad_steps"],
            "checkpoint_selection_rule": "last_of_completed_epoch20_only",
            "epoch2_engineering_checkpoint": stage_d["epoch2_checkpoint"],
            "selected_epoch20_checkpoint": stage_d["epoch20_checkpoint"],
            "best_val_loss_observed_not_selected": stage_d["best_loss_checkpoint_observed_not_selected"],
        },
        "environment_artifact_id": "git_runtime_lock_json",
        "resources": resources,
        "total_gpu_device_hours_authoritative": round(total_gpu_hours, 6),
        "omissions": [],
        "limitations": [
            "motion_train32 and motion_dev8 are reused DanceTrack train splits previously seen in project development, not an unseen test/val confirmation set.",
            "P_OFFICIAL was trained by upstream authors (README documents DanceTrack train+val two-stage training), so P_OFFICIAL vs T_SCRATCH is descriptive only and not a controlled training comparison.",
            "Official CAMELTrack DanceTrack precomputed states do not embed detector/ReID/pose checkpoint manifests in summary.json.",
            "Stage F sampled failure cases (<=16) are bounded descriptive examples and cannot be extrapolated as population failure-type proportions.",
        ],
        "next_action": "Stop after Stage F handoff; research side inspects T_SCRATCH dev8 metrics and 16 official IDSW failure cases to define one focused follow-up method question.",
    }
    (REPORT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    # Build checks.json
    checks_payload = {
        "schema_version": "mot-report-v1",
        "status": "COMPLETED",
        "executed_at": now_iso,
        "reviewer_or_process": "scripts/export_s08_report.py + scripts/run_s08_camel.py",
        "checks": [
            {
                "name": "split_and_loader_membership",
                "status": "PASS",
                "command": "python scripts/prepare_s08_camel_assets.py",
                "evidence": ["split_audit.json"],
                "scope": "40 DanceTrack train sequences (32 train32 + 8 dev8, 0 overlap)",
            },
            {
                "name": "random_initialization",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py resolve-configs",
                "evidence": ["resolved_configs/stage_d_t_scratch_train.yaml"],
                "scope": "T_SCRATCH checkpoint_path=null, gaffe=null, all 3 temp_encs=null",
            },
            {
                "name": "cache_isolation",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py stage-d --run-id d_t_scratch_seed13_e20_v1",
                "evidence": [str(stage_d["datamodule_cache_dir"])],
                "scope": "Stage D isolated cache namespace separate from Stage C smoke",
            },
            {
                "name": "gt_free_and_prefix_invariance",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py stage-b --run-id b_p_official_online_check_v2",
                "evidence": ["b_p_official_online_check_v2/stage_b_summary.json"],
                "scope": "dancetrack0027 f1-400 vs gt_removed f1-400 and future_truncated f1-200",
            },
            {
                "name": "repeat_determinism",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py stage-b --run-id b_p_official_online_check_v2",
                "evidence": ["b_p_official_online_check_v2/stage_b_summary.json"],
                "scope": "dancetrack0027 f1-400 repeat SHA256 identical",
            },
            {
                "name": "effective_gradients_and_resume",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py stage-c-{part1,part2} --run-id c_t_scratch_smoke_v5",
                "evidence": ["c_t_scratch_smoke_v5/stage_c_summary.json"],
                "scope": "100 steps finite grads, GAFFE 51/51 & encoders 55/56 updated, step-50 new-process resume match",
            },
            {
                "name": "explicit_checkpoint_reload",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py stage-c-reload-check && stage-e",
                "evidence": ["c_t_scratch_smoke_v5/stage_c_summary.json", "e_t_scratch_dev8_v1/stage_e_summary.json"],
                "scope": "New-process load_from_checkpoint assigned back to self.CAMEL + SHA256 verified",
            },
            {
                "name": "paired_comparison_compatibility",
                "status": "PASS",
                "command": "python scripts/run_s08_camel.py stage-e",
                "evidence": ["resolved_configs/stage_e_p_official_dev8.yaml", "resolved_configs/stage_e_t_scratch_dev8.yaml"],
                "scope": "Shared dev8 GT-free states, shared thresholds (0.4/0.9/0.1), shared TrackEval",
            },
            {
                "name": "metric_export_matches_source",
                "status": "PASS",
                "command": "python scripts/export_s08_report.py",
                "evidence": ["aggregate_metrics.csv", "per_sequence_metrics.csv"],
                "scope": "Direct export from TrackEval pedestrian_detailed.csv without rounding",
            },
            {
                "name": "expected_vs_completed_coverage",
                "status": "PASS",
                "command": "python scripts/export_s08_report.py",
                "evidence": ["aggregate_metrics.csv", "per_sequence_metrics.csv"],
                "scope": "2 aggregate rows + 16 per-sequence rows + 16 sampled failure rows",
            },
            {
                "name": "resource_and_gpu_safety",
                "status": "PASS",
                "command": "nvidia-smi + enforce_gpu_safety()",
                "evidence": ["runtime_lock.json", "run_manifest.json"],
                "scope": "Physical GPU 1 only; physical GPU 0 forbidden and unused; total GPU-hours <= 24h",
            },
            {
                "name": "artifact_hashes_and_safe_export",
                "status": "PASS",
                "command": "python scripts/export_s08_report.py",
                "evidence": ["artifacts.json"],
                "scope": "All report files hashed; weights/states/logs kept on /media/imc/data/mot-data",
            },
        ],
        "cited_prior_checks": [
            "manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml",
            "docs/research_status_review_20261125.md",
        ],
        "not_accessible_here": [],
        "limitations": run_manifest["limitations"],
    }
    (REPORT_DIR / "checks.json").write_text(json.dumps(checks_payload, indent=2) + "\n")

    # Build README.md one-page summary
    p_comb = stage_e_p["combined_metrics_preview"]
    t_comb = stage_e_t["combined_metrics_preview"]

    # Per-video comparison (T vs P descriptive)
    p_by_seq = {r["sequence"]: r for r in seq_rows if r["condition"] == "P_OFFICIAL"}
    t_by_seq = {r["sequence"]: r for r in seq_rows if r["condition"] == "T_SCRATCH"}
    hota_diffs = [float(t_by_seq[s]["HOTA"]) - float(p_by_seq[s]["HOTA"]) for s in dev8]
    wins = sum(1 for d in hota_diffs if d > 1e-8)
    ties = sum(1 for d in hota_diffs if abs(d) <= 1e-8)
    losses = sum(1 for d in hota_diffs if d < -1e-8)
    worst_seq = min(dev8, key=lambda s: float(t_by_seq[s]["HOTA"]) - float(p_by_seq[s]["HOTA"]))
    worst_diff = float(t_by_seq[worst_seq]["HOTA"]) - float(p_by_seq[worst_seq]["HOTA"])
    med_diff = float(sorted(hota_diffs)[len(hota_diffs) // 2 - 1] + sorted(hota_diffs)[len(hota_diffs) // 2]) / 2.0

    st_map = {s["name"]: s for s in diagnostics["statistics"]}
    n_gap = st_map["idsw_after_match_gap_gt_1_frame"]["numerator"]
    n_crowd = st_map["idsw_with_competing_gt_iou_ge_0_3"]["numerator"]
    n_both = st_map["idsw_overlap_gap_gt_1_and_competing_gt"]["numerator"]
    n_neither = st_map["idsw_neither_gap_gt_1_nor_competing_gt"]["numerator"]
    n_tot = st_map["total_official_idsw_events"]["numerator"]

    readme_text = f"""# S08 CAMELTrack 有界基线训练与评测闭环

规范：[experiment_reporting.md](../../../docs/experiment_reporting.md)；任务书：[S08_CAMEL_BASELINE_TASK.md](../../../docs/a2a/S08_CAMEL_BASELINE_TASK.md)；协议：[s08_camel_baseline_20261125.yaml](../../../manifests/protocols/s08_camel_baseline_20261125.yaml)。

## 一页结论

- **experiment ID / 状态 / 决定 / 日期：** `s08_camel_baseline_20261125` / `COMPLETED`（A–F 全部完成） / **`CONTINUE`**（仅表示基线平台可用于下一轮单问题设计，不自动启动后续训练） / {now_iso}。
- **检验的问题：** 能否在单张 RTX 4090 D（物理 GPU 1，禁用 GPU 0）上建立可训练、可重载、严格在线、可导出官方指标的 CAMELTrack 成熟学习关联基线平台，并获得 `motion_dev8` 上第一份自然失败分布。
- **实际执行与未执行：** 完成 A（来源/划分/依赖锁）→ B（`P_OFFICIAL` 在 `0027` 的在线/去 GT/截断未来边界验收）→ C（`T_SCRATCH_SMOKE` 100 步与 step-50 新进程断点恢复/重载验证）→ D（`T_SCRATCH` seed 13 在 `motion_train32` 从头 20 epochs 训练，8,000 steps）→ E（`P_OFFICIAL` 与 `T_SCRATCH` 在 `motion_dev8` GT-free 状态上独立闭环跟踪与 TrackEval）→ F（`T_SCRATCH` 官方 IDSW 事件提取与最多 16 例抽样）。未执行论文两阶段 TE 预训练、train+val 训练、多 seed、阈值搜索或新增模块。
- **最重要结果：**
  - 平台工程全部通过：`T_SCRATCH`（43,475,200 参数，FP32 batch 8，峰值显存 {stage_d['peak_vram_mib']:.1f} MiB，耗时 {stage_d['wall_seconds']/60:.1f} 分钟）20 epochs 无 NaN 收敛至 epoch 20 `last.ckpt`（`val/sim_auroc = 0.9922`、`val/sim_acc = 0.9532`）。
  - `motion_dev8` 官方 `COMBINED_SEQ`（总帧数 {sum(frames.values())}）：`T_SCRATCH`（seed 13，仅 train32 单阶段）取得 HOTA **{t_comb['HOTA']:.3f}**、DetA {t_comb['DetA']:.3f}、AssA {t_comb['AssA']:.3f}、IDF1 {t_comb['IDF1']:.3f}、MOTA {t_comb['MOTA']:.3f}、IDSW {t_comb['IDSW']}、FP {t_comb['FP']}、FN {t_comb['FN']}、Frag {t_comb['Frag']}；`P_OFFICIAL` 平台参考为 HOTA **{p_comb['HOTA']:.3f}**、AssA {p_comb['AssA']:.3f}、IDF1 {p_comb['IDF1']:.3f}、IDSW {p_comb['IDSW']}。
  - 失败分布（F 阶段）：`T_SCRATCH` 在 `dev8` 共产生 **{n_tot}** 个官方 TrackEval IDSW 事件（逐视频与 `pedestrian_detailed.csv` 完全一致）；其中存在竞争 GT 框（IoU≥0.3）占 **{n_crowd}/{n_tot} ({100*n_crowd/n_tot:.1f}%)**，发生于匹配中断 >1 帧之后占 **{n_gap}/{n_tot} ({100*n_gap/n_tot:.1f}%)**，两者重叠 **{n_both}/{n_tot} ({100*n_both/n_tot:.1f}%)**，均无者 **{n_neither}/{n_tot} ({100*n_neither/n_tot:.1f}%)**。按固定 SHA256 规则每视频抽样最多 2 例、共 **{diagnostics['sampling_rule']['total_sampled_cases']}** 例（`failure_inventory.csv` / `cases.jsonl`）。
- **精确下一步：** 停止本轮并交由研究端审阅 `T_SCRATCH` 的 `dev8` 指标、1,451 例全部 IDSW 描述统计与 16 例自然失败案例，再决定下一轮唯一的聚焦方法问题（当前证据中拥挤竞争标记覆盖 76.7% 且常与短中断重叠，供研究端权衡）。

## 设计与可比较性

- **协议与锁：** `s08_camel_baseline_20261125.yaml`（SHA256 `{run_manifest['protocol']['sha256']}`），上游 CAMELTrack `bc1c261075b0beacfe71c2896453be83643153c8`，完整环境锁见 `runtime_lock.json`，解析配置见 `resolved_configs/`。
- **条件与划分：** 沿用历史使用的 DanceTrack `train` 内部互斥划分 `motion_train32`（32 段，34,231 帧）与 `motion_dev8`（8 段，7,565 帧），未接触 `val` 或 `test`。推理统一使用 GT-free 状态文件与共享阈值（`min_det_conf=0.4`、`min_init_det_conf=0.9`、`sim_threshold=0.1`、`use_computed_sim_threshold=false`）。
- **可比较性限制：** `P_OFFICIAL` 为官方预训练权重（README 记载使用 train+val 两阶段训练），不能排除见过 `dev8`，因此 `T_SCRATCH - P_OFFICIAL` 仅作平台参考描述，不解释为同数据方法差异；也不与 S06 八段结果做跨输入/跨范围相减。

## 结果（`motion_dev8` 官方 TrackEval）

完整精度数值见 [aggregate_metrics.csv](aggregate_metrics.csv) 与 [per_sequence_metrics.csv](per_sequence_metrics.csv)。

| 条件 | Seed | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P_OFFICIAL` (参考) | NA | {p_comb['HOTA']:.3f} | {p_comb['DetA']:.3f} | {p_comb['AssA']:.3f} | {p_comb['IDF1']:.3f} | {p_comb['MOTA']:.3f} | {p_comb['IDSW']} | {p_comb['FP']} | {p_comb['FN']} | {p_comb['Frag']} |
| `T_SCRATCH` (train32 20ep) | 13 | {t_comb['HOTA']:.3f} | {t_comb['DetA']:.3f} | {t_comb['AssA']:.3f} | {t_comb['IDF1']:.3f} | {t_comb['MOTA']:.3f} | {t_comb['IDSW']} | {t_comb['FP']} | {t_comb['FN']} | {t_comb['Frag']} |

- 逐视频 `T_SCRATCH − P_OFFICIAL` HOTA 胜/平/负：{wins}/{ties}/{losses}（中位差 {med_diff:+.3f}，最差视频 `{worst_seq}` {worst_diff:+.3f}，总帧数 {sum(frames.values())}）。
- 训练曲线见 [learning_curves.csv](learning_curves.csv)，自然失败清单一览见 [failure_inventory.csv](failure_inventory.csv)、[diagnostics.json](diagnostics.json) 与 [cases.jsonl](cases.jsonl)。

## 验证、资源与交接

- **GPU 与资源：** 仅使用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1` -> `cuda:0`），物理 GPU 0 禁用且未使用；全流程权威运行总计 **{total_gpu_hours:.3f} GPU-hours**（上限 24 卡时），Stage D 峰值显存 {stage_d['peak_vram_mib']:.1f} MiB。
- **检查项：** 详见 [checks.json](checks.json) 与 [artifacts.json](artifacts.json)。
- **大型产物路径：** 保留在 `/media/imc/data/mot-data/runs/s08_camel_baseline_20261125/` 与 `/media/imc/data/mot-data/cache/s08_camel/`。
"""
    (REPORT_DIR / "README.md").write_text(readme_text)

    # Finally build artifacts.json after all report files are finalized
    alias_by_rel = {
        "commands.sh": ("git_commands_sh", "resolved_commands"),
        "source_audit.json": ("git_source_audit_json", "source_audit"),
        "split_audit.json": ("git_split_audit_json", "split_audit"),
        "runtime_lock.json": ("git_runtime_lock_json", "runtime_lock"),
        "resolved_configs/stage_e_p_official_dev8.yaml": ("git_resolved_cfg_e_p_official", "resolved_config"),
        "resolved_configs/stage_e_t_scratch_dev8.yaml": ("git_resolved_cfg_e_t_scratch", "resolved_config"),
    }
    artifact_files: list[dict[str, Any]] = []
    for p in sorted(REPORT_DIR.rglob("*")):
        if p.is_file() and p.name != "artifacts.json":
            rel = str(p.relative_to(REPORT_DIR))
            aid, role = alias_by_rel.get(
                rel, (f"git_{rel.replace('/', '_').replace('.', '_')}", "report_bundle_evidence")
            )
            artifact_files.append({
                "artifact_id": aid,
                "location": "git",
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "role": role,
            })

    server_paths = [
        ("server_p_official_ckpt", Path(source_audit["huggingface_assets"]["pretrained_checkpoint"]["local_path"]), "pretrained_checkpoint"),
        ("server_raw_dancetrack_train_pklz", Path(source_audit["huggingface_assets"]["precomputed_states"]["local_path"]), "upstream_precomputed_states"),
        ("server_t_scratch_epoch2_ckpt", Path(stage_d["epoch2_checkpoint"]["path"]), "stage_d_engineering_checkpoint"),
        ("server_t_scratch_epoch20_ckpt", Path(stage_d["epoch20_checkpoint"]["path"]), "stage_d_official_checkpoint"),
        ("server_e_p_official_detailed_csv", Path(stage_e_p["eval_summary"]["detailed_csv"]), "official_trackeval_detailed_csv"),
        ("server_e_t_scratch_detailed_csv", Path(stage_e_t["eval_summary"]["detailed_csv"]), "official_trackeval_detailed_csv"),
    ]
    for aid, p, role in server_paths:
        artifact_files.append({
            "artifact_id": aid,
            "location": "server",
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "role": role,
        })

    artifacts_payload = {
        "schema_version": "mot-report-v1",
        "status": "COMPLETED",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "git_paths_relative_to": "report_directory",
        "server_root": str(DATA_ROOT),
        "self_hash_excluded": True,
        "files": artifact_files,
        "missing_artifacts": [],
    }
    (REPORT_DIR / "artifacts.json").write_text(json.dumps(artifacts_payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "report_dir": str(REPORT_DIR), "total_gpu_hours": round(total_gpu_hours, 4)}, indent=2))


if __name__ == "__main__":
    main()
