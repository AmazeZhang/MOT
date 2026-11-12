#!/usr/bin/env python3
"""Launch, inspect and aggregate failure-isolated GPU experiment jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from mot_research.experiments.async_queue import (
    aggregate_passed_jobs,
    atomic_write_json,
    file_hash_index,
    latest_statuses,
    next_attempt_directory,
    queue_directory,
    read_status,
    sha256_file,
    utc_now,
    validate_queue_manifest,
)


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_queue_manifest(yaml.safe_load(path.read_text()))


def git_state(project_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_root, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": head, "clean": not bool(porcelain), "porcelain": porcelain.splitlines()}


def gpu_inventory() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    devices = []
    uuid_to_index: dict[str, int] = {}
    for line in gpu.stdout.splitlines():
        index, uuid, name, memory, utilization = [item.strip() for item in line.split(",", 4)]
        uuid_to_index[uuid] = int(index)
        devices.append({
            "index": int(index), "uuid": uuid, "name": name,
            "memory_used_mib": int(memory), "utilization_percent": int(utilization),
        })
    processes_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    processes = []
    for line in processes_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory, process_name = [item.strip() for item in line.split(",", 3)]
        processes.append({
            "physical_gpu": uuid_to_index.get(uuid), "gpu_uuid": uuid,
            "pid": int(pid), "used_memory_mib": int(memory), "process_name": process_name,
        })
    return {"devices": devices, "compute_processes": processes}


def ensure_queue_snapshot(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    root = queue_directory(manifest)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "queue_manifest.yaml"
    source_hash = sha256_file(manifest_path)
    if snapshot.exists() and sha256_file(snapshot) != source_hash:
        raise RuntimeError("queue manifest snapshot exists with a different hash")
    if not snapshot.exists():
        temporary = snapshot.with_name(snapshot.name + f".tmp-{os.getpid()}")
        shutil.copyfile(manifest_path, temporary)
        os.replace(temporary, snapshot)
    atomic_write_json(root / "queue_metadata.json", {
        "queue_id": manifest["queue_id"],
        "manifest_source": str(manifest_path.resolve()),
        "manifest_sha256": source_hash,
        "project_root": manifest["project_root"],
        "created_or_checked_at_utc": utc_now(),
    })
    return root


def select_jobs(manifest: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return list(manifest["jobs"])
    wanted = set(requested)
    jobs = [job for job in manifest["jobs"] if job["job_id"] in wanted]
    missing = wanted - {job["job_id"] for job in jobs}
    if missing:
        raise ValueError(f"unknown jobs: {sorted(missing)}")
    return jobs


def command_launch(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    project_root = Path(manifest["project_root"])
    state = git_state(project_root)
    if manifest["git_commit"] and state["commit"] != manifest["git_commit"]:
        raise RuntimeError(f"Git commit mismatch: {state['commit']} != {manifest['git_commit']}")
    if manifest["require_clean_git"] and not state["clean"]:
        raise RuntimeError("queue requires a clean Git worktree")
    root = ensure_queue_snapshot(args.manifest, manifest)
    inventory = gpu_inventory()
    atomic_write_json(root / "gpu_inventory_at_launch.json", inventory)
    selected = select_jobs(manifest, args.jobs)
    selected_gpus = {job["physical_gpu"] for job in selected}
    busy = [
        process for process in inventory["compute_processes"]
        if process["physical_gpu"] in selected_gpus
    ]
    if busy and not args.allow_busy:
        raise RuntimeError(f"selected GPUs already have compute processes: {busy}")

    launched = []
    script_path = Path(__file__).resolve()
    for job in selected:
        prior_attempts = (root / "jobs" / job["job_id"] / "attempts")
        if prior_attempts.exists():
            latest_dirs = sorted(path for path in prior_attempts.iterdir() if path.is_dir())
            if latest_dirs:
                latest = read_status(latest_dirs[-1])
                if latest.get("state") in {"PENDING", "RUNNING"}:
                    raise RuntimeError(f"{job['job_id']}: latest attempt is still active")
                if latest.get("state") == "PASS" and not args.retry_passed:
                    raise RuntimeError(f"{job['job_id']}: already PASS; use --retry-passed explicitly")
        attempt = next_attempt_directory(manifest, job["job_id"])
        attempt.mkdir(parents=True, exist_ok=False)
        resolved = {
            **job,
            "queue_id": manifest["queue_id"],
            "project_root": manifest["project_root"],
            "git": state,
            "manifest_sha256": sha256_file(args.manifest),
            "attempt": attempt.name,
            "attempt_dir": str(attempt),
            "created_at_utc": utc_now(),
            "gpu_inventory_at_launch": inventory,
        }
        resolved_path = attempt / "resolved_job.json"
        atomic_write_json(resolved_path, resolved)
        atomic_write_json(attempt / "status.json", {
            "state": "PENDING", "job_id": job["job_id"],
            "physical_gpu": job["physical_gpu"], "attempt": attempt.name,
            "created_at_utc": utc_now(),
        })
        process = subprocess.Popen(
            [sys.executable, str(script_path), "_run-job", "--resolved-job", str(resolved_path)],
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=os.environ.copy(),
        )
        atomic_write_json(attempt / "launch.json", {
            "worker_pid": process.pid, "launched_at_utc": utc_now(),
            "launcher_pid": os.getpid(),
        })
        launched.append({
            "job_id": job["job_id"], "physical_gpu": job["physical_gpu"],
            "attempt": attempt.name, "attempt_dir": str(attempt), "worker_pid": process.pid,
        })
    print(json.dumps({"queue_id": manifest["queue_id"], "launched": launched}, indent=2))


def command_run_job(args: argparse.Namespace) -> None:
    resolved_path = args.resolved_job.resolve()
    job = json.loads(resolved_path.read_text())
    attempt = Path(job["attempt_dir"])
    status_path = attempt / "status.json"
    artifact_partial = attempt / "artifacts.partial"
    artifact_final = attempt / "artifacts"
    artifact_failed = attempt / "artifacts.failed"
    try:
        artifact_partial.mkdir()
        environment = os.environ.copy()
        environment.update(job["environment"])
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(job["physical_gpu"]),
            "MOT_PHYSICAL_GPU": str(job["physical_gpu"]),
            "MOT_PROCESS_DEVICE": "cuda:0",
            "MOT_QUEUE_ID": job["queue_id"],
            "MOT_JOB_ID": job["job_id"],
            "MOT_ATTEMPT": job["attempt"],
            "MOT_JOB_ARTIFACT_DIR": str(artifact_partial),
        })
        atomic_write_json(status_path, {
            "state": "RUNNING", "job_id": job["job_id"],
            "physical_gpu": job["physical_gpu"], "CUDA_VISIBLE_DEVICES": str(job["physical_gpu"]),
            "process_device": "cuda:0", "attempt": job["attempt"],
            "worker_pid": os.getpid(), "started_at_utc": utc_now(),
            "command": job["command"],
        })
        with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen(
                job["command"], cwd=job["project_root"], env=environment,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                shell=False,
            )
            return_code = child.wait()
        missing = [
            path for path in job["expected_artifacts"]
            if not (artifact_partial / path).is_file()
        ]
        passed = return_code == 0 and not missing
        if passed:
            os.replace(artifact_partial, artifact_final)
            artifact_hashes = file_hash_index(artifact_final)
            state = "PASS"
        else:
            os.replace(artifact_partial, artifact_failed)
            artifact_hashes = file_hash_index(artifact_failed)
            state = "FAIL"
        atomic_write_json(status_path, {
            "state": state, "job_id": job["job_id"],
            "physical_gpu": job["physical_gpu"], "CUDA_VISIBLE_DEVICES": str(job["physical_gpu"]),
            "process_device": "cuda:0", "attempt": job["attempt"],
            "worker_pid": os.getpid(), "child_pid": child.pid,
            "started_at_utc": json.loads(status_path.read_text())["started_at_utc"],
            "completed_at_utc": utc_now(), "return_code": return_code,
            "expected_artifacts": job["expected_artifacts"],
            "missing_expected_artifacts": missing,
            "artifact_hashes": artifact_hashes,
        })
    except Exception as error:
        if artifact_partial.exists() and not artifact_failed.exists():
            os.replace(artifact_partial, artifact_failed)
        atomic_write_json(status_path, {
            "state": "FAIL", "job_id": job.get("job_id"),
            "physical_gpu": job.get("physical_gpu"), "attempt": job.get("attempt"),
            "worker_pid": os.getpid(), "completed_at_utc": utc_now(),
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        raise


def command_status(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    print(json.dumps({
        "queue_id": manifest["queue_id"],
        "checked_at_utc": utc_now(),
        "jobs": latest_statuses(manifest),
    }, indent=2))


def command_wait(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    deadline = time.monotonic() + args.timeout
    while True:
        statuses = latest_statuses(manifest)
        if all(status["state"] in {"PASS", "FAIL", "INTERRUPTED"} for status in statuses):
            print(json.dumps({"queue_id": manifest["queue_id"], "jobs": statuses}, indent=2))
            return
        if time.monotonic() >= deadline:
            print(json.dumps({"queue_id": manifest["queue_id"], "jobs": statuses}, indent=2))
            raise TimeoutError("queue wait timeout")
        time.sleep(args.poll_interval)


def command_aggregate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    aggregate = aggregate_passed_jobs(manifest)
    root = queue_directory(manifest)
    atomic_write_json(root / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--jobs", nargs="*")
    launch.add_argument("--allow-busy", action="store_true")
    launch.add_argument("--retry-passed", action="store_true")
    launch.set_defaults(function=command_launch)
    status = subparsers.add_parser("status")
    status.add_argument("--manifest", type=Path, required=True)
    status.set_defaults(function=command_status)
    wait = subparsers.add_parser("wait")
    wait.add_argument("--manifest", type=Path, required=True)
    wait.add_argument("--timeout", type=float, default=60.0)
    wait.add_argument("--poll-interval", type=float, default=0.5)
    wait.set_defaults(function=command_wait)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.set_defaults(function=command_aggregate)
    worker = subparsers.add_parser("_run-job")
    worker.add_argument("--resolved-job", type=Path, required=True)
    worker.set_defaults(function=command_run_job)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

