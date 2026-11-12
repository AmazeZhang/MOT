"""Failure-isolated asynchronous job queue helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOB_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
TERMINAL_STATES = {"PASS", "FAIL", "INTERRUPTED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_relative_artifact(path_text: str) -> None:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts or path_text in {"", "."}:
        raise ValueError(f"expected artifact must be a safe relative path: {path_text!r}")


def validate_queue_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the static queue contract."""
    queue_id = str(manifest.get("queue_id", ""))
    if not JOB_ID_PATTERN.fullmatch(queue_id):
        raise ValueError(f"invalid queue_id: {queue_id!r}")
    project_root = Path(str(manifest.get("project_root", "")))
    queue_root = Path(str(manifest.get("queue_root", "")))
    if not project_root.is_absolute() or not queue_root.is_absolute():
        raise ValueError("project_root and queue_root must be absolute")
    try:
        queue_root.resolve().relative_to(project_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("queue_root must be outside the Git project")

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")
    seen_ids: set[str] = set()
    seen_gpus: set[int] = set()
    normalized_jobs: list[dict[str, Any]] = []
    for raw in jobs:
        if not isinstance(raw, dict):
            raise ValueError("each job must be a mapping")
        job_id = str(raw.get("job_id", ""))
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError(f"invalid job_id: {job_id!r}")
        if job_id in seen_ids:
            raise ValueError(f"duplicate job_id: {job_id}")
        seen_ids.add(job_id)
        physical_gpu = int(raw.get("physical_gpu", -1))
        if physical_gpu == 0:
            raise ValueError(f"{job_id}: physical GPU 0 is permanently forbidden")
        if physical_gpu not in range(1, 8):
            raise ValueError(f"{job_id}: physical_gpu must be one of 1..7")
        if physical_gpu in seen_gpus:
            raise ValueError(f"physical GPU {physical_gpu} assigned to multiple concurrent jobs")
        seen_gpus.add(physical_gpu)
        command = raw.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError(f"{job_id}: command must be a non-empty string list")
        environment = raw.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError(f"{job_id}: environment must be a mapping")
        if "CUDA_VISIBLE_DEVICES" in environment:
            raise ValueError(f"{job_id}: CUDA_VISIBLE_DEVICES is controlled by the queue")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
            raise ValueError(f"{job_id}: environment keys and values must be strings")
        expected = raw.get("expected_artifacts", ["metrics.json"])
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{job_id}: expected_artifacts must be a non-empty list")
        for artifact in expected:
            if not isinstance(artifact, str):
                raise ValueError(f"{job_id}: expected artifact paths must be strings")
            _validate_relative_artifact(artifact)
        normalized_jobs.append({
            "job_id": job_id,
            "physical_gpu": physical_gpu,
            "command": command,
            "environment": environment,
            "expected_artifacts": expected,
            "description": str(raw.get("description", "")),
        })
    return {
        "queue_id": queue_id,
        "project_root": str(project_root),
        "queue_root": str(queue_root),
        "require_clean_git": bool(manifest.get("require_clean_git", True)),
        "git_commit": str(manifest.get("git_commit", "")),
        "jobs": normalized_jobs,
    }


def queue_directory(manifest: dict[str, Any]) -> Path:
    return Path(manifest["queue_root"]) / manifest["queue_id"]


def job_directory(manifest: dict[str, Any], job_id: str) -> Path:
    return queue_directory(manifest) / "jobs" / job_id


def attempt_directories(manifest: dict[str, Any], job_id: str) -> list[Path]:
    root = job_directory(manifest, job_id) / "attempts"
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"attempt_[0-9]{3}", path.name)),
        key=lambda path: path.name,
    )


def next_attempt_directory(manifest: dict[str, Any], job_id: str) -> Path:
    attempts = attempt_directories(manifest, job_id)
    number = int(attempts[-1].name.split("_")[-1]) + 1 if attempts else 1
    return job_directory(manifest, job_id) / "attempts" / f"attempt_{number:03d}"


def read_status(attempt_dir: Path) -> dict[str, Any]:
    path = attempt_dir / "status.json"
    if not path.is_file():
        return {"state": "UNKNOWN", "attempt_dir": str(attempt_dir)}
    value = json.loads(path.read_text())
    value["attempt_dir"] = str(attempt_dir)
    return value


def latest_statuses(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        attempts = attempt_directories(manifest, job["job_id"])
        if not attempts:
            statuses.append({
                "job_id": job["job_id"],
                "physical_gpu": job["physical_gpu"],
                "state": "PENDING",
                "attempt_dir": None,
            })
            continue
        status = read_status(attempts[-1])
        status.setdefault("job_id", job["job_id"])
        status.setdefault("physical_gpu", job["physical_gpu"])
        statuses.append(status)
    return statuses


def file_hash_index(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def aggregate_passed_jobs(manifest: dict[str, Any]) -> dict[str, Any]:
    """Select the newest PASS attempt per job and never ingest failed artifacts."""
    passed: list[dict[str, Any]] = []
    missing_or_failed: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        attempts = attempt_directories(manifest, job["job_id"])
        selected: tuple[Path, dict[str, Any]] | None = None
        for attempt in reversed(attempts):
            status = read_status(attempt)
            if status.get("state") == "PASS":
                selected = (attempt, status)
                break
        if selected is None:
            latest = read_status(attempts[-1]) if attempts else {"state": "PENDING"}
            missing_or_failed.append({
                "job_id": job["job_id"],
                "physical_gpu": job["physical_gpu"],
                "latest_state": latest.get("state", "UNKNOWN"),
            })
            continue
        attempt, status = selected
        artifact_root = attempt / "artifacts"
        metrics_path = artifact_root / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
        passed.append({
            "job_id": job["job_id"],
            "physical_gpu": job["physical_gpu"],
            "attempt": attempt.name,
            "attempt_dir": str(attempt),
            "status_sha256": sha256_file(attempt / "status.json"),
            "artifact_hashes": file_hash_index(artifact_root),
            "metrics": metrics,
            "completed_at_utc": status.get("completed_at_utc"),
        })
    return {
        "queue_id": manifest["queue_id"],
        "generated_at_utc": utc_now(),
        "aggregation_policy": "newest PASS attempt per job; non-PASS artifacts excluded",
        "complete": not missing_or_failed,
        "passed_job_count": len(passed),
        "required_job_count": len(manifest["jobs"]),
        "passed_jobs": passed,
        "missing_or_failed_jobs": missing_or_failed,
    }

