from __future__ import annotations

import json
from pathlib import Path

import pytest

from mot_research.experiments.async_queue import (
    aggregate_passed_jobs,
    atomic_write_json,
    validate_queue_manifest,
)


def manifest(tmp_path: Path) -> dict:
    return {
        "queue_id": "queue_test",
        "project_root": str(tmp_path / "project"),
        "queue_root": str(tmp_path / "runs"),
        "require_clean_git": False,
        "git_commit": "",
        "jobs": [
            {
                "job_id": "job1", "physical_gpu": 1,
                "command": ["python", "job.py"],
                "expected_artifacts": ["metrics.json"],
            },
            {
                "job_id": "job2", "physical_gpu": 2,
                "command": ["python", "job.py"],
                "expected_artifacts": ["metrics.json"],
            },
        ],
    }


def test_gpu_zero_and_duplicate_gpu_are_rejected(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    value["jobs"][0]["physical_gpu"] = 0
    with pytest.raises(ValueError, match="GPU 0"):
        validate_queue_manifest(value)
    value = manifest(tmp_path)
    value["jobs"][1]["physical_gpu"] = 1
    with pytest.raises(ValueError, match="multiple concurrent jobs"):
        validate_queue_manifest(value)


def test_queue_controls_cuda_and_artifact_paths(tmp_path: Path) -> None:
    value = manifest(tmp_path)
    value["jobs"][0]["environment"] = {"CUDA_VISIBLE_DEVICES": "7"}
    with pytest.raises(ValueError, match="controlled by the queue"):
        validate_queue_manifest(value)
    value = manifest(tmp_path)
    value["jobs"][0]["expected_artifacts"] = ["../escape.json"]
    with pytest.raises(ValueError, match="safe relative path"):
        validate_queue_manifest(value)


def test_aggregate_ingests_only_pass_attempts(tmp_path: Path) -> None:
    value = validate_queue_manifest(manifest(tmp_path))
    for job_id, state in (("job1", "PASS"), ("job2", "FAIL")):
        attempt = Path(value["queue_root"]) / value["queue_id"] / "jobs" / job_id / "attempts" / "attempt_001"
        artifact = attempt / ("artifacts" if state == "PASS" else "artifacts.failed")
        artifact.mkdir(parents=True)
        (artifact / "metrics.json").write_text(json.dumps({"job": job_id}) + "\n")
        atomic_write_json(attempt / "status.json", {"state": state, "job_id": job_id})
    aggregate = aggregate_passed_jobs(value)
    assert aggregate["passed_job_count"] == 1
    assert aggregate["passed_jobs"][0]["job_id"] == "job1"
    assert aggregate["missing_or_failed_jobs"] == [
        {"job_id": "job2", "physical_gpu": 2, "latest_state": "FAIL"}
    ]

