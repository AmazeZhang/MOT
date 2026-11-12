#!/usr/bin/env python3
"""Tiny job used only to verify asynchronous queue isolation and provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("pass", "fail"), required=True)
    args = parser.parse_args()
    artifact_dir = Path(os.environ["MOT_JOB_ARTIFACT_DIR"])
    metrics = {
        "engineering_smoke_only": True,
        "requested_outcome": args.outcome,
        "queue_id": os.environ["MOT_QUEUE_ID"],
        "job_id": os.environ["MOT_JOB_ID"],
        "attempt": os.environ["MOT_ATTEMPT"],
        "physical_gpu": int(os.environ["MOT_PHYSICAL_GPU"]),
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "process_device_label": os.environ["MOT_PROCESS_DEVICE"],
    }
    with (artifact_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    if args.outcome == "fail":
        raise RuntimeError("intentional queue-isolation smoke failure")


if __name__ == "__main__":
    main()

