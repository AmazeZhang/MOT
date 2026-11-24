#!/usr/bin/env python3
"""Create non-overwriting S05-A retry locks after an environment-only failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--failure-reason", required=True)
    args = parser.parse_args()
    if args.attempt < 2:
        raise ValueError("retry attempt must be >= 2")
    source_root = args.run_root / "tracking"
    retry_root = args.run_root / f"tracking_attempt{args.attempt}"
    if retry_root.exists():
        raise FileExistsError(retry_root)
    for source_dir in sorted(source_root.glob("dancetrack*")):
        source_lock_path = source_dir / "runtime_lock.json"
        lock = json.loads(source_lock_path.read_text())
        sequence = lock["selected_sequences"][0]
        output = retry_root / sequence
        output.mkdir(parents=True)
        lock.update({
            "run_id": f"s05a_stability_{sequence}_attempt{args.attempt}",
            "status": "locked_before_tracking_retry",
            "output_root": str(output),
            "parent_attempt_id": f"s05a_stability_{sequence}_attempt1",
            "retry_reason": args.failure_reason,
            "environment_change_only": "set PYTHONPATH to repository src directory",
        })
        (output / "runtime_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED_RETRY", "attempt": args.attempt,
                      "sequences": len(list(retry_root.iterdir()))}, sort_keys=True))


if __name__ == "__main__":
    main()
