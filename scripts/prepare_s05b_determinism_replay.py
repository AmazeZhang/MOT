#!/usr/bin/env python3
"""Prepare a non-overwriting identical S05-B replay for determinism evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-check", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    source_lock = json.loads(args.source_lock.read_text())
    source_check = json.loads(args.source_check.read_text())
    sequence = source_lock["selected_sequences"][0]
    output = args.output_root / sequence
    output.mkdir(parents=True)
    source_lock.update({
        "run_id": f"s05b_candidate_coverage_{sequence}_determinism_attempt2",
        "status": "locked_identical_determinism_replay",
        "output_root": str(output),
        "parent_attempt_id": source_check["run_id"],
        "retry_reason": None,
        "purpose": "repeat identical replay and require tracker plus trace SHA256 equality",
        "expected_parent_track_sha256": source_check["result"]["track_sha256"],
        "expected_parent_trace_sha256": source_check["result"]["trace_sha256"],
    })
    (output / "runtime_lock.json").write_text(json.dumps(source_lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "sequence": sequence,
                      "run_id": source_lock["run_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
