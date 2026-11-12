#!/usr/bin/env python3
"""Validate a frozen motion train/dev manifest against local DanceTrack train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mot_research.data.motion_split import validate_motion_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    summary = validate_motion_split(manifest, args.train_root)
    summary["manifest"] = str(args.manifest.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

