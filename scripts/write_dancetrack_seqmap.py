#!/usr/bin/env python3
"""Write a TrackEval seqmap from the sequence directories in a split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.split_dir.is_dir():
        raise FileNotFoundError(args.split_dir)
    names = sorted(path.name for path in args.split_dir.iterdir() if path.is_dir())
    if not names:
        raise RuntimeError(f"no sequence directories under {args.split_dir}")
    partial = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {partial}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as handle:
        handle.write("name\n")
        handle.writelines(f"{name}\n" for name in names)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, args.output)
    print(f"wrote {len(names)} sequences to {args.output}")


if __name__ == "__main__":
    main()
