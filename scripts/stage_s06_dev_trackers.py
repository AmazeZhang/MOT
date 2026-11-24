#!/usr/bin/env python3
"""Stage checked S06 dev tracker files by model/checkpoint for TrackEval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


DEV6 = ["dancetrack0008", "dancetrack0039", "dancetrack0055",
        "dancetrack0066", "dancetrack0082", "dancetrack0099"]


def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    records = []
    for mode in ("Local", "Context"):
        for epoch in range(5, 31, 5):
            method = f"{mode}_risk_e{epoch:02d}"
            target_root = args.output_root / method
            target_root.mkdir(parents=True)
            for sequence in DEV6:
                root = args.tracking_root / f"{mode}_e{epoch:02d}" / sequence
                check = json.loads((root / "engineering_checks.json").read_text())
                if check["status"] != "ENGINEERING_PASS":
                    raise RuntimeError(f"incomplete {method}/{sequence}")
                source = root / "trackers/M" / f"{sequence}.txt"
                if sha(source) != check["result"]["track_sha256"]:
                    raise RuntimeError("tracker hash mismatch")
                target = target_root / f"{sequence}.txt"
                shutil.copy2(source, target)
                records.append({"method": method, "sequence": sequence,
                                "source": str(source), "sha256": sha(target),
                                "changed_components": check["selector_changed_components"]})
    (args.output_root / "manifest.json").write_text(json.dumps(
        {"status": "COMPLETED", "records": records}, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "trackers": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
