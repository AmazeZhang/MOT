#!/usr/bin/env python3
"""Stage all seven frozen S06 evaluation8 conditions for official TrackEval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


EVAL8 = ["dancetrack0074", "dancetrack0032", "dancetrack0069", "dancetrack0015",
         "dancetrack0023", "dancetrack0080", "dancetrack0049", "dancetrack0083"]
NEW = ["V", "Local_no_extra_risk", "Local_risk", "Context_no_extra_risk", "Context_risk"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-root", type=Path, required=True)
    p.add_argument("--s05a-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    if args.output_root.exists(): raise FileExistsError(args.output_root)
    records=[]
    for condition in ["M", "G", *NEW]:
        target_root=args.output_root/condition; target_root.mkdir(parents=True)
        for sequence in EVAL8:
            if condition == "G":
                source=args.s05a_root/sequence/"trackers/G"/f"{sequence}.txt"
                expected=sha(source); changed=None
            elif condition == "M":
                lock=json.loads((args.s05a_root/sequence/"runtime_lock.json").read_text())
                source=Path(lock["reused_M_tracker"]); expected=lock["reused_M_tracker_sha256"]
                changed=None
            else:
                root=args.tracking_root/condition/sequence
                check=json.loads((root/"engineering_checks.json").read_text())
                if check["status"]!="ENGINEERING_PASS": raise RuntimeError("incomplete tracker")
                stream="V" if condition=="V" else "M"
                source=root/"trackers"/stream/f"{sequence}.txt"
                expected=check["result"]["track_sha256"]
                changed=check.get("selector_changed_components")
            if sha(source)!=expected: raise RuntimeError("source tracker hash mismatch")
            target=target_root/f"{sequence}.txt"; shutil.copy2(source,target)
            records.append({"condition":condition,"sequence":sequence,"sha256":sha(target),
                            "source":str(source),"changed_components":changed})
    (args.output_root/"manifest.json").write_text(json.dumps(
        {"status":"COMPLETED","records":records},indent=2)+"\n")
    print(json.dumps({"status":"COMPLETED","trackers":len(records)},sort_keys=True))


if __name__ == "__main__": main()
