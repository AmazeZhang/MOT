#!/usr/bin/env python3
"""Freeze internal-dev6 risk-policy closed-loop locks for all checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEV6 = ["dancetrack0008", "dancetrack0039", "dancetrack0055",
        "dancetrack0066", "dancetrack0082", "dancetrack0099"]
CODE = ["scripts/run_s06_selector_closed_loop.py", "scripts/run_s01_visual_baselines.py",
        "scripts/run_e1_mature_reemit_factorial.py", "scripts/analyze_p1_common_state_mechanism.py",
        "src/mot_research/models/component_action_selector.py",
        "src/mot_research/analysis/component_action_features.py",
        "src/mot_research/analysis/multi_assignment_components.py",
        "src/mot_research/analysis/runtime_record.py",
        "src/mot_research/association/appearance_core.py",
        "src/mot_research/tracking/first_round_adapter.py",
        "src/mot_research/tracking/natural_motion_policy.py",
        "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
        "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--s05-lock-root", type=Path, required=True)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    if sha(args.protocol) != args.protocol_sha256 or args.output_root.exists():
        raise RuntimeError("protocol changed or output exists")
    args.output_root.mkdir(parents=True)
    snapshot = args.output_root / "source_snapshot"
    for path in [*CODE, args.protocol.relative_to(ROOT).as_posix()]:
        target = snapshot / path; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    code_hashes = {path: sha(ROOT / path) for path in CODE}
    count = 0
    for mode in ("Local", "Context"):
        for epoch in range(5, 31, 5):
            checkpoint = args.training_root / mode / f"epoch_{epoch:02d}.pt"
            for sequence in DEV6:
                source_path = args.s05_lock_root / sequence / "runtime_lock.json"
                source = json.loads(source_path.read_text())
                output = args.output_root / f"{mode}_e{epoch:02d}" / sequence
                output.mkdir(parents=True)
                lock = {
                    **{key: source[key] for key in ("train_root", "detection_root",
                        "input_detection_sha256", "feature_cache_path", "feature_cache_sha256")},
                    "run_id": f"s06_dev_{mode}_risk_e{epoch:02d}_{sequence}_retry3_attempt1",
                    "status": "locked_before_closed_loop_replay", "selected_sequences": [sequence],
                    "model": mode, "epoch": epoch, "policy": "risk", "risk_multiplier": 2.0,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                    "protocol_path": str(args.protocol), "protocol_sha256": args.protocol_sha256,
                    "code_hashes": code_hashes, "output_root": str(output),
                    "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
                }
                (output / "runtime_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
                count += 1
    print(json.dumps({"status": "LOCKED", "jobs": count}, sort_keys=True))


if __name__ == "__main__":
    main()
