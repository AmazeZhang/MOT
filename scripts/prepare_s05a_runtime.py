#!/usr/bin/env python3
"""Create immutable S05-A runtime locks from authoritative S04 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = [
    "scripts/run_s05_visual_stability.py",
    "scripts/run_s01_visual_baselines.py",
    "scripts/run_e1_mature_reemit_factorial.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/association/appearance_core.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          text=True, stdout=subprocess.PIPE).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--s04-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    protocol_rel = protocol.relative_to(ROOT).as_posix()
    protocol_sha = sha256(protocol)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    patch_path = args.output_root / "source_dirty.patch"
    patch_path.write_text(git("diff", "--binary", "HEAD"))
    status = git("status", "--short")
    snapshot_root = args.output_root / "source_snapshot"
    for path in [*CODE_PATHS, protocol_rel]:
        target = snapshot_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    source_index = {
        path: sha256(ROOT / path) for path in CODE_PATHS
    }
    metadata = {
        "project_commit": git("rev-parse", "HEAD").strip(),
        "project_status": status,
        "dirty_patch_path": str(patch_path),
        "dirty_patch_sha256": sha256(patch_path),
        "source_snapshot_root": str(snapshot_root),
        "protocol_path": protocol_rel,
        "protocol_sha256": protocol_sha,
        "code_hashes": source_index,
    }
    (args.output_root / "source_index.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for baseline_dir in sorted((args.s04_root / "baseline").glob("dancetrack*")):
        source_lock_path = baseline_dir / "runtime_lock.json"
        source_check_path = baseline_dir / "engineering_checks.json"
        source_lock = json.loads(source_lock_path.read_text())
        source_check = json.loads(source_check_path.read_text())
        sequence = source_lock["selected_sequences"][0]
        reused_m = baseline_dir / "trackers" / "M" / f"{sequence}.txt"
        reported_hash = source_check["results"]["M"]["track_sha256"]
        if sha256(reused_m) != reported_hash:
            raise RuntimeError(f"{sequence}: S04 M output disagrees with engineering record")
        output = args.output_root / "tracking" / sequence
        output.mkdir(parents=True)
        lock = {
            "run_id": f"s05a_stability_{sequence}_attempt1",
            "status": "locked_before_tracking",
            **metadata,
            "selected_sequences": [sequence],
            "train_root": source_lock["train_root"],
            "detection_root": source_lock["detection_root"],
            "input_detection_sha256": source_lock["input_detection_sha256"],
            "feature_cache_path": source_lock["feature_cache_path"],
            "feature_cache_sha256": source_lock["feature_cache_sha256"],
            "output_root": str(output),
            "reused_M_tracker": str(reused_m),
            "reused_M_tracker_sha256": reported_hash,
            "reused_M_source_run_id": source_lock["run_id"],
            "reused_M_runtime_lock_path": str(source_lock_path),
            "reused_M_runtime_lock_sha256": sha256(source_lock_path),
            "cuda_visible_devices": "",
            "physical_gpus": [],
            "logical_to_physical": {},
            "GPU0_used": False,
        }
        lock_path = output / "runtime_lock.json"
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "protocol_sha256": protocol_sha,
                      "sequences": len(list((args.output_root / "tracking").iterdir())),
                      "output_root": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
