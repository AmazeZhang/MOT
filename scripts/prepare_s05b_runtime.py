#!/usr/bin/env python3
"""Create immutable S05-B locks from authoritative S01 M artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = [
    "scripts/run_s05_candidate_coverage.py",
    "scripts/run_s01_visual_baselines.py",
    "scripts/run_e1_mature_reemit_factorial.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/multi_assignment_components.py",
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
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--s01-frozen-inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    frozen = json.loads(args.s01_frozen_inputs.read_text())
    protocol = args.protocol.resolve()
    protocol_rel = protocol.relative_to(ROOT).as_posix()
    args.output_root.mkdir(parents=True)
    patch_path = args.output_root / "source_dirty.patch"
    patch_path.write_text(git("diff", "--binary", "HEAD"))
    snapshot = args.output_root / "source_snapshot"
    for path in [*CODE_PATHS, protocol_rel]:
        destination = snapshot / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)
    code_hashes = {path: sha256(ROOT / path) for path in CODE_PATHS}
    artifacts = {(row["sequence"], row["condition"], row["kind"]): row
                 for row in frozen["artifacts"]}
    features = {row["sequence"]: row for row in frozen["feature_artifacts"]}
    checks = {row["sequence"]: row for row in frozen["engineering_checks"]}
    for sequence in frozen["sequences"]:
        source_check = checks[sequence]
        source_lock_path = Path(source_check["path"]).parent / "runtime_lock.json"
        source_lock = json.loads(source_lock_path.read_text())
        m_artifact = artifacts[(sequence, "M", "trackers")]
        if sha256(Path(m_artifact["path"])) != m_artifact["sha256"]:
            raise RuntimeError(f"{sequence}: authoritative M tracker changed")
        feature = features[sequence]
        if sha256(Path(feature["path"])) != feature["sha256"]:
            raise RuntimeError(f"{sequence}: feature cache changed")
        output = args.output_root / sequence
        output.mkdir()
        lock = {
            "run_id": f"s05b_candidate_coverage_{sequence}_attempt1",
            "status": "locked_before_candidate_replay",
            "project_commit": git("rev-parse", "HEAD").strip(),
            "project_status": git("status", "--short"),
            "dirty_patch_path": str(patch_path),
            "dirty_patch_sha256": sha256(patch_path),
            "source_snapshot_root": str(snapshot),
            "protocol_path": protocol_rel,
            "protocol_sha256": sha256(protocol),
            "code_hashes": code_hashes,
            "selected_sequences": [sequence],
            "train_root": source_lock["train_root"],
            "detection_root": source_lock["detection_root"],
            "input_detection_sha256": {sequence: source_lock["input_detection_sha256"][sequence]},
            "feature_cache_path": feature["path"],
            "feature_cache_sha256": feature["sha256"],
            "authoritative_M_tracker": m_artifact["path"],
            "authoritative_M_tracker_sha256": m_artifact["sha256"],
            "authoritative_S01_runtime_lock": str(source_lock_path),
            "authoritative_S01_runtime_lock_sha256": sha256(source_lock_path),
            "output_root": str(output),
            "cuda_visible_devices": "",
            "physical_gpus": [],
            "logical_to_physical": {},
            "GPU0_used": False,
        }
        (output / "runtime_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "sequences": len(frozen["sequences"]),
                      "protocol_sha256": sha256(protocol)}, sort_keys=True))


if __name__ == "__main__":
    main()
