from __future__ import annotations

import configparser
import hashlib
from pathlib import Path

import pytest

from mot_research.data.motion_split import (
    ranked_extra_dev,
    sequence_names_sha256,
    validate_motion_split,
)


def write_sequence(root: Path, name: str, frames: int) -> None:
    sequence = root / name
    (sequence / "gt").mkdir(parents=True)
    config = configparser.ConfigParser()
    config["Sequence"] = {"name": name, "seqLength": str(frames)}
    with (sequence / "seqinfo.ini").open("w") as handle:
        config.write(handle)
    (sequence / "gt" / "gt.txt").write_text("1,1,0,0,1,1,1,1,1\n")
    (sequence / "img1").mkdir()
    for frame in range(1, frames + 1):
        payload = hashlib.sha256(f"{name}:{frame}".encode()).digest()
        (sequence / "img1" / f"{frame:08d}.jpg").write_bytes(payload)


def build_manifest(names: list[str], train: list[str], dev: list[str]) -> dict:
    anchors = [dev[0]]
    extra = ranked_extra_dev(names, anchors, 7, len(dev) - 1)
    assert set(dev) == set(anchors + extra)
    return {
        "split_id": "test",
        "dataset": {
            "expected_sequences": len(names),
            "inventory_sequence_names_sha256": sequence_names_sha256(names),
        },
        "selection": {
            "split_seed": 7,
            "target_counts": {"motion_train": len(train), "motion_dev": len(dev)},
            "diagnostic_anchor_dev": anchors,
            "extra_dev_rank_order": extra,
            "expected_exact_sentinel_duplicate_groups": 0,
        },
        "motion_train": {
            "sequences": train,
            "sequence_count": len(train),
            "frame_count": len(train) * 3,
            "sequence_names_sha256": sequence_names_sha256(train),
        },
        "motion_dev": {
            "sequences": dev,
            "sequence_count": len(dev),
            "frame_count": len(dev) * 3,
            "sequence_names_sha256": sequence_names_sha256(dev),
        },
    }


def test_frozen_hash_rank_and_split_validation(tmp_path: Path) -> None:
    names = ["seq1", "seq2", "seq3", "seq4"]
    for name in names:
        write_sequence(tmp_path, name, 3)
    anchor = "seq1"
    extra = ranked_extra_dev(names, [anchor], 7, 1)
    dev = sorted([anchor, *extra])
    train = sorted(set(names) - set(dev))
    summary = validate_motion_split(build_manifest(names, train, dev), tmp_path)
    assert summary["status"] == "PASS"
    assert summary["overlap_count"] == 0
    assert summary["motion_train"]["sequence_count"] == 2
    assert summary["motion_dev"]["frame_count"] == 6


def test_overlap_is_rejected(tmp_path: Path) -> None:
    names = ["seq1", "seq2", "seq3", "seq4"]
    for name in names:
        write_sequence(tmp_path, name, 3)
    anchor = "seq1"
    extra = ranked_extra_dev(names, [anchor], 7, 1)
    dev = sorted([anchor, *extra])
    train = sorted(set(names) - set(dev))
    manifest = build_manifest(names, train, dev)
    manifest["motion_train"]["sequences"] = sorted([*train, dev[0]])
    with pytest.raises(ValueError, match="overlap"):
        validate_motion_split(manifest, tmp_path)

