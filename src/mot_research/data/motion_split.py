"""Deterministic validation helpers for frozen video-level motion splits."""

from __future__ import annotations

import configparser
import hashlib
from pathlib import Path
from typing import Any


def sequence_names_sha256(sequence_names: list[str]) -> str:
    """Hash sorted sequence names with a terminal newline."""
    payload = "".join(f"{name}\n" for name in sorted(sequence_names))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def ranked_extra_dev(
    sequence_names: list[str], anchors: list[str], seed: int, count: int
) -> list[str]:
    """Rank non-anchor sequences using the frozen language-independent rule."""
    anchor_set = set(anchors)
    ranked = sorted(
        (
            hashlib.sha256(f"{seed}:{name}".encode("ascii")).hexdigest(),
            name,
        )
        for name in sequence_names
        if name not in anchor_set
    )
    return [name for _, name in ranked[:count]]


def _sequence_frame_count(sequence_dir: Path) -> int:
    config = configparser.ConfigParser()
    seqinfo = sequence_dir / "seqinfo.ini"
    if not config.read(seqinfo):
        raise FileNotFoundError(seqinfo)
    return config["Sequence"].getint("seqLength")


def _sentinel_image_hashes(sequence_dir: Path) -> dict[str, str]:
    config = configparser.ConfigParser()
    seqinfo = sequence_dir / "seqinfo.ini"
    if not config.read(seqinfo):
        raise FileNotFoundError(seqinfo)
    info = config["Sequence"]
    length = info.getint("seqLength")
    image_dir = sequence_dir / info.get("imDir", "img1")
    extension = info.get("imExt", ".jpg")
    frames = sorted({1, (length + 1) // 2, length})
    hashes: dict[str, str] = {}
    for frame in frames:
        image_path = image_dir / f"{frame:08d}{extension}"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        hashes[str(frame)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    return hashes


def validate_motion_split(manifest: dict[str, Any], train_root: Path) -> dict[str, Any]:
    """Validate membership, frozen selection rule, files, hashes and frame totals."""
    dataset_sequences = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    expected = int(manifest["dataset"]["expected_sequences"])
    if len(dataset_sequences) != expected:
        raise ValueError(f"expected {expected} dataset sequences, found {len(dataset_sequences)}")
    inventory_hash = sequence_names_sha256(dataset_sequences)
    if inventory_hash != manifest["dataset"]["inventory_sequence_names_sha256"]:
        raise ValueError("dataset inventory sequence-name hash mismatch")
    sentinel_owners: dict[str, list[str]] = {}
    for sequence in dataset_sequences:
        for frame, digest in _sentinel_image_hashes(train_root / sequence).items():
            sentinel_owners.setdefault(digest, []).append(f"{sequence}:{frame}")
    duplicate_sentinels = sorted(
        owners for owners in sentinel_owners.values() if len(owners) > 1
    )
    expected_duplicate_groups = int(
        manifest["selection"]["expected_exact_sentinel_duplicate_groups"]
    )
    if len(duplicate_sentinels) != expected_duplicate_groups:
        raise ValueError(
            "exact sentinel-image duplicate group count mismatch: "
            f"{len(duplicate_sentinels)} != {expected_duplicate_groups}"
        )

    motion_train = list(manifest["motion_train"]["sequences"])
    motion_dev = list(manifest["motion_dev"]["sequences"])
    if motion_train != sorted(motion_train) or motion_dev != sorted(motion_dev):
        raise ValueError("split sequence lists must be sorted")
    if len(motion_train) != len(set(motion_train)) or len(motion_dev) != len(set(motion_dev)):
        raise ValueError("duplicate sequence within a split")
    overlap = sorted(set(motion_train) & set(motion_dev))
    if overlap:
        raise ValueError(f"motion train/dev overlap: {overlap}")
    if sorted(motion_train + motion_dev) != dataset_sequences:
        raise ValueError("motion split is not exhaustive over the dataset inventory")

    selection = manifest["selection"]
    anchors = list(selection["diagnostic_anchor_dev"])
    if not set(anchors).issubset(motion_dev):
        raise ValueError("a diagnostic anchor is absent from motion_dev")
    extra_count = int(selection["target_counts"]["motion_dev"]) - len(anchors)
    expected_extra = ranked_extra_dev(
        dataset_sequences, anchors, int(selection["split_seed"]), extra_count
    )
    if expected_extra != list(selection["extra_dev_rank_order"]):
        raise ValueError("extra-dev rank order does not match the frozen hash rule")
    if set(motion_dev) != set(anchors) | set(expected_extra):
        raise ValueError("motion_dev membership differs from anchors plus ranked extras")

    group_summaries: dict[str, dict[str, Any]] = {}
    for group_name, sequences in (("motion_train", motion_train), ("motion_dev", motion_dev)):
        declared = manifest[group_name]
        if len(sequences) != int(declared["sequence_count"]):
            raise ValueError(f"{group_name} sequence count mismatch")
        digest = sequence_names_sha256(sequences)
        if digest != declared["sequence_names_sha256"]:
            raise ValueError(f"{group_name} sequence-name hash mismatch")
        frame_count = 0
        for sequence in sequences:
            sequence_dir = train_root / sequence
            frame_count += _sequence_frame_count(sequence_dir)
            if not (sequence_dir / "gt" / "gt.txt").is_file():
                raise FileNotFoundError(sequence_dir / "gt" / "gt.txt")
        if frame_count != int(declared["frame_count"]):
            raise ValueError(f"{group_name} frame count mismatch")
        group_summaries[group_name] = {
            "sequence_count": len(sequences),
            "frame_count": frame_count,
            "sequence_names_sha256": digest,
        }
    return {
        "status": "PASS",
        "split_id": manifest["split_id"],
        "dataset_inventory": {
            "sequence_count": len(dataset_sequences),
            "sequence_names_sha256": inventory_hash,
        },
        **group_summaries,
        "overlap_count": 0,
        "exact_sentinel_duplicate_groups": duplicate_sentinels,
        "official_val_used": False,
    }

