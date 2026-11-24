#!/usr/bin/env python3
"""Export frozen S06 component/action samples from verified causal traces."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN18 = {
    "dancetrack0087", "dancetrack0098", "dancetrack0072", "dancetrack0001",
    "dancetrack0075", "dancetrack0057", "dancetrack0061", "dancetrack0051",
    "dancetrack0002", "dancetrack0006", "dancetrack0068", "dancetrack0052",
    "dancetrack0062", "dancetrack0029", "dancetrack0045", "dancetrack0016",
    "dancetrack0027", "dancetrack0096",
}
DEV6 = {"dancetrack0008", "dancetrack0039", "dancetrack0055",
        "dancetrack0066", "dancetrack0082", "dancetrack0099"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--component-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.protocol) != args.protocol_sha256:
        raise RuntimeError("retry2 protocol hash changed")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    labels = {row["component_id"]: row for row in
              (json.loads(line) for line in args.component_results.read_text().splitlines())}
    if len(labels) != 242:
        raise RuntimeError("expected 242 frozen S05 component labels")
    args.output_root.mkdir(parents=True)
    output = args.output_root / "samples.jsonl"
    counts: Counter[str] = Counter()
    per_video: dict[str, Counter[str]] = {}
    trace_hashes = {}
    with output.open("w") as handle:
        sequences = sorted(TRAIN18 | DEV6)
        for sequence in sequences:
            split = "train18" if sequence in TRAIN18 else "internal_dev6"
            trace = args.trace_root / sequence / "traces/M" / f"{sequence}.jsonl"
            trace_hashes[sequence] = sha256(trace)
            video_counts: Counter[str] = Counter()
            for line in trace.read_text().splitlines():
                frame_row = json.loads(line)
                features = frame_row["s06_features"]
                for component_index, component in enumerate(frame_row["components"]):
                    component_id = f"{sequence}:{frame_row['frame']}:{component_index}"
                    label = labels[component_id]
                    if label["sequence"] != sequence or label["frame"] != frame_row["frame"]:
                        raise RuntimeError("component label alignment changed")
                    actions = component["actions"]
                    union = sorted({tuple(edge) for edges in actions.values() for edge in edges})
                    edge_rows = []
                    for d, t in union:
                        edge_rows.append({
                            "detection_row": d, "tracker_row": t,
                            "iou": features["iou"][d][t],
                            "direction_score": features["direction_score"][d][t],
                            "detection_score": features["detection_scores"][d],
                            "ema_detection_cosine": features["ema_detection_cosine"][d][t],
                            "cosine_valid": features["cosine_valid"][d][t],
                            "observation_gap": frame_row["observation_gaps"][t],
                            "valid_history_length": frame_row["history_lengths"][t],
                            "row_best_margin": features["row_best_margin"][d][t],
                            "column_best_margin": features["column_best_margin"][d][t],
                        })
                    action_rows = []
                    for action in component["unique_actions"]:
                        action_edges = {tuple(edge) for edge in actions[action]}
                        m_edges = {tuple(edge) for edge in actions["M"]}
                        action_label = label.get("actions", {}).get(action)
                        if label["fully_labelable"] and action_label is None:
                            raise RuntimeError("missing label for unique action")
                        k = len(component["tracker_rows"])
                        action_rows.append({
                            "name": action,
                            "selected_edge_indices": [i for i, edge in enumerate(union)
                                                      if edge in action_edges],
                            "differs_from_M": [int((edge in action_edges) != (edge in m_edges))
                                               for edge in union],
                            "g": None if action_label is None else action_label["g"],
                            "h": None if action_label is None else action_label["h"],
                            "g_div_k": None if action_label is None else action_label["g"] / k,
                            "h_div_k": None if action_label is None else action_label["h"] / k,
                        })
                    record = {
                        "component_id": component_id, "sequence": sequence,
                        "frame": frame_row["frame"], "split": split,
                        "fully_labelable": label["fully_labelable"],
                        "detection_count": len(component["detection_rows"]),
                        "track_count": len(component["tracker_rows"]),
                        "edges": edge_rows, "actions": action_rows,
                    }
                    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                    counts[f"{split}_components"] += 1
                    video_counts["components"] += 1
                    if label["fully_labelable"]:
                        counts[f"{split}_labelable_components"] += 1
                        video_counts["labelable_components"] += 1
                        counts[f"{split}_supervised_actions"] += len(action_rows)
                        video_counts["supervised_actions"] += len(action_rows)
                        if label.get("profitable_actions"):
                            counts[f"{split}_profitable_components"] += 1
                            video_counts["profitable_components"] += 1
            per_video[sequence] = video_counts
    summary = {
        "status": "COMPLETED",
        "protocol": str(args.protocol),
        "protocol_sha256": args.protocol_sha256,
        "component_label_source": str(args.component_results),
        "component_label_source_sha256": sha256(args.component_results),
        "samples": str(output), "samples_sha256": sha256(output),
        "counts": dict(sorted(counts.items())),
        "per_video": {key: dict(sorted(value.items())) for key, value in per_video.items()},
        "trace_sha256": trace_hashes,
        "sampling_cap_per_video": 2000,
        "cap_triggered_videos": [],
        "unlabelable_retained_with_null_targets": True,
        "GT_used_as_input_feature": False,
        "CUDA_VISIBLE_DEVICES": "",
        "physical_gpus": [],
        "GPU0_used": False,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
