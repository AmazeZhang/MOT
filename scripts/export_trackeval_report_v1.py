#!/usr/bin/env python3
"""Export a saved S01-style TrackEval summary into report-v1 CSV files."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from pathlib import Path


FIELDS = [
    "run_id", "condition", "seed", "scope_id", "sequence", "status",
    "num_frames", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW",
    "FP", "FN", "Frag", "source_artifact_id", "source_locator",
    "missing_reason",
]


def detailed_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {row["seq"]: row for row in csv.DictReader(handle)}
    if "COMBINED" not in rows:
        raise ValueError(f"{path}: missing COMBINED row")
    return rows


def frame_count(dataset_root: Path, sequence: str) -> int:
    config = configparser.ConfigParser()
    path = dataset_root / sequence / "seqinfo.ini"
    if not config.read(path):
        raise FileNotFoundError(path)
    return config["Sequence"].getint("seqLength")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--source-artifact-id", required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    sequences = summary["sequences"]
    results = summary["results"]
    if len(sequences) != len(set(sequences)):
        raise ValueError("duplicate sequence in summary scope")
    frames = {sequence: frame_count(args.dataset_root, sequence) for sequence in sequences}
    aggregate: list[dict[str, object]] = []
    per_sequence: list[dict[str, object]] = []
    for condition, condition_result in results.items():
        official_path = args.official_root / condition / "pedestrian_detailed.csv"
        official = detailed_rows(official_path)
        expected = set(sequences) | {"COMBINED"}
        if set(official) != expected:
            raise ValueError(f"{condition}: detailed rows differ from expected scope")
        run_id = f"{args.run_id_prefix}_{condition}"
        for sequence in ["COMBINED_SEQ", *sequences]:
            source_key = "COMBINED_SEQ" if sequence == "COMBINED_SEQ" else sequence
            summary_metrics = (condition_result["COMBINED_SEQ"] if sequence == "COMBINED_SEQ"
                               else condition_result["per_sequence"][sequence])
            official_key = "COMBINED" if sequence == "COMBINED_SEQ" else sequence
            detail = official[official_key]
            for metric in ("HOTA", "DetA", "AssA", "IDF1"):
                detail_key = f"{metric}___AUC" if metric != "IDF1" else "IDF1"
                expected_value = float(detail[detail_key]) * 100.0
                if abs(float(summary_metrics[metric]) - expected_value) > 1e-10:
                    raise ValueError(f"{condition}/{sequence}/{metric}: summary/detail mismatch")
            for metric in ("IDSW", "FP", "FN", "Frag"):
                detail_key = {"FP": "CLR_FP", "FN": "CLR_FN"}.get(metric, metric)
                if int(summary_metrics[metric]) != int(float(detail[detail_key])):
                    raise ValueError(f"{condition}/{sequence}/{metric}: summary/detail mismatch")
            row = {
                "run_id": run_id,
                "condition": condition,
                "seed": "NA",
                "scope_id": args.scope_id,
                "sequence": sequence,
                "status": "COMPLETED",
                "num_frames": sum(frames.values()) if sequence == "COMBINED_SEQ" else frames[sequence],
                "HOTA": repr(float(summary_metrics["HOTA"])),
                "DetA": repr(float(summary_metrics["DetA"])),
                "AssA": repr(float(summary_metrics["AssA"])),
                "IDF1": repr(float(summary_metrics["IDF1"])),
                "MOTA": repr(float(detail["MOTA"]) * 100.0),
                "IDSW": int(summary_metrics["IDSW"]),
                "FP": int(summary_metrics["FP"]),
                "FN": int(summary_metrics["FN"]),
                "Frag": int(summary_metrics["Frag"]),
                "source_artifact_id": args.source_artifact_id,
                "source_locator": f"results.{condition}.{source_key}; official/{condition}/pedestrian_detailed.csv:{official_key}",
                "missing_reason": "",
            }
            (aggregate if sequence == "COMBINED_SEQ" else per_sequence).append(row)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("aggregate_metrics.csv", aggregate),
                       ("per_sequence_metrics.csv", per_sequence)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": "COMPLETED", "conditions": list(results),
                      "aggregate_rows": len(aggregate),
                      "per_sequence_rows": len(per_sequence)}, sort_keys=True))


if __name__ == "__main__":
    main()
