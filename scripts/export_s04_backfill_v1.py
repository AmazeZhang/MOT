#!/usr/bin/env python3
"""Export the authoritative S04 event scores into a compact report-v1 backfill."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scores_path = args.source_root / "scores.jsonl"
    summary_path = args.source_root / "summary.json"
    per_video_path = args.source_root / "per_video.json"
    scores = [json.loads(line) for line in scores_path.read_text().splitlines() if line]
    summary = json.loads(summary_path.read_text())
    per_video = json.loads(per_video_path.read_text())
    if len(scores) != len({row["case_id"] for row in scores}):
        raise ValueError("duplicate S04 case_id")
    kinds = Counter(row["kind"] for row in scores)
    if kinds != Counter({"wrong": summary["totals"]["wrong"]["cases"],
                         "clean": summary["totals"]["clean"]["cases"]}):
        raise ValueError("S04 score rows disagree with summary case totals")
    for kind in ("wrong", "clean"):
        rows = [row for row in scores if row["kind"] == kind]
        if sum(row["frozen_minus_single"]["all_prior_anchor_correct_frame_delta"] for row in rows) != \
                summary["totals"][kind]["frozen_minus_single_all_anchor_delta"]:
            raise ValueError(f"{kind}: event scores disagree with summary delta")
        if sum(bool(row["future_frames_with_frozen_minus_single_output_change"]) for row in rows) != \
                summary["totals"][kind]["frozen_minus_single_changed_cases"]:
            raise ValueError(f"{kind}: changed-case count disagrees with summary")

    normalized = []
    for row in scores:
        normalized.append({
            "case_id": row["case_id"],
            "kind": row["kind"],
            "sequence": row["sequence"],
            "frame": row["frame"],
            "track_id": row["track_id"],
            "owner_gt_id": row["owner_gt_id"],
            "paired_case_id": row["matched_control_id"] or row["matched_wrong_id"],
            "future_frames_scored": row["future_frames_scored"],
            "prior_output_anchor_count": row["prior_output_anchor_count"],
            "owner_in_prior_output_anchors": row["owner_in_prior_output_anchors"],
            "single_minus_factual": row["single_minus_factual"],
            "frozen_minus_factual": row["frozen_minus_factual"],
            "frozen_minus_single": row["frozen_minus_single"],
            "frozen_suppressed_write_count": row["frozen_suppressed_write_count"],
            "future_frames_with_single_output_change": row["future_frames_with_single_output_change"],
            "future_frames_with_frozen_output_change": row["future_frames_with_frozen_output_change"],
            "future_frames_with_frozen_minus_single_output_change": row["future_frames_with_frozen_minus_single_output_change"],
            "future_target_edge_exposure": row["future_target_edge_exposure"],
            "first_round_legal_screen": row["first_round_legal_screen"],
            "source": {"artifact_id": "s04_scores", "locator": f"case_id={row['case_id']}"},
        })

    statistics = []
    for kind in ("wrong", "clean"):
        total = summary["totals"][kind]
        for name, value in total.items():
            statistics.append({
                "name": f"{kind}_{name}",
                "definition": f"authoritative S04 summary field totals.{kind}.{name}",
                "value": value,
                "unit": "case_count" if name.endswith("cases") else "count_or_frame_delta",
                "numerator": value,
                "denominator": total["cases"],
                "scope": "S04 development8 selected local branches",
                "gt_role": "privileged offline event selection and scoring",
                "source": {"artifact_id": "s04_summary", "locator": f"totals.{kind}.{name}"},
            })
    diagnostics = {
        "schema_version": "mot-report-v1",
        "status": "COMPLETED",
        "comparison_type": "GT_privileged_local_branch_counterfactual",
        "unit_of_analysis": "selected_event_branch",
        "definitions": {"authoritative_decision": summary["decision"]},
        "sampling_rule": "all selected wrong events plus frozen matched clean controls",
        "selection_seed": None,
        "statistics": statistics,
        "per_sequence_statistics": per_video,
        "overlap_accounting": summary["overlap"],
        "unlabeled_or_excluded": [],
        "case_sampling": {
            "rule": "all_30_cases_exported_no_case_sampling",
            "maximum_per_stratum": None,
            "actual_counts": dict(kinds),
            "source_artifact_id": "s04_scores",
            "worst_case_additions_separately_labeled": False,
        },
        "limitations": summary["limitations"],
        "source_hashes": {
            "scores.jsonl": sha256(scores_path),
            "summary.json": sha256(summary_path),
            "per_video.json": sha256(per_video_path),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_event_results.jsonl").open("w") as handle:
        for row in normalized:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "events": len(normalized),
                      "kinds": dict(kinds)}, sort_keys=True))


if __name__ == "__main__":
    main()
