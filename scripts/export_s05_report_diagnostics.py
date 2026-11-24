#!/usr/bin/env python3
"""Export compact S05-B diagnostics and deterministic cases for Git handoff."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest_key(seed: str, component_id: str) -> str:
    return hashlib.sha256(f"{seed}:{component_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.analysis_root / "summary.json").read_text())
    rows = [json.loads(line) for line in (args.analysis_root / "components.jsonl").read_text().splitlines()]
    if len(rows) != len({row["component_id"] for row in rows}):
        raise ValueError("duplicate component ID")
    profitable = [row for row in rows if row.get("profitable_actions")]
    if len(profitable) != summary["coverage_gate"]["observed_component_frames"]:
        raise ValueError("component rows disagree with coverage summary")
    if len({row["sequence"] for row in profitable}) != summary["coverage_gate"]["observed_unique_videos"]:
        raise ValueError("video coverage disagrees with summary")
    statistics = []
    reason_counts = dict(summary["multi_label_reason_counts"])
    reason_counts["correct_alternative_with_M_wrong"] = sum(
        row["fully_labelable"] and any(
            name != "M" and value["g"] > 0
            for name, value in row["actions"].items())
        for row in rows)
    for name, value in sorted(summary["counts"].items()):
        statistics.append({"name": name,
            "definition": f"frozen S05-B summary count: {name}", "value": value,
            "unit": "count", "numerator": value, "denominator": summary["counts"]["frames"],
            "scope": "S05 fit24 same-state M/G/V candidate coverage",
            "gt_role": "offline IoU-0.5 detection and stable-owner proxy labeling",
            "source": {"artifact_id": "s05b_summary", "locator": f"counts.{name}"}})
    for name, value in sorted(reason_counts.items()):
        statistics.append({"name": f"reason_{name}",
            "definition": f"multi-label component reason: {name}", "value": value,
            "unit": "component_frame", "numerator": value,
            "denominator": summary["counts"]["component_frames"],
            "scope": "S05 fit24 candidate difference components",
            "gt_role": "offline diagnostic proxy labeling",
            "source": {"artifact_id": "s05b_summary",
                       "locator": ("derived_from_component_results:any_unique_alternative_g>0"
                                   if name == "correct_alternative_with_M_wrong" else
                                   f"multi_label_reason_counts.{name}")}})
    priority = ["unlabelable", "positive_detection_absent",
                "only_low_score_or_later_round", "IoU_gate_excluded",
                "no_correct_action_in_M_G_V", "recoverable_M_wrong",
                "M_correct_and_alternative_harms"]
    exclusive = Counter()
    intersections = Counter()
    for row in rows:
        reasons = set(row["reasons"])
        if row["fully_labelable"] and any(
                name != "M" and value["g"] > 0
                for name, value in row["actions"].items()):
            reasons.add("correct_alternative_with_M_wrong")
        exclusive[next((name for name in priority if name in reasons), "other")] += 1
        ordered = sorted(reasons)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1:]:
                intersections[f"{first}&{second}"] += 1
    diagnostics = {
        "schema_version": "mot-report-v1", "status": "COMPLETED",
        "comparison_type": "same_state_M_G_V_first_round_candidate_coverage",
        "unit_of_analysis": "component_frame",
        "definitions": {"profitable": "fully labelable, action differs from M, and g-h > 0",
                        "g": "new correct pairs relative to M",
                        "h": "M-correct pairs destroyed by the alternative",
                        "correct_alternative_with_M_wrong": "at least one non-M action has g>0; may also have h>0",
                        "recoverable_M_wrong": "at least one unique non-M action has g-h>0; this is the 41-component resource gate",
                        "exclusive_funnel_priority": priority},
        "sampling_rule": "all fit24 frames and every non-identical union component",
        "selection_seed": None,
        "statistics": statistics,
        "per_sequence_statistics": summary["per_video"],
        "overlap_accounting": {"component_frames": len(rows),
                               "unique_component_ids": len({row["component_id"] for row in rows}),
                               "profitable_unique_frames": summary["coverage_gate"]["observed_unique_frames"]},
        "unlabeled_or_excluded": [{"reason": "unstable_or_missing_owner",
                                   "count": summary["counts"]["unlabelable_component_frames"]}],
        "case_sampling": {"rule": "sha256(s05-cases-v1:component_id) within improved/worsened/unchanged",
                          "maximum_per_stratum": 3, "actual_counts": {},
                          "source_artifact_id": "s05b_component_results",
                          "worst_case_additions_separately_labeled": False},
        "limitations": summary["limitations"],
        "coverage_gate": summary["coverage_gate"],
        "decision": summary["decision"],
        "multi_label_reason_counts": reason_counts,
        "reason_intersections": dict(sorted(intersections.items())),
        "exclusive_funnel_counts": dict(exclusive),
    }
    strata = {"improved": [], "worsened": [], "unchanged": []}
    for row in rows:
        if row.get("profitable_actions"):
            strata["improved"].append(row)
        elif row["fully_labelable"] and any(
                value["h"] > value["g"] for name, value in row["actions"].items() if name != "M"):
            strata["worsened"].append(row)
        elif row["fully_labelable"] and all(
                value["g"] == value["h"] == 0 for name, value in row["actions"].items() if name != "M"):
            strata["unchanged"].append(row)
    cases = []
    for stratum, values in strata.items():
        selected = sorted(values, key=lambda row: digest_key("s05-cases-v1", row["component_id"]))[:3]
        diagnostics["case_sampling"]["actual_counts"][stratum] = len(selected)
        for row in selected:
            cases.append({"case_id": f"s05:{row['component_id']}", "stratum": stratum,
                          "selection_key": digest_key("s05-cases-v1", row["component_id"]),
                          "comparison_type": "same_state", "gt_role": "offline diagnostic proxy",
                          **row})
    (args.report_root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    with (args.report_root / "component_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (args.report_root / "cases.jsonl").open("w") as handle:
        for row in cases:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "COMPLETED", "components": len(rows),
                      "profitable": len(profitable), "cases": len(cases),
                      "strata": {name: len(value) for name, value in strata.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
