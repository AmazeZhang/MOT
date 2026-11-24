#!/usr/bin/env python3
"""Read-only arithmetic review of S05/S06 Git exports; never runs a tracker."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
S05 = ROOT / "reports/diagnostics/s05_visual_decision_coverage_20261123"
S06 = ROOT / "reports/feasibility/s06_component_selector_seed13_retry3_20261124"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path; do not overwrite a prior review.")
    inputs = {}

    def read(path):
        data = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def load(path):
        return json.loads(read(path))

    def csv_rows(path):
        return list(csv.DictReader(read(path).splitlines()))

    aggregate_rows = csv_rows(S06 / "aggregate_metrics.csv")
    sequence_rows = csv_rows(S06 / "per_sequence_metrics.csv")
    aggregates = {r["condition"]: r for r in aggregate_rows}
    sequences = {}
    for row in sequence_rows:
        group = sequences.setdefault(row["condition"], {})
        assert row["sequence"] not in group
        group[row["sequence"]] = row
    assert len(aggregate_rows) == len(aggregates) == 7
    assert len(sequence_rows) == 56
    assert all(set(g) == set(sequences["M"]) for g in sequences.values())
    assert len(sequences["M"]) == 8
    assert all(r["status"] == "COMPLETED" for r in aggregate_rows + sequence_rows)
    comparisons = {}
    for condition in aggregates:
        for baseline in ("M", "V"):
            if condition == baseline:
                continue
            delta = {s: float(sequences[condition][s]["HOTA"]) -
                     float(sequences[baseline][s]["HOTA"]) for s in sequences[baseline]}
            values = list(delta.values())
            comparisons[f"{condition}_vs_{baseline}"] = {
                "official_aggregate_delta": {
                    m: float(aggregates[condition][m]) - float(aggregates[baseline][m])
                    for m in ("HOTA", "AssA", "DetA", "IDF1", "IDSW", "FP", "FN")},
                "per_sequence_HOTA_delta": delta,
                "wins_ties_losses": [sum(x > 0 for x in values), sum(x == 0 for x in values),
                                     sum(x < 0 for x in values)],
                "macro_mean_HOTA_delta_not_official_aggregate": statistics.mean(values),
                "median_HOTA_delta": statistics.median(values),
            }
    gate = load(S06 / "seed13_gate.json")
    cr = comparisons["Context_risk_vs_M"]
    cv = comparisons["Context_risk_vs_V"]["official_aggregate_delta"]
    recomputed = [cr["official_aggregate_delta"]["HOTA"] >= 0.5,
                  cr["official_aggregate_delta"]["AssA"] >= 0.5,
                  cr["official_aggregate_delta"]["IDSW"] <= 0,
                  cr["wins_ties_losses"][0] >= 5, cv["HOTA"] > 0, cv["AssA"] > 0,
                  float(aggregates["Context_risk"]["HOTA"]) > float(aggregates["Local_risk"]["HOTA"]),
                  float(aggregates["Context_risk"]["AssA"]) > float(aggregates["Local_risk"]["AssA"])]
    assert recomputed == list(gate["gate_clauses"].values())
    compared_cells = 0
    for row in aggregate_rows + sequence_rows:
        result = gate["results"][row["condition"]]
        source = result["COMBINED_SEQ"] if row["sequence"] == "COMBINED_SEQ" else result["per_sequence"][row["sequence"]]
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag"):
            assert float(row[metric]) == source[metric]
            compared_cells += 1
    hash_checks = []
    unresolved = {}
    for directory in (S05, S06):
        manifest = load(directory / "artifacts.json")
        identifiers = {r["artifact_id"] for r in manifest["files"]}
        for artifact in manifest["files"]:
            if artifact["location"] != "git":
                continue
            path = directory / artifact["path"]
            read(path)
            digest = inputs[str(path.relative_to(ROOT))]
            assert digest == artifact["sha256"], str(path)
            if "bytes" in artifact:
                assert path.stat().st_size == artifact["bytes"], str(path)
            hash_checks.append(str(path.relative_to(ROOT)))
        rows = csv_rows(directory / "aggregate_metrics.csv")
        unresolved[directory.name] = sorted({r["source_artifact_id"] for r in rows} - identifiers)
    cases = [json.loads(line) for line in read(S06 / "cases.jsonl").splitlines()]
    case_summary = {}
    for condition in sorted({r["condition"] for r in cases}):
        rows = [r for r in cases if r["condition"] == condition]
        changed = [r for r in rows if r["chosen"] != "M"]
        identity_predictions = [next(v for v in r["predictions"] if v["name"] == "M") for r in rows]
        case_summary[condition] = {
            "changed": len(changed), "unchanged_sampled_controls": len(rows) - len(changed),
            "chosen_action_counts": dict(Counter(r["chosen"] for r in changed)),
            "known_zero_M_predictions_on_exported_cases_only": {
                metric: {"min": min(r[metric] for r in identity_predictions),
                         "mean": statistics.mean(r[metric] for r in identity_predictions),
                         "max": max(r[metric] for r in identity_predictions)}
                for metric in ("g_hat", "h_hat")},
        }
    selections = load(S06 / "checkpoint_selection.json")
    curves = csv_rows(S06 / "learning_curves.csv")
    losses = {mode: {r["epoch"]: float(r["value"]) for r in curves
                    if r["run_id"] == f"s06_seed13_{mode}"}
              for mode in ("Local", "Context")}
    protocol = read(ROOT / "manifests/protocols/s06_component_selector_seed13_20261123.yaml")
    train = set(re.search(r"train18:\n    sequences: \[(.*?)\]", protocol).group(1).split(", "))
    labels = [json.loads(line) for line in read(S05 / "component_results.jsonl").splitlines()]
    examples = []
    for component in labels:
        if component["sequence"] not in train or not component["fully_labelable"]:
            continue
        for action in component["unique_actions"]:
            label = component["actions"][action]
            k = len(component["tracker_ids"])
            examples.append((component["sequence"], action, label["g"] / k, label["h"] / k))
    counts = Counter(row[0] for row in examples)
    weights = [1 / (len(counts) * counts[row[0]]) for row in examples]
    means = [sum(w * row[i] for w, row in zip(weights, examples)) for i in (2, 3)]
    training = {
        "unique_labelable_components": sum(r["sequence"] in train and r["fully_labelable"] for r in labels),
        "unique_action_examples": len(examples),
        "updates_per_epoch_from_executed_128_action_batching": (len(examples) + 127) // 128,
        "selected_epochs": {m: selections["selected"][m]["epoch"] for m in losses},
        "training_loss_by_epoch": losses,
        "video_balanced_target_means_g_h": means,
        "constant_zero_expected_MSE": sum(w * (r[2]**2 + r[3]**2) / 2 for w, r in zip(weights, examples)),
        "constant_mean_expected_MSE": sum(w * ((r[2]-means[0])**2 + (r[3]-means[1])**2) / 2
                                          for w, r in zip(weights, examples)),
        "constant_baseline_scope": "analytic training-label expectations, not held-out or same-resample model scores",
    }
    read(Path(__file__).resolve())
    result = {
        "review_date": "2026-11-24", "status": "LOCAL_GIT_EVIDENCE_REVIEW_ONLY",
        "server_raw_results_recomputed": False,
        "verification": {"aggregate_rows": 7, "per_sequence_rows": 56,
                         "CSV_vs_gate_common_metric_cells": compared_cells,
                         "verified_git_artifact_hashes": hash_checks,
                         "gate_passed_clauses": sum(recomputed),
                         "unresolved_metric_artifact_ids": unresolved},
        "comparisons": comparisons, "cases": case_summary, "training": training,
        "input_sha256": inputs,
        "limitations": ["No model/checkpoint/GT/trajectory access or new tracking execution",
                        "No confidence intervals or independent generalization claim",
                        "Case predictions are selected records, not population calibration estimates",
                        "Per-video mean is not official combined HOTA"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "gate_passed_clauses": sum(recomputed),
                      "compared_cells": compared_cells, "verified_hashes": len(hash_checks)}))


if __name__ == "__main__":
    main()
