#!/usr/bin/env python3
"""Apply the frozen P1 natural closed-loop decision gate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use METHOD=PATH")
        method, path = value.split("=", 1)
        if not method or method in result:
            raise ValueError(f"invalid or duplicate {label} method: {method!r}")
        result[method] = Path(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-summary", action="append", required=True)
    parser.add_argument("--trackeval-summary", action="append", required=True)
    parser.add_argument("--recovery-summary", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise RuntimeError("protocol SHA256 mismatch")
    protocol = yaml.safe_load(args.protocol.read_text())
    methods = set(protocol["runs"])
    run_paths = parse_mapping(args.run_summary, "run-summary")
    trackeval_paths = parse_mapping(args.trackeval_summary, "trackeval-summary")
    recovery_paths = parse_mapping(args.recovery_summary, "recovery-summary")
    for label, mapping in (
        ("run", run_paths), ("TrackEval", trackeval_paths), ("recovery", recovery_paths)
    ):
        if set(mapping) != methods:
            raise RuntimeError(f"{label} methods differ: {sorted(set(mapping) ^ methods)}")
    output_partial = args.output_dir.with_name(args.output_dir.name + ".partial")
    if args.output_dir.exists() or output_partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    output_partial.mkdir(parents=True)

    runs = {method: json.loads(path.read_text()) for method, path in run_paths.items()}
    trackeval = {
        method: json.loads(path.read_text()) for method, path in trackeval_paths.items()
    }
    recovery = {
        method: json.loads(path.read_text()) for method, path in recovery_paths.items()
    }
    for method in sorted(methods):
        run = runs[method]
        if (
            run["execution_status"] != "PASS"
            or run["protocol_sha256"] != args.protocol_sha256
            or run["protocol_run_id"] != method
            or run["frames_processed"] != protocol["scope"]["frame_count"]
            or run["sequence_count"] != len(protocol["scope"]["sequences"])
            or run["research_metrics_computed"]
            or run["ground_truth_read"]
        ):
            raise RuntimeError(f"{method}: invalid tracker run summary")
        if trackeval[method]["execution_status"] != "PASS":
            raise RuntimeError(f"{method}: TrackEval did not pass")
        if recovery[method]["execution_status"] != "PASS":
            raise RuntimeError(f"{method}: recovery evaluation did not pass")
        if recovery[method]["events_sha256"] != protocol["fixed_inputs"]["common_events"]["sha256"]:
            raise RuntimeError(f"{method}: recovery event hash mismatch")

    official = {
        method: value["official_trackeval_combined"] for method, value in trackeval.items()
    }
    baseline_expected = protocol["fixed_inputs"]["registered_B0_official_combined"]
    for metric, expected in baseline_expected.items():
        if not np.isclose(official["b0_upstream"][metric], expected, atol=1e-12):
            raise RuntimeError(f"fresh B0 {metric} differs from registered baseline")

    learned_ids = sorted(method for method in methods if method.startswith("learned_seed"))
    comparator_ids = list(protocol["decision_gate"]["official_gain"]["comparator_pool"])
    simple_ids = ["zero_delta", "constant_velocity"]
    learned_mean = {
        metric: float(np.mean([official[method][metric] for method in learned_ids]))
        for metric in ("HOTA", "AssA", "DetA", "IDF1", "IDSW", "MOTA", "Frag")
    }
    best_comparator = {
        metric: max(official[method][metric] for method in comparator_ids)
        for metric in ("HOTA", "AssA")
    }
    gate = protocol["decision_gate"]
    gain = gate["official_gain"]
    safety = gate["detection_safety"]
    recovery_gate = gate["recovery_mechanism"]
    long_gate = gate["long_gap_observation"]
    engineering_gate = gate["engineering_safety"]

    sequence_positive = 0
    per_sequence_mean_learned = {}
    for sequence in protocol["scope"]["sequences"]:
        mean_hota = float(np.mean([
            trackeval[method]["per_sequence"][sequence]["HOTA"] for method in learned_ids
        ]))
        difference = mean_hota - trackeval["b0_upstream"]["per_sequence"][sequence]["HOTA"]
        sequence_positive += int(difference > 0)
        per_sequence_mean_learned[sequence] = {
            "HOTA": mean_hota,
            "HOTA_difference_vs_B0": difference,
        }

    learned_recoveries = [
        recovery[method]["overall"]["baseline_failure_recoveries"] for method in learned_ids
    ]
    learned_harms = [
        recovery[method]["overall"]["success_control_harms"] for method in learned_ids
    ]
    simple_recoveries = {
        method: recovery[method]["overall"]["baseline_failure_recoveries"]
        for method in simple_ids
    }
    learned_long = [
        recovery[method]["by_gap"]["gap16_31"].get("persistence_success", 0)
        for method in learned_ids
    ]
    simple_long = {
        method: recovery[method]["by_gap"]["gap16_31"].get("persistence_success", 0)
        for method in simple_ids
    }

    engineering_by_method = {}
    for method in learned_ids:
        sequences = runs[method]["sequences"]
        maximum_p95 = max(
            float(row["runner"]["tracker_update_latency"]["p95_ms"])
            for row in sequences
        )
        nonfinite = sum(
            int(row["runner"]["policy"]["counters"].get("fallback_nonfinite_prediction", 0))
            for row in sequences
        )
        invalid = sum(
            int(row["runner"]["policy"]["counters"].get("fallback_invalid_prediction_box", 0))
            for row in sequences
        )
        birth = {
            key: sum(
                int(row["runner"]["birth_mapping_counters"].get(key, 0))
                for row in sequences
            )
            for key in ("mapped_birth", "unresolved_birth", "ambiguous_birth")
        }
        birth_total = sum(birth.values())
        bad_fraction = (
            (birth["unresolved_birth"] + birth["ambiguous_birth"]) / birth_total
            if birth_total else 0.0
        )
        engineering_by_method[method] = {
            "maximum_sequence_p95_ms": maximum_p95,
            "nonfinite_prediction_fallbacks": nonfinite,
            "invalid_prediction_box_fallbacks": invalid,
            "birth_mapping": birth,
            "unresolved_plus_ambiguous_birth_fraction": bad_fraction,
        }

    checks = {
        "learned_mean_HOTA_gain": learned_mean["HOTA"] - best_comparator["HOTA"]
        >= float(gain["learned_mean_HOTA_above_best_comparator_percentage_points"]),
        "learned_mean_AssA_gain": learned_mean["AssA"] - best_comparator["AssA"]
        >= float(gain["learned_mean_AssA_above_best_comparator_percentage_points"]),
        "minimum_seeds_above_best_HOTA_and_AssA": sum(
            official[method]["HOTA"] > best_comparator["HOTA"]
            and official[method]["AssA"] > best_comparator["AssA"]
            for method in learned_ids
        ) >= int(gain["minimum_learned_seeds_strictly_above_best_comparator_on_both_HOTA_and_AssA"]),
        "worst_seed_HOTA_safety": min(
            official[method]["HOTA"] - official["b0_upstream"]["HOTA"]
            for method in learned_ids
        ) >= float(gain["worst_learned_seed_HOTA_vs_B0_minimum_percentage_points"]),
        "paired_sequence_HOTA_consistency": sequence_positive
        >= int(gain["minimum_sequences_with_positive_mean_learned_HOTA_difference_vs_B0"]),
        "learned_mean_DetA_safety": learned_mean["DetA"] - official["b0_upstream"]["DetA"]
        >= float(safety["learned_mean_DetA_vs_B0_minimum_percentage_points"]),
        "worst_seed_DetA_safety": min(
            official[method]["DetA"] - official["b0_upstream"]["DetA"]
            for method in learned_ids
        ) >= float(safety["worst_learned_seed_DetA_vs_B0_minimum_percentage_points"]),
        "persistent_recovery_gain_vs_simple": float(np.mean(learned_recoveries))
        - max(simple_recoveries.values())
        >= float(recovery_gate["learned_mean_additional_persistent_recoveries_vs_best_simple_predictor_minimum"]),
        "net_recoveries_minus_harms": float(np.mean(np.asarray(learned_recoveries) - np.asarray(learned_harms)))
        >= float(recovery_gate["learned_mean_net_recoveries_minus_success_control_harms_minimum"]),
        "mean_success_control_harms": float(np.mean(learned_harms))
        <= float(recovery_gate["learned_mean_success_control_harms_maximum"]),
        "each_seed_success_control_harms": max(learned_harms)
        <= int(recovery_gate["each_learned_seed_success_control_harms_maximum"]),
        "long_gap_each_seed_at_least_B0": min(learned_long)
        >= recovery["b0_upstream"]["by_gap"]["gap16_31"].get("persistence_success", 0),
        "long_gap_mean_at_least_best_simple": float(np.mean(learned_long))
        >= max(simple_long.values()),
        "learned_latency": all(
            value["maximum_sequence_p95_ms"]
            <= float(engineering_gate["each_learned_sequence_tracker_update_p95_ms_maximum"])
            for value in engineering_by_method.values()
        ),
        "learned_predictions_finite_and_valid": all(
            value["nonfinite_prediction_fallbacks"]
            <= int(engineering_gate["nonfinite_prediction_fallbacks_maximum_per_run"])
            and value["invalid_prediction_box_fallbacks"]
            <= int(engineering_gate["invalid_prediction_box_fallbacks_maximum_per_run"])
            for value in engineering_by_method.values()
        ),
        "learned_birth_mapping": all(
            value["unresolved_plus_ambiguous_birth_fraction"]
            <= float(engineering_gate["unresolved_plus_ambiguous_birth_fraction_maximum"])
            for value in engineering_by_method.values()
        ),
    }
    decision = "GO" if all(checks.values()) else "NO_GO"
    summary = {
        "execution_status": "PASS",
        "decision": decision,
        "protocol_sha256": args.protocol_sha256,
        "checks": checks,
        "official_combined": official,
        "learned_mean": learned_mean,
        "best_comparator": best_comparator,
        "per_sequence_mean_learned": per_sequence_mean_learned,
        "positive_HOTA_sequence_count": sequence_positive,
        "recovery": {
            "learned_baseline_failure_recoveries": dict(zip(learned_ids, learned_recoveries)),
            "learned_success_control_harms": dict(zip(learned_ids, learned_harms)),
            "simple_baseline_failure_recoveries": simple_recoveries,
            "learned_gap16_31_persistence_success": dict(zip(learned_ids, learned_long)),
            "simple_gap16_31_persistence_success": simple_long,
        },
        "engineering": engineering_by_method,
        "mandatory_interpretation": {
            "long_gap": long_gate["mandatory_interpretation"],
            "scope": gate["interpretation"],
            "official_val_used": False,
        },
    }
    (output_partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_partial / "input_hashes.json").write_text(json.dumps({
        "protocol": sha256_file(args.protocol),
        "run_summaries": {method: sha256_file(path) for method, path in run_paths.items()},
        "trackeval_summaries": {
            method: sha256_file(path) for method, path in trackeval_paths.items()
        },
        "recovery_summaries": {
            method: sha256_file(path) for method, path in recovery_paths.items()
        },
        "aggregator": sha256_file(Path(__file__)),
    }, indent=2, sort_keys=True) + "\n")
    os.replace(output_partial, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
