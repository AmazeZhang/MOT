#!/usr/bin/env python3
"""Frozen linear fit and fixed-coverage, video-held-out R1-B diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import expit

from run_p1_natural_closed_loop import sha256_file
from run_r1_stage_b_labels import FEATURE_NAMES


def load_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def design(rows, mean, scale):
    raw = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows],
                     dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE_NAMES) or not np.isfinite(raw).all():
        raise RuntimeError("invalid feature table")
    normalized = (raw - mean) / scale
    return np.column_stack([np.ones(len(rows)), normalized])


def fit_linear_harm(x, rows, label_name, ridge_l2, logistic_l2):
    y = np.asarray([row[label_name]["net_correct_frame_delta"] for row in rows], dtype=np.float64)
    harm = np.asarray([row[label_name]["harmed_anchors"] > 0 for row in rows], dtype=np.float64)
    penalty = np.diag([0.0] + [ridge_l2] * (x.shape[1] - 1))
    ridge = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    if len(np.unique(harm)) == 1:
        rate = (harm.sum() + 0.5) / (len(harm) + 1.0)
        logistic = np.zeros(x.shape[1], dtype=np.float64)
        logistic[0] = math.log(rate / (1 - rate))
    else:
        def objective(weight):
            logits = x @ weight
            reg = 0.5 * logistic_l2 * np.sum(weight[1:] ** 2)
            value = np.logaddexp(0, logits).sum() - np.dot(harm, logits) + reg
            grad = x.T @ (expit(logits) - harm)
            grad[1:] += logistic_l2 * weight[1:]
            return value, grad
        result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if not result.success:
            raise RuntimeError(f"harm logistic fit failed: {result.message}")
        logistic = result.x
    return ridge, logistic


def metrics(rows):
    return {
        "selected": len(rows),
        "net_correct_frame_delta": int(sum(r["future_label"]["net_correct_frame_delta"] for r in rows)),
        "benefited_anchors": int(sum(r["future_label"]["benefited_anchors"] for r in rows)),
        "harmed_anchors": int(sum(r["future_label"]["harmed_anchors"] for r in rows)),
        "selected_direct_anchor_actions": int(sum(r["direct_anchor_count"] > 0 for r in rows)),
        "direct_anchor_net_delta": int(sum(r["future_label"]["net_correct_frame_delta"]
                                           for r in rows if r["direct_anchor_count"] > 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = yaml.safe_load(args.protocol.read_text())
    summary = json.loads((args.label_dir / "summary.json").read_text())
    if summary["status"] != "complete" or summary["protocol_sha256"] != sha256_file(args.protocol):
        raise RuntimeError("label run incomplete or protocol hash mismatch")
    fit_seq = protocol["split"]["fit_sequences"]
    heldout_seq = protocol["split"]["heldout_sequences"]
    inputs = {"summary.json": sha256_file(args.label_dir / "summary.json")}
    by_sequence = {}
    for sequence in fit_seq + heldout_seq:
        path = args.label_dir / f"{sequence}.jsonl"
        rows = load_rows(path)
        inputs[path.name] = sha256_file(path)
        record = next(row for row in summary["sequence_summaries"] if row["sequence"] == sequence)
        if len(rows) != record["selected"]:
            raise RuntimeError(f"row count mismatch for {sequence}")
        if any(row["sequence"] != sequence or row["group"] != record["group"] for row in rows):
            raise RuntimeError(f"sequence/group mismatch in {sequence}")
        if len({row["frame"] for row in rows}) != len(rows):
            raise RuntimeError(f"duplicate selected frame in {sequence}")
        by_sequence[sequence] = rows
    fit_rows = [row for seq in fit_seq for row in by_sequence[seq]]
    heldout_rows = [row for seq in heldout_seq for row in by_sequence[seq]]
    if not fit_rows or not heldout_rows:
        raise RuntimeError("fit or heldout labels empty")
    raw_fit = np.asarray([[row["features"][name] for name in FEATURE_NAMES]
                          for row in fit_rows], dtype=np.float64)
    mean = raw_fit.mean(axis=0)
    scale = raw_fit.std(axis=0)
    scale[scale == 0] = 1.0
    x_fit = design(fit_rows, mean, scale)
    x_heldout = design(heldout_rows, mean, scale)
    learning = protocol["learning"]
    models = {}
    for label_name in ("future_label", "immediate_label"):
        ridge, logistic = fit_linear_harm(
            x_fit, fit_rows, label_name,
            float(learning["ridge_l2"]), float(learning["logistic_l2"]),
        )
        models[label_name] = {
            "signed_delta_weights": ridge.tolist(), "harm_logistic_weights": logistic.tolist(),
            "fit_rows": len(fit_rows),
            "fit_harm_prevalence": float(np.mean([row[label_name]["harmed_anchors"] > 0
                                                  for row in fit_rows])),
        }
    heldout_score = {}
    for label_name, method in (("future_label", "future_linear"),
                               ("immediate_label", "immediate_linear")):
        w = np.asarray(models[label_name]["signed_delta_weights"])
        h = np.asarray(models[label_name]["harm_logistic_weights"])
        scores = x_heldout @ w - expit(x_heldout @ h)
        heldout_score[method] = scores
    score_by_key = {}
    for index, row in enumerate(heldout_rows):
        score_by_key[(row["sequence"], row["frame"])] = {
            method: float(values[index]) for method, values in heldout_score.items()
        }
    per_video = []
    fraction = float(protocol["evaluation"]["selection_fraction_per_sequence"])
    for sequence in heldout_seq:
        rows = by_sequence[sequence]
        k = math.ceil(fraction * len(rows))
        method_results = {}
        for method in ("future_linear", "immediate_linear", "simple_iou_difference"):
            def score(row):
                if method == "simple_iou_difference":
                    return row["features"]["iou_difference"]
                return score_by_key[(row["sequence"], row["frame"])][method]
            ranked = sorted(rows, key=lambda r: (-score(r), r["rank"]))
            method_results[method] = metrics(ranked[:k])
        random_results = []
        for seed in range(20):
            ranked = sorted(rows, key=lambda row: hashlib.sha256(
                f"r1-stage-b-random:{seed:02d}:{sequence}:{row['frame']:06d}:{row['component_signature']}".encode("ascii")
            ).hexdigest())
            random_results.append(metrics(ranked[:k]))
        method_results["random_20_fixed_seeds_mean"] = {
            key: float(np.mean([value[key] for value in random_results]))
            for key in random_results[0]
        }
        method_results["no_intervention"] = metrics([])
        method_results["all_intervention"] = metrics(rows)
        per_video.append({"sequence": sequence, "candidates": len(rows),
                          "direct_anchor_candidate_actions": sum(r["direct_anchor_count"] > 0 for r in rows),
                          "selected_count": k, "methods": method_results})

    aggregate = {}
    for method in per_video[0]["methods"]:
        aggregate[method] = {key: float(sum(video["methods"][method][key] for video in per_video))
                             for key in per_video[0]["methods"][method]}
        selected = aggregate[method]["selected"]
        aggregate[method]["net_per_selected_action"] = (
            aggregate[method]["net_correct_frame_delta"] / selected if selected else None)
    methods = aggregate
    future = methods["future_linear"]
    simple = methods["simple_iou_difference"]
    immediate = methods["immediate_linear"]
    random = methods["random_20_fixed_seeds_mean"]
    gate = protocol["evaluation"]["gates_for_considering_stage_c"]
    checks = {
        "all_heldout_videos_present": sum(video["candidates"] > 0 for video in per_video)
            >= gate["minimum_heldout_videos_with_candidates"],
        "direct_anchor_coverage": sum(video["direct_anchor_candidate_actions"] for video in per_video)
            >= gate["minimum_direct_anchor_actions_total"],
        "future_vs_simple_net": future["net_per_selected_action"] - simple["net_per_selected_action"]
            >= gate["future_minus_simple_mean_net_per_selected_action"],
        "future_vs_random_net": future["net_per_selected_action"] - random["net_per_selected_action"]
            >= gate["future_minus_random_mean_net_per_selected_action"],
        "future_vs_immediate_net": future["net_per_selected_action"] - immediate["net_per_selected_action"]
            >= gate["future_minus_immediate_mean_net_per_selected_action"],
        "future_beats_simple_videos": sum(
            video["methods"]["future_linear"]["net_correct_frame_delta"] >
            video["methods"]["simple_iou_difference"]["net_correct_frame_delta"]
            for video in per_video) >= gate["minimum_videos_future_beats_simple_net"],
        "future_beats_immediate_videos": sum(
            video["methods"]["future_linear"]["net_correct_frame_delta"] >
            video["methods"]["immediate_linear"]["net_correct_frame_delta"]
            for video in per_video) >= gate["minimum_videos_future_beats_immediate_net"],
        "future_harm_no_more_than_simple": future["harmed_anchors"] <= simple["harmed_anchors"],
    }
    output = {
        "status": "complete", "protocol_sha256": sha256_file(args.protocol),
        "label_run_dir": str(args.label_dir.resolve()), "input_hashes": inputs,
        "fit_videos": len(fit_seq), "heldout_videos": len(heldout_seq),
        "fit_actions": len(fit_rows), "heldout_actions": len(heldout_rows),
        "fit_direct_anchor_actions": sum(r["direct_anchor_count"] > 0 for r in fit_rows),
        "heldout_direct_anchor_actions": sum(r["direct_anchor_count"] > 0 for r in heldout_rows),
        "features": list(FEATURE_NAMES), "train_normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "models": models, "per_video": per_video, "aggregate": aggregate,
        "stage_c_gate": {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks},
        "interpretation": "offline, fixed-budget short-horizon proxy; no online threshold or TrackEval claim",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "fit_actions": len(fit_rows),
                      "heldout_actions": len(heldout_rows), "aggregate": aggregate,
                      "stage_c_gate": output["stage_c_gate"]}), flush=True)


if __name__ == "__main__":
    main()
