#!/usr/bin/env python3
"""Frozen CPU-only R2 P1 score comparison on captured common B0 states."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.special import logsumexp

from analyze_p1_common_state_mechanism import read_events
from mot_research.analysis.runtime_record import (
    runtime_record, sha256_file, verify_code_unchanged,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/r2_decision_rules_p1_20261118.yaml"
RULES = ("S0", "S1", "S2", "S3", "S4")
CODE_PATHS = (
    "scripts/rescore_r2_decision_rules.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/runtime_record.py",
)


def valid_box(box):
    box = np.asarray(box, dtype=np.float64)
    return bool(np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1])


def box_from_log_state(values):
    cx, cy, log_area, log_aspect = values
    log_area = float(np.clip(log_area, -20, 30))
    log_aspect = float(np.clip(log_aspect, -8, 8))
    width = math.exp(0.5 * (log_area + log_aspect))
    height = math.exp(0.5 * (log_area - log_aspect))
    return np.array([cx - width / 2, cy - height / 2,
                     cx + width / 2, cy + height / 2], dtype=np.float64)


def iou_many(boxes, detections):
    boxes = np.asarray(boxes, dtype=np.float64)
    detections = np.asarray(detections, dtype=np.float64)
    if not len(boxes) or not len(detections):
        return np.zeros((len(detections), len(boxes)), dtype=np.float64)
    left = np.maximum(detections[:, None, :2], boxes[None, :, :2])
    right = np.minimum(detections[:, None, 2:], boxes[None, :, 2:])
    wh = np.maximum(right - left, 0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    da = np.prod(np.maximum(detections[:, 2:] - detections[:, :2], 0), axis=1)
    ba = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    union = da[:, None] + ba[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def kf_samples(record, row, count, height, width, sequence, frame, track_id, counters):
    fallback = np.repeat(np.asarray(record["predicted_kf_boxes_xyxy"][row],
                                    dtype=np.float64)[None, :], count, axis=0)
    x = np.asarray(record["kf_x_after_predict"][row], dtype=np.float64)
    p = np.asarray(record["kf_P_after_predict"][row], dtype=np.float64)
    if x.shape != (7,) or p.shape != (7, 7) or not np.isfinite(x).all() or not np.isfinite(p).all():
        counters["kf_invalid_shape_or_nonfinite"] += 1
        return fallback
    area, aspect = x[2], x[3]
    if area <= 1e-8 or aspect <= 1e-8:
        counters["kf_invalid_area_aspect"] += 1
        return fallback
    mean = np.array([x[0], x[1], math.log(area), math.log(aspect)])
    jac = np.diag([1, 1, 1 / area, 1 / aspect])
    covariance = jac @ p[:4, :4] @ jac
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        counters["kf_invalid_covariance"] += 1
        return fallback
    eigval, eigvec = np.linalg.eigh(covariance)
    covariance = (eigvec * np.maximum(eigval, 1e-10)) @ eigvec.T
    caps = np.array([.2 * width, .2 * height, 1., 1.])
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-10))
    factor = np.minimum(1., caps / std)
    covariance *= factor[:, None] * factor[None, :]
    eigval, eigvec = np.linalg.eigh((covariance + covariance.T) / 2)
    transform = eigvec @ np.diag(np.sqrt(np.maximum(eigval, 1e-10)))
    seed_text = f"r2-p1-20261118:{sequence}:{frame:06d}:{track_id}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("ascii")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = mean[None, :] + rng.standard_normal((count, 4)) @ transform.T
    boxes = np.stack([box_from_log_state(sample) for sample in samples])
    if not all(valid_box(box) for box in boxes):
        counters["kf_sample_invalid_box"] += 1
        return fallback
    return boxes


def turn_modes(record, row, frame, counters):
    fallback = np.repeat(np.asarray(record["predicted_kf_boxes_xyxy"][row],
                                    dtype=np.float64)[None, :], 2, axis=0)
    history = record["history_observations"][row]
    if len(history) < 3:
        counters["turn_insufficient_history"] += 1
        return fallback
    latest = history[-3:]
    times = np.asarray([item["frame"] for item in latest], dtype=np.float64)
    boxes = np.asarray([item["box"] for item in latest], dtype=np.float64)
    if not (np.all(np.diff(times) > 0) and all(valid_box(box) for box in boxes)):
        counters["turn_invalid_history"] += 1
        return fallback
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    v1 = (centers[1] - centers[0]) / (times[1] - times[0])
    v2 = (centers[2] - centers[1]) / (times[2] - times[1])
    if min(np.linalg.norm(v1), np.linalg.norm(v2)) < 1e-8:
        counters["turn_near_zero_velocity"] += 1
        return fallback
    angle = math.atan2(np.linalg.det(np.stack([v1, v2])), float(np.dot(v1, v2)))
    angle_per_frame = float(np.clip(angle / (times[2] - times[1]), -math.pi / 4, math.pi / 4))
    gap = frame - times[2]
    if gap <= 0:
        counters["turn_invalid_gap"] += 1
        return fallback
    size = boxes[2, 2:] - boxes[2, :2]
    proposals = []
    for sign in (-1, 1):
        theta = sign * angle_per_frame * gap
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        center = centers[2] + gap * (rotation @ v2)
        proposals.append(np.r_[center - size / 2, center + size / 2])
    counters["turn_applied"] += 1
    return np.stack(proposals)


def box_features(boxes, height, width):
    boxes = np.asarray(boxes, dtype=np.float64)
    size = np.maximum(boxes[:, 2:] - boxes[:, :2], 1e-12)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    return np.column_stack((center[:, 0] / width, center[:, 1] / height,
                            np.log(size[:, 0] / width), np.log(size[:, 1] / height)))


def score_frame(record, variant, protocol, counters, height, width):
    detections = np.asarray(record["detections_xyxy_score"], dtype=np.float64)
    det_boxes = detections[:, :4].reshape(-1, 4)
    tracker_ids = record["tracker_ids"]
    n_d, n_t = len(det_boxes), len(tracker_ids)
    scores = {key: np.full((n_d, n_t), -np.inf) for key in RULES}
    eligible = np.zeros((n_d, n_t), dtype=bool)
    if not n_t or not n_d:
        return scores, eligible
    for col, track_id in enumerate(tracker_ids):
        kf = kf_samples(record, col, 20, height, width,
                        record["sequence"], record["frame"], track_id, counters)
        turns = turn_modes(record, col, record["frame"], counters)
        if variant == "b0_upstream":
            point = np.asarray(record["predicted_kf_boxes_xyxy"][col], dtype=np.float64)
        else:
            point = np.asarray(record["proposals_xyxy"][variant][col], dtype=np.float64)
        if variant.startswith("learned_"):
            turns = np.repeat(point[None, :], 2, axis=0)
        hypotheses = np.concatenate((kf, turns), axis=0)
        if not valid_box(point) or not all(valid_box(box) for box in hypotheses):
            counters["invalid_hypothesis_or_point"] += 1
            continue
        ious = iou_many(hypotheses, det_boxes)
        eligibility = np.max(ious, axis=1) >= protocol["inputs"]["tracker"]["iou_threshold"]
        eligible[:, col] = eligibility
        point_iou = iou_many(point[None, :], det_boxes)[:, 0]
        scores["S0"][:, col] = point_iou
        scores["S1"][:, col] = np.mean(ious, axis=1)
        scores["S2"][:, col] = np.max(ious, axis=1)
        std = np.array([.02, .02, .10, .10])
        residual = (box_features(det_boxes, height, width)[:, None, :]
                    - box_features(hypotheses, height, width)[None, :, :]) / std
        log_kernel = -.5 * np.sum(residual ** 2, axis=2)
        scores["S3"][:, col] = logsumexp(log_kernel, axis=1) - math.log(len(hypotheses))
    # A common gate makes all five rules comparable within a variant.
    for key in ("S0", "S1", "S2", "S3"):
        scores[key][~eligible] = -np.inf
    base = scores["S2"]
    margin_score = base.copy()
    for i in range(n_d):
        for j in range(n_t):
            if not eligible[i, j]:
                continue
            row_other = [base[i, k] for k in range(n_t) if k != j and eligible[i, k]]
            col_other = [base[k, j] for k in range(n_d) if k != i and eligible[k, j]]
            row_margin = base[i, j] - max(row_other) if row_other else math.inf
            col_margin = base[i, j] - max(col_other) if col_other else math.inf
            margin = min(row_margin, col_margin)
            margin_score[i, j] = base[i, j] - .2 * max(0, .05 - margin)
    scores["S4"] = margin_score
    return scores, eligible


def evaluate_event(record, event, variant, scores, eligible):
    det_ids, track_ids = record["detection_ids"], record["tracker_ids"]
    intended = int(event["intended_track_id"])
    positive = int(event["positive_detection_id"])
    if positive not in det_ids:
        raise RuntimeError(f"{event['event_id']}: positive detection missing")
    det_row = det_ids.index(positive)
    applicable = intended in track_ids
    target_col = track_ids.index(intended) if applicable else None
    out = []
    for rule in RULES:
        candidate_cols = [j for j in range(len(track_ids)) if eligible[det_row, j]]
        order = sorted(candidate_cols, key=lambda j: (-scores[rule][det_row, j], track_ids[j]))
        winner = track_ids[order[0]] if order else None
        success = applicable and winner == intended
        target_eligible = bool(applicable and eligible[det_row, target_col])
        target_score = (float(scores[rule][det_row, target_col]) if target_eligible else None)
        next_score = (float(scores[rule][det_row, order[1]]) if len(order) > 1 else None)
        out.append({
            "event_id": event["event_id"], "sequence": event["sequence"],
            "frame": event["after_frame"], "baseline_stratum": event["baseline_stratum"],
            "gap_bin": event["gap_bin"], "variant": variant, "rule": rule,
            "applicable": applicable, "target_eligible": target_eligible,
            "eligible_count": len(order), "winner_track_id": winner,
            "target_track_id": intended, "positive_detection_id": positive,
            "target_score": target_score,
            "target_rank": (order.index(target_col) + 1 if target_eligible else None),
            "winner_margin": (float(scores[rule][det_row, order[0]]) - next_score
                              if len(order) > 1 else None),
            "top1_correct": bool(success),
        })
    return out


def decision_summary(rows, protocol):
    by_key = {(r["event_id"], r["variant"], r["rule"]): r for r in rows}
    primary = protocol["decision"]["primary_variant"]
    event_ids = sorted({r["event_id"] for r in rows})
    applicable = [event_id for event_id in event_ids
                  if by_key[event_id, primary, "S1"]["applicable"]]
    baseline_fail = [e for e in applicable if not by_key[e, primary, "S1"]["top1_correct"]]
    baseline_success = [e for e in applicable if by_key[e, primary, "S1"]["top1_correct"]]
    results = {}
    for rule in protocol["decision"]["candidate_rules"]:
        rescued = [e for e in baseline_fail if by_key[e, primary, rule]["top1_correct"]]
        harmed = [e for e in baseline_success if not by_key[e, primary, rule]["top1_correct"]]
        video_rescue = Counter(by_key[e, primary, rule]["sequence"] for e in rescued)
        fraction = len(rescued) / len(baseline_fail) if baseline_fail else None
        harm_fraction = len(harmed) / len(baseline_success) if baseline_success else None
        passed = bool(fraction is not None and harm_fraction is not None and
                      fraction >= protocol["decision"]["minimum_rescue_fraction"] and
                      harm_fraction <= protocol["decision"]["maximum_new_harm_fraction"] and
                      len(video_rescue) >= 5)
        results[rule] = {"rescued": len(rescued), "harmed": len(harmed),
                         "rescue_fraction": fraction, "harm_fraction": harm_fraction,
                         "rescue_video_count": len(video_rescue),
                         "rescued_event_ids": rescued, "harmed_event_ids": harmed,
                         "promotion_gate_pass": passed}
    descriptive = {}
    for variant in protocol["hypotheses"]["variants"]:
        descriptive[variant] = {}
        for rule in RULES:
            selected = [by_key[e, variant, rule] for e in event_ids]
            descriptive[variant][rule] = {
                "registered": len(selected),
                "applicable": sum(r["applicable"] for r in selected),
                "target_eligible": sum(r["target_eligible"] for r in selected),
                "top1_correct": sum(r["top1_correct"] for r in selected),
            }
    return {"primary_variant": primary, "applicable": len(applicable),
            "S1_failures": len(baseline_fail), "S1_successes": len(baseline_success),
            "rules": results, "descriptive": descriptive,
            "any_promotion_gate_pass": any(x["promotion_gate_pass"] for x in results.values())}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("R2 P1 rescore requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    capture = Path(protocol["outputs"]["capture_run"])
    capture_summary = json.loads((capture / "summary.json").read_text())
    if capture_summary["status"] != "PASS" or capture_summary["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("capture failed or protocol changed")
    events_path = Path(protocol["events"]["path"])
    if sha256_file(events_path) != protocol["events"]["sha256"]:
        raise RuntimeError("events hash mismatch")
    output = Path(protocol["outputs"]["rescore_run"])
    partial = output.with_name(output.name + ".partial")
    failed = output.with_name(output.name + ".failed")
    if any(p.exists() for p in (output, partial, failed)):
        raise FileExistsError(f"rescore output already exists: {output}")
    partial.mkdir(parents=True)
    try:
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"protocol_sha256": sha256_file(PROTOCOL),
                           "capture_summary_sha256": sha256_file(capture / "summary.json"),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        events = read_events(events_path)
        by_frame = defaultdict(lambda: defaultdict(list))
        for event in events:
            by_frame[event["sequence"]][event["after_frame"]].append(event)
        rows = []
        counters = Counter()
        with (partial / "frame_scores.jsonl").open("w") as score_sink:
            for sequence in protocol["scope"]["sequence_order"]:
                capture_file = capture / "states" / f"{sequence}.jsonl"
                expected_hash = next(s["state_sha256"] for s in capture_summary["sequences"]
                                     if s["sequence"] == sequence)
                if sha256_file(capture_file) != expected_hash:
                    raise RuntimeError(f"{sequence}: capture state hash mismatch")
                config = __import__("configparser").ConfigParser()
                config.read(Path(protocol["inputs"]["train_root"]) / sequence / "seqinfo.ini")
                height = config["Sequence"].getint("imHeight")
                width = config["Sequence"].getint("imWidth")
                seen = set()
                with capture_file.open() as handle:
                    for line in handle:
                        record = json.loads(line)
                        frame = record["frame"]
                        if frame in seen or frame not in by_frame[sequence]:
                            raise RuntimeError(f"{sequence}: duplicate or unexpected frame {frame}")
                        seen.add(frame)
                        if set(record["event_ids"]) != {e["event_id"] for e in by_frame[sequence][frame]}:
                            raise RuntimeError(f"{sequence}:{frame}: event IDs mismatch")
                        for variant in protocol["hypotheses"]["variants"]:
                            scores, eligible = score_frame(record, variant, protocol,
                                                           counters, height, width)
                            score_sink.write(json.dumps({
                                "sequence": sequence, "frame": frame,
                                "variant": variant, "detection_ids": record["detection_ids"],
                                "tracker_ids": record["tracker_ids"],
                                "eligibility": eligible.tolist(),
                                "scores": {k: [[float(x) if np.isfinite(x) else None for x in line]
                                               for line in v]
                                           for k, v in scores.items()},
                            }, separators=(",", ":"), allow_nan=False) + "\n")
                            for event in by_frame[sequence][frame]:
                                rows.extend(evaluate_event(record, event, variant, scores, eligible))
                if seen != set(by_frame[sequence]):
                    raise RuntimeError(f"{sequence}: missing captured frames")
                print(f"{sequence}: rescored {len(seen)} event frames", flush=True)
        expected = len(events) * len(protocol["hypotheses"]["variants"]) * len(RULES)
        if len(rows) != expected or len({(r["event_id"], r["variant"], r["rule"]) for r in rows}) != expected:
            raise RuntimeError(f"row coverage mismatch: {len(rows)} != {expected}")
        rows.sort(key=lambda r: (r["event_id"], r["variant"], r["rule"]))
        with (partial / "event_rule_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        decision = decision_summary(rows, protocol)
        verify_code_unchanged(ROOT, provenance)
        summary = {"status": "PASS", "scope": "offline common-state positive-detection rank only",
                   "registered_events": len(events), "row_count": len(rows),
                   "protocol_sha256": provenance["protocol_sha256"],
                   "capture_summary_sha256": provenance["capture_summary_sha256"],
                   "event_rule_rows_sha256": sha256_file(partial / "event_rule_rows.csv"),
                   "frame_scores_sha256": sha256_file(partial / "frame_scores.jsonl"),
                   "fallback_counters": dict(counters), "decision": decision,
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "limitations": ["No Hungarian assignment or tracker state writeback.",
                                   "Motion-dev is a development split, not official validation.",
                                   "The mean IoU formula is borrowed as an aggregation; the hypothesis generator is different."]}
        (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(partial, output)
        print(json.dumps({"status": "PASS", "output": str(output),
                          "decision": decision["rules"]}, indent=2), flush=True)
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
