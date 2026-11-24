#!/usr/bin/env python3
"""Replay factual M and archive frozen causal S06 component features."""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import sys

import numpy as np

import run_s01_visual_baselines as s01
from mot_research.analysis.component_action_features import (
    cosine_matrix, signed_competition_margins, upstream_direction_score,
)
from mot_research.analysis.multi_assignment_components import components
from mot_research.association.appearance_core import fuse_appearance_score


ROOT = Path(__file__).resolve().parents[1]


class S06FeaturePolicy(s01.VisualPolicy):
    """Execute M unchanged while recording only causal observable features."""

    def __call__(self, request, upstream_associate):
        if self.begun:
            raise RuntimeError("appearance memory begin_frame repeated")
        if len(request.detections) != len(self.det_ids):
            raise RuntimeError("source high detection IDs misaligned")
        past, written = self.memory.begin_frame(request.frame_id, request.tracker_ids)
        self.begun = True
        history = self.cv_policy.history
        lengths, gaps = [], []
        for tid in request.tracker_ids:
            obs = history.observations(tid)
            lengths.append(min(len(obs), history.history_length))
            gaps.append(request.frame_id - obs[-1].frame_id if obs else -1)
        self.trace.update({
            "requested_det_ids": list(self.det_ids),
            "requested_tracker_ids": list(request.tracker_ids),
            "memory_last_write_frames": written.tolist(),
            "history_lengths": lengths,
            "observation_gaps": gaps,
        })

        def associate(dets, trks, threshold, velocities, observations, inertia):
            iou = self.assoc_module.iou_batch(dets, trks) if len(dets) and len(trks) \
                else np.empty((len(dets), len(trks)), dtype=np.float64)
            direction = upstream_direction_score(dets, velocities, observations, inertia)
            cosine, cosine_valid = cosine_matrix(self.det_features, past)
            fused_m, used_m = fuse_appearance_score(
                iou + direction, self.det_features, past, iou >= 0.3, weight=0.25)
            row_margin, column_margin = signed_competition_margins(fused_m)
            eligible = iou > threshold
            fast_path = bool(len(dets) and len(trks) and
                             eligible.sum(1).max() == 1 and eligible.sum(0).max() == 1)
            original_lap = self.assoc_module.linear_assignment

            def candidate(weight):
                used_edges = 0
                if weight == 0:
                    return upstream_associate(dets, trks, threshold, velocities,
                                              observations, inertia), used_edges
                def visual_lap(cost):
                    nonlocal used_edges
                    fused, used = fuse_appearance_score(
                        -cost, self.det_features, past, iou >= 0.3, weight=weight)
                    used_edges = int(used.sum())
                    return original_lap(-fused)
                self.assoc_module.linear_assignment = visual_lap
                try:
                    result = upstream_associate(dets, trks, threshold, velocities,
                                                observations, inertia)
                finally:
                    self.assoc_module.linear_assignment = original_lap
                return result, used_edges

            proposals, used = {}, {}
            for name, weight in (("G", 0.0), ("V", 0.5), ("M", 0.25)):
                proposals[name], used[name] = candidate(weight)
            assignments = {name: value[0] for name, value in proposals.items()}
            parts = components(assignments, len(dets), len(trks), reference="M")
            self.trace.update({
                "fast_path": fast_path,
                "appearance_eligible_edges": used["M"],
                "candidate_appearance_eligible_edges": used,
                "candidate_first_pairs": {
                    name: self._pairs(value[0], request.tracker_ids)
                    for name, value in proposals.items()
                },
                "candidate_row_pairs": {
                    name: np.asarray(value[0], dtype=np.int64).reshape(-1, 2).tolist()
                    for name, value in proposals.items()
                },
                "components": [{
                    "detection_rows": list(part.detection_rows),
                    "tracker_rows": list(part.tracker_rows),
                    "actions": {name: [list(edge) for edge in edges]
                                for name, edges in part.actions},
                    "unique_actions": list(part.unique_actions(("M", "G", "V"))),
                } for part in parts],
                "s06_features": {
                    "detection_scores": np.asarray(dets[:, 4], dtype=float).tolist(),
                    "iou": iou.tolist(),
                    "direction_score": direction.tolist(),
                    "ema_detection_cosine": cosine.tolist(),
                    "cosine_valid": cosine_valid.tolist(),
                    "M_fused_score": fused_m.tolist(),
                    "row_best_margin": row_margin.tolist(),
                    "column_best_margin": column_margin.tolist(),
                    "M_used_appearance": used_m.tolist(),
                },
            })
            self.trace["base_first_pairs"] = self.trace["candidate_first_pairs"]["G"]
            self.trace["first_pairs"] = self.trace["candidate_first_pairs"]["M"]
            return proposals["M"]

        return self.cv_policy(request, associate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S06 instrumentation must use CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if s01.sha256(ROOT / lock["protocol_path"]) != lock["protocol_sha256"]:
        raise RuntimeError("S06 protocol changed after runtime lock")
    for path, expected in lock["code_hashes"].items():
        if s01.sha256(ROOT / path) != expected:
            raise RuntimeError(f"source changed after runtime lock: {path}")
    sequence = lock["selected_sequences"][0]
    cache_file = Path(lock["detection_root"]) / f"{sequence}.csv"
    feature_file = Path(lock["feature_cache_path"])
    if s01.sha256(cache_file) != lock["input_detection_sha256"][sequence] or \
            s01.sha256(feature_file) != lock["feature_cache_sha256"]:
        raise RuntimeError("S06 input cache changed")
    sys.path.insert(0, str((ROOT / "third_party/OC_SORT").resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module
    config = configparser.ConfigParser()
    seq_dir = Path(lock["train_root"]) / sequence
    if not config.read(seq_dir / "seqinfo.ini"):
        raise FileNotFoundError(seq_dir / "seqinfo.ini")
    info = config["Sequence"]
    cached, _ = s01.read_detection_cache(cache_file)
    source_ids = s01.all_detection_ids(cache_file)
    feature_map = s01.load_features(feature_file)
    original_policy = s01.VisualPolicy
    s01.VisualPolicy = S06FeaturePolicy
    try:
        result = s01.run_condition(
            "M", lock, sequence, info.getint("seqLength"), cached, source_ids,
            feature_map, Path(lock["output_root"]), ocsort_module, assoc_module,
            info.getint("imHeight"), info.getint("imWidth"), info.getfloat("frameRate"))
    finally:
        s01.VisualPolicy = original_policy
    tracker = Path(lock["output_root"]) / "trackers/M" / f"{sequence}.txt"
    if s01.sha256(tracker) != lock["authoritative_M_tracker_sha256"]:
        raise RuntimeError("instrumented S06 M differs from authoritative S01 M")
    trace = Path(lock["output_root"]) / "traces/M" / f"{sequence}.jsonl"
    result.update({"S06_feature_trace_sha256": s01.sha256(trace),
                   "authoritative_M_output_exact": True})
    checks = {"status": "ENGINEERING_PASS", "run_id": lock["run_id"],
              "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
              "result": result}
    path = Path(lock["output_root"]) / "engineering_checks.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
