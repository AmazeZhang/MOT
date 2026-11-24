#!/usr/bin/env python3
"""Natural closed-loop replay for one frozen S06 selector checkpoint/policy."""

from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import run_s01_visual_baselines as s01
from mot_research.analysis.component_action_features import (
    cosine_matrix, signed_competition_margins, upstream_direction_score,
)
from mot_research.analysis.multi_assignment_components import apply_action, components
from mot_research.association.appearance_core import fuse_appearance_score
from mot_research.models.component_action_selector import (
    CONTINUOUS_FEATURES, ComponentActionSelector, FeatureNormalization,
)


ROOT = Path(__file__).resolve().parents[1]


class SelectorPolicy(s01.VisualPolicy):
    def __init__(self, *args, model: ComponentActionSelector, risk_multiplier: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.risk_multiplier = risk_multiplier

    def __call__(self, request, upstream_associate):
        if self.begun:
            raise RuntimeError("appearance memory begin_frame repeated")
        past, written = self.memory.begin_frame(request.frame_id, request.tracker_ids)
        self.begun = True
        history = self.cv_policy.history
        lengths, gaps = [], []
        for tid in request.tracker_ids:
            obs = history.observations(tid)
            lengths.append(min(len(obs), history.history_length))
            gaps.append(request.frame_id - obs[-1].frame_id if obs else -1)
        self.trace.update({"requested_det_ids": list(self.det_ids),
                           "requested_tracker_ids": list(request.tracker_ids),
                           "memory_last_write_frames": written.tolist()})

        def associate(dets, trks, threshold, velocities, observations, inertia):
            iou = self.assoc_module.iou_batch(dets, trks) if len(dets) and len(trks) \
                else np.empty((len(dets), len(trks)), dtype=np.float64)
            direction = upstream_direction_score(dets, velocities, observations, inertia)
            cosine, cosine_valid = cosine_matrix(self.det_features, past)
            fused_m, _ = fuse_appearance_score(
                iou + direction, self.det_features, past, iou >= 0.3, weight=0.25)
            row_margin, col_margin = signed_competition_margins(fused_m)
            original_lap = self.assoc_module.linear_assignment

            def candidate(weight):
                if weight == 0:
                    return upstream_associate(dets, trks, threshold, velocities,
                                              observations, inertia)
                def visual_lap(cost):
                    fused, _ = fuse_appearance_score(
                        -cost, self.det_features, past, iou >= 0.3, weight=weight)
                    return original_lap(-fused)
                self.assoc_module.linear_assignment = visual_lap
                try:
                    return upstream_associate(dets, trks, threshold, velocities,
                                              observations, inertia)
                finally:
                    self.assoc_module.linear_assignment = original_lap

            proposals = {name: candidate(weight) for name, weight in
                         (("G", 0.0), ("V", 0.5), ("M", 0.25))}
            assignments = {name: result[0] for name, result in proposals.items()}
            parts = components(assignments, len(dets), len(trks), reference="M")
            selected_records = []
            matched = np.asarray(proposals["M"][0], dtype=np.int64).reshape(-1, 2)
            with torch.no_grad():
                for part in parts:
                    action_map = dict(part.actions)
                    union = sorted(set().union(*[set(value) for value in action_map.values()]))
                    continuous = torch.tensor([[
                        iou[d, t], direction[d, t], dets[d, 4], cosine[d, t],
                        gaps[t], lengths[t], row_margin[d, t], col_margin[d, t],
                    ] for d, t in union], dtype=torch.float32)
                    det_rows = torch.tensor([d for d, _ in union])
                    trk_rows = torch.tensor([t for _, t in union])
                    m_edges = set(action_map["M"])
                    predictions = []
                    for priority, name in enumerate(part.unique_actions(("M", "G", "V"))):
                        action_edges = set(action_map[name])
                        binary = torch.tensor([[
                            float(cosine_valid[d, t]),
                            float((edge in action_edges) != (edge in m_edges)),
                        ] for edge, (d, t) in zip(union, union)], dtype=torch.float32)
                        selected_set = {i for i, edge in enumerate(union) if edge in action_edges}
                        selected = torch.tensor([i in selected_set for i in range(len(union))])
                        comp = torch.tensor([np.log1p(len(part.detection_rows)),
                                             np.log1p(len(part.tracker_rows)),
                                             np.log1p(len(selected_set))], dtype=torch.float32)
                        pred = self.model(continuous, binary, det_rows, trk_rows, selected, comp)
                        g_hat, h_hat = (float(value) for value in pred)
                        score = g_hat - self.risk_multiplier * h_hat
                        predictions.append((score, -priority, name, g_hat, h_hat))
                    best = max(predictions)
                    chosen = best[2] if best[0] > 0 else "M"
                    if chosen != "M":
                        matched, _, _ = apply_action(matched, part, chosen, len(dets), len(trks))
                    selected_records.append({"chosen": chosen, "score": best[0],
                                             "predictions": [{"name": row[2], "score": row[0],
                                                              "g_hat": row[3], "h_hat": row[4]}
                                                             for row in predictions]})
            unmatched_d = np.asarray(sorted(set(range(len(dets))) - set(matched[:, 0])), dtype=np.int64)
            unmatched_t = np.asarray(sorted(set(range(len(trks))) - set(matched[:, 1])), dtype=np.int64)
            self.trace.update({
                "base_first_pairs": self._pairs(proposals["G"][0], request.tracker_ids),
                "first_pairs": self._pairs(matched, request.tracker_ids),
                "selector_components": selected_records,
                "selector_changed_components": sum(row["chosen"] != "M" for row in selected_records),
                "appearance_eligible_edges": int((iou >= 0.3).sum()),
                "fast_path": bool(len(dets) and len(trks) and
                                  (iou > threshold).sum(1).max() == 1 and
                                  (iou > threshold).sum(0).max() == 1),
            })
            return matched, unmatched_d, unmatched_t

        return self.cv_policy(request, associate)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-lock", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S06 closed-loop checkpoint replay is CPU-only")
    lock = json.loads(args.runtime_lock.read_text())
    for path, expected in lock["code_hashes"].items():
        if s01.sha256(ROOT / path) != expected:
            raise RuntimeError(f"source changed: {path}")
    checkpoint_path = Path(lock["checkpoint_path"])
    if s01.sha256(checkpoint_path) != lock["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization = FeatureNormalization.from_dict(checkpoint["normalization"])
    model = ComponentActionSelector(checkpoint["mode"], normalization)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sequence = lock["selected_sequences"][0]
    cache_file = Path(lock["detection_root"]) / f"{sequence}.csv"
    feature_file = Path(lock["feature_cache_path"])
    if s01.sha256(cache_file) != lock["input_detection_sha256"][sequence] or \
            s01.sha256(feature_file) != lock["feature_cache_sha256"]:
        raise RuntimeError("input cache changed")
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
    original = s01.VisualPolicy
    def factory(*values, **kwargs):
        return SelectorPolicy(*values, model=model, risk_multiplier=lock["risk_multiplier"], **kwargs)
    s01.VisualPolicy = factory
    try:
        result = s01.run_condition(
            "M", lock, sequence, info.getint("seqLength"), cached, source_ids,
            feature_map, Path(lock["output_root"]), ocsort_module, assoc_module,
            info.getint("imHeight"), info.getint("imWidth"), info.getfloat("frameRate"))
    finally:
        s01.VisualPolicy = original
    trace = Path(lock["output_root"]) / "traces/M" / f"{sequence}.jsonl"
    changed = total = 0
    for line in trace.read_text().splitlines():
        row = json.loads(line); total += len(row.get("selector_components", []))
        changed += row.get("selector_changed_components", 0)
    check = {"status": "ENGINEERING_PASS", "run_id": lock["run_id"],
             "CUDA_VISIBLE_DEVICES": "", "physical_gpus": [], "GPU0_used": False,
             "selector_components": total, "selector_changed_components": changed,
             "result": result}
    path = Path(lock["output_root"]) / "engineering_checks.json"
    path.write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
