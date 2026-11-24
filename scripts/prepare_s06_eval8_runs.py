#!/usr/bin/env python3
"""Freeze the one-time S06 seed-13 evaluation8 V and learned run locks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL8 = ["dancetrack0074", "dancetrack0032", "dancetrack0069", "dancetrack0015",
         "dancetrack0023", "dancetrack0080", "dancetrack0049", "dancetrack0083"]
COMMON = ["scripts/run_s01_visual_baselines.py", "scripts/run_e1_mature_reemit_factorial.py",
          "scripts/analyze_p1_common_state_mechanism.py", "src/mot_research/analysis/runtime_record.py",
          "src/mot_research/association/appearance_core.py",
          "src/mot_research/tracking/first_round_adapter.py",
          "src/mot_research/tracking/natural_motion_policy.py",
          "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
          "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py"]
LEARNED = ["scripts/run_s06_selector_closed_loop.py",
           "src/mot_research/models/component_action_selector.py",
           "src/mot_research/analysis/component_action_features.py",
           "src/mot_research/analysis/multi_assignment_components.py", *COMMON]
FIXED = ["scripts/run_s06_fixed_v.py", *COMMON]


def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest(); return h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--s05a-lock-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    if sha(args.protocol) != args.protocol_sha256 or args.output_root.exists():
        raise RuntimeError("protocol changed or output exists")
    selection = json.loads(args.selection.read_text())
    if selection["evaluation8_used"] or set(selection["selected"]) != {"Local", "Context"}:
        raise RuntimeError("checkpoint selection is not frozen")
    args.output_root.mkdir(parents=True)
    jobs = []
    for sequence in EVAL8:
        source_path = args.s05a_lock_root / sequence / "runtime_lock.json"
        source = json.loads(source_path.read_text())
        base = {key: source[key] for key in ("train_root", "detection_root",
                "input_detection_sha256", "feature_cache_path", "feature_cache_sha256")}
        v_out = args.output_root / "V" / sequence; v_out.mkdir(parents=True)
        v_lock = {**base, "run_id": f"s06_eval8_V_{sequence}_retry3_attempt1",
                  "status": "locked_before_one_time_evaluation8_tracking",
                  "selected_sequences": [sequence], "condition": "V",
                  "protocol_path": str(args.protocol), "protocol_sha256": args.protocol_sha256,
                  "code_hashes": {path: sha(ROOT / path) for path in FIXED},
                  "output_root": str(v_out), "CUDA_VISIBLE_DEVICES": "",
                  "physical_gpus": [], "GPU0_used": False}
        (v_out / "runtime_lock.json").write_text(json.dumps(v_lock, indent=2) + "\n")
        jobs.append(str(v_out / "runtime_lock.json"))
        for mode in ("Local", "Context"):
            chosen = selection["selected"][mode]
            for policy, multiplier in (("no_extra_risk", 1.0), ("risk", 2.0)):
                condition = f"{mode}_{policy}"
                out = args.output_root / condition / sequence; out.mkdir(parents=True)
                lock = {**base,
                    "run_id": f"s06_eval8_{condition}_{sequence}_seed13_retry3_attempt1",
                    "status": "locked_before_one_time_evaluation8_tracking",
                    "selected_sequences": [sequence], "model": mode,
                    "epoch": chosen["epoch"], "policy": policy, "risk_multiplier": multiplier,
                    "checkpoint_path": chosen["checkpoint"],
                    "checkpoint_sha256": chosen["checkpoint_sha256"],
                    "selection_summary": str(args.selection), "selection_summary_sha256": sha(args.selection),
                    "protocol_path": str(args.protocol), "protocol_sha256": args.protocol_sha256,
                    "code_hashes": {path: sha(ROOT / path) for path in LEARNED},
                    "output_root": str(out), "CUDA_VISIBLE_DEVICES": "",
                    "physical_gpus": [], "GPU0_used": False}
                (out / "runtime_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
                jobs.append(str(out / "runtime_lock.json"))
    manifest = {"status": "LOCKED_BEFORE_EVALUATION8_TRACKING", "jobs": jobs,
                "conditions": ["V", "Local_no_extra_risk", "Local_risk",
                               "Context_no_extra_risk", "Context_risk"],
                "sequences": EVAL8, "selection_sha256": sha(args.selection),
                "protocol_sha256": args.protocol_sha256}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "LOCKED", "jobs": len(jobs)}, sort_keys=True))


if __name__ == "__main__":
    main()
