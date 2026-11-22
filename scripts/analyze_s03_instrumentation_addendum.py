#!/usr/bin/env python3
"""Posthoc read-only checks of S03 branch instrumentation and dependence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from mot_research.association.appearance_core import normalize_features
from run_s03_single_memory_write_counterfactual import match_cost

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/s03_instrumentation_addendum_20261122.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise RuntimeError(f"source hash changed: {path}")


def stats(values: list[float]) -> dict:
    return {"count": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S03 addendum requires CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if lock["protocol_sha256"] != sha256(PROTOCOL) or \
            lock["analysis_code_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("addendum code/protocol changed")
    for rel, expected in lock["dependency_code_sha256"].items():
        checked(ROOT / rel, expected)
    if lock["environment_versions"] != {"python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pyyaml": yaml.__version__}:
        raise RuntimeError("addendum dependency versions changed")
    plan = yaml.safe_load(PROTOCOL.read_text())
    inputs = plan["inputs"]
    for name in ("s03_summary", "s03_event_selection", "s03_branches_frozen", "s01_frozen_inputs"):
        checked(Path(inputs[name]), inputs[name + "_sha256"])
    s03 = json.loads(Path(inputs["s03_summary"]).read_text())
    if s03["status"] != "COMPLETE_PRIVILEGED_DIAGNOSTIC":
        raise RuntimeError("S03 parent result incomplete")
    selection = json.loads(Path(inputs["s03_event_selection"]).read_text())["cases"]
    branch_manifest = json.loads(Path(inputs["s03_branches_frozen"]).read_text())
    frozen = json.loads(Path(inputs["s01_frozen_inputs"]).read_text())
    feature = {x["sequence"]: x for x in frozen["feature_artifacts"]}
    trace = {(x["sequence"], x["condition"]): x for x in frozen["artifacts"]
             if x["kind"] == "traces"}
    byseq: dict[str, list[dict]] = defaultdict(list)
    byid = {}
    for case in selection:
        byseq[case["sequence"]].append(case)
        cid = f"{case['kind']}:{case['sequence']}:{case['frame']}:{case['track_id']}"
        if cid in byid:
            raise RuntimeError("duplicate case key")
        byid[cid] = case
    if len(selection) != 172:
        raise RuntimeError("S03 selected-case count changed")
    vector_delta: dict[str, list[float]] = {"wrong": [], "clean": []}
    seq_length = {}
    for seq, cases in byseq.items():
        fp = Path(feature[seq]["path"])
        checked(fp, feature[seq]["sha256"])
        tp = Path(trace[(seq, "M")]["path"])
        checked(tp, trace[(seq, "M")]["sha256"])
        with np.load(fp, allow_pickle=False) as data:
            features = {(int(k[0]), int(k[1])): v.astype(np.float64)
                        for k, v in zip(data["keys"], data["features"])}
        selected = {(case["frame"], case["track_id"]): case for case in cases}
        tids = {case["track_id"] for case in cases}
        memory: dict[int, np.ndarray] = {}
        lines = tp.read_text().splitlines()
        seq_length[seq] = len(lines)
        for line in lines:
            row = json.loads(line)
            frame = int(row["frame"])
            for tid, did, _ in row["final_actual_updates"]:
                if tid not in tids:
                    continue
                value, ok = normalize_features(features[(frame, did)][None, :])
                if not ok[0]:
                    raise RuntimeError("invalid S01 selected feature")
                old = memory.get(tid)
                selected_case = selected.get((frame, tid))
                if selected_case is not None:
                    if old is None:
                        raise RuntimeError("selected memory has no prior feature")
                    new, ok = normalize_features((.9 * old + .1 * value[0])[None, :])
                    if not ok[0]:
                        raise RuntimeError("selected EMA cancelled")
                    vector_delta[selected_case["kind"]].append(float(np.linalg.norm(new[0] - old)))
                if old is not None:
                    value, ok = normalize_features((.9 * old + .1 * value[0])[None, :])
                    if not ok[0]:
                        raise RuntimeError("EMA cancelled")
                memory[tid] = value[0]
    if {k: len(v) for k, v in vector_delta.items()} != {"wrong": 86, "clean": 86} or \
            any(value <= 0 for values in vector_delta.values() for value in values):
        raise RuntimeError("selected write did not change target memory vector")
    trace_counts = {kind: Counter() for kind in ("wrong", "clean")}
    for entry in branch_manifest["artifacts"]:
        path = Path(entry["path"])
        checked(path, entry["sha256"])
        for line in path.read_text().splitlines():
            branch = json.loads(line)
            kind = branch["case"]["kind"]
            count = trace_counts[kind]
            count["cases"] += 1
            if any(a != b for a, b in zip(branch["factual_rows"], branch["suppressed_rows"])):
                count["cases_with_output_change"] += 1
            changed_frames = 0
            for offset, (a, b) in enumerate(zip(branch["factual_traces"], branch["suppressed_traces"])):
                if offset == 0 or a == b:
                    continue
                fa, sb = json.loads(a), json.loads(b)
                fields = {field for field in fa.keys() | sb.keys() if fa.get(field) != sb.get(field)}
                if fields != {"memory_last_write_frames"}:
                    raise RuntimeError(f"unexpected branch trace difference: {fields}")
                count["memory_visible_frames"] += 1
                count["nonfast_memory_visible_frames"] += int(not fa["fast_path"])
                count["memory_visible_frames_with_eligible_edges"] += int(fa["appearance_eligible_edges"] > 0)
                count["eligible_edges_during_memory_visibility"] += int(fa["appearance_eligible_edges"])
                changed_frames += 1
            count["cases_with_memory_visibility"] += int(changed_frames > 0)
    exposure = {}
    for kind in ("wrong", "clean", "all"):
        cases = [x for x in selection if kind == "all" or x["kind"] == kind]
        counts = Counter()
        for case in cases:
            for frame in range(case["frame"] + 1, min(case["frame"] + 30, seq_length[case["sequence"]]) + 1):
                counts[(case["sequence"], frame)] += 1
        exposure[kind] = {"cases": len(cases),
            "distinct_onset_frames": len({(x["sequence"], x["frame"]) for x in cases}),
            "future_frame_exposures": sum(counts.values()),
            "unique_future_video_frames": len(counts),
            "reused_future_video_frames": sum(n > 1 for n in counts.values()),
            "maximum_frame_multiplicity": max(counts.values()),
            "video_end_truncated_cases": sum(x["frame"] + 30 > seq_length[x["sequence"]] for x in cases)}
    wrong = [x for x in selection if x["kind"] == "wrong"]
    control = [x for x in selection if x["kind"] == "clean"]
    if len(control) != len({(x["sequence"], x["frame"], x["track_id"]) for x in control}):
        raise RuntimeError("matched controls reused")
    costs = []
    balances: dict[str, list[float]] = defaultdict(list)
    for case in wrong:
        paired = byid[case["matched_control_id"]]
        if paired["sequence"] != case["sequence"] or \
                any(abs(paired["frame"] - other["frame"]) <= 30
                    for other in wrong if other["sequence"] == case["sequence"]):
            raise RuntimeError("control same-video/exclusion rule broken")
        cost = match_cost(case["covariates"], paired["covariates"])
        if abs(cost - paired["match_cost"]) > 1e-12:
            raise RuntimeError("control match cost changed")
        costs.append(cost)
        for field in case["covariates"]:
            balances[field].append(abs(case["covariates"][field] - paired["covariates"][field]))
    result = {"status": "POSTHOC_INSTRUMENTATION_CHECK_PASS",
        "claim_level": "no_new_research_outcome",
        "protocol_sha256": sha256(PROTOCOL),
        "runtime_lock_sha256": sha256(args.runtime_lock),
        "parent_s03_summary_sha256": sha256(Path(inputs["s03_summary"])),
        "target_memory_vector_delta_L2": {k: stats(v) for k, v in vector_delta.items()},
        "branch_trace_difference": {k: dict(v) for k, v in trace_counts.items()},
        "window_overlap": exposure,
        "control_match_cost": stats(costs),
        "control_absolute_covariate_difference": {k: stats(v) for k, v in balances.items()},
        "limitations": ["posthoc instrumentation checks do not change frozen S03 outcomes",
            "nonzero target memory difference does not imply a changed assignment",
            "matching distance has outliers, so clean controls are imperfectly balanced"]}
    output = Path(plan["outputs"]["root"])
    if output.exists() or str(output) != lock["output_root"]:
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
        "summary_sha256": sha256(output / "summary.json"),
        "target_memory_vector_delta_L2": result["target_memory_vector_delta_L2"],
        "window_overlap": result["window_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
