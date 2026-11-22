#!/usr/bin/env python3
"""GT-privileged, 30-frame single-memory-write forks of frozen S01 M."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from analyze_p1_common_state_mechanism import read_detection_cache
from analyze_s02_natural_identity_mechanism import stable_anchor
from mot_research.analysis.failure_events import (
    Candidate, hungarian_iou_matches, pairwise_iou, read_detections, read_gt,
    read_tracker, xywh_to_xyxy,
)
from run_s01_visual_baselines import all_detection_ids, load_features
from run_s03_memory_clone_smoke import MRunner, require_hash, sha256, tracker_equal

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/s03_single_memory_write_counterfactual_20261121.yaml"


def key(case: dict) -> str:
    return f"{case['kind']}:{case['sequence']}:{case['frame']}:{case['track_id']}"


def filehash_bytes(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def case_covariates(frame: int, tid: int, did: int, birth: dict, last: dict,
                    live: list[int], high: list[Candidate], rows: dict[int, np.ndarray]) -> dict:
    selected = rows[did]
    other = [item.box_xyxy for item in high if item.candidate_id != did]
    neighbor = float(pairwise_iou(selected[:4][None, :], np.stack(other)).max()) if other else 0.0
    return {"track_age": frame - birth[tid], "update_gap": frame - last[tid],
        "score": float(selected[4]), "live_count": len(live),
        "high_detection_count": len(high), "max_neighbor_detection_iou": neighbor}


def match_cost(a: dict, b: dict) -> float:
    return (abs(math.log1p(a["track_age"]) - math.log1p(b["track_age"])) +
        abs(math.log1p(a["update_gap"]) - math.log1p(b["update_gap"])) +
        2 * abs(a["score"] - b["score"]) +
        .05 * abs(a["live_count"] - b["live_count"]) +
        .05 * abs(a["high_detection_count"] - b["high_detection_count"]) +
        abs(a["max_neighbor_detection_iou"] - b["max_neighbor_detection_iou"]))


def select_cases(sequence: str, wrong: list[dict], traces: list[dict], gt: dict,
                 detections: dict, source_rows: dict[int, dict[int, np.ndarray]]) -> list[dict]:
    positives, labels = {}, {}
    for frame, trace in enumerate(traces, start=1):
        matches, _ = hungarian_iou_matches(gt.get(frame, {}), detections.get(frame, []), .5)
        positives[frame] = {gid: did for gid, (did, _) in matches.items()}
        labels[frame] = {did: gid for gid, did in positives[frame].items()}
    by_wrong = {(int(e["frame"]), int(e["track_id"])): e for e in wrong}
    history: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    birth: dict[int, int] = {}
    last: dict[int, int] = {}
    wrong_cases, clean_pool = [], []
    for frame, trace in enumerate(traces, start=1):
        live = trace["requested_tracker_ids"]
        anchors = {tid: stable_anchor(history[tid], frame) for tid in live}
        high = detections.get(frame, [])
        for tid, did, stage in trace["final_actual_updates"]:
            birth.setdefault(tid, frame)
            anchor = anchors.get(tid)
            label = labels[frame].get(did)
            if anchor is not None and stage != "birth":
                cov = case_covariates(frame, tid, did, birth, last, live,
                                      high, source_rows[frame])
                wrong_e = by_wrong.get((frame, tid))
                if wrong_e is not None:
                    if label != wrong_e["foreign_gt_id"] or did != wrong_e["selected_det_id"] or \
                            anchor != wrong_e["anchor_gt_id"]:
                        raise RuntimeError("S02 wrong event/source history mismatch")
                    wrong_cases.append({"kind": "wrong", "sequence": sequence,
                        "frame": frame, "track_id": tid, "owner_gt_id": anchor,
                        "selected_det_id": did,
                        "owner_positive_det_id": wrong_e["owner_positive_det_id"],
                        "foreign_gt_id": label, "covariates": cov})
                elif label == anchor and positives[frame].get(anchor) == did and \
                        any(other != tid and owner is not None and owner != anchor
                            for other, owner in anchors.items()):
                    clean_pool.append({"kind": "clean", "sequence": sequence,
                        "frame": frame, "track_id": tid, "owner_gt_id": anchor,
                        "selected_det_id": did, "owner_positive_det_id": did,
                        "foreign_gt_id": None, "covariates": cov})
            history[tid].append((frame, label))
            history[tid] = history[tid][-3:]
            last[tid] = frame
    if len(wrong_cases) != len(wrong):
        raise RuntimeError(f"{sequence}: not all frozen S02 wrong events found")
    wrong_frames = [int(e["frame"]) for e in wrong]
    clean_pool = [c for c in clean_pool if all(abs(c["frame"] - f) > 30 for f in wrong_frames)]
    chosen: set[tuple[int, int]] = set()
    result = []
    for wrong_case in sorted(wrong_cases, key=lambda x: (x["frame"], x["track_id"])):
        available = [c for c in clean_pool if (c["frame"], c["track_id"]) not in chosen]
        wrong_case["matched_control_id"] = None
        result.append(wrong_case)
        if available:
            def order(control):
                tie = hashlib.sha256(
                    f"s03:{sequence}:{control['frame']}:{control['track_id']}".encode("ascii")
                ).hexdigest()
                return (match_cost(wrong_case["covariates"], control["covariates"]), tie)
            selected = min(available, key=order)
            chosen.add((selected["frame"], selected["track_id"]))
            selected["match_cost"] = match_cost(wrong_case["covariates"], selected["covariates"])
            selected["matched_wrong_id"] = key(wrong_case)
            wrong_case["matched_control_id"] = key(selected)
            result.append(selected)
    return result


class MultiCVTap:
    def __init__(self, base, visual, assoc_module, cases: list[dict]):
        self.base, self.visual, self.assoc_module = base, visual, assoc_module
        self.cases_by_frame: dict[int, list[dict]] = defaultdict(list)
        for case in cases:
            self.cases_by_frame[case["frame"]].append(case)
        self.capture = {}

    def __call__(self, request, associate):
        def observed(dets, trks, threshold, velocities, observations, inertia):
            for case in self.cases_by_frame.get(request.frame_id, []):
                dids, tids = self.visual.det_ids, request.tracker_ids
                did, tid = case["owner_positive_det_id"], case["track_id"]
                if did not in dids or tid not in tids:
                    self.capture[key(case)] = {"requested": False, "legal": False,
                        "first_round_iou_after_CV": None}
                    continue
                iou = self.assoc_module.iou_batch(dets, trks)
                value = float(iou[dids.index(did), tids.index(tid)])
                self.capture[key(case)] = {"requested": True, "legal": value >= .3,
                    "first_round_iou_after_CV": value}
            return associate(dets, trks, threshold, velocities, observations, inertia)
        return self.base(request, observed)


def assert_memory_isolation(factual: MRunner, branch: MRunner, frame: int,
                            target: int) -> None:
    if not tracker_equal(factual, branch):
        raise RuntimeError("event-frame tracker/CV state differs")
    if factual.memory._memory[target][0] != frame or branch.memory._memory[target][0] >= frame:
        raise RuntimeError("selected memory write was not suppressed")
    if factual.memory._memory.keys() != branch.memory._memory.keys():
        raise RuntimeError("memory track keys changed")
    for tid, value in factual.memory._memory.items():
        if tid != target and (value[0] != branch.memory._memory[tid][0] or
                not np.array_equal(value[1], branch.memory._memory[tid][1])):
            raise RuntimeError("another track's visual memory changed")


def run_branch(case: dict, snapshot: MRunner, length: int,
               source_rows: dict[int, str], source_traces: list[str]) -> dict:
    factual, suppressed = snapshot.fork(), snapshot.fork()
    first, end = case["frame"], min(length, case["frame"] + 30)
    fact_rows, branch_rows, fact_traces, branch_traces = [], [], [], []
    for frame in range(first, end + 1):
        arow, atrace = factual.step(frame)
        brow, btrace = suppressed.step(frame, case["track_id"] if frame == first else None)
        if arow != source_rows.get(frame, "") or atrace != source_traces[frame - 1]:
            raise RuntimeError(f"factual clone differs from S01: {key(case)}/{frame}")
        if frame == first:
            if arow != brow or atrace != btrace:
                raise RuntimeError(f"event-frame association/output differs: {key(case)}")
            assert_memory_isolation(factual, suppressed, frame, case["track_id"])
        fact_rows.append(arow); branch_rows.append(brow)
        fact_traces.append(atrace); branch_traces.append(btrace)
    return {"case": case, "last_frame": end,
        "factual_rows": fact_rows, "suppressed_rows": branch_rows,
        "factual_traces": fact_traces, "suppressed_traces": branch_traces,
        "future_frames_with_changed_rows": sum(a != b for a, b in zip(fact_rows[1:], branch_rows[1:]))}


def text_candidates(value: str) -> list[Candidate]:
    result = []
    for row in csv.reader(value.splitlines()):
        if len(row) != 10:
            raise RuntimeError("branch MOT output schema changed")
        result.append(Candidate(int(row[1]), xywh_to_xyxy(*map(float, row[2:6]))))
    return result


def score_case(branch: dict, gt: dict, prior_outputs: dict[int, list[Candidate]]) -> dict:
    case = branch["case"]
    first = case["frame"]
    anchors = hungarian_iou_matches(gt.get(first - 1, {}),
                                    prior_outputs.get(first - 1, []), .5)[0]
    anchor_ids = {gid: tid for gid, (tid, _) in anchors.items()}
    target = (case["owner_gt_id"], case["track_id"])
    scores = {"factual": Counter(), "suppressed": Counter()}
    for offset in range(1, len(branch["factual_rows"])):
        frame = first + offset
        frame_gt = gt.get(frame, {})
        for name, rows in (("factual", branch["factual_rows"]),
                           ("suppressed", branch["suppressed_rows"])):
            matches = hungarian_iou_matches(frame_gt, text_candidates(rows[offset]), .5)[0]
            if target[0] in frame_gt and matches.get(target[0], (None,))[0] == target[1]:
                scores[name]["target_correct_frames"] += 1
            for gid, tid in anchor_ids.items():
                if gid in frame_gt and matches.get(gid, (None,))[0] == tid:
                    scores[name][f"anchor:{gid}"] += 1
    delta = {gid: scores["suppressed"][f"anchor:{gid}"] -
            scores["factual"][f"anchor:{gid}"] for gid in anchor_ids}
    return {"case_id": key(case), "kind": case["kind"],
        "sequence": case["sequence"], "frame": first, "track_id": case["track_id"],
        "owner_gt_id": case["owner_gt_id"], "prior_output_anchor_count": len(anchor_ids),
        "owner_in_prior_output_anchors": anchor_ids.get(target[0]) == target[1],
        "target_correct_frame_delta": scores["suppressed"]["target_correct_frames"] -
                                      scores["factual"]["target_correct_frames"],
        "all_prior_anchor_correct_frame_delta": sum(delta.values()),
        "other_prior_anchor_correct_frame_delta": sum(v for gid, v in delta.items() if gid != target[0]),
        "other_prior_anchor_harmed": any(v < 0 for gid, v in delta.items() if gid != target[0]),
        "per_anchor_delta": delta,
        "future_frames_with_changed_rows": branch["future_frames_with_changed_rows"],
        "future_frames_scored": len(branch["factual_rows"]) - 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S03 requires CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if lock["protocol_sha256"] != sha256(PROTOCOL) or \
            lock["analysis_code_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("S03 protocol/analysis code differs from pre-run lock")
    for rel, expected in lock["dependency_code_sha256"].items():
        require_hash(ROOT / rel, expected)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "pyyaml": yaml.__version__}
    if versions != lock["environment_versions"]:
        raise RuntimeError(f"S03 environment changed: {versions}")
    plan = yaml.safe_load(PROTOCOL.read_text())
    inp = plan["inputs"]
    for key_name in ("s01_frozen_inputs", "s02_events", "s02_summary",
                     "smoke_engineering_checks", "s01_eval_runtime_lock", "s01_protocol"):
        source = Path(inp[key_name])
        source = source if source.is_absolute() else ROOT / source
        require_hash(source, inp[f"{key_name}_sha256"])
    smoke = json.loads(Path(inp["smoke_engineering_checks"]).read_text())
    if smoke["status"] != inp["smoke_required_status"] or smoke["GT_read"]:
        raise RuntimeError("S03 clone smoke not accepted")
    frozen = json.loads(Path(inp["s01_frozen_inputs"]).read_text())
    s02 = json.loads(Path(inp["s02_summary"]).read_text())
    events = [json.loads(line) for line in Path(inp["s02_events"]).read_text().splitlines()]
    wrong_events = [e for e in events if e["condition"] == "M" and e["actionable_competition"]]
    if len(wrong_events) != 86 or s02["totals"]["M"]["actionable_competition_first_onsets"] != 86:
        raise RuntimeError("frozen S02 wrong-event population changed")
    if frozen["sequences"] != lock["sequences"] or \
            lock["output_root"] != plan["execution"]["output_root"]:
        raise RuntimeError("S03 scope/output differs from lock")
    gt_hashes = json.loads((ROOT / inp["s01_eval_runtime_lock"]).read_text())["gt_file_sha256"]
    train_root = Path(yaml.safe_load((ROOT / inp["s01_protocol"]).read_text())["inputs"]["train_root"])
    artifacts = {(a["sequence"], a["condition"], a["kind"]): a for a in frozen["artifacts"]}
    features = {a["sequence"]: a for a in frozen["feature_artifacts"]}
    checks = {a["sequence"]: a for a in frozen["engineering_checks"]}
    source: dict[str, dict] = {}
    input_record = {"status": "all_S01_sources_frozen_before_GT_load",
        "protocol_sha256": sha256(PROTOCOL), "runtime_lock_sha256": sha256(args.runtime_lock),
        "s01_frozen_inputs_sha256": sha256(Path(inp["s01_frozen_inputs"])),
        "s02_events_sha256": sha256(Path(inp["s02_events"])),
        "sequences": frozen["sequences"], "artifacts": []}
    for seq in frozen["sequences"]:
        check = checks[seq]
        check_path = Path(check["path"])
        require_hash(check_path, check["sha256"])
        track_lock_path = check_path.parent / "runtime_lock.json"
        require_hash(track_lock_path, check["runtime_lock_sha256"])
        track_lock = json.loads(track_lock_path.read_text())
        for rel, expected in track_lock["code_hashes"].items():
            require_hash(ROOT / rel, expected)
        det_path = Path(track_lock["detection_root"]) / f"{seq}.csv"
        require_hash(det_path, track_lock["input_detection_sha256"][seq])
        feature = Path(features[seq]["path"])
        require_hash(feature, features[seq]["sha256"])
        track_path = Path(artifacts[(seq, "M", "trackers")]["path"])
        trace_path = Path(artifacts[(seq, "M", "traces")]["path"])
        require_hash(track_path, artifacts[(seq, "M", "trackers")]["sha256"])
        require_hash(trace_path, artifacts[(seq, "M", "traces")]["sha256"])
        source[seq] = {"check": check_path, "tracking_lock": track_lock_path,
            "det": det_path, "feature": feature, "tracker": track_path,
            "trace": trace_path}
        for path in (check_path, track_lock_path, det_path, feature, track_path, trace_path):
            input_record["artifacts"].append({"path": str(path), "sha256": sha256(path)})
    output = Path(lock["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "frozen_sources.json").write_text(json.dumps(input_record, indent=2) + "\n")

    # Offline GT is used only to select diagnostic cases and later to score
    # frozen branch outputs. It never enters MRunner.step or the association.
    selection: dict[str, list[dict]] = {}
    contexts: dict[str, dict] = {}
    for seq in frozen["sequences"]:
        gt_path = train_root / seq / "gt/gt.txt"
        require_hash(gt_path, gt_hashes[seq])
        gt = read_gt(gt_path)
        detections = read_detections(source[seq]["det"], .6)
        traces = [json.loads(line) for line in source[seq]["trace"].read_text().splitlines()]
        cached, _ = read_detection_cache(source[seq]["det"])
        dids = all_detection_ids(source[seq]["det"])
        rows = {frame: {did: np.asarray(row, np.float64)
                        for did, row in zip(dids.get(frame, []), cached.get(frame, []))}
                for frame in range(1, len(traces) + 1)}
        wrong = [e for e in wrong_events if e["sequence"] == seq]
        selection[seq] = select_cases(seq, wrong, traces, gt, detections, rows)
        contexts[seq] = {"gt": gt, "cached": cached, "dids": dids,
                         "traces": traces, "rows": rows}
    all_cases = [case for seq in frozen["sequences"] for case in selection[seq]]
    if len([c for c in all_cases if c["kind"] == "wrong"]) != 86:
        raise RuntimeError("selected wrong cases differ from S02 population")
    (output / "frozen_event_selection.json").write_text(json.dumps({
        "status": "selected_before_any_branch_output", "cases": all_cases,
        "wrong_count": 86,
        "clean_count": sum(c["kind"] == "clean" for c in all_cases)}, indent=2) + "\n")

    sys.path.insert(0, str((ROOT / "third_party/OC_SORT").resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module
    branch_artifacts = []
    per_video_engineering = []
    for seq in frozen["sequences"]:
        seq_output = output / seq
        seq_output.mkdir()
        try:
            cfg = configparser.ConfigParser()
            if not cfg.read(train_root / seq / "seqinfo.ini"):
                raise FileNotFoundError(seq)
            info = cfg["Sequence"]
            height, width, fps, length = (info.getint("imHeight"), info.getint("imWidth"),
                info.getfloat("frameRate"), info.getint("seqLength"))
            context = contexts[seq]
            if len(context["traces"]) != length:
                raise RuntimeError("S01 trace/seqinfo length differs")
            source_rows: dict[int, str] = defaultdict(str)
            for line in source[seq]["tracker"].read_text().splitlines(keepends=True):
                source_rows[int(line.split(",", 1)[0])] += line
            source_traces = source[seq]["trace"].read_text().splitlines(keepends=True)
            cases = selection[seq]
            frames = {c["frame"] for c in cases}
            runner = MRunner(seq, context["cached"], context["dids"],
                load_features(source[seq]["feature"]), height, width, fps,
                ocsort_module, assoc_module, None)
            tap = MultiCVTap(runner.visual.cv_policy, runner.visual, assoc_module, cases)
            runner.visual.cv_policy = tap
            snapshots: dict[int, MRunner] = {}
            with (seq_output / "full_replay_tracker.txt").open("w") as track_handle, \
                    (seq_output / "full_replay_trace.jsonl").open("w") as trace_handle:
                for frame in range(1, length + 1):
                    if frame in frames:
                        snapshots[frame] = runner.fork()
                    rows, trace = runner.step(frame)
                    if rows != source_rows.get(frame, "") or trace != source_traces[frame - 1]:
                        raise RuntimeError(f"full M replay differs at {seq}/{frame}")
                    track_handle.write(rows); trace_handle.write(trace)
            require_hash(seq_output / "full_replay_tracker.txt", sha256(source[seq]["tracker"]))
            require_hash(seq_output / "full_replay_trace.jsonl", sha256(source[seq]["trace"]))
            branch_path = seq_output / "branches.jsonl"
            with branch_path.open("w") as handle:
                for case in cases:
                    result = run_branch(case, snapshots[case["frame"]], length,
                                        source_rows, source_traces)
                    result["first_round_legal_screen"] = tap.capture.get(key(case))
                    if result["first_round_legal_screen"] is None:
                        raise RuntimeError(f"first-round request not captured: {key(case)}")
                    handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            branch_artifacts.append({"sequence": seq, "path": str(branch_path),
                                     "sha256": sha256(branch_path), "case_count": len(cases)})
            per_video_engineering.append({"sequence": seq, "status": "ENGINEERING_PASS",
                "frames_replayed": length, "case_count": len(cases),
                "tracker_sha256": sha256(seq_output / "full_replay_tracker.txt"),
                "trace_sha256": sha256(seq_output / "full_replay_trace.jsonl"),
                "branches_sha256": sha256(branch_path)})
            (seq_output / "engineering_checks.json").write_text(json.dumps(
                per_video_engineering[-1], indent=2) + "\n")
        except Exception as exc:
            failure = {"sequence": seq, "status": "ENGINEERING_FAIL",
                       "reason": repr(exc), "traceback": traceback.format_exc()}
            (seq_output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            per_video_engineering.append(failure)
            (output / "partial_engineering.json").write_text(json.dumps(
                per_video_engineering, indent=2) + "\n")
            raise
    (output / "branches_frozen.json").write_text(json.dumps({
        "status": "all_branch_outputs_frozen_before_GT_scoring",
        "frozen_sources_sha256": sha256(output / "frozen_sources.json"),
        "frozen_event_selection_sha256": sha256(output / "frozen_event_selection.json"),
        "artifacts": branch_artifacts}, indent=2) + "\n")

    scores = []
    per_video = []
    for seq in frozen["sequences"]:
        gt = contexts[seq]["gt"]
        prior_outputs = read_tracker(source[seq]["tracker"])
        scored = []
        branch_path = output / seq / "branches.jsonl"
        require_hash(branch_path, next(a["sha256"] for a in branch_artifacts if a["sequence"] == seq))
        for line in branch_path.read_text().splitlines():
            branch = json.loads(line)
            value = score_case(branch, gt, prior_outputs)
            value["first_round_legal_screen"] = branch["first_round_legal_screen"]
            value["matched_control_id"] = branch["case"].get("matched_control_id")
            value["matched_wrong_id"] = branch["case"].get("matched_wrong_id")
            value["match_cost"] = branch["case"].get("match_cost")
            scored.append(value); scores.append(value)
        (output / seq / "scores.jsonl").write_text("".join(
            json.dumps(x, separators=(",", ":")) + "\n" for x in scored))
        groups = {}
        for kind in ("wrong", "clean"):
            items = [x for x in scored if x["kind"] == kind]
            groups[kind] = {"cases": len(items),
                "legal_owner_detection_edges": sum(bool(x["first_round_legal_screen"]["legal"]) for x in items),
                "all_prior_anchor_correct_frame_delta": sum(x["all_prior_anchor_correct_frame_delta"] for x in items),
                "target_correct_frame_delta": sum(x["target_correct_frame_delta"] for x in items),
                "other_prior_anchor_correct_frame_delta": sum(x["other_prior_anchor_correct_frame_delta"] for x in items),
                "other_prior_anchor_harmed_cases": sum(x["other_prior_anchor_harmed"] for x in items),
                "cases_with_any_output_change": sum(x["future_frames_with_changed_rows"] > 0 for x in items)}
        per_video.append({"sequence": seq, "groups": groups})
    (output / "scores.jsonl").write_text("".join(
        json.dumps(x, separators=(",", ":")) + "\n" for x in scores))
    (output / "per_video.json").write_text(json.dumps(per_video, indent=2) + "\n")
    totals = {}
    for kind in ("wrong", "clean"):
        totals[kind] = {field: sum(row["groups"][kind][field] for row in per_video)
                        for field in per_video[0]["groups"][kind]}
    legal_videos = sum(row["groups"]["wrong"]["legal_owner_detection_edges"] > 0 for row in per_video)
    pair_fraction = totals["clean"]["cases"] / totals["wrong"]["cases"]
    decision = plan["decision"]
    exposure_ok = (pair_fraction >= .8 and totals["wrong"]["legal_owner_detection_edges"] >= 20
                   and legal_videos >= 6)
    summary = {"status": "COMPLETE_PRIVILEGED_DIAGNOSTIC", "scope": "24_reused_train_fit_videos",
        "claim_level": plan["claim_level"], "protocol_sha256": sha256(PROTOCOL),
        "runtime_lock_sha256": sha256(args.runtime_lock), "sequences": frozen["sequences"],
        "frozen_sources_sha256": sha256(output / "frozen_sources.json"),
        "frozen_event_selection_sha256": sha256(output / "frozen_event_selection.json"),
        "branches_frozen_sha256": sha256(output / "branches_frozen.json"),
        "scores_sha256": sha256(output / "scores.jsonl"),
        "per_video_sha256": sha256(output / "per_video.json"),
        "totals": totals, "wrong_legal_videos": legal_videos,
        "control_pair_fraction": pair_fraction,
        "decision": {"exposure_gate_met": exposure_ok,
                     "next": "return_for_integrity_and_claim_review_no_auto_training" if exposure_ok else
                             "report_sparse_or_unbalanced_no_training"},
        "limitations": ["GT-selected memory suppression is privileged and not deployable",
            "local overlapping 30-frame branches are not complete-sequence TrackEval",
            "prior-output anchor coverage may omit un-emitted tracks",
            "S02 identity labels use IoU-0.5 one-to-one proxy",
            "reused train-fit, possible appearance-pretraining overlap, no independent generalization"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "totals": totals,
                      "decision": summary["decision"],
                      "summary_sha256": sha256(output / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
