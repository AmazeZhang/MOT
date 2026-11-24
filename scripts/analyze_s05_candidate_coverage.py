#!/usr/bin/env python3
"""Offline GT labeling and coverage gate for frozen S05-B candidate traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from mot_research.analysis.failure_events import hungarian_iou_matches, read_detections, read_gt


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_anchor(history: list[tuple[int, int | None]], frame: int) -> int | None:
    if len(history) < 3 or frame - history[-1][0] > 30:
        return None
    labels = [label for _, label in history[-3:]]
    return labels[0] if labels[0] is not None and labels.count(labels[0]) == 3 else None


def label_action(edges, trackers, detections, anchors, detection_labels, m_correct):
    mapping = {int(t): int(d) for d, t in edges}
    correct = {}
    for tracker_row in trackers:
        owner = anchors[tracker_row]
        det_row = mapping.get(tracker_row)
        correct[tracker_row] = det_row is not None and detection_labels.get(detections[det_row]) == owner
    g = sum(not m_correct[t] and correct[t] for t in trackers)
    h = sum(m_correct[t] and not correct[t] for t in trackers)
    return {"g": g, "h": h, "g_minus_h": g - h, "correct_by_tracker_row": correct}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-lock", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("S05-B analysis must use CUDA_VISIBLE_DEVICES=''")
    lock = json.loads(args.runtime_lock.read_text())
    if sha256(ROOT / lock["protocol_path"]) != lock["protocol_sha256"] or \
            sha256(Path(__file__)) != lock["analysis_code_sha256"]:
        raise RuntimeError("protocol or S05-B analysis source changed")
    for path, expected in lock["dependency_code_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"analysis dependency changed: {path}")
    for item in lock["inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen analysis input changed: {item['path']}")
    output = Path(lock["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    frozen = {"status": "candidate_traces_frozen_before_GT_read",
              "protocol_sha256": lock["protocol_sha256"],
              "runtime_lock_sha256": sha256(args.runtime_lock),
              "inputs": lock["inputs"]}
    (output / "frozen_inputs.json").write_text(json.dumps(frozen, indent=2) + "\n")

    totals = Counter()
    reason_totals = Counter()
    per_video = []
    profitable_videos = set()
    unique_profitable_frames = set()
    unique_profitable_tracks = set()
    with (output / "components.jsonl").open("w") as component_handle:
        for sequence in lock["sequences"]:
            seq = lock["per_sequence"][sequence]
            gt_path = Path(lock["dataset_root"]) / sequence / "gt/gt.txt"
            if sha256(gt_path) != lock["gt_file_sha256"][sequence]:
                raise RuntimeError(f"GT changed: {sequence}")
            gt = read_gt(gt_path)
            high_detections = read_detections(Path(seq["detection_path"]), .6)
            all_detections = read_detections(Path(seq["detection_path"]), -1e30)
            trace_path = Path(seq["trace_path"])
            traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
            histories: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
            video = Counter()
            video_reasons = Counter()
            for frame, trace in enumerate(traces, start=1):
                if trace["frame"] != frame or trace["sequence"] != sequence:
                    raise RuntimeError(f"trace order changed: {sequence}/{frame}")
                high_match, _ = hungarian_iou_matches(gt.get(frame, {}), high_detections.get(frame, []), .5)
                all_match, _ = hungarian_iou_matches(gt.get(frame, {}), all_detections.get(frame, []), .5)
                high_positive = {gid: did for gid, (did, _) in high_match.items()}
                all_positive = {gid: did for gid, (did, _) in all_match.items()}
                detection_labels = {did: gid for gid, did in high_positive.items()}
                requested_tids = trace.get("requested_tracker_ids", [])
                requested_dids = trace.get("requested_det_ids", [])
                anchors_by_row = {row: stable_anchor(histories[tid], frame)
                                  for row, tid in enumerate(requested_tids)}
                totals["frames"] += 1
                video["frames"] += 1
                if trace.get("fast_path"):
                    totals["fast_path_frames"] += 1
                    video["fast_path_frames"] += 1
                parts = trace.get("components", [])
                if parts:
                    totals["candidate_difference_frames"] += 1
                    video["candidate_difference_frames"] += 1
                for index, part in enumerate(parts):
                    trackers = [int(value) for value in part["tracker_rows"]]
                    actions = {name: [tuple(edge) for edge in edges]
                               for name, edges in part["actions"].items()}
                    component_id = f"{sequence}:{frame}:{index}"
                    totals["component_frames"] += 1
                    video["component_frames"] += 1
                    fully_labelable = all(anchors_by_row[row] is not None for row in trackers)
                    record = {"component_id": component_id, "sequence": sequence,
                              "frame": frame, "tracker_ids": [requested_tids[row] for row in trackers],
                              "detection_ids": [requested_dids[row] for row in part["detection_rows"]],
                              "unique_actions": part["unique_actions"],
                              "fully_labelable": fully_labelable, "actions": {}, "reasons": []}
                    if not fully_labelable:
                        record["reasons"].append("unlabelable")
                        totals["unlabelable_component_frames"] += 1
                        video["unlabelable_component_frames"] += 1
                    else:
                        totals["fully_labelable_component_frames"] += 1
                        video["fully_labelable_component_frames"] += 1
                        m_map = {int(t): int(d) for d, t in actions["M"]}
                        m_correct = {row: row in m_map and
                            detection_labels.get(requested_dids[m_map[row]]) == anchors_by_row[row]
                            for row in trackers}
                        for name in ("M", "G", "V"):
                            record["actions"][name] = label_action(
                                actions[name], trackers, requested_dids, anchors_by_row,
                                detection_labels, m_correct)
                        alternatives = [name for name in part["unique_actions"] if name != "M"]
                        profitable = [name for name in alternatives
                                      if record["actions"][name]["g_minus_h"] > 0]
                        m_wrong_rows = [row for row in trackers if not m_correct[row]]
                        legal = {(int(did), int(tid)) for did, tid, _ in trace.get("legal_edges", [])}
                        final_updates = {int(tid): (int(did), stage)
                                         for tid, did, stage in trace.get("final_actual_updates", [])}
                        flags = set()
                        for row in m_wrong_rows:
                            owner, tid = anchors_by_row[row], requested_tids[row]
                            if owner not in all_positive:
                                flags.add("positive_detection_absent")
                            elif owner not in high_positive:
                                flags.add("only_low_score_or_later_round")
                            else:
                                owner_did = high_positive[owner]
                                if (owner_did, tid) not in legal:
                                    flags.add("IoU_gate_excluded")
                            final = final_updates.get(tid)
                            if final is not None and final[1] == "later" and detection_labels.get(final[0]) == owner:
                                flags.add("only_low_score_or_later_round")
                        if m_wrong_rows and not any(record["actions"][name]["g"] > 0 for name in alternatives):
                            flags.add("no_correct_action_in_M_G_V")
                        if profitable:
                            flags.add("recoverable_M_wrong")
                            totals["profitable_component_frames"] += 1
                            video["profitable_component_frames"] += 1
                            profitable_videos.add(sequence)
                            unique_profitable_frames.add((sequence, frame))
                            unique_profitable_tracks.update((sequence, requested_tids[row]) for row in trackers)
                        if any(record["actions"][name]["h"] > 0 for name in alternatives):
                            flags.add("M_correct_and_alternative_harms")
                        record["profitable_actions"] = profitable
                        record["reasons"].extend(sorted(flags))
                    for reason in record["reasons"]:
                        reason_totals[reason] += 1
                        video_reasons[reason] += 1
                    component_handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                for tid, did, _stage in trace.get("final_actual_updates", []):
                    histories[int(tid)].append((frame, detection_labels.get(int(did))))
                    histories[int(tid)] = histories[int(tid)][-3:]
            per_video.append({"sequence": sequence, "counts": dict(video),
                              "reason_counts": dict(video_reasons)})
            totals.update({f"videos_with_{key}": int(value > 0)
                           for key, value in video.items() if key != "frames"})
    gate_pass = totals["profitable_component_frames"] >= 40 and len(profitable_videos) >= 4
    decision = "CONTINUE_TO_SEPARATELY_FROZEN_S06" if gate_pass else \
        "INCONCLUSIVE_STOP_CURRENT_SELECTOR"
    summary = {
        "status": "COMPLETED",
        "decision": decision,
        "coverage_gate": {"required_component_frames": 40, "required_unique_videos": 4,
                          "observed_component_frames": totals["profitable_component_frames"],
                          "observed_unique_frames": len(unique_profitable_frames),
                          "observed_unique_videos": len(profitable_videos),
                          "observed_unique_video_track_ids": len(unique_profitable_tracks),
                          "passed": gate_pass},
        "counts": dict(totals),
        "multi_label_reason_counts": dict(reason_totals),
        "per_video": per_video,
        "protocol_sha256": lock["protocol_sha256"],
        "runtime_lock_sha256": sha256(args.runtime_lock),
        "frozen_inputs_sha256": sha256(output / "frozen_inputs.json"),
        "limitations": ["GT IoU-0.5 owner labels are offline diagnostic proxies",
                        "fit24 are reused development sequences",
                        "coverage does not prove learnability or closed-loop gain"],
    }
    (output / "per_video.json").write_text(json.dumps(per_video, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "decision": decision,
                      "coverage_gate": summary["coverage_gate"],
                      "reason_counts": summary["multi_label_reason_counts"]}, indent=2))


if __name__ == "__main__":
    main()
