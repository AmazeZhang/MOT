#!/usr/bin/env python3
"""Read-only CV replay addendum for frozen E1 first-round pair counts."""

from __future__ import annotations

import configparser
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from analyze_p1_common_state_mechanism import (
    read_detection_cache, read_events, read_tracker_rows, write_tracker_row,
)
from mot_research.analysis.d2_first_association import compare_tracker_replay
from mot_research.analysis.runtime_record import (
    peak_rss_kib, runtime_record, sha256_file, verify_code_unchanged,
)
from mot_research.tracking.natural_motion_policy import (
    NaturalHistoryTrackerRunner, NaturalMotionFirstRoundPolicy,
    TrackerDetectionHistory,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/protocols/e1_mature_reemit_factorial_20261119.yaml"
E0_RUN = Path("/media/imc/data/mot-data/runs/e0_reassociation_output_chain_20261119")
E1_RUN = Path("/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119")
OUTPUT = Path("/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119_internal_pairs_addendum")
CODE_PATHS = (
    "scripts/derive_e1_internal_pairs.py",
    "scripts/analyze_p1_common_state_mechanism.py",
    "src/mot_research/analysis/d2_first_association.py",
    "src/mot_research/analysis/runtime_record.py",
    "src/mot_research/tracking/first_round_adapter.py",
    "src/mot_research/tracking/natural_motion_policy.py",
    "src/mot_research/data/motion_samples.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py",
    "third_party/OC_SORT/trackers/ocsort_tracker/association.py",
)


class PairRecorder:
    def __init__(self, base, frame_events, detection_ids, sink):
        self.base = base
        self.history = base.history
        self.frame_events = frame_events
        self.detection_ids = detection_ids
        self.sink = sink
        self.seen = set()
        self.counts = Counter()

    def snapshot(self):
        return {"mode": "CV_with_read_only_pair_recorder", "base": self.base.snapshot(),
                "counts": dict(self.counts)}

    def __call__(self, request, upstream_associate):
        result = self.base(request, upstream_associate)
        events = self.frame_events.get(request.frame_id, [])
        if not events:
            return result
        if request.frame_id in self.seen:
            raise RuntimeError(f"duplicate event frame {request.frame_id}")
        self.seen.add(request.frame_id)
        ids = self.detection_ids.get(request.frame_id, [])
        if len(ids) != len(request.detections):
            raise RuntimeError("detection ID/value mismatch")
        pairs = {(int(ids[int(d)]), int(request.tracker_ids[int(t)]))
                 for d, t in np.asarray(result[0], dtype=np.int64).reshape(-1, 2)}
        owner = {d: t for d, t in pairs}
        for event in events:
            positive = int(event["positive_detection_id"])
            intended = int(event["intended_track_id"])
            if positive not in ids:
                raise RuntimeError(f"{event['event_id']}: positive detection missing")
            correct = (positive, intended) in pairs
            self.sink.append({"event_id": event["event_id"],
                              "sequence": event["sequence"],
                              "frame": request.frame_id,
                              "positive_detection_id": positive,
                              "intended_track_id": intended,
                              "first_round_positive_pair": correct,
                              "positive_detection_first_round_owner": owner.get(positive)})
            self.counts["event_rows"] += 1
            self.counts["first_round_positive_pair"] += int(correct)
        return result


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("internal pair addendum is CPU only")
    protocol = yaml.safe_load(PROTOCOL.read_text())
    e1 = json.loads((E1_RUN / "summary.json").read_text())
    e0 = json.loads((E0_RUN / "summary.json").read_text())
    if e1["status"] != "PASS" or e0["status"] != "PASS":
        raise RuntimeError("source runs incomplete")
    if e1["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("E1 protocol changed")
    event_path = Path(protocol["events"]["path"])
    if sha256_file(event_path) != protocol["events"]["sha256"]:
        raise RuntimeError("event hash mismatch")
    prior_path = Path(protocol["inputs"]["prior_hashes"])
    if sha256_file(prior_path) != protocol["inputs"]["prior_hashes_sha256"]:
        raise RuntimeError("prior hashes changed")
    prior = json.loads(prior_path.read_text())
    historical_cv = json.loads((Path(protocol["inputs"]["cv_tracker_root"]).parent /
                                "summary.json").read_text())
    historical_hashes = {x["sequence"]: x["tracker_output_sha256"]
                         for x in historical_cv["sequences"]}
    oc_commit = subprocess.run(["git", "-C", str(ROOT / protocol["inputs"]["oc_sort_root"]),
                                "rev-parse", "HEAD"], check=True,
                               capture_output=True, text=True).stdout.strip()
    if oc_commit != protocol["inputs"]["oc_sort_commit"]:
        raise RuntimeError("OC-SORT commit mismatch")
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    failed = OUTPUT.with_name(OUTPUT.name + ".failed")
    if any(p.exists() for p in (OUTPUT, partial, failed)):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    partial.mkdir(parents=True)
    start = time.monotonic()
    cpu_start = time.process_time()
    try:
        sys.path.insert(0, str((ROOT / protocol["inputs"]["oc_sort_root"]).resolve()))
        import trackers.ocsort_tracker.ocsort as ocsort_module
        provenance = runtime_record(ROOT, CODE_PATHS)
        provenance.update({"E1_summary_sha256": sha256_file(E1_RUN / "summary.json"),
                           "E0_summary_sha256": sha256_file(E0_RUN / "summary.json"),
                           "protocol_sha256": sha256_file(PROTOCOL),
                           "CUDA_VISIBLE_DEVICES": "", "physical_gpu": None,
                           "started_utc": datetime.now(timezone.utc).isoformat()})
        (partial / "runtime.json").write_text(json.dumps(provenance, indent=2) + "\n")
        events = read_events(event_path)
        by_seq_frame = defaultdict(lambda: defaultdict(list))
        for event in events:
            by_seq_frame[event["sequence"]][int(event["after_frame"])].append(event)
        rows = []
        replay_root = partial / "cv_replay"
        replay_root.mkdir()
        summaries = []
        for sequence in protocol["scope"]["sequences"]:
            seq_dir = Path(protocol["inputs"]["train_root"]) / sequence
            seqinfo = seq_dir / "seqinfo.ini"
            cache = Path(protocol["inputs"]["detection_root"]) / f"{sequence}.csv"
            reference = Path(protocol["inputs"]["cv_tracker_root"]) / f"{sequence}.txt"
            e1_reference = E1_RUN / "trackers" / "C" / f"{sequence}.txt"
            for key, path in (("seqinfo", seqinfo), ("detections", cache)):
                if sha256_file(path) != prior["sequences"][sequence][key]:
                    raise RuntimeError(f"{sequence}: input {key} hash mismatch")
            if sha256_file(reference) != historical_hashes[sequence]:
                raise RuntimeError(f"{sequence}: historical CV hash mismatch")
            config = configparser.ConfigParser()
            if not config.read(seqinfo):
                raise FileNotFoundError(seqinfo)
            info = config["Sequence"]
            length, height, width = (info.getint("seqLength"), info.getint("imHeight"),
                                     info.getint("imWidth"))
            fps = info.getfloat("frameRate")
            scale = min(800 / height, 1440 / width)
            cached, detection_ids = read_detection_cache(cache)
            tracker = ocsort_module.OCSort(**protocol["inputs"]["tracker"])
            history = TrackerDetectionHistory(history_length=protocol["inputs"]["cv_history"]["length"],
                                              score_threshold=.6)
            base = NaturalMotionFirstRoundPolicy(
                mode="constant_velocity", history=history,
                image_width=width, image_height=height, fps=fps,
                minimum_observations=protocol["inputs"]["cv_history"]["minimum_observations"],
                maximum_frame_gap=protocol["inputs"]["cv_history"]["maximum_frame_gap"])
            recorder = PairRecorder(base, by_seq_frame[sequence], detection_ids, rows)
            runner = NaturalHistoryTrackerRunner(tracker, ocsort_module, recorder)
            replay = replay_root / f"{sequence}.txt"
            with replay.open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                for frame in range(1, length + 1):
                    values = np.asarray(cached.get(frame, []), dtype=np.float64)
                    tracker_input = None if values.size == 0 else values.reshape(-1, 5).copy()
                    if tracker_input is not None:
                        tracker_input[:, :4] *= scale
                    tracks = runner.update(tracker_input, (height, width),
                                           (800, 1440), frame_id=frame)
                    for track in tracks:
                        write_tracker_row(writer, frame, track)
            if recorder.seen != set(by_seq_frame[sequence]):
                raise RuntimeError(f"{sequence}: missing event frames")
            rkeys, rboxes = read_tracker_rows(reference)
            keys, boxes = read_tracker_rows(replay)
            validation = compare_tracker_replay(rkeys, rboxes, keys, boxes, 1e-4)
            if validation["status"] != "PASS":
                raise RuntimeError(f"{sequence}: CV replay mismatch")
            if sha256_file(replay) != sha256_file(e1_reference):
                raise RuntimeError(f"{sequence}: CV replay differs from E1 C bytes")
            summary = {"sequence": sequence, "events": recorder.counts["event_rows"],
                       "CV_first_round_positive_pair": recorder.counts["first_round_positive_pair"],
                       "reference_replay": validation, "replay_sha256": sha256_file(replay)}
            summaries.append(summary)
            (partial / "sequence_progress.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"{sequence}: {summary['CV_first_round_positive_pair']} CV positive pairs", flush=True)
        if len(rows) != 831 or len({r["event_id"] for r in rows}) != 831:
            raise RuntimeError("CV pair event coverage mismatch")
        b0_rows = [json.loads(line) for path in sorted((E0_RUN / "event_rows").glob("*.jsonl"))
                   for line in path.open()]
        if len(b0_rows) != 831:
            raise RuntimeError("E0 B0 pair rows incomplete")
        b0_by_id = {r["event_id"]: r for r in b0_rows}
        if set(b0_by_id) != {r["event_id"] for r in rows}:
            raise RuntimeError("B0/CV event IDs differ")
        combined = []
        for row in sorted(rows, key=lambda r: r["event_id"]):
            combined.append({"event_id": row["event_id"], "sequence": row["sequence"],
                             "frame": row["frame"],
                             "B0_first_round_positive_pair": b0_by_id[row["event_id"]]["first_round_positive_pair"],
                             "CV_first_round_positive_pair": row["first_round_positive_pair"],
                             "CV_positive_detection_first_round_owner": row["positive_detection_first_round_owner"]})
        path = partial / "event_internal_pairs.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(combined)
        b0_count = sum(r["B0_first_round_positive_pair"] for r in combined)
        cv_count = sum(r["CV_first_round_positive_pair"] for r in combined)
        if b0_count != e0["reconciled_counts"]["first_round_positive_pair"]:
            raise RuntimeError("B0 first-round count mismatch")
        verify_code_unchanged(ROOT, provenance)
        result = {"status": "PASS", "scope": "secondary E1 internal pair count addendum",
                  "protocol_sha256": provenance["protocol_sha256"],
                  "E1_summary_sha256": provenance["E1_summary_sha256"],
                  "E0_summary_sha256": provenance["E0_summary_sha256"],
                  "event_count": len(combined),
                  "first_round_positive_pair": {"A_and_B": b0_count, "C_and_D": cv_count},
                  "sequences": summaries, "csv_sha256": sha256_file(path),
                  "wall_seconds": time.monotonic() - start,
                  "cpu_core_hours": (time.process_time() - cpu_start) / 3600,
                  "peak_rss_kib": peak_rss_kib(),
                  "main_E1_gate_changed": False,
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        (partial / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        os.replace(partial, OUTPUT)
        print(json.dumps({"status": "PASS", "counts": result["first_round_positive_pair"],
                          "output": str(OUTPUT)}, indent=2), flush=True)
    except Exception as exc:
        (partial / "failure.json").write_text(json.dumps({
            "status": "FAIL", "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        os.replace(partial, failed)
        raise


if __name__ == "__main__":
    main()
