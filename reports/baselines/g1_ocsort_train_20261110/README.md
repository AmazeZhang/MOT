# G1 OC-SORT DanceTrack train evidence

Recorded: 2026-11-10 (Asia/Shanghai)

Scope: unchanged OC-SORT baseline on all 40 DanceTrack **train** sequences. This is a train diagnostic baseline, not a validation result and not a new-method result.

Files:

- `trackeval_summary.txt`: normalized LF copy of the official TrackEval combined summary.
- `per_sequence_metrics.csv`: compact projection of official per-sequence TrackEval output.
- `artifact_index.csv`: per-sequence cache/tracker counts and SHA256 values.
- `replay_validation.csv`: per-sequence direct-versus-cache-replay validation.
- `source_hashes.yaml`: SHA256 values of the primary external run artifacts.

Large artifacts intentionally excluded from Git:

- 355,231-row fixed-detection caches;
- 331,420-row direct and replay tracker files;
- DanceTrack images/GT;
- detector checkpoint;
- full run directories.

The canonical configuration and limitations are in `manifests/detections/ocsort_yolox_dance_train.yaml`.
