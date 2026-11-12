# G1 OC-SORT DanceTrack val evidence

Recorded: 2026-11-11 (Asia/Shanghai)

Scope: unchanged OC-SORT baseline on all 25 DanceTrack **val** sequences under a protocol frozen before metric inspection. This is the G1 reference validation baseline, not a new-method result. Val results must not be used for method or hyperparameter selection.

Files:

- `trackeval_summary.txt`: normalized LF copy of the official TrackEval combined summary.
- `per_sequence_metrics.csv`: compact projection of official per-sequence TrackEval output.
- `artifact_index.csv`: per-sequence cache/tracker counts and SHA256 values.
- `replay_validation.csv`: per-sequence direct-versus-cache-replay validation.
- `source_hashes.yaml`: SHA256 values of the primary external run artifacts and frozen protocol/config.

Large artifacts intentionally excluded from Git:

- 238,084-row fixed-detection caches;
- 202,711-row direct and replay tracker files;
- DanceTrack images/GT;
- detector checkpoint;
- full run directories.

The frozen protocol is `manifests/protocols/g1_dancetrack_val_freeze_20261111.yaml`; canonical configuration and limitations are in `manifests/detections/ocsort_yolox_dance_val.yaml`.
