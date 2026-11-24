# S06 component selector seed 13

Status: **FAILED_ENGINEERING / NO_SCIENTIFIC_DECISION**. No S06 training, checkpoint, learned closed-loop run, or TrackEval result exists.

The executable protocol was frozen before S06-specific extraction at `manifests/protocols/s06_component_selector_seed13_20261123.yaml`, SHA256 `290a95e3e36e96dbc62cd20e7661a6b1c80b1627a2849586cb37c412f2afdf51`. It allowed seed 13 only, one disclosed engineering repair, and no automatic seed 37/73 execution.

## Attempts

1. `feature_instrumentation_attempt1/dancetrack0027` stopped on its first no-tracker frame before producing a completed trace or tracker result. The causal direction-feature helper required a two-dimensional velocity/observation shape, while upstream uses empty arrays at tracker count zero.
2. Attempt 2 used a new immutable root after adding an empty-velocity test. It stopped at the same first no-tracker boundary because upstream OC-SORT also represents empty `k_observations` as shape `(0,)`, not `(0,5)`. This consumed the protocol's one repair allowance.

The two directories are immutable and distinct, but the preparer hard-coded the internal `run_id` suffix `attempt1`; attempt 2 therefore has a duplicate internal run ID. No outputs were overwritten, and this metadata defect is retained rather than silently edited. It is an additional requirement for any future retry preparer.

Read-only source inspection confirmed the construction in `third_party/OC_SORT/trackers/ocsort_tracker/ocsort.py`: both arrays are built from the empty tracker list and therefore become one-dimensional. The helper and regression test have since been corrected in the working tree, but that post-failure change was **not run as attempt 3** and is not S06 evidence. A future run requires a new protocol/version and renewed authorization; it must not silently reuse the stopped protocol.

Both attempts used `CUDA_VISIBLE_DEVICES=''`. Physical GPU0 and all other GPUs were unused. The only outputs are runtime locks, source snapshots/dirty patches, and partial empty files created before the exception. They are retained under `/media/imc/data/mot-data/runs/s06_component_selector_seed13_20261123/`.

Decision: do not interpret this as evidence for or against Local, Context, the risk policy, or the frozen metric gate. Seeds 13, 37 and 73 are all `NOT_RUN` scientifically. The immediate next choice is whether to authorize a versioned S06 retry protocol containing the now-tested empty-tracker fix; no further run is authorized by this report.
