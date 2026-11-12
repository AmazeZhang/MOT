# Asynchronous GPU queue engineering smoke

This is an engineering-only failure-isolation test. It does not allocate CUDA tensors and contains no research metric.

Three independent jobs were launched asynchronously with explicit physical GPU labels:

- `pass_gpu1`: PASS; finalized to `artifacts/`.
- `expected_fail_gpu2`: intentional non-zero exit; logs and partial evidence preserved in `artifacts.failed/`.
- `pass_gpu3`: PASS despite the independent GPU2 failure; finalized to `artifacts/`.

The aggregate contains only the two PASS attempts and lists the GPU2 job under `missing_or_failed_jobs`. Each attempt records its command, Git state, physical GPU, `CUDA_VISIBLE_DEVICES`, process-local `cuda:0`, worker/child PIDs, timestamps, logs and artifact hashes. Retry creates a new numbered attempt and never overwrites an old attempt.

Safety properties covered by tests and smoke:

- Physical GPU0 is rejected by manifest validation.
- Jobs cannot override queue-controlled `CUDA_VISIBLE_DEVICES`.
- A concurrent batch cannot assign the same physical GPU twice.
- Selected GPUs are checked for existing compute processes before launch.
- Artifacts use `artifacts.partial` then atomic rename; failed artifacts remain separate.
- Aggregation selects the newest PASS attempt per job and never reads failed artifacts.

Raw run: `/media/imc/data/mot-data/runs/async_queue_smoke_20261112/`.

