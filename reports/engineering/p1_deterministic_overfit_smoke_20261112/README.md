# P1 deterministic GRU overfit smoke

This is an engineering-only capacity and CUDA-path check. It deliberately trains and evaluates on the same 128 motion-train samples and therefore is not held-out evidence or a tracking result.

## Frozen setup

- Protocol SHA256: `b9c339caf5fdf29a0d0368c6917358ec1be01dcf6b6f2f7517b8372391003aa9`
- Deterministic sample-key subset SHA256: `70a8206a5e9b42492883c23620916858c981f148d756725af8ee4bddca272750`
- Model: one-layer GRU, hidden size 64, four-dimensional normalized delta output.
- Optimization: full 128-sample batch, Adam at 0.003, 800 steps, seed 20261112.
- Frozen acceptance: final normalized MSE at most 0.05, final/initial at most 0.1, finite losses, and zero prediction difference after checkpoint reload.

## Result

- Queue state: PASS; one required job and one passed job.
- Initial normalized MSE: `0.7186536192893982`.
- Final normalized MSE: `0.00010839459719136357`.
- Final/initial: `0.0001508300999006221`.
- Checkpoint reload maximum absolute prediction difference: `0.0`.
- Device record: physical GPU1, `CUDA_VISIBLE_DEVICES=1`, process `cuda:0`, NVIDIA GeForce RTX 4090 D, PyTorch 2.6.0+cu124 / CUDA 12.4.
- GPU0 was never exposed; the post-run snapshot remained 413 MiB and 0% utilization.

Raw queue and checkpoint artifacts are under `/media/imc/data/mot-data/runs/p1_deterministic_overfit_smoke_20261112/`. The checkpoint SHA256 is recorded below but the file is intentionally not stored in Git.

## Interpretation

The result verifies that masking, normalization, optimization, actual CUDA execution, atomic queue artifacts, and checkpoint reload work together, and that this small model can memorize a tiny subset. It says nothing about motion-dev generalization, association decisions, closed-loop identity stability, HOTA/AssA/IDF1, or the research idea's value.
