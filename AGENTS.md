# MOT Workspace Rules

## GPU safety policy

- Physical GPU 0 is reserved for the Ubuntu graphical desktop. Never use GPU 0 for this project's training, evaluation, inference, smoke tests, or other CUDA workloads.
- Project CUDA jobs must explicitly exclude physical GPU 0. For a single-GPU job, select one of physical GPUs 1-7 after checking current allocation; for example, use `CUDA_VISIBLE_DEVICES=1`.
- Inside a process started with `CUDA_VISIBLE_DEVICES=1`, `cuda:0` refers to physical GPU 1. Record both the environment variable and the physical GPU selection in run metadata.
- Do not terminate or interfere with unrelated GPU processes. Inspect allocation before starting a job.

## Local storage policy

- Keep the Git repository at `/home/imc/yzy/MOT`.
- Keep large datasets, downloads, weights, caches, and run outputs outside the repository under `/media/imc/data/mot-data` unless the user explicitly changes the location.
- Keep machine-specific environment records under the ignored `local/` directory; do not commit credentials, host secrets, raw datasets, or model weights.

## Completion and handoff policy

- Before finishing each meaningful work stage, update `docs/progress_log.md` with the date, completed work, verification evidence, important artifact paths, known limitations, and the exact next step.
- Keep the progress log factual. Distinguish engineering smoke-test metrics from research results, and never promote planned or partial runs into claims.
- In the user-facing handoff, explicitly explain what was done and what should be done next.
- Do not create Git commits or push changes unless the user explicitly asks. Leave a reviewable working tree so the user can record the work with Git.
