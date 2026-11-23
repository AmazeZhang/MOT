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

## Experiment reporting and Git handoff policy

- Follow `docs/experiment_reporting.md` for every new experiment, including negative, partial, failed, diagnostic, and engineering runs. Start from `docs/templates/report_bundle/`; its `TEMPLATE` status is not a result.
- Before a research run, freeze the question, controls, exact sequence lists, primary metric, selection/stopping rules, seeds, and budget in a versioned protocol. Record subsequent deviations and whether results had already been inspected. A file hash alone does not prove preregistration.
- Give each attempt an immutable run ID. Never overwrite failed attempts or silently replace an evaluation. Link retries, reused artifacts, and the authoritative evaluation explicitly.
- A completed tracking comparison must return Git-readable full-precision aggregate and per-sequence metrics for every condition/seed, with expected-versus-completed coverage. Server-only paths, hashes, screenshots, or rounded Markdown tables are insufficient.
- Reports must distinguish official TrackEval aggregate metrics, per-video descriptive summaries, local/oracle diagnostics, and engineering checks. Record metric units, denominators, data reuse, weight training-data overlap, GT usage, and unavailable evidence explicitly.
- Return reproducible success/failure/unchanged case samples and compact decision evidence when claiming an association mechanism; include learning curves and checkpoint-selection records when training. Report all planned seeds, failures, collateral identity harm, and compute costs, not only the best result.
- Keep large data, weights, trajectories, and logs on the data disk. Commit-ready report bundles contain only small summaries, configurations, safe evidence excerpts, and artifact hashes. Record code commit plus relevant dirty-source contents/hashes so uncommitted runs are reproducible; never export secrets.
- Before handoff, check metric export against source results, group/sequence/seed completeness, comparison compatibility, and report-file hashes. State exactly which checks were executed and which were not; a report format check is not an independent experiment reproduction.
- Update `reports/README.md`, `docs/progress_log.md`, and `docs/a2a/LATEST.md` after each meaningful experimental handoff. The latest handoff must name the authoritative report, decision, limitations, and exact next action. Preparing these files does not authorize committing or pushing.
