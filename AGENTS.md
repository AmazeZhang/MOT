# MOT Workspace Rules

## Thesis research objective and anti-rabbit-hole policy

- **Primary Goal:** Deliver a **Master's-thesis-grade MOT research contribution** combining genuine methodological novelty with clear, reproducible closed-loop HOTA/AssA/IDF1 gains over the strongest calibrated baselines—moving from bounded development (`motion_dev8`) to multi-seed confirmation (`13, 37, 73`) and authoritative full benchmarks (DanceTrack `val`/`test`, plus a second benchmark such as SportsMOT or MOT17 when justified), backed by clean ablations and mechanism evidence.
- **Anti-Rabbit-Hole Discipline:** Do not let the project drift into endless incremental micro-tuning or multi-round diagnostic rabbit holes on small subsets (`motion_dev8`, single seed) without a thesis-level hypothesis. Every new experiment must state upfront: (1) which thesis-level innovation/chapter claim it tests, (2) a concrete closed-loop promotion gate over the strongest calibrated baseline (`T2_TH040 = 57.572` on `dev8` unless superseded) required before multi-seed/full-`val` escalation, and (3) a strict **1–2 round stop/pivot boundary** per route so dead ends or plateaus are terminated cleanly.
- **Guardrails Against Refuted Loops:** Do not re-run already-refuted micro-variants (`G1–G4` write-only gates, naive uncalibrated `B1/B2`, synthetic whole-frame swaps `C2/C3`, or `M3` `tau_neg=0.35` margin distortion) without a substantially distinct mechanism; always inspect score-scale / `val/computed_sim_threshold` calibration before assuming a model failed on merit.

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

## Research-side experiment instruction policy

- Follow `docs/experiment_task_contract.md` when proposing or handing off any new experiment. Use `docs/templates/server_experiment_task.md`; the research side owns the scientific choices and must not leave them for the server to guess.
- `docs/a2a/LATEST.md` is the current task entry point. It must distinguish the current task and its execution status from completed or stopped historical tasks. A proposal or a checked-in template is not an executed experiment.
- Deliver a task document plus a versioned protocol before execution. Specify the question and permitted claim, exact conditions and strongest relevant controls, input/split history, initialization, GT boundary, checkpoint rule, execution order, gates, resource limits, and the artifacts required for each stage. Mark engineering acceptance separately from evidence for an innovation.
- Explicitly assign command/configuration implementation to the server when it is not yet implemented locally. Separate verified upstream commands from illustrative commands; the server must record resolved commands/configs and dependency/input locks before running dependent stages. Do not label untested instructions as runnable scripts or a draft as a frozen runtime protocol.
- Predeclare which stages continue automatically after their gates pass. Within the user's authorized scope, routine environment, path, logging, and API compatibility fixes may proceed under the stated repair budget, preserving attempts. Changes to scientific inputs, splits, losses, controls, selection rules, or resource limits require a documented revised task/protocol; do not silently expand a failed experiment or repeatedly ask about already authorized steps.
- Choose controls that test the actual claim, including simple or constant policies and analytic identities when relevant. For learned comparisons, record effective trainable capacity, optimizer steps and sample exposure, not just nominal parameters and epochs. Identify the real reference policy and same-state versus independent-rollout comparisons.
- On return, the research side must reconcile planned versus completed work, recompute accessible metrics, assess compatibility and uncertainty, and state one exact next action. Request missing existing evidence before proposing reruns; bounded diagnostics must have a question they can resolve and a stopping point. Preserve negative results and do not replace failed primary gates with favorable secondary outcomes.
