# Experiment Audit Report

**Date**: 2026-11-10

**Auditor route**: GPT-5.6-Sol ultra, fresh same-family agent

**Project**: MOT / DanceTrack OC-SORT train baseline

## Overall Verdict: REVIEW_UNAVAILABLE

## Integrity Status: unavailable

The independent semantic reviewer hit an account usage limit before returning an audit. Per the `experiment-audit` protocol, this is recorded as unavailable rather than replaced with an executor-authored PASS. The raw request/error trace is retained locally under `.aris/traces/experiment-audit/2026-11-10_run01/` and is ignored by Git.

## Deterministic checks completed

- All 40 expected DanceTrack train sequence summaries exist; their recorded cache and tracker SHA256 values match the files.
- The split summary totals match the 40 per-sequence summaries: 41,796 frames, 355,231 detections, 331,420 tracker rows, and 0 empty-detection frames.
- The final CPU cache replay covers 40 sequences and 331,420 rows; frame/ID pairs are identical and maximum XYXY error is `3.9e-5` pixels under a fixed `1e-4` tolerance.
- The TrackEval summary and detailed result files exist and are hashed in the train detection manifest.
- Two failed `1e-5` replay attempts and the later tolerance probe are retained and identified separately from the final replay validation.

These checks establish artifact consistency only. They do not substitute for the unavailable adversarial semantic review.

## Claim impact

- Full 40-sequence DanceTrack **train** baseline metrics: may be reported as artifact-backed train diagnostics, with the exact scope and evaluator stated.
- Replay consistency: may be reported with the fixed `1e-4` pixel tolerance and the two earlier failed attempts disclosed.
- DanceTrack val performance, generalization, robustness, or new-method improvement: unsupported by this run.

## Action item

- Rerun `experiment-audit` when reviewer capacity is available. Until then, integrity acceptance remains unavailable, not provisional PASS.
