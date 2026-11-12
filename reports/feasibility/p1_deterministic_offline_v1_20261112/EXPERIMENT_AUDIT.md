# Experiment Audit Report

**Date:** 2026-11-12

**Auditor:** fresh GPT-6.1-Sol ultra, read-only, same-family provisional review

**Requested reviewer note:** the skill named GPT-5.6-Sol, which was unavailable; the available fresh same-family reviewer was used.

**Overall verdict:** WARN

**Integrity status:** warn

## Checks

### A. Ground-truth provenance: PASS with scope qualification

All 347,312 train/dev target states were independently checked against raw DanceTrack GT with maximum absolute error 0; all 40 GT files matched their archived members. Samples are emitted before the current observation is appended, so no current/future detection leakage was found. Histories nevertheless use privileged GT-to-detection identity correspondence and are not tracker-owned histories.

### B. Score normalization: PASS

History and delta statistics are fit on motion-train only. Dev does not refit them. Normalized MSE divides by frozen train target-delta standard deviations, not model prediction statistics. Zero-delta and constant-velocity predictions match their registered definitions.

### C. Result existence and numeric fidelity: PASS

All three seeds have one PASS attempt, return code 0 and four complete artifacts. The reviewer passed 50 source/sample/status/artifact checks and 120 GT/seqinfo/detection hashes. Saved prediction keys and targets cover all 80,458 dev samples. Independent metric recomputation matched every saved metric leaf within `3.33e-16`, and all five gate criteria reproduced.

Running all three checkpoints on CPU differed from saved GPU state predictions by at most `7.71e-5`; the CPU recomputation still passed all five gates. This is consistency evidence, not bitwise cross-device reproducibility.

### D. Dead code: PASS

The formal pipeline executes learned, zero-delta and constant-velocity prediction; overall, per-sequence, gap and detection-presence metrics; saved predictions; and the corrected multi-seed gate. The requested audit-side tests passed.

### E. Scope: WARN

- Fit/dev scope is exactly 32/8 complete sequences and three fixed seeds.
- Four of eight dev sequences are deliberately selected G2 anchors.
- Only samples with at least two GT-matched detector observations and gap at most 31 are represented.
- 75,428/80,458 dev samples have gap 1.
- Gap16-31 IoU is worse than zero delta in every seed, with mean difference `-0.0081100634`, despite improved normalized MSE.
- The GO permits only a controlled natural closed-loop test.

### F. Evaluation type: PASS — real_gt

Classification: real-GT supervised offline prediction with privileged GT-aligned detector histories. It is not synthetic GT or simulation. There are no official HOTA, AssA, IDF1, persistent recovery or closed-loop results in this artifact.

### G. Aggregation addendum: technical PASS, provenance WARN

The original exact comparator equality check reproducibly stops before summary creation. Differences are only `2.22e-16–3.33e-16` IoU float leaves. The corrected `1e-12` absolute tolerance retains exact structures/counts, does not modify learned branches and preserves every original research threshold.

The claims that learned results were not inspected before the addendum and that the parent protocol preceded results are consistent with the records but cannot be independently authenticated from mutable files and self-recorded timestamps. The integrity helper also lacks an explicit non-finite-float rejection; all 1,599 float leaves in this aggregate are finite, so it did not affect this result.

## Supported claims

- The deterministic GRU passes the five preregistered offline feasibility criteria on the frozen eligible motion-dev population against the two registered simple comparators.
- All exact aggregate ratios, IoU difference, seed stability and 8/8 per-sequence MSE wins are supported.
- The narrow decision `GO_TO_NATURAL_CLOSED_LOOP` is supported.

## Unsupported claims

- Tracking, association, identity recovery, HOTA, AssA, IDF1 or deployment improvement.
- Uniform improvement across gap lengths.
- Unbiased confirmation, unseen-dataset generalization or broad robustness.
- Learned uncertainty, Gaussian/MDN/Flow Matching effectiveness, or justification for expanding model complexity.
- Independently authenticated preregistration chronology.

## Action items

1. Preserve the gap16-31 adverse IoU result in all future summaries.
2. Use natural tracker-owned histories and official tracking metrics in the next gate.
3. Add explicit finite-float checks to future aggregation protocols before execution.
4. Do not escalate to probabilistic models unless the deterministic method passes the natural closed-loop gate.

`review_independence: same-family`

`acceptance_status: provisional`
