# E2 experiment audit

**Date:** 2026-11-19<br>
**Reviewer:** fresh Codex agent, same-family provisional<br>
**Overall verdict:** WARN

| Check | Verdict | Evidence |
|---|---|---|
| A. GT provenance | PASS | `scripts/run_e2_reemit_guard.py:140-179` uses DanceTrack train GT through TrackEval and fixed-event evaluation; 8 GT files exist. |
| B. Score normalization | PASS | `scripts/analyze_e1_mature_reemit.py:41-52` only converts TrackEval fractions to percentage points. No prediction-derived denominator. |
| C. Result existence and arithmetic | PASS | All E1 source and E2 output hashes match. C's 73,658 rows are preserved; E/F add exactly 920/1,056 filtered rows. Official summaries and risk counts match the report; read-only recovery reevaluation reproduces all 831 event aggregates. |
| D. Metric execution | PASS | `scripts/run_e2_reemit_guard.py:153-214` calls TrackEval, event recovery, proxy labels and utility gate, which appear in the run summary. |
| E. Scope | WARN | Protocol `manifests/protocols/e2_reemit_guard_20261119.yaml:14-15,48-50` admits threshold selection after reading GT-labeled risk on the same 8 reused train-dev videos. No independent confirmation. |
| F. Evaluation type | PASS with qualification | Official metrics are real-GT TrackEval. Fixed-event recovery is a custom dataset-GT diagnostic. Extra-row TP/FP is a custom Hungarian proxy, not official FP. |

The untracked protocol file timestamp precedes the run and its SHA matches the run summary, but this does not independently prove preregistration. E/F passing the exploratory utility gate supports a development-set tradeoff only. D has higher HOTA; E/F still add IDSW and official FP relative to C. No internal-association, novelty, or generalization claim follows.

**Action:** Keep E as a simple baseline. Confirm any future method only after a new frozen design and independent scope; do not tune further on these 8 videos while treating them as validation.

**Raw audit trace:** `.aris/traces/experiment-audit/2026-11-19_run01/` (ignored by Git).
