# E3 fit experiment audit

**Date:** 2026-11-19<br>
**Reviewer:** fresh Codex agent, same-family provisional<br>
**Overall verdict:** WARN

| Check | Verdict | Evidence |
|---|---|---|
| A. GT provenance and timing | WARN | TrackEval uses dataset GT. `scripts/run_e3_short_gap_support.py:214-246` computes output rules from current detections and prior outputs, with no GT feature. But line 259 hashes each GT file before all outputs finish, contrary to `manifests/protocols/e3_short_gap_support_20261119.yaml:51`. This is a protocol chronology deviation without observed label leakage. |
| B. Score normalization | PASS | `scripts/analyze_e1_mature_reemit.py:41-52` converts TrackEval proportions into percentage points; no prediction-derived normalization. |
| C. Artifacts, nesting and gate | PASS | Independent checks covered split/protocol/code hashes, 24 GT/cache/seqinfo hashes, 96 tracker hashes and 24 extra JSONL hashes. Every full row satisfies exact `C ⊂ E ⊂ G ⊂ D`; all 2,854 extra rows match saved score/gap/IoU decisions. Official detailed CSVs match four conditions' combined and per-video metrics. |
| D. Metric execution | PASS | `scripts/run_e3_short_gap_support.py:116-124` instantiates and runs HOTA, CLEAR and Identity; official outputs exist. |
| E. Scope and decision | PASS with limits | 24 fit videos, one rule and one run. G−E HOTA +0.049978 point fails ≥+0.100; FP +84 fails ≤+50. AssA +0.003402, IDF1 +0.020441, IDSW +2 and 21/24 HOTA video wins pass their clauses. `fit_gate_pass: false` is correct; heldout directory is absent. |
| F. Evaluation type | real_gt | DanceTrack train GT is passed to TrackEval. This is fit development evidence, not an independent test. |

`status: PASS` in the run summary means execution succeeded; the research gate **failed**. The protocol file timestamp precedes runtime and result files, and the run records its SHA, but uncommitted files cannot independently prove preregistration. The early GT checksum read did not parse GT or affect any output decision; it remains an explicit protocol deviation. Keep the failed gate and do not run E3 heldout or tune the same rule on fit.

**Action for future protocols:** calculate GT checksums after all tracker outputs are complete when the protocol requires no GT-file access during output generation.

**Raw audit trace:** `.aris/traces/experiment-audit/2026-11-19_run02/` (ignored by Git).
