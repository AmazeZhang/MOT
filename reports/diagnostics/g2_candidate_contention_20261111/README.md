# G2 candidate-contention addendum

Recorded: 2026-11-11 (Asia/Shanghai)

Scope: post-extraction exploratory decomposition of all 797 frozen G2 **train-only** events. No val data or metrics were read.

Each parent gap frame is classified as:

- `no_raw_candidate`: no score > 0.6 raw detection reaches IoU 0.5 with the target GT;
- `candidate_contention`: at least one such raw candidate exists, but the target is unmatched after the frozen one-to-one Hungarian assignment.

This uses GT and is diagnostic only. A contending candidate is not proof of correct detector identity, and neither class is a causal failure label available to an online tracker.

Results:

- Events: 385 contention-only, 305 absence-only, 107 mixed.
- Gap frames: 1,291 contention and 1,076 raw-candidate absence.
- Identity-change events containing contention: 45/53 (84.9%).
- Identity-preserving controls containing contention: 97/112 (86.6%).
- Because contention is similarly common in both outcomes, its presence alone does not explain identity change.

Files:

- `summary.json`: exact external-run summary.
- `sampled_contention.csv`: decomposition for the fixed 20+20 visual-review sample.
- `outcome_summary.csv`: event-class counts by parent outcome.
- `source_hashes.yaml`: hashes of the frozen protocol, parent events, and complete external result table.

The complete 797-row table remains on the data disk at `/media/imc/data/mot-data/runs/g2_candidate_contention_20261111/event_candidate_contention.csv`.
