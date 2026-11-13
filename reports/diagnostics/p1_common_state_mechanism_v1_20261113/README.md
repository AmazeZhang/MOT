# P1 common-state mechanism diagnosis

Decision: **stop/pivot the learned-motion branch**. None of the four frozen aggregate learned-versus-constant-velocity disadvantage checks fired, so the diagnostic did not identify the mechanism required to authorize a correction.

## What was tested

- Fixed protocol SHA256: `48e431925f92212583e3182b0498122f17364b5855be0ec367f93329df11622a`.
- All 831 previously frozen recovery events on the same eight motion-dev sequences.
- B0/KF, zero delta, constant velocity and all three learned checkpoints.
- At each registered reappearance frame, every method received the identical B0 live tracks, natural detection history, detections, direction inputs, gate and assignment backend.
- Alternative results were one-step counterfactuals only. B0 alone updated tracker/KF/history; no alternative pair or box was written back.
- The complete output is 831 × 6 = 4,986 rows. Two events retain a dead intended B0 track and are non-applicable for every method; the remaining 829 are common and eligible.

This is a mechanism diagnostic, not a new natural closed-loop method run and not official TrackEval.

## Common-state results

| Method | Target IoU mean | Center error mean | Gate eligible /829 | Immediate positive /829 |
| --- | ---: | ---: | ---: | ---: |
| B0/KF | **0.625057** | **0.015774** | 771 | **575** |
| Zero delta | 0.608120 | 0.016791 | 771 | 551 |
| Constant velocity | 0.522368 | 0.022220 | 692 | 509 |
| Learned 20261112 | 0.617239 | 0.016212 | 771 | 555 |
| Learned 20261113 | 0.617172 | 0.016217 | **774** | 562 |
| Learned 20261114 | 0.615925 | 0.016266 | 768 | 559 |

On identical B0 state, learned is not worse than constant velocity on the registered aggregate stages. All three learned seeds have higher target IoU than CV on 601/829 events; at least two do on 628/829. All three learned seeds immediately match while CV misses on 81 events, while CV matches and all three learned miss on 28.

## Frozen checks

The signed value is defined as learned disadvantage relative to CV. Negative values favor learned.

| Stage | Observed disadvantage | Trigger threshold | Disadvantaged seeds | Triggered |
| --- | ---: | ---: | ---: | --- |
| Geometry IoU | -0.094411 | +0.010 | 0 | no |
| Center error | -0.005988 | +0.005 | 0 | no |
| Gate rate | -0.095296 | +0.020 | 0 | no |
| Immediate assignment rate | -0.059912 | +0.020 | 0 | no |

No stage triggered. The prospective rule therefore allows zero deterministic corrections and stops/pivots this learned-motion branch. Thresholds were not changed after reveal.

## Relation to the negative natural closed loop

The earlier natural closed-loop result remains:

- CV HOTA/AssA: `51.056654/32.623675`;
- learned mean HOTA/AssA: `49.270966/30.541388`;
- CV recovery/harm: `23/20`;
- learned recovery/harm: `19/21`, `17/20`, `17/19`.

The common-state result does not overturn those numbers. It shows an inversion: learned is better than CV on the registered one-step B0-state measurements but worse once each method evolves its own closed-loop state and history.

This inversion does **not** prove a feedback/history mechanism. The diagnostic does not replay method-specific evolving histories or identify causal mediation. Immediate first-round positive also differs sharply from reconnect/persistence: B0 has 575 immediate positives but only 158 of them reconnect, and one reconnect occurs without an immediate first-round positive.

## Integrity and claim gate

- Fresh same-family audit: provisional `WARN` for selected development scope, one-step boundary, 17 long-gap events, descriptive outcome linkage and mutable chronology.
- The audit independently replayed 7,565 frames / 73,479 tracker rows and all 4,986 rows / 189,468 CSV cells with zero differences. It found no fake GT, self-normalization, phantom result, dropped event or writeback.
- Result-to-claim: `claim_supported=no`, `route=pivot`, `confidence=high`.
- Currently justified deterministic corrections: `0`.
- Gaussian, MDN, Flow Matching and official-val confirmation: not authorized.

The exact defensible claim is:

> On 831 fixed recovery events from eight selected DanceTrack motion-dev sequences, none of four registered aggregate learned-versus-CV disadvantage checks fired on the 829 common applicable B0-state events. Learned was stronger on these one-step aggregate measurements despite weaker prior natural closed-loop tracking. The diagnostic therefore did not explain the closed-loop deficit, and the frozen resource-allocation rule stops/pivots this branch.

Do not rewrite this as “no feedback/history mechanism exists” or as a learned tracking benefit.

## Storage

- Raw output: `/media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113/`.
- Raw event-method CSV SHA256: `d08df0baec806d732fa04b463d073b0d0b9bb6ac5b3954fa1249fae77354a73e`.
- Raw summary SHA256: `d2e9d4138701de0105750cf0d6e836bb5164068358ac5a0cf65df9d73b4a998c`.
- Full audit: `EXPERIMENT_AUDIT.md`.
- Local review traces under `.aris/traces/` are intentionally ignored by Git.

No model weights, raw replay files or per-event CSVs are stored in Git.
