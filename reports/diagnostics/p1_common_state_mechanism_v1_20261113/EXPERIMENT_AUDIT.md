# P1 common-state mechanism experiment audit

**Date:** 2026-11-13
**Auditor:** fresh GPT-6.1-Sol ultra, read-only, same-family
**Acceptance:** provisional
**Overall verdict:** `WARN`

The requested GPT-5.6-Sol reviewer was unavailable. The closest available reviewer independently inspected and replayed the full diagnostic. It found no fabricated GT, active self-normalization fraud, phantom result, dropped event, or counterfactual writeback. WARN is retained for selected development scope, the narrow one-step interpretation, only 17 long-gap events, descriptive linkage to method-specific closed-loop outcomes, and chronology represented only by mutable local evidence.

## A. Ground-truth provenance: PASS

- Dataset GT is loaded from each DanceTrack `gt/gt.txt` by `scripts/analyze_p1_common_state_mechanism.py` and `src/mot_research/analysis/failure_events.py`.
- Both source archives match the dataset manifest. All eight GT and eight `seqinfo.ini` files are byte-identical to their archive members; all 81,099 GT rows were parsed.
- Instrumentation verified that all six method outputs were fixed before each of the 831 GT accesses. GT scores geometry and identifies the already registered positive detection; it never constructs B0 state/history or updates the tracker.
- All 76,759 history appends match real current-frame fixed detections with score strictly greater than 0.6.

## B. Score normalization: PASS

- Center error uses image-diagonal normalization from dataset dimensions, not prediction statistics.
- Size errors are absolute log predicted/GT ratios; target overlap is raw continuous-coordinate IoU.
- Association score is raw IoU plus detection-score-weighted velocity-direction consistency, matching upstream OC-SORT.
- The auditor checked 9,773 active LAP matrices / 2,671,263 elements bitwise against the recorded score definition.
- An unused max-normalized `ct_dist` function exists upstream but is inactive because this run fixes `asso_func="iou"`.

## C. Result existence and numbers: PASS; chronology WARN

- Exactly 831 × 6 = 4,986 unique event-method rows exist.
- Every method has the same 829 applicable events. The two retained non-applicable events have a dead intended B0 track.
- All 829 applicable intended tracks have eligible histories; all five alternative predictors apply at those rows.
- All 54 method/gap/stratum summary blocks, all closed-loop linkage counts and 79,716 parent/geometry cells independently reproduce.
- Fifty-six recorded file-hash checks, two source-archive hashes and 16 archive-member byte comparisons pass.
- Protocol mtime precedes surviving outputs and no threshold mutation was detected, but mutable local files cannot independently prove historical preregistration or absence of deleted earlier outputs.

Key method aggregates on the common B0 state:

| Method | Target IoU mean | Center error mean | Gate eligible /829 | Immediate positive /829 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.625057 | 0.015774 | 771 | 575 |
| Zero delta | 0.608120 | 0.016791 | 771 | 551 |
| Constant velocity | 0.522368 | 0.022220 | 692 | 509 |
| Learned 20261112 | 0.617239 | 0.016212 | 771 | 555 |
| Learned 20261113 | 0.617172 | 0.016217 | 774 | 562 |
| Learned 20261114 | 0.615925 | 0.016266 | 768 | 559 |

All four frozen learned-versus-CV disadvantage checks reproduce:

| Check | Signed disadvantage | Threshold | Consistent disadvantaged seeds | Trigger |
| --- | ---: | ---: | ---: | --- |
| Geometry IoU | -0.094410774660 | 0.01 | 0 | false |
| Geometry center | -0.005988448890 | 0.005 | 0 | false |
| Gate rate | -0.095295536791 | 0.02 | 0 | false |
| Immediate assignment rate | -0.059911540008 | 0.02 | 0 | false |

Negative disadvantage means learned is better than CV on the aggregate common-state one-step measurement. No registered disadvantage check fires.

## D. Execution and no-writeback: PASS

The auditor independently replayed the entire experiment in memory:

- 7,565 frames and 73,479 tracker rows;
- all eight B0 replays byte-identical to registered and stored B0;
- all 4,986 rows / 189,468 CSV cells reproduced with zero differences;
- 749 event-bearing calls preserve serialized tracker/KF state and history;
- 2,996 original input-array isolation checks pass;
- all counters, birth mappings, method outputs and final source frames reproduce.

The adapter exposes read-only requests, alternative policies modify only copied tracker XYXY arrays, and the capture policy always returns the original B0 association for the actual update. Three targeted unit tests pass.

## E. Scope and interpretation: WARN

This is one detector/tracker configuration on eight selected DanceTrack train/motion-dev sequences, three learned seeds and a fixed 831-event population. It is a one-step diagnostic on B0-owned state, not a new natural closed-loop experiment. The 17 gap16--31 events cannot support a broad long-gap claim.

Immediate first-round match is not reconnect or persistence. Even for B0, 575 immediate positives contain only 158 reconnects; 417 do not reconnect, and one reconnect occurs without an immediate first-round positive. Output/min-hit gates, later matching and GT-to-output evaluation intervene.

The method-specific closed-loop outcomes use their own histories and stable IDs; joining them to common-B0-state rows is descriptive, not causal mediation.

## F. Evaluation classification: PASS

`real_gt_with_disclosed_common_state_counterfactual`. It is neither synthetic GT nor simulation-only evaluation. Fixed events are diagnostic proxies, not official physical-occlusion annotations.

## Claim impact and action

Supported:

- common-state isolation and exact numeric results;
- no registered aggregate learned-versus-CV disadvantage fired;
- the frozen resource-allocation rule therefore says to stop this learned-motion branch.

Unsupported:

- absence of feedback, history, lifecycle, subgroup or other unregistered mechanisms;
- learned closed-loop improvement;
- causal mediation from the descriptive closed-loop join;
- validation generalization, broad long-gap safety, or Gaussian/MDN/FM value.

The defensible wording is: **none of the four registered aggregate learned-versus-CV disadvantage checks fired on common B0 state; the registered stopping policy therefore stops this learned-motion branch.** It must not be rewritten as “no mechanism exists.”

## Artifact identities

- Raw summary: `d2e9d4138701de0105750cf0d6e836bb5164068358ac5a0cf65df9d73b4a998c`
- Event-method CSV: `d08df0baec806d732fa04b463d073b0d0b9bb6ac5b3954fa1249fae77354a73e`
- Input hashes: `5f0e4fb762aeae8bc1911f705947a3cdd3e563d41a9bf62af3d81883e82b21fc`
- Runtime: `a1261cab885a540ed1fe8504b520a49a9acd2e953ecf8ffae899b3e3ab954720`

Full local trace: `.aris/traces/experiment-audit/2026-11-13_run02/` (intentionally ignored).
