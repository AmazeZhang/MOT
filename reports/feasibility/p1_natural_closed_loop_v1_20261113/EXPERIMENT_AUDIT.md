# P1 natural closed-loop experiment audit

**Date:** 2026-11-13
**Auditor:** fresh GPT-6.1-Sol ultra, same-family read-only reviewer (requested GPT-5.6-Sol unavailable)
**Review independence:** same-family
**Acceptance status:** provisional
**Overall verdict:** WARN

The recorded `NO_GO` is numerically correct. The audit found no fabricated ground truth, fraudulent score normalization, phantom primary results, dropped method runs, or intervention leakage. WARN is retained because this is selected development evidence on one dataset/detector/configuration, the long-gap stratum has only 17 events and no success controls, and mutable local files cannot independently authenticate preregistration chronology.

## A. Ground-truth provenance: PASS

- DanceTrack dataset `gt.txt` supplies evaluation boxes and identities. All eight evaluated GT files are byte-identical to their original `train1.zip`/`train2.zip` members.
- Official quality metrics are produced by pinned TrackEval HOTA/CLEAR/Identity, including official `COMBINED_SEQ`; they are not home-written substitutes.
- Recovery events are a disclosed real-GT diagnostic proxy conditioned on fixed detections and registered B0, not annotations of physical occlusion and not model-generated GT.

Evidence: `failure_events.py:113-123`, `build_p1_closed_loop_event_set.py:97-103`, `evaluate_p1_recovery_events.py:166-174`, `evaluate_p1_trackeval.py:145-178`.

## B. Score normalization: PASS

- Official TrackEval values are projected to percentages only by multiplication by 100; counts remain raw integers.
- Recovery/harm values are raw counts on the frozen common denominator.
- The decision gate uses predeclared differences and means. No metric is divided by maxima, means, or other statistics of the evaluated prediction.

Evidence: `evaluate_p1_trackeval.py:47-60`, `evaluate_p1_recovery_events.py:81-101`, `summarize_p1_natural_closed_loop.py:104-233`.

## C. Result existence and numeric fidelity: PASS; chronology WARN

The reviewer independently verified:

- 48 tracker files, six methods × eight sequences.
- 889 recorded hash comparisons plus 27 training-lineage/artifact checks, with zero failures.
- All learned jobs were `attempt_001`, return code 0, no missing artifacts and empty stderr.
- Fresh B0 has all 73,479 frame/ID rows equal to registered G1 B0; maximum reconstructed XYXY difference is `7.85e-6` pixels, below the registered `1e-4` serialization tolerance.
- All 831 common events were rebuilt independently: 14,127 CSV cells exact, with 254 precondition and eight overlength exclusions reproduced.
- All 4,986 method-event outcomes were independently recomputed: 59,832 cells exact, including every grouping.
- Official TrackEval was independently rerun in memory for all six methods and eight sequences. Across 594 projected metric cells the maximum difference was zero.
- All 16 frozen decision checks were independently reproduced: eight PASS and eight FAIL; `NO_GO` is correct.

Primary summary SHA256: `3014c4a5d8e78ce7ad839e59326f98295e72d5f6e6609cbcd66380f9c7150ecc`.

Chronology is internally consistent: B0 evaluator acceptance preceded the main protocol; protocol mtime preceded queue start and all surviving candidate outputs. But local mtimes, hashes and self-recorded timestamps cannot prove that no earlier output was inspected/deleted or that files were never retroactively changed.

## D. Dead code and metric execution: PASS

Official metrics, recovery outcomes, grouped counts and all decision checks are invoked by the executed pipelines and appear in raw results. No claimed primary metric exists only as an unused function. `TorchMotionPredictor.synchronize()` is unused, but `.cpu().numpy()` waits for inference output before the measured update returns, and this is not a phantom metric.

## E. Scope: WARN

- One dataset: DanceTrack train; one detector cache and OC-SORT configuration.
- Eight motion-dev sequences, 7,565 frames and 81,099 GT detections.
- Three existing learned seeds and one deterministic run for each simple comparator.
- Four of eight sequences are deliberately selected difficult diagnostic anchors.
- Official val is excluded.
- Only 17 gap16-31 events exist, 12 from one sequence; every method has zero persistent successes and B0 has no success controls in this stratum. Its `0>=0` checks do not establish long-gap safety.

The evidence supports a development decision, not significance, broad robustness, validation generalization or independent dataset replication.

## F. Evaluation classification: PASS

- Official HOTA/AssA/DetA/CLEAR/Identity: `real_gt`.
- Recovery/persistence/harm: `real_gt`-based diagnostic proxy using a frozen B0-conditioned event population.
- Interface smoke and natural tracker outputs: engineering execution evidence, not standalone quality claims.

## G. Intervention integrity: PASS

- The adapter intercepts only first-round `associate`, presents inputs read-only and restores the upstream symbol in `finally`.
- Histories contain prior real detections that updated a tracker, append after the complete frame update, purge deleted IDs and reject current/future observations.
- Eligible policies replace only XYXY in a copy of the first-round tracker array. KF state, direction input, gates, detector input, rematching and lifecycle are not directly overwritten.
- Instrumented first-200-frame CPU replays across all six methods preserved serialized tracker/KF state through 1,200 policy calls and validated 12,360 appended observations against actual current-frame high-score detection rows.
- Full learned runs recorded 89,631/89,630/89,703 replacements; all fallbacks were insufficient-history cases. Birth mappings were 342/345/346 with zero unresolved/ambiguous cases and zero nonfinite/invalid predictions.

## H. Decision interpretation: PASS

| Method | HOTA | AssA | IDF1 | IDSW | Recoveries /699 | Harms /81 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 50.059599 | 31.662792 | 49.554270 | 762 | 0 | 0 |
| Zero delta | 47.664771 | 28.677977 | 46.052512 | 879 | 22 | 23 |
| Constant velocity | 51.056654 | 32.623675 | 49.605511 | 725 | 23 | 20 |
| Learned 20261112 | 49.052182 | 30.296771 | 47.105216 | 763 | 19 | 21 |
| Learned 20261113 | 49.205286 | 30.460473 | 47.637951 | 768 | 17 | 20 |
| Learned 20261114 | 49.555429 | 30.866921 | 47.908976 | 754 | 17 | 19 |

Mean learned HOTA trails the best comparator by `1.785689` percentage points; mean learned AssA trails by `2.082287`. Mean additional persistent recoveries versus constant velocity are `-5.333333`; learned net recoveries minus harms are `-2.333333`; harm mean/max are `20/21`, exceeding frozen limits. Engineering latency/numeric/birth checks pass.

## Claim impact

Supported:

- The frozen learned P1 failed its natural closed-loop development gate.
- Learned aggregate association performance trails both B0 and constant velocity despite successful engineering execution.
- Constant velocity is the strongest method in this frozen comparison and must be retained as the simple comparator.

Unsupported:

- Learned tracking improvement, broad long-gap safety or official-val/generalization benefit.
- A causal claim that history pollution or objective mismatch alone caused failure.
- Gaussian, MDN or Flow Matching necessity or likely benefit.

## Required next action

Follow the frozen negative branch: retain constant velocity, inspect natural-history distribution and association-objective mismatch without dev retraining, and pause complexity expansion. Any limited correction or mechanism probe requires a new prospective protocol.
