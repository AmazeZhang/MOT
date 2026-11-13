# P1 natural closed-loop v1

Decision: **`NO_GO`**. The frozen deterministic learned predictor does not improve natural closed-loop association over the registered B0 or the same-interface constant-velocity predictor on the selected eight-sequence DanceTrack motion-dev split.

## Frozen setup

- Protocol SHA256: `48c69f27aac2da9eac6c69314a4bba1616280895cc08cd78c566ed7c7a982534`.
- Six complete method runs: B0/upstream, zero delta, constant velocity and three existing learned checkpoints (`20261112/13/14`). No retraining, seed selection or dev early stopping.
- All methods cover the same eight motion-dev sequences and 7,565 frames using fixed detections.
- Learned jobs ran independently on physical GPU1/2/3 mapped to `cuda:0`; GPU0 was never exposed.
- Only eligible first-round association tracker boxes were replaced. KF state, gates, direction input, rematching and lifecycle were not directly overwritten.
- Evaluation used official TrackEval `COMBINED_SEQ` and the fixed 831-event recovery denominator. Official val was not used.

## Official results

| Method | HOTA | DetA | AssA | IDF1 | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0/upstream | 50.059599 | 79.323649 | 31.662792 | 49.554270 | 762 |
| Zero delta | 47.664771 | 79.418960 | 28.677977 | 46.052512 | 879 |
| Constant velocity | **51.056654** | **80.111086** | **32.623675** | **49.605511** | **725** |
| Learned seed 20261112 | 49.052182 | 79.620732 | 30.296771 | 47.105216 | 763 |
| Learned seed 20261113 | 49.205286 | 79.669277 | 30.460473 | 47.637951 | 768 |
| Learned seed 20261114 | 49.555429 | 79.748087 | 30.866921 | 47.908976 | 754 |

Learned mean HOTA/AssA is `49.270966/30.541388`: `−0.788633/−1.121403` percentage points versus B0 and `−1.785689/−2.082287` versus constant velocity. Mean learned HOTA is higher than B0 on 5/8 individual sequences, but losses on the other three outweigh those gains in official combined aggregation.

## Recovery and harm

The common event population contains 699 B0 persistence failures and 81 B0 success controls.

| Method | Persistent recoveries /699 | Success-control harms /81 | Net recovery minus harm |
| --- | ---: | ---: | ---: |
| B0/upstream | 0 | 0 | 0 |
| Zero delta | 22 | 23 | −1 |
| Constant velocity | **23** | 20 | **+3** |
| Learned seed 20261112 | 19 | 21 | −2 |
| Learned seed 20261113 | 17 | 20 | −3 |
| Learned seed 20261114 | 17 | 19 | −2 |

All methods have zero persistent successes on the 17 gap16–31 events. This stratum is small, has no B0 success controls and is concentrated in three sequences; equality at zero is not evidence of long-gap safety.

## Frozen gate

Eight of 16 required checks failed:

- learned mean HOTA and AssA gains over the best comparator;
- at least two seeds above the best comparator on both primary metrics;
- worst-seed HOTA safety versus B0;
- persistent recovery gain over the best simple predictor;
- net recovery minus harm;
- mean and per-seed success-control harm limits.

Passed checks include 5/8 paired-sequence HOTA direction, DetA safety, observational long-gap non-decrease at zero, latency, finite/valid predictions and birth mapping. Because `all_required=true`, the decision is `NO_GO`.

## Engineering result

The negative quality result is not an execution failure. All six tracker runs and all evaluations passed. Learned maximum per-sequence tracker-update p95 latencies were `10.9287/10.5127/11.0010 ms`, below the frozen 15 ms ceiling. The three runs recorded `342/345/346` mapped births, zero unresolved/ambiguous births and zero nonfinite/invalid predictions.

## Integrity and claim review

- Fresh same-family integrity audit: provisional `WARN` for selected development scope, weak long-gap evidence and mutable local chronology. It found no fake GT, metric normalization fraud, phantom results, dropped runs or intervention leakage.
- The auditor independently rebuilt all 831 events, all 4,986 method-event outcomes, reran official TrackEval for all six methods, checked 916 lineage/hash relations and reproduced all 16 decision gates.
- Result-to-claim verdict: `claim_supported=no`, confidence `high`, route `pivot`, integrity `warn`.

See `EXPERIMENT_AUDIT.md` for full evidence boundaries.

## Storage

- CPU tracker runs: `/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_20261113/`.
- Learned queue: `/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_learned_20261113/`.
- Raw evaluations: `/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_evaluation_20261113/`.
- Raw decision summary: `/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_summary_20261113/`.
- Tracker outputs, outcomes, official detailed CSVs and checkpoints remain outside Git.

## Defensible conclusion and next branch

The frozen learned P1 fails the natural closed-loop development gate. Constant velocity is the strongest method in this comparison and must remain a required baseline, although its 20/81 harms mean it is not generally safe.

Do not proceed directly to Gaussian, MDN, FM or official-val confirmation. The next bounded step is a prospective mechanism diagnosis on identical recorded B0 histories/candidate sets: compare KF, zero delta, constant velocity and all learned seeds on center/size error, IoU, correct-candidate availability, gate passage, competitor ranking and Hungarian assignment, then relate immediate effects to natural closed-loop harms. If this does not identify a discriminating mechanism, or one justified correction again fails, stop/pivot the learned-motion direction.
