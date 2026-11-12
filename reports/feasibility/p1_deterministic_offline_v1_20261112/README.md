# P1 deterministic offline feasibility

This is the first full motion-train to held-out motion-dev prediction experiment. It is an offline real-GT evaluation with GT-aligned detector histories, not a closed-loop tracking result.

## Frozen setup

- Parent protocol SHA256: `5f918470aac7220073744d947e939ecb754cf7072756b9028168b51523057aea`.
- Aggregation addendum SHA256: `773024f3e9f10f363ef03b89da07ba1425db347d97742b26698b9e09e76e890d`.
- Fit: 266,854 samples from 32 motion-train sequences.
- Development evaluation: 80,458 samples from 8 disjoint motion-dev sequences.
- Three fixed seeds, ten fixed epochs, no dev early stopping and no seed selection.
- Physical GPU1/2/3 were assigned one seed each. GPU0 was never exposed.
- Comparators: zero delta and last-two-observation constant velocity in normalized center/log-size state space.

## Registered gate result

Decision: `GO_TO_NATURAL_CLOSED_LOOP`. This authorizes only the next controlled closed-loop experiment.

| Criterion | Observed | Registered requirement | Result |
| --- | ---: | ---: | --- |
| Mean learned macro-MSE / best comparator | 0.9156314457 | <= 0.98 | PASS |
| Every-seed macro MSE | 1.380908 / 1.385514 / 1.383229 | each <= 1.510670 | PASS |
| Mean learned gap2-31 MSE / best comparator | 0.9256415992 | <= 0.95 | PASS |
| Mean learned macro IoU | 0.8505008015 | >= 0.8386281957 | PASS |
| Relative macro-MSE seed range | 0.0033295921 | <= 0.10 | PASS |

Zero delta is the strongest comparator for macro MSE, gap2-31 MSE and macro IoU. Each learned seed has lower sequence-level normalized delta MSE on all 8/8 motion-dev sequences.

## Important adverse stratum

The registered overall gate passes, but the 650 gap16-31 samples do not show uniform IoU benefit. Zero-delta mean IoU is `0.3758572396`; learned seed IoUs are `0.3661499387`, `0.3721634479`, and `0.3649281420`, for a mean difference of `-0.0081100634`. Their normalized delta MSE still improves by about 9.28%. Any closed-loop design and report must preserve this mixed long-gap result.

The dev population is also dominated by gap1 samples: 75,428/80,458, with only 5,030 gap2-31 samples.

## Aggregation correction

The original aggregator stopped before creating a summary because it required exact Python dictionary equality for independently reduced comparator metrics. The only observed differences were IoU mean/median float leaves at `2.22e-16` to `3.33e-16`; structures, counts and non-float values matched.

Before reading learned gate results, the addendum froze an absolute `1e-12` comparator-float tolerance while retaining exact structure/count checks and all original research thresholds. The corrected aggregator canonicalizes comparator branches only and never changes learned branches. The integrity audit reproduced this behavior.

## Integrity audit

Fresh same-family review verdict: provisional `WARN`. It independently verified real GT, train-only normalization, all saved metrics, checkpoint predictions, comparator formulas, artifacts and gate numbers. WARN is caused by the selected/privileged development setup, adverse long-gap IoU, lack of closed-loop official metrics, and inability to independently authenticate preregistration chronology from mutable files.

See `EXPERIMENT_AUDIT.md` for supported and unsupported claims.

## Storage

- Raw queue: `/media/imc/data/mot-data/runs/p1_deterministic_offline_v1_20261112/`
- Raw gate summary: `/media/imc/data/mot-data/runs/p1_deterministic_offline_v1_summary_20261112/`
- Checkpoints and full dev predictions remain outside Git.

## Interpretation limit

The defensible conclusion is only that this deterministic GRU passes the preregistered offline development gate against the two specified simple predictors. There is no evidence yet of association improvement, persistent identity recovery, HOTA, AssA, IDF1, deployment benefit, unbiased confirmation, or a need for Gaussian/MDN/Flow Matching models.
