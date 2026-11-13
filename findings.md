# Research findings

## 2026-11-11: D2 result-to-claim gate

- Verdict: **partial**, with medium confidence.
- Review status: fresh GPT-5.6-Sol reviewer; same-family and provisional. Independent integrity verdict is unavailable.
- Deterministic evidence precheck: project-specific file/hash/range checks passed; the canonical generic evidence checker could not be resolved, so semantic acceptance is not upgraded.
- Positive evidence: D2 recovers 22/53 selected identity-change failures at the first association round across four difficult train sequences.
- Boundary evidence: 30/53 remain unrecovered; every positive detection is owned by another track. The intervention harms 2/112 identity-preserving controls.
- Unsupported: learned uncertainty, Flow Matching, deployability, closed-loop recovery, and official tracking-metric gains.
- Route: supplementary diagnostics. Run frozen D1 and D3 plus competing-track cost/margin decomposition before implementing learned predictors.

The exact defensible wording and missing evidence are maintained in `CLAIMS_FROM_RESULTS.md`.

## 2026-11-11: D1 history diagnostics and integrity gate

- V1 is invalid as a history-pollution estimand: its latest-owner relabel labels 92/112 success controls as harmed without changing tracking state or pairs.
- V2 clean-gap history yields 0 newly positive assignments in 53 identity-change failures, 1 newly positive and 7 newly negative assignments in 112 controls, and 0/30 recovery among D2 non-recoveries.
- Exposure matters: only 12/53 failures and 9/30 D2 non-recoveries had an intended-track gap update to suppress. Zero recovery is evidence against this narrow intervention, not against history modelling in general.
- Fresh same-family integrity review verdict: `WARN`, provisional. All counts, hashes, parent-event derivations, replay rows, and event fields reproduced; no phantom or post-selected result was found.
- The missing v2 association total-score margins are closed by a separately frozen addendum. It shows that all 30 D2 non-recoveries remain below their best eligible competitor after D1, even though 9 target scores improve; this is descriptive competition evidence, not a causal Hungarian decomposition.
- Remaining obligations: preserve the one ambiguous anchor, retain the 9/30 nonzero-intervention qualifier, and run D3 before learned predictors.

## 2026-11-12: D3 and D1/D2/D3 result-to-claim postmortem

- D3 yields 14 newly positive first-round assignments among 53 selected failures, with 1 newly positive and 7 newly negative controls. It recovers 9/30 D2-oracle-negative failures.
- D2/D3 newly-positive sets overlap by 5; their separate-counterfactual union is 31/53. Twenty-one events remain first-round negative under both, while one additional neither-new event was already baseline-positive.
- Integrity verdict for D3 is provisional `WARN` solely for selected train-only scope and wording; intervention, numeric, score-matrix, replay and provenance checks pass.
- The motion/distribution-dominance claim fails the result-to-claim gate: `claim_supported: no`, medium confidence. The diagnostics show bounded position-related headroom but do not rank motion above observation error, candidate competition, detection absence or history contamination.
- Constraint for future work: do not start MDN or Flow Matching from oracle headroom alone. A deterministic interface must first demonstrate natural closed-loop selection and persistence gains on frozen video splits.
- Bounded route: first register direction-cost and truly combined D2+D3 controls on the 21 still-negative events; then establish B0/B1 equivalence and run one deterministic P1 pilot with cached detections, official tracking metrics, control harms, post-gap persistence and latency. Include a simple observation/association comparator because D3 has 9 unique newly-positive failures.
- Stop/pivot if position prediction improves only oracle/error metrics, if simple observation or association controls explain the gain, or if natural closed-loop identity stability does not improve.
- Only after P1 is credible may an equal-encoder Gaussian be considered. MDN requires residual decision-relevant multimodality; FM requires a credible matched MDN comparison and must be dropped if it does not beat MDN after latency/budget accounting.

## 2026-11-12: D4/D5 final diagnostic closure

- D4 global direction-term removal yields 0 newly positive failures and 3 newly negative controls. It does not rescue any of the 21 baseline/D2/D3-negative focus events.
- D5 actually executes the simultaneous D2+D3 coordinate construction and yields 52 newly positive failures plus one newly positive control, including all 21 focus events.
- D5's positive result is dominated by construction: both known pair boxes equal current GT, so intended IoU is exactly one. It is an assignment-consistency check, not evidence that two causal mechanisms are synergistic or that a deployable model can recover 52 events.
- Fresh same-family audit reproduces all cells/matrices/replays and permits the bounded final-diagnostic stage to close with provisional `WARN`.
- G2 is now complete. Next work is split freeze, B0/B1 equivalence and a deterministic P1 natural closed-loop pilot; Gaussian remains HOLD, MDN/FM remain NO-GO.

## 2026-11-13: P1 natural closed-loop negative result

- All six frozen methods completed on the same eight motion-dev sequences. The one-time aggregate is `NO_GO`; 8/16 required checks fail.
- Mean learned HOTA/AssA is `49.270966/30.541388`, below B0 (`50.059599/31.662792`) and constant velocity (`51.056654/32.623675`). No learned seed beats constant velocity on both primary metrics.
- Fixed-event persistence recoveries/harms are learned `19/21`, `17/20`, `17/19`, versus constant velocity `23/20`. Every learned seed has negative net recovery minus harm.
- Five of eight sequences show positive learned-vs-B0 HOTA direction, but official combined aggregation is negative; sequence win counts cannot replace the registered combined metric.
- The offline development success did not transfer to robust natural closed-loop association. This is the principal scientific result of P1, not a failed execution.
- Fresh same-family audit independently reproduces all 831 events, 4,986 method-event outcomes, six official TrackEval evaluations and all gates. Verdict remains provisional `WARN` for selected scope, only 17 long-gap events and mutable chronology.
- Constraint: retain constant velocity in every subsequent comparator set; do not advance directly to Gaussian/MDN/FM or official val.
- Route: a bounded prospective mechanism diagnosis on identical B0 states/candidates. At most one correction is allowed if the diagnosis identifies a specific discriminating failure mechanism; otherwise stop/pivot this learned-motion direction.

## 2026-11-13: P1 common-state diagnosis closes learned-motion branch

- The diagnostic evaluates 831 fixed event reappearance frames with identical B0 tracker state, natural history, detections, direction term, gate and assignment backend; alternatives never write back. Coverage is 4,986 unique method-event rows, with 829 common applicable events.
- Learned is substantially better than constant velocity on the registered aggregate one-step measurements: mean target IoU is about `0.6168` versus `0.5224`, mean center error about `0.01623` versus `0.02222`, mean gate count 771 versus 692, and mean immediate-positive count 558.7 versus 509.
- Consequently, all four registered learned-disadvantage checks are negative and zero stages trigger. The planned mechanism needed to justify one correction was not found.
- This does not rescue learned P1: actual natural closed loop still favors CV in HOTA/AssA and recovery/harm. The discrepancy is an inversion between common-state one-step scores and evolving closed-loop outcomes.
- The inversion is not causal evidence for history feedback. Immediate first-round matches differ from reconnect/persistence, and the method-specific outcome join uses different evolving histories.
- Full same-family audit reproduces all raw rows and state isolation with provisional `WARN` only for scope/chronology. Result-to-claim is `no`, route `pivot`, confidence high.
- Constraint: do not relax thresholds, mine subgroups post hoc, train a correction, add Gaussian/MDN/FM, or use official val for this branch. Current authorized correction count is zero.
- P1 learned-motion is closed as a development branch. Future feedback/history/lifecycle work would be a separately authorized idea with a new prospective protocol, not continuation evidence for P1.
