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

## 2026-11-19: E0–E3 result-to-claim gate closes output-only identity-method claim

- Fresh same-family provisional result-to-claim verdict: `claim_supported=no` for the broad claim that mature-track re-emission and risk gates are a new method improving **internal** identity continuity; confidence high for this rejection. The canonical generic `evidence_check.py` was unavailable, so this is a semantic review over recorded experiment reports, raw summaries and provisional integrity audits.
- E0 found 424 previously mature tracks updated by a positive detection but not emitted, as well as 242 events without a positive update. E1 CV+U HOTA +0.477 but IDSW +91 and official FP +328; U cannot change internal association by construction. E2 score gate retained a modest HOTA gain with costs; E3 G failed the fit gate on HOTA +0.050 and FP +84, and its heldout set remains unused.
- Narrow support: output-only policies can improve visible continuity and modestly lift development HOTA; E is a strong simple baseline with measured costs. AssA/IDF1 changes under output-only policies do not prove internal matching was repaired. Novelty, safe identity improvement and generalization remain unsupported.
- Route: stop U/G as identity-method candidates. One bounded GT-privileged association headroom experiment may test whether correcting missed positive reappearance matches has downstream five-frame value. The oracle is diagnostic only; even a positive outcome does not establish deployability, new method performance or official TrackEval gains. Do not restart Gaussian/MDN/FM or tune output thresholds from these results.

## 2026-11-19: E4 privileged association headroom

- E4 retry3 completed 24 fit videos after three preserved engineering failures. From 1,903 GT/C-defined candidate gaps, 1,073 were hash-selected; 454 were mature, live and not correctly updated by the registered positive detection in normal CV+E. A single forced correct first-round pair yielded +1,426 target and +1,643 all-anchor correct-ID frames over overlapping five-frame windows; target harm 9/454, other-anchor harm 37/454. All five frozen headroom checks pass.
- Fresh same-family integrity audit is provisional `WARN`: source/event hashes and gate arithmetic pass, one event's oracle branch was independently replayed, but all full oracle trajectories were not archived; selected fit events and overlapping windows limit inference. Three prior runs failed engineering acceptance and contribute no research result.
- Fresh same-family result-to-claim judgment supports only the literal conditional privileged headroom claim (`yes`, medium confidence). It does not support an online decision rule, complete-video TrackEval gains, novelty, heldout generalization or FM/MDN training. The oracle is not a formal optimization upper bound.
- Next allowed experiment: one locked non-oracle information-source test against CV+E and matched-budget simple IoU/margin+score control, calibrated on fit then evaluated once on eight train-heldout videos. Stop if it cannot beat the simple control on net all-anchor benefit and harm; do not tune repeatedly on heldout.

## 2026-11-20: E5 fixed non-GT HSV selector fails fit gate

- From 1,515 GT-free observable gap episodes across 24 train-fit videos, the frozen hash chose 664 and 576 led to matched local one-pair actions. The HSV and geometry rules disagreed on 47 actions. Both branch trajectories were saved before dataset GT scoring.
- Five-frame all-anchor correct-output net against unchanged CV+E: geometry −1,317, HSV −1,323; HSV−geometry **−6**, versus frozen minimum +58. Other-anchor harm events 420 versus 422. The two benefit/harm gate clauses fail; 100-action and 20-disagreement exposure clauses pass. E5 train-heldout remains untouched.
- Fresh same-family experiment audit is provisional `WARN` for fit-only, overlapping local forks, while all 576 raw rows, source hashes and gate arithmetic independently rescore without discrepancy. Fresh result-to-claim is `no`, high confidence for this exact fit-gate decision.
- This is not evidence that appearance is generally useless or geometry always better. The shared trigger is appearance-confidence-conditioned, 529/576 targets have a prior GT anchor and only 233/576 have a GT-positive current detection; both forced rules harm the original baseline substantially.
- Constraint: stop the fixed HSV selector; do not tune it on the same fit, open its heldout, or train FM/MDN from E4 headroom. A new idea would need read-only error analysis, a materially different observable trigger/selection mechanism, and a new prospective protocol.

## 2026-11-20: E5 read-only failure mechanism breakdown

- All 576 E5 actions were joined to saved five-frame outputs, GT scores, and a new GT-free normal first-round replay. Source hashes, E3 C/E parent replay and exact event keys match; the fresh same-family integrity audit is provisional `WARN` for fit-only posthoc scope, with no numeric mismatch.
- The methods pick the same detection in 529 actions and produce byte-equivalent saved branch outputs there; 383 of those events harm other prior C-anchored identities under both methods. Among 47 disagreements, appearance−geometry net is −6 correct frames, from 14 positive events totaling +47, 13 negative totaling −53, and 20 zeros. Only five disagreements had the registered target-positive detection in the alternative pool; geometry picked it four times and HSV once.
- The 343 events without an assigned target-positive detection under the fixed one-to-one IoU≥0.5 label split into 47 without a historical target GT anchor, 107 with target GT absent at the current frame, 64 with no high detection reaching IoU 0.5, and 125 with an overlapping high detection assigned elsewhere by Hungarian matching. This is not a detector-miss count. Of 233 with an assigned positive, normal first round already paired it to target in 121, paired it to another track in 105, and left it unpaired in seven.
- Normal first round had already paired E5's selected detection to another track in 567/576 geometry and 566/576 HSV actions. In 414 events where both branches harmed other anchors, 400 per branch removed a pair consistent with that detection's GT owner and prior C anchor; 398 events did so in both branches. Pair displacement follows from the forced rule, but the five-frame harm may also involve second-round and later updates.
- Interpretation: this fixed action trigger often overrides existing associations without a registered target-positive alternative. HSV ranking changes too few decisions to offset the shared damage. No validated online signal for safe abstention emerges from these GT posthoc groups. Stop E5 and its heldout; future work needs a distinct prospectively registered observable trigger or abstention mechanism, with no threshold mining on this fit.
