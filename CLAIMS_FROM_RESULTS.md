# Claims from completed experiments

This file records what the current evidence supports before later method work is allowed to strengthen the wording.

## 2026-11-19: E0–E3 output and identity claim gate

Verdict: **not supported** for the proposed broad claim that mature-track re-emission plus observable risk gates form a new method that improves internal identity continuity. Fresh same-family semantic review is provisional; E0–E3 audits retain their `WARN` qualifications. See [decision record](docs/e0_e3_claim_gate_20261119.md).

Supported: on selected DanceTrack train development videos and fixed detections, re-emitting previously mature tracks recovers visible rows and modestly raises TrackEval HOTA, with measurable FP/IDSW costs. E is a simple comparison baseline. E3's short-gap rule did not clear its frozen train-fit benefit/cost gate.

Unsupported: internal association repair, novelty, safe identity improvement, generalization, or official validation of a method. Changes in AssA/IDF1 after output filtering are evaluation effects, not proof that the tracker chose better identities.

Next evidence: only a bounded, explicitly privileged association-headroom test is warranted before choosing a new state-changing mechanism. A positive oracle result would show room for an online information source; it would not validate a deployable algorithm.

## 2026-11-19: E4 conditional association headroom

Verdict: **supported only as a conditional GT-privileged local counterfactual** (same-family provisional, medium overall confidence; integrity `WARN`). See [E4 claim decision](docs/e4_claim_gate_20261119.md), [report](reports/diagnostics/e4_oracle_association_headroom_20261119/README.md) and [audit](docs/audits/e4_oracle_association_headroom_20261119/EXPERIMENT_AUDIT.md).

Supported: on 454 selected mature-track reappearance failures from 24 train-fit videos, forcing the GT-positive detection to the prior track once increases target same-ID output by 1,426 correct frames and all anchored identities by 1,643 correct frames over overlapping five-frame windows. The five frozen headroom checks pass.

Unsupported: available online features can find those pairs; a deployable algorithm exists; full-sequence HOTA/AssA/IDF1 improves; the effect generalizes to heldout videos; a learned motion distribution or FM is justified. “Headroom” is constructive and conditional, not a proven optimal bound.

Next evidence: one prelocked non-oracle appearance information test versus CV+E and a matched-budget simple geometry/score control, with a one-time train-heldout evaluation and a stop rule.

## 2026-11-20: E5 fixed non-GT appearance selector

Verdict: **not supported** for the proposed claim that this fixed HSV crop selector improves association correction over a matched-action geometry and score rule. The fresh same-family result-to-claim judgment is provisional; confidence is high for the fit-gate decision and the integrity audit is `WARN` for scope. See [E5 decision](docs/e5_claim_gate_20261120.md), [report](reports/feasibility/e5_nonoracle_appearance_local_20261119/README.md), and [audit](docs/audits/e5_nonoracle_appearance_local_20261119/EXPERIMENT_AUDIT.md).

Supported: on 24 DanceTrack train-fit videos, 576 paired local five-frame actions yielded appearance minus geometry all-anchor correct-output frames **−6**, against the frozen requirement of at least +58; other-anchor harm events were **422 versus 420**. Both methods were worse than unchanged CV+E on this local aggregate. The fit gate failed, so the eight heldout videos were not used.

Unsupported: a general claim that appearance cues are ineffective or geometry is superior; a deployable selector, full-sequence gains, official TrackEval changes, or heldout generalization. E4's GT-privileged headroom does not establish that online features can find its correct pairs.

Route: stop this selector without retuning it on fit or opening heldout. Any next association hypothesis needs a materially different non-GT mechanism and a new locked protocol after read-only error analysis.

Subsequent [read-only error analysis](reports/diagnostics/e5_error_analysis_20261120/README.md) found 529/576 identical selections and 414 events where both branches harmed other historical anchors. A GT-free normal first-round replay showed that chosen detections were usually already paired to another track. These posthoc diagnostics explain the risk of the tested trigger; they do not provide a new online rule or change the `no` verdict.

## 2026-11-11: G2 D2 current-position counterfactual

Verdict: **partially supported**. The external review used a fresh GPT-5.6-Sol reviewer, but it is a same-family review and therefore provisional rather than independently accepted. The integrity-review verdict remains unavailable because that reviewer hit a usage limit.

### Supported

- On the frozen 165 resolved train events, replacing only the intended history track's current predicted box with the current GT box changes 22/53 selected identity-change failures from an incorrect to a positive first-round assignment.
- The effect appears in all four selected difficult train sequences.
- Perfect current position is insufficient for most selected failures: 30/53 identity-change events remain unrecovered, and their positive detection is assigned to another live track.
- The intervention has measurable global-assignment cost: 2/112 identity-preserving controls are harmed.

### Not supported

- No learned uncertainty model, Gaussian/MDN predictor, Flow Matching model, deployable recovery method, or closed-loop tracking gain has been demonstrated.
- The event counts do not establish HOTA, AssA, IDF1, or persistent identity recovery.
- The result does not generalize beyond the four deliberately difficult train sequences or to the 632 unresolved parent events.
- D2 is a privileged, independent, first-round-only diagnostic and is not an achievable performance upper bound.

### Evidence still required

1. Freeze and run D1 history-correspondence and D3 detection-coordinate counterfactuals on the same event/control population.
2. Decompose the 30 D2 non-recoveries by competing-track ownership and association-cost margins.
3. Predefine a decision threshold for whether motion uncertainty has enough residual headroom to justify model development.
4. Compare deterministic, Gaussian, and MDN baselines before Flow Matching, followed by closed-loop video evaluation if the diagnostics justify it.
5. Obtain an independent integrity review before using the numbers in a paper claim.

### Current defensible claim

On a frozen, train-only set of resolved recovery events from four difficult DanceTrack sequences, privileged correction of the intended track's current position exposes limited first-round association headroom: 22/53 selected identity-change failures become recoverable, while 30/53 remain constrained by history/global competition and 2/112 success controls are harmed. This motivates D1/D3 and competition diagnostics; it does not yet justify a learned motion generator.

Raw evidence: `reports/diagnostics/g2_d2_first_association_v2_20261111/`. Review trace: `.aris/traces/result-to-claim/2026-11-11_run01/` (local, intentionally ignored).

## 2026-11-11: G2 D1 history diagnostics

Integrity verdict: **WARN**, from a fresh same-family reviewer and therefore provisional. The reviewer reproduced all 165 events in an independent CPU replay and found no fabricated GT, numeric mismatch, phantom result, or event post-selection.

V1 semantic-owner relabelling is a failed operationalization, not a tracking result. It labelled 92/112 success controls as harmed because all 92 owners came from the final gap frame's contaminating observation, while leaving every tracker state and assignment pair unchanged.

D1 v2 is an explicitly post-v1, exploratory intervention. It copies the target track at the pre-gap endpoint, suppresses all of that copy's gap updates, and evaluates the resulting first-round association without writeback. It produces zero newly positive first-round assignments among 53 identity-change failures; one failure was already positive and remains positive. Among 112 success controls it changes 111 baseline positives to 105, comprising one newly positive and seven newly negative controls. It recovers 0/30 D2 non-recoveries.

The defensible conclusion is narrow: removing target-track updates during the frozen gap did not repair these selected identity-change events. It does **not** show that history is irrelevant or that general history correction cannot help. Only 12/53 failures and 9/30 D2 non-recoveries had any target-track gap update to remove; pre-anchor history and competitor histories were unchanged.

Known protocol deviations and limits:

- The v2 frozen protocol promised association total-score margins, but the original parent CSV contains only IoUs, coordinate changes, and pair changes. This deviation is now closed by a separately frozen and audited addendum; the original run and protocol remain unchanged.
- One of 165 operational anchors maps its most recent observation to another GT; it remains included and changes no association pair.
- Original D1 summaries do not capture runtime device, command, source-script hashes, or `seqinfo.ini` hashes. CPU-only execution was requested and independently reproduced, but the original shell environment is not authenticated by the raw result files.
- Neither D1 version writes back oracle decisions or supports HOTA, AssA, IDF1, deployability, or learned-method claims.

Raw evidence: `reports/diagnostics/g2_d1_history_correspondence_20261111/` and `reports/diagnostics/g2_d1_clean_gap_history_v2_20261111/`.

### D1 score-margin addendum

The post-audit addendum reproduces all 165 D1 v2 events and records the raw OC-SORT IoU-plus-direction score. Among 53 identity-change events, clean history raises the intended score in 11, leaves 41 unchanged, and lowers 1; three tracks gain the IoU gate, but no event becomes newly positive. Every failure has an eligible competitor, and 52/53 clean intended tracks remain below the best eligible competitor; the sole positive margin is the event already correctly matched before D1.

For the 30 D2 non-recoveries, 9 scores improve, 3 gain the gate, and 2 global pair sets change, but all 30 intended margins remain negative, with median `-0.2871` and range `[-0.6387, -0.0421]`. This is consistent with strong score competition, not proof that competition causally determines the global Hungarian solution. Only 9/30 received a nonzero D1 history intervention.

The fresh same-family addendum audit is provisional `WARN` for inherited exploratory scope; every implementation, numeric, lineage, hash, and replay check passed. Raw evidence: `reports/diagnostics/g2_d1_score_margin_addendum_20261111/`.

## 2026-11-12: G2 D3 and combined D1/D2/D3 claim gate

D3 independently replaces only an already existing target-positive detection's XYXY with current target GT. Among 53 selected identity-change events it produces 14 newly positive first-round intended-track assignments and zero newly negative assignments. Among 112 controls it produces one newly positive and seven newly negative assignments. It produces 9 newly positive assignments among the 30 D2-oracle-negative failures.

D2 and D3 newly-positive failure sets contain 22 and 14 events, with overlap 5. Their separate-counterfactual set union is 31/53: D2-only 17, D3-only 9, and both 5. Of the remaining 22 events, 21 stay first-round negative under both interventions and one was already baseline-positive. This is not a combined D2+D3 oracle result.

Fresh D3 integrity review verdict: provisional `WARN` for selected train-only scope and interpretation. All intervention bytes, 6,435 result cells, 324 score matrices, 48,726 replay rows, recovery-set arithmetic and provenance checks reproduce. Wording is restricted to non-identical isolated first-round headroom; distinct causal mechanisms, closed-loop recovery, and official metric gains remain unsupported.

### Result-to-claim verdict

The intended claim that motion/distribution inadequacy is the dominant actionable cause is **not supported** (`claim_supported: no`, medium confidence). This is failure to establish dominance, not proof that the hypothesis is false. C4 remains `NOT_SUPPORTED_BY_PROVENANCE`.

Strongest defensible claim:

> On 165 resolved natural detector-gap events from four selected DanceTrack train sequences, independent privileged correction of target-track position and existing-detection coordinates produces non-identical first-round assignment headroom: 22 and 14 newly positive assignments among 53 identity-change events, respectively, with overlap five. Neither intervention establishes closed-loop recovery or causal error prevalence. Narrow gap-update suppression produces no newly positive failure assignments, but affects only 12 of those 53 events.

Unsupported conclusions include motion dominance, probabilistic-over-deterministic benefit, Flow Matching benefit, conditional predictability of GT hints, causal complementarity, general history irrelevance, deployability, persistence, HOTA/AssA/IDF1 gains, and generalization beyond the selected train clusters.

Routing decision:

- Minimal deterministic motion interface: **GO**, only as bounded engineering/P1 exploration after split freeze and B1 equivalence.
- Gaussian: **HOLD** until deterministic P1 and natural closed-loop evaluation are credible.
- MDN: **NO-GO now**.
- Flow Matching: **NO-GO now**.

Raw D3 evidence: `reports/diagnostics/g2_d3_gt_detection_coordinate_20261112/`. Result-to-claim acceptance is same-family/provisional; generic evidence checker remains unresolved and earlier D2 semantic integrity acceptance remains unavailable.

## 2026-11-12: D4 direction ablation and D5 joint construction

D4 globally removes only the first-round velocity-direction term. It creates zero newly positive assignments among 53 selected identity-change events and three newly negative assignments among 112 controls. The 21 failures still negative under baseline and separate D2/D3 also gain zero positives. This supports a narrow negative result for this ablation; it does not establish that direction, motion or history is irrelevant.

D5 is a true simultaneous counterfactual, not the set union of D2 and D3. It sets the known intended track and its existing positive detection to the same current GT rectangle, making their IoU exactly one by construction. All 165 intended pairs are selected: 52 failures and one control are newly positive, with no newly negative assignment. All 21 previously still-negative failures become positive.

The D5 result is an assignment-consistency stress test with a near-tautological geometric advantage. It supports only that retained direction/global assignment do not prevent the pair under this exact privileged construction. It does not support causal D2/D3 synergy, an achievable recovery ceiling, predictability of the corrections, closed-loop persistence, learned-model benefit, official metric gains or generalization.

Fresh same-family integrity review is provisional `WARN` for scope, chronology evidence and interpretation. It reproduces all result cells, solver matrices, parent/event derivations, hashes and baseline replay with no numerical correction. D4/D5 close the planned G2 diagnostic stage; C4 remains unsupported. Raw evidence: `reports/diagnostics/g2_d4_direction_cost_ablation_20261112/` and `reports/diagnostics/g2_d5_joint_d2_d3_20261112/`.

## 2026-11-13: P1 natural closed-loop claim gate

Verdict: **not supported** for the intended learned-motion improvement claim. The frozen six-method experiment completed successfully, but the aggregate decision is `NO_GO` with 8/16 required checks failing. A fresh same-family integrity audit is provisional `WARN`; it found no fake GT, score-normalization fraud, phantom result, dropped run, or intervention leakage.

### Supported

- On the selected eight-sequence DanceTrack motion-dev split, the three deterministic learned seeds have mean HOTA/AssA `49.270966/30.541388`, below B0 by `0.788633/1.121403` points and below constant velocity by `1.785689/2.082287` points.
- Constant velocity is the strongest registered method on this split: HOTA `51.056654`, AssA `32.623675`, IDSW `725`; it nevertheless harms 20/81 B0 success controls and is not generally safe.
- On the fixed event denominator, learned seeds recover `19/17/17` of 699 B0 persistence failures while harming `21/20/19` of 81 B0 success controls. Their recovery-minus-harm values are all negative.
- The negative result is not an engineering failure: all runs/evaluations passed, predictions were finite and valid, and learned p95 tracker-update latency stayed below the frozen 15 ms ceiling.

### Not supported

- The deterministic learned predictor does not improve natural closed-loop association over B0 or the strongest simple predictor.
- The 17 gap16--31 events all remain unsuccessful for every method; equality at zero does not establish long-gap safety.
- The result does not justify Gaussian, MDN, Flow Matching, official-val confirmation, or claims beyond this selected development split.

### Current defensible claim and route

On eight selected DanceTrack motion-dev sequences with fixed detections, a deterministic GRU that passed the frozen offline prediction gate failed the preregistered natural closed-loop tracking gate, while constant velocity was stronger overall but caused material success-control harm. This shows that the offline prediction gain did not transfer to robust closed-loop identity association under the tested interface.

Route: **pivot** with high confidence. Before any new motion model, run one bounded prospective mechanism diagnosis on identical B0 histories and candidate sets, retaining constant velocity as a required comparator. Compare geometric error, positive-candidate availability, gate passage, competitor rank, Hungarian assignment, and observed harms. Permit at most one mechanism-justified motion-train-only correction; otherwise stop the learned-motion branch.

Raw evidence: `reports/feasibility/p1_natural_closed_loop_v1_20261113/`. Local review trace: `.aris/traces/result-to-claim/2026-11-13_run01/` (intentionally ignored).

## 2026-11-13: P1 common-state mechanism stop gate

Verdict: **not supported** for the intended discriminating-mechanism claim. On 831 fixed recovery events, the registered common-B0-state diagnostic did not find a stable learned disadvantage versus constant velocity in geometry IoU, center error, gate passage or immediate first-round assignment. Result-to-claim route is `pivot` with high confidence.

### Supported

- All six predictors share identical B0 state, natural history, detections, direction inputs, gate and assignment backend; 829 events are commonly applicable and two dead-track events are retained for every method.
- Learned target-IoU means are `0.617239/0.617172/0.615925` versus CV `0.522368`; learned immediate-positive counts are `555/562/559` versus CV `509`.
- The four signed registered disadvantages are `-0.094411/-0.005988/-0.095296/-0.059912`, with zero consistent disadvantaged seeds and zero triggered stages.
- The frozen resource rule therefore stops/pivots the learned-motion branch. Currently justified deterministic corrections: zero.

### Not supported

- The common-state measurements do not identify why learned is worse in method-specific natural closed loop.
- A one-step immediate positive is not reconnect, persistence or HOTA/AssA improvement. B0 has 575 immediate positives but only 158 of those reconnect.
- The descriptive join to method-specific outcomes is not causal mediation because those methods evolved different histories and IDs.
- No evidence authorizes a deterministic correction, Gaussian, MDN, Flow Matching or official-val confirmation.
- No-trigger does not establish absence of feedback, history, lifecycle, subgroup or other unregistered mechanisms.

### Defensible conclusion

On 831 fixed recovery events from eight selected DanceTrack motion-dev sequences, none of four registered aggregate learned-versus-CV disadvantage checks triggered on the 829 common applicable B0-state events. Learned was stronger on these one-step aggregate measurements despite weaker prior natural closed-loop tracking. The diagnostic therefore did not explain the closed-loop deficit, and the frozen resource-allocation rule stops/pivots this branch.

Integrity status is same-family provisional `WARN` for scope and chronology, with all numeric/no-writeback checks passing. Raw evidence: `reports/diagnostics/p1_common_state_mechanism_v1_20261113/`. Local result-to-claim trace: `.aris/traces/result-to-claim/2026-11-13_run02/`.

## 2026-11-21: S01 deep appearance baseline versus candidate A mechanism claim

Verdict: **not supported** for the claim that S01 already demonstrates naturally persistent competitive identity-memory contamination and therefore warrants immediate candidate A training. Fresh GPT-6.1-Sol ultra result-to-claim review is same-family, provisional; integrity status `WARN`. See [S01 baseline report](reports/feasibility/s01_visual_association_20261120/README.md), [audit](reports/feasibility/s01_visual_association_20261120/EXPERIMENT_AUDIT.md), and [S02 prewritten protocol](manifests/protocols/s02_natural_identity_mechanism_20261121.yaml).

Supported: a fixed deep-appearance first-round association baseline is integrated and supplies descriptive visual-history information on 24 reused DanceTrack train-fit videos. Compared with G, M raises COMBINED_SEQ HOTA by 2.46, AssA by 3.32 and IDF1 by 2.56 percentage points at the fixed 0.25 appearance weight and 0.9 EMA. Some videos decline; 132/138 L/M changed first-round frames compare against no-appearance association on each condition's own state.

Unsupported: prevalence, duration or causal effect of wrong-identity memory writes; benefit of structured rollout training over matched alternatives; independent generalization; MOTIP research performance or new-method novelty. S01 traces lack GT ownership and persistence labels. M versus L also changes ordinary temporal smoothing, so aggregate differences cannot identify contamination.

Route: **S02 read-only natural-error diagnosis first**, followed by a controlled single-memory-write intervention only if observational opportunity is sufficient. No candidate A training from S01 alone. The reviewer has high confidence in this insufficiency judgment; the mechanism itself remains unknown.

## 2026-11-22: S03 single visual-memory write causal gate

Verdict: **not supported** for the claim that one wrong visual-memory write is a major cause of subsequent natural ID errors and already justifies candidate A structured-rollout training. Fresh same-family GPT-6.1-Sol ultra `result-to-claim` review is provisional, with high confidence for the scoped null and training stop; integrity audit is `WARN`. See [S03 decision](docs/s03_claim_gate_20261122.md), [report](reports/diagnostics/s03_single_memory_write_counterfactual_20261122/README.md), and [audit](reports/diagnostics/s03_single_memory_write_counterfactual_20261122/EXPERIMENT_AUDIT.md).

Supported: in frozen S01 M on 24 reused DanceTrack train-fit videos, suppressing exactly one target memory write in each of 86 GT-proxy wrong events and 86 selected clean updates changed no saved output in the available future windows of up to 30 frames. Every recorded local target/other/all-anchor correct-frame delta was zero. All target vectors changed; one actual subsequent fused-score probe changed, so the intervention reached the scoring path.

Unsupported: major causal influence of one wrong write, sufficient feasibility signal for candidate A training, general harmlessness of appearance memory contamination, repeated-write or longer-horizon nulls, an online method, full-sequence TrackEval gain, or independent generalization. Retry2 corrected controls after the first zero result, so it is not a blind independent replication.

Route: **stop candidate A training under this rationale now**. No further run of the unchanged single-write hypothesis is needed for that decision. Any continuation requires a distinct prospectively frozen mechanism diagnostic, explicit clean-update costs and collateral-identity accounting, and later independent matched training/validation if that new mechanism passes. Generic `evidence_check.py` was unavailable; S03 has separate deterministic artifact and scoring review.

## 2026-11-23: S04 bounded cumulative visual-memory freeze gate

Verdict: **not supported** for the claim that bounded repeated wrong visual-memory writes materially drive later ID errors and justify candidate A structured-rollout training. Fresh same-family GPT-6.1-Sol ultra result-to-claim judgment is provisional `no / stop`, high confidence for rejecting this current claim and enforcing the frozen stop; integrity review is provisional `WARN`. See [S04 decision](docs/s04_claim_gate_20261123.md), [report](reports/diagnostics/s04_bounded_memory_freeze_20261122/README.md), and [audit](reports/diagnostics/s04_bounded_memory_freeze_20261122/EXPERIMENT_AUDIT.md).

Supported: in frozen S01 M on eight reused DanceTrack train videos, all three saved arms are output-identical for 15 GT-selected wrong updates and 15 paired clean updates over 30 future frames. All local identity deltas are zero. Every case has multiple actual target writes suppressed and altered actual target fused-LAP scores. The stricter owner-positive competition with changed bounded-versus-single scores covers 14/15 wrong cases across four videos, not all 15 across five videos.

Unsupported: a positive identity rescue, a training signal, universal irrelevance of cumulative memory, population equivalence, an online event selector, official full-sequence gain, or independent validation. Bounded freezing also suppresses potentially corrective writes and preserves the wrong association/tracker/CV state. Overlapping 900 window-frame exposures cover 604 distinct frames; controls lack demonstrated covariate balance.

Route: **close this cumulative-freeze training rationale**. No extra S04 run is needed to enact the frozen stop. Any future mechanism work must be genuinely distinct and prelocked, with actual owner/rival competition and assignment margins, clean costs, collateral harm, video-level gate, and fresh evidence before training is reconsidered. Generic `evidence_check.py` remained unavailable; separate deterministic artifact/replay/scoring review passed.

## 2026-11-23: S05 visual stability and legal component-action coverage

Verdict: **supported only for the frozen development/resource gate** (`claim_supported: yes`, confidence medium, same-family provisional integrity `WARN`). S05 does not establish learnability, closed-loop gain, novelty, or generalization.

On the reused eight-video development scope, official combined TrackEval gives M versus G `+0.615954` HOTA, `+0.740692` AssA and `-35` IDSW. M wins HOTA on 5/8 videos but loses on 3/8, including `-7.1388` on `dancetrack0074`; unweighted per-video mean deltas are negative. The pooled result therefore supports continued use of M as a development baseline, not uniform robustness or independent confirmation.

On M's own causal states across the reused 24-video fit scope, a GT-free generator produced 242 non-identical M/G/V union components in 239 of 25,328 frames. Offline stable-owner proxy scoring found 126 fully labelable components and 41 with a legal deduplicated G/V action satisfying frozen `g-h>0`, across 17 videos. This narrowly clears the prewritten `40 components / 4 videos` resource gate. Forty-six components have some `g>0`; five also destroy one M-correct pair and are not net-profitable. Three fully labelable components have owner collisions; none are profitable.

Fresh same-family audit independently read all 152 JSON inputs, 335,407 tracker rows, 25,328 candidate traces and 242 components, and independently reproduced all 27 official TrackEval rows. It found no fake GT, prediction-dependent normalization, phantom results, duplicate inflation, or GPU0 use. The `WARN` is for reused development/fit data, proxy labels, unknown appearance-pretraining overlap and scope. The local protocol timestamp supports a pre-execution freeze on this machine, not external preregistration.

Route: `implement_and_run_seed13_only`. Seed 13 may proceed only under a separate frozen executable S06 protocol. Seeds 37 and 73 remain blocked until the seed-13 gate passes. The strongest current claim is candidate-resource feasibility; no model, training or tracking improvement has yet been demonstrated.

## 2026-11-24: S06 seed-13 component selector gate

Verdict: **not supported** for the intended claim that Context+risk provides sufficient natural closed-loop gain over M, fixed V and Local+risk to justify further seeds (`claim_supported: no`, confidence high, same-family provisional integrity `WARN`).

On the reused eight-video DanceTrack development scope, Context+risk versus M is HOTA `+0.09456944456033511`, AssA `+0.1427358275802959`, IDSW `+4`, and only 1/8 positive-HOTA videos. It trails fixed V by `0.8234863186461467` HOTA and `1.3287845120047663` AssA. Only 2/8 frozen clauses pass, so the exact decision is `STOP_S06_NO_SEEDS_37_73`.

The predefined Context-no-extra-risk condition is a descriptive secondary observation: versus M it has `+1.4801012144080374` HOTA, `+2.2036032163611523` AssA and `−9` IDSW; versus V, `+0.5620454512015556` HOTA, `+0.7320828767760901` AssA and `−20` IDSW. It improves HOTA on 4/8 videos, declines on 2/8 and ties on 2/8. It cannot replace the frozen primary policy after inspection, establish a context mechanism, authorize seeds 37/73, or support generalization.

Fresh same-family audit independently reproduced 693 final TrackEval cells exactly and all 378 proxy action labels. Integrity is `WARN`, not fabrication: the trainer batched 128 action examples rather than the frozen 128 components, and Local has 8,192 structurally inactive neighbor-input weights despite equal allocated parameter count. These weaken protocol fidelity and Local/Context attribution but do not change the observed metrics or stop decision.

Route: stop S06 and preserve the negative result. No new performance run is automatically warranted. A future no-extra-risk study would require a new question and protocol, corrected component batching, an effective-capacity control and a defensible untouched evaluation split; it would not be a corrective continuation of S06.
