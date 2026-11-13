# Claims from completed experiments

This file records what the current evidence supports before later method work is allowed to strengthen the wording.

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
