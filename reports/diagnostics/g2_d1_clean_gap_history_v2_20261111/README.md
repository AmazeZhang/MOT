# G2 D1 v2 clean-gap-history diagnostic

Scope: train-only, event-level privileged counterfactual on the same 165 resolved G2 events. V2 was designed after v1 failed its success-control sanity check and is therefore exploratory.

For each event, the intended target track is copied immediately after the parent `before_frame` update. The copy preserves the existing noisy observation and OC-SORT Kalman state. Across the frozen detector-gap frames it runs the same predict step but receives `update(None)`, preventing detections mapped to another GT from entering the target copy. At the post-gap frame, only that copy's predicted box, velocity, and previous observation replace the intended track's first-round association inputs. Detections, all other tracks, the solver, thresholds, baseline state, and lifecycle remain unchanged; no oracle result is written back.

## Results

- 165/165 events were applicable; baseline replay retained exact frame/track IDs with maximum XYXY error `8.0e-6` under the frozen `1e-4` tolerance.
- The baseline target track received at least one detection during the gap in 124/165 events; 104/165 received at least one detection whose one-to-one GT owner was another identity.
- Among 53 identity-change failures, clean history recovered 0 and harmed 0. The one baseline first-round positive match remained positive.
- Among 112 identity-preserving controls, clean history added one first-round positive match but destroyed seven; the count changed from 111 baseline positives to 105 clean-history positives.
- None of the 30 D2 position-oracle non-recoveries was recovered by D1 v2.
- Only 12/53 failure events had a target-track gap update to suppress; 41/53 therefore had identical clean/baseline predictions. Among the 30 D2 non-recoveries, only 9 received a nonzero intervention. Global first-round pairs changed in only 2/53 failures, versus 8/112 controls.

These results do not show that history is irrelevant. They show that this narrow intervention—removing target-track updates during the already frozen gap while leaving competitors unchanged—does not repair the selected identity-change failures and can remove useful positional support in controls. It is not a closed-loop tracker or a learned method result.

## Anchor audit

All 165 parent `before_frame` tracker endpoints match the target GT under the frozen endpoint rule. In 164 events, the latest tracker observation maps to the target GT and its box agrees with the target-positive detection at IoU at least 0.99. One crowded success-control anchor (`dancetrack0012:gt4:f106-108`) instead maps its latest observation to GT9; its box and the GT4-positive detection have IoU 0.616. That event changes no association pair and does not affect the aggregate recovery/harm counts, but it remains included because the protocol did not post-hoc exclude it. “Trusted anchor” is therefore operational, not guaranteed identity-clean.

The frozen protocol promised association total-score margins, while the original CSV contains IoUs, coordinate changes, and pair changes only. This is a disclosed protocol/output deviation; it is now closed by the separately frozen `g2_d1_score_margin_addendum_20261111` without modifying this parent run. The original summary also did not capture runtime device, command, source-script hashes, or `seqinfo.ini` hashes, so CPU-only execution is not independently authenticated by the raw parent run itself.

Raw run: `/media/imc/data/mot-data/runs/g2_d1_clean_gap_history_v2_20261111/`.
