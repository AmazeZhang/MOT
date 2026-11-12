# G2 D5 simultaneous D2+D3 assignment-consistency diagnostic

Scope: train-only, isolated first-round privileged construction on the same 165 resolved events. For each known target pair, D5 simultaneously sets the intended track prediction and its existing positive detection to the same current GT XYXY. It changes no score, candidate ID/count/order, other detection/track, velocity, previous observation, solver or tracker lifecycle. Decisions are not written back.

## Results

- All 165 intended pairs have IoU `1` by construction and are selected in the first round.
- Among 53 selected identity-change events, baseline has one positive assignment and D5 has 53: `52` are newly positive and none newly negative.
- Among 112 controls, baseline has 111 positives and D5 has 112: one is newly positive and none newly negative.
- All 21 failures still negative under baseline and separate D2/D3 are positive under this exact joint privileged construction.
- The D5 newly-positive failure set contains the D2 set (22) and D3 set (14) and adds those 21 events beyond their separate union of 31. This is an executed simultaneous counterfactual, not set arithmetic.

## Interpretation ceiling

D5 is an assignment-consistency stress test, not an achievable recovery ceiling. It uses the known correct pair and gives both sides identical GT geometry, producing a very strong association advantage: in this run the intended score is the largest in its detection row and target-track column for every event. The supported conclusion is only that retained direction and global assignment do not block the intended pair once this exact privileged geometry is supplied. It does not establish causal D2/D3 complementarity, predictability, learned-model benefit, closed-loop recovery or tracking-metric gains.

## Verification and provenance

- Same-family provisional integrity verdict: `WARN` for the near-tautological geometric construction, selected train-only scope and chronology limitation; implementation and numbers pass.
- Independent review reproduced all D4/D5 15,180 CSV cells, 489 assignment matrices, 48,726 baseline replay rows and 52 input-hash checks.
- Baseline replay frame/track IDs are exact; maximum XYXY error is `7.85e-6` under `1e-4`.
- Runtime records `lap.lapjv 0.5.12`, empty `CUDA_VISIBLE_DEVICES`, and OC-SORT commit `bb2f6867a41082558d95ab47a854a3918de7831b`.
- Protocol hash: `e47a740857929815e01bdc9a3af66b4ba4cb79d52cd138af5d72ccb747adf6fd`.

Raw run: `/media/imc/data/mot-data/runs/g2_d5_joint_d2_d3_20261112/`.
