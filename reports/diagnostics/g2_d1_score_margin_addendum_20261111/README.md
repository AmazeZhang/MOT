# G2 D1 v2 association-score margin addendum

Scope: post-audit supplement for the unchanged 165 D1 v2 train events. It completes a measurement promised by the frozen parent protocol but omitted from the original parent CSV. It does not replace, modify, or reselect any D1 event or outcome.

The addendum independently replays OC-SORT, reconstructs the same clean-gap target state, and asserts all 165 parent outcomes, IoUs, prediction changes, positive-detection owners, and pair-change counts against the original D1 v2 event table. It then records the raw first-round score used by upstream OC-SORT: IoU plus score-weighted velocity-direction consistency, without normalization.

## Results

- Parent reproduction: 165/165 events passed.
- Four baseline replays retained exact frame/track IDs; maximum XYXY error was `7.85e-6`, below `1e-4`.
- None of the 165 event frames used OC-SORT's one-to-one IoU shortcut, before or after intervention. These crowded frames therefore enter the global Hungarian score path.
- Among 53 identity-change events, the intended target score improved in 11, was unchanged in 41, and worsened in 1. Three intended tracks gained the IoU gate and none lost it; global pair sets changed in 2 events, but there were still zero newly positive target assignments.
- Every failure event had at least one IoU-eligible competitor for the target-positive detection. After clean history, 52/53 intended tracks still scored below the best eligible competitor. The only positive margin belongs to the single failure event that was already a correct first-round target match before D1.
- Among the 30 D2 position-oracle non-recoveries, D1 raised the target score in 9 and left 21 unchanged; 3 gained the gate and 2 changed a global pair set. Nevertheless, the clean target margin remained negative against the best eligible competitor in all 30, with median `-0.2871` and range `[-0.6387, -0.0421]`.
- Among 112 success controls, the target score improved in 31 and worsened in 81; one lost the IoU gate. These are descriptive score changes, not closed-loop tracking changes.

The addendum sharpens the D1 interpretation: even when clean history improves the intended target's local score or gate eligibility, the positive detection remains more attractive to a competing live track in the D2 non-recovery set. This is consistent with strong score competition on these event frames, but pairwise margins are not an assignment certificate or causal decomposition because Hungarian optimizes the whole matrix. Two controls remain correctly assigned despite a negative pairwise margin, illustrating this boundary.

## Runtime provenance

- Environment: `/home/imc/anaconda3/envs/mot-det/bin/python`, Python 3.10.22, NumPy 1.26.4.
- `CUDA_VISIBLE_DEVICES` was recorded as the empty string in the raw runtime file.
- OC-SORT commit: `bb2f6867a41082558d95ab47a854a3918de7831b`.
- Script, helper, upstream source, `seqinfo.ini`, GT, detection, baseline, parent result, and protocol hashes are recorded in `source_hashes.yaml` and the raw `input_hashes.json`.
- Physical GPU0 remained at 413 MiB and 0% utilization after the run; the addendum itself records the empty CUDA environment rather than relying only on that snapshot.

Raw run: `/media/imc/data/mot-data/runs/g2_d1_score_margin_addendum_20261111/`.
