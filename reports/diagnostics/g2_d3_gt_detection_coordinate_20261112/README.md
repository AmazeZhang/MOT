# G2 D3 privileged detection-coordinate diagnostic

Scope: train-only, independent first-round counterfactual on the same 165 resolved G2 events. For each event, only the existing target-positive detection's XYXY coordinates are replaced with the current target GT box. The candidate ID, confidence, count, order, all track states, all other detections, solver, thresholds, and lifecycle are unchanged. The D3 result is not written back.

## Results

- 165/165 events were applicable. Baseline replay retained exact frame/track IDs with maximum XYXY error `7.85e-6` under the frozen `1e-4` tolerance.
- Among 53 identity-change events, baseline has one positive first-round assignment and D3 has 15: 14 are newly positive and none is harmed.
- Among 112 identity-preserving controls, D3 produces one newly positive and seven newly negative assignments, changing 111 baseline positives to 105.
- Among the 30 failures not recovered by the D2 perfect-track-position oracle, D3 produces 9 newly positive assignments. Thus D2 and D3 show non-identical isolated first-round assignment headroom; this does not establish distinct causal mechanisms.
- D2 produces 22 newly positive failure assignments and D3 produces 14; their intersection is 5. Their union contains 31/53 selected events. Of the other 22, 21 remain first-round negative under both interventions and one was already first-round positive in baseline, D2, and D3.
- D3 changes a global first-round pair set in 17/53 failures and 8/112 controls. The intended target score improves in 34 failures and worsens in 19; exact GT detection coordinates do not guarantee a better intended-track score because direction consistency and global competition remain active.
- All 165 baseline and D3 event frames enter the global Hungarian path; none uses OC-SORT's one-to-one IoU shortcut.

These counts show event-level headroom from correcting the location of an already existing detection. They do not measure detector recall, add missing detections, write corrected observations into the tracker, or establish HOTA/AssA/IDF1 gains. Seven control harms demonstrate that perfect target coordinates can still perturb the global assignment adversely.

## D2/D3 coverage

- D2 only: 17 failure events.
- D3 only: 9 failure events.
- Both D2 and D3: 5 failure events.
- Neither newly positive: 22 failure events, comprising 21 still first-round negative and one already baseline-positive event.

The union is a descriptive union of separate, non-simultaneous privileged counterfactuals, not the result of combining D2 and D3 in one tracker state.

## Runtime provenance

- Environment: `/home/imc/anaconda3/envs/mot-det/bin/python`, Python 3.10.22, NumPy 1.26.4, SciPy 1.13.0.
- Raw runtime records `CUDA_VISIBLE_DEVICES` as the empty string.
- OC-SORT commit: `bb2f6867a41082558d95ab47a854a3918de7831b`.
- Script/helper/upstream source, protocol, parent, `seqinfo.ini`, GT, detection and baseline hashes are recorded in `source_hashes.yaml` and the raw input index.
- The original runtime did not record the selected linear-assignment backend or `lap` version. The audit environment used `lap 0.5.12`, but that does not retrospectively prove the original version; future runs must record it.

Raw run: `/media/imc/data/mot-data/runs/g2_d3_gt_detection_coordinate_20261112/`.
