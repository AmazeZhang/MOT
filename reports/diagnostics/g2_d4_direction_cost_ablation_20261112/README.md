# G2 D4 direction-cost ablation

Scope: train-only, isolated first-round counterfactual on the same 165 resolved G2 events. D4 changes only OC-SORT's velocity-direction weight from `0.2` to `0.0`; detections, track predictions, velocities, previous observations, IoU gate, solver and tracker lifecycle are unchanged. Decisions are not written back.

## Results

- All 165 events are applicable. Among 53 selected identity-change events, D4 creates `0` newly positive and `0` newly negative intended assignments; three pair sets change.
- Among 112 success controls, D4 creates `0` newly positive and `3` newly negative assignments; seven pair sets change.
- Among the 21 failures still negative under baseline and separate D2/D3, D4 creates `0` newly positive assignments; one pair set changes.
- Removing the direction term improves the intended-pair raw total score in 21 failures and worsens it in 32 because the original signed direction term may help or penalize a pair. It changes no IoU gate.

This supports only the narrow observation that globally removing the first-round direction term does not repair these selected failure assignments. It does not establish that motion direction or history is irrelevant, nor isolate a causal source of failure.

## Verification and provenance

- Same-family provisional integrity verdict: `WARN` for selected train-only scope and interpretation; implementation and numbers pass.
- Independent review reproduced all D4/D5 15,180 CSV cells, 489 assignment matrices, 48,726 baseline replay rows and 52 input-hash checks.
- Baseline replay frame/track IDs are exact; maximum XYXY error is `7.85e-6` under `1e-4`.
- Runtime records `lap.lapjv 0.5.12`, empty `CUDA_VISIBLE_DEVICES`, and OC-SORT commit `bb2f6867a41082558d95ab47a854a3918de7831b`.
- Protocol hash: `ab329426deffcefcb50d38803fc78481cb03402194650041d04d34d8955dad7b`.

Raw run: `/media/imc/data/mot-data/runs/g2_d4_direction_cost_ablation_20261112/`.
