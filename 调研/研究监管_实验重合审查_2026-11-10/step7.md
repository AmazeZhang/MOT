# Step 7 — Delta and Verdict

Timestamp: 2026-11-10 (Asia/Shanghai)

## Delta

Unlike **DiffuTrack**, which trains a diffusion box-motion model with a motion objective and attaches identity supervision to a separate time-aware appearance branch, the proposed work backpropagates current candidate-set association loss through the actual gap-conditioned CFM sampling path and compares it against stop-gradient, matched-head MDN/Gaussian/regression, and a generator-free classifier, yielding causal evidence about whether decision supervision must reshape the motion distribution itself.

## Verdict

**Level 2 — High Overlap.**

DiffuTrack and DiffMOT already cover the broad combination of generative motion distributions, online MOT and association-centric use, while DMMTracker covers the motivation of identity-discriminative motion prediction. Constraint-Aware Flow Matching and TrajFlow further remove any broad claim that downstream-aware FM or ranked multi-candidate FM is new. The remaining defensible delta is therefore narrow: the exact gradient path from real MOT candidate competition into the motion generator and the matched factorial controls that isolate its effect. No fully identical four-axis work was found, but the delta is fragile and should not be generalized to "first probabilistic/generative/FM motion tracker" or "first association-aware motion prediction."

## Experiment-overlap classification

| Current experiment block | Closest public precedent | Overlap judgment | Can it be claimed as innovation? |
|---|---|---|---|
| G1 baseline replay, official TrackEval sanity, repeatability | Standard reproducibility practice; TrackEval | Methodologically important, not a research novelty | No |
| G2 D1/D2/D3 privileged-information error attribution | OA-SORT, DOA-SORT, DiffuTrack diagnostic analyses | Same failure factors; the three-way decomposition is customized | At most an analysis contribution, not the main method |
| G3 regression/Gaussian/MDN/CFM under a shared encoder | TrajE, DiffMOT, DiffuTrack, TrajFlow/MoFlow | Strong overlap in model families and K/step ablations | No; it is a fairness/necessity test |
| Natural gaps plus artificial deletion and gap-length strata | DiffuTrack, GenPT and occlusion-robust tracking literature | Common robustness evaluation | No |
| K and ODE-step latency/accuracy curves | DiffMOT, DiffuTrack, MoFlow, TrajFlow | Standard generative-inference ablation | No |
| G4 predictor × motion/joint objective | DMMTracker, Constraint-Aware FM, differentiable association literature | Motivation overlaps; exact fixed-protocol factorial not found | Potentially, but only as causal experimental evidence |
| stop-gradient vs true sampling-path backprop | No exact public MOT match found; closest is downstream-aware FM outside MOT | Narrowest and strongest remaining delta | Yes, if implemented correctly and it changes closed-loop association |
| generator-free geometry classifier with matched head | MOTIP/MOTIP2, FAMNet, Deep Network Flow | Direct association learning is old; using it as a necessity control is sound | Control, not standalone novelty |
| Second dataset / multiple training and inference seeds / video bootstrap | Standard evaluation/statistical practice | Necessary validation | No |
