# Step 3 — Abstract-Level Triage

Timestamp: 2026-11-10 (Asia/Shanghai)

Overlap score counts plausible matches on problem framing, core mechanism, key insight, and application domain.

## Structured papers

### TrajE: Multiple Object Tracking with a Learned Radius of Gyration

- Date: 2021-06
- Problem framing: Improve MOT under nonlinear/ambiguous motion using multiple trajectory hypotheses.
- Core mechanism: Recurrent mixture-density trajectory estimator with multiple hypotheses and beam search.
- Key insight: A distribution of plausible motion can support tracking better than one deterministic extrapolation.
- Application domain: Online 2D MOT.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2106.10950

### DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction

- Date: 2024-06
- Problem framing: Online MOT with nonlinear and diverse motion.
- Core mechanism: History-conditioned diffusion motion predictor, optimized for few/one-step sampling, followed by standard association.
- Key insight: Generative motion prediction can replace simple linear propagation in MOT.
- Application domain: DanceTrack, SportsMOT and related online MOT settings.
- Overlap score: 3/4.
- Source: https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html

### Learning Discriminative Motion Models for Multi-Object Tracking (DMMTracker)

- Date: 2024
- Problem framing: Motion prediction should reduce identity ambiguity rather than only minimize position error.
- Core mechanism: Task-specific features and sampled four-dimensional state prediction trained for identity-discriminative tracking.
- Key insight: Position prediction should be optimized for downstream identity distinction.
- Application domain: Multi-object tracking.
- Overlap score: 3/4.
- Source: https://orca.cardiff.ac.uk/id/eprint/170666/

### MOTIP: Multiple Object Tracking as ID Prediction

- Date: 2024-03 preprint / CVPR 2025
- Problem framing: Replace hand-crafted association with learned ID prediction.
- Core mechanism: Contextual identity tokens and direct ID prediction.
- Key insight: Association can be learned as an identity prediction task rather than fixed heuristics.
- Application domain: MOT including DanceTrack.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2403.16848

### MoFlow: One-Step Flow Matching for Human Trajectory Forecasting

- Date: 2025-03
- Problem framing: Efficient multimodal human motion forecasting.
- Core mechanism: Flow-matching teacher with correlated K-mode prediction and one-step distillation/IMLE.
- Key insight: Multiple coherent futures can be produced efficiently and jointly.
- Application domain: Known-identity trajectory prediction, not detection-to-track assignment.
- Overlap score: 1/4.
- Source: https://arxiv.org/abs/2503.09950

### TrajFlow: Multi-modal Motion Prediction via Flow Matching

- Date: 2025-06
- Problem framing: Efficient multimodal future trajectory prediction.
- Core mechanism: Flow matching that emits multiple candidates in one run, Plackett–Luce ranking loss, self-conditioning.
- Key insight: Candidate diversity and ranking should be trained together for usable multimodal prediction.
- Application domain: Autonomous-driving motion forecasting with known agent identities.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2506.08541

### GenPT: Generative Point Tracking with Flow Matching

- Date: 2025-10
- Problem framing: Track queried points through occlusion with multiple hypotheses.
- Core mechanism: Flow matching, cross-window priors and iterative refinement.
- Key insight: Generative hypotheses help maintain query identity through uncertain/occluded intervals.
- Application domain: Point tracking with identity supplied by the query, not general MOT assignment.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2510.20951

### DiffuTrack: Probabilistic multi-object tracking under occlusion

- Date: 2026
- Problem framing: Online MOT under nonlinear motion, short gaps and identity ambiguity.
- Core mechanism: Conditional diffusion over 4-D boxes; one or multiple hypotheses enter motion/appearance association; time-aware appearance prototypes are trained separately.
- Key insight: Distributional propagation should be judged by association continuity, not only point forecast error.
- Application domain: DanceTrack/MOT17/MOT20-style online MOT.
- Overlap score: 3/4.
- Source: https://www.nature.com/articles/s41598-026-50731-8

### OA-SORT: Occlusion-Aware SORT

- Date: 2026-03
- Problem framing: Occlusion corrupts detection observations and identity association.
- Core mechanism: Occlusion-aware cost correction and Kalman observation/update handling.
- Key insight: Observation bias, rather than insufficient motion expressiveness, can dominate failures.
- Application domain: Online MOT including DanceTrack.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2603.06034

### Constraint-Aware Flow Matching

- Date: 2026-05
- Problem framing: Align flow-generated samples with downstream constraints/decisions.
- Core mechanism: Differentiable constraint projection or decision-aware training of a flow-matching model.
- Key insight: A generative model can be trained for downstream feasibility/utility rather than pure distribution fit.
- Application domain: Not MOT.
- Overlap score: 2/4.
- Source: https://arxiv.org/html/2605.12754v1

### DOA-SORT: Directional Occlusion-Aware Multi-Object Tracking

- Date: 2026-09
- Problem framing: Online MOT identity association under directional occlusion and box deformation.
- Core mechanism: Clean-plus-four-direction mixture observation model; mixture likelihood used in several association stages.
- Key insight: Direction-specific observation distributions can improve association without a richer motion generator.
- Application domain: DanceTrack online MOT.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2609.22706

### MOTIP2

- Date: 2026-10
- Problem framing: Improve ID prediction for overlapping and spatially confusing targets.
- Core mechanism: Identity perturbation, spatial ID loss and spatial anchors.
- Key insight: Hard spatial competitors and geometry-aware ID supervision matter for association.
- Application domain: End-to-end MOT.
- Overlap score: 2/4.
- Source: https://arxiv.org/html/2610.10391v1

### Deep Network Flow for Multi-Object Tracking

- Date: 2017-06
- Problem framing: Learn MOT association costs from final assignment supervision.
- Core mechanism: Differentiable smoothed network-flow optimization.
- Key insight: Gradients from an assignment objective can train upstream association features/costs.
- Application domain: Multi-object tracking.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/1706.08482

### FAMNet: Joint Learning of Feature, Affinity and Multi-dimensional Assignment for Online MOT

- Date: 2019-04
- Problem framing: Jointly learn features, affinities and assignment for online MOT.
- Core mechanism: Differentiable feature/affinity/assignment network supervised by assignment ground truth.
- Key insight: Direct assignment supervision can replace separately tuned tracking modules.
- Application domain: Online MOT.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/1904.04989
