# Step 5 — Full-Paper Deep Dive

Timestamp: 2026-11-10 (Asia/Shanghai)

The repository already contains extracted full text for DiffMOT, DiffuTrack, TrajFlow, MoFlow, GenPT, OA-SORT and DOA-SORT from the 2026-11-03 review. Those local extracts were re-used. TrajE and DMMTracker remain limited to the workshop/author-page level noted below.

## DiffuTrack

- Problem framing (verified): Predict–associate–update online MOT; fixed box histories; short occlusions and nonlinear motion; association-centric outcomes on DanceTrack/MOT17/MOT20.
- Core mechanism (verified): Conditional diffusion in normalized 4-D box state; DDIM-style accelerated sampling; one or multiple samples converted into motion compatibility and fused with a separate appearance cost; TPCL supplies time-aware appearance supervision.
- Key insight (verified): A plausible support region can preserve matchability after gaps even when a single propagated point drifts.
- Application domain (verified): Online 2-D MOT.
- Venue: Scientific Reports, 2026.
- Assumptions & scope: Requires existing track history and candidate detections; does not train the motion generator with candidate-set identity competition; public code/evaluation differs from official TrackEval in ways documented in the earlier review.
- Closest-passage evidence: Introduction/contributions describe MDM as a motion diffusion model and emphasize association-centric outcomes; method sections "Accelerated sampling and multi-hypothesis use" and "Association and track update" specify multiple box hypotheses and Hungarian association.
- Refined overlap: problem match; core mechanism partial; key insight match; domain match. **3 axes match.**

## DiffMOT

- Problem framing (verified): Tracking-by-detection with nonlinear and diverse motion.
- Core mechanism (verified): History-conditioned D2MP diffusion model generates current motion; predictions are associated with detections using the standard tracker stack; public sampler supports very few, including one, steps.
- Key insight (verified): A learned generative motion prior can outperform linear propagation in complex motion.
- Application domain (verified): Online MOT on DanceTrack and SportsMOT.
- Venue: CVPR 2024.
- Assumptions & scope: Its contribution is motion generation plus standard association; it does not isolate association gradients into the motion generator. Public sampling code returns detached output in the inspected version.
- Closest-passage evidence: §3.1 defines detection, D2MP motion prediction and association as separate components; §3.2 defines D2MP as modeling the dataset motion distribution; §3.2.4 and the sampler implement accelerated sampling.
- Refined overlap: problem match; core mechanism partial-to-match at the generative-motion level; key insight differs from decision-supervised distribution shaping; domain match. **3 broad axes match, with the unique axis being the training objective/gradient path.**

## DMMTracker

- Problem framing (verified only at author-page abstract level): Learn motion predictions that help separate identities in MOT.
- Core mechanism: Task-specific visual/SOT features and sampling of four-dimensional states.
- Key insight: Motion prediction should be identity-discriminative, not only geometrically accurate.
- Application domain: MOT.
- Venue: IEEE Transactions on Multimedia, 2024.
- Assumptions & scope: Input includes image/SOT features, unlike the proposed pure-history fixed-detection setup. Full-text gradient path was not re-verified in this pass.
- Closest-passage evidence: Author repository abstract states that task-specific features and four-dimensional state sampling are learned for identity-discriminative position prediction.
- Refined overlap: problem match; mechanism partial; key insight match; domain match. **3 axes match, with evidence limitation.**

## TrajE

- Problem framing (verified at workshop-paper level): Improve MOT using learned multiple trajectory hypotheses.
- Core mechanism: Recurrent mixture-density model and beam search.
- Key insight: Multiple plausible trajectories can support tracking under ambiguous motion.
- Application domain: MOT.
- Venue: CVPR 2021 RVSU workshop.
- Assumptions & scope: No FM and no evidence of candidate-set association loss reshaping the generator.
- Closest-passage evidence: Method overview describes a recurrent MDN producing multiple trajectory hypotheses and their integration into existing MOT systems.
- Refined overlap: problem match; mechanism partial; key insight partial; domain match. **2 axes match conservatively.**

## Constraint-Aware Flow Matching

- Problem framing (verified from arXiv HTML): Train flow-generated samples to obey downstream constraints/utility.
- Core mechanism: Differentiable projection/constraint-aware objective integrated into FM training.
- Key insight: Distribution fit alone may not align samples with how a downstream decision consumes them.
- Application domain: Non-MOT constrained generation.
- Venue: arXiv preprint, 2026.
- Assumptions & scope: No identities, detections, births/deaths or Hungarian competition.
- Closest-passage evidence: §4–5 and Eq. 8 integrate downstream constraint information into flow training.
- Refined overlap: problem differs; core mechanism match at the decision-aligned FM level; key insight match; domain differs. **2 axes match.**

## TrajFlow

- Problem framing (verified): Predict K ranked future trajectories for a known agent in autonomous driving.
- Core mechanism: Flow matching, multi-shot candidate generation, Plackett–Luce ranking loss and self-conditioning.
- Key insight: Candidate generation should account for diversity and ranking confidence.
- Application domain: Waymo motion prediction, identities given.
- Venue: IROS 2025.
- Assumptions & scope: No detection-to-track assignment, births/deaths or closed-loop MOT identity state.
- Closest-passage evidence: Abstract/contributions state that the model produces multiple trajectories in one sampling run and trains their ranking with a Plackett–Luce loss; §III-B defines FM for multimodal motion prediction.
- Refined overlap: problem differs; core mechanism partial; key insight partial; domain differs. **1 axis match conservatively.**

## DOA-SORT

- Problem framing (verified): Online MOT identity association under directional occlusion and detection-box deformation.
- Core mechanism: One clean plus four directional observation components; likelihood is used for primary, low-confidence and recovery association.
- Key insight: The uncertainty needing a distribution may be the observation process, not the latent motion.
- Application domain: DanceTrack online MOT.
- Venue: arXiv preprint, 2026-09.
- Assumptions & scope: Hand-structured observation mixture, no learned FM generator; main DanceTrack result is validation-only.
- Closest-passage evidence: Abstract and §3.4–3.6 define the five-component observation mixture and use it in multi-stage association; §5.2 lists sensitivity to scale, occlusion density and motion distribution.
- Refined overlap: problem match; core differs; key insight differs/competes; domain match. **2 axes match.**
