# Step 6 — Compare Against the Proposed Novelty

Timestamp: 2026-11-10 (Asia/Shanghai)

- **Proposed work**
  - Title: Association-directed motion distribution learning for short observation gaps
  - Date: 2026-11 planning stage
  - Source: local experiment handbook
  - Problem framing: Fixed-detection, strictly online MOT under short gaps and identity competition; closed-loop association is the primary endpoint.
  - Core mechanism: Gap-conditioned GRU+CFM box-increment distribution, sample compatibility cost, and candidate-set association loss backpropagated through the actual sampling path.
  - Key insight: Stop-gradient, joint and direct-classifier controls are needed to establish whether association supervision must reshape the generator rather than merely train a readout.
  - Application domain: DanceTrack-first online 2-D MOT.

- **DiffuTrack**
  - Title: DiffuTrack
  - Date: 2026
  - Source: https://www.nature.com/articles/s41598-026-50731-8
  - Problem framing: Same short-gap/nonlinear-motion online MOT regime.
  - Core mechanism: Diffusion box distribution plus association; identity objective lives in a separate appearance branch.
  - Key insight: Distributional propagation should improve association continuity.
  - Application domain: Same online 2-D MOT family.
  - Result: **3 axes matching → Level 2 — High Overlap (one axis differs).**

- **DiffMOT**
  - Title: DiffMOT
  - Date: 2024-06
  - Source: https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html
  - Problem framing: Same nonlinear-motion online MOT regime.
  - Core mechanism: History-conditioned generative motion predictor plus standard association.
  - Key insight: Focuses on generative motion, not association-gradient reshaping.
  - Application domain: Same DanceTrack/SportsMOT family.
  - Result: **3 axes matching → Level 2 — High Overlap (one axis differs).**

- **DMMTracker**
  - Title: Learning Discriminative Motion Models for Multi-Object Tracking
  - Date: 2024
  - Source: https://orca.cardiff.ac.uk/id/eprint/170666/
  - Problem framing: Motion prediction serving identity distinction in MOT.
  - Core mechanism: Task-specific features and sampled 4-D states rather than fixed-history CFM.
  - Key insight: Motion should be trained for identity discriminability.
  - Application domain: MOT.
  - Result: **3 axes matching → Level 2 — High Overlap (one axis differs), with full-text limitation.**

- **TrajE**
  - Title: TrajE
  - Date: 2021-06
  - Source: https://arxiv.org/abs/2106.10950
  - Problem framing: Multiple motion hypotheses for MOT.
  - Core mechanism: Recurrent MDN and beam search.
  - Key insight: Multi-hypothesis motion can help tracking, without decision-gradient isolation.
  - Application domain: Online MOT.
  - Result: **2 axes matching → Level 3 — Medium Overlap (two axes differ).**

- **Constraint-Aware Flow Matching**
  - Title: Constraint-Aware Flow Matching
  - Date: 2026-05
  - Source: https://arxiv.org/html/2605.12754v1
  - Problem framing: Different downstream constrained-generation task.
  - Core mechanism: Downstream-aware FM training.
  - Key insight: Decision constraints should shape the generator.
  - Application domain: Not MOT.
  - Result: **2 axes matching → Level 3 — Medium Overlap (two axes differ).**

- **TrajFlow**
  - Title: TrajFlow
  - Date: 2025-06
  - Source: https://arxiv.org/abs/2506.08541
  - Problem framing: Known-agent future trajectory prediction.
  - Core mechanism: FM multiple candidates plus ranking.
  - Key insight: Rank/usefulness supervision complements generation.
  - Application domain: Autonomous-driving forecasting.
  - Result: **1 axis matching → Level 4 — Low Overlap (three axes differ).**

- **DOA-SORT**
  - Title: DOA-SORT
  - Date: 2026-09
  - Source: https://arxiv.org/abs/2609.22706
  - Problem framing: Same occlusion/identity-association regime.
  - Core mechanism: Directional observation mixture rather than a motion generator.
  - Key insight: Models observation bias as the competing explanation.
  - Application domain: Same DanceTrack online MOT family.
  - Result: **2 axes matching → Level 3 — Medium Overlap (two axes differ).**

Overall minimum level: **Level 2 — High Overlap.** No candidate matched all four axes, but three candidates matched three axes.
