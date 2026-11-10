# Step 2 — Search and Deduplicate

Timestamp: 2026-11-10 (Asia/Shanghai)

## Queries

1. `online multi-object tracking probabilistic motion prediction identity association supervision`
2. `motion distribution multi-object tracking occlusion association`
3. `flow matching multi-object tracking association loss`

Window: 2021–2026, with backward citation tracing for foundational association-learning papers.

## Search execution and limitations

- The installed `paper-search` documentation required `python` and `--json`; the host only exposes `python3`, and the installed script rejects `--json`. Both errors were surfaced rather than interpreted as negative search results.
- The programmatic six-source run was attempted. DBLP returned `Expecting value: line 1 column 1 (char 0)` for all three queries. The aggregate process then stalled while awaiting other remote workers and was interrupted after more than six minutes.
- A targeted web search restricted technical conclusions to primary paper pages (arXiv/CVF/publisher/author pages). It recovered DiffMOT and several broad association papers but no exact `FM + online MOT + candidate-set association loss backpropagated through the motion generator` match.
- The main evidence set therefore combines the 2026-11-03 six-source search and seven full-paper extracts already stored in this repository with the 2026-11-10 targeted update. The one-week-old search had 36 raw API records, 30 normalized-title groups, and explicit source failures; it is not represented as exhaustive.

## Deduplicated relevant set

The abstract-level set retained for Step 3 contains 14 works: TrajE, DiffMOT, DMMTracker, MOTIP, MoFlow, TrajFlow, GenPT, DiffuTrack, OA-SORT, DOA-SORT, MOTIP2, Constraint-Aware Flow Matching, Deep Network Flow for MOT, and FAMNet. Generic MOT papers, single-object tracking papers, unrelated flow-matching papers, surveys, and motion-prediction papers without a material connection to the claim were filtered as irrelevant after title/abstract inspection.

## Recall augmentation

Deep Network Flow for MOT (2017) and FAMNet (2019) were added by backward citation/targeted retrieval because they establish that supervision through association/assignment is not new in itself. They are outside the primary date window and are used only to bound the association-learning claim.
