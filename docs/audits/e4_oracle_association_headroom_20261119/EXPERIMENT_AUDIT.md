# E4 retry3 experiment audit

**Date:** 2026-11-19<br>
**Reviewer:** fresh Codex agent, same-family provisional<br>
**Overall verdict:** WARN; numerical headroom gate PASS

| Check | Verdict | Evidence |
|---|---|---|
| A. GT provenance | PASS | `src/mot_research/analysis/failure_events.py:113-124` reads DanceTrack train GT. `scripts/run_e4_oracle_association_headroom.py:106-132,300-344` explicitly uses GT to choose candidates and force the correct pair. This is a privileged diagnostic. |
| B. Normalization | PASS | `scripts/run_e4_oracle_association_headroom.py:177-197,418-436` counts raw correct frames and divides only by the eligible event count for per-event reporting. |
| C. Artifacts and arithmetic | PASS | Independently verified 24 event files, 454 unique event rows, all event/source hashes and 1,903 candidate episodes → 1,073 hash-selected episodes. Every saved event belongs to the selected set; C anchors, positive detections and baseline five-frame scores were independently checked against GT and E3 E outputs. Event-row aggregation exactly reproduces +1,426 target and +1,643 all-anchor correct frames, 9 target harms and 37 other-anchor harms. |
| D. Branch behavior | PASS with auditability warning | Code removes conflicting first-round pairs, forces a one-to-one pair, verifies real positive update, same-state parent isolation, exact E3 C/E baseline replay and separate global birth counters (`scripts/run_e4_oracle_association_headroom.py:73-89,314-400`). Independent replay of dancetrack0087 frame 146 reproduced target +3 and all-anchor +6. Only per-event scores and checks were archived, so full verification of every oracle trajectory requires rerunning. |
| E. Scope | WARN | 24 train-fit videos; eligible set conditions on baseline failure. 454 five-frame windows have 2,270 frame exposures across 1,750 distinct video frames; 387 frames appear in multiple windows. These are not independent frame or video samples. |
| F. Evaluation type | PASS with qualification | Real dataset GT, explicitly privileged local branch counterfactual. No coherent full-sequence oracle output or official TrackEval result; no deployable method claim. |

Frozen gate arithmetic: 454≥100; 1,426/454=3.141≥0.5; 9/454=1.98%≤5%; 37/454=8.15%≤10%; 1,643/454=3.619≥0.25. The gate is numerically **PASS**. It supports conditional short-range association headroom only. “Upper bound” here is informal headroom language, not a proven optimization bound; a forced pair can harm another identity.

Original, retry1 and retry2 runs failed engineering checks before a complete result. Their failure directories and archived executed scripts explain the input-array, static ID counter and C/E anchor corrections. The retry3 protocol was saved before its run; local file times and hashes make chronology plausible but do not independently establish preregistration. No source outputs or failed runs were overwritten.

**Raw audit trace:** `.aris/traces/experiment-audit/2026-11-19_run03/` (ignored by Git).
