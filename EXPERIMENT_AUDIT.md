# Experiment Audit Report

**Date**: 2026-11-10; val/G2 extensions and D1 semantic audit 2026-11-11; D3 semantic audit 2026-11-12

**Auditor route**: earlier GPT-5.6-Sol ultra call unavailable; D1 reviewed by fresh GPT-6.1-Sol ultra same-family agent

**Project**: MOT / DanceTrack OC-SORT G1 baseline and G2 train-only failure diagnostics

## Overall Verdict: WARN for D1; earlier scopes REVIEW_UNAVAILABLE

## Integrity Status: warn_d1_prior_unavailable

The independent semantic reviewer hit an account usage limit before returning an audit. Per the `experiment-audit` protocol, this is recorded as unavailable rather than replaced with an executor-authored PASS. The raw request/error trace is retained locally under `.aris/traces/experiment-audit/2026-11-10_run01/` and is ignored by Git.

The frozen val result was added on 2026-11-11. The existing reviewer route and its child checks still report the same usage-limit error; per protocol it was not downgraded to another model or repeatedly retried. Therefore the val extension also has no independent semantic verdict.

The G2 train-only event extraction was added later on 2026-11-11 under the same unavailable reviewer condition. Its deterministic checks below do not constitute an independent semantic or causal review.

The D1 history diagnostics were subsequently reviewed by a new same-family reviewer with an independent CPU replay. Its verdict is provisional `WARN`: numerical and provenance checks passed, but interpretation, protocol completion, one anchor, and runtime metadata require qualification. This D1 verdict does not retroactively upgrade the earlier G1/G2 scopes whose reviewer was unavailable.

## Deterministic checks completed

- All 40 expected DanceTrack train sequence summaries exist; their recorded cache and tracker SHA256 values match the files.
- The split summary totals match the 40 per-sequence summaries: 41,796 frames, 355,231 detections, 331,420 tracker rows, and 0 empty-detection frames.
- The final CPU cache replay covers 40 sequences and 331,420 rows; frame/ID pairs are identical and maximum XYXY error is `3.9e-5` pixels under a fixed `1e-4` tolerance.
- The TrackEval summary and detailed result files exist and are hashed in the train detection manifest.
- Two failed `1e-5` replay attempts and the later tolerance probe are retained and identified separately from the final replay validation.
- The val configuration and protocol hashes were fixed before metric inspection and still match after export.
- All 25 expected DanceTrack val sequence summaries exist; recorded cache and tracker hashes, split totals, and device fields were cross-checked.
- The val CPU replay covers 25 sequences and 202,711 rows; frame/ID pairs are identical and maximum original-image XYXY error is `3.2e-5` pixels under the frozen `1e-4` tolerance.
- The val TrackEval combined row (HOTA 52.059, DetA 77.400, AssA 35.130, IDF1 51.623, MOTA 87.265, IDSW 1,936) matches both the raw summary and the committed evidence projection.
- The G2 protocol hash remained `a825161d9800180559012b2b2cee5117e744e8d2aac31d1e2739d01a7dee6645` after event extraction, and its inputs name only four DanceTrack train sequences.
- All 797 G2 event IDs are unique and satisfy the frozen frame-order/gap-length invariants; outcome counts are 53 identity changes, 112 identity-preserving controls, and 632 unresolved endpoints.
- The 20+20 review sample is exactly reproducible from seed 20261111, and all four detection/tracker source hashes match the prior G1 train evidence.
- All 40 external review images match their indexed hashes. Only four were spot-checked for rendering alignment; all 40 manual-category rows remain explicitly `pending`.
- The contention addendum preserves all 797 parent event IDs and partitions all 2,367 gap frames into 1,291 candidate-contention and 1,076 raw-candidate-absence frames; event class counts sum exactly to 797.
- The addendum protocol, parent event table, complete contention table, and run summary hashes match the committed evidence. No validation path appears in the analysis inputs.
- Codex visually reviewed all 40 fixed sheets and recorded reviewer, confidence, visibility, and observable pattern. This is agent annotation, not human-expert ground truth or an independent audit.
- D2 protocol v1 failed before producing results because its byte-identical replay condition contradicted the already accepted G1 numerical replay contract. Protocol v2 preserves v1, records both pre-result engineering failures, and changes only replay acceptance to exact frame/ID plus `1e-4` XYXY tolerance.
- All 165 resolved parent events appear exactly once in the D2 table. Four baseline replays have exact frame/track IDs and maximum XYXY error `8.0e-6`; their reference hashes match G1 evidence.
- D2 counts recompute from the event table: 112 baseline positive matches, 132 oracle positive matches, 22 newly recoverable, and 2 harmed. All 30 oracle-unrecovered identity-change events have their positive detection assigned to another live track.

These checks establish artifact consistency only. They do not substitute for the unavailable adversarial semantic review.

## D1 semantic integrity extension: WARN (provisional)

- The reviewer rebuilt all 797 parent events and contention classes from the real DanceTrack GT, cached detections, and baseline tracker outputs.
- Both D1 tables contain the same 165 resolved parent IDs exactly once. All summaries and grouping counts recompute exactly; 632 unresolved parent events remain visible and excluded by protocol rather than silently dropped.
- Four saved replays cover 48,726 tracker rows with exact frame/track IDs and maximum XYXY error `8.0e-6` under `1e-4`.
- An independent CPU in-memory replay reproduced all 165 events, passing 3,795 v1 field comparisons and 2,640 v2 field comparisons.
- No fake GT, active self-normalization, phantom result, numeric mismatch, hidden negative effect, or event post-selection was found.
- V1's 92 harmed-label controls all take their semantic owner from `after_frame-1` inside the gap. This supports the disclosed conclusion that v1 is a failed semantic-relabel operationalization; it did not harm tracker state because it changed no state or pair.
- V2 faithfully deep-copies the target after `before_frame`, suppresses its gap updates with the same Kalman model, substitutes only the target association inputs at `after_frame`, and discards the oracle decision.
- V2 has zero newly positive first-round assignments among 53 identity-change failures; one failure was already positive. Controls change from 111 to 105 positives through one newly positive and seven newly negative assignments. D2 non-recoveries gain 0/30.
- Scope warning: only 12/53 failures and 9/30 D2 non-recoveries had a target-track gap update to remove. Zero recovery does not establish that history is irrelevant or that general history correction has no headroom.
- Anchor warning: one event (`dancetrack0012:gt4:f106-108`) satisfies the parent endpoint rule but its latest anchor observation maps to GT9 rather than target GT4. It remains included and changes no pair. “Trusted” is operational, not guaranteed identity-clean.
- Protocol warning: v2 promised association total-score margins, but the original output contains IoUs, coordinate changes, and pair changes only. The frozen protocol and run remain unchanged; this measurement gap is now closed by the separately labelled and audited addendum below.
- Metadata warning: original D1 summaries do not capture the command, runtime device, source hashes, or `seqinfo.ini` hashes. CPU execution was requested, observed, and independently reproduced, but the original raw run alone does not authenticate GPU usage.

Review independence is same-family and acceptance remains provisional. Trace: `.aris/traces/experiment-audit/2026-11-11_run02/`.

## D1 score-margin addendum: WARN (provisional)

- A separately frozen post-audit addendum reproduces all 165 D1 v2 events and supplies the omitted raw OC-SORT IoU-plus-direction score margins without changing the parent run.
- The fresh reviewer verified the active score formula, orientations, detection-score weighting, observation mask, strict shortcut threshold, inclusive final IoU gate, and absence of normalization.
- Independent CPU replay reproduced 5,610 parent-field and 5,610 addendum-field comparisons with exactly zero numeric difference. It captured 324 active Hungarian matrices containing 151,784 elements; addendum score matrices were bitwise equal to the solver's negative costs.
- All 165 IDs, all summary groups, every lightweight outcome cell, four 48,726-row replays, runtime metadata, source/parent/protocol hashes, 16 sequence hashes, and the upstream clean commit were verified.
- Among 53 failures the target score improves 11 times and gains the gate 3 times, but 52/53 clean margins remain negative; the sole positive-margin event was already correctly matched. All 30 D2 non-recoveries retain negative margins, despite 9 score improvements and 3 gate gains.
- Verdict remains `WARN` only because this inherits the four-sequence train-only, post-v1 exploratory scope. Pairwise margins are descriptive, not assignment certificates: global Hungarian can select a target even when its local margin is negative.
- Runtime provenance now records `CUDA_VISIBLE_DEVICES=""`, exact argv, environment, source hashes, `seqinfo.ini` hashes, inputs, and OC-SORT commit.

No numeric or implementation correction is required. Trace: `.aris/traces/experiment-audit/2026-11-11_run03/`.

## D3 detection-coordinate diagnostic: WARN (provisional)

- A fresh reviewer independently reconstructed all 797 parent events and contention categories, verified D3 retains exactly the 165 resolved IDs, and found no additional post-selection.
- For all 165 interventions, the corrected positive detection XYXY equals current target GT exactly; ID, confidence, count, order, all other detections, serialized tracker state, and every original association input remain unchanged. Only baseline association results are written into tracker state.
- All 6,435 raw result cells recompute with zero numeric difference. Failure assignments change 1→15 through 14 newly positive and 0 newly negative; controls change 111→105 through 1 newly positive and 7 newly negative; 9/30 D2-oracle-negative failures become newly positive under D3.
- The D2/D3 newly-positive sets are 22/14 with intersection 5, D2-only 17, D3-only 9, and union 31. The other 22 contain 21 events still first-round negative under both and one already baseline-positive event; they must not all be called unrecovered first-round negatives.
- Independent CPU replay checked 324 active cost matrices and 48,726 tracker rows. Score matrices match the upstream solver inputs; replay frame/track IDs are exact and maximum XYXY error is `7.85e-6` under `1e-4`.
- Forty provenance/hash consistency checks pass, including archived GT bytes, input/source/protocol hashes, raw artifacts and clean upstream commit. Runtime records exact argv, Python/NumPy/SciPy and empty CUDA visibility.
- Verdict is `WARN` for selected four-sequence train-only scope and interpretation. D2/D3 demonstrate non-identical isolated first-round headroom, not complementary causal mechanisms or a combined intervention result.
- Minor provenance limitation: original runtime does not record the selected assignment backend or `lap` version. The audit environment used `lap 0.5.12`, which does not retrospectively prove the original version.

No numerical or implementation correction is required. Trace: `.aris/traces/experiment-audit/2026-11-12_run04/`.

## D4/D5 final counterfactual diagnostics: WARN (provisional)

- A fresh reviewer independently reproduced all 15,180 D4/D5 CSV cells, 489 active assignment matrices containing 231,063 elements, 48,726 baseline replay rows, all 797 parent events, contention outputs and 52 input-hash checks. Maximum numerical difference is zero.
- D4 changes only direction weight 0.2→0. It produces zero newly positive failure assignments and three newly negative controls. This does not establish direction/history irrelevance.
- D5 changes exactly the known intended track and its existing positive detection to current target GT, leaving all other inputs and serialized state unchanged. It produces 52 newly positive failures and one newly positive control, with no newly negative assignment.
- D5 intended-pair IoU is exactly one by construction. In all 165 matrices its score is largest in the full detection row and target-track column. Therefore it is a privileged assignment-consistency stress test, not an achievable recovery ceiling or evidence of causal D2/D3 synergy.
- All 21 baseline/D2/D3-negative focus events become positive under D5 and none under D4. The remaining failure was already baseline-positive.
- Runtime provenance records `lap.lapjv 0.5.12`, empty CUDA visibility, source/input/protocol hashes and the clean upstream commit. Baseline frame/ID keys are exact and maximum XYXY error is `7.85e-6` under `1e-4`.
- Scope remains four deliberately difficult train videos and 165 resolved events; 632/797 parent events are excluded by endpoint resolution. Protocol mtimes precede outputs, but mutable times and self-recorded timestamps cannot independently prove freeze chronology.

No numerical or implementation correction is required. The bounded G2 final-diagnostic stage may close, but C4 and all learned/closed-loop claims remain unsupported. Trace: `.aris/traces/experiment-audit/2026-11-12_run05/`.

## Claim impact

- Full 40-sequence DanceTrack **train** baseline metrics: may be reported as artifact-backed train diagnostics, with the exact scope and evaluator stated.
- Replay consistency: may be reported with the fixed `1e-4` pixel tolerance and the two earlier failed attempts disclosed.
- Frozen DanceTrack val G1 metrics: may be reported as artifact-backed reference metrics, with the one-shot scope and precision/fusion caveat stated.
- G2 event counts and proxy co-occurrence: may be reported as descriptive train-only diagnostics with the selected-sequence scope and unresolved events disclosed.
- Candidate-contention versus raw-candidate-absence counts: may be reported as GT-assisted exploratory diagnostics; the similar contention prevalence in failures and controls may be reported descriptively.
- Codex visual-review fields: may be reported as agent observations with confidence and sampling limitations, not as true causal categories.
- D2: may support the claim that privileged current GT position creates first-round association headroom in 22/53 selected identity-change events while leaving 30/53 unrecovered.
- D2: does not support closed-loop recovery, tracking-metric improvement, deployability, or a learned motion-model claim because counterfactual decisions are not written back.
- D1 v1: supports only the claim that the semantic-owner relabel definition failed its success-control sanity check; it is not evidence that history correction harms tracking.
- D1 v2: supports the qualified claim of zero newly positive first-round assignments among 53 selected identity-change events under gap-update suppression; it does not support a general “history is irrelevant” claim.
- D1 v2 controls: one newly positive and seven newly negative first-round assignments are artifact-backed; they are not closed-loop trajectory changes.
- D1/D2 intersection: 0/30 D2 non-recoveries become positive under D1 v2, but only 9/30 received a nonzero history intervention.
- D1 score margins: supported as artifact-backed descriptive evidence that all 30 D2 non-recoveries remain below their best eligible competitor after D1; causal global-competition wording remains unsupported.
- D3: supports 14 newly positive isolated first-round assignments among 53 selected failures, with 1 newly positive and 7 newly negative controls; closed-loop recovery and official tracking-metric improvement remain unsupported.
- D2/D3 union: supports a separate-counterfactual newly-positive set union of 31/53, not combined-oracle efficacy or causal mechanism prevalence.
- D4: supports zero newly positive assignments among 53 selected failures and three newly negative controls under global direction-term removal only.
- D5: supports 52 newly positive failures and one newly positive control under the exact identical-GT-pair construction; it is not a deployable recovery count, synergy result or upper bound.
- Causal labels such as true occlusion, motion-model failure, or detector failure: unsupported until human-expert review where needed and the planned D1--D3 diagnostics are complete.
- Generalization beyond DanceTrack val, robustness, or new-method improvement: unsupported by these runs.

## Action item

- Preserve the D1 `WARN`, both original runs, and the separately labelled score-margin addendum. Its runtime/source/sequence provenance closes the specific metadata omissions for the addendum but not retrospectively for the parent run.
- Earlier G1/G2 scopes still need a semantic audit when reviewer capacity is available; their integrity acceptance remains unavailable, not provisional PASS.
