# G2 train failure-attribution evidence

Recorded: 2026-11-11 (Asia/Shanghai)

Scope: exploratory, train-only diagnostics on the preselected `dancetrack0012`, `dancetrack0020`, `dancetrack0037`, and `dancetrack0044` G1 failure sequences. No DanceTrack val data or metrics were used.

The frozen event is a bounded run of frames in which a continuously annotated GT identity has no one-to-one matched detector candidate at IoU 0.5 using candidates with score strictly above 0.6, bracketed by matched detections. This is matching unavailability, not proof that no raw candidate exists. Endpoint tracker matches divide events into identity-change failures, identity-preserving controls, and unresolved cases.

Files:

- `summary.json`: exact run summary with counts and limitations.
- `sequence_summary.csv`: counts for each of the four train sequences.
- `sampled_events.csv`: frozen-seed 20 failure + 20 success-control review sample.
- `descriptive_summary.csv`: descriptive rates only; events are clustered within videos/identities and are not independent trials.
- `overlay_index.csv`: hashes for 40 external review sheets; the JPG files remain on the data disk.
- `manual_review.csv`: all 40 sheets reviewed by Codex with visibility, candidate class, endpoint identity observation, visual pattern, confidence, and notes.
- `agent_review_summary.csv`: descriptive counts from the Codex visual review; this is not human-expert annotation.
- `source_hashes.yaml`: hashes of the full external event table and primary run artifacts.

Observed scope:

- 797 bounded matching-gap events: 53 identity changes, 112 identity-preserving controls, and 632 unresolved endpoints.
- The resolved subset contains 53/165 identity changes (32.1%), but this is a deliberately difficult four-sequence diagnostic subset and is not a population failure rate.
- Neighbor-overlap and history-ID-pollution proxies are common in both failure and success groups. Their co-occurrence is not causal evidence and currently has weak discriminative value.
- All 40 sheets were visually reviewed by Codex. Failure and success samples both predominantly show dense overlap or candidate contention; the review does not reveal a single clean separating pattern.
- In the fixed 20+20 sample, target visibility was marked `not_visually_resolvable` for 5 failures and 2 controls, but the sample is outcome-stratified, video-clustered, and mostly medium/low confidence. This is descriptive only, not a causal or population estimate.
- No human-expert review has been performed. The visual labels describe observable patterns and must not be presented as ground-truth causes.

Large artifacts intentionally excluded from Git:

- all 797 event rows (`all_recovery_events.csv`);
- the four source videos/images, GT, fixed detections, and tracker outputs;
- 40 rendered review JPG files.

Canonical definitions are frozen in `manifests/protocols/g2_train_failure_attribution_20261111.yaml`. This stage changes no tracker behavior and is not a new-method result.
