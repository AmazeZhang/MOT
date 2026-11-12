# G2 D2 privileged first-association headroom

Recorded: 2026-11-11 (Asia/Shanghai)

Scope: train-only, event-level counterfactual on the 165 resolved G2 events. This is a privileged diagnostic, not a deployable method and not a closed-loop tracking evaluation. No val data or metrics were used.

For each event at the post-gap frame, the unchanged OC-SORT baseline is replayed. In a side-effect-free copy of the first-round association inputs, only the intended pre-gap track's predicted XYXY box is replaced by the current target GT box. Detection candidates, scores, all other tracks, velocity/direction cost, Hungarian assignment, thresholds, second-round logic, and lifecycle are unchanged. The oracle association is never written back.

Results:

- All 165 resolved events are applicable; the intended pre-gap track is still alive and the positive detection exists in every event.
- Baseline intended-track/positive-detection matches: 112/165. Oracle matches: 132/165.
- Among 53 identity-change failures, baseline is positive in 1, oracle is positive in 23, and 22 (41.5%) are newly recoverable.
- The other 30 failures remain unrecovered; in every case the positive detection is assigned to another track. The intended track is unmatched in 24 and matched to a wrong detection in 6.
- Among 112 identity-preserving controls, baseline is positive in 111; the oracle harms 2. This exposes global assignment side effects even with a perfect target position hint.
- The 22 recoverable failures comprise 15 contention-only, 4 mixed, and 3 absence-only parent events.

Interpretation boundary:

- There is meaningful first-round position headroom for a subset of failures, so motion should not be rejected outright.
- A majority of identity-change events (30/53) are not repaired by perfect current position alone; D1/history and global competition must be examined before building a learned motion generator.
- These counts do not imply HOTA, AssA, IDF1, or closed-loop identity gains because no oracle decision is written back.

Files:

- `summary.json`: exact external-run summary.
- `event_d2_first_association.csv`: complete 165-event counterfactual table.
- `failure_diagnosis.csv`: focused 53-row identity-change analysis joined to parent gap/proxy fields.
- `outcome_summary.csv`: compact result counts by parent outcome.
- `engineering_attempts.md`: the two pre-result failed starts and the frozen protocol replacement.
- `source_hashes.yaml`: protocol and external artifact hashes.

Four baseline CPU replays have exact frame/track IDs and maximum XYXY error `8.0e-6`, below the frozen G1 `1e-4` contract. Physical GPU 0 was not used.
