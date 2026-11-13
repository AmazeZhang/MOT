# P1 closed-loop common recovery events

This is the fixed method-independent denominator for natural closed-loop recovery and harm diagnostics. It was built from motion-dev GT, fixed detections, and the already registered B0 tracker before any full candidate-method output was produced.

## Frozen definition

- Protocol SHA256: `ddbd280e57119aa030b99b9e8c0d893c34fd86836cde3a637b3454b3d34c47aa`.
- Detection and tracker matching: global one-to-one Hungarian matching at IoU `>=0.5`.
- A gap starts after a matched fixed detection, contains 1--31 consecutive frames without an assigned positive detection, and ends when a positive detection reappears.
- B0 must have a matched tracker before the gap. Candidate methods do not define or filter the event population.
- Reconnection asks whether the method's own pre-gap ID is matched at reappearance.
- Persistence asks whether that same ID remains matched for five consecutive GT-visible frames starting at reappearance. Events without five remaining visible frames are right-censored for persistence.

## Population

There are 831 fixed events: 604 in the four selected diagnostic anchors and 227 in the other four motion-dev sequences. B0 reconnects its prior ID on 159/831 events. Persistence is evaluable on 780 events; B0 succeeds on 81 and fails on 699. Another 51 are right-censored.

| Gap bin | Events | B0 reconnect success | B0 persistent success controls |
| --- | ---: | ---: | ---: |
| 1 | 398 | 95 | 52 |
| 2--3 | 264 | 47 | 25 |
| 4--7 | 118 | 15 | 4 |
| 8--15 | 34 | 2 | 0 |
| 16--31 | 17 | 0 | 0 |

Per sequence, event counts are: dancetrack0012 132, 0020 226, 0024 12, 0033 99, 0037 71, 0044 175, 0053 41, and 0086 75. Construction excluded 254 gaps because B0 lacked the required pre-gap match and eight gaps longer than 31 frames.

The long-gap stratum is small and has no B0 success controls. It can reveal observed recoveries or degradation against another fixed method, but it cannot support a broad long-gap safety claim. The full protocol must retain exact counts and label this stratum inconclusive even if its mean looks favorable.

## Build integrity and storage

- Successful event CSV SHA256: `25c2602e9f5f2a6352b9bef3fc6ffdff0879b68016f0c06359c46b1ec867d2fc`.
- Successful summary SHA256: `3550e55faf879e843a30f68f9b13c8aab8b4103882ab1fcfc235df9661e475ca`.
- Raw successful build: `/media/imc/data/mot-data/runs/p1_closed_loop_common_events_v1_retry1_20261112/`.
- The first output path ended in `.failed` after event construction because the summary writer used the invalid Python token `false`; it remains preserved. The corrected retry used a new directory.
- Event rows and full GT/detection/tracker inputs remain outside Git. This report stores only counts and hashes.

## Interpretation limit

These events are development diagnostics conditioned on a B0 pre-gap match. They do not replace official TrackEval HOTA/AssA aggregation, are not an unbiased sample of all identity failures, and do not provide a significance guarantee.
