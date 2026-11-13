# P1 natural closed-loop interface smoke

Decision: `ENGINEERING_PASS`. This validates the enabled natural-history path on one fixed fragment. It is not a tracking-quality result and does not authorize a HOTA/AssA claim.

## Frozen run

- Protocol SHA256: `f77a2a65f5d9a6c58915ca0debaacb47d3a8ad7983e8522aa5cc73847eb93b74`.
- Sequence/frame range: `dancetrack0012`, frames 1--200.
- Fixed checkpoint: seed `20261112`, SHA256 `a05c039ce19a4d9e5543216612e216bb0cd8b9c81507389e04aee232cd31432a`.
- The lowest numbered preregistered seed was used for engineering only; no seed, checkpoint, threshold, or method was selected from the smoke output.
- Physical GPU1 was the only exposed device and appeared inside the process as `cuda:0`. Physical GPU0 remained excluded.
- No GT was read and no HOTA, AssA, IDF1, DetA, IDSW, recovery, or harm metric was computed.

## Result

| Check | Observed | Frozen requirement | Result |
| --- | ---: | ---: | --- |
| Frames | 200 | exactly 200 | PASS |
| First-round calls | 200 | exactly 200 | PASS |
| Replaced tracker rows | 2,516 | at least 1 | PASS |
| Recorded birth observations | 17 | at least 1 | PASS |
| Unresolved births | 0 | at most 0 | PASS |
| Ambiguous births | 0 | at most 0 | PASS |
| Non-finite prediction fallbacks | 0 | at most 0 | PASS |
| Invalid prediction-box fallbacks | 0 | at most 0 | PASS |

The run saw 2,560 first-round tracker rows. Of these, 2,516 were eligible and replaced; 44 retained the upstream KF box because they had only one prior real observation. It wrote 1,945 MOT rows. All 17 new tracker births were uniquely mapped to their real high-score birth detections. The bounded histories recorded 2,060 detector observations, purged three deleted tracker histories, and never used current/future observations in a prediction.

Tracker-update latency after a fixed 20-frame warm-up was p50 `4.7521 ms`, p95 `7.0502 ms`, mean `5.1281 ms` over 180 cached-detection frames on the recorded GPU. This is tracker-only engineering timing, not end-to-end detector FPS and not yet a comparative latency result.

## Failure preservation

`attempt_001` failed before tracker execution because `mot-det` did not contain PyYAML. Its logs/status were preserved under `artifacts.failed`. PyYAML `6.0.2` was added to the declared environment and lock; `attempt_002` then passed with all six expected artifacts. No failed attempt was overwritten or included as a successful result.

## Storage and provenance

- Raw queue: `/media/imc/data/mot-data/runs/p1_natural_closed_loop_interface_smoke_v1_20261112/`.
- Successful tracker output SHA256: `9f6dd69221d49df790f1bedff3387cb196abaa78a90032dd077f492a6a46ea36`.
- Successful summary SHA256: `f3467492094b62414f01eab7777004f41b7c6d73c3415332a58912b91dac681f`.
- Upstream OC-SORT commit: `bb2f6867a41082558d95ab47a854a3918de7831b`.
- Full tracker rows, call logs, checkpoint, and raw queue remain outside Git.

## Interpretation limit and next step

This run establishes that the enabled path executes with tracker-owned histories, post-update appends, deletion cleanup, unique birth mapping, KF fallback, and first-round-only replacement. It does not show that associations or identities improve.

Next, define and freeze the complete six-run natural closed-loop protocol and its common-denominator evaluator before revealing full motion-dev method results. The main comparison remains B0/KF, zero delta, constant velocity, and all three existing learned seeds over all eight motion-dev sequences.
