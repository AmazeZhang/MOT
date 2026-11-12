# P1 motion sample construction

This snapshot records the frozen, strictly-online sample construction used for the first deterministic motion-model feasibility study. Large NPZ arrays remain outside Git.

## Result

- Source: the 32/8 DanceTrack train-derived motion split; official DanceTrack val was not read.
- Motion-train: 266,854 samples from 32 sequences; 261,069 targets have a current positive detection and 5,785 do not.
- Motion-dev: 80,458 samples from 8 sequences; 75,668 targets have a current positive detection and 4,790 do not.
- Every valid history observation precedes its target. Each sample has at least two observations, at most eight stored observations, and target-to-last-observation gap 1--31 frames.
- Train/dev sample-key overlap is zero. Normalization was fit once on motion-train (266,854 samples and 2,128,015 valid history observations) and reused unchanged on motion-dev.
- State encode/decode round-trip maximum absolute error is `2.220446049250313e-15`, below the frozen `1e-10` tolerance.

Two independent builds produced identical summary, normalization, input provenance, compressed NPZ SHA256 values, content hashes, and arrays element-by-element when the intentionally missing positive IoUs are compared with `NaN == NaN` semantics:

| Group | Array content SHA256 | NPZ SHA256 |
| --- | --- | --- |
| motion-train | `2d5c8b35546f18d66f306f771eabb294cdca4d93f1e628fc0b946b9385aec711` | `d1b4bf29d057c18ff98eb03a7a57f0fedfe598936542289ddf211474235016b7` |
| motion-dev | `e055e17ce0dc2e24bf2fd899ed24a266359aecd34f78f33d3f1ac8828e68d141` | `763e286a7a0a17ec688a51d2279a228eea26c61269532f6a12bcb69e0c2fd5b1` |

## Storage

- Primary raw build: `/media/imc/data/mot-data/runs/p1_motion_samples_v1_20261112/`
- Independent repeat: `/media/imc/data/mot-data/runs/p1_motion_samples_v1_repeat_20261112/`
- Runtime: `/home/imc/anaconda3/envs/mot-det/bin/python`, CPU-only with an empty `CUDA_VISIBLE_DEVICES`; physical GPU0 was not exposed.

## Interpretation limits

GT is used only to construct supervised targets and training correspondence. Histories contain earlier GT-matched real detector boxes, not closed-loop tracker-owned histories. The four previously inspected G2 sequences are deliberate motion-dev anchors. Therefore this artifact proves data-contract integrity and repeatability only; it is not evidence of prediction quality, tracker improvement, or generalization.
