# B0/B1 off-state equivalence on frozen motion-dev

The frozen eight-sequence motion-dev split was replayed twice from the same registered detection cache:

- B0 calls the fixed upstream OC-SORT tracker directly.
- B1 wraps only the module-level first-round association call and uses the off-state policy, which delegates the exact read-only inputs to the original upstream function.

## Acceptance result

- 8 sequences, 7,565 frames, 73,479 tracker rows and 7,565 B1 first-round calls.
- B0 and B1 frame/track IDs are exactly equal.
- B0 and B1 original-image float64 XYXY arrays are elementwise exactly equal.
- Both match the registered G1 tracker frame/IDs exactly. Maximum serialization/replay XYXY error is `7.85e-6` pixels, below `1e-4`.
- The adapter restores the upstream function after every update; second-round association, old-observation rematching, KF/ORU, lifecycle and output rules remain upstream code.

This validates only the B1 wrapper with research features disabled. It does not validate a learned P1 score, training, modified state writeback or metric improvement. Motion-dev contains four deliberately selected G2 diagnostic anchors and is development evidence.

Runtime: CPU-only with empty `CUDA_VISIBLE_DEVICES`; `lap.lapjv 0.5.12`; OC-SORT commit `bb2f6867a41082558d95ab47a854a3918de7831b`.

Raw run: `/media/imc/data/mot-data/runs/b0_b1_equivalence_motion_dev_20261112/`.

