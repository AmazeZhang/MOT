# S06 seed 13 experiment audit

Date: 2026-11-24. Auditor: fresh same-family GPT-6.1-Sol ultra, read-only, provisional. Overall verdict: **WARN**.

The seed-13 result is reproducible and the frozen gate correctly returns `STOP_S06_NO_SEEDS_37_73`. No fabricated GT, prediction-dependent metric normalization, phantom result, or incorrect gate arithmetic was found. WARN reflects a material training-unit discrepancy, unequal effective model capacity, reused development scope, proxy supervision and incomplete historical GPU telemetry.

## A. Ground-truth provenance: PASS with proxy qualification

Final and checkpoint evaluations use dataset DanceTrack GT through official TrackEval HOTA/CLEAR/Identity. Training targets are explicitly offline proxies: current-frame detection/GT IoU≥0.5 plus a past-only stable owner from factual M updates. Inference does not load GT or owner labels.

The auditor independently reconstructed 24 sequences, 25,328 frames, 242 components, 126 fully labelable components and all 378 M/G/V action labels with zero mismatches. Evidence: `/media/imc/data/mot-data/audits/s06_seed13_integrity_20261124/proxy_target_reproduction.json`.

## B. Score normalization: PASS

Official metrics retain their standard definitions. `g/k,h/k` are structural training targets, not benchmark scores. Train-only normalization was independently recomputed from 273 edge rows × 8 features and matched exactly; no dev6/evaluation8 inputs entered it.

## C. Result existence and numeric fidelity: PASS

The auditor independently ran CPU-only official TrackEval on all seven evaluation8 tracker directories. All 7 conditions × 9 aggregate/sequence rows × 11 metrics, 693 comparisons, matched exactly with maximum absolute difference 0.0. Evidence: `/media/imc/data/mot-data/audits/s06_seed13_integrity_20261124/official_reproduction.json`.

Completeness/hash checks passed for 1 training lock, 72 dev locks, 40 new evaluation locks, 12 checkpoints, 12 dev and 7 final summaries, 384 GT/seqinfo/tracker hashes, 72 dev staged trackers and 56 final staged trackers.

## D. Executed code and protocol fidelity: WARN

The selector, metrics and frozen gate paths execute; 1,032 dev and 608 evaluation8 learned predictions are finite and choices match the strictly-positive rule and M/G/V tie priority.

Two qualifications were discovered after result inspection:

- The protocol says `batch_components: 128`; training expands each component into independent action examples, resamples 208 actions per epoch, and steps after 128 actions plus 80 actions. The last 80-action batch is still divided by 128. Both models share the implementation, but exact component-batch fidelity is unsupported.
- Local and Context each allocate 25,794 parameters, but Local supplies zeros to the two 64-wide neighbor inputs. Consequently 8,192 context-layer weights are structurally inactive for Local; input-dependent counts are 17,602 versus 25,794. Equal allocated parameters do not establish equal effective capacity.

These issues weaken mechanism comparison; they do not alter the official metrics or frozen stop decision and do not authorize a posthoc repaired rerun.

## E. Scope, chronology, failures and hardware: WARN

This is one seed on reused DanceTrack train development: train18, internal-dev6 and evaluation8 are disjoint but not untouched final evaluation. Appearance pretraining overlap remains unknown. Local chronology is consistent with protocol freeze before training/selection/evaluation, but hashes and filesystem mtimes are not external preregistration.

Earlier instrumentation/training failures remain in distinct directories. The final training environment requires `CUDA_VISIBLE_DEVICES=1`, one visible device, `cuda:0 → physical GPU1`, and deterministic CuBLAS; replay/evaluation require empty CUDA visibility. No immutable prelaunch allocation snapshot or independent historical GPU telemetry exists, so the audit verifies the code/environment contract rather than historical hardware utilization.

## F. Evaluation classification: PASS

- Final and dev tracking comparisons: real GT, official TrackEval.
- Training: real GT with a derived past-only stable-owner proxy.
- Trace/hash/finite checks: engineering checks, not performance evidence.

## Gate reproduction and claim impact

Context-risk minus M is HOTA `+0.09456944456033511`, AssA `+0.1427358275802959`, IDSW `+4`, and positive HOTA in 1/8 videos. Versus V it is HOTA `−0.8234863186461467` and AssA `−1.3287845120047663`. Only 2/8 frozen clauses pass. `dancetrack0049` loses HOTA `−5.36265255157533`, while `dancetrack0074` gains `+11.685902771291353`; six videos are unchanged.

Supported: stop S06 under this protocol and do not run seeds 37/73. Supported with scope qualifiers: all seven aggregate/per-sequence results. Unsupported: sufficient Context-risk gain, identity safety, exact component-batch compliance, effective capacity matching, method validation or generalization. Context-no-extra-risk is a predefined descriptive positive condition, but cannot replace the inspected frozen Context-risk primary gate.
