# S10 Report: CAMELTrack Pure Skip-Only Decomposition & Anti-Contamination Training

- **Protocol**: `s10_camel_anticontamination_20261127` (v1, protocol SHA256 `d89609d243ad...`)
- **Status**: **COMPLETED** (Stages A–E passed within budget on `CUDA_VISIBLE_DEVICES=1`; GPU 0 untouched; **0.0** TrackEval export diff)
- **Compute & Storage**: `NVIDIA GeForce RTX 4090 D` (physical GPU 1 only); total **4.072 GPUh** (B `0.391h`, C `3.384h`, D `0.297h`, budget `<= 12.0h`); extra storage **4.27 GiB** (budget `<= 25.0 GiB`).
- **Split & GT Boundary**: Strict `motion_train32` (C1 TE ckpt init + random GAFFE, `seed=13`, 20 epochs, `last.ckpt`, `sim_threshold=0.50`); evaluation on GT-free `motion_dev8` (`7,565` frames). **`D2_SKIP_ONLY`** (`b_d2_skip_only_v2`, `gt_injected_on_miss=0`) is strictly isolated as `privileged_skip_only_probe`. Official DanceTrack `val`/`test` untouched.

## 1. Results on `motion_dev8` (Official `COMBINED_SEQ`)

### Online Closed-Loop Conditions (Deployable, GT-Free)

| Condition | Inference Mode | Threshold | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T_TWOSTAGE_TH050` *(S09 Clean Ref)* | `s09_readonly_online_closed_loop` | `0.5000` | 54.443 | 80.969 | 36.704 | 53.629 | 90.373 | 786 | 1406 | 5615 | 1332 |
| `C0_SIMPLE_WRITE_GATE_TAU060` | `online_closed_loop` | `0.5000` | 48.838 | 80.090 | 29.847 | 44.474 | 87.603 | 1225 | 1138 | 7691 | 1534 |
| `C0_SIMPLE_WRITE_GATE_TAU065` | `online_closed_loop` | `0.5000` | 43.509 | 79.289 | 23.925 | 36.904 | 85.507 | 1666 | 994 | 9094 | 1622 |
| `C1_MATCHED_IID_SWAP` (`p_iid=0.0769`) | `online_closed_loop` | `0.5000` | 51.780 | 80.822 | 33.256 | 49.964 | 89.796 | 888 | 1251 | 6136 | 1395 |
| `C2_CONTIGUOUS_TAKEOVER` | `online_closed_loop` | `0.5000` | 48.210 | 80.528 | 28.936 | 45.176 | 89.025 | 1228 | 1290 | 6383 | 1445 |
| `C3_CONSISTENCY_GATED_TE` | `online_closed_loop` | `0.5000` | 49.781 | 80.752 | 30.765 | 46.604 | 89.507 | 1110 | 1331 | 6069 | 1445 |
| `C3_WITH_SOFT_WRITE_FILTER` (`w<0.50`) | `online_closed_loop` | `0.5000` | 49.313 | 80.739 | 30.201 | 45.506 | 89.382 | 1106 | 1351 | 6154 | 1462 |

### Isolated Privileged Counterfactual Probes (Not Deployable)

| Condition | Inference Mode | Threshold | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`D2_SKIP_ONLY` (`b_d2_skip_only_v2`, `gt_injected_on_miss=0`)** | `privileged_skip_only_probe` | `0.5000` | **63.781** | 81.568 | **49.940** | **63.186** | 88.958 | 936 | 1133 | 6886 | 1454 |
| `T_TWOSTAGE_TH050_PRIVILEGED_CLEAN_HISTORY` *(S09 D2 `v3` w/ miss GT)* | `s09_readonly_privileged_clean_history_probe` | `0.5000` | 81.907 | 83.835 | 80.063 | 92.153 | 91.673 | 830 | 1338 | 4585 | 1442 |

## 2. Direct Scientific Answers to S10 Questions

1. **Stage B Pure Skip-Only Upper Bound (`D2_SKIP_ONLY`)**:
   - With `gt_injected_on_miss == 0` (`2,170` cross-ID + `1,136` FP matches skipped; `71,681` clean updates), **`D2_SKIP_ONLY` achieves `63.781` HOTA / `49.940` AssA / `63.186` IDF1 / `936` IDSW** — a **`+9.338` HOTA (`+13.236` AssA, `+9.557` IDF1)** gain over S09 `T_TWOSTAGE_TH050` (`54.443` / `36.704`).
   - This isolates **+9.338 HOTA (+13.236 AssA)** attributable purely to halting cross-ID/FP gallery writes vs **`+18.126` HOTA (+30.123 AssA)** from S09 `D2 v3`'s `else:` GT re-injection on missed/misassociated frames (`63.781 -> 81.907`).

2. **Zero-Retrain Scalar Write Gate (`C0`) & Controlled Stage C (`C1` vs `C2` vs `C3`)**:
   - **Zero-retrain scalar write gate (`C0`) hurts strongly**: `tau=0.60` drops HOTA to `48.838` (`3,208` writes skipped), `tau=0.65` drops to `43.509` (`7,475` writes skipped). Low similarity often occurs during legitimate rapid motion/pose changes; blocking those gallery updates freezes stale templates.
   - **Swap-rate calibration (`200` batches)**: `C2_CONTIGUOUS_TAKEOVER` swaps `0.071268` of valid obs (`54.22%` tracklets), matched by `C1_MATCHED_IID_SWAP` (`p_iid=0.076880`) at `0.071080` (`0.26%` relative difference, well within `5%`). `C3` ConsistencyGateHead adds `382,785` params (`0.83% < 2%`).
   - **Stage C outcome (all negative vs clean `T_TWOSTAGE_TH050` `54.443` HOTA)**:
     - `C1_MATCHED_IID_SWAP`: `51.780` HOTA / `33.256` AssA / `888` IDSW
     - `C2_CONTIGUOUS_TAKEOVER`: `48.210` HOTA / `28.936` AssA / `1228` IDSW
     - `C3_CONSISTENCY_GATED_TE`: `49.781` HOTA / `30.765` AssA / `1110` IDSW
     - Adding causal consistency-gated attention (`C3`) recovers **`+1.570` HOTA / `+1.829` AssA / `-118` IDSW** relative to `C2`, confirming partial mitigation of the synthetic takeover degradation, **but fails to beat either `C1` (`-2.000` HOTA) or clean-trained `T_TWOSTAGE_TH050` (`-4.662` HOTA)**.

3. **Stage D Soft Write Filter & Real Closed-Loop Pollution AUROC**:
   - Adding causal soft write filtering (`C3_WITH_SOFT_WRITE_FILTER`, `w_last < 0.50`, `750 / 74,826` writes skipped) yields `49.313` HOTA / `30.201` AssA / `1106` IDSW (no online gain).
   - On `T_TWOSTAGE_TH050`'s real closed-loop `dev8` writes (`42,847` clean, `31,940` cross-ID, `1,350` FP), `C3`'s consistency head (despite `val/cons_acc = 0.9896` on synthetic takeovers) scores **`AUROC = 0.5002`**, **`F1 = 0.0259`** globally (`0.5032` mature `hits>=3`), compared to `1 - sim` (`AUROC = 0.5198` all / `0.5244` mature).
   - **Root-cause decomposition (`d_real_pollution_scores.csv`)**: First, `31,791 / 31,940` cross-ID frames are *post-switch persisted tails* where the tracklet's recent window is internally consistent with the *new* wrong person (`AUROC = 0.4997`). Second, even restricted to the **`149` onset switch frames** (first switch per tracklet), `C3` reaches only **`AUROC = 0.6220`** (`4.70%` triggered at `>=0.5`) vs **`0.8512` for raw `1 - sim`** (`sim_val` mean `0.6620` onset vs `0.7919` clean), exposing a large **sim-to-real mismatch** between synthetic same-frame neighbor swaps and natural closed-loop identity transitions.

## 3. Preserved Failed Engineering Attempts

- `b_d2_skip_only_v1`: Pre-tracking `AttributeError` (`load_dev8_gt_boxes` missing on `s09`), superseded by `b_d2_skip_only_v2`.
- `c1_matched_iid_seed13_e20_v1`: Epoch-1 callback `AttributeError` (`S10EpochLogger.t0`), superseded by `c1_matched_iid_seed13_e20_v2`.
- `d_c3_soft_write_filter_w050_v1`: Frame-12 `int64` `NaN` pad overflow in `compute_candidate_w_last`, superseded by `d_c3_soft_write_filter_w050_v2`.
