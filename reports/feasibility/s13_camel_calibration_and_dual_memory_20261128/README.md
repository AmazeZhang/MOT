# S13 Report: CAMELTrack Scale Calibration, Occlusion-Drop Supervision & K-Anchor Dual Memory (`K-AnchorCAMEL`)

- **Protocol**: `s13_camel_calibration_and_dual_memory_20261128` (v1, SHA256 `7c1a9f344d39...`)
- **Status**: **COMPLETED** (Stages A–E finished; `CUDA_VISIBLE_DEVICES=1` only; GPU 0 untouched; **0.0** TrackEval export diff; **exact** MOT match on all anatomy replays)
- **Thesis Escalation / Pivot Verdict**: **`CASE_3_STOP_CAMEL_LOCAL_PATCH_AND_PIVOT`** (`N3_DUAL_ANCHOR_CAMEL_TH040` HOTA **`57.519`**, AssA **`40.787`**, `seq_wins_vs_T2=4/8`, `post_switch_cross=42454`)
- **Compute & Storage**: `NVIDIA GeForce RTX 4090 D` (physical GPU 1); authoritative clean compute **7.264 GPUh** (`<= 8.5h` budget: B `0.909h`, C `6.028h`, D `0.328h`); gross compute including preserved failed/superseded `v1`/`v2` repair runs = **10.011 GPUh**; extra storage **5.063 GiB** (`<= 15.0 GiB`).

## 1. Closed-Loop Results on `motion_dev8` (`COMBINED_SEQ`)

| Condition | Thresh | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | ms/f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T2_TH040` *(S11 Baseline Ref)* | `0.40` | 57.572 | 81.394 | 40.845 | 60.117 | 91.184 | 832 | 1676 | 4642 | n/a |
| `M1_C_KIN_EDGE_TH040` *(S12 Ref)* | `0.40` | 57.033 | 80.990 | 40.275 | 60.417 | 91.079 | 957 | 1723 | 4555 | 64.93 |
| `D2_SKIP_ONLY` *(S10 Oracle Ref)* | `0.50` | 63.781 | 81.568 | 49.940 | 63.186 | 88.958 | 936 | 1133 | 6886 | n/a |
| `B_M1_CAL_055` | `0.55` | 57.001 | 80.989 | 40.229 | 60.359 | 91.071 | 960 | 1722 | 4559 | 64.02 |
| `B_M1_CAL_060` | `0.60` | 56.994 | 81.146 | 40.139 | 59.759 | 91.213 | 954 | 1720 | 4452 | 64.40 |
| `B_M1_CAL_064` | `0.64` | 56.208 | 81.434 | 38.916 | 58.267 | 91.136 | 956 | 1716 | 4517 | 64.71 |
| `B_M1_CAL_068` | `0.68` | 57.079 | 81.120 | 40.270 | 59.972 | 91.108 | 945 | 1690 | 4576 | 64.92 |
| `B_M1_B1_CAL_058` | `0.58` | 55.953 | 81.014 | 38.773 | 58.448 | 91.091 | 971 | 1700 | 4554 | 66.41 |
| `B_M1_B1_CAL_062` | `0.62` | 56.167 | 80.898 | 39.112 | 59.203 | 91.037 | 1011 | 1703 | 4555 | 66.58 |
| `N1_KIN_SCALE_ANCHORED` (@0.40) | `0.40` | 54.461 | 81.332 | 36.593 | 57.602 | 91.004 | 1030 | 1705 | 4561 | 65.30 |
| `N1_KIN_SCALE_ANCHORED` (@0.54) | `0.54` | 54.246 | 81.298 | 36.321 | 57.100 | 91.010 | 1022 | 1701 | 4568 | 64.35 |
| `N2_KIN_OCC_DROP_SUP` (@0.40) | `0.40` | 57.194 | 79.919 | 41.063 | 60.528 | 91.138 | 1031 | 1725 | 4431 | 62.41 |
| `N2_KIN_OCC_DROP_SUP` (@0.50) | `0.50` | 55.470 | 80.025 | 38.597 | 57.365 | 90.895 | 1095 | 1766 | 4523 | 65.08 |
| **`N3_DUAL_ANCHOR_CAMEL`** (@0.40, *Stage C Winner*) | `0.40` | 57.519 | 81.365 | 40.787 | 59.550 | 90.874 | 1187 | 1729 | 4485 | 75.84 |
| `N3_DUAL_ANCHOR_CAMEL` (@0.48) | `0.48` | 57.453 | 81.110 | 40.831 | 59.674 | 91.007 | 1193 | 1733 | 4367 | 73.91 |
| `N3_PLUS_B1_THCAL` (@0.48, dustbin=0.48) | `0.48` | 57.467 | 81.447 | 40.680 | 60.405 | 90.925 | 1114 | 1701 | 4545 | 76.72 |

## 2. Direct Mechanism Findings & Hypothesis Verdicts

1. **Stage B (`H_cal` & `0024/0053` Attribution — `FALSIFIED` for net HOTA gain)**:
   - Calibrating S12 `M1` across `0.55–0.68` yields at best `57.079` HOTA at `@0.68` (`+0.046` vs `M1@0.40 = 57.033`), still `-0.492` below `T2_TH040` (`57.572`).
   - Matching `dustbin_score = sim_threshold` in `B_M1_B1_CAL_062` unlocks non-zero dustbin rejections (`4`) and cuts `0024/0053` `post_switch_cross` (`3,893 -> 3,507`), but degrades overall `dev8` HOTA to `56.167`.
   - Targeted `0024/0053` anatomy audits prove **`8/10` (`80%`) onset switches happen when the true GT detection is absent (`true_det_absent`)** while `10/10` (`100%`) have `IoU >= 0.30` overlap competition.

2. **Stage C Controlled 3-Arm Ablation (`N1 -> N2 -> N3`)**:
   - **`N1_KIN_SCALE_ANCHORED`**: `center_loss` (`lambda=0.50`) tightly centered `delta_s` (`train |delta_s| = 0.00986`, `val = 0.08641`), yet `epoch20_val_th = 0.6804` (`>= 0.62`), and `@0.40` dropped to `54.461` HOTA (`@0.54`: `54.246`).
   - **`N2_KIN_OCC_DROP_SUP` (Major Positive Mechanism Delta)**: Targeted training-time `p_occ_drop=0.25` + `L_occ` rejection supervision recovers **`+2.733` HOTA (`57.194`)** and **`+4.471` AssA (`41.063`)** over `N1 @0.40` (lower `val_th = 0.6398`).
   - **`N3_DUAL_ANCHOR_CAMEL` (Stage C Winner)**: Adding permanent `K=4` early anchor 4D edge consistency + `p_hist_pollute=0.25` recent-slot corruption (`epoch20_val_th = 0.6209`) raises `@0.40` to **`57.519` HOTA** (`+0.325` over `N2`, `+0.486` over S12 `M1_TH040 = 57.033`; `@0.48`: `57.453` HOTA / `40.831` AssA / `4,367` FN). However, `N3` still finishes `-0.052` HOTA behind `T2_TH040` (`57.572`) and well below the `59.1` escalation bar.

3. **Stage D Synergy & Full-`dev8` Anatomy Replay (`N3_DUAL_ANCHOR_CAMEL @0.40`)**:
   - `N3_PLUS_B1_THCAL` (`@0.48`, dustbin=0.48): `57.467` HOTA / `40.680` AssA (`dustbin_rejected = 7`).

| Run / Condition | Clean Writes | Onset Cross | Post-Switch Cross | FP Writes | gap==1 Onsets | IoU>=0.30 Onsets | True Absent | True Present |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S11 `T_TWOSTAGE_TH050` | 42847 | 149 | 31791 | 1350 | 111 | 134 | 94 | 55 |
| S12 `M1_C_KIN_EDGE @0.40` | 35844 | 95 | 40570 | 1649 | 82 | 83 | 57 | 38 |
| S13 `N3_DUAL_ANCHOR_CAMEL @0.40` | 34578 | 104 | 42454 | 1094 | 80 | 97 | 76 | 28 |

## 3. Verification & Preserved Repair History

- All checks in `checks.json` passed (`checks["all_passed"] = True`), with `0.0` TrackEval CSV diff across 13 online evals (`104` per-seq rows), `0` nonfinite grad steps, `0.0` checkpoint reload diff, exact MOT match on anatomy replays, and `16` paired case bundles under `evidence/`.
- Preserved transparent failed/superseded attempts (excluded from authoritative claims, included in gross GPUh): `b_m1_cal_th055_v1`, `b_audit_0024_0053_m1_th068_v1`, `c_n1_scale_anchored_seed13_e20_v1` & `c_eval_n1_th040_v1` (`-inf*0=NaN` in zero `occ_loss` branch), `c_n1_scale_anchored_seed13_e20_v2`, and `c_n2_occ_drop_seed13_e20_v1` (`TkAgg` `SIGILL` at epoch 10 before `MPLBACKEND=Agg`).
