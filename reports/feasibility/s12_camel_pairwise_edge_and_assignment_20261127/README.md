# S12 Report: CAMELTrack Pairwise Edge Bias (C') and Assignment Alignment (A')

- **Protocol**: `s12_camel_pairwise_edge_and_assignment_20261127` (v1, SHA256 `8095ff00d557...`)
- **Status**: **COMPLETED** (Stages A–E finished; `CUDA_VISIBLE_DEVICES=1` only; GPU 0 untouched; **0.0** TrackEval export diff; **0.0** Stage D anatomy MOT diff)
- **Compute & Storage**: `NVIDIA GeForce RTX 4090 D` (physical GPU 1); total **8.871 GPUh** (B `0.313h`, C `7.936h` [train `6.810h` + eval `1.126h`], D `0.623h`; total budget `<= 9.0h`, Stage C exceeded `6.5h` sub-estimate due to 20-epoch joint speed `~1.70h/arm`); extra storage **4.097 GiB** (`<= 15.0 GiB`).
- **Stage C Winner**: **`M1_C_KIN_EDGE`** (highest `dev8` `COMBINED_SEQ` HOTA `@0.40` = **`57.033`**, AssA = **`40.275`**).

## 1. Closed-Loop Results on `motion_dev8` (`COMBINED_SEQ`)

| Condition | Thresh | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | ms/f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T2_TH040` *(S11 Ref)* | `0.40` | 57.572 | 81.394 | 40.845 | 60.117 | 91.184 | 832 | 1676 | 4642 | n/a |
| `T_TWOSTAGE_TH050` *(S09 Ref)* | `0.50` | 54.443 | 80.969 | 36.704 | 53.629 | 90.373 | 786 | 1406 | 5615 | n/a |
| `D2_SKIP_ONLY` *(S10 Oracle Ref)* | `0.50` | 63.781 | 81.568 | 49.940 | 63.186 | 88.958 | 936 | 1133 | 6886 | n/a |
| `B1_A_DUAL_SOFTMAX_TH040` | `0.40` | 55.919 | 80.999 | 38.718 | 58.154 | 90.967 | 931 | 1682 | 4713 | 58.73 |
| `B2_A_MARGIN_2CYCLE_DEFER_TH040` | `0.40` | 54.356 | 81.950 | 36.173 | 52.843 | 90.163 | 1218 | 1521 | 5239 | 85.62 |
| `M0_NODE_CAP_CTRL` (@0.40) | `0.40` | 54.179 | 81.299 | 36.244 | 55.354 | 90.975 | 879 | 1707 | 4733 | 65.24 |
| `M0_NODE_CAP_CTRL` (@0.50) | `0.50` | 53.577 | 81.252 | 35.432 | 53.416 | 90.499 | 882 | 1499 | 5324 | 64.05 |
| **`M1_C_KIN_EDGE`** (@0.40, *Stage C Best*) | `0.40` | 57.033 | 80.990 | 40.275 | 60.417 | 91.079 | 957 | 1723 | 4555 | 64.93 |
| `M1_C_KIN_EDGE` (@0.50) | `0.50` | 57.033 | 80.986 | 40.276 | 60.418 | 91.078 | 956 | 1723 | 4557 | 64.34 |
| `M2_C_KIN_DOA_EDGE` (@0.40) | `0.40` | 56.670 | 81.051 | 39.762 | 59.318 | 91.218 | 903 | 1690 | 4529 | 65.77 |
| `M2_C_KIN_DOA_EDGE` (@0.50) | `0.50` | 56.670 | 81.051 | 39.762 | 59.318 | 91.218 | 903 | 1690 | 4529 | 65.44 |
| `M3_C_PRIME_PLUS_A_LOSS` (@0.40) | `0.40` | 52.025 | 80.866 | 33.586 | 54.434 | 90.896 | 954 | 1685 | 4744 | 64.52 |
| `M3_C_PRIME_PLUS_A_LOSS` (@0.50) | `0.50` | 52.545 | 81.015 | 34.196 | 53.687 | 90.782 | 932 | 1632 | 4912 | 63.53 |
| `D_M1_PLUS_B1_TH040` | `0.40` | 57.009 | 80.857 | 40.322 | 59.952 | 91.064 | 980 | 1701 | 4566 | 66.56 |
| `D_M1_PLUS_B2_TH040` | `0.40` | 52.627 | 81.138 | 34.239 | 54.849 | 91.023 | 1002 | 1729 | 4549 | 86.65 |

## 2. Direct Answers to S12 Research Questions

1. **Q1 — Zero-Retrain `A'_infer` (`B1` Dual-Softmax & `B2` 2-cycle/defer vs `T2_TH040 = 57.572`)**:
   - **Negative**: `B1_A_DUAL_SOFTMAX_TH040` drops to **55.919** (`-1.653` HOTA, AssA `38.718`, IDSW `931`, `10` dustbin rejections); `B2_A_MARGIN_2CYCLE_DEFER_TH040` drops to **54.356** (`-3.215` HOTA, IDSW `1218`, FN `5239`, `220` 2-cycle swaps, `1,244` deferrals). Post-hoc assignment tweaks on S09 `T_TWOSTAGE` scores hurt both HOTA and IDSW.

2. **Q2 — Pairwise Edge Bias `M1` & `M2` vs Capacity-Matched Node Control `M0_NODE_CAP_CTRL`**:
   - **Causal Pairwise Edge Bias Validated vs Capacity Control**: With fewer extra parameters (`M1: 146,305` and `M2: 151,105` vs `M0: 266,240`), **`M1_C_KIN_EDGE` gains `+2.854` HOTA (`57.033` vs `54.179`), `+4.031` AssA (`40.275` vs `36.244`), and `+5.063` IDF1 (`60.417` vs `55.354`)** over `M0` (`+2.491` HOTA for `M2`).
   - **DOA Incremental Effect (`M2 - M1`)**: Adding 13D DOA occlusion + self spatial adjacency (`M2`, `56.670`) reduces IDSW (`903` vs `957`) and FN (`4,529` vs `4,555`) slightly vs `M1`, but is **`-0.363` HOTA / `-0.513` AssA** lower overall.
   - **Comparison to S11 `T2_TH040` (`57.572`)**: While `M1` beats `T_TWOSTAGE_TH050` (`54.443`, `+2.590` HOTA) and edges out `T2_TH040` in `IDF1` (`60.417` vs `60.117`) and `FN` (`4,555` vs `4,642`), **`M1` still trails S11 `T2_TH040` by `-0.539` HOTA (`-0.570` AssA, `+125` IDSW)** — and `M0` trails `T2_TH040` by `-3.393` HOTA.

3. **Q3 — `M3_C_PRIME_PLUS_A_LOSS` (`A'_train`) and Stage D (`M1 + B1/B2`) Synergy**:
   - **`A'_train` Loss (`M3`) Negative**: Adding `L_bidir_margin + L_unmatched_rej` degrades `@0.40` HOTA to **`52.025`** (`-4.645` vs `M2`, AssA `33.586`). `train/*` losses and all `8,000` training gradient steps stayed strictly finite (`nonfinite_grad_steps=0`), while `val/unmatched_rej_loss` logged `nan` on epochs [2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20] without impacting optimization.
   - **Stage D Synergy (`M1 + B1`, `M1 + B2`)**: `D_M1_PLUS_B1_TH040` achieves **57.009** HOTA (`-0.024` vs `M1`), and `D_M1_PLUS_B2_TH040` achieves **52.627** HOTA (`-4.406` vs `M1`).

4. **Q4 — Stage D Onset & Contamination Anatomy (`M1_C_KIN_EDGE` vs S11 `T_TWOSTAGE_TH050`)**:

| Run / Condition | Clean Writes | Onset Cross | Post-Switch Cross | FP Writes | gap==1 Onsets | IoU>=0.30 Onsets | True Absent | True Present | 2-Cycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S11 `T_TWOSTAGE_TH050` | 42847 | 149 | 31791 | 1350 | 111 | 134 | 94 | 55 | 34 |
| S12 `M1_C_KIN_EDGE @0.50` | 35844 | 95 | 40568 | 1649 | 82 | 83 | 57 | 38 | 20 |
| S12 `M1_C_KIN_EDGE @0.40` | 35844 | 95 | 40570 | 1649 | 82 | 83 | 57 | 38 | 20 |

## 3. Verification Status & Artifacts

- All checks in `checks.json` passed (`checks["all_passed"] = True`), including `0.0` TrackEval preview diff across all `12` online runs, `0.0` checkpoint reload diff and `0` non-finite training gradient steps across all `4` trained arms, bit-exact MOT match on both Stage D anatomy replays, and `16` paired case files under `evidence/`.
