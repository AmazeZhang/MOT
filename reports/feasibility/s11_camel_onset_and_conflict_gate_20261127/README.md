# S11 Report: CAMELTrack Low-Threshold Calibration, 149-Onset Anatomy & Conflict-Gated Write Probe

- **Protocol**: `s11_camel_onset_and_conflict_gate_20261127` (v1, SHA256 `bcf8ec627d54...`)
- **Status**: **COMPLETED** (0 training runs; Stages A–E finished within budget on `CUDA_VISIBLE_DEVICES=1`; GPU 0 untouched; **0.0** TrackEval export diff)
- **Compute & Storage**: `NVIDIA GeForce RTX 4090 D` (physical GPU 1 only); total **1.008 GPUh** (B `0.353h`, C `0.132h`, D `0.523h`, budget `<= 2.0h`); extra storage **0.086 GiB** (budget `<= 5.0 GiB`).
- **Scope & GT Boundary**: Strict zero-retrain reuse of S09 `T_TWOSTAGE` ckpt (`20be35af...`, `seed=13`) on GT-free `motion_dev8` (`7,565` frames). Stage C GT is used solely for post-hoc offline event labeling (`exact_mot_sha256_match_vs_s09_t_twostage_th050 = True`). Official DanceTrack `val`/`test` untouched.

## 1. Closed-Loop Results on `motion_dev8` (Official `COMBINED_SEQ`)

| Condition | Group | Threshold | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`T2_TH040`** | Stage B | `0.40` | **57.572** | **81.394** | **40.845** | **60.117** | **91.184** | 832 | 1676 | **4642** | 1385 |
| `T2_TH030` | Stage B | `0.30` | 56.595 | 81.295 | 39.537 | 59.679 | 91.092 | 876 | 1689 | 4659 | 1388 |
| `T2_TH010` | Stage B | `0.10` | 56.594 | 81.294 | 39.537 | 59.679 | 91.090 | 877 | 1690 | 4659 | 1388 |
| `T_TWOSTAGE_TH050` *(S09 Ref)* | Baseline | `0.50` | 54.443 | 80.969 | 36.704 | 53.629 | 90.373 | **786** | 1406 | 5615 | **1332** |
| `T_TWOSTAGE_TH_COMP` *(S09 Ref)* | Baseline | `0.5460` | 51.712 | 80.462 | 33.319 | 49.292 | 88.893 | 889 | 1138 | 6981 | 1313 |
| `G1_CONFLICT_IOU030_TAU065` | Stage D | `0.50` | 48.686 | 79.958 | 29.715 | 45.128 | 87.084 | 1338 | 1082 | 8055 | 1524 |
| `C0_SIMPLE_WRITE_GATE_TAU060` *(S10 Ref)* | Ref Gate | `0.50` | 48.838 | 80.090 | 29.847 | 44.474 | 87.603 | 1225 | 1138 | 7691 | 1534 |
| `G2_MARGIN_M010_TAU070` | Stage D | `0.50` | 48.139 | 79.451 | 29.249 | 42.769 | 85.960 | 1480 | 1029 | 8877 | 1626 |
| `G3_HYBRID_GATE_NO_PIN` | Stage D | `0.50` | 44.442 | 78.911 | 25.088 | 38.553 | 84.842 | 1734 | 971 | 9588 | 1651 |
| `C0_SIMPLE_WRITE_GATE_TAU065` *(S10 Ref)* | Ref Gate | `0.50` | 43.509 | 79.289 | 23.925 | 36.904 | 85.507 | 1666 | 994 | 9094 | 1622 |
| `G4_HYBRID_GATE_EARLY_PIN5` | Stage D | `0.50` | 43.380 | 78.891 | 23.918 | 37.064 | 84.742 | 1698 | **917** | 9759 | 1658 |

## 2. Direct Answers to S11 Research Questions

1. **Stage B Lower-Threshold Calibration (`0.10 / 0.30 / 0.40` vs `0.50`)**:
   - Lowering `sim_threshold` to `0.40` (`T2_TH040`) recovers **`973` FN** (`5615 -> 4642`) while raising IDSW only modestly (`786 -> 832`), lifting **HOTA `+3.129` (`54.443 -> 57.572`), AssA `+4.141` (`36.704 -> 40.845`), and IDF1 `+6.488` (`53.629 -> 60.117`)**. Lowering further to `0.30`/`0.10` saturates FN recovery (`4659`) and slightly worsens IDSW (`876/877`, HOTA `56.595`).

2. **Stage C 149-Onset Anatomy & Early Anchor Drift Signal (`c_onset_anatomy_v2`)**:
   - **Bit-exact baseline replay**: All 8 sequence MOT files match S09 `T_TWOSTAGE_TH050` identically; `76,658` writes = `42,847` clean, `149` onset_cross, `31,791` post_switch_cross, `1,350` fp, `521` unanchored.
   - **149 Onset Mechanism Distribution**:
     - Continuous frame switches (`gap == 1`): **`111 / 149` (`74.5%`)** vs `gap > 1`: `38 / 149` (`25.5%`).
     - Spatial overlap conflict (`IoU >= 0.30`): **`134 / 149` (`89.9%`)** (median `max_conflict_iou = 0.556`).
     - Low two-sided assignment margin (`< 0.10`): **`111 / 149` (`74.5%`)** (median `0.0456`); `2-cycle` mutual swaps: **`34 / 149` (`22.8%`)**.
     - True target present in frame: `55 / 149` (`34` strong reversals `<-0.05`, `16` ties, `5` Hungarian losses); true target absent: `94 / 149`.
   - **Offline Predictors & Early-Anchor Separation**:
     - On `onset` (`149`) vs `clean`: **`neg_assignment_margin` AUROC = `0.9432`** (`73.8%` TPR at `5%` clean FPR), followed by `max_conflict_iou` (`0.8666`) and `1 - sim` (`0.8512`).
     - On `post_switch` (`31,791`) vs `clean`: **`anchor_reid_dist` AUROC = `0.7062`** (`0.7058` on all `31,940` cross-ID), breaking S10's self-consistency trap compared to `recent_reid_dist` (`0.5251`) and `1 - sim` (`0.5182`).

3. **Stage D Zero-Retrain Conflict-Gated Write & Early Pinning (`G1`–`G4` -> S12 Decision = Case 2)**:
   - Adding a spatial contention precondition (`G1`, `HOTA = 48.686`, `FN = 8,055`, `4,733` skipped) beats unconditional `C0_TAU065` (`43.509`, `FN = 9,094`, `7,475` skipped) by **`+5.177` HOTA (`-1,039` FN, `-328` IDSW)**.
   - However, **ALL `G1`–`G4` still underperform `T_TWOSTAGE_TH050` (`54.443`)** because spatial/margin conflict occurs on `~42%` of mature clean frames in crowded DanceTrack (`29,714 / 71,986` in `G1`), triggering `4,733`–`7,857` write skips (`>98%` false skips vs `~82` caught onsets). Starving galleries during crowded overlaps causes `FN` to climb to `8,055`–`9,759` (`IDSW` `1,338`–`1,734`), and pinning 5 stale birth slots (`G4`, `43.380`) hurts an additional `-1.062` HOTA vs `G3` (`44.442`).
   - **S12 Decision (Case 2 confirmed)**: Write-side gating alone cannot cure `T_TWOSTAGE`; S12 must move directly to association scoring and assignment (**`C'`** GAFFE pairwise motion/occlusion edge bias + **`A'`** margin/2-cycle/deferred disambiguation, anchored at calibrated threshold regime `0.40`).

## 3. Preserved Failed Attempt

- `c_onset_anatomy_v1`: Pre-tracking `AttributeError` (`name` attribute missing before adding `__getattr__`), superseded by `c_onset_anatomy_v2`.
