# S09 Report: CAMELTrack Threshold Calibration, Two-Stage Baseline, and Failure Attribution

- **Protocol**: `s09_camel_calibration_and_attribution_20261126` (v1, protocol SHA256 `b586284b396c...`)
- **Status**: **COMPLETED** (Stages A–E passed within budget; GPU 0 untouched; **0.0** TrackEval export diff)
- **Physical GPU**: `CUDA_VISIBLE_DEVICES=1` (`NVIDIA GeForce RTX 4090 D`); total compute **8.486 GPUh** (B `0.495h`, C `7.477h`, D `0.514h`, budget `<= 16.0h`); extra storage **6.30 GiB** (`1.3 TiB` free).
- **Split & GT Boundary**: `motion_train32` (32 seqs, 34,231 frames) for C1/C2 training; `motion_dev8` (8 seqs, 7,565 frames) GT-free for B/C3/D1; **Stage D2** (`v3`) is strictly isolated as `privileged_clean_history_probe` and never mixed into online rankings. Official DanceTrack `val`/`test` untouched.

## 1. Online Closed-Loop Results (`motion_dev8`, official `COMBINED_SEQ`)

| Condition | Mode | Threshold | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P_OFFICIAL` *(S08 Ref)* | `s08_readonly_reference_th010` | `0.1000` | 66.993 | 82.524 | 54.516 | 71.763 | 91.591 | 638 | 1646 | 4536 | 1397 |
| `P_OFFICIAL_TH050` | `online_closed_loop` | `0.5000` | 63.874 | 82.657 | 49.399 | 64.332 | 90.771 | 606 | 1147 | 5732 | 1398 |
| `T_SCRATCH` *(S08 Ref)* | `s08_readonly_reference_th010` | `0.1000` | 53.981 | 81.018 | 36.096 | 56.695 | 90.621 | 1451 | 1721 | 4434 | 1428 |
| `T_SCRATCH_TH046` | `online_closed_loop` | `0.4600` | 53.348 | 81.014 | 35.260 | 55.168 | 90.692 | 1337 | 1573 | 4639 | 1445 |
| `T_SCRATCH_TH050` | `online_closed_loop` | `0.5000` | 55.215 | 80.690 | 37.900 | 56.204 | 90.158 | 1333 | 1448 | 5201 | 1490 |
| `T_SCRATCH_TH_COMP` | `online_closed_loop` | `0.5197` | 54.444 | 80.653 | 36.875 | 53.869 | 89.636 | 1412 | 1362 | 5631 | 1524 |
| `T_TWOSTAGE_TH050` | `online_closed_loop` | `0.5000` | 54.443 | 80.969 | 36.704 | 53.629 | 90.373 | 786 | 1406 | 5615 | 1332 |
| `T_TWOSTAGE_TH_COMP` | `online_closed_loop` | `0.5460` | 51.712 | 80.462 | 33.319 | 49.292 | 88.893 | 889 | 1138 | 6981 | 1313 |

### Privileged Offline Counterfactual (Isolated — Not Deployable)

| Condition | Mode | Threshold | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T_TWOSTAGE_TH050_PRIVILEGED_CLEAN_HISTORY` (`v3`) | `privileged_clean_history_probe` | `0.5000` | **81.907** | 83.835 | **80.063** | **92.153** | 91.673 | 830 | 1338 | 4585 | 1442 |

*(Note: Preserved earlier non-authoritative attempts `d2_..._v1` [shared uncopied `clean_det` / multi-clone `.age` aliasing on `dancetrack0037`] and `d2_..._v2` [`Tensor` `.age` TypeError] remain intact on data disk and are excluded from authoritative rankings.)*

## 2. Direct Answers to S09 Research Questions

1. **Threshold Calibration & Two-Stage Training (Stages B & C)**:
   - Calibrating `T_SCRATCH` from `0.10` to `0.50` improves HOTA modestly (`53.981 -> 55.215`, AssA `36.096 -> 37.900`, IDSW `1451 -> 1333`), while `T_SCRATCH_TH_COMP` (`0.519729`) scores `54.444` HOTA / `36.875` AssA. Threshold miscalibration is **not** the primary bottleneck (`P_OFFICIAL @ 0.50` drops to `63.874` from `66.993` @ `0.10` due to higher `FN`).
   - Strict `train32`-only Two-Stage CAMELTrack (`C1` 3 single-cue TE pretrains + `C2` 20 epochs joint GAFFE, `val/sim_auroc = 0.99225`, `val/loss = 0.2391`) cuts `T_TWOSTAGE_TH050` official `dev8` IDSW nearly in half (**`786`** vs `1333` for `T_SCRATCH_TH050` and `1451` for `T_SCRATCH @ 0.10`), yet global `AssA` stays **`36.704`** (`HOTA = 54.443`, `IDF1 = 53.629`).

2. **D1 Similarity Margin Attribution (100% Official IDSW Coverage)**:
   - **`T_TWOSTAGE_TH050` (`786/786` official IDSW)**:
     - Misassociation reversal (`s_pos < s_best_competing - 0.05`): **411 / 786 (52.29%)**
     - High-score ties (`|s_pos - s_best_competing| < 0.05`): **204 / 786 (25.95%)**
     - Unmatched / spawned new track: **137 / 786 (17.43%)**
     - Positive scored higher but lost Hungarian 1-to-1 contention: **34 / 786 (4.33%)**
     - Context: **60.05%** (`472/786`) follow `gap > 1`, **76.08%** (`598/786`) have competing GT `IoU >= 0.3`, median `margin = -0.0755`.
   - **`T_SCRATCH_TH_COMP` (`1412/1412`)**: reversals **57.72%** (`815`), ties **29.18%** (`412`), unmatched **11.76%** (`166`), Hungarian **1.35%** (`19`).

3. **D2 Privileged Clean-History Recovery & S10 Decision (`v3`)**:
   - Keeping tracklet galleries strictly 1-to-1 GT-clean (`1,108` cross-ID matches + `1,319` FP matches prevented from polluting galleries; `76,509` gallery updates across `77,689` predicted matches) elevates `T_TWOSTAGE_TH050` by **+43.359 AssA** (`36.704 -> 80.063`), **+38.524 IDF1** (`53.629 -> 92.153`), and **+27.465 HOTA** (`54.443 -> 81.907`) — well above even `P_OFFICIAL @ 0.10` (`54.516` AssA).
   - Hardest failure sequences recover dramatically:
     - `dancetrack0037`: AssA `21.796 -> 84.008` (**+62.212**), HOTA `44.271 -> 87.607`, IDF1 `37.438 -> 92.974`
     - `dancetrack0044`: AssA `24.757 -> 78.349` (**+53.592**), HOTA `44.318 -> 80.110`, IDF1 `44.968 -> 94.173`
   - **Routing for S10**: **Primary route `B'` (anti-contamination / self-healing gallery update)** is overwhelmingly supported as the dominant cause of `T_TWOSTAGE`'s closed-loop AssA/IDF1 collapse. Note honestly that **`1,108` single-frame cross-ID predictions (`830` CLEAR IDSWs)** still fire even from a clean gallery (`B'` prevents those momentary swaps from permanently poisoning future tracklet galleries, whereas suppressing the remaining `1,108` single-frame reversals is secondary).
