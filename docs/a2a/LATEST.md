# Latest Experiment Handoff (`a2a/LATEST`)

更新：2026-11-27。本文件是可更新导航；历史报告、协议和失败记录仍保留。

## 1. Latest Completed Task: S10 CAMELTrack 纯拒写基准分解与抗闭环污染实验 (`COMPLETED`)

- **任务书与协议：** [S10 具体执行与回传要求](S10_CAMEL_ANTICONTAMINATION_TASK.md)；[版本化协议](../../manifests/protocols/s10_camel_anticontamination_20261127.yaml)。
- **执行状态：** **COMPLETED（Stages A–E 在物理 GPU 1 上完成并停止在 Stage E，等待研究端审阅）**
- **权威报告：** [reports/feasibility/s10_camel_anticontamination_20261127/README.md](../../reports/feasibility/s10_camel_anticontamination_20261127/README.md)
- **资源与检查：** `CUDA_VISIBLE_DEVICES=1`（`NVIDIA GeForce RTX 4090 D`），物理 GPU 0 禁用且未触碰；总耗时 **`4.072 GPUh`**（预算 `<= 12.0h`），新增存储 **`4.27 GiB`**（预算 `<= 25.0 GiB`）；`checks.json` **14/14 通过**，TrackEval 导出与来源 CSV 误差 **`0.0`**；保留 3 项可追溯失败 attempt（`b_d2_skip_only_v1`、`c1_matched_iid_seed13_e20_v1`、`d_c3_soft_write_filter_w050_v1`）。

### Key Authoritative Findings (`motion_dev8`, `COMBINED_SEQ`, 7,565 frames)

1. **Stage B `D2_SKIP_ONLY`（`gt_injected_on_miss = 0`，严格隔离探针）vs `C0` 零重训标量门控：**
   - **`D2_SKIP_ONLY`**（跳过 `2,170` 跨 ID + `1,136` FP 写入、0 漏配 GT 注入）：`HOTA = 63.781`、`DetA = 81.568`、`AssA = 49.940`、`IDF1 = 63.186`、`MOTA = 88.958`、`IDSW = 936`。相对 S09 `T_TWOSTAGE_TH050`（`54.443 / 36.704`）净增 **`+9.338` HOTA（`+13.236` AssA）**；并与 S09 `D2 v3`（`81.907 / 80.063`）清晰剥离出 `+18.126` HOTA 属于 `else:` 分支对漏配/错配帧的 GT 重连贡献。
   - **`C0_SIMPLE_WRITE_GATE`（零重训）**：`tau=0.60` `HOTA = 48.838`（`AssA = 29.847`、`IDSW = 1,225`）；`tau=0.65` `HOTA = 43.509`（`AssA = 23.925`、`IDSW = 1,666`）。硬截断低分刷新会损伤快速运动正匹配的模板适应。
2. **Stage C 受控 20-epoch 训练（`train32`，C1 TE 初始化，seed 13，`@ 0.50` 在线闭环）：**
   - `C1_MATCHED_IID_SWAP`（标定 `p_iid = 0.076880`，相对偏差 `0.26%`）：`HOTA = 51.780`、`AssA = 33.256`、`IDF1 = 49.964`、`IDSW = 888`；
   - `C2_CONTIGUOUS_TAKEOVER`：`HOTA = 48.210`、`AssA = 28.936`、`IDF1 = 45.176`、`IDSW = 1,228`；
   - `C3_CONSISTENCY_GATED_TE`（ConsistencyGateHead 参数增量 `0.83%`）：`HOTA = 49.781`、`AssA = 30.765`、`IDF1 = 46.604`、`IDSW = 1,110`；虽相比 `C2` 回升 `+1.570` HOTA / `+1.829` AssA，但含合成污染三组均低于 S09 无污染 `T_TWOSTAGE_TH050`（`54.443`）。
3. **Stage D 软拒写滤网与真实闭环污染 AUROC：**
   - `C3_WITH_SOFT_WRITE_FILTER`（`w_last < 0.50`）：`HOTA = 49.313`、`AssA = 30.201`、`IDSW = 1,106`。
   - `T_TWOSTAGE_TH050` 真实闭环写入（`76,658` 次）：`C3` 全局跨 ID `AUROC = 0.5002`、`F1 = 0.0259`（成熟轨 `hits>=3` `0.5032`），尽管合成验证集 `val/cons_acc = 0.9896`。拆解显示 `31,791/31,940` 跨 ID 写入为切换后已自洽的漂移尾段（`AUROC = 0.4997`）；即便仅在 **`149` 例首次切换 onset** 上，`C3` 仅得 `AUROC = 0.6220`，亦大幅落后于 `1 - sim`（`AUROC = 0.8512`）。

### Decision, Limitations & Exact Next Action

- **Decision & Limitations**: **对当前合成替换假设（`C2` 连续接管训练 + `C3` 当前局部因果一致性头）为明确的负结果，不扩展 multi-seed 37/73，不碰官方 `val/test`。** 纯停写污染确有 `+9.338` HOTA（`+13.236` AssA）空间，但合成同帧替框污染训出的识别器存在显著 sim-to-real 失配，且短窗局部一致性难以检测漂移数帧后已内部自洽的错误轨迹。本轮仅单 seed (`13`)、复用 `motion_dev8`。
- **Exact Next Action**: 交由研究端审阅 [S10 报告包](../../reports/feasibility/s10_camel_anticontamination_20261127/README.md)（`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`learning_curves.csv`、`diagnostics.json`、`cases.jsonl`），基于 `D2_SKIP_ONLY`（`+9.34 HOTA`）、`149` 例 onset `1-sim AUROC=0.8512` 与 `C2/C3` 失败原因重新决定下一轮方向。工作树保留未提交、未推送。

---

## 2. Latest Completed: S09 CAMELTrack Threshold Calibration, Two-Stage Baseline & Failure Attribution (`COMPLETED`)

- **Protocol**: `s09_camel_calibration_and_attribution_20261126` (`v1`, [protocol manifest](../../manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml))
- **Execution Status**: **COMPLETED (Stages A–E finished on physical GPU 1; stopped at Stage E pending research review)**
- **Authoritative Report**: [reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md](../../reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md)
- **Compute & Storage**: `CUDA_VISIBLE_DEVICES=1` (`NVIDIA GeForce RTX 4090 D`), physical GPU 0 unused; total `8.486 GPUh` (B `0.495h`, C `7.477h`, D `0.514h`, limit `<= 16h`); extra off-repo storage `6.30 GiB` (limit `<= 40 GiB`, `1.3 TiB` free). All `checks.json` assertions pass (`13/13`, `0.0` TrackEval export max diff).

### Key Authoritative Findings (`motion_dev8`, `COMBINED_SEQ`, 7,565 frames)

1. **Online Closed-Loop Calibration & Two-Stage Baseline (`online_closed_loop`, GT-free)**:
   - Calibrating S08 `T_SCRATCH` (`@ 0.1`: HOTA `53.981`, AssA `36.096`, IDSW `1451`) to `0.50` improves HOTA modestly to `55.215` / AssA `37.900` / IDSW `1333` (`@ 0.519729`: `54.444` / `36.875` / `1412`), whereas `P_OFFICIAL @ 0.50` scores `63.874` / `49.399` / `606` (vs `66.993` / `54.516` / `638` `@ 0.1`). Threshold miscalibration is not the bottleneck.
   - `train32`-only Two-Stage (`C1` 3 single-cue TE pretrains + `C2` 20-epoch joint GAFFE, `val/sim_auroc = 0.99225`, `val/loss = 0.2391`) cuts `T_TWOSTAGE_TH050` (`0.50`) official IDSW nearly in half (**`786`** vs `1333`/`1451`), while global AssA stays **`36.704`** (`HOTA = 54.443`, `IDF1 = 53.629`, `MOTA = 90.373`). `@ 0.5460` (`T_TWOSTAGE_TH_COMP`): `51.712` HOTA / `33.319` AssA / `889` IDSW.
2. **D1 Similarity Margin Attribution (`786/786` & `1412/1412` 100% official IDSW coverage)**:
   - `T_TWOSTAGE_TH050`: **52.29%** (`411`) misassociation reversals (`margin < -0.05`), **25.95%** (`204`) high-score ties (`|margin| < 0.05`), **17.43%** (`137`) unmatched/new track, **4.33%** (`34`) Hungarian contention losses; `60.05%` after `gap > 1`, `76.08%` with competing GT `IoU >= 0.3`, median margin `-0.0755`.
3. **D2 Privileged Clean-History Probe (Strictly Isolated `privileged_clean_history_probe`, `v3`)**:
   - Preventing `1,108` cross-ID and `1,319` FP observations from contaminating tracklet galleries under strict 1-to-1 GT routing raises `T_TWOSTAGE_TH050` by **+43.359 AssA** (`36.704 -> 80.063`), **+38.524 IDF1** (`53.629 -> 92.153`), and **+27.465 HOTA** (`54.443 -> 81.907`), surpassing `P_OFFICIAL @ 0.1` (`54.516` AssA); `dancetrack0037` recovers AssA `21.796 -> 84.008` (`+62.212`), `dancetrack0044` recovers AssA `24.757 -> 78.349` (`+53.592`).
   - *(Preserved non-authoritative earlier attempts on disk: `d2_..._v1` multi-clone/shared-`age` artifact [`IDSW=2281`]; `d2_..._v2` `Tensor.age` `TypeError`.)*

### Decision, Limitations & Exact Next Action

- **Recommended S10 Direction**: Route primarily to **`B'` (anti-contamination / self-healing gallery update)**. **Explicit limitation**: `D2 v3` is a non-deployable GT-routed counterfactual on reused `motion_dev8` (`seed 13` only), and **`1,108` momentary single-frame cross-ID predictions (`830` CLEAR IDSWs)** still fire even from clean galleries (though clean galleries immediately self-heal on subsequent frames rather than cascading).
- **Exact Next Action**: Research side reconciles [S09 report bundle](../../reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md) (`aggregate_metrics.csv`, `per_sequence_metrics.csv`, `d1_idsw_margins_summary.json`, `cases.jsonl`, `diagnostics.json`) and freezes an explicit **S10** task & versioned protocol (`B'` primary). Working tree remains uncommitted and unpushed per handoff policy.

---

## 2. Completed / Stopped Historical Tasks

- **S08 (COMPLETED, superseded by S09 for threshold/two-stage analysis)**: [S08 Task](S08_CAMEL_BASELINE_TASK.md) | [S08 Report](../../reports/baselines/s08_camel_baseline_20261125/README.md) (`CONTINUE`, `T_SCRATCH @ 0.1` vs `P_OFFICIAL @ 0.1`).
- **2026-11-26 Research Review & Literature Audit**: [docs/research_status_review_20261126.md](../research_status_review_20261126.md).
- **Stopped Branches**: `P1`, `R1`, `R2`, `E3`, `E5`, `S03`, `S04`, `S06` (`STOP_S06_NO_SEEDS_37_73`; CPU postmortem in [S07 report](../../reports/diagnostics/s07_s06_checkpoint_postmortem_20261124/README.md)).
