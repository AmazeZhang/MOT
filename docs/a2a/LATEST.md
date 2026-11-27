# Latest Experiment Handoff (`a2a/LATEST`)

更新：2026-11-27。本文件是可更新导航；历史报告、协议和失败记录仍保留。

## 1. Current Task: S10 CAMELTrack 纯拒写基准分解与抗闭环污染实验 (`NOT_RUN`)

- **任务书：** [S10 具体执行与回传要求](S10_CAMEL_ANTICONTAMINATION_TASK.md)；[版本化协议](../../manifests/protocols/s10_camel_anticontamination_20261127.yaml)。前置复盘：[2026-11-27 S09 审阅复盘与设计说明](../research_status_review_20261127.md)。
- **执行状态：** `DESIGN_READY_RUNTIME_LOCK_PENDING` / `NOT_RUN`（尚未交给服务器执行）。用户交给服务器后，Stage A–E 在前置门通过且预算内自动顺序执行；停在 Stage E 或阻断/预算上限处回传。
- **核心问题：**
  1. 先在 Stage B 用 `D2_SKIP_ONLY` 剥离 S09 D2 `v3` 中“纯拒写污染”与“GT 补写漏配帧（`else:` 分支）”各自贡献，并用零重训 `C0_SIMPLE_WRITE_GATE`（`tau_write ∈ {0.60, 0.65}`）建立强简单对照；
  2. 在严格同 `motion_train32` / C1 TE 初始化 / 20 epochs / seed 13 / `sim_threshold=0.50` 下，对比等噪声量独立随机交换（`C1_MATCHED_IID_SWAP`）、连续异主接管污染训练（`C2_CONTIGUOUS_TAKEOVER`）与多线索历史一致性门控聚合（`C3_CONSISTENCY_GATED_TE` + 真实闭环污染检出 AUROC）。
- **预算与硬边界：** 物理 GPU 1–7 单卡（GPU 0 禁用）；总 GPU 预算 `≤ 12.0 GPUh`、墙钟 `≤ 36h`、新增存储 `≤ 25 GiB`。不运行多 seed 37/73、官方 `val`/`test`、SportsMOT、A'/C' 模块或自动 `git commit/push`。
- **回传目录（待生成）：** `reports/feasibility/s10_camel_anticontamination_20261127/`（当前尚未产生）。

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
