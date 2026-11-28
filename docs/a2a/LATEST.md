# Latest Experiment Handoff (`a2a/LATEST`)

更新：2026-11-28。本文件是可更新导航；历史报告、协议和失败记录仍保留。

## 0. Current Active Task To Execute: **NONE (`S13 COMPLETED` — Stopped at Stage E; Verdict = `CASE_3_STOP_CAMEL_LOCAL_PATCH_AND_PIVOT`)**

- **当前状态：** **S13（`s13_camel_calibration_and_dual_memory_20261128`）已于 2026-11-28 在物理 GPU 1 上完成 Stages A–E 全阶段并停止在 Stage E，等待研究端审阅与架构路线切换决策。** 工作树保持未提交、未推送。
- **毕业论文核心晋级/止损裁决：** **`CASE_3_STOP_CAMEL_LOCAL_PATCH_AND_PIVOT`** — Stage C 最优组 `N3_DUAL_ANCHOR_CAMEL_TH040`（`57.519` HOTA / `40.787` AssA，`dev8` 相对 `T2_TH040` `4/0/4`，`post_switch_cross = 42,454`）未能超越无新增头基线 `T2_TH040`（`57.572`）且未达 `59.1` 晋级门；依 [AGENTS.md](../../AGENTS.md) 终止 CAMEL 局部边/槽记忆修补分支，不扩展 `37/73` 或官方 `val`/`test`，交由研究端决策转向 **Route B（轨迹级交叉后漂移检测与全局图重连）** 或 **Route C（更高天花板骨干如 MOTIP2 / End-to-End）**。

---

## 1. Latest Completed Task: S13 CAMELTrack 标尺校准、缺检拒识与早锚双时间尺度抗污染记忆 (`K-AnchorCAMEL`) (`COMPLETED`)

- **任务书与协议：** [S13 任务书](S13_CAMEL_CALIBRATION_AND_DUAL_MEMORY_TASK.md)；[版本化协议](../../manifests/protocols/s13_camel_calibration_and_dual_memory_20261128.yaml)。
- **执行状态：** **COMPLETED（Stages A–E 在物理 GPU 1 上顺序完成并停止在 Stage E；未执行 `git commit` / `git push`）**
- **权威报告：** [reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/README.md](../../reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/README.md)
- **资源与检查：** `CUDA_VISIBLE_DEVICES=1`（`NVIDIA GeForce RTX 4090 D`），物理 GPU 0 禁用且未触碰；权威净耗时 **`7.264 GPUh`**（`<= 8.5h` 预算：B `0.909h`、C `6.028h`、D `0.328h`），含透明保留之失败/重跑维修 `v1/v2` 记录的毛耗时 **`10.011 GPUh`**；额外数据盘占用 **`5.063 GiB`**（`<= 15.0 GiB`）；`checks.json` **10/10 全通过**（`24,957` 轨迹 `monotone_violations = 0`、`nonfinite_grad_steps = 0`、`reload_diff = 0.0`、936 行曲线 `0` NaN、TrackEval 导出差 `0.0`、Stage B/D 解剖 MOT 逐字节/双射一致、`16` 例配对样本、[`test_s13_camel.py`](../../tests/test_s13_camel.py) `3 passed`）。

### Key Authoritative Findings (`motion_dev8`, `COMBINED_SEQ`, 7,565 frames)

1. **Stage B（零重训 S12 `M1` 标尺扫阈与 `0024/0053` 定向审计，`H_cal` 证伪）：**
   - `M1` `@ {0.55, 0.60, 0.64, 0.68}` 最高仅在 **`B_M1_CAL_068`**（`@0.68`）取得 **`57.079` HOTA / `40.270` AssA / `59.972` IDF1 / `945` IDSW**（较 `M1@0.40 = 57.033` 仅 `+0.046`，仍低于 `T2_TH040 = 57.572` `-0.492` HOTA）。
   - 同步校准 `dustbin=sim_threshold` 组合 `B1` 在 `B_M1_B1_CAL_062` 首次激活拒识（`4` 次）并将 `0024/0053` 后切换尾段从 `3,893` 压降至 `3,507`（`clean 6,423 -> 6,809`），但总体闭环 HOTA 跌至 `56.167`（`B_M1_B1_CAL_058` `55.953`、`0` 拒识）。两例 `0024/0053` 定向解剖确认 **`8/10` (`80%`) onset 发生于真目标缺检瞬间（`true_det_absent`）** 且 `10/10` 存在 `IoU >= 0.30` 竞争。
2. **Stage C（同 S09 C1 TE 初始化、20ep `motion_train32` 受控三联组消融，`seed=13`、`last.ckpt` @ `0.40` & `th_cal`）：**
   - `N1_KIN_SCALE_ANCHORED`（`146,305` 新增参数）：`center_loss` 有效压低 `delta_s`（train `|delta_s| = 0.00986`、val `0.08641`），但 `epoch20_val_th = 0.6804`（`th_cal = 0.54`），闭环跌至 `@0.40` `54.461` HOTA / `36.593` AssA（`@0.54`: `54.246`）。
   - `N2_KIN_OCC_DROP_SUP`（`146,305` 新增参数，`train/occ_drop_count = 38.39`，`val_th = 0.6398`，`th_cal = 0.50`）：相对 `N1 @0.40` 净增 **`+2.733` HOTA / `+4.471` AssA**，取得 `@0.40` **`57.194` HOTA / `41.063` AssA / `60.528` IDF1**（逐视频 `6/0/2` 胜 `N1`；`@0.50`: `55.470`）。
   - **`N3_DUAL_ANCHOR_CAMEL`**（`147,073` 新增参数，永久 `K=4` 早锚 4D 一致性 + `p_hist_pollute=0.25`，`train/hist_pollute_count = 62.09`，`val_th = 0.6209`，`th_cal = 0.48`，**Stage C 获胜组**）：`@0.40` 取得 **`57.519` HOTA / `81.365` DetA / `40.787` AssA / `59.550` IDF1 / `1,187` IDSW / `4,485` FN / `75.84 ms/f`**（再增 `+0.325` vs `N2`、`+0.486` vs S12 `M1_TH040` [`5/0/3`]）；`@0.48` 取得 `57.453` HOTA / `40.831` AssA / `59.674` IDF1 / `4,367` FN（逐视频 `5/0/3` 胜 `T2_TH040`）。总体较 `T2_TH040` 仍微差 `-0.052` HOTA（`4/0/4`）。
3. **Stage D（`N3+B1` 协同与全 `dev8` 解剖回放）：**
   - `N3_PLUS_B1_THCAL`（`@0.48`、`dustbin=0.48`）：`57.467` HOTA / `40.680` AssA / `60.405` IDF1 / `1,114` IDSW（`dustbin_rejected = 7`）。
   - 全 `dev8` 解剖回放（`d_winner_anatomy_dev8_v1`，与在线 MOT `8/8` 逐字节一致）：`clean_match = 34,578`、`onset_cross = 104`（`76` 例真目标缺检、`28` 例真目标在场）、`post_switch_cross = 42,454`（高于 S11 `31,791` 与 S12 `40,570`）、`fp_pollution = 1,094`（低于 S11 `1,350` 与 S12 `1,649`）。

### Decision, Limitations & Exact Next Action

- **Decision & Limitations**: **`CASE_3_STOP_CAMEL_LOCAL_PATCH_AND_PIVOT`** — 内部消融清晰证实“遮挡缺检监督 `N2` (`+2.733`) + 永久早锚双时间尺度记忆 `N3` (`+0.325`)”有效，但 `K-AnchorCAMEL`（`57.519` HOTA）仍未超过无新增头基线 `T2_TH040`（`57.572`）且后切换尾段扩至 `42,454`，距 `59.1` 晋级门差 `-1.581` HOTA。依 1–2 轮硬止损规则终止 CAMEL 局部修补路线，不扩 multi-seed `37/73` 或官方 `val`/`test`。局限性：仅单 seed (`13`)、复用 `motion_dev8`；`S13CAMELTrackWrapper.stats["tracks_with_early_anchors"]` 汇总计数器未自增，帧级已实际执行 `_record_early_anchor` / `_build_anchor_override`；保留 `b_m1_cal_th055_v1`、`b_audit_0024_0053_m1_th068_v1`、`c_n1_*_v1/v2`、`c_n2_*_v1` 失败尝试。
- **Exact Next Action**: 交由研究端审阅 [S13 报告包](../../reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/README.md)（`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`s12_reference_comparison.csv`、`learning_curves.csv`、`diagnostics.json`、`cases.jsonl`）并立项 Route B（轨迹级漂移检测与全局重连）或 Route C（更高天花板端到端/序列关联模型）。工作树保持未提交、未推送。

---

## 2. Earlier Completed Task: S12 CAMELTrack 成对运动/方向遮挡边偏置 (`C'`) 与分配对齐 (`A'`) (`COMPLETED`)

- **任务书与协议：** [S12 具体执行与回传要求](S12_CAMEL_PAIRWISE_EDGE_AND_ASSIGNMENT_TASK.md)；[版本化协议](../../manifests/protocols/s12_camel_pairwise_edge_and_assignment_20261127.yaml)。
- **执行状态：** **COMPLETED（Stages A–E 在物理 GPU 1 上顺序完成并停止在 Stage E，等待研究端审阅与 S13 裁决）**
- **权威报告：** [reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/README.md](../../reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/README.md)
- **资源与检查：** `CUDA_VISIBLE_DEVICES=1`（`NVIDIA GeForce RTX 4090 D`），物理 GPU 0 禁用且未触碰；总耗时 **`8.871 GPUh`**（预算上限 `<= 9.0h`；B `0.313h`、C `7.936h` [20ep 四臂训练 `6.810h` + 8 次评测 `1.126h`；高于阶段子估计 `6.5h`]、D `0.623h`），新增数据盘存储 **`4.097 GiB`**（预算 `<= 15.0 GiB`）；`checks.json` **10/10 全通过**（`nonfinite_grad_steps=0`、`reload_diff=0.0`、TrackEval 导出差 `0.0`、Stage D MOT 逐字节一致、`16` 例配对证据、[`test_s12_camel.py`](../../tests/test_s12_camel.py) `3 passed`）。

### Key Authoritative Findings (`motion_dev8`, `COMBINED_SEQ`, 7,565 frames)

1. **Stage B（零重训 `A'_infer` 复用 S09 `T_TWOSTAGE` @ `0.40` vs S11 `T2_TH040` = `57.572` HOTA / `40.845` AssA / `60.117` IDF1 / `832` IDSW）：**
   - `B1_A_DUAL_SOFTMAX_TH040`：`55.919` HOTA / `38.718` AssA / `58.154` IDF1 / `931` IDSW / `4,713` FN（`10` dustbin 拒识，`-1.653` HOTA）；
   - `B2_A_MARGIN_2CYCLE_DEFER_TH040`：`54.356` HOTA / `36.173` AssA / `52.843` IDF1 / `1,218` IDSW / `5,239` FN（`220` 次 `2-cycle` 互换、`1,244` 次暂缓，`-3.215` HOTA）。纯推理分配改法未改善、反而损伤基线。
2. **Stage C（同 S09 C1 TE 初始化、20ep `motion_train32` 受控四联组，`seed=13`、`last.ckpt` @ `0.40` 主基点 / `0.50` 参照）：**
   - `M0_NODE_CAP_CTRL`（`266,240` 新增参数）：`@0.40` `54.179` HOTA / `36.244` AssA / `55.354` IDF1 / `879` IDSW / `4,733` FN（`@0.50` `53.577`）。
   - **`M1_C_KIN_EDGE`（`146,305` 新增参数，Stage C 获胜组）**：`@0.40` **`57.033`** HOTA / **`40.275`** AssA / **`60.417`** IDF1 / `957` IDSW / `4,555` FN（`@0.50` `57.033` HOTA / `40.276` AssA / `956` IDSW）。相对同初始化容量对照组 `M0` 净增 **`+2.854` HOTA / `+4.031` AssA / `+5.063` IDF1**，显著优于 `T_TWOSTAGE_TH050`（`54.443`, `+2.590`）；但在 `motion_dev8` 上总体仍落后无新增边头的 `T2_TH040`（`57.572`）`-0.539` HOTA（`+125` IDSW，逐视频胜平负 `3/0/5`）。
   - `M2_C_KIN_DOA_EDGE`（`151,105` 新增参数）：`@0.40` `56.670` HOTA / `39.762` AssA / `59.318` IDF1 / `903` IDSW / `4,529` FN（`@0.50` 相同）；DOA 略降 IDSW（`-54`）与 FN（`-26`），总体 `-0.363` HOTA vs `M1`。
   - `M3_C_PRIME_PLUS_A_LOSS`（`151,105` 新增参数）：`@0.40` `52.025` HOTA / `33.586` AssA / `54.434` IDF1 / `954` IDSW / `4,744` FN（`@0.50` `52.545`），较 `M2` 退化 `-4.645` HOTA（`train/*` 全有限，仅个别 val epoch `val/unmatched_rej_loss=nan`）。
3. **Stage D（`M1_C_KIN_EDGE` 协同与 S11 同口径 Onset 解剖回放）：**
   - `D_M1_PLUS_B1_TH040`：`57.009` HOTA (`-0.024` vs `M1`)；`D_M1_PLUS_B2_TH040`：`52.627` HOTA (`-4.406` vs `M1`，`257` 次 `2-cycle` 互换严重损伤精度)。
   - Onset 解剖（`@0.50` 与 `@0.40` 均与在线 MOT 逐字节一致）：`onset_cross` 相比 S11 `149` 例降低 **`-36.2%` 至 `95` 例**（`gap==1`: `111->82`、`IoU>=0.30`: `134->83`、真目标缺检误抓: `94->57`、`2-cycle`: `34->20`），但同时 `post_switch_cross` 从 `31,791` 增至 `40,568/40,570`（`clean_match` `42,847->35,844`），残余切换形成更长持续污染段，未收敛至 `D2_SKIP_ONLY`（`63.781` HOTA）。

### Decision, Limitations & Exact Next Action

- **Decision & Limitations**: `C'` 成对因果运动偏置相对同容量节点对照 `M0` 成效明确（`+2.854` HOTA），并真实压降 `36.2%` onset 切换；但整体闭环暂未能超越未加边头基线 `T2_TH040`（`57.033` vs `57.572`），且 `A'_infer`（`B1/B2`）与当前 `A'_train`（`M3`）均呈负面或无协同增益。仅单 seed (`13`)、复用 `motion_dev8`；不扩 multi-seed 37/73 或官方 `val`/`test`。
- **Exact Next Action**: 交由研究端审阅 [S12 报告包](../../reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/README.md)（`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`learning_curves.csv`、`diagnostics.json`、`cases.jsonl`）并决定 S13 研究方向。工作树保持未提交、未推送。

---

## 3. Earlier Completed Task: S11 CAMELTrack 低阈值标定、149 例 Onset 解剖与冲突感知拒写无重训验证 (`COMPLETED`)

- **任务书与协议：** [S11 具体执行与回传要求](S11_CAMEL_ONSET_AND_CONFLICT_GATE_TASK.md)；[版本化协议](../../manifests/protocols/s11_camel_onset_and_conflict_gate_20261127.yaml)。
- **执行状态：** **COMPLETED（Stages A–E 在物理 GPU 1 上完成，已由研究端审阅并推进至 S12）**
- **权威报告：** [reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md](../../reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md)
- **资源与检查：** 0 次训练、`CUDA_VISIBLE_DEVICES=1`（`NVIDIA GeForce RTX 4090 D`），物理 GPU 0 禁用且未触碰；总耗时 **`1.008 GPUh`**（预算 `<= 2.0h`），新增存储 **`0.086 GiB`**（预算 `<= 5.0 GiB`）；`checks.json` **10/10 通过**，TrackEval 导出误差 `0.0`，Stage C MOT 与 S09 基准逐字节一致；保留 1 项预运行失败 record `c_onset_anatomy_v1`。

### Key Authoritative Findings (`motion_dev8`, `COMBINED_SEQ`, 7,565 frames)

1. **Stage B 低阈值补充标定（`T_TWOSTAGE` @ `0.10 / 0.30 / 0.40` vs S09 `0.50`）：**
   - **`T2_TH040`** (`0.40`) 取得 **`57.572` HOTA / `40.845` AssA / `60.117` IDF1 / `91.184` MOTA / `832` IDSW / `4,642` FN** — 相对 `0.50`（`54.443 / 36.704 / 786 / 5,615`）净回收 `973` 个 FN，带来 **`+3.129` HOTA / `+4.141` AssA / `+6.488` IDF1**；继续降至 `0.30` / `0.10` 则略回落至 `56.595 / 56.594` HOTA。
2. **Stage C `149` 例真实闭环 Onset 解剖与早期锚点信号：**
   - **病因构成**：`74.5%`（`111/149`）为连续帧 `gap==1` 切换，`89.9%`（`134/149`）伴 `IoU >= 0.30` 空间重叠（中位 `0.556`），`74.5%`（`111/149`）`assignment_margin < 0.10`（中位 `0.0456`），`22.8%`（`34/149`）为 `2-cycle` 互换；`55` 例真目标框在场中 `34` 例强反转、`16` 例平局、`5` 例匈牙利竞争落败。
   - **可分性**：`-assignment_margin` 在 onset 上离线 AUROC 达 **`0.9432`**（5% FPR 召回 `73.8%`）；早期无冲突锚点 `anchor_reid_dist` 打破后切换自洽陷阱，在 `31,791` 例 `post_switch` 上 AUROC 达 **`0.7062`**（全量跨 ID `0.7058` vs `recent_reid_dist` `0.5251`）。
3. **Stage D 冲突感知拒写与早锚保留（`G1`–`G4` @ `0.50`）：**
   - `G1_CONFLICT_IOU030_TAU065`（`48.686` HOTA / `8,055` FN / `4,733` 跳写）相对无条件 `C0_TAU065`（`43.509` HOTA / `9,094` FN）挽回 `+5.177` HOTA，但 **`G1`–`G4`（`43.380–48.686` HOTA）全数低于 `T_TWOSTAGE_TH050`（`54.443`）与 `T2_TH040`（`57.572`）**；`G4` 早锚槽固定（`43.380`）进一步低于 `G3`（`44.442`）。拥挤区约 `42%` 成熟正匹配触发 `IoU>=0.30` 导致 `>98%` 误跳过并在遮挡时饿死模板更新。

### Decision, Limitations & Exact Next Action

- **Decision (Case 2 Confirmed)**: 仅在写库端拒写无法弥补关联端强反转与竞争，**S12 直接立项训练端与分配端联合改进：`C'`（GAFFE 成对运动/方向遮挡边偏置）+ `A'`（双向边际/`2-cycle`/延迟解歧）**，同时纳入校准低阈基点 `0.40`（`57.572`）及 Stage C 发现的 `anchor_reid_dist`（`0.7062` 后切换可分信号）。本轮仅单 seed `13` 零重训推理、复用 `motion_dev8`，不宣称架构创新或未知集泛化。
- **Exact Next Action**: 交由研究端审阅 [S11 报告包](../../reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md)（`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`diagnostics.json`、`cases.jsonl`）并冻结 S12（`C' + A'`）协议。工作树保持未提交、未推送。

---

## 4. Earlier Completed Task: S10 CAMELTrack 纯拒写基准分解与抗闭环污染实验 (`COMPLETED`)

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

## 5. Earlier Completed Task: S09 CAMELTrack Threshold Calibration, Two-Stage Baseline & Failure Attribution (`COMPLETED`)

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

## 6. Completed / Stopped Historical Tasks

- **S08 (COMPLETED, superseded by S09 for threshold/two-stage analysis)**: [S08 Task](S08_CAMEL_BASELINE_TASK.md) | [S08 Report](../../reports/baselines/s08_camel_baseline_20261125/README.md) (`CONTINUE`, `T_SCRATCH @ 0.1` vs `P_OFFICIAL @ 0.1`).
- **2026-11-26 Research Review & Literature Audit**: [docs/research_status_review_20261126.md](../research_status_review_20261126.md).
- **Stopped Branches**: `P1`, `R1`, `R2`, `E3`, `E5`, `S03`, `S04`, `S06` (`STOP_S06_NO_SEEDS_37_73`; CPU postmortem in [S07 report](../../reports/diagnostics/s07_s06_checkpoint_postmortem_20261124/README.md)).
