# S09 复盘审阅与 S10 抗污染实验设计说明

日期：2026-11-27；核查仓库版本：`f8c6dc3`（`Complete S09 CAMELTrack threshold calibration, two-stage baseline, and D1/D2 failure attribution report`）。
关联文件：[S09 权威报告](../reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md)、[S10 任务书](a2a/S10_CAMEL_ANTICONTAMINATION_TASK.md)、[S10 协议](../manifests/protocols/s10_camel_anticontamination_20261127.yaml)、[创新主张台账](research_governance/innovation_registry.md)。

> 证据边界声明：本复盘基于本地对 S09 Git 报告包与 `scripts/run_s09_camel.py` / `tests/test_s09_camel.py` 的核查，未连接服务器重跑大产物。S10 处于 `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）状态，尚未执行任何训练或评测。

---

## 1. S09 核查结论：排除了两个假瓶颈，定位到真实主因

S09 在物理 GPU 1（`8.486 GPUh`、`6.30 GiB` 新增存储、`13/13` 验收检查通过、`0.0` TrackEval 导出差异）上完成了全部 A–E 阶段：

1. **推理阈值 `0.10` 不是核心瓶颈**：
   - 将 S08 `T_SCRATCH` 从 `0.10` 调至 `0.50` / `0.519729`，`motion_dev8`（7,565 帧）AssA 仅从 `36.096` 微升至 `37.900` / `36.875`，IDSW 仍为 `1,333` / `1,412`；`P_OFFICIAL` 调到 `0.50` 后 HOTA 反而从 `66.993` 降至 `63.874`（FN 从 `4,536` 增至 `5,732`）。
2. **补齐 `train32` 两阶段 TE 预训练也不是全部答案**：
   - Stage C 严格只在 `motion_train32` 执行了 3 个单线索 TE 20-epoch 预训练 + 20-epoch 联合 GAFFE 训练，`T_TWOSTAGE_TH050`（`0.50`）将 IDSW 几乎减半（`1,333 → 786`），在 `dancetrack0044` 上把 AssA 从 `16.498` 提升至 `24.757`；但全集 AssA 仍仅 **`36.704`**（HOTA `54.443`、IDF1 `53.629`、FN `5,615`）。
3. **D1（100% 官方 IDSW 覆盖）与 D2（`v3` 干净历史探针）锁定“闭环 gallery 污染级联”**：
   - D1 显示 `T_TWOSTAGE_TH050` 的 `786` 例 IDSW 中，错配反超（`margin < -0.05`）占 `52.29%`（411）、高分平局（`|margin| < 0.05`）占 `25.95%`（204）、漏连新建占 `17.43%`（137）、匈牙利冲突丢失占 `4.33%`（34）；`60.05%` 发生于 `gap > 1`，`76.08%` 伴随竞争 GT `IoU >= 0.3`。
   - D2 `v3` 使用完全相同的 `T_TWOSTAGE_TH050` 权重与阈值，在阻止 `1,108` 次异主写入与 `1,319` 次 FP 写入并保持 1-to-1 GT 干净 gallery 后，AssA 从 `36.704` 跃升至 **`80.063`**（`+43.359`），HOTA 从 `54.443` 升至 **`81.907`**（`+27.465`），IDF1 从 `53.629` 升至 **`92.153`**（`+38.524`）；其中 `dancetrack0037` AssA 由 `21.796 → 84.008`，`dancetrack0044` 由 `24.757 → 78.349`。

---

## 2. 对 S09 D2 `v3` 实现源码的关键口径修正（S10 必先拆解）

审阅 [scripts/run_s09_camel.py](../scripts/run_s09_camel.py#L1206-L1225) 中的 `CleanHistoryCAMELTrack.process` 发现，D2 `v3` 的 `+43.359` AssA 增益实际上叠加了**两项不同的特权干预**：

1. **拒写污染（Do-No-Harm）**：阻止 `1,108` 次跨 ID 错配和 `1,319` 次 FP 错配进入 `trk.detections`。
2. **漏配/错配帧的 GT 强制补写（`else:` 分支）**：当 `clean_matched` 为空（该轨迹当帧并未正确匹配到自己目标检测）时，源码第 1215–1224 行仍将当帧 GT 对应的真实检测 `detections[target_det_idx]` 注入 `chosen_trk.detections`（`gt_routed_gallery_updates = 76,509` > 干净预测匹配数 `75,262`，至少多注入 `1,247+` 次观测；这也是 `FN` 从 `5,615` 降到 `4,585` 的重要原因）。

**科学影响**：线上不可部署方法无法凭空执行第 (2) 步“用 GT 补回漏掉的帧”，只能做到接近第 (1) 步的“识别/降权或跳过疑似污染观测”。因此 S10 第一步（Stage B，`~0.15 GPUh`）必须先运行去掉 `else:` 补写的 **`D2_SKIP_ONLY`（纯拒写、不补漏）** 探针，定量拆出“仅靠不写脏数据”在当前模型上究竟能挽回多少 AssA，再据实解读后续在线方法上限。

---

## 3. S10 实验设计的科学逻辑与创新边界

在 [2026-11-26 文献复盘](research_status_review_20261126.md) 已确认 `B'` 赛道（MOTIP2、MOTRv3、HMP、LiAM-SAM、TCEI）重合风险较高的前提下，S10 绝不能把“置信度低就不写入 gallery”或“随机交换增强”包装成新方法。因此 S10 采用**强简单对照 + 单因子正交拆解**：

1. **零重训强简单对照（`C0_SIMPLE_WRITE_GATE`）**：
   - 在完全不改 S09 `T_TWOSTAGE_TH050` 权重的前提下，预设两个固定写入阈值 `tau_write ∈ {0.60, 0.65}`（已确定历史长度 `hits >= 3` 时，若当帧 `sim < tau_write` 则仅输出预测但不污染 `trk.detections`）。如果仅靠这行简单启发式门控就能拿走大部分可部署收益，后续就不再夸大复杂学习模块的价值。
2. **分布对齐因子对照（`C1_MATCHED_IID_SWAP` vs. `C2_CONTIGUOUS_TAKEOVER`）**：
   - CAMELTrack 原版 ACT 的 `SwapSporadic` / `SwapOccluded` 是各帧独立随机交换，与闭环下“早期错配后连续 $L$ 帧异主观测持续接管 gallery 尾部”的动力学不同。
   - `C2` 在 `motion_train32` 上构造**连续异主接管污染训练（Contiguous Takeover）**（从同视频空间邻近竞争轨迹注入连续 $L \in [2, 12]$ 帧异主观测，要求前缀干净长度 $\ge 2L$），而 `C1` 以**严格配平的相同总交换帧比例**执行独立随机交换。二者共享完全相同的架构、C1 TE 初始化、20-epoch 调度与损失函数，专门检验“连续接管结构”是否超越“等量 i.i.d. 噪声”。
3. **机制监督与可微去污聚合（`C3_CONSISTENCY_GATED_TE`）**：
   - 不依赖简单匹配分数标量阈值，而是在 `TemporalEncoder` / `GAFFE` 前的历史编码端增加一个**轻量化、严格因果的多线索历史一致性门控头**：在训练期利用 ACT 已知的历史污染掩码 `corrupted_mask`（推理期完全无 GT！）直接监督预测每个历史观测与当前轨迹主身份的一致概率 $w_t \in (0, 1)$，并以 $\log(w_t + \epsilon)$ 作为时序/跨目标注意力的软偏置，抑制污染历史帧对关联打分与特征聚合的毒害。
   - **机制验证**：除了完整的 `dev8` TrackEval 指标，S10 Stage D 还将直接评测该一致性头在 `T_TWOSTAGE` 真实闭环 `dev8` 上**对那 `1,108` 次实际异主污染写入的检出 AUROC / F1**，并对比“仅注意软降权”与“联合推理期低一致写入过滤”的增益与副作用。
