# S08 复盘、前沿查重与下一步研究路线（A' / B' / C' 与 S09 裁决）

日期：2026-11-26；核查仓库版本：`d662023`（`s08: complete CAMELTrack baseline loop (A-F) and handoff report`）。
关联文件：[S08 权威报告](../reports/baselines/s08_camel_baseline_20261125/README.md)、[创新主张台账](research_governance/innovation_registry.md)、[S09 任务书](a2a/S09_CAMEL_CALIBRATION_AND_ATTRIBUTION_TASK.md)、[S09 协议](../manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml)。

> 证据边界声明：本文件基于本地仓库拉取 `d662023` 后已落盘的 S08 报告、配置与案例文件，以及 2026-11-26 联网检索到的公开论文页面/摘要与仓库既有核查记录。本轮未连接服务器、未执行训练或新 TrackEval；对 2026 年部分预印本（未全文通读前）的机制差异仍标记为待复核。

---

## 1. 当前实现状况与 S08 暴露的核心反常

S08 已在物理 GPU 1（禁用 GPU 0）上完成 CAMELTrack（`bc1c261075b0beacfe71c2896453be83643153c8`）A–F 全流程闭环，总计耗时 `4.465` GPU 卡时，工程可控性与在线因果边界（去 GT、未来截断前缀、重复确定性、step-50 RNG 恢复、新进程显式重载）全部通过（[checks.json](../reports/baselines/s08_camel_baseline_20261125/checks.json)）。

然而，**离线验证指标与闭环跟踪指标之间出现了极显著的断层**：

- **离线指标极高**（[learning_curves.csv](../reports/baselines/s08_camel_baseline_20261125/learning_curves.csv)）：`T_SCRATCH`（seed 13，仅 `motion_train32` 单阶段训练 20 epochs / 8,000 steps，43.48M 参数）在 epoch 20 达到 `train/loss_epoch = 0.0357`、`val/loss = 0.2021`、**`val/sim_auroc = 0.9922`、`val/sim_acc = 0.9532`**，`val/computed_sim_threshold = 0.519729`。
- **闭环指标偏弱**（[aggregate_metrics.csv](../reports/baselines/s08_camel_baseline_20261125/aggregate_metrics.csv)，`motion_dev8`，7,565 帧）：
  - `T_SCRATCH`：`HOTA = 53.981`、`DetA = 81.018`、`AssA = 36.096`、`IDF1 = 56.695`、`MOTA = 90.621`、`IDSW = 1451`、`FP = 1721`、`FN = 4434`、`Frag = 1428`。
  - `P_OFFICIAL`（官方两阶段 + `train+val` 预训练，仅作平台参考）：`HOTA = 66.993`、`DetA = 82.524`、`AssA = 54.516`、`IDF1 = 71.763`、`MOTA = 91.591`、`IDSW = 638`。
  - 逐视频差距全部落在关联侧（[per_sequence_metrics.csv](../reports/baselines/s08_camel_baseline_20261125/per_sequence_metrics.csv)）：例如 `dancetrack0044` IDSW 563 vs 142（AssA 13.789 vs 32.969），`dancetrack0037` IDSW 仅多 16 次（45 vs 29、Frag 同为 84）但 AssA 从 `61.154` 暴跌至 **`26.498`**。
- **自然失败分布特征**（Stage F，[diagnostics.json](../reports/baselines/s08_camel_baseline_20261125/diagnostics.json) & [cases.jsonl](../reports/baselines/s08_camel_baseline_20261125/cases.jsonl)）：
  - 全部 `1,451` 例官方 IDSW 中，**100%（1,451/1,451）** 当帧存在 IoU ≥ 0.5 合格检测；
  - **76.71%（1,113/1,451）** 存在同帧 IoU ≥ 0.3 竞争 GT 框；
  - **59.27%（860/1,451）** 发生在 **`gap == 1` 连续帧（无检漏间隔）**，仅 40.73%（591/1,451）发生在 `gap > 1` 之后；
  - 16 例抽样直观暴露两类现象：密集重叠下的**连续帧 2-cycle 双向互换**（如 `s08_idsw_dancetrack0044_f742_gt1`、`s08_idsw_dancetrack0020_f358_gt32`），以及无强竞争下失联数十帧旧轨迹的**跨轨强行接管**（如 `s08_idsw_dancetrack0020_f433_gt32` 中 `pred_id=29` 在 `last_frame=400` 失联 33 帧后抢占 `gt_id=32`）。

### 为什么出现“18.42 分 AssA 落差”？三个待剥离原因

1. **推理阈值未校准（配置约束，非模型上限，推断可快速剥离）**：S08 Stage E 为固定共享政策将 `sim_threshold` 锁定为 `0.1` 且关闭 `use_computed_sim_threshold`（[stage_e_t_scratch_dev8.yaml](../reports/baselines/s08_camel_baseline_20261125/resolved_configs/stage_e_t_scratch_dev8.yaml#L256-L267)）。但 `T_SCRATCH` 训练末期验证阈值是 `0.519729`（官方默认 override 为 `0.5`、`P_OFFICIAL` ckpt 内置 `0.46`）。`0.1` 配合 `max_wo_hits=150` 允许大量低置信度失联旧轨迹强行匹配。
2. **单阶段训练 vs. 官方两阶段 TE 预训练**：官方论文/仓库先用 `gaffe: Identity` 对三个单线索 `TemporalEncoder` 分别预训练，再联合训练 `GAFFE`。S08 仅跑了单阶段从头训练，因此 `53.981` 尚不是同数据 `train32` 下 CAMELTrack 的正常基线强度。
3. **结构性科研瓶颈（决定下一步创新方向）**：即便校准阈值并补全两阶段预训练，CAMELTrack 仍可能受制于三类固有结构问题之一——**(a) 训练目标（成对 InfoNCE）与全局一对一分配/拒识失配**；**(b) 闭环误差级联（gallery 连续异主污染）**；**(c) `bbox_encoder` 仅占 0.46% 参数且缺少显式成对相对运动/方向遮挡几何先验**。

---

## 2. 2025–2026 联网前沿查重与“撞车红线”

结合 2026-11-26 联网检索及仓库既往记录（[2026-11-20 文献目录](../调研/研究方向发散_2026-11-20/allinone.md)），梳理最贴近的三类先例与不可触碰的重合红线：

| 类别 | 代表文献与可信链接 | 核心机制（检索核实） | 对本项目选题的限制 / 差异要求 |
|---|---|---|---|
| **底座平台** | **CAMELTrack** (arXiv 2025)<br>[arXiv:2505.01257](https://arxiv.org/abs/2505.01257)<br>[GitHub@bc1c2610](https://github.com/TrackingLaboratory/CAMELTrack/tree/bc1c261075b0beacfe71c2896453be83643153c8) | TE + GAFFE 多线索融合，ACT 图像无关增强（`SwapOccluded` / `SwapSporadic` / `DropoutFeatures`），InfoNCE + 单阶段匈牙利匹配 + `sim_threshold` | 两阶段 TE 预训练与 IoU 交换增强已是平台既有内容，不能作为创新 |
| **分配损失 / Sinkhorn / Dustbin / Margin** | **SLAck** (ECCV 2024): [arXiv:2409.11235](https://arxiv.org/abs/2409.11235)<br>**HDST-GNN** (2026-06): [arXiv:2606.05587](https://arxiv.org/abs/2606.05587)<br>**MM-Tracker** (AAAI 2025): [arXiv:2407.10485](https://arxiv.org/abs/2407.10485)<br>**TrackTrack** (CVPR 2025): [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)<br>**DeconfuseTrack** (CVPR 2024): [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf) | SLAck 与 HDST-GNN 已使用 log-space **Sinkhorn + 可学习 dustbin 行列**处理未匹配目标；HDST-GNN 与 MM-Tracker 已使用 **margin loss**；TrackTrack / DeconfuseTrack 处理轨迹视角分配与全局分配混淆 | **红线**：不能宣称“首次在 MOT 中使用 Sinkhorn + Dustbin + Margin Loss”。必须窄化到：在同构多线索 Transformer 关联器中隔离“成对 InfoNCE + 静态全局阈值”与“双向分配边际对齐 + 上下文拒识边界”对连续帧 2-cycle 互换的因果效应 |
| **闭环曝光偏差 / 自生成历史 / 可靠记忆** | **Exposure Bias in MOT** (CVPR 2019): [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Maksai_Eliminating_Exposure_Bias_and_Metric_Mismatch_in_Multiple_Object_Tracking_CVPR_2019_paper.pdf)<br>**Event-Aware Learning** (UAI 2022): [PMLR](https://proceedings.mlr.press/v180/lee22a.html)<br>**MOTRv3** (2023): [arXiv:2305.14298](https://arxiv.org/abs/2305.14298)<br>**MOTIP** (CVPR 2025): [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html)<br>**MOTIP2** (BMVC 2026): [arXiv:2610.10391](https://arxiv.org/abs/2610.10391)<br>**MeMOTR** (ICCV 2023): [arXiv:2307.15700](https://arxiv.org/abs/2307.15700)<br>**HMP** (Electronics 2026): [MDPI](https://www.mdpi.com/2079-9292/15/11/2357)<br>**LiAM-SAM** (2026): [arXiv:2609.28078](https://arxiv.org/abs/2609.28078)<br>**TCEI / Dual-level Adaptation** (CVPR 2026): [arXiv:2603.21629](https://arxiv.org/abs/2603.21629) | MOTIP2 已有空间相关 ID 交换；MOTRv3 已有 Track Group Denoising；HMP/LiAM-SAM/TCEI 已覆盖置信度门控记忆写入、生命周期管理与测试时经验校准 | **红线**：此方向最拥挤。不能把空间重叠交换或置信度不写入记忆当创新；只有先实证“连续自相关污染主导 AssA 崩塌”，且配平噪声量后仍优于独立/空间交换，才可研究结构一致性去污 |
| **运动先验 / 方向遮挡 / 空间结构** | **TWiX / C-TWiX** (PR 2025): [arXiv:2403.08018](https://arxiv.org/abs/2403.08018)<br>**OA-SORT** (CVPR 2026): [arXiv:2603.06034](https://arxiv.org/abs/2603.06034)<br>**DOA-SORT** (2026-09): [arXiv:2609.22706](https://arxiv.org/abs/2609.22706)<br>**HyperSSM** (2026-04): [arXiv:2604.12665](https://arxiv.org/abs/2604.12665)<br>**FDTA** (CVPR 2026): [arXiv:2512.02392](https://arxiv.org/abs/2512.02392) / [GitHub](https://github.com/Spongebobbbbbbbb/FDTA)<br>**DiffMOT** (CVPR 2024): [arXiv:2403.02075](https://arxiv.org/abs/2403.02075) | DOA-SORT 在纯运动卡尔曼系首次显式建模左/右/上/下方向性遮挡观测偏移（DanceTrack val 63.00→66.34 HOTA）；HyperSSM 用超图+SSM 做多目标运动协同（test 65.7 HOTA / 52.6 AssA）；FDTA 在 DETR 上加深度/时序适配器；C-TWiX 为纯坐标成对 Transformer | **机制交叉空白**：尚未见到把“显式成对运动外推残差 + DOA-SORT 式方向性遮挡几何先验”作为 Pairwise Edge Bias 注入已具备 KPR 外观与关键点的多线索学习关联器（CAMELTrack GAFFE）并验证其增量价值的受控研究 |

---

## 3. 细化后的三个候选研究方向（S10 候选题池）

### 方向 A'：面向全局二分分配的边际对齐与动态拒识边界
- **核心假设**：CAMELTrack 离线 AUROC 达 0.9922 却在连续帧频繁互换，根本原因在于 **Pairwise InfoNCE 训练目标与逐帧 1-to-1 全局分配决策失配**，且全局静态 `sim_threshold` 无法同时抑制长距低分强连与高二义性互换。
- **具体改动**：同构 GAFFE 下，在相似度矩阵 $S$ 上加入帧内竞争边际约束（抑制 2-cycle 交叉交换得分与行/列强竞争负对）＋上下文预测的轨迹/检测级未匹配边界（替代静态全局阈值）。
- **必须对照**：同框架 InfoNCE + 静态最优阈值扫描、单纯边际损失（不含动态边界）、以及与 `HDST-GNN`/`MM-Tracker` 的差异边界说明。

### 方向 B'：抗闭环误差级联的自相关污染训练与结构一致性去污
- **核心假设**：若 `0037` 类视频 AssA 暴跌主要由早期偶发错配后错人特征持续污染 50 帧 gallery 所致，则无记忆的随机增强（`SwapSporadic`/`SwapOccluded`）无法对齐真实连续异主漂移分布。
- **具体改动**：先用“配平污染率/连续长度的滚动污染历史”对比独立/空间交换；若分布有效，再在 `TemporalEncoder` 内引入历史观测簇一致性降权（不采用已被 HMP 覆盖的置信度阈值硬拒写）。
- **风险**：近邻密集（MOTIP2、MOTRv3、HMP、LiAM-SAM、TCEI），除非 S09 实证历史污染主导失败，否则不作首选。

### 方向 C'：多线索 Transformer 中显式成对运动残差与方向性遮挡边偏置
- **核心假设**：在 DanceTrack 有限训练视频（`train32`）和同外观密集竞争（76.71% IDSW 有近邻 GT）下，`bbox_encoder` 仅占 0.46% 参数且只编码单目标绝对坐标，难以为 GAFFE 提供足够的相对运动学与局部遮挡可靠性线索。
- **具体改动**：在 GAFFE 的轨迹-检测跨目标注意力/边打分路径注入轻量化成对几何偏置——(1) 轨迹外推位置（线性/KF 预测）与候选框的运动学残差（中心偏移、方向角、尺度比、CIoU）；(2) 相对邻近目标的**方向性遮挡几何特征**（左/右/上/下被遮挡比例），辅助 GAFFE 在局部遮挡时动态降权受损外观/边界线索。
- **必须对照**：增大 `bbox_encoder` 隐层容量（排掉纯参数量效应）、仅运动学残差（不含方向遮挡）vs. 完整成对偏置。

---

## 4. 用 S09 低成本实证裁决下一轮方向

**绝对禁止在未校准阈值、未补齐 `train32` 两阶段基线前，直接把 S08 的 `53.981` 当靶子开发新模块。** 先执行有界实验 **S09**（[任务书](a2a/S09_CAMEL_CALIBRATION_AND_ATTRIBUTION_TASK.md)、[协议](../manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml)，预算总计 ≤ 16 GPU 卡时）：

1. **Stage B（~0.5 GPUh，零重训）**：复用 S08 `T_SCRATCH` 与 `P_OFFICIAL` ckpt，在 `dev8` 测 `sim_threshold ∈ {0.46, 0.50, 0.519729}`（与 S08 `0.1` 并列对照），直接分离“阈值未校准”占比。
2. **Stage C（~6–8 GPUh）**：严格只在 `motion_train32` 执行官方两阶段训练（Stage 1 三个 TE 分别用 `gaffe=Identity` 预训练 → Stage 2 联合训练 GAFFE 20 epochs，seed 13）得到 `T_TWOSTAGE`，并在 `0.50` 与自身 `computed_sim_threshold` 上评测 `dev8`。
3. **Stage D（~1 GPUh，只读机制归因）**：
   - **D1（内部分数与边际剖面）**：导出校准后残余 IDSW 帧的 `s_pos`、`s_assigned`、`s_best_competing` 及 `s_pos - s_best_competing`，区分“低分漏网”、“高分平局互换（`|margin| < 0.05`）”和“错配反超（`s_pos < s_neg - 0.05`）”；
   - **D2（GT 路由干净历史探针，特权诊断）**：同权重下若用 GT 身份路由保持历史 gallery 不被异主污染，能恢复多少 AssA / 减少多少 IDSW（尤其核查 `0037` 与 `0044`）。

### S09 → S10 唯一选择规则

- **情况 1**：D1 显示残余错误中大量为高分平局互换或阈值敏感两难，且 D2 干净历史增益有限 → **S10 立项方向 A'**；
- **情况 2**：D2 干净历史探针大幅拉回 `0037` 等 AssA 跌落、消除了多数连续 IDSW → **S10 立项方向 B'**；
- **情况 3**：即使在 D2 干净历史下，模型在 `gap == 1` / 近邻竞争帧依然大量给出 `s_neg > s_pos`（单步判别本身失效）→ **S10 立项方向 C'**。

---

## 5. 长期确认集规划提醒

`motion_train32` 与 `motion_dev8` 均来自 DanceTrack `train` 且已反复用于项目开发。未来在 S10 冻结单一创新方法后，不得继续仅用 `dev8` 宣称最终泛化结论；应在冻结方法与超参后，前往未用于 CAMELTrack 调参的**官方 DanceTrack `val`**（须诚实披露早期 G1 OC-SORT 基线使用史）、**DanceTrack `test`**（官方评测服务器）以及第二数据集（如 **SportsMOT**）上用多随机种子（如 13/37/73）完成确认评测。
