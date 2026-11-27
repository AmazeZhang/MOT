# 创新主张台账

更新日期：2026-11-27。本表是主张级别的单一事实源；报告文案不得超过本表状态。

**2026-11-27 S11 复盘与 S12（`C' + A'`）立项登记：** S11（[权威报告](../../reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md)）已完成全部 A–E 阶段并确认 **Situation 2**：① `T2_TH040`（`0.40`）将 `T_TWOSTAGE` 标定基线提升至 **`57.572` HOTA / `40.845` AssA / `60.117` IDF1 / `832` IDSW / `4,642` FN**；② `149` 例真实 onset 解剖揭示 `74.5%` 为 `gap==1`、`89.9%` 伴 `IoU>=0.30`、`63.1%`（`94/149`）真目标框缺检却误抓重叠邻框、`74.5%` 双向 `margin < 0.10`（`-margin` 离线 AUROC `0.9432`）、`22.8%`（`34/149`）为 `2-cycle` 互换，且 `anchor_reid_dist` 在 `31,791` 帧后切换尾段获 `0.7062` AUROC；③ 纯写库侧门控 `G1–G4`（`43.380–48.686` HOTA）均因拥挤区 `~42%` 误拒写致 `FN` 飙升（`8,055–9,759`）而全部劣于 `T_TWOSTAGE_TH050`（`54.443`）与 `T2_TH040`（`57.572`），对应纯写库端门控分支登记为 `REJECTED`。研究端已冻结 [S12 任务书](../a2a/S12_CAMEL_PAIRWISE_EDGE_AND_ASSIGNMENT_TASK.md) 与 [S12 协议](../../manifests/protocols/s12_camel_pairwise_edge_and_assignment_20261127.yaml)（`DESIGN_READY_RUNTIME_LOCK_PENDING` / `NOT_RUN`），在同初始化（S09 C1 TE）及单节点容量对照（`M0`）与零重训分配对照（`B1/B2`）约束下检验 **`C'`（GAFFE 成对运动外推残差 + DOA 式方向遮挡/前后序边偏置）** 与 **`A'`（双向竞争边际/未匹配防误抓训练损失与分配解歧）**；两项主张当前状态为 `NOT_SUPPORTED_BY_PROVENANCE`，且显式保留不与 [DOA-SORT](https://arxiv.org/abs/2609.22706)、[HyperSSM](https://arxiv.org/abs/2604.12665)、[VLA-ReID](https://arxiv.org/abs/2607.17157)、[C-TWiX](https://arxiv.org/abs/2403.08018)、[SLAck](https://arxiv.org/abs/2409.11235)、[HDST-GNN](https://arxiv.org/abs/2606.05587)、[MM-Tracker](https://arxiv.org/abs/2407.10485) 混淆的首创边界。

**2026-11-27 S10 复盘与 S11 登记：** S10（[权威报告](../../reports/feasibility/s10_camel_anticontamination_20261127/README.md)）已完成并证伪当前 `B'_takeover`（`C2` HOTA `48.210`）与 `B'_consistency_gate`（`C3` HOTA `49.781`、真实跨 ID 写入全局 AUROC `0.5002`）局部滑窗合成接管假设（状态更新为 `REJECTED` 仅限该局部合成接管机制，不扩展 multi-seed）；同时 Stage B `D2_SKIP_ONLY`（`gt_injected_on_miss=0`）证实纯拒写脏特征写入存在 `+9.338` HOTA（`+13.236` AssA，`54.443 -> 63.781`）真实空间，且 Stage D 揭示 `31,791/31,940` 跨 ID 污染源于 `149` 例首次切换 onset（模型原生 `1-sim` 对其达 `0.8512` AUROC）。研究端已下发零重训过渡任务 [S11 任务书](../a2a/S11_CAMEL_ONSET_AND_CONFLICT_GATE_TASK.md)（[协议](../../manifests/protocols/s11_camel_onset_and_conflict_gate_20261127.yaml)，已由 S11 执行完毕）。

**2026-11-27 S09 复盘与 S10 候选登记：** 审阅 S09 (`f8c6dc3`) 后确认：① 阈值 `0.1→0.50` 仅将 `T_SCRATCH` AssA 提至 `37.900`；② 两阶段 TE 预训练将 `T_TWOSTAGE_TH050` IDSW 减半（`1333→786`）但 AssA 仍为 `36.704`；③ D2 `v3`（AssA `80.063`）同时包含“阻止 1,108 异主 + 1,319 FP 写入”与 `else:` 分支“GT 强制补写漏配帧（≥1,247 次）”。已制定 [S10 任务](../a2a/S10_CAMEL_ANTICONTAMINATION_TASK.md) 与 [2026-11-27 复盘](../research_status_review_20261127.md)，通过 `D2_SKIP_ONLY`、`C0_SIMPLE_WRITE_GATE`（`0.60/0.65`）、`C1_MATCHED_IID_SWAP`（配平交换率）严格检验 `B'_takeover`（`C2`）与 `B'_consistency_gate`（`C3`）。当前状态仍为 `NOT_SUPPORTED_BY_PROVENANCE`（未执行，不得宣称 S10 机制成立或否定）。

**2026-11-26 S08 复盘与 S10 候选更新：** 基于 S08 `d662023` 闭环结果与最新联网文献窄查重（见 [2026-11-26 复盘](../research_status_review_20261126.md)），整理三个待 S09 实证裁决的 S10 候选方向：`A'`（训练目标-全局二分分配边际对齐与动态拒识边界，须区分 [SLAck](https://arxiv.org/abs/2409.11235)、[HDST-GNN](https://arxiv.org/abs/2606.05587)、[MM-Tracker](https://arxiv.org/abs/2407.10485)）、`B'`（抗闭环自相关污染训练与一致性去污，受限于 [MOTIP2](https://arxiv.org/abs/2610.10391)、[MOTRv3](https://arxiv.org/abs/2305.14298)、[HMP](https://www.mdpi.com/2079-9292/15/11/2357)、[TCEI](https://arxiv.org/abs/2603.21629) 高重合，仅在 S09-D2 证实历史污染主导时启动）、`C'`（CAMELTrack GAFFE 显式成对运动学残差与方向性遮挡边偏置，桥接 [DOA-SORT](https://arxiv.org/abs/2609.22706)/[HyperSSM](https://arxiv.org/abs/2604.12665) 与多线索 Transformer）。三者当前均处于未实验、未立项状态（`NOT_SUPPORTED_BY_PROVENANCE`），待 [S09 任务](../a2a/S09_CAMEL_CALIBRATION_AND_ATTRIBUTION_TASK.md) 标定与归因后单选其一。

**2026-11-25 S07 更新：** [补充体检复盘](../research_status_review_20261125.md)确认同 epoch Context 有固定样本回归优势，但开发动作一致率未超过 always-M，结果后动作类型均值对照的 MSE 更低。保留原 S06 单 seed 闭环数值，不升级 V1/V2，不把 22 个分量中的一例差异解释为显著否定所有上下文/风险方法。当前稀疏分歧双头选择器建议结束；下一研究平台优先成熟学习关联基线，具体创新主张尚待确定。C1–C5 与其他停止判定不变。

**2026-11-24 证据更新：** S05/S06 已完成，见 [最新复盘](../research_status_review_20261124.md)。S06 当前 Context-risk 冻结门 2/8，通过数不足，按协议停止；V2 的当前实现没有得到支持，不等于所有风险建模被否定。Context 无额外风险对 M/V 有单 seed 的描述性官方汇总收益，但对 V 为 2 胜/2 负/4 平，且 Local/Context 选中 epoch5/30、有效容量不匹配、Local 本身也使用竞争 margin；尚不足确认 V1 的机制。V1/V2 状态继续为 `NOT_SUPPORTED_BY_PROVENANCE`（主张所需证据不足，并非没有任何真实结果）。不升级 C1–C5 或恢复旧停止路线。下列 11 月 23 日文字是候选登记时的历史记录。

## 2026-11-23 视觉决策候选（尚未运行）

S01 支持固定深度 EMA 基线在复用开发集上的收益；S03/S04 不支持继续当前错配后冻结记忆机制。既有 C1–C5 与 R1 停止判定不变。新候选来自 [S05/S06 建议](../a2a/NEXT_ROUND_20261123.md)，不是原 R1 结果升级。

| ID | 待验主张 | 最近先例与风险 | 必须的证据 | 状态 |
|---|---|---|---|---|
| V1 | 同一合法候选集合中，竞争上下文比只看动作自身的局部线索更能改善自然闭环关联 | CAMELTrack 已有目标间关系和多线索学习；Deep OC-SORT/DeconfuseTrack 已涉及自适应线索与混淆处理；高重合 | Context 对 Local、固定 M/V、近邻学习基线；冻结输入与预算，完整序列/多 seed，排除候选集变化 | `NOT_SUPPORTED_BY_PROVENANCE` |
| V2 | 对完整匹配分量显式预测损伤并保留不干预，可改善新增正确与破坏原正确的权衡 | 与本项目 R1、既有关联决策/保守选择思路重合；不能宣称风险建模首次 | 相同 checkpoint 的风险/无额外风险代价对照、所有受影响身份与全序列指标，跨视频和种子复核 | `NOT_SUPPORTED_BY_PROVENANCE` |

新颖性边界：仅做定向先例复查，未完成方法级排重；“上下文关联网络”本身不能成为新贡献。候选架构、损失或停止门在实现前修改须修订协议，不得事后选择性报告。2026-11-23 未运行 S05/S06，没有新方法结果。

## 历史主张与阶段记录

本日阶段复盘补充：P1 确定性模型已通过三 seed 离线开发门，但历史由 GT 对齐，且 gap16–31 IoU 退化。该证据支持自然闭环尝试，不涉及分布相对确定性的优势，也不支持 FM 或关联监督的必要性，故 C1–C5 状态不变。执行顺序见 [阶段复盘](../experiment_review_20261112.md)；C4 的 `dominance verdict=no` 表示未建立主导性，不能解释为已证明运动因素无效。

2026-11-13 补充：P1 三 seed 在自然闭环冻结门得到 `NO_GO`，平均 HOTA/AssA 同时低于 B0 和恒速对照，且每个 seed 的恢复减损伤均为负。离线预测改善没有转化为稳健身份关联；恒速虽为最强对照，仍损伤 20/81 成功事件。该结果不支持升级 C1–C5，也不授权 Gaussian、MDN 或 FM。下一步只允许一次有界机制诊断及至多一个由诊断明确支持的确定性修正。

同日机制诊断补充：831 个固定事件的同 B0 状态比较中，learned 相对恒速的注册几何/中心误差/门控/即时匹配劣势全部未触发，且单步聚合反而更优；这没有解释真实闭环中 learned 更差的原因，也不证明反馈/history/lifecycle 机制。按预先冻结的资源规则，当前 learned-motion 分支停止，获准修正数为 0。C1–C5 继续不升级，Gaussian、MDN、FM 与 official val 均不获授权。

## 状态定义

2026-11-13 新候选登记（不是 C1–C5 升级）：R1“同状态成对短程回放监督的关联改动收益/损伤选择”，当时状态为 `NOT_SUPPORTED_BY_PROVENANCE`，当时只有 [研究路线建议](../research_route_proposal_20261113.md)。最近先例包括 Learning to Track、MARLMOT、DeconfuseTrack、Sentinel；宽泛的决策学习/置信度门控存在高重合风险，拟采用的动作和监督组合尚未完成新颖性认证。R1 不恢复已停止的 P1 或授权 FM 训练。

2026-11-18 R1 更新：阶段 A 的动作和短程代理标签工程验收通过；阶段 B 在 24 段 fit、8 段训练内留出比较未来标签线性选择器、即时标签选择器、简单 IoU 规则与随机对照。预定阶段 C 门槛 `FAIL`：未来选择器在 83 次固定覆盖动作上的净身份正确帧差 +168，简单 IoU 规则 +239；逐视频仅 1/8 严格胜出。审计为同系列临时 `WARN`，未发现改变门槛判定的错误。**当前线性选择器路线按注册停止规则结束**；不能把“未来监督优于即时监督/随机”提升为“未来监督优于强简单规则或改善自然闭环”。R1 的广义研究问题未被普遍否定，但本轮没有可升级的主张。证据见 [阶段 B 报告](../r1_stage_b_report_20261118.md)与[完整性审阅](../audits/r1_stage_b_20261118/EXPERIMENT_AUDIT.md)。C1–C5 状态不变。

- `ALIGNED`：表述与已核验文献或项目证据一致。
- `OVERSTATED`：有一定证据，但表述超过证据边界。
- `NOT_SUPPORTED_BY_PROVENANCE`：尚无真实实验或可核验来源支持，不得当作结论。
- `PROVENANCE_INSUFFICIENT`：有结果线索，但配置、数据、代码或评测记录不完整。
- `REJECTED`：已被文献或实验明确否定，不再作为项目主张。

## 台账

| ID | 待审主张 | 创新重合风险 | 最近先例 | 必须的项目证据 | 对应实验 | 当前状态 |
|---|---|---|---|---|---|---|
| C0 | 未检出与“真实候选集关联损失经 CFM 实际采样链改变运动生成器＋匹配 stop-gradient/joint/直接分类对照”完全相同的工作 | 高；只是“未检出” | DiffuTrack、DMMTracker、Constraint-Aware FM | 方法定稿后再做窄检索；不得改写为绝对首创 | 无 | `ALIGNED` |
| C1 | 运动分布比同编码器确定性回归更能改善自然缺失后的闭环关联 | 中；概率运动 MOT 已有 | TrajE、DiffMOT、DiffuTrack | 自然视频、固定检测、配对输入、多视频/种子的 AssA/HOTA/恢复差异 | G3 P1–P4；G4 motion | `NOT_SUPPORTED_BY_PROVENANCE` |
| C2 | 关联监督必须改变生成器才能获得额外闭环收益 | 高；身份导向运动和下游导向 FM 已有 | DMMTracker、Constraint-Aware FM | joint 相对 detach/motion 的稳定配对改善；覆盖/能量分数与延迟不产生无法解释的恶化 | G4 joint/detach/direct classifier | `NOT_SUPPORTED_BY_PROVENANCE` |
| C3 | FM 相对 Gaussian/MDN 在同延迟或同训练预算下有额外价值 | 高；FM 不自动优于轻量分布 | TrajE、MoFlow、TrajFlow、DiffMOT | 同历史编码、同数据暴露、同调参预算和延迟曲线；闭环而非只有 oracle | G3 P2/P3/P4；K/S 切片 | `NOT_SUPPORTED_BY_PROVENANCE` |
| C4 | 当前主要错误来自运动分布不足，而不是观测偏移、检测缺失或历史 ID 污染 | 高；已有直接竞争解释 | OA-SORT、DOA-SORT | D1/D2/D3 分开诊断，自然失败/成功事件对照，影响范围和不明确映射单列 | G2 D1--D5 已完成；D4 无恢复，D5 为构造性压力测试；dominance verdict=`no` | `NOT_SUPPORTED_BY_PROVENANCE` |
| C5 | 当前最有可辩护性的贡献是梯度路径与因果式对照，不是“FM 用于 MOT” | 中；取决于后续窄检索和实验 | DiffuTrack、DMMTracker、Constraint-Aware FM | C0 继续成立，且 C2 获得支持；若 C2 失败则收缩到评价/负结果 | G4 完整机制对照 | `NOT_SUPPORTED_BY_PROVENANCE` |

## 升级与降级规则

1. 计划中的实验不是 provenance；只有可复查的完整 run 才是。
2. C1–C5 任一主张准备升级时，先填写决策记录，列出最强负结果、方差和未解释异常。
3. 发现完全先例时，C0 立即降级为 `OVERSTATED` 或 `REJECTED`，C5 同步重新评估。
4. MDN 与 FM 相当、直接分类器更好、只有人工缺失改善或只有 oracle 改善，都必须保留并触发主张收缩。
