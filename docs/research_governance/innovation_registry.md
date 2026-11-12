# 创新主张台账

更新日期：2026-11-12。本表是主张级别的单一事实源；报告文案不得超过本表状态。

## 状态定义

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
