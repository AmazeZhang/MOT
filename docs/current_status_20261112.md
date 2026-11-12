# 当前研究状态总结（2026-11-12）

## 一句话结论

项目已经完成环境/数据闭环、OC-SORT 基线、G2 错误归因、B0/B1 等价、异步 GPU 队列和确定性 P1 离线预测实验。确定性 GRU 通过了冻结的三 seed 离线开发门，下一步可以进入自然闭环跟踪实验；目前仍没有新方法的 HOTA、AssA、IDF1 或身份恢复提升证据，也没有证据支持直接扩展到 Gaussian、MDN 或 Flow Matching。

## 当前阶段

| 阶段 | 状态 | 已获得的信息 |
| --- | --- | --- |
| G0 环境与数据 | 完成 | 三个隔离环境、DanceTrack train/val、TrackEval、固定检测缓存和外部数据盘布局已验收 |
| G1 基线 | 完成 | OC-SORT 完整 train/val 检测、重放和官方 TrackEval 基线已登记 |
| G2 诊断 | 完成 | D1--D5 只做 train 事件级归因/特权反事实，不是闭环方法效果 |
| motion split | 完成 | DanceTrack train 冻结为 32 段 motion-train 与 8 段 motion-dev；官方 val 不参与开发 |
| B0/B1 等价 | 完成 | motion-dev 上 73,479 行输出逐元素一致，研究接口关闭时不改变上游行为 |
| 异步 GPU 层 | 完成 | GPU0 硬拒绝、GPU1--7 独立 attempt、失败隔离和只聚合 PASS 已验收 |
| P1 数据与 smoke | 完成 | 266,854/80,458 train/dev 样本，两次构建一致；真实 GPU1 过拟合 smoke 通过 |
| P1 离线可行性 | 完成并 GO | 三 seed 全量训练和 motion-dev 对照通过全部五项冻结门 |
| P1 自然闭环 | 下一步 | 尚未运行；必须使用 tracker 自身历史和官方跟踪指标 |

## 基线与诊断发现

### G1 OC-SORT 基线

- DanceTrack train：HOTA `62.203`、DetA `85.665`、AssA `45.212`、IDF1 `61.779`。
- DanceTrack val：HOTA `52.059`、DetA `77.400`、AssA `35.130`、IDF1 `51.623`。
- val 在查看指标前冻结，只作为最终参考基线，不用于当前模型选择。

### G2 对“运动/分布不足是主导原因”的约束

- D2 特权修正 target track 当前预测位置，在 53 个身份改变事件中新增 22 个首轮正匹配。
- D3 特权修正已有正检测坐标，新增 14/53；D2/D3 交集 5、独立并集 31，但这不是同时干预结果。
- D1 窄历史干预没有新增失败恢复，且只实际改变少量事件，不能证明历史无关。
- D4 全局移除方向代价新增 0/53，并破坏 3/112 成功对照；只能否定这一窄操作。
- D5 把已知正确 pair 两侧设为同一个 GT 框后新增 52/53；这是 IoU=1 的 assignment-consistency 压力测试，不是可实现恢复上界。
- 因而“运动/分布误差是主导可行动原因”仍未建立。G2 只支持存在有限位置相关 headroom，并授权一个受限的确定性 P1 探索。

## 确定性 P1 离线结果

### 数据与训练

- motion-train：32 段、266,854 个样本。
- motion-dev：8 段、80,458 个样本；其中 4 段是已检查的困难 G2 anchors。
- 输入历史是过去 GT 匹配到的真实检测框，当前/未来检测不进入历史；这仍是训练/离线用的 GT 对齐历史，不等同 tracker 自身历史。
- 三个固定 seed `20261112/20261113/20261114` 分别使用物理 GPU1/2/3，固定训练 10 epoch，无 dev 早停、无最佳 seed 选择。

### 冻结门结果

| 指标 | 结果 | 解释 |
| --- | ---: | --- |
| learned 平均宏 MSE / 最佳对照 | `0.915631` | 相对零 delta 改善约 8.44% |
| learned gap2--31 MSE / 最佳对照 | `0.925642` | 改善约 7.44% |
| learned 宏 IoU - 最佳对照 | `+0.006873` | 通过整体 IoU 安全门 |
| seed 宏 MSE 相对极差 | `0.003330` | 三 seed 稳定 |
| 序列级 MSE 胜出 | 每个 seed `8/8` | 所有 motion-dev 序列均低于零 delta |

五项条件全部通过，决策为 `GO_TO_NATURAL_CLOSED_LOOP`。这只授权下一道实验，不是跟踪提升结论。

### 必须同时报告的不利结果

- motion-dev 中 75,428/80,458 个样本是 gap1，gap2--31 只有 5,030 个。
- 650 个 gap16--31 样本上，learned MSE 仍改善约 9.28%，但平均 IoU 比零 delta 低 `0.008110`，三个 seed 都出现下降。
- 因此不能声称长 gap 全面改善，也不能把离线预测分数替代闭环关联和身份持续性。

### 完整性状态

- fresh same-family 审计为 provisional `WARN`。
- 审计独立核验 347,312 个 target state 对原始 GT 最大误差 0、40 个 GT 归档、80,458 个 dev 预测键/目标、三 seed artifact、50 项 provenance、120 项输入哈希和全部指标；独立复算最大差为 `3.33e-16`。
- WARN 来自 GT 对齐历史、4/8 已选 anchors、长 gap IoU 退化、尚无官方闭环指标，以及冻结 chronology 不能由不可变时间证据独立认证。

## 当前可以与不可以声称什么

可以声称：

- 确定性 GRU 在冻结的 motion-dev eligible population 上，相对两个注册的简单预测器通过五项离线开发门。
- 三 seed 结果稳定，并获准进入自然闭环测试。

不可以声称：

- 已改善关联、身份恢复、HOTA、AssA、IDF1 或部署表现。
- 所有 gap 长度都获益，或已证明对未见数据泛化。
- 已证明运动分布是主导问题。
- 已证明 Gaussian、MDN 或 Flow Matching 有必要或有效。

## GPU、环境和产物约束

- 物理 GPU0 永久留给 Ubuntu 图形界面，所有代码和队列都硬拒绝 GPU0。
- 当前训练只使用 GPU1--3；其它 GPU 保持可用于后续独立作业。
- Python 环境：`mot-core` 负责 PyTorch 训练，`mot-det` 负责检测/OC-SORT，`mot-eval` 负责 TrackEval。
- 大数据、检测缓存、raw runs、checkpoint 和逐样本预测统一位于 `/media/imc/data/mot-data/`，不进入 Git。
- Git 只保存代码、冻结协议、轻量摘要、输入/产物哈希和审计报告。

## 原始与轻量记录位置

- 总进展：`docs/progress_log.md`
- P1 离线轻量报告：`reports/feasibility/p1_deterministic_offline_v1_20261112/`
- P1 raw queue：`/media/imc/data/mot-data/runs/p1_deterministic_offline_v1_20261112/`
- P1 raw summary：`/media/imc/data/mot-data/runs/p1_deterministic_offline_v1_summary_20261112/`
- G1/G2/B0-B1 轻量凭证：`reports/baselines/` 与 `reports/diagnostics/`

## 精确下一步

1. 冻结 P1 自然闭环协议，不在看结果后修改门槛。
2. 用 tracker 自身在线历史驱动 B1，三个 seed 全部运行，不挑最佳 seed。
3. 同时运行 B0 和一个可实现的简单运动/关联对照。
4. 在完整 motion-dev 上评价官方 HOTA、AssA、IDF1、gap 后身份持续性、成功对照副作用和延迟。
5. 为 gap16--31 单独设置安全门；若闭环无收益或副作用明显，停止扩大运动模型。
6. 只有确定性闭环可信后，才讨论 Gaussian；MDN/FM 继续保持 NO-GO。
