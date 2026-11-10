# 科研 Agent 并行任务简报

目标不是继续扩展泛文献列表，而是尽快判断当前窄差异能否同时通过“新颖性”和“有效性”两道门。

## Lane R1：完全先例阻断检索

- 对应 claim：C0、C2、C5。
- 查找组合：MOT / tracking-by-detection＋probabilistic or generative motion＋identity/association loss＋gradient into predictor/generator＋stop-gradient or detach ablation。
- 必查引用链：DiffuTrack、DMMTracker、DiffMOT、TrajE、Constraint-Aware Flow Matching，以及它们的 related work 和 citing papers。
- 交付：最接近 3–7 篇的四轴对比；是否存在真正阻断 delta 的文献；可辩护的一句话 delta。

## Lane R2：FM 必要性与对照公平性

- 对应 claim：C1、C3。
- 检索对象：MOT 中的 deterministic/Gaussian/MDN/diffusion/flow motion predictor，以及多模态运动预测中的公平效率对比。
- 核心问题：哪些模型有精确密度；哪些必须采样；K、积分步数、延迟、参数量和训练预算应如何对齐。
- 交付：G3/G4 公平性检查清单；必须补充或删除的 baseline；不得混合比较的指标。

## Lane R3：竞争性解释与可否定性

- 对应 claim：C4，并约束 C1/C2。
- 主题：检测缺失、观测偏移、历史 ID 污染、相机运动与真实运动不确定性的区分。
- 必查文献：OA-SORT、DOA-SORT、OC-SORT 及其观测中心系先例，DiffuTrack 的失败分析。
- 交付：G2 D1/D2/D3 是否能区分上述解释；还缺少哪个最小诊断；什么结果应立即停止 FM 主线。

## 共同输出要求

1. 先读 [创新主张台账](innovation_registry.md)和[2026-11-10 重合审查](../../调研/研究监管_实验重合审查_2026-11-10/allinone.md)。
2. 把文献事实、代码事实、项目结果、推断和建议分开。
3. 只用原始论文、出版社/会议页、作者项目和可定位代码支持技术结论。
4. 负结果必须回传；“没找到”必须附检索式、日期、来源与失败记录。
5. 使用 [agent_handoff.md](agent_handoff.md) 作为最终回传格式。
