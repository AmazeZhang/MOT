# Step 1 — Decompose the Novelty

日期：2026-11-03（Asia/Shanghai）

研究目的：硕士毕业设计选题；探索阶段；资源为 8 张 RTX 4090 与 1 张 RTX 3060 12GB。
流程：deep-research / lit-review，结合科学批判与先例重合核查。输出中文 Markdown 报告及可追溯证据记录。此次不运行真实 MOT 训练或评测。

- Problem framing：固定检测输入的单摄像机在线 MOT；在交叉、短时缺失、非线性运动时提高身份关联正确性。
- Core mechanism：历史检测、缺失掩码、邻域作为条件的低维 Flow Matching；以生成样本构建匹配分数；身份正负匹配监督回传生成器；可选风险校准与自适应采样。
- Key insight：位置预测误差与关联决策误差不等价；需要让分布服务于身份竞争，并检验预测不确定性的实际用途。
- Application domain：DanceTrack 为主，SportsMOT/MOT17 为外部验证；不使用未来帧推理。

初始计划：1 分解主张；2 检索去重；3 摘要初筛；4 选择最近工作；5 正文核查；6 四轴比较；7 明确差异及可证伪实验。

2026-11-03 更新：本轮七步调研已完成，过程见 `检索与核查记录.md`，结论见 `FlowMatching_MOT_批判性调研报告.md`。方法仍为研究假设，不构成已验证创新或性能承诺；没有执行真实训练或 MOT 复现。
