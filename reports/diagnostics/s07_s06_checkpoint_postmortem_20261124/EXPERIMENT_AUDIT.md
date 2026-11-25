# Experiment integrity audit

日期：2026-11-24。审阅者：fresh GPT-6.1-Sol ultra，同家族暂定审阅。`review_independence: same-family`，`acceptance_status: provisional`。

## 总体结论：WARN

checkpoint 诊断的数值证据成立；没有发现假 GT、自归一化造高、幽灵结果或挑选 checkpoint 隐藏失败。WARN 来自协议/实现分母偏差、案例 attempt1 错误、代理与复用范围，以及原 S06 的公平对照限制。

### A. GT 来源：PASS（代理限定）

标签来自 DanceTrack GT 的检测 IoU 匹配与过去三次一致 tracker owner，属于 GT 辅助局部代理，不是官方 tracking utility。S07 模型输入没有 GT 身份。

### B. 分数归一化：PASS

`g/k,h/k` 的分母是分量 tracker 数，不是预测最大值或模型自身统计。特征标准化只使用 train18 输入，误差保留原始比例单位。

### C. 结果存在性与数值：WARN

3,024 行预测覆盖 12 checkpoint 的全部固定可标注动作。独立复算 checkpoint 指标、常数、CSV 与同 epoch 差值 1,104 项，最大差 `1.6653345369377348e-16`；107 个哈希引用覆盖 70 个唯一文件，无不匹配。不同 Torch 版本 CPU 重推理最大绝对差 `5.960464477539063e-08`。

偏差：协议全局写视频等权，但 decision accuracy 实际按分量 pooled。主 MSE 正确使用视频等权。补充包保留两种分母，并明确是在看结果后发现。

### D. 执行路径：WARN

指标函数确实执行且产物存在。原 S06 Local 有 8,192 个结构性不活跃的邻边输入权重，Local/Context 不是有效容量严格匹配；训练还存在 action-example batching 与末批 `/128`，限制机制解释。

### E. 范围：WARN

只有 seed13；train18 104/199 个可标注分量，internal-dev6 22/43 个，dancetrack0066 没有标签样本。internal-dev6 已参与原 checkpoint 选择。15 个案例只覆盖指定四段的全部改动，不是 evaluation8 全部 19 个改动，也不是代表性随机样本。

### F. 类型：PASS

分类为 `real-GT-assisted local proxy diagnostics`，不是合成 GT，也不是新 TrackEval 性能实验。

## 案例审计

attempt1 把代码中明确来自 G proposal 的 `base_first_pairs` 错标为 baseline M，并混用了独立闭环 M owner；保留为无效失败。attempt2 修正为同状态 G-vs-applied，独立重建 868 项完全一致。但同状态 M 配对和精确合法 mask 不存在；4 个选 G 案例相对 G 差异为空，14 条配对的 GT owner 无法解析。30 帧后续是描述性 Hungarian IoU 代理，不是官方 IDSW 或因果归因。

## 主张影响

- 支持：后期 checkpoint 相对冻结的 action-agnostic 常数降低固定样本回归误差。
- 需限定：结果后 action-type 常数 internal-dev6 MSE `0.05587726359924876`，优于 Context epoch30 的 `0.06909482306121303`；原常数对照不足证明 component-specific learning。
- 不支持：可迁移动作排序、Context 机制优越、S06 恢复、泛化或新 tracking 性能结论。
- 路由：保持 S06 STOP；未来实验必须另立协议。
