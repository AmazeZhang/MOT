# MOT 毕业设计研究

研究问题：在短时观测缺失与身份竞争下，运动分布能否改善在线多目标跟踪？Flow Matching（FM）是候选实现，需要与确定性预测、高斯和混合密度网络（MDN）比较。

**当前阶段：确定性 P1 已通过三 seed 完整 motion-train→motion-dev 离线可行性门，并获准进入自然闭环。** 相对最强零-delta 对照，宏平均 normalized MSE 比值为 0.9156、gap2--31 为 0.9256，宏平均 IoU 增加 0.00687；但 gap16--31 IoU 平均下降 0.00811。独立审计为 same-family provisional WARN。下一步冻结自然 tracker 历史、官方指标和副作用门；尚无新方法闭环跟踪效果结果。

## 阅读顺序

准备去服务器安装时，直接从 [服务器环境配置指南](服务器环境配置指南.md) 开始：按步骤安装、运行 GPU 验收并保存反馈信息。

1. [实验手册总览](docs/experiment_handbook/00_overview.md)：当前需求、优先级和每阶段交付物。
2. [验证路线与实验矩阵](docs/experiment_handbook/01_validation_plan.md)：先验证什么、如何对照、何时继续或调整。
3. [服务器与环境配置](docs/experiment_handbook/02_environment.md)：硬件盘点、候选版本、安装和验收步骤。
4. [数据、检测缓存与评测](docs/experiment_handbook/03_data_and_evaluation.md)：官方下载、目录、划分、权重来源和 TrackEval。
5. [实现步骤与接口约定](docs/experiment_handbook/04_implementation.md)：按依赖顺序实现的任务、验收条件和边界。
6. [服务器反馈与协作](docs/experiment_handbook/05_server_handoff.md)：你需要反馈的信息、运行记录和后续远程协作。
7. [研究监管与创新有效性台账](docs/research_governance/README.md)：记录每个创新主张的最近先例、实验证据、审查状态和人工决策点，供本项目与科研 agent 并行更新。
8. [项目目录与产物约定](docs/project_structure.md)：代码、配置、来源清单、本地环境、数据缓存和实验输出分别放在哪里。
9. [项目进展日志](docs/progress_log.md)：每阶段实际完成内容、验收证据、限制和下一步。
10. [轻量实验凭证](reports/README.md)：进入 Git 的官方评测摘要、逐序列指标和产物哈希；大型原始输出仍留在数据盘。
11. [当前研究状态总结](docs/current_status_20261112.md)：截至 2026-11-12 的主要发现、证据边界、阶段位置、产物路径和精确下一步。
12. [已完成实验复盘与下一轮安排](docs/experiment_review_20261112.md)：判断是否继续原计划；明确先做 P1 自然闭环、暂缓概率模型矩阵，以及最小对照与决策分支。

可填写模板：[服务器信息](docs/templates/server_inventory.md)、[实验记录](docs/templates/experiment_record.md)、[决策记录](docs/templates/decision_record.md)、[实验清单](docs/templates/experiment_matrix.csv)。模板目前没有真实实验结果。

前期依据：[2026-11-03 批判性调研](调研/FlowMatching_深入调研_2026-11-03/FlowMatching_MOT_批判性调研报告.md)、[检索核查记录](调研/FlowMatching_深入调研_2026-11-03/检索与核查记录.md)、[2026-11-10 当前实验重合审查](调研/研究监管_实验重合审查_2026-11-10/allinone.md)。调研中的论文 PDF、原始检索返回与代码阅读快照仅保存在本地，不随仓库分发；原始来源链接保留在报告中。

## 第一轮的范围

- DanceTrack train/val；固定检测输入；严格在线；先不启用 ReID、补框或离线插值。
- OC-SORT 官方实现作为参考基线；研究接口的零改动版本必须先通过一致性检查。
- 单张 4090 起步。其他卡用于独立实验；3060 用于小样本与环境验证。
- 先完成数据/评测闭环与错误归因，再比较简单预测器和 FM；有证据后再加入关联监督。

数据、权重、原始输出、环境私有配置放在仓库外或被忽略的目录。Git 保存代码、配置、划分清单、来源哈希和经整理的实验摘要。后续引入第三方代码时单独记录版本与许可证，不将整个外部仓库混入本项目历史。

手册版本：2026-11-12，v0.10。G1 train/val 基线、G2 首轮归因和 D1--D5 事件级特权诊断已完成，P1 数据契约已验收；这些诊断和数据构建都不是方法效果，待实现方法仍没有闭环对照支持。
