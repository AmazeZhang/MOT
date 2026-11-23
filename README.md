# MOT 毕业设计研究

初始研究问题：在短时观测缺失与身份竞争下，运动分布能否改善在线多目标跟踪？Flow Matching（FM）是历史候选，目前暂缓；当前建议聚焦深度外观基线上的竞争关联决策，见下方最新交接。

**下一轮与报告要求（2026-11-23）：** [S05/S06 建议](docs/a2a/NEXT_ROUND_20261123.md)规定先补八段完整对照与一次有界候选覆盖检查，再决定小型学习方法；CAMELTrack 等先例已覆盖泛化的上下文关联，候选创新尚未成立。[AGENTS.md](AGENTS.md) 已加入 [报告规范 v1](docs/experiment_reporting.md)，使用 [可复制模板](docs/templates/report_bundle/README.md)回传完整精度总体/逐视频指标、诊断案例、训练曲线及来源。服务器从 [LATEST](docs/a2a/LATEST.md) 读取当前任务；新实验尚未运行。

**当前阶段（2026-11-23）：S01–S04 已完成。** 固定深度外观 EMA 基线在 24 段开发集上相对 CV+E 有 +2.46 HOTA、+3.32 AssA、+2.56 IDF1；这是已有视觉机制的正向基线，不是已确认的新方法或泛化结果。S03 单次跳写和 S04 连续冻结均未改变所选分支输出，停止据此启动候选 A 训练。下一步重心转向错误发生时的视觉/几何竞争与合法关联决策，先补稳定性对照和决策覆盖诊断，不继续扫冻结窗口。见 [S01–S04 最新复盘](docs/research_status_review_20261123.md)、[S01 结果](reports/feasibility/s01_visual_association_20261120/README.md)、[S04 结果](reports/diagnostics/s04_bounded_memory_freeze_20261122/README.md)。

**整体复盘（2026-11-20）：** [两周进展与方向评估](docs/research_status_review_20261120.md)区分了已获得的简单改进、机制发现与尚未成立的方法创新。建议收束当前规则链，先进行有边界的方向发散；不是继续执行旧实验清单。

**方向调研（2026-11-20）：** [方向评估与最小验证方案](调研/研究方向发散_2026-11-20/方向评估与最小验证方案.md)建议先建立深度视觉对照与 MOTIP 验收，再验证竞争身份污染或合法冲突集合决策；可靠记忆与 FM 暂缓。MOTIP2、早期 exposure-bias 工作等已覆盖部分直觉，不把候选当已确认创新。[检索与代码核查记录](调研/研究方向发散_2026-11-20/allinone.md)列出原始来源、API 失败和代码可用性限制。本轮未实现或运行新实验。

**服务器 agent 交接：** [S0/S1 任务书](docs/a2a/S01_SERVER_TASK.md)及[固定对照协议](manifests/protocols/s01_visual_association_handoff_20261120.yaml)已执行，真实数据接入、外观提取、固定对照与 MOTIP 单视频验收见[服务器回传](docs/a2a/s01_server_return.md)。后续 S02–S04 见[进展日志](docs/progress_log.md)与最新复盘；旧任务书保留作复现说明，不是新的待运行清单。

**历史阶段（2026-11-20，后续见上方最新状态）：P1、R1 当前选择器分支和 R2 P1 离线排序均已按预定门槛停止。E0/E1 找到“匹配并更新但未输出”的断点，E2 的简单门控有开发集收益与代价，E3 的短缺失补回未过 fit 门。E4 的 GT 特权局部反事实证明选中错配事件存在短程收益；E5 用固定 HSV 外观选择可观察事件，在 24 段 train-fit 上相对配平几何规则净差 −6 帧、其他锚点损伤 422 对 420，未过晋级门，8 段留出未运行。事后只读分析显示 576 次动作中 529 次两规则同选，414 次共同损伤其他锚点；正常首轮常已把候选检测配给别的身份。** 目前仍无经独立确认的新方法或新颖性结论。当前外观选择器停止，尚无已验证的安全在线触发信号；只有提出实质不同的可观察机制后才启动新协议。见 [E5 失误分析](reports/diagnostics/e5_error_analysis_20261120/README.md)、[E5 实验报告](reports/feasibility/e5_nonoracle_appearance_local_20261119/README.md)、[E5 主张决策](docs/e5_claim_gate_20261120.md)、[进展日志](docs/progress_log.md)；[11 月 19 日复盘](docs/experiment_review_20261119.md)保留当时的执行建议。

R1 的原始设计见 [P1 停止后的研究路线建议](docs/research_route_proposal_20261113.md)。[决策规则对照执行草案](docs/next_branch_decision_rule_plan_20261118.md) 已产生独立冻结的 R2 P1 实验，其余阶段未执行；草案中的归因推断和三条件设计需按最新复盘修正，不能直接作为下一轮运行清单。

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

手册状态更新：2026-11-13。G1/G2 与 P1 离线、自然闭环及一次机制诊断均已完成；P1 闭环负结果保留，当前分支停止。早期实验矩阵为候选设计，不是当前待运行清单。
