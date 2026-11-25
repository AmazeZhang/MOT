# 当前 A2A 交接入口

更新：2026-11-25。本文件是可更新导航；历史报告、协议和失败记录仍保留。

## 当前待执行：S08 CAMELTrack 基线闭环

- **任务书：** [S08 具体执行与回传要求](S08_CAMEL_BASELINE_TASK.md)；[版本化协议](../../manifests/protocols/s08_camel_baseline_20261125.yaml)。设计已给出，执行状态 `NOT_RUN`，运行锁待服务器生成；没有新模型结果。
- **执行顺序：** A来源/环境/划分锁 → B官方预训练推理与在线验收 → C随机训练/恢复smoke → D seed13从头20epoch单阶段训练 → E两条件dev8完整闭环 → F有界失败摘要。用户交给服务器执行后，门通过且预算内自动继续；F或阻断/预算上限处停止回传。
- **数据与范围：** 已有 motion_train32 / motion_dev8，均为历史使用的官方train；不要混用S06的evaluation8。官方三cue；P预训练参考、T随机初始化；P可能训练过dev8，不做同训练数据受控优劣结论。不是论文两阶段复现或创新验证。
- **预算：** 空闲物理GPU1–7中的单张4090，GPU0禁用；总≤24GPU卡时、≤72小时墙钟、主动工程≤2工作日；具体分阶段上限、修复和存储限制见任务书。
- **回传：** `reports/baselines/s08_camel_baseline_20261125/` 的标准报告包，加来源/实际split/解析配置/命令/运行锁及失败事件摘要。当前该目录尚未产生，不是已有结果链接。
- **精确下一步：** 服务器从A开始落实并冻结运行锁；不要仅回复环境安装成功或只给服务器文件路径。研究端收到包后核查可用性与自然失败，再确定一个方法问题。

## 长期协作规则

[研究端任务规范](../experiment_task_contract.md) → [任务书模板](../templates/server_experiment_task.md) → [服务器报告规范](../experiment_reporting.md) → [报告包模板](../templates/report_bundle/README.md)。研究端决定科学设计和行动边界；服务器落实环境/脚本并回传证据。无明确授权不commit/push。

## 最新已完成结果与停止范围

- [S07权威报告](../../reports/diagnostics/s07_s06_checkpoint_postmortem_20261124/README.md)与[研究端复盘](../research_status_review_20261125.md)：Context epoch30 dev MSE 0.069095，动作类型常数0.055877；无额外风险决策13/22，always-M14/22。未建立component-specific decision价值。案例仍缺同状态M和合法mask，不再重复同类补检。
- [S06 retry3权威报告](../../reports/feasibility/s06_component_selector_seed13_retry3_20261124/README.md)：Context-risk主门2/8，保持 `STOP_S06_NO_SEEDS_37_73`。Context-no-risk的单seed正向描述不替换主门；容量、选模及复用开发限制保留。S06的停止不禁止单独设计的S08平台验收。
- [S05权威报告](../../reports/diagnostics/s05_visual_decision_coverage_20261123/README.md)及[S05/S06复盘](../research_status_review_20261124.md)是历史依据，原任务已完成。S01–S04和更早分支见[进展日志](../progress_log.md)与[报告索引](../../reports/README.md)，不自动复跑。

本轮研究端实际只核查官方源码并完善任务/规则；没有服务器访问、环境安装、数据下载、训练、CUDA或新TrackEval。服务器交接时更新当前执行状态、权威报告、完成范围、偏离/限制和精确下一步，不只追加“实验完成”。
