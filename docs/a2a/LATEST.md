# 当前 A2A 交接入口

更新：2026-11-25。本文件是可更新导航；历史报告、协议和失败记录仍保留。

## 最新已完成：S08 CAMELTrack 有界基线闭环（`COMPLETED` / `CONTINUE`）

- **任务与协议：** [S08 任务书](S08_CAMEL_BASELINE_TASK.md)；[版本化协议](../../manifests/protocols/s08_camel_baseline_20261125.yaml)。执行状态：`COMPLETED`（A–F 全部完成，权威报告：[S08 报告](../../reports/baselines/s08_camel_baseline_20261125/README.md)）。
- **主要结论与指标（`motion_dev8`，7,565 帧）：**
  - **平台验收：** 阶段 A–C 全部通过（无 GT / 截断未来 1–200 前缀 / 重跑均一致；step 50 新进程断点恢复后参数/优化器/scheduler/step/RNG 精确一致；`T_SCRATCH` 43.48M 参数在物理 GPU 1 上 20 epochs / 8,000 步无 NaN 收敛，峰值显存 2,801.8 MiB，全流程总耗时 4.465 GPU 卡时，GPU 0 禁用且未使用）。
  - **TrackEval `COMBINED_SEQ`：** `T_SCRATCH`（seed 13，train32 单阶段 20 epoch `last.ckpt`）取得 HOTA `53.981`、DetA `81.018`、AssA `36.096`、IDF1 `56.695`、MOTA `90.621`、IDSW `1451`、FP `1721`、FN `4434`、Frag `1428`；`P_OFFICIAL` 平台参考为 HOTA `66.993`、DetA `82.524`、AssA `54.516`、IDF1 `71.763`、MOTA `91.591`、IDSW `638`。
  - **自然失败（F 阶段）：** `T_SCRATCH` 在 `dev8` 共产生 `1,451` 个官方 TrackEval IDSW 事件（每视频按固定 SHA256 抽样最多 2 例、共 16 例，见 `failure_inventory.csv` / `cases.jsonl` / `diagnostics.json`）。全部 1,451 事件中，76.71%（1,113/1,451）存在 IoU≥0.3 的同帧竞争 GT 框，40.73%（591/1,451）发生于匹配中断 >1 帧之后，两者重叠 33.91%（492/1,451）。
- **限制与范围：** `motion_train32` 与 `motion_dev8` 均为项目历史复用划分，非新留出确认集；`P_OFFICIAL` 为官方两阶段/train+val 权重，与 `T_SCRATCH` 仅作平台参考对比；未启动两阶段训练、新增模块、阈值网格或 S06 续跑。
- **精确下一步（停在 F 交接研究端）：** 本轮已于阶段 F 结束并停止自动推进。请研究端审阅 [S08 报告](../../reports/baselines/s08_camel_baseline_20261125/README.md) 与 `failure_inventory.csv` / `cases.jsonl`，据此确定下一轮唯一的方法改进问题与任务书；未经明确授权不执行 `git commit` / `git push`。

## 长期协作规则

[研究端任务规范](../experiment_task_contract.md) → [任务书模板](../templates/server_experiment_task.md) → [服务器报告规范](../experiment_reporting.md) → [报告包模板](../templates/report_bundle/README.md)。研究端决定科学设计和行动边界；服务器落实环境/脚本并回传证据。无明确授权不commit/push。

## 最新已完成结果与停止范围

- [S07权威报告](../../reports/diagnostics/s07_s06_checkpoint_postmortem_20261124/README.md)与[研究端复盘](../research_status_review_20261125.md)：Context epoch30 dev MSE 0.069095，动作类型常数0.055877；无额外风险决策13/22，always-M14/22。未建立component-specific decision价值。案例仍缺同状态M和合法mask，不再重复同类补检。
- [S06 retry3权威报告](../../reports/feasibility/s06_component_selector_seed13_retry3_20261124/README.md)：Context-risk主门2/8，保持 `STOP_S06_NO_SEEDS_37_73`。Context-no-risk的单seed正向描述不替换主门；容量、选模及复用开发限制保留。S06的停止不禁止单独设计的S08平台验收。
- [S05权威报告](../../reports/diagnostics/s05_visual_decision_coverage_20261123/README.md)及[S05/S06复盘](../research_status_review_20261124.md)是历史依据，原任务已完成。S01–S04和更早分支见[进展日志](../progress_log.md)与[报告索引](../../reports/README.md)，不自动复跑。

本轮研究端实际只核查官方源码并完善任务/规则；没有服务器访问、环境安装、数据下载、训练、CUDA或新TrackEval。服务器交接时更新当前执行状态、权威报告、完成范围、偏离/限制和精确下一步，不只追加“实验完成”。
