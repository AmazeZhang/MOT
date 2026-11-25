# S08 CAMELTrack 有界基线训练与评测闭环

规范：[experiment_reporting.md](../../../docs/experiment_reporting.md)；任务书：[S08_CAMEL_BASELINE_TASK.md](../../../docs/a2a/S08_CAMEL_BASELINE_TASK.md)；协议：[s08_camel_baseline_20261125.yaml](../../../manifests/protocols/s08_camel_baseline_20261125.yaml)。

## 一页结论

- **experiment ID / 状态 / 决定 / 日期：** `s08_camel_baseline_20261125` / `COMPLETED`（A–F 全部完成） / **`CONTINUE`**（仅表示基线平台可用于下一轮单问题设计，不自动启动后续训练） / 2026-11-25T15:38:46.314175+08:00。
- **检验的问题：** 能否在单张 RTX 4090 D（物理 GPU 1，禁用 GPU 0）上建立可训练、可重载、严格在线、可导出官方指标的 CAMELTrack 成熟学习关联基线平台，并获得 `motion_dev8` 上第一份自然失败分布。
- **实际执行与未执行：** 完成 A（来源/划分/依赖锁）→ B（`P_OFFICIAL` 在 `0027` 的在线/去 GT/截断未来边界验收）→ C（`T_SCRATCH_SMOKE` 100 步与 step-50 新进程断点恢复/重载验证）→ D（`T_SCRATCH` seed 13 在 `motion_train32` 从头 20 epochs 训练，8,000 steps）→ E（`P_OFFICIAL` 与 `T_SCRATCH` 在 `motion_dev8` GT-free 状态上独立闭环跟踪与 TrackEval）→ F（`T_SCRATCH` 官方 IDSW 事件提取与最多 16 例抽样）。未执行论文两阶段 TE 预训练、train+val 训练、多 seed、阈值搜索或新增模块。
- **最重要结果：**
  - 平台工程全部通过：`T_SCRATCH`（43,475,200 参数，FP32 batch 8，峰值显存 2801.8 MiB，耗时 249.4 分钟）20 epochs 无 NaN 收敛至 epoch 20 `last.ckpt`（`val/sim_auroc = 0.9922`、`val/sim_acc = 0.9532`）。
  - `motion_dev8` 官方 `COMBINED_SEQ`（总帧数 7565）：`T_SCRATCH`（seed 13，仅 train32 单阶段）取得 HOTA **53.981**、DetA 81.018、AssA 36.096、IDF1 56.695、MOTA 90.621、IDSW 1451、FP 1721、FN 4434、Frag 1428；`P_OFFICIAL` 平台参考为 HOTA **66.993**、AssA 54.516、IDF1 71.763、IDSW 638。
  - 失败分布（F 阶段）：`T_SCRATCH` 在 `dev8` 共产生 **1451** 个官方 TrackEval IDSW 事件（逐视频与 `pedestrian_detailed.csv` 完全一致）；其中存在竞争 GT 框（IoU≥0.3）占 **1113/1451 (76.7%)**，发生于匹配中断 >1 帧之后占 **591/1451 (40.7%)**，两者重叠 **492/1451 (33.9%)**，均无者 **239/1451 (16.5%)**。按固定 SHA256 规则每视频抽样最多 2 例、共 **16** 例（`failure_inventory.csv` / `cases.jsonl`）。
- **精确下一步：** 停止本轮并交由研究端审阅 `T_SCRATCH` 的 `dev8` 指标、1,451 例全部 IDSW 描述统计与 16 例自然失败案例，再决定下一轮唯一的聚焦方法问题（当前证据中拥挤竞争标记覆盖 76.7% 且常与短中断重叠，供研究端权衡）。

## 设计与可比较性

- **协议与锁：** `s08_camel_baseline_20261125.yaml`（SHA256 `dbbe9a777cab2f7b47d330117ba5040e4863da912e5090d76d4768a6e21b77f9`），上游 CAMELTrack `bc1c261075b0beacfe71c2896453be83643153c8`，完整环境锁见 `runtime_lock.json`，解析配置见 `resolved_configs/`。
- **条件与划分：** 沿用历史使用的 DanceTrack `train` 内部互斥划分 `motion_train32`（32 段，34,231 帧）与 `motion_dev8`（8 段，7,565 帧），未接触 `val` 或 `test`。推理统一使用 GT-free 状态文件与共享阈值（`min_det_conf=0.4`、`min_init_det_conf=0.9`、`sim_threshold=0.1`、`use_computed_sim_threshold=false`）。
- **可比较性限制：** `P_OFFICIAL` 为官方预训练权重（README 记载使用 train+val 两阶段训练），不能排除见过 `dev8`，因此 `T_SCRATCH - P_OFFICIAL` 仅作平台参考描述，不解释为同数据方法差异；也不与 S06 八段结果做跨输入/跨范围相减。

## 结果（`motion_dev8` 官方 TrackEval）

完整精度数值见 [aggregate_metrics.csv](aggregate_metrics.csv) 与 [per_sequence_metrics.csv](per_sequence_metrics.csv)。

| 条件 | Seed | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P_OFFICIAL` (参考) | NA | 66.993 | 82.524 | 54.516 | 71.763 | 91.591 | 638 | 1646 | 4536 | 1397 |
| `T_SCRATCH` (train32 20ep) | 13 | 53.981 | 81.018 | 36.096 | 56.695 | 90.621 | 1451 | 1721 | 4434 | 1428 |

- 逐视频 `T_SCRATCH − P_OFFICIAL` HOTA 胜/平/负：0/0/8（中位差 -13.562，最差视频 `dancetrack0037` -25.499，总帧数 7565）。
- 训练曲线见 [learning_curves.csv](learning_curves.csv)，自然失败清单一览见 [failure_inventory.csv](failure_inventory.csv)、[diagnostics.json](diagnostics.json) 与 [cases.jsonl](cases.jsonl)。

## 验证、资源与交接

- **GPU 与资源：** 仅使用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1` -> `cuda:0`），物理 GPU 0 禁用且未使用；全流程权威运行总计 **4.465 GPU-hours**（上限 24 卡时），Stage D 峰值显存 2801.8 MiB。
- **检查项：** 详见 [checks.json](checks.json) 与 [artifacts.json](artifacts.json)。
- **大型产物路径：** 保留在 `/media/imc/data/mot-data/runs/s08_camel_baseline_20261125/` 与 `/media/imc/data/mot-data/cache/s08_camel/`。
