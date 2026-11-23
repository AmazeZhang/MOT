# S04：有界连续冻结视觉记忆的局部机制实验

日期：2026-11-22。状态：**运行、独立完整性审阅及结果到主张门完成**。这项实验接续 [S03 主张门](../../../docs/s03_claim_gate_20261122.md)，测试与“只跳过一次错写”不同的累积记忆机制；没有训练候选 A。事件/对照仍由 GT 离线选取，分支推理不使用未来 GT，因此这是特权局部诊断，不是在线方法或完整序列 TrackEval。

## 事前范围与运行

[协议](../../../manifests/protocols/s04_bounded_memory_freeze_20261122.yaml) SHA256 `b6e40f6a24d7c8f79a41e43e89ce21de403bc43f2872e01597c81285d04fd90f` 在新特征提取前写定：使用 `dancetrack0074/0032/0069/0015/0023/0080/0049/0083` 八段 DanceTrack **train**，共 8,903 帧。它们与 S01/S03 的 24 段 fit 不重叠，但在项目既有 R1 实验中曾被使用，不能称独立未接触测试。固定 S01 M 的检测、Deep-OC-SORT FP32 外观、CV+E、外观权重 0.25 和 EMA 0.9。

特征提取[运行锁](../../../manifests/runtime/s04_heldout_features1_20261122.json) SHA256 `4cf1cae88643a63f5e0f4ee1c9117cec5a172d352d456cd019f4ab409687bef3`，输入图片逐帧哈希清单在数据盘 `input_image_hashes.json`，SHA256 `3f7b55e803b46f8126f141549dbb9a78c8bd86294835521eba8ae87030e5d23a`。8 段共提取 76,123 条高分检测特征，0 无效，800 个固定键的缓存/直接重算通过。此步只使用物理 GPU1，显式 `CUDA_VISIBLE_DEVICES=1`；物理 GPU0 未使用。8 段特征各有独立输出、元数据及日志，队列状态在 `/media/imc/data/mot-data/runs/s04_bounded_memory_freeze_20261122/features/status.jsonl`。

M 基线[运行锁](../../../manifests/runtime/s04_M_baseline1_20261122.json) SHA256 `64cb02eca137193b536490b4397bfe0e95ae50c5e6c2112813d99e8b306fe32c`。8 段使用 S01 独立实现完整重放，逐段 `ENGINEERING_PASS`；每段轨迹、trace、锁、校验及日志位于数据盘 `baseline/<sequence>/`。基线与后续分支全部使用 `CUDA_VISIBLE_DEVICES=''`、CPU。

首次三份分支运行锁因版本字段和协议字段读取错误，在**输出目录创建及 GT 读取前**停止，失败锁与 `branch_run_failed_env1.log`、`branch_run_retry2.log`、`branch_run_retry3.log` 完整保留；没有从失败尝试看研究结果。修正后的[权威 retry4 运行锁](../../../manifests/runtime/s04_bounded_memory_freeze_retry4_20261122.json) SHA256 `16ac218cf4122938b1fa032e4895233b6bfb1dda1a0aacad249b0626462869cb`，只修复运行环境记录与协议字段读取，未改实验干预、事件定义或判定门。

## 干预、结果和原始证据

S04 用与 S02 相同的稳定历史/IoU 身份代理，在新八段 M 轨迹中记录全部 67 个首次异主更新，其中 15 个符合事前定义的错误组，分布于 5 段；各配一条同视频、避开全部首次异主更新 ±30 帧的干净对照。每个事件在同一预帧状态克隆三臂：事实 M、仅事件帧跳过一次目标写入、事件帧起连续至多 30 帧禁止目标所有视觉记忆写入。后两臂保持事件帧关联、tracker/CV 状态和输出与事实臂一致；未来各自自然闭环。其他轨迹记忆照常更新。三臂均对干净对照执行相同规则。

原始产物在 `/media/imc/data/mot-data/runs/s04_bounded_memory_freeze_20261122/branches/`。GT 读取前冻结 `frozen_sources.json`（SHA256 `db7762a7cf9add077918706386a191edd2b42c5e425cf791aff3a6d982e0ec27`），任何分支前冻结 `frozen_event_selection.json`（`b538f3a54eccb0ac2202021cfb2b37ee5d9e13a514e79f8df28e6c152bd5b807`）；全部三臂输出先以 `branches_frozen.json`（`afa6547c766ade1c2f3babe72ae1ccbb48f853c1ac3fe9c2edc9b18255a21ae9`）冻结，之后 GT 评分。`scores.jsonl` SHA256 `d385864931f6d8f406a52870cb6486055b33735339beca6c66307341cb8a3ee3`；`per_video.json` `bd850f649c1e2974c24c0e12924afc3db4c788a4129303ed03be8c6c7ce0b0a1`；`summary.json` `589323a38141cb9964d5a0ee4eef6ee1514896fed64e097223a3de070920e128`。

| 组别 | 事件数 | 至少两次写入被连续冻结 | 单次跳写改变未来输出 | 连续冻结改变未来输出 | 连续冻结−单次跳写的全部前帧输出锚点正确帧净差 |
|---|---:|---:|---:|---:|---:|
| 异主首次更新 | 15 | 15 | 0 | 0 | **0** |
| 干净更新对照 | 15 | 15 | 0 | 0 | **0** |

错误组每个事件连续冻结实际拦截 10–31 次目标写入；15/15 个事件在未来有目标列非快路径且 IoU≥0.3 的检测边。该原始暴露量**不等于**正确身份候选参与竞争且增量分数发生变化。独立实际 LAP 探针显示，严格的 owner-positive 竞争与连续冻结相对单次跳写分数变化交集为错误组 **14/15 例、133 帧**，干净对照 **15/15 例、241 帧**。余下一例的唯一 owner-positive 竞争发生在两臂记忆尚相同之时，不能据此判其增量机制阴性。三臂在全部 30 个窗口未来输出逐帧相同，目标、其他既有锚点和全部锚点的 GT 正确帧净差均为 0。事件数、视频数、15/15 多写入暴露及对照覆盖均达到预写可检验门；收益门未达到，运行摘要判 `NO_SIGNAL_STOP_CURRENT_RATIONALE`。这只说明**当前冻结 M、所选 train 事件与 30 帧窗口里，连续冻结目标记忆没有改变保存的输出**，不能证明外观记忆普遍无用或长期污染无害。30 个窗口共 900 次未来帧暴露，对应 604 个不同视频帧，最多 4 重；没有截断窗口，不能按 30 个独立样本推断。[独立完整性审阅](EXPERIMENT_AUDIT.md)为同家族暂定 **WARN**：9,371 次哈希比较、8,903 帧基线回放、全部 30 个三臂分支与计分复算一致，未发现虚假 GT、分数造高或死计分。当前结果不授权训练。

| 视频 | 错误事件 / 干净对照 | 多写入暴露的错误事件 | 连续冻结−单次跳写全部锚点正确帧净差 |
|---|---:|---:|---:|
| dancetrack0074 | 2 / 2 | 2 | 0 |
| dancetrack0032 | 0 / 0 | 0 | 0 |
| dancetrack0069 | 1 / 1 | 1 | 0 |
| dancetrack0015 | 2 / 2 | 2 | 0 |
| dancetrack0023 | 4 / 4 | 4 | 0 |
| dancetrack0080 | 0 / 0 | 0 | 0 |
| dancetrack0049 | 0 / 0 | 0 | 0 |
| dancetrack0083 | 6 / 6 | 6 | 0 |

## 下一步

已完成独立核对和[结果到主张门](../../../docs/s04_claim_gate_20261123.md)：同家族暂定判定 `claim_supported=no`，按预写决策停止当前有界记忆冻结的候选 A 训练理由；不在同一 15 个事件上调窗口或筛选找正例。若继续研究，需另提可证伪的在线可观测机制、独立来源及预写判定门。
