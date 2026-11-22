# S03：单次视觉记忆写入的受控局部分支

日期：2026-11-22。状态：**COMPLETE_PRIVILEGED_DIAGNOSTIC；完整性审阅 WARN（同家族暂定）**。本实验复用 24 段 DanceTrack train-fit 与 S01 M 冻结输入，只评估 GT 条件化选中的局部机制；没有训练候选 A、使用 heldout/val/test 或执行完整序列新方法评测。全程 `CUDA_VISIBLE_DEVICES=''`、CPU，物理 GPU0 未使用。

## 运行来源与历史

[单视频克隆工程协议](../../../manifests/protocols/s03_memory_clone_smoke_20261121.yaml)及[运行锁](../../../manifests/runtime/s03_memory_clone_smoke1_20261121.json)先验证 `dancetrack0087:f195:t1`：完整 M 回放和事件事实分支与 S01 逐字节相同；仅跳过该次目标视觉记忆写入，当帧输出与 tracker 状态相同。工程验收文件位于 `/media/imc/data/mot-data/runs/s03_memory_clone_smoke_20261121/engineering_checks.json`，SHA256 `e90c968b7ceed412e12e70faa086742cc967ca74246bb61f3fe2bb3141f5cd9b`。smoke 不读取原始 GT 或评分，但事件来自 S02 的 GT 辅助清单。

[首版协议](../../../manifests/protocols/s03_single_memory_write_counterfactual_20261121.yaml)和[运行锁](../../../manifests/runtime/s03_single_memory_write_counterfactual1_20261121.json)保留在仓库，原始结果保留于 `/media/imc/data/mot-data/runs/s03_single_memory_write_counterfactual_20261121/`；原始摘要 SHA256 `20699a65eff5384f4bfbb9a9edffd18e7084c4059886b496e02ad0b2961ef0cc`。首版代码选干净对照时仅避开 86 个 S02 可干预首次错写事件的 ±30 帧；按“全部 M 首次事件”的更严格解释，有 40/86 个对照落在其余首次事件附近。首版 86 个错误事件的零输出结果保留，不倒改。

在已知首版零结果后，另建[retry2 协议](../../../manifests/protocols/s03_single_memory_write_counterfactual_retry2_20261122.yaml) SHA256 `960becb0af4842176433834573a7f2df6f5bc08b797965f9b2a3e7d71bf12cee`、[运行锁](../../../manifests/runtime/s03_single_memory_write_counterfactual_retry2_20261122.json) SHA256 `72948a9ffd5a8b19e22b1413a68733b3de1a92ca8e85f62c1b18da17bc2050f2`，将同视频**全部 278 个 M 首次事件**的 ±30 帧排除。修正版的 86 个错误事件与首版相同，重新选出 86 个干净对照，其中 40 个更换；全部新对照距任一同视频 M 首次事件至少 31 帧。这是控制口径修复重跑，**不是独立盲化确认实验**。新对照匹配成本 min/median/max 为 `0.0562/0.3865/3.1305`，无匹配距离上限；86/86 配对覆盖不等于状态平衡。

## 权威 retry2 结果

原始文件位于 `/media/imc/data/mot-data/runs/s03_single_memory_write_counterfactual_20261122_retry2/`。运行先冻结来源 `frozen_sources.json`（SHA256 `f814db15b0cc16ead74633b570e232a9633eb4f0ac4d5e311df86e48a582b8e9`）和事件 `frozen_event_selection.json`（`5c71eafe1e3d11e4a021298b2f3490a6bcf238927434b809cd1babc01965bccb`）；24 段 S01 M 事实全序列回放的轨迹/trace 完全一致。分支文件先以 `branches_frozen.json`（`19320a027e1cd43b2604008eab0e53cf6bae2494b38d4c0a4ad3525eb8c9214b`）冻结，再用 GT 离线评分。`scores.jsonl` SHA256 `8bdc92423012bb0e75dfead6d9c090b9fbd7a03131b3b6532f49bb42de202e27`；`per_video.json` `b3995df3616c1bb71e3aed450f0a1b6f81181066ac98f9104bcf2b00f731e006`；`summary.json` `12293779fa5ceb1930eb9f59d0c99f07ee670d5bcfb3a38cbf6e8c4c7cd472bc`。

| 事件组 | 分支数 | 事件帧目标正检测有合法边 | 任一未来输出行变化 | 未来同 ID 正确帧净差：目标 / 其他既有锚点 / 全部既有锚点 |
|---|---:|---:|---:|---:|
| 异主写入，跳过该次写入 | 86 | 70 | **0** | **0 / 0 / 0** |
| 配平干净写入，跳过该次写入 | 86 | 85 | **0** | **0 / 0 / 0** |

事件帧合法边按 CV 调整后的首轮预测框与当前高分正检测 IoU≥0.3 定义。错误组的 70 个合法边覆盖 21 段视频；它们不是后续分支中目标记忆实际改变决策的保证。86 个错误分支在排除配对 ID 元数据后与首版逐项相同，46 个保留的干净分支也相同。修正版 172/172 个分支的事实与跳写未来输出行一致，因此全部局部 GT 正确帧差为零。投入数量门通过只表示覆盖足够，**不表示机制成功**。

[事后仪表补检协议](../../../manifests/protocols/s03_instrumentation_addendum_20261122.yaml)和[运行锁](../../../manifests/runtime/s03_instrumentation_addendum1_20261122.json)针对首版，摘要 SHA256 `e2c5be870f2bb9215d8594e9e100c3541079442015f498594960b3d69ce4de7c`。独立审阅又从冻结源重建 retry2 的目标 EMA：172/172 次目标记忆向量扰动非零；错误组 L2 min/median/max `0.0158/0.0539/0.1037`，新对照 `0.0075/0.0441/0.0845`。修正版未来 trace 差异仅在 `memory_last_write_frames`，错误/干净组分别 153/98 帧。实际错误事件的后续一帧 probe 中，目标向量改动了外观融合矩阵两个单元，最大绝对差约 `0.00458`，但输出未变。这说明干预到达了真实打分路径；不证明所有后续候选边都受目标向量影响。

## 范围、审阅与决策

独立完整性审阅[记录](EXPERIMENT_AUDIT.md)为 **WARN、同家族 provisional**：339 项哈希核对、24 视频 25,328 帧源回放、全部 172 个事实分支与独立 IoU/Hungarian 评分吻合；代表性错误/干净事件的完整 tracker/history 状态探针一致，除目标记忆外无差异。记录的限制是初版控制排除偏差、retry2 为事后修正、对照匹配无平衡保证、局部窗口依赖、前帧输出锚点范围，以及未做全部事件的完整隐藏状态逐字段核验。

retry2 的 172 个未来窗口名义暴露 5,106 帧次，只覆盖 3,562 个不同视频帧；93 对窗口重叠，最大 6 重。5 个窗口因视频末尾少于 30 个未来帧（错误 3、干净 2），最短 4 帧。目标身份属于前帧**输出**锚点的事件为错误 61/86、干净 75/86；单独目标指标覆盖全部目标。事件标签基于 IoU≥0.5 的 GT 离线代理，外观权重可能与 train 数据重叠。这里不做事件级独立样本显著性推断，也不报告完整序列 TrackEval 增益。

**可支持的窄结论：** 在冻结 M、这 86 个 GT 辅助选中的错误事件及至多 30 帧闭环窗口里，保持当帧错误关联和其他状态不变，仅禁止目标一次视觉记忆写入，没有改变保存的跟踪输出。这个结果不能证明外观记忆普遍无用、多次错误写入不会累积危害，或另一个训练方案无效。按原先 S03 停止条件，本轮没有支持直接训练候选 A 的单次错写机制信号；新上下文 `result-to-claim` 判该训练主张 **no**、当前停止判断高信心（同家族暂定）。详见[主张门记录](../../../docs/s03_claim_gate_20261122.md)。后续需先另行提出可观测、与本干预不同的机制及独立验证设计。
