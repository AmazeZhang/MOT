# E0–E3 结果到主张的决策

**日期：** 2026-11-19<br>
**审阅：** 新鲜 Codex 代理，同系列 provisional；E0–E3 实验完整性审计均有 `WARN` 范围或协议事项。当前环境未找到 `evidence_check.py`，因此没有额外的确定性 claim 文件预检；原始摘要与实验审计仍可逐项查验。

## 审阅对象与判定

待判断的宽主张是：“成熟轨迹即时再输出及可观察风险门控已经构成能改善内部身份连续性的 MOT 新方法。”**判定：不支持（claim_supported=no；对否定宽主张的信心高）。**

E0 在固定 831 事件中找到 424 个曾成熟输出、由正检测更新但未当帧输出的轨迹，同时有 242 个目标未由正检测更新。E1 的 U 在 CV 分支使 HOTA +0.477 点，但 IDSW +91、Frag +174、官方 FP +328；其内部匹配状态按构造完全不变。E2 的分数≥0.8 简单门控降低 U 的代价，但相对原 CV 仍多 39 次 IDSW、42 个官方 FP，且 0.8 来自看过同一开发集的标签。E3 在另 24 段 train-fit 上的短缺失空间补回 G 相对 E 仅 HOTA +0.050、FP +84，未通过冻结门槛，8 段留出没有运行。

因此可保留的窄结论是：**在这些开发视频和固定检测上，成熟轨迹再输出能改善部分可见连续性与 HOTA；E 是一个有明确代价的简单基线。** AssA/IDF1 的评测变化不证明内部关联修复。新颖性、稳定身份改善、跨范围泛化和独立确认均未建立。U/G 作为身份方法候选停止，不再在同一视频上调输出阈值。

## 下一次实验的唯一问题

若继续身份方法研究，先判定“关联决策本身是否有足够可利用的闭环空间”。E4 只在固定 train-fit 重现候选上，复制同一预帧 CV 状态；一支原样关联，另一支用 GT **特权**强制目标与注册正检测一对一配对，随后按同一 CV+E 规则各运行 4 帧。比较五帧同 ID 正确输出与其他锚定身份损伤。它是可证伪的上界诊断，不是在线可部署方法，也不构成官方 TrackEval 完整序列结果。

E4 的候选抽样、特权动作、分母、验收、预算与停止门见 `manifests/protocols/e4_oracle_association_headroom_20261119_retry3.yaml`。前三次失败均为工程或评价锚点验收，未产生完整研究结论，失败产物留在数据盘。若特权配对仍无足够净收益，则停止当前关联修复方向；若通过，只说明存在 headroom，下一阶段还需独立的信息来源与非 oracle 闭环方法。此前 D1 的窄范围阴性历史干预仍是约束，不能把本次设计解释为已经证明状态写入是主要原因。

**证据：** [E0](../reports/diagnostics/e0_reassociation_output_chain_20261119/README.md)、[E1](../reports/feasibility/e1_mature_reemit_factorial_20261119/README.md)、[E2](../reports/feasibility/e2_reemit_guard_20261119/README.md)、[E3](../reports/feasibility/e3_short_gap_support_20261119/README.md)及各自审计。审阅 trace 位于 Git 忽略的 `.aris/traces/result-to-claim/2026-11-19_run01/`。
