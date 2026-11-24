# 当前 A2A 交接入口

更新：2026-11-24。本文件是可更新导航；历史报告、协议和失败记录仍保留。

- **研究端最新复盘：** [S05–S06 本地审阅](../research_status_review_20261124.md)。确认原 Context-risk 门失败；保留 Context 无额外风险的描述性正向信号，但其相对 V 为 2 胜/2 负/4 平，收益集中，Local epoch5/Context epoch30 的选择和有效容量差异限制机制归因。本地复算 Git 内 441 个共同指标单元格、12 个清单文件哈希，不冒充服务器独立重评。

- **最近完成：** [S05 权威报告](../../reports/diagnostics/s05_visual_decision_coverage_20261123/README.md)。八段复用 development 上 M 的 pooled HOTA/AssA 胜 G，但仅 5/8 视频为正；fit24 同状态 M/G/V 普查得到 41 个 `g-h>0` 的完整可标注分量，跨 17 视频，预写 40/4 资源门通过。
- **决定与边界：** 同家族暂定完整性审计 `WARN` 仅因复用范围、离线 owner 代理和预训练重叠未知；确定性和官方指标复算通过。S05 只授权 S06 seed 13 可行性，不支持 learnability、闭环增益、新颖性或泛化主张。
- **权威协议：** S05 协议 `manifests/protocols/s05_visual_decision_coverage_20261123.yaml`；S06 seed13 协议 `manifests/protocols/s06_component_selector_seed13_20261123.yaml`，SHA256 `290a95e3e36e96dbc62cd20e7661a6b1c80b1627a2849586cb37c412f2afdf51`。后者在任何 S06 专用提取/训练/结果前冻结。
- **S06 权威状态：** [retry3 完整报告](../../reports/feasibility/s06_component_selector_seed13_retry3_20261124/README.md)。seed13 七条件自然闭环和官方 TrackEval 已完成；冻结 Context-risk 主门仅过 2/8，`claim_supported=no`、confidence high，决定 `STOP_S06_NO_SEEDS_37_73`。旧工程失败报告保留为 lineage，不再是最新状态。
- **主要结果：** Context-risk 相对 M `+0.094569 HOTA / +0.142736 AssA / +4 IDSW`，1/8 视频 HOTA 为正，并落后 V。Context-no-risk 是计划内次要描述：相对 M `+1.480101/+2.203603/−9 IDSW`，相对 V `+0.562045/+0.732083/−20`；不能替换主门或建立机制主张。
- **审计边界：** 同家族暂定 WARN；693 个官方指标和 378 个代理标签独立零差复现。训练按 action examples 而非冻结 components batching，且 Local/Context 有效容量不匹配；这削弱机制解释但不改变停止决定。范围仍是一 seed、复用 DanceTrack train development、外观预训练重叠未知。
- **精确下一步：** 不运行 seeds37/73，不修补后重跑当前门。建议仅用保存的 checkpoint 在固定 train18/internal-dev6 上做 CPU 预测体检，补 g/h 标签与预测、常数参照、M 零点误差、同 epoch 对照；补现有 0074/0032/0049/0023 决策配对及后续身份证据和 artifact 引用，最多一工作日/8 CPU 核小时。详见最新复盘第 6 节。未来 Context-no-risk 须另立主问题和评估协议，当前结果不自动授权新训练或性能试验。
- **GPU 与 Git：** retry3 训练显式 `CUDA_VISIBLE_DEVICES=1`，逻辑 `cuda:0→物理 GPU1`；全部回放/评测 CPU-only。物理 GPU0 未暴露给项目任务，但没有独立历史 telemetry 快照。服务器结果已进入本地 `54d03cc`；本轮研究端仅新增复盘与算术检查，未训练、未提交、未推送。

服务器每次交接应更新这里的“权威报告/完成范围/决定/限制/精确下一步”，不要仅追加“实验完成”。
