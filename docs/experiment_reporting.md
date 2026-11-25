# 实验报告与 Git 回传规范 v1

生效日期：2026-11-23。适用于服务器 agent 和本地研究端；规则入口是 [AGENTS.md](../AGENTS.md)。目标是研究端仅凭 Git 就能判断“跑了什么、效果如何、为什么这样解释、还缺什么”。大型原始产物继续在 `/media/imc/data/mot-data`，不为报告而重新跑已经可核验的实验。

研究端下发前须同时遵守 [实验指导与任务规范](experiment_task_contract.md)，使用 [任务书模板](templates/server_experiment_task.md) 明确实验矩阵、前置门、预算、自动继续和停止边界。报告中的 `protocol` 指向本次实际执行快照；设计稿不等于已冻结或已运行。

## 1. 每次交付是什么

一个实验系列可用一个报告目录，里面每个配置、种子和重试都有唯一 `run_id`。路径沿用 `reports/{engineering,diagnostics,feasibility,baselines}/<experiment_id>/`。先复制 [模板目录](templates/report_bundle/README.md)，填真实信息，再更新 [报告索引](../reports/README.md)、[进展日志](progress_log.md)、[最新交接](a2a/LATEST.md)。不要求为每个 epoch 单独写报告。

| 文件 | 要求 | 用途 |
|---|---|---|
| `README.md` | 所有报告 | 问题、设计、结果、解释、限制、决定、下一步；一页摘要优先 |
| `run_manifest.json` | 所有报告 | 运行身份、配置、数据范围、全部计划/实际 run、时间和资源、版本与来源 |
| `checks.json` | 所有报告 | 实际验收项、状态和证据定位；不能只有“全部通过” |
| `artifacts.json` | 所有报告 | Git 内小文件及服务器原始文件的路径、用途、大小、SHA256、可访问范围 |
| `aggregate_metrics.csv` | 有完整序列跟踪评测时必需 | 每个 condition × seed × scope 的官方总体指标 |
| `per_sequence_metrics.csv` | 同上 | 上述每个组合下全部预期视频，包括失败/未完成行 |
| `diagnostics.json` | 机制、局部分支、关联改动研究必需 | 定义、分子分母、分层统计、有效覆盖、重叠与其他身份损伤 |
| `cases.jsonl` 和 `evidence/` | 声称机制、成功恢复或失败类型时必需 | 可核查案例与小型分数/配对/轨迹片段；无法提供须说明原因 |
| `learning_curves.csv` | 有训练时必需 | 全种子损失、开发指标、训练进度、checkpoint 选择轨迹 |

纯工程测试不强制产生 HOTA；纯局部诊断不伪造完整 TrackEval。没有运行某项时，manifest 的 `omissions` 写文件名和原因，不制造零值填空。已开始但失败的跟踪组合仍在计划清单和指标表中保留行。

运行状态统一使用 `NOT_RUN / RUNNING / PARTIAL / COMPLETED / FAILED`；`TEMPLATE` 只属于模板。状态表示工作完成情况，不表示研究成功。科学决定另写 `CONTINUE / STOP_CURRENT_METHOD / INCONCLUSIVE / ENGINEERING_ONLY` 及理由。报告系列有多组时，缺任何计划组即不得称整个系列 `COMPLETED`；取消组须记录协议修订，不能事后删除计划。

## 2. 开跑前锁定什么

协议至少包含：可否定假设、最强对照、唯一变化因素、在线可用输入、GT 使用边界、完整序列列表与历史使用情况、训练/开发/最终评估职责、种子、主指标和代价指标、调参/选 checkpoint 规则、继续/停止规则与预算。新方案先登记到 [创新台账](research_governance/innovation_registry.md)，候选不能写成已成立结论。

记录冻结时间（ISO 8601 含时区）、协议路径/哈希和当时已知结果。运行前协议与运行后修改分别保存；结果后制定的规则只能叫探索性。无需为了冻结协议违反“不主动提交 Git”的规则：保留带时间的版本文件、实际执行快照及事后披露，不能假装完成外部预注册。

manifest 至少逐项提供：

- 代码：本项目 commit、dirty 状态、实际执行源文件清单/哈希；dirty 时保存与 HEAD 的相关安全补丁及未跟踪源码快照。第三方 commit、相关补丁、评测器版本、导出器版本也记录。只记 HEAD 不能复现未提交代码。
- 输入：dataset/revision、split 清单与哈希、精确视频/帧范围、检测缓存与外观缓存索引/哈希、预处理、权重来源/SHA256、已知训练数据及重叠审计。未知写 `unknown`，不推定无泄漏。
- 配置：每个条件的完整配置与准确命令、工作目录、训练/推理种子、baseline run、相较基线改变的所有因素。秘密变量保留名称而不导出值。
- 环境与资源：机器代号、OS/Python/关键库/驱动/CUDA、GPU 型号；物理 GPU 号、`CUDA_VISIBLE_DEVICES` 与逻辑映射；开始/结束时间、退出码、墙钟/CPU 核时/GPU 卡时、峰值显存和测量方法。CPU-only 明记 `CUDA_VISIBLE_DEVICES=''`。详细机器记录放忽略的 `local/`。
- 运行谱系：每次 attempt 的状态、失败原因、父 run、替代关系、权威 eval ID、复用产物来源及哈希验证。重评与重训分开，不覆盖第一次失败或不利结果。

## 3. 指标表的统一口径

两张 CSV 使用相同列（见模板）；每行键为 `(run_id, condition, seed, scope_id, sequence)`。`seed` 对确定性非训练基线写 `NA`；学习模型写真实训练 seed，额外推理 seed 在 manifest 中登记。`scope_id` 在 manifest 映射到固定序列列表、帧范围、数据使用历史和 evaluator 配置。总体表的 `sequence` 固定为 `COMBINED_SEQ`，逐视频表使用真实名称。

必备列：`run_id,condition,seed,scope_id,sequence,status,num_frames,HOTA,DetA,AssA,IDF1,MOTA,IDSW,FP,FN,Frag,source_artifact_id,source_locator,missing_reason`。可追加列，不能暗改已有语义。`source_locator` 指向原始 JSON key、CSV 行/列或评测结果位置；artifact ID 在清单中可解析。

- 比例指标统一导出为 **0–100 分**，差值是百分点；MOTA 可为负，不截断。计数为整数。数值保留原始评测器可提供的精度；README 才四舍五入。若唯一存档只有两位小数，明确精度限制，不能补零冒充原始精度。
- HOTA/DetA/AssA 使用官方评测器跨阈值处理后的值；`COMBINED_SEQ` 必须来自官方聚合，不能取逐视频简单平均，也不能用 `sqrt(mean(DetA)*mean(AssA))` 代替 HOTA。
- 源 summary 未提供 Frag 等字段时，空值并注明缺失字段和原因；不能将缺失写成 0。JSON 用 `null`，不接受 NaN/Infinity；CSV 空单元格同时有 `missing_reason`。主指标未完成则状态必须反映这一点。
- 部分视频完成可导出 `PARTIAL` 总体行，但给它单独 scope，清楚列出覆盖。不得与完整 baseline scope 直接相减。需要比较时，对双方共同完成范围另做相同官方聚合，并保留原来缺失组。
- README 主表列出全部条件的 HOTA/AssA/DetA/IDF1/IDSW/FP/FN、相对最强基线差值；同时报告逐视频胜/平/负、差值中位数、最差视频和总帧数。平局按完整精度及预写容差判断，不能从舍入表猜。
- 对比必须匹配检测、外观权重、视频/帧、输出政策和评测器；同时修改的因素逐项明列。别的论文原表或不同 detector 的全系统结果只作为独立参考。
- 多 seed 必须有每个 seed 的全部结果，再给均值/标准差；共享确定性基线不重复伪装为多个独立样本。视频是配对分析单位，相关窗口/相邻帧不是独立重复。开发集上反复选择后得到的区间不称独立确认。
- 如计算区间，报告方法、重采样单位、重复次数、seed 和 estimand。逐视频平均差的区间与官方总体差不同；官方总体指标的 bootstrap 须使用评测器所需统计量重新聚合。没有这些信息就先报告描述值，不制造“显著”。

## 4. 诊断必须能解释“哪里变了”

`diagnostics.json` 不固定某一方法的所有字段，但每项统计必须有 `name / definition / value / unit / numerator / denominator / scope / gt_role / source`。不适用的分子分母可为 null 并说明。至少覆盖本实验涉及的下列层级：

1. 全部预期视频/帧/轨迹 → 可评价单位 → 触发 → 可执行动作 → 实际分数变化 → 配对变化 → 完整输出变化。每层写分母；“进入函数”不代表模型影响了结果。
2. 正检测是否存在、首轮/后续轮是否可见、合法门控是否通过、正确候选是否在实际动作集中、外观是否可用；记录快路径/回退/无候选。GT 只用于离线标注，不能混入触发或推理特征。
3. 对所有受影响身份分别统计新增正确、破坏原正确、保持不变、无法标注；损伤其他身份不能仅用目标身份收益遮盖。局部代理不改称官方 IDSW 或 HOTA。
4. 事件数量、唯一视频/帧/身份数、窗口暴露帧数、去重帧数及重叠；抽样和筛选规则、被排除样本数与原因。缺少覆盖时结论是证据不足，而非方法已被否定。
5. 同状态反事实与各方法独立闭环分开。两种运行中同一个数字 track ID 通常不保证同一身份；跨分支比较需要明确的冻结锚点或评测器 GT 映射。

工程自检和研究诊断各自报告。零变化可能来自合法决策余量、候选无差异或代码未生效，需要分数、候选和输出证据区分，不能一律解释成“无价值”。

## 5. 案例、曲线与时间开销

案例默认按**固定种子哈希顺序**选成功、失败、无变化各最多 3 例；类为空就写 0，不补造。追加 1–2 个最严重失败用于排障时，单独标注“按最差结果选取”。这是解释性抽样，不用来估计总体成功率。抽样规则、总候选数和未覆盖原因写入 diagnostics。

每条 JSONL 包含 case ID、run/condition/baseline、序列/帧范围、抽样分层与排序键、比较类型（同状态/独立闭环）、track/det 的稳定映射、GT 的诊断角色、原/新配对、关键分数、合法边 mask、变更原因、该窗口所有受影响身份的结果及 evidence 路径。数字 track ID 不作为跨运行身份对齐依据。图表可在许可允许时附少量带帧号、GT/基线/方法一致配色的拼图；不能分发原始视频或整库图像。没有图像也必须给可解释的坐标/配对/分数证据。

训练曲线每行记录 run、seed、epoch、step、split、metric、value、unit、已见样本数、elapsed_seconds、checkpoint ID。训练/开发 loss、学习率、开发主指标及评测间隔必需；未每轮评测写真实频率，不插值冒充观测。manifest 登记固定 checkpoint 选择规则、实际被选 checkpoint 的步数/SHA256、总参数/可训练参数、batch/优化器/调度/精度、全部重启与发散。推理不得用 GT 选模型或阈值。

成本分开报告 detector、ReID 抽取、缓存跟踪、学习头、评测、诊断多分支。记录预热、CUDA 同步、batch、硬件、采样次数与 p50/p95 的定义。缓存回放 FPS 不能叫端到端 FPS；带额外 oracle/trace 的时间不能当部署速度。尚未测量写未测，不从文献或设备型号推算。

## 6. 服务器提交前自检与本地核查边界

`checks.json` 的每项用 `PASS / FAIL / NOT_RUN / NOT_APPLICABLE`，并给命令/证据路径与范围。按实验需要检查：

- JSON/CSV 可读、键不重复、计划的组/seed/视频无静默遗漏、缺失值原因完整；完整表覆盖与 manifest 一致。
- 导出器从原始 evaluator 对象提取的数值逐项一致，单位转换明确，README 差值与完整精度数据一致；来源 SHA256 和 source locator 可定位。只检查 Markdown 算术不等于源指标校验。
- 条件间输入与政策一致；零改动适配器可恢复基线；时序、det ID、memory 写入、合法一对一匹配等与改动直接相关的工程门通过。无需为无关文档修改重复整套训练测试。
- `artifacts.json` 的每项标明 `git / server / external`、路径、字节数、SHA256、角色；所有随 Git 交付的结果文件入清单，清单不哈希自己。服务器目录通过文件索引哈希，不对目录名“算哈希”。授权凭证、环境秘密和大型缓存不进入报告。
- README 明列“实际执行的验证”“引用既有审计”“本机无法访问”。独立审阅属于额外证据，不用自动生成一句“已审计通过”替代检查，也不要求每个小测试都重新找 agent 审阅。

指标导出和清单生成应由脚本实现，详细版本由服务器 agent 编写。**本规范及模板不是已实现的验证器。** 接收端可复算差值、覆盖、案例和格式，只有访问并复跑服务器原始产物后才能声称独立复现。

## 7. 历史报告如何补齐

优先补 S01 全 24 段 G/Z/L/M 原始完整精度表和 S04 的逐事件机器可读结果/有效覆盖；从原有权威 summary、分支输出、审计索引导出，不从 README 反推精度，不重跑训练。新增 `backfill` 目录/报告修订说明，记录提取时间、原始 run/eval ID 和输入哈希，不改旧结论或旧运行时间。原始文件不可用则明确列为缺口。历史规范欠缺不自动使已有结果失效。
