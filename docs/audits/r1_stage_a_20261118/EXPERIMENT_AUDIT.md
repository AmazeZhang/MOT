# R1 阶段 A 实验完整性审阅

- 日期：2026-11-18
- 审阅：新 Codex 代理，同系列临时审阅；指定的 GPT-5.6-Sol ultra 型号在本会话不可用
- 总判定：**WARN**
- 审阅对象：`/media/imc/data/mot-data/runs/r1_stage_a_20261118_retry1/`
- 追踪：`.aris/traces/experiment-audit/2026-11-18_run01/`（本机忽略目录）

| 检查 | 结论 | 证据与限制 |
| --- | --- | --- |
| A. GT 来源 | PASS | `scripts/run_r1_stage_a.py:181,238` 从 DanceTrack train 的 `gt/gt.txt` 加载，候选选择后才读取 GT；`src/mot_research/analysis/failure_events.py:113` 解析真实标注。四段 GT 哈希与运行记录吻合。 |
| B. 分数归一化 | PASS | `scripts/run_r1_stage_a.py:117` 统计原始身份正确帧数之差，没有用模型自身最大值/均值归一化。 |
| C. 文件与数字 | PASS | 四个 JSONL 分别 50、50、50、14 行。独立逐行合计与 `summary.json` 一致：164 个已执行；净正 34、净负 48、零 82；受益锚点 43、受损锚点 58。哈希排序、标签算术、GT 可见帧数和 12 个输入哈希均核对通过。没有保存分支逐帧框，因此不能独立重算每个标签。 |
| D. 执行路径 | PASS | `scripts/run_r1_stage_a.py:44,224,251-289` 实际生成提案、选分量、双分支回放、计算并写出标签；结果文件存在。 |
| E. 范围 | WARN | 只有四段 fit 视频和一次确定性工程 smoke。164 个动作中有 64 个分量没有任何 `t−1` 可锚定的分量内目标，且这些样本净标签全为零；只有 100 个至少含一个分量内锚点。`executed=164` 不代表 164 个都有直接动作监督。没有训练或评估选择器。 |
| F. 评价类型 | PASS（限定） | 真正的数据集 GT + 自定义 10 帧身份代理，见 `scripts/run_r1_stage_a.py:87` 和 `manifests/protocols/r1_stage_a_20261118.yaml:23`。不是 TrackEval/HOTA，也不是现实世界因果效应。 |

分支检查包括首轮整分量一对一约束、独立 tracker 深拷贝、OC-SORT 全局 ID 计数器复位、A0 逐帧精确复现和父状态不变；见 `src/mot_research/analysis/r1_components.py:37,69` 与 `scripts/run_r1_stage_a.py:243-286`。协议哈希与运行记录一致；本轮协议文件未提交到 Git，其创建时间不能由独立 Git 时间戳认证。

**可支持的表述：** 工程管道在这四段视频上产生了内部一致的短程正/负标签。**尚不支持：** 标签可学习、选择器改善闭环、官方指标提升、新颖性已成立。下一步须先把分量内锚点覆盖率写入阶段 B 样本方案，并在留出结果出现前冻结 B 阶段门槛。
