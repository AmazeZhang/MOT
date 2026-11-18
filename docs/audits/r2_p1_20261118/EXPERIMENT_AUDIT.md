# R2 P1 实验完整性审阅

**日期：** 2026-11-18
**审阅方式：** 新鲜 Codex 子代理，只读，同系列，provisional。技能指定的 GPT-5.6-Sol ultra 路由在当前会话不可用，实际使用继承模型。
**总体结论：WARN。** 原始离线排序结果和 P1 停止门槛可复算；次要 margin 字段不符合协议，且未提交协议的冻结时间不能由哈希独立证明。

| 项目 | 结论 | 独立审阅证据 |
|---|---|---|
| A. 标签来源 | PASS | 事件构建器从 DanceTrack `gt/gt.txt` 与固定检测、B0 轨迹产生正检测和目标 ID；本次采集与评分不把 GT 作为特征。见 `scripts/build_p1_closed_loop_event_set.py:97`、`src/mot_research/evaluation/closed_loop_events.py:101`、`scripts/capture_r2_decision_states.py:139`。 |
| B. 分数归一化 | PASS | `scripts/rescore_r2_decision_rules.py:149-199` 使用 IoU、固定核宽和固定边际惩罚；没有除以模型自己的最大/最小输出。 |
| C. 文件与数字 | PASS | 831 个事件、749 个事件帧、20,775 个唯一结果行、3,745 个完整矩阵行；逐矩阵复算获胜轨迹与主指标 0 处不一致。8 段 B0 重放均 PASS。输入及输出哈希与运行摘要相符。 |
| D. 代码实际调用 | PASS | `scripts/rescore_r2_decision_rules.py:329-355` 调用打分、事件判定和门槛汇总；输出文件存在。 |
| E. 评价范围 | PASS，有限定 | 8 段选定的 DanceTrack train motion-dev；官方 val 未使用。摘要明确为离线排序代理指标，没有声称闭环或官方基准收益。 |
| F. 评价类型/次要字段 | WARN | 固定事件表有真实 GT 锚点，但指标是自定义的正检测轨迹排序代理。协议要求目标轨迹的 margin，原 CSV 保存的是获胜者的 margin；错误排序时两者不同。见协议第 67 行、评分脚本第 219–231 行。 |

审阅从原始 CSV 重算：829 个可评价事件中，S1 成功 545、失败 284；救回至少需要 29，新增伤害至多 10。S2 为救回 57 / 伤害 38，S3 为 91 / 63，S4 为 64 / 39。三个规则都有 7 段视频出现救回，但都超过伤害上限，因此**均未晋级 P2**。

审阅另指出 B0 和 CV 的 S1–S4 分数相同：CV 点框只参与 S0，不能把主门槛结果归因给 CV 提案。协议 SHA256 与运行记录一致，但协议尚未提交 Git，审阅无法独立认证“先于结果冻结”的时间顺序。

**处理：** 保留原始评分产物不变；用 `scripts/derive_r2_target_margin.py` 从已存完整矩阵补算了协议所需的目标 margin，并写在独立附录目录。附录通过源哈希、目标分数、候选资格和全 20,775 行键的核对。此修复不改变主指标或 P1 门槛。研究报告把原 `winner_margin` 与补算的 `target_minus_strongest_other` 分开解释。下一次运行应在 Git 可追踪状态下冻结协议，并记录峰值 RSS 与 CPU 核时。

完整审阅输入/输出与原文见忽略的 `.aris/traces/experiment-audit/2026-11-18_run01/`。
