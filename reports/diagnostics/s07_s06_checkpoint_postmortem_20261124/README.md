# S07：S06 既有 checkpoint 学习质量体检

状态：`COMPLETED_DIAGNOSTIC / STOP_NO_NEW_RUN_AUTHORIZED`。日期：2026-11-24。权威 run ID：`s07_s06_checkpoint_postmortem_cpu_attempt1`。本轮只在固定样本上读取既有 checkpoint 做 CPU 推理；没有训练、调参、checkpoint 重选、闭环跟踪或 TrackEval。

## 问题与决定

问题是：S06 的 Local/Context checkpoint 是否在固定 train18/internal-dev6 代理标签上学到了超越常数预测器的动作效用与可迁移动作排序？结论分成两层：**标量 g/h 回归误差确有下降；可迁移动作排序没有成立。** result-to-claim 为 `partial`（同家族暂定、confidence high），但研究路由仍是保持 S06 STOP，不运行 seeds37/73，不自动启动新性能实验。

## 固定样本结果

所有 12 个 checkpoint（Local/Context × epoch 5/10/15/20/25/30）均完成。train18 为 104 分量/208 动作；internal-dev6 为 22 分量/44 动作，实际覆盖 5 个有标签视频，dancetrack0066 没有可标注分量。MSE 使用视频等权、视频内动作等权。

| 条件 | train18 mean MSE | internal-dev6 mean MSE |
|---|---:|---:|
| 常数零预测 | 0.13560599028435288 | 0.12657407407407406 |
| train18 视频等权全局均值 | 0.10106821534683674 | 0.09706207694768555 |
| Local epoch5 | 0.16925734365891143 | 0.16122084165155168 |
| Local epoch30 | 0.08302594144457168 | 0.08368197978582628 |
| Context epoch5 | 0.15505424682389463 | 0.14674816851532418 |
| Context epoch30 | **0.054256786158627404** | **0.06909482306121303** |

Context epoch30 在 internal-dev6 上比 Local epoch30 低 0.014587156724613248 MSE，主要来自 h 误差下降。这只支持固定样本的标量代理回归改善；internal-dev6 已用于原 S06 checkpoint 选择，且只有 seed13。

## 动作决定没有迁移

原导出把 decision accuracy 按分量 pooled，虽协议写了视频等权；这是结果后发现的执行偏差。补充文件并列保留两种分母，不覆盖原数值。

| internal-dev6 条件 | pooled | 视频等权 | 视频等权 always-M |
|---|---:|---:|---:|
| Local epoch30，无额外风险 | 14/22 = 0.636364 | 0.686667 | 0.686667 |
| Context epoch30，无额外风险 | 13/22 = 0.590909 | 0.646667 | 0.686667 |
| Context epoch30，risk | 14/22 = 0.636364 | 0.686667 | 0.686667 |

Context epoch30 无额外风险在 train18 的 pooled 决策一致率为 80/104（always-M 为 71/104），但 internal-dev6 只有 13/22（always-M 14/22）；其中只找回 2/8 个真实非 M 决策，漏 6 个并产生 3 个错误非 M 决策。risk 在 internal-dev6 完全等同 always-M。故较低 MSE 不能解释成学会了可迁移动作排序或可靠拒绝。

结果后透明增加了两个**探索性而非冻结对照**：把 M 固定为解析零、其他动作使用 train18 视频等权均值，internal-dev6 MSE 为 0.06755007982129707；按 M/G/V 分别使用 train18 均值为 0.05587726359924876，均低于 Context epoch30。这说明原 action-agnostic 常数对照太弱，当前结果不足证明 component-specific utility learning。

## 案例补充与失败谱系

attempt1 把 S06 trace 的 `base_first_pairs`（代码实际为 G 候选）误标成 baseline M；独立审计发现后保留为数据盘上的 `*_attempt1_invalid*`，不进入权威 Git 案例。attempt2 是权威版本，共覆盖指定四段中的全部 15 个 Context-no-extra-risk 改动决定。

attempt2 只能比较**同一方法状态的 G 候选与实际应用配对**，并把独立闭环 M 轨迹用于描述性 30 帧后续对照。保留 trace 没有同状态 M 候选配对和精确合法边 mask；4 个选 G 案例相对 G 自然没有差异集合，14 条配对无法以 IoU≥0.5 确认 GT owner。后续 Hungarian IoU 对齐不是官方 IDSW，也不能把整段 HOTA 差归因给单个动作。因此案例补充提高了可追踪性，但没有补成机制因果证据。

## 执行、验证与限制

- 命令使用 `CUDA_VISIBLE_DEVICES=''`、8 CPU 线程；物理 GPU 列表为空，GPU0 未使用。主推理耗时 1.5301 秒，CPU 核时上界约 0.0034。
- 临时目录完整重复推理后，3,024 行预测和 checkpoint CSV 与权威产物逐字节一致。
- 独立审阅复算 24 个 checkpoint/split 指标、常数与同 epoch 差值共 1,104 项，最大差约 1.67e-16；另用不同 Torch 版本重推理，最大绝对预测差约 5.96e-08。
- 没有新的官方 tracking 指标；`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`learning_curves.csv` 不适用。
- 代理标签仍是离线 stable-owner 定义；范围是一 seed、小样本、复用 development，且原 S06 的 action batching 与有效容量偏离仍存在。

## 下一步

保持 `STOP_S06_NO_SEEDS_37_73`。当前不授权重训、阈值扫描、FM 或新性能运行。若以后另立方法问题，协议至少要包含：M 解析零点、action-type 强常数对照、相对排序/utility regret/拒绝校准指标、公平有效容量与 component batching、更充分监督，以及未用于开发的最终评估。
