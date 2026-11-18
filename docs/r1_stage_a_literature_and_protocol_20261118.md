# R1 阶段 A：先例核查与已冻结的最小试验

日期：2026-11-18。此文记录阶段 A 的方法边界；数值结果另见运行目录和进展日志。

## 先例核查

| 原始来源 | 已能确认的相关机制 | 对 R1 的约束 |
| --- | --- | --- |
| [Learning to Track（作者项目）](https://cvgl.stanford.edu/projects/MDP_tracking/)、[作者代码](https://github.com/yuxng/MDP_Tracking) | 将在线多目标跟踪作为决策过程学习 | “学习跟踪/关联决策”不能作为新颖性主张；本次只确认项目与代码入口，没有运行其代码 |
| [Event-aware Learning，UAI 2022](https://proceedings.mlr.press/v180/lee22a.html)、[论文 PDF](https://proceedings.mlr.press/v180/lee22a/lee22a.pdf) | 训练时模拟 tracker，取真实候选与约 10–20 帧片段构造事件监督 | “用未来事件训练关联”已有直接近邻；R1 须进一步证明同状态成对动作和显式收益/损伤标签的必要性 |
| [DeconfuseTrack，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf) | 分解混淆关联并对关联候选作选择性处理 | “把冲突匹配分组并选择性干预”也不能单独主张创新 |
| [Sentinel，Scientific Reports 2026](https://www.nature.com/articles/s41598-026-43938-2) | 可访问的索引/摘要显示它涉及轨迹状态与关联策略 | 正文访问受限，尚不能比较具体实现；不能据此声称 R1 不重合 |

这不是系统性文献综述，也不是新颖性认证。尚未逐行比较 Event-aware Learning、DeconfuseTrack 与 Sentinel 的公开实现；Sentinel 正文访问受限。阶段 B 前需要继续做全文与源码级核查。

### 阶段 B 前的定向补查（2026-11-18）

- [Event-aware Learning 正文 §3.3–3.5](https://proceedings.mlr.press/v180/lee22a/lee22a.pdf) 使用事件类别与 GT 关联矩阵监督网络；训练片段随机裁成 10–20 帧，模拟 tracker 从片段首帧的检测初始化。文中还在训练时将部分缺失目标的 GT 框加入候选，并将当前检测框替换为相应 GT 框作回归训练。我们目前的 R1-B 使用固定真实检测、同一 B0 状态的两条可执行首轮动作和后续 B0 回放标签；这是机制差异，不保证足以构成论文创新。
- [DeconfuseTrack 正文 §3](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf) 的 DDA 在初始全局匹配后顺序处理检测、轨迹和关联歧义，属于非学习的选择性关联改动；R1-B 的简单 IoU 规则对照必须保留，不能仅与 B0 比较。
- 检索到的 [SEAT_tracktor1 仓库](https://github.com/cocoen12/SEAT_tracktor1) README 自称 Event-aware Learning 实现，但网页可见仓库文件只有 README、LICENSE 和演示 GIF；它还保留 DeepMOT/Tracktor 说明。没有据此声称完成 SEAT 源码比对。DeconfuseTrack 的 CVF 页面未给出可核对的官方实现入口。本轮源码级重合判断仍不完整。

## 冻结协议

机器可读协议为 [r1_stage_a_20261118.yaml](../manifests/protocols/r1_stage_a_20261118.yaml)。它在读取本轮 GT 标签前落盘，固定以下内容：

- 对原 motion-train 32 段按序列名的 SHA256 排名，前 24 段 fit、后 8 段 calibration；原 motion-dev 8 段和官方 val 均不参与。
- 从 fit 排名前四的 `dancetrack0087/0098/0072/0001` 做工程 smoke。每段对 B0 与恒速首轮匹配的完整分歧分量按固定哈希取最多 50 个，合计上限 200。
- A0 是原 OC-SORT 首轮匹配。A1 只把所选连通分量的首轮匹配切换为同状态 CV 提案；其余匹配维持 A0。两条分支后续都运行原始 B0，窗口为包含干预帧在内的 10 帧。
- 在 `t−1` 用 GT 与基线 tracker 输出按 IoU ≥ 0.5 一对一匹配确定身份锚点。标签统计 `t..t+9` 中 GT 可见帧上原 tracker ID 的正确输出数；所有锚点参与，包含分量外。GT 不进入 tracker 策略、候选排序或未来检测。
- 序列尾不足 10 帧的分量排除并计数。预定工程门包括基线分支逐帧完全一致、父状态不变、匹配一对一、至少 100 个已执行样本，以及正/负净标签各至少 3 个。数值门只用于决定是否值得审阅下一阶段，不证明可学习或方法有效。

本阶段没有选择器训练或官方 TrackEval 闭环方法评价。输出是确定性软件分支中的短程身份代理标签，不是自然场景中的因果效应，也不是 HOTA/AssA。阶段 B 的训练与开发门仍须在查看留出结果之前另行冻结。
