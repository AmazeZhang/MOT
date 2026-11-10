# 当前实验安排与公开文献重合审查

日期：2026-11-10（Asia/Shanghai）。状态：方案审查，尚无本项目训练结果。

## Verdict

**Level 2 — High Overlap.**

没有检出四个轴完全相同的公开工作，但大框架已被明显挤压：DiffuTrack 已做“框运动分布＋短遮挡/缺失＋在线关联”，DiffMOT 已做“生成式运动预测＋MOT”，DMMTracker 已把运动预测定位为服务身份区分。因此当前方案只剩下一条窄但可辩护的差异：真实候选检测集的关联损失是否经实际采样链改变运动生成器，以及 stop-gradient / joint / 简单概率模型 / 无生成器分类器的匹配对照能否证明这一点。

## Delta

Unlike **DiffuTrack**, which trains a diffusion box-motion model with a motion objective and attaches identity supervision to a separate time-aware appearance branch, the proposed work backpropagates current candidate-set association loss through the actual gap-conditioned CFM sampling path and compares it against stop-gradient, matched-head MDN/Gaussian/regression, and a generator-free classifier, yielding causal evidence about whether decision supervision must reshape the motion distribution itself.

## Decomposed claim

- **Problem framing:** 固定检测池、严格在线 MOT；聚焦短时观测缺失后的重新关联和近邻身份竞争；以 HOTA/AssA/IDF1 和闭环恢复为主终点。
- **Core mechanism:** GRU 历史编码＋缺失间隔条件化的框增量 CFM；K 个样本通过核兼容度转为匹配代价；关联监督可沿真实采样链回传至生成器。
- **Key insight:** 必须区分“学会读出已有分布”和“关联目标真正改变生成分布”；不能用 best-of-K 有真值选择代替真实关联。
- **Application domain:** DanceTrack 为主的在线 2-D MOT，固定检测输入，首轮不使用 ReID、离线插值或补框。

## Structured papers

- **TrajE**
  - Date: 2021-06
  - Problem framing: 以多轨迹假设改善非线性/模糊运动下的 MOT。
  - Core mechanism: 循环 MDN＋beam search。
  - Key insight: 多个可能运动比单点更适合跟踪。
  - Application domain: 在线 2-D MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/2106.10950

- **DiffMOT**
  - Date: 2024-06
  - Problem framing: 非线性运动下的在线 MOT。
  - Core mechanism: 历史条件扩散运动预测＋标准关联，支持少步/单步。
  - Key insight: 生成式运动模型可替代线性传播。
  - Application domain: DanceTrack/SportsMOT。
  - Overlap score: 3/4.
  - Source: https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html

- **DMMTracker**
  - Date: 2024
  - Problem framing: 运动预测服务于身份可分性。
  - Core mechanism: 任务特定特征＋4-D 状态采样。
  - Key insight: 位置预测不应只优化几何误差。
  - Application domain: MOT。
  - Overlap score: 3/4.
  - Source: https://orca.cardiff.ac.uk/id/eprint/170666/

- **MOTIP**
  - Date: 2024-03 preprint / CVPR 2025
  - Problem framing: 将关联改写为 ID 预测。
  - Core mechanism: 上下文 ID token 和直接 ID 分类。
  - Key insight: 关联可以数据驱动学习。
  - Application domain: MOT/DanceTrack。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/2403.16848

- **MoFlow**
  - Date: 2025-03
  - Problem framing: 高效多模态人体轨迹预测。
  - Core mechanism: FM 教师、K 组相关预测、单步蒸馏/IMLE。
  - Key insight: 联合多未来可以高效产生。
  - Application domain: 已知身份轨迹预测，不是 MOT 分配。
  - Overlap score: 1/4.
  - Source: https://arxiv.org/abs/2503.09950

- **TrajFlow**
  - Date: 2025-06
  - Problem framing: 高效多模态未来轨迹预测。
  - Core mechanism: FM 多候选、Plackett–Luce 排序、自条件。
  - Key insight: 多样性和排序应联合学习。
  - Application domain: Waymo 运动预测，身份已知。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/2506.08541

- **GenPT**
  - Date: 2025-10
  - Problem framing: 遮挡下的多假设点跟踪。
  - Core mechanism: FM＋跨窗口先验＋迭代精化。
  - Key insight: 生成假设有助于穿过不确定区间。
  - Application domain: 查询点跟踪，ID 由查询给定。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/2510.20951

- **DiffuTrack**
  - Date: 2026
  - Problem framing: 非线性运动、短缺失与身份模糊下的在线 MOT。
  - Core mechanism: 4-D 框扩散分布＋多假设关联；另有时间加权外观原型。
  - Key insight: 分布传播应以关联连续性评价。
  - Application domain: DanceTrack/MOT17/MOT20 在线 MOT。
  - Overlap score: 3/4.
  - Source: https://www.nature.com/articles/s41598-026-50731-8

- **OA-SORT**
  - Date: 2026-03
  - Problem framing: 遮挡导致检测观测偏移和关联错误。
  - Core mechanism: 遮挡感知的代价、测量修正和 KF 更新。
  - Key insight: 错误可能主要来自观测偏移。
  - Application domain: 在线 MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/2603.06034

- **Constraint-Aware Flow Matching**
  - Date: 2026-05
  - Problem framing: 生成样本与下游约束/决策对齐。
  - Core mechanism: 可微投影/约束感知 FM 训练。
  - Key insight: 分布拟合不等于下游效用。
  - Application domain: 非 MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/html/2605.12754v1

- **DOA-SORT**
  - Date: 2026-09
  - Problem framing: 方向性遮挡与框变形下的 ID 关联。
  - Core mechanism: 一个干净＋四个方向的混合观测分布，用于多阶段关联。
  - Key insight: 可能应建模观测分布，而不是运动分布。
  - Application domain: DanceTrack 在线 MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/2609.22706

- **MOTIP2**
  - Date: 2026-10
  - Problem framing: 重叠和空间混淆目标的 ID 预测。
  - Core mechanism: ID 扰动、空间 ID loss、空间锚点。
  - Key insight: 困难空间负样本对关联很重要。
  - Application domain: 端到端 MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/html/2610.10391v1

- **Deep Network Flow for MOT**
  - Date: 2017-06
  - Problem framing: 从最终分配监督中学关联代价。
  - Core mechanism: 可微平滑网络流优化。
  - Key insight: 分配损失可回传到上游代价/特征。
  - Application domain: MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/1706.08482

- **FAMNet**
  - Date: 2019-04
  - Problem framing: 联合学习特征、亲和度和分配。
  - Core mechanism: 可微特征/亲和/多维分配网络。
  - Key insight: 分配标注可直接监督整个关联流程。
  - Application domain: 在线 MOT。
  - Overlap score: 2/4.
  - Source: https://arxiv.org/abs/1904.04989

## Comparison result

- **Proposed work**
  - Title: Association-directed motion distribution learning for short observation gaps
  - Date: 2026-11 planning stage
  - Source: local experiment handbook
  - Problem framing: 固定检测、严格在线、短缺失和身份竞争。
  - Core mechanism: gap-conditioned CFM＋样本兼容度＋关联损失通过真实采样链回传。
  - Key insight: 用 stop-gradient/joint/直接分类器分离读出与生成器重塑。
  - Application domain: DanceTrack-first 在线 MOT。

- **DiffuTrack**
  - Title: DiffuTrack
  - Date: 2026
  - Source: https://www.nature.com/articles/s41598-026-50731-8
  - Problem framing: 同类短缺失/非线性运动 MOT。
  - Core mechanism: 扩散框分布＋关联；ID 监督在独立外观分支。
  - Key insight: 分布传播用于保持可匹配性。
  - Application domain: 同类在线 2-D MOT。
  - Result: **3 axes → Level 2 — High Overlap (one axis differs).**

- **DiffMOT**
  - Title: DiffMOT
  - Date: 2024-06
  - Source: https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html
  - Problem framing: 同类非线性运动 MOT。
  - Core mechanism: 历史条件生成式运动预测＋标准关联。
  - Key insight: 不包含关联梯度重塑生成器的因果拆分。
  - Application domain: 同类 DanceTrack/SportsMOT。
  - Result: **3 axes → Level 2 — High Overlap (one axis differs).**

- **DMMTracker**
  - Title: Learning Discriminative Motion Models for MOT
  - Date: 2024
  - Source: https://orca.cardiff.ac.uk/id/eprint/170666/
  - Problem framing: 运动预测服务身份可分性。
  - Core mechanism: 不同；含图像/SOT 特征与状态采样。
  - Key insight: 身份导向运动学习。
  - Application domain: MOT。
  - Result: **3 axes → Level 2 — High Overlap (one axis differs; full-text limitation).**

- **TrajE**
  - Title: TrajE
  - Date: 2021-06
  - Source: https://arxiv.org/abs/2106.10950
  - Problem framing: MOT 多运动假设。
  - Core mechanism: MDN＋beam search。
  - Key insight: 有多假设价值，但无关联梯度隔离。
  - Application domain: 在线 MOT。
  - Result: **2 axes → Level 3 — Medium Overlap (two axes differ).**

- **Constraint-Aware Flow Matching**
  - Title: Constraint-Aware Flow Matching
  - Date: 2026-05
  - Source: https://arxiv.org/html/2605.12754v1
  - Problem framing: 不同的约束生成任务。
  - Core mechanism: 下游感知 FM。
  - Key insight: 决策约束应改变生成器。
  - Application domain: 非 MOT。
  - Result: **2 axes → Level 3 — Medium Overlap (two axes differ).**

- **TrajFlow**
  - Title: TrajFlow
  - Date: 2025-06
  - Source: https://arxiv.org/abs/2506.08541
  - Problem framing: 已知 agent 轨迹预测。
  - Core mechanism: FM 多候选＋排序。
  - Key insight: 效用/排序监督辅助生成。
  - Application domain: 自动驾驶预测。
  - Result: **1 axis → Level 4 — Low Overlap (three axes differ).**

- **DOA-SORT**
  - Title: DOA-SORT
  - Date: 2026-09
  - Source: https://arxiv.org/abs/2609.22706
  - Problem framing: 同类遮挡/ID 关联。
  - Core mechanism: 方向观测混合，不是运动生成。
  - Key insight: 观测偏移是竞争性解释。
  - Application domain: 同类 DanceTrack MOT。
  - Result: **2 axes → Level 3 — Medium Overlap (two axes differ).**

## 当前实验项的重合分类

| 实验块 | 最近先例 | 结论 |
|---|---|---|
| G1 baseline replay／TrackEval 人工例／重复性 | TrackEval 与常规复现规范 | 重要但不是创新 |
| G2 D1/D2/D3 特权信息归因 | OA-SORT、DOA-SORT、DiffuTrack | 失败因素重合；三分诊断是定制化分析，可作分析贡献 |
| G3 regression/Gaussian/MDN/CFM 公平对照 | TrajE、DiffMOT、DiffuTrack、TrajFlow/MoFlow | 强重合，是必要性测试而非新方法 |
| 自然缺失＋人工删检测＋缺失长度分层 | DiffuTrack、GenPT、遮挡鲁棒跟踪文献 | 常规鲁棒性评价 |
| K/积分步数/延迟曲线 | DiffMOT、DiffuTrack、MoFlow、TrajFlow | 标准生成推理消融 |
| G4 predictor × motion/joint | DMMTracker、Constraint-Aware FM、可微关联 | 动机重合；完整固定协议因子矩阵未发现完全先例 |
| stop-gradient vs 真实采样链回传 | 最近为 MOT 外的 Constraint-Aware FM | 当前最强窄差异；需要闭环关联收益才成立 |
| 无生成器几何分类器 | MOTIP/MOTIP2、FAMNet、Deep Network Flow | 直接关联学习已有；作为必要性对照很强，但不是创新 |
| 多种子／多数据／视频 bootstrap | 常规统计与评价实践 | 必要验证，不是创新 |

## 检索边界

三条检索式均指向“在线 MOT 概率运动/关联监督”和“FM＋MOT＋association loss”。2026-11-10 的六源聚合调用受安装版本与远程 API 阻塞影响；DBLP 三次解析失败，聚合调用最终手动中止。本结论主要依据 2026-11-03 已保存的六源结果、7 篇全文抽取与 2026-11-10 定向原始论文页更新。“未检出完全相同”不等于“世界上不存在”；方法最终定稿后必须按具体 loss、梯度路径和候选集定义再做一次窄检索。
