# Step 1 — Decompose the Novelty

Timestamp: 2026-11-10 (Asia/Shanghai)

## Research problem

在固定检测候选池与严格在线多目标跟踪中，当目标出现短时观测缺失、定位偏移和近邻身份竞争时，学习得到的运动分布是否比单点或简单概率预测更能改善实际关联与身份稳定性。

## Novelty claim under review

将缺失间隔条件化的低维运动分布（候选实现为 CFM）通过样本兼容度接入检测分配，并用当前帧真实候选集中的身份竞争监督回传至生成器；通过同编码器 GRU/Gaussian/MDN/CFM、stop-gradient/joint/直接分类器及自然缺失闭环评价，分离“非线性均值”、“简单概率分布”、“只学匹配头”与“真正改变运动分布”的贡献。

## Four atomic axes

- **Problem framing:** 固定检测池、严格在线 MOT；聚焦短时观测缺失后的重新关联和近邻身份竞争；主要终点是 HOTA/AssA/IDF1 与闭环恢复，不是只看 best-of-K 预测误差。
- **Core mechanism:** GRU 历史编码＋缺失间隔条件化的框增量分布；CFM 生成 K 个样本，核兼容度转化为匹配代价；关联监督可沿真实采样链回传至生成器。
- **Key insight:** 运动分布的价值应由无真值选择时的身份决策来验证；关联头能读出分布与关联目标真正改变生成分布必须通过梯度路径消融分离。
- **Application domain:** DanceTrack 为主的在线 2D MOT，固定检测输入，初期不使用 ReID、离线插值或补框；可向 SportsMOT/MOT17 扩展。
