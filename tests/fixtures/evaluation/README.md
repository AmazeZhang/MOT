# Evaluation fixtures

TrackEval 固定人工例放在这里。当前三个序列均为两帧、完美定位框：

| 序列 | 构造 | 已验收期望 |
|---|---|---|
| `perfect` | 两个目标，两帧 ID 完全一致 | HOTA/DetA/AssA/MOTA/IDF1 = 1，IDSW = 0 |
| `miss` | 一个目标，仅第一帧有预测 | HOTA/DetA/AssA/MOTA = 0.5，IDF1 = 2/3 |
| `swap` | 两个目标第二帧交换预测 ID | DetA = 1，AssA = 1/3，HOTA = sqrt(1/3)，IDSW = 2，IDF1 = 0.5 |

这些数值已在固定 TrackEval commit 的 `MotChallenge2DBox` 适配器上实际运行通过。测试关闭预处理以隔离指标行为；真实 DanceTrack 评测仍需单独验收类别和预处理配置。
