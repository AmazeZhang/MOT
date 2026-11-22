# S01 实验完整性审阅

日期：2026-11-21。审阅者：新上下文 GPT-6.1-Sol ultra，同模型家族，**provisional**；原技能指定的 GPT-5.6-Sol 在当前环境不可用。结论：**WARN**。警示涉及数据范围、历史锁与成本解释；没有发现伪造 GT、自归一化分数、空结果、未调用的指标路径或当前帧外观记忆泄漏。

权威输入：`/media/imc/data/mot-data/runs/s01_visual_association_20261120/s1_full_evaluation2/summary.json` SHA256 `b3585deb2eebc5132ac898478a69d8bdb803f9b989c0bdaf8c88ea0bf6ddabb0`；`frozen_inputs.json` SHA256 `ed50243913084d3df2840423feb2bb3ffb3b010667dbc8cf4ebf8d483340fcc2`；[运行锁](../../../manifests/runtime/s01_visual_association_s1_full_eval2_20261121.json) SHA256 `473a092e59f06033419f2bf2f145e5476e871f9fdbc923c2b721c9d0b7fab460`。

| 检查 | 判定 | 证据与解释 |
|---|---|---|
| A. GT 来源 | PASS | [评测入口](../../../scripts/summarize_s01_visual_baselines.py) 第 121 行先冻结轨迹与 trace 哈希，第 126 行后才读取 GT，第 143–157 行调用 TrackEval。审阅者将 24 个 `gt.txt` 与归档 `train1.zip/train2.zip` 中对应文件逐字节核对；共 190,627 行，mark/class 均为 1/1。真实数据标注，没有从轨迹生成参考。|
| B. 分数归一化 | PASS | [指标投影](../../../scripts/analyze_e1_mature_reemit.py) 第 41 行对官方 TrackEval HOTA 阈值求均值并转百分数；CLEAR 计数保持原值。外观 L2 与余弦是特征/关联计算，不是性能分数的自归一化。官方 detailed CSV 保留。|
| C. 结果存在与数值 | PASS，历史 caveat | pilot、全量首评与重评文件均存在。审阅者核对首评/pilot 1,080 个、重评 900 个数值字段与官方 CSV 一致；重评 192 个轨迹/trace、96 个复制轨迹、24 个特征、24 个工程记录、46 个 TrackEval 源文件及 24 个 GT 哈希均一致。首评锁没有完整固定 TrackEval 本地改动及投影脚本；重评补齐，且 results/deltas/costs/视频顺序与首评精确相同。|
| D. 指标是否调用 | PASS | 评测入口第 153、160–179 行实际构造、运行、投影并写入 HOTA/CLEAR/Identity；报告数值在官方文件和摘要中均存在。|
| E. 证据范围 | WARN | [划分](../../../manifests/protocols/r1_stage_b_20261118.yaml) 的 24 段复用 train-fit 共 25,328 帧，每组一个固定确定性配置；四段 pilot 包含在 24 段内，非独立重复。外观权重训练集重叠未知；检测器来源文档说明其用 DanceTrack train 训练。没有独立泛化、多随机种子或完整 Deep-OC-SORT 复现证据。|
| F. 评测类型 | PASS：real_gt | 自然闭环轨迹与数据集 GT 使用官方 TrackEval 实现评测；MOTIP 仅单独工程验收，未纳入四组指标。|

审阅者还核查全部 186,676 个特征键/框/分数与源检测文件一致；101,312 条条件帧 trace 中记忆写入严格来自更早帧。外观只改变首轮合法 LAP 分数，L/M 的 132/138 个变化帧确实改变配对集合；24 个 G/Z 输出逐字节相同，24 个 G 也与历史 E3 E 逐字节相同。

## 保留的限制与行动

1. `s1_pilot_features1/runtime_lock.json` 曾记录不参与特征提取的回放脚本旧哈希；实际 pilot tracking 锁记录正确回放脚本哈希。该历史差异保留并在下次生成锁时避免记录无关代码。
2. L/M 回放各额外运行无外观关联用于反事实 trace，且包括写文件与状态检查；记录的核时和墙钟只能解释为带仪表的实验成本，不能当部署 FPS 或纯外观开销。
3. 当前只支持“固定检测与固定首轮外观接入，在复用 train-fit 的描述性方向为正、部分视频为负”的结论。独立泛化、跨视频稳健优势与新方法有效性均无足够证据。后续候选 A 应另立预写协议。

审阅 trace：`.aris/traces/experiment-audit/2026-11-21_run01/`；该目录在本地保留。审阅判定是同家族暂定结论，不等同于跨家族独立认可。
