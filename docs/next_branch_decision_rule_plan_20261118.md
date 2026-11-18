# 下一分支候选：分布假设下的决策规则对照（执行草案）

**记录时间：** 2026-11-18 19:23（Asia/Shanghai）
**状态：** 草案。未冻结、未实现、未运行、未分配 GPU。
**基线提交：** `62bad22`（记录时工作树仅 `docs/progress_log.md` 有未提交追加）
**触发：** 2026-11-18 DiffuTrack 全文核查 + R1 阶段 B 停止后的路线选择
**性质：** 这是给人执行的方案，不是冻结协议。真正运行前须按 §8 另写 `manifests/protocols/*.yaml`，并在读取本轮标签前落盘。

**给服务器的第一句话：** 先做 §6 的 P0，只做核查、不跑 tracker；P0 的结论决定 P1 是"直接重排"还是"先补一次 831 事件重放"。

---

## 0. 一句话

在**同一组运动假设**下比较不同的打分/分配规则，分离"分布打分"与"轨迹续存"两条通道；不训练新的生成器。

## 1. 为什么选这个方向

### 1.1 DiffuTrack 已经占掉了"分布用于关联"

外部论文 `s41598-026-50731-8.pdf`（Scientific Reports 2026, 16:19960；DOI [10.1038/s41598-026-50731-8](https://doi.org/10.1038/s41598-026-50731-8)；本机下载副本与哈希见 §11）即台账已登记的 DiffuTrack：

- MDM：条件扩散在 4D 归一化框状态上生成多假设，历史 L=8，DDIM 10 步，残差 MLP 去噪。
- 关联代价：`C_mot = 1 − (1/S)·Σ_s IoU(b_s, d)`（Eq. 10），即**样本均值 IoU**，与外观代价 0.55/0.45 融合后匈牙利分配。
- 输出：匹配上时输出检测框；未匹配时用扩散续轨框，patience 8 帧。
- DanceTrack test（共享 YOLOX-X 检测）：DiffuTrack 63.5 HOTA / 80.8 DetA / 50.2 AssA / 64.1 IDF1；OC-SORT 55.1 / 80.4 / 38.0 / 54.2；DiffMOT 62.3 / 82.5 / 47.2 / 63.0。

因此**不能**把"概率运动改善关联"或"解析混合算期望 IoU"作为方法创新；后者在机制上就是 Eq. 10 的蒙特卡洛版本。

### 1.2 但它没有隔离打分通道

- 表 10：纯扩散输出 vs 融合输出，**AssA 都是 50.1**，差异全在 DetA（75.4 vs 80.8）。
- 表 11：fallback 续轨触发于 31.6% 轨迹、平均 2.7 帧、**73.9% 再关联成功**、持续 fallback 后失败 17.8%。
- 正文 p.6 自述："This 'continue under uncertainty' mechanism is the primary pathway through which diffusion improves association."

亦即：该文的收益主要来自"轨迹在缺失期间不死、之后还能匹配"，而不是"分布让打分更准"。这条通道没有被反事实实验拆开。

### 1.3 我们自己的证据指向同一处

- P1 闭环：CV(51.057 HOTA) > B0(50.060) > learned(49.271)。恒速单步 IoU 最差（0.522 vs learned 0.617、KF 0.625），闭环却最好。
- 按 `HOTA=√(DetA·AssA)` 分解：B0→CV 的 +0.997 中 DetA 约 25%、AssA 约 75%；CV→learned 的 −1.786 中 AssA 约 92%。**要解释反转必须解释身份。**
- 机制诊断在共同 B0 状态下做单步旁路，按构造看不到方法自身的状态演化，因此四项聚合检查不触发不构成"没有机制"的证据。
- R1：未来标签选择器 +168/83，简单 IoU 规则 +239/83，1/8 视频严格胜出——动作级价值学不出来，但**边际可观测**。

### 1.4 引用该文时要注意的三点

- 组件消融基线是 KF+CE（DanceTrack val 48.5 HOTA / 35.1 AssA），不是 OC-SORT（test 55.1 / 38.0）；"MDM +9.7 HOTA" 是相对弱基线测得，且 TPCL 依赖外观。
- 表 7 的覆盖率对比用扩散经验分位数 vs KF 模型协方差，KF CR@95% 仅 42.1%，不是同口径校准比较；不能据此说分布"更校准"。
- 主表单次运行、无置信区间；未提供代码链接。引用数字时限定为"该文报告值"，不作为我们的基线或复现目标。

## 2. 假设

| 编号 | 假设 | 判据（见 §7） |
|---|---|---|
| H1 | 打分规则本身影响 AssA：同假设下 S2/S3/S4 优于 S1 | P2 闭环 AssA 与 IDSW |
| H2 | 均值 IoU 不是似然，多模态下会反转正确候选排序 | P1 离线 rank/margin |
| H3 | 收益主要来自续轨而不是打分 | P3 三条件归因 |

H1 与 H3 可同时成立（打分有增益但续轨更大），设计要能分别估计，不预设二选一。

## 3. 边界（不做什么）

- 不训练扩散/流匹配/MDN 生成器；不引入外观、ReID 或新检测器。
- 不改检测缓存与 OC-SORT 的 KF/生命周期；S4 只改分配层的代价，不改预测与更新。
- 不重开 R1 的选择器路线，不放宽其已注册门槛。
- 8 段 motion-dev 只作开发证据，不称独立确认；官方 val 确证须另立协议、锁定后一次性使用。
- 不声称复现或超过 DiffuTrack：检测器、外观分支与训练预算都不同。

## 4. 假设集与打分规则

### 4.1 假设集（不使用 GT、不使用未来帧）

对每条轨迹 `i` 在帧 `t`：

1. **KF 采样**：取 OC-SORT 在 `t−1` 的 KF 均值与协方差，predict 到 `t` 后采样 `S=20` 个框。
2. **转向模式**：用最近两帧位移估计航向 `θ` 与角速度 `ω`（按轨迹长度截断），生成左/右 `±ω` 两个解析模式。
3. **可选模式**：P1 checkpoint 的点预测作为一个模式；只有 checkpoint 仍在时使用，且必须作为独立消融变体记录。

权重先用均匀；若要拟合权重，只允许在 24 段 fit 上拟合并固定。

### 4.2 五种打分规则

记假设样本为 `b_s`（中心+尺度），检测为 `d`：

```
S0  点预测        IoU(mean_box, d)                      # 传统 tracker
S1  样本均值 IoU  (1/S) Σ_s IoU(b_s, d)                 # DiffuTrack Eq.10
S2  样本最大 IoU  max_s IoU(b_s, d)                     # 模式感知
S3  混合似然      log Σ_k w_k N(d; μ_k, Σ_k)            # 概率正确用法
S4  边际感知分配  argmax Σ s'_ij,
                  s'_ij = s_ij − β·max(0, τ − m_ij),
                  m_ij  = min(该检测上的行边际, 该轨迹上的列边际)
```

- `m_ij` 只用分数矩阵的前二值，`O(N log N)` 可算。
- `S4` 可叠加在 `S2` 或 `S3` 上；`β=0` 退化为对应基线。
- 规则必须只作用于 OC-SORT 走全局 Hungarian 的帧；稀疏帧的 one-to-one IoU 捷径保持原样（G2 记录 165/165 事件都走全局路径，作用面正确）。

**为什么 S1 可能反转（H2 的技术依据）：** `E[IoU]` 随假设集变宽而下降，对正确检测和错误检测同时下降。若正确检测只被 1/3 的模式覆盖、某个错误检测被 2/3 的模式覆盖，均值 IoU 会偏好错误检测——这恰好发生在该方法最想解决的多模态场景。`S2`（只看最可能模式）和 `S3`（用密度而不是重叠）是对这个问题的两种修正。

## 5. 指标与事件集

- 官方 TrackEval `COMBINED_SEQ`：HOTA / DetA / AssA / IDF1 / IDSW / Frag。
- 固定 831 事件集（699 个 B0 persistence failures + 81 个 B0 success controls）上的 recovery / harm。
- 逐序列表 + 官方聚合；8 段 motion-dev，7,565 帧。
- 迁移性对照：至少同时报告 S0（点）与 S1（DiffuTrack 规则），否则无法判断增益来自规则还是来自假设集。

## 6. 阶段与工作量

### P0 服务器事实核查（0.5 天，不跑 tracker）

1. 查 `/media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113/` 的逐事件 CSV：38 列/行里有没有**逐候选分数或候选排名**。有 → P1 可直接重排；没有 → 需要一次 831 事件的 CPU 重放来重建候选集。
2. 记录检测缓存来源（detector、阈值、输入尺寸）、OC-SORT commit、`mot-det` 版本、GPU 使用（应为 CPU）。
3. 确认 P1 checkpoint 是否仍可访问（决定"可选模式"是否可用）。
4. 产出 `reports/engineering/r2_recon_<date>/README.md`，含输入哈希与列字典。

**P0 决策：** 若候选分数可重建，P1 为 1–2 天；若必须重放，先做重放并冻结重放协议。

### P1 离线重排（1–2 天）

- 对每个事件 × 方法 × 规则，重建候选集，算：正确候选是否 top-1、正确候选的 margin、翻转数、对 S1 成功事件的伤害。
- 方法集先取 B0/KF、CV、learned 三个 seed；不必一开始就上六方法。
- 产出：`analysis.json` + 逐事件 CSV（留在数据盘）。

**门槛（草案，冻结时复核）：** 至少一个规则使 S1 失败事件中 ≥10% 翻为正确，且 S1 成功事件的新增伤害 ≤2%。

### P2 闭环对照（3–5 天）

- 在首轮适配器里实现 S1–S4（不改既有模块，新增 `scripts/run_r2_rules.py` 与 `src/mot_research/tracking/r2_scoring.py`）。
- 8 段 motion-dev，CPU，`CUDA_VISIBLE_DEVICES=''`，逐规则跑 TrackEval。

**门槛（草案）：** 相对 S1，AssA ≥ +0.3 点、DetA ≥ −0.2 点、IDSW 不增、HOTA ≥ S1；831 事件上 recovery−harm ≥ S1 + 1。

### P3 通道归因（2–3 天）

三条件同状态对照：

| 条件 | 打分 | 续轨 |
|---|---|---|
| A | S2/S3 | 关（未匹配即按原生命周期） |
| B | S1 | 开（fallback 续轨） |
| C | S2/S3 | 开 |

用成对分支把 AssA 变化拆到"打分 / 续轨 / 交互"。这一条是对 DiffuTrack 缺失消融的直接回应。

**判据：** 哪条通道解释 ≥50% 的 AssA 增益，就按那条写结论；不预设必须落在打分上。

### P4 可选

- S4 的 β/τ 敏感性；官方 val 锁定后确证（另立协议）；第二 tracker 泛化（可选章节，不承诺）。

## 7. 判据与停止规则（冻结前数值可改，改动须在结果前）

- **P1 不通过**：所有规则相对 S1 的 rank 改善都在噪声内 → 停在离线，写"打分规则不敏感"的归因结论。
- **P2 不通过**：S2/S3/S4 无一提过 S1 门槛 → 不进入 S4 调参，不换更复杂模型续试，转 P3 写通道归因。
- **P3 显示续轨主导**：如实写"收益主要来自续存而非打分"，并把方法主张收缩为归因/协议。
- 任一阶段出现 A0 分支不能逐帧复现、父状态改变、匹配非一对一，立即停止并记录为工程失败，不重试调参。
- 全部阶段不得读取 motion-dev 之外的新划分来挑规则；官方 val 只能在 P4 锁定后使用一次。

## 8. 冻结前必须写进 `manifests/protocols/*.yaml` 的字段

- 事件集定义与分母（831 / 699 / 81）。
- 假设集：采样数、转向模式定义、可选 checkpoint 版本、是否拟合权重。
- 五种规则的精确公式与数值稳定性处理（S3 的协方差正则）。
- P1/P2 门槛数值、预算（CPU 核时、墙钟）、停止规则。
- 输入哈希：split、检测缓存、GT、OC-SORT commit、协议自身 SHA256。
- 输出目录与保留策略（原始 CSV 留数据盘，仓库只放摘要）。

## 9. 运行环境与命令模板

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=src:scripts
/home/imc/anaconda3/envs/mot-det/bin/python -c "import numpy, scipy, filterpy, yaml; print('ok')"
```

拟新增脚本（尚未实现，命名可改）：

```bash
# P1 离线重排（若 P0 发现候选分数不可重建，先做 831 事件重放）
python scripts/rescore_r2_events.py \
  --protocol manifests/protocols/r2_decision_rules_<date>.yaml \
  --split manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml \
  --mechanism-dir /media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113 \
  --train-root <DanceTrack>/train \
  --detection-root <detection-cache> \
  --oc-sort-root <oc-sort-checkout> \
  --run-dir /media/imc/data/mot-data/runs/r2_rescore_<date>/

# P2 闭环
python scripts/run_r2_rules.py ... --run-dir /media/imc/data/mot-data/runs/r2_rules_<date>/
python scripts/analyze_r2_rules.py --protocol ... --label-dir ... --out .../analysis.json
```

**注意：** 本机（Windows 副本）`core.autocrlf=true` 且无 `.gitattributes`，协议哈希只有在 LF 规范化后才与记录一致。服务器上用 LF 检出的工作树计算哈希；若在 Windows 上核对，先做 `\r\n → \n` 规范化再比。

## 10. 记录要求

- 每个 run 目录写 `summary.json`：`git_commit`、`CUDA_VISIBLE_DEVICES`、物理 GPU（应为无）、协议哈希、输入哈希、墙钟、峰值 RSS。
- 运行摘要里写入代码与依赖版本（Python、NumPy、SciPy、FilterPy、lap、OC-SORT commit）。
- 每次变更单独目录，失败现场保留为 `.failed/`，不覆盖。
- 仓库只提交摘要、逐序列表与哈希；原始 JSONL/CSV/输出框留数据盘。

## 11. 定位锚点

- 本文件创建：2026-11-18 19:23（Asia/Shanghai）。
- 基线提交：`62bad22`（`research: record R1 paired-action feasibility and stop gate`）。
- 同日前序记录：`docs/progress_log.md` 的 2026-11-18 R1 阶段 B 条目、分析轮条目、DiffuTrack 核查条目。
- 外部论文：`s41598-026-50731-8.pdf`，13 页，SHA256 `a2fa4794efbe800011de3e78c4c77b4f9f06e228ac5462eaa045c93b553870b7`；本机副本位于 `C:\Users\Administrator\AppData\Local\Temp\MicrosoftEdgeDownloads\951f7623-8698-4bac-8aab-12f2aacc44a8\`。
- 关键证据：`reports/feasibility/p1_natural_closed_loop_v1_20261113/`、`reports/diagnostics/p1_common_state_mechanism_v1_20261113/`、`reports/diagnostics/g2_d1_score_margin_addendum_20261111/`、`docs/audits/r1_stage_b_20261118/`。

## 12. 参考

- Chen X. *Robust online multi-object tracking with conditional diffusion motion hypotheses and time-aware contrastive prototypes.* Scientific Reports 16:19960 (2026). https://doi.org/10.1038/s41598-026-50731-8
- Lv W. et al. *DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction.* CVPR 2024.
- Huang C. et al. *DeconfuseTrack: Dealing with Confusion for Multi-Object Tracking.* CVPR 2024.
- Sentinel, Scientific Reports 2026: https://www.nature.com/articles/s41598-026-43938-2
- 本仓库：`docs/r1_stage_a_literature_and_protocol_20261118.md`、`docs/r1_stage_b_report_20261118.md`、`docs/research_route_proposal_20261113.md`、`docs/research_governance/innovation_registry.md`。
