# S11：CAMELTrack 低阈值标定、149 例 Onset 解剖与冲突感知拒写/干净锚点无重训验证

> 状态：**设计已给出，服务器尚未执行，运行锁待服务器生成（`NOT_RUN`）**。
> 协议：[s11_camel_onset_and_conflict_gate_20261127.yaml](../../manifests/protocols/s11_camel_onset_and_conflict_gate_20261127.yaml)。
> 前置权威报告：[S09 报告](../../reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md) 与 [S10 报告](../../reports/feasibility/s10_camel_anticontamination_20261127/README.md)。
> 规范依据：[研究端任务规范](../experiment_task_contract.md)、[任务书模板](../templates/server_experiment_task.md)、[服务器报告规范](../experiment_reporting.md)。

## 身份与决策

- **Task ID / 版本 / 日期：** `S11`（`s11_camel_onset_and_conflict_gate_20261127`） / v1 / 2026-11-27。
- **前置报告与停止范围：** S10（`s10_camel_anticontamination_20261127`）`COMPLETED`。`P1/R1/R2/E3/E5/S03/S04/S06` 与 S10 `C2/C3` 局部合成接管路线停止决定维持不变，不重跑训练、不扩展 `C3`。
- **当前执行状态与自动推进依据：** `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）。交付服务器后，Stage A–E 在前置门通过且预算内自动顺序推进；到达 Stage E 结束或触发阻断/预算上限即停止回传，不开启任何新训练或官方 `val`/`test` 提交。
- **唯一主问题：**
  严格零重训复用 S09 `T_TWOSTAGE` checkpoint，在不借助任何 GT、也不改变特征抽取与网络权重的前提下：
  1. 补齐 `T_TWOSTAGE` 在 `0.50` 以下阈值（`0.10 / 0.30 / 0.40`）的 `dev8` 标定，确认降阈是否能像 `P_OFFICIAL` 那样拉回 `FN`（`5615`）；
  2. 在 `T_TWOSTAGE_TH050` 真实闭环回放上，对 S10 Stage D 暴露的 **`149` 例首次起跳切换（onset）** 与 **`31,791` 例漂移后自洽尾段** 做 100% 机制解剖（`gap`、空间重叠竞争、匹配边际、`2-cycle` 互换、以及早期干净锚点 vs 近期滑窗外观漂移的可分性）；
  3. 验证与 S10 `C0`（全局卡 `sim < 0.60/0.65` 导致 `FN` 暴增）截然不同的 **竞争冲突感知条件拒写（Conflict-Gated Write）与早期干净锚点保留（Early-Anchor Pinning）** 能否在不伤及孤立快动目标的前提下，在 `dev8` 闭环中稳健超越 `T_TWOSTAGE_TH050`（`54.443` HOTA / `36.704` AssA）并吃下 `D2_SKIP_ONLY`（`63.781` / `49.940`）的显著部分？
- **任务类别、允许主张与不允许主张：**
  - 类别：零重训推理标定 + 机制归因诊断 + 启发式可部署基线可行性检验（`0` 训练运行，复用 `seed 13` 权重与 `motion_dev8`）。
  - 允许主张：`T_TWOSTAGE` 的低阈值特性、149 例真实 onset 切换的定量病因构成、长期锚点对比对“后切换自洽尾段”的判别力，以及冲突感知拒写/锚点保留策略在复用 `dev8` 上的真实闭环净效果与与代价。
  - 不允许主张：**严禁将任何纯推理拒写/锚点保留启发式包装为“方法级架构创新”**；不得用复用 `dev8` 宣称未知数据集泛化；不得在看到 Stage B/D 结果后事后微调新阈值。
- **结果如何决定 S12：**
  - **情况 1**：`G1`–`G4` 中至少一组在不引发 `FN` 恶化的前提下明显优于 `T_TWOSTAGE_TH050`（`54.443` HOTA / `36.704` AssA）与 `C0` → 确立首个胜过基线的零重训防污染底座，并依据 Stage C 149 例 onset 的主导病因（遮挡反转 vs `gap>1` 重连 vs `2-cycle` 互换）决定 S12 主攻 **`C'`（GAFFE 成对运动/方向遮挡边偏置）** 还是叠加 **`A'`（边际/2-cycle/延迟解歧）**；
  - **情况 2**：无论全局阈值还是冲突条件拒写都无法显著改善闭环（净增有限或仍大量漏掉高分反转），且 Stage C 显示多数 onset 为高遮挡强反转（`s_pos < s_neg - 0.05`）或 `2-cycle` 互换 → 明确“只在写库侧刹车不足以止损”，S12 直接从关联打分/分配端立项 **`C'` + `A'`**（并评估 Stage C 中早期锚点双层对账是否值得保留为推理组件）。

## 科学设计

- **可否定假设：**
  - `H_lowth`：降低 `T_TWOSTAGE` 的 `sim_threshold`（`0.10/0.30/0.40`）虽然能减少 `FN`，但是否同步推高 onset 切换（`IDSW`）而抵消收益？
  - `H_anchor`：引入轨迹“创生早期干净特征锚点 vs 近期滑窗”的 ReID 余弦距离 `anchor_reid_dist`，能够在离线只读诊断上打破 S10 的后切换自洽陷阱（使 `post_switch` 与全局 `cross_id` 判别 AUROC 显著脱离 `0.500` 水平）。
  - `H_conflict_gate`：把 `C0` 的全局 `sim < tau` 改写为**仅当空间邻近竞争（`IoU >= 0.30`）或分配低边际（`margin < 0.10`）存在时**才跳过写入 gallery（并在 `G4` 中保护最早 5 个无冲突成熟帧不被 FIFO 淘汰），能够消除 `C0` 对 `dancetrack0024` 等孤立快动目标的 `FN` 崩塌，并在 `motion_dev8` 上同时超越 `T_TWOSTAGE_TH050` 与 `C0`。
- **最强相关对照与唯一变化因素（全部预写固定，禁止事后调参）：**
  - **既有参考基线（直接引用，不重跑）**：S09 `T_TWOSTAGE_TH050`（`0.50`, `54.443` HOTA / `36.704` AssA / `786` IDSW / `5615` FN）、`T_TWOSTAGE_TH_COMP`（`0.5460`, `51.712` HOTA）、S10 `C0_TAU060`（`48.838`）/ `C0_TAU065`（`43.509`）、S10 `D2_SKIP_ONLY`（特权纯拒写参考上限 `63.781` HOTA / `49.940` AssA）。
  - **Stage B 阈值对照（唯一变化：`sim_threshold`）**：`T2_TH010`（`0.10`）、`T2_TH030`（`0.30`）、`T2_TH040`（`0.40`）。
  - **Stage D 推理门控四组（全部固定 `sim_threshold=0.50`, `min_hits_to_gate=3`）**：
    - `G1_CONFLICT_IOU030_TAU065`：对 `trk.hits >= 3` 的匹配，计算 `det_max_other_iou`（与同帧其他有效检测的最大 IoU）与 `trk_max_other_iou`（与同帧其他活跃轨迹 `last_detection` 的最大 IoU）。**仅当 `(det_max_other_iou >= 0.30 or trk_max_other_iou >= 0.30)` 且 `sim_val < 0.65` 时**跳过写库（仍更新 `hits/hit_streak/time_wo_hits=0/last_detection`）。与 `C0_TAU065` 仅差“空间竞争前置条件”。
    - `G2_MARGIN_M010_TAU070`：计算双向分配边际 `assignment_margin = sim_val - max(best_other_trk_sim, best_other_det_sim)`（对同一检测取除当前轨外的最高相似度轨迹，对同一轨迹取除当前检测外的最高相似度检测，无第二候选时边际记为 `+inf`）。**仅当 `assignment_margin < 0.10` 且 `sim_val < 0.70` 时**跳过写库。检验“低边际竞争门控”相对 `G1` 的差异。
    - `G3_HYBRID_GATE_NO_PIN`：**当 `(det_max_other_iou >= 0.30 or trk_max_other_iou >= 0.30 or assignment_margin < 0.10)` 且 `sim_val < 0.68` 时**跳过写库。
    - `G4_HYBRID_GATE_EARLY_PIN5`：在与 `G3` 完全相同的写库过滤规则基础上，**唯一增量**：当轨迹成熟且发生**无冲突、已接受**的早期更新时，允许固定保留最多 `K = 5` 个最早的无冲突检测槽位在 `trk.detections[:K]` 中；当 `len(trk.detections) > max_track_gallery_size (50)` 时，仅从 `[K:]` 队列尾部执行 FIFO 弹出（不改变 `track.forward()` 对全部 `trk.detections` 正常递增 `age`、不增加 gallery 上限）。
- **实现责任（明确标注服务器待实现项）：**
  - 本地仅交付任务书与协议；服务器在 Stage A 核对 S09/S10 资产后，编写并单测 `scripts/run_s11_camel.py`、`scripts/export_s11_report.py` 与 `tests/test_s11_camel.py`：
    1. 复用 `scripts/run_s09_camel.py` 的资产加载与 `TrackEval` 评测函数；
    2. 在不访问任何 GT 的前提下实现 `ConflictGatedCAMELTrack`（支持 `G1/G2/G3/G4` 与统计计数）；
    3. 实现 Stage C `OnsetAnatomyReplayCAMELTrack`：GT 仅用于帧外标签归因（严禁流入检测、相似度、边际或任何决策逻辑），并在 `tracklet` 初始化/未受冲突时记录无冲突 ReID embedding 早期均值，计算 `anchor_reid_dist = 1 - cosine(det_app_emb, anchor_app_emb)` 以及 `2-cycle` 互换标记；
    4. `tests/test_s11_camel.py` 必须用合成张量验证：竞争 IoU 计算、排除自身的双向边际计算、`2-cycle` 检测、`G1–G3` 拒写时不改动 `trk.detections`、`G4` 严格满足 `len <= 50` 且前 5 个干净锚点在 80 步追加后保持不变。
- **数据来源、精确 split 与 GT 边界：**
  - 全部复用 [split 清单](../../manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml)（SHA256 `dd0b7fefd353532e04fb36206c7a1cf624cc661054b198653173f23e320926ff`）中的 `motion_dev8`（8 段，7,565 帧）：`[dancetrack0012, dancetrack0020, dancetrack0024, dancetrack0033, dancetrack0037, dancetrack0044, dancetrack0053, dancetrack0086]`。
  - 状态文件：复用 `states_dev8_gt_free.pklz` 执行所有闭环推理（Stage B、Stage D 及 Stage C 主推演）；仅 Stage C 离线标签回放与最终 TrackEval 读取 `motion_dev8` GT，且必须验证 Stage C 回放出的 `T_TWOSTAGE_TH050` MOT 输出与 S09 权威记录 / `T_TWOSTAGE_TH050` 计数对齐。
  - 共享推理政策（除 Stage B 标明的 `sim_threshold` 外）：`min_det_conf=0.4`、`min_init_det_conf=0.9`、`min_num_hits=0`、`max_wo_hits=150`、`max_track_gallery_size=50`、`use_computed_sim_threshold=false`、无插值后处理。
- **主指标、辅助指标与配对单位：**
  - Stage B/D 闭环主指标：`COMBINED_SEQ` 官方 TrackEval HOTA、AssA、IDF1（0–100 全精度）；次要指标：DetA、MOTA、IDSW、FP、FN、Frag；配对单位：8 段序列（逐视频与相对 `T_TWOSTAGE_TH050` / `C0` 胜/平/负及中位差）。
  - Stage C 离线机制指标：AUROC / AUPRC / F1 / 检出率 `@ 0.5` 与 `@ clean_FPR=5%` 分界点下的识别表现，及 149 例 onset 的完整分类计数表。

## 执行矩阵

| stage / ID | 输入与范围 | 操作 / 命令入口及实现责任 | 必须回传 | 通过条件 | 通过后 / 失败后 |
|---|---|---|---|---|---|
| **A：资产核验、脚本实现与单测** | S09 `T_TWOSTAGE` ckpt、S08 `states_dev8_gt_free.pklz` | 服务器校验 S09 `source_audit.json` 与 `20be35af...` 实际文件 SHA256；实现并运行 `scripts/run_s11_camel.py` 与 `tests/test_s11_camel.py` | `source_audit.json`、`runtime_lock.json`、`resolved_configs/`、`commands.sh`、单测输出 | SHA256 精确匹配且全部单元测试 PASS | 通过自动进 B；修复超过 2 次仍失败则停止 |
| **B：低阈值补充标定** | `T_TWOSTAGE` ckpt + `states_dev8_gt_free.pklz` | 依次运行 `T2_TH010` (`0.10`)、`T2_TH030` (`0.30`)、`T2_TH040` (`0.40`) 闭环跟踪与 TrackEval | 3×8=24 条逐视频与 3 条 `COMBINED_SEQ` 全精度指标 | 全部 24+3 行导出完整，导出误差 `0.0` | **通过自动进 C** |
| **C：149 例 Onset 解剖与锚点漂移离线探针** | `T_TWOSTAGE_TH050` 回放 + `dev8` GT（只读标签） | 运行 `OnsetAnatomyReplayCAMELTrack` 回放 `dev8`，对全部 `clean/onset/post_switch/fp` 写入导出 `gap`、`IoU`、双向 `margin`、`2-cycle` 标、早期干净锚点余弦距离，并复核各类判别 AUROC 与最多 16 例 onset 案例 | `diagnostics.json`（完整 149 例 onset 解剖 + `anchor_reid_dist` AUROC + 对比 `1-sim`）、`cases.jsonl` & `evidence/` | 覆盖全部 `dev8` 8 段，与 S10 `d_real_pollution_auroc_v1` 计数逻辑可对账，零 GT 泄漏进推理 | **通过自动进 D** |
| **D：零重训冲突感知拒写与早锚固定闭环评测** | `T_TWOSTAGE` ckpt (`@0.50`) + `states_dev8_gt_free.pklz` | 依次独立运行 `G1_CONFLICT_IOU030_TAU065`、`G2_MARGIN_M010_TAU070`、`G3_HYBRID_GATE_NO_PIN`、`G4_HYBRID_GATE_EARLY_PIN5` 闭环跟踪与 TrackEval | 4×8=32 条逐视频与 4 条 `COMBINED_SEQ` 指标 + 写入跳过率/触发条件统计 (`diagnostics.json`) | 零 GT 访问，全部 32+4 行完整且导出误差 `0.0` | 通过自动进 E；若有工程报错先修一次并保留失败目录 |
| **E：导出标准报告与同步交接** | A–D 全部产物 | 运行 `scripts/export_s11_report.py` 生成报告、校验哈希，更新 `reports/README.md`、`docs/progress_log.md`、`docs/a2a/LATEST.md` | `reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/` 完整报告包 | `checks.json` 与 `artifacts.json` 校验全通过 | **停在 E，交回研究端定案 S12 (`C'`/`A'`)**；无明确授权不 commit/push |

- **预期规模计数：**
  - 训练运行：**`0` 次**。
  - 在线闭环评测：**`7` 组**（Stage B 3 组低阈值 + Stage D 4 组冲突拒写/早锚策略；共 `7 × 8 = 56` 条逐视频 + `7` 条 `COMBINED_SEQ`）＋ S09/S10 参考只读引用。
  - 离线机制回放与解剖：**`1` 组**（Stage C，覆盖 `dev8` 全量写库与 `149` 例 onset 事件）。
- **待解析项：**
  - 核对 S09 `source_audit.json` 中 `c2_twostage_joint_seed13_e20_v1/CAMEL/last.ckpt` 的真实完整 SHA256 与文件路径（注：S10 协议中曾记录前缀 `20be35af...`，请 Stage A 直接以 S09 盘上文件实际哈希锁定并记录）。
  - 确认 `det.embeddings`（ReID 特征维数与 L2 归一化状态）及 `CAMELDataset`/`CAMEL` 推理时 `trk.detections` 内各帧特征的设备与形状；若多尺度/分块 embedding 存在多向量（例如 `6 x 128` 身体分区特征），在 `anchor_reid_dist` 中按分区平均有效余弦距离或全局主分量明确记录计算方式并在 `source_audit.json` 说明。
- **预算与安全约束：**
  - 物理 GPU 1–7 单卡（首选物理 GPU 1，**GPU 0 永久禁用且严禁触碰**）；
  - 总 GPU 预算 **`≤ 2.0` 卡时**（A ≤ 0.2h、B ≤ 0.4h、C ≤ 0.3h、D ≤ 0.9h、E ≤ 0.2h）；墙钟 **≤ 8 小时**；新增数据盘存储 **`≤ 5 GiB`**（保留 ≥ 50 GiB 空闲）；
  - 每项工程阻断最多 **2 次**有针对性兼容修复，保留失败 `run_id` 目录。

## 回传和停止点

- **权威报告目录：** `reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/`（大型产物放 `/media/imc/data/mot-data/runs/s11_camel_onset_and_conflict_gate_20261127/`）。
- **研究端需要仅凭 Git 回答的三个问题：**
  1. `T_TWOSTAGE` 在低阈值 `0.10 / 0.30 / 0.40` 下，究竟恢复了多少 `FN`、是否同步增加了 onset 切换与 `IDSW`？
  2. `149` 例真实闭环 onset 切换的具体物理分布是什么（`gap>1` vs 连续帧、`IoU>=0.3` 竞争比例、反转 vs 平局、`2-cycle` 互换占比）？“早期干净锚点 vs 近期滑窗”余弦距离 `anchor_reid_dist` 能否有效识别 `31,791` 帧漂移自洽尾段与 `149` 例 onset？
  3. 零重训的竞争条件拒写与早期锚点保留（`G1`–`G4`）是否克服了 S10 `C0` 的 `FN` 灾难并在 `dev8` 闭环显著提升 AssA / HOTA？还剩多少结构性误差必须靠 S12 改训练端/关联器（`C'`/`A'`）解决？
- **最后一个允许自动执行的步骤与本轮不运行项：**
  - 最后自动步骤为 **Stage E（报告导出与文档同步）**。
  - **本轮明确不运行**：任何模型训练、阈值网格搜索（超出 `0.10/0.30/0.40` 与 `G1–G4` 预设之外）、multi-seed（37/73）、官方 DanceTrack `val`/`test`、SportsMOT、以及未授权的 `git commit` / `git push`。
