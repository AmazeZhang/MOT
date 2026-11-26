# S09：CAMELTrack 阈值标定、两阶段基线补齐与闭环失败归因

> 状态：**设计已给出，服务器尚未执行，运行锁待服务器生成（`NOT_RUN`）**。
> 协议：[s09_camel_calibration_and_attribution_20261126.yaml](../../manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml)。
> 前置权威报告：[S08 报告](../../reports/baselines/s08_camel_baseline_20261125/README.md) 与 [S08 复盘及文献审查](../research_status_review_20261126.md)。
> 规范依据：[研究端任务规范](../experiment_task_contract.md)、[任务书模板](../templates/server_experiment_task.md)、[服务器报告规范](../experiment_reporting.md)。

## 身份与决策

- **Task ID / 版本 / 日期：** `S09`（`s09_camel_calibration_and_attribution_20261126`） / v1 / 2026-11-26。
- **前置报告与停止范围：** S08（`s08_camel_baseline_20261125`）`COMPLETED / CONTINUE`。既有 P1/R1/R2/E0–E5/S06/S07 停止决定继续保持，不重启、不续跑 seeds 37/73。
- **当前执行状态与自动推进依据：** `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）。用户将本任务交给服务器后，Stage A–E 在前置门通过且预算内自动顺序推进；到 Stage E 结束、或遇到未解决阻断/预算上限时停止并回传，不自动启动 S10 新方法训练。
- **唯一主问题（平台标定与机制归因）：** 在严格仅使用 `motion_train32` 与 GT-free `motion_dev8` 的条件下，S08 中 `T_SCRATCH`（HOTA `53.981` / AssA `36.096` / IDSW `1451`）与平台参考之间的差距，有多少分别来源于 **(1) `sim_threshold=0.1` 未校准**、**(2) 缺少官方 Stage 1 单线索 TE 预训练**、以及 **(3) 残余闭环错误中“高分平局/边际失配 vs. 历史污染级联 vs. 单步特征几何不可分辨”三类内因的相对占比**？
- **任务类别、允许主张与不允许主张：**
  - 类别：基线标定 + 只读/特权机制诊断（**非创新方法验证**）。
  - 允许主张：同数据同平台下校准阈值及两阶段训练的描述性基线变化；`motion_dev8` 上残余 IDSW 的内部分数与干净历史探针诊断占比。
  - 不允许主张：不得将 `dev8` 上选出的阈值直接宣称为独立泛化；不得将 D2（GT 路由干净历史探针）输出当成可部署方法或在线指标；不得借 S09 提前宣称 A'/B'/C' 任一创新成立。
- **结果如何决定 S10：**
  - 若 Stage D 显示残余 IDSW 集中在低边际双高分互换（`|s_pos - s_best_competing| < 0.05`）或静态阈值两难，且 D2 干净历史恢复有限 → S10 转向 **A'（分配边际对齐与动态拒识）**；
  - 若 D2 干净历史探针明显消除了 `0037`/`0044` 类长程 AssA 崩塌与多数连续 IDSW → S10 转向 **B'（抗闭环自相关污染与一致历史聚合）**；
  - 若即使在 D2 干净历史下，近邻竞争帧仍大量出现 `s_best_competing > s_pos` → S10 转向 **C'（GAFFE 显式成对运动残差与方向遮挡边偏置）**；
  - 若 S09 执行遇不可解决工程阻断或结果互相矛盾 → 停止回传 `INCONCLUSIVE`，研究端先行排障。

## 科学设计

- **可否定假设（诊断判别式）：**
  - H_thresh：S08 `T_SCRATCH` 的部分 IDSW 源于 `sim_threshold=0.1` 过松；校准至 `0.50` / `computed_sim_threshold` 应能减少跨轨低分误接，但不足以独自解决密集交叉区的连续帧互换。
  - H_twostage：补齐官方 Stage 1 单线索 TE 预训练后，`T_TWOSTAGE` 在 `motion_train32` 下的 `dev8` AssA 应明显高于单阶段 `T_SCRATCH`。
  - H_mechanism：Calibration 与 Two-Stage 之后的残余官方 IDSW，将由 D1（内部分数 margin）与 D2（GT 路由干净历史 vs. 闭环历史）分离为“决策边际/阈值两难”、“闭环历史污染”或“单步几何/表征不可分辨”三类可观测份额。
- **最强相关对照与唯一变化因素：**
  - Stage B（阈值对照）：固定 S08 `T_SCRATCH` epoch 20 `last.ckpt`（SHA256 `fb86a65eab1d72c271c73b27e8155dae5f162545d5c14f0e1b61daad2aed7808`）与 `P_OFFICIAL`（SHA256 `cecce022fb51aa58e149ebb3f3cd408c6e4086266d7b126576f99bdc1b3f9075`），唯一变化因素为推理 `sim_threshold`；
  - Stage C（训练范式对照）：同 `motion_train32`、同三线索输入、同 seed 13、同共享推理政策，对比单阶段 `T_SCRATCH`（S08 已有）与官方两阶段 `T_TWOSTAGE`（Stage 1 分别预训练 3 个 TE + Stage 2 联合 GAFFE 训练 20 epochs）；
  - Stage D（机制归因对照）：在 `T_TWOSTAGE` 同一权重、同一检测输入下，对比**正常闭环推理**与 **D2 干净历史路由探针**（唯一变化因素为推理时 tracklet 历史 gallery 是否由 GT 保持身份纯净）。
- **核心方法与实现责任：**
  - **已核验上游依据**（[CAMELTrack@bc1c2610](https://github.com/TrackingLaboratory/CAMELTrack/tree/bc1c261075b0beacfe71c2896453be83643153c8)）：Stage 1 单线索预训练通过将 `gaffe._target_` 设为 `cameltrack.architecture.Identity`、仅保留单个 encoder（`app_encoder` / `kp_encoder` / `bbox_encoder`）完成；Stage 2 通过 `temporal_encoders.<cue>.checkpoint_path` 加载各 TE 权重并联合训练 GAFFE。
  - **服务器待实现项（本地未编写执行脚本）**：
    1. 在 Stage A 核对已固定本地仓库 `/media/imc/data/mot-data/third_party/CAMELTrack` 中 Stage 1（3 个 TE 各自配置文件/默认 epoch/学习率/样本数）与 `load_from_checkpoint` 对部分 encoder 加载的实际代码路径，导出解析配置并记录完整命令；
    2. 编写 `scripts/run_s09_camel.py` 与 `scripts/export_s09_report.py`（复用 S08 隔离缓存与 GPU 安全检查），支持 Stage B 阈值扫描、Stage C 两阶段训练（新进程显式重载校验）、Stage D1 内部分数记录、Stage D2 GT 路由干净历史探针；
    3. 本任务书给出的配置键仅为上游机制说明，**不是已测试可直接运行的启动命令**；实际解析命令须写入 `commands.sh` 与 `resolved_configs/`。
- **数据来源、精确 split 与历史使用：**
  - 严格沿用 S08 经校验且未变动的 DanceTrack `train` 内部互斥划分（[split 清单](../../manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml)，SHA256 `dd0b7fefd353532e04fb36206c7a1cf624cc661054b198653173f23e320926ff`）：
    - `motion_train32`（32 段，34,231 帧）：`dancetrack0001, 0002, 0006, 0008, 0015, 0016, 0023, 0027, 0029, 0032, 0039, 0045, 0049, 0051, 0052, 0055, 0057, 0061, 0062, 0066, 0068, 0069, 0072, 0074, 0075, 0080, 0082, 0083, 0087, 0096, 0098, 0099`；
    - `motion_dev8`（8 段，7,565 帧）：`dancetrack0012, 0020, 0024, 0033, 0037, 0044, 0053, 0086`。
  - 直接复用 S08 阶段 A 生成的隔离状态文件（须在 S09 Stage A 校验 SHA256 与 S08 一致，不得覆盖原 S08 运行目录）：`states_train32.pklz`、`states_dev8_train_ref.pklz`、`states_dev8_gt_free.pklz`。
  - 历史使用声明：`motion_train32` / `motion_dev8` 均属历史复用开发划分，非全新留出集；官方 `val` 与 `test` 本轮均不碰。`P_OFFICIAL`（train+val 预训练）仅作平台参考，不用于宣称受控训练优势。
- **GT 边界与因果时序：**
  - Stage B & Stage C3 推理严格使用 `states_dev8_gt_free.pklz`，禁止访问 GT 或未来帧；
  - Stage C1/C2 仅在 `motion_train32` 训练目标中使用 GT；
  - Stage D1（相似矩阵记录）严格在 GT-free 正常闭环推理中记录相似度分数，仅离线对照 TrackEval IDSW 切片读取；
  - **Stage D2（GT 路由干净历史探针）明确标记为特权离线诊断**（`gt_role: privileged_clean_history_counterfactual_not_deployable`）：每帧决策仍只读取过去至多 `max_track_gallery_size=50` 帧的检测特征且不得读取未来帧，但更新轨迹历史缓存时由当帧 GT 一对一匹配路由决定特征归属（避免异主观测写入 gallery）。D2 指标必须单列，不得混入在线模型排名或声称线上性能。
- **初始化、训练配置、参数与样本暴露：**
  - `T_TWOSTAGE`（seed 13）全部从随机初始化开始（`app_encoder` / `kp_encoder` / `bbox_encoder` / `gaffe` 均禁止加载官方 `P_OFFICIAL` 权重）；
  - Stage C1（三个单 cue TE 预训练）：优先沿用 pinned CAMELTrack 仓库自带的单线索 TE 预训练配置（学习率、epoch 数、batch、loss、增广）；若仓库中存在多套预设（如不同数据集/通用配置），按“DanceTrack 配置优先 → 默认通用配置”解析并在运行前写入 `resolved_configs/`，记录每个 TE 独立训练的步数、有效参数量、总耗时和 checkpoint SHA256；
  - Stage C2（联合训练）：固定与 S08 Stage D 相同的数据预算与调度（`train_add_val=false`、`multi_dataset_training=false`、20 epochs、FP32、batch 8、`num_samples=32`、`samples_per_video=100`、`max_length=50`、`lr=1e-4`、`weight_decay=1e-4`、seed 13），从 C1 的三个 TE checkpoint 加载编码器权重，GAFFE 随机初始化，唯一选择规则为 `epoch 20 last.ckpt`。记录总参数、分模块可训练参数与累计更新步数（C1 + C2 分别列明）。
- **指标、配对单位与缺失规则：**
  - 主指标：`COMBINED_SEQ` 官方 TrackEval HOTA、AssA、IDF1、DetA、MOTA（0–100 分，全精度导出），计数 IDSW、FP、FN、Frag；配对单位为视频（8 段 `dev8`）。
  - 每次独立闭环推理必须在独立 `run_id` 目录执行，新进程显式重载指定 ckpt 并核验 SHA256，每个 run 导出全部 8 段 + `COMBINED_SEQ`。失败或未运行组保留空行并写 `missing_reason`，不以部分视频替代完整聚合。

## 执行矩阵

| stage / ID | 输入与范围 | 操作 / 命令入口及实现责任 | 必须回传 | 通过条件 | 通过后 / 失败后 |
|---|---|---|---|---|---|
| **A：资产复核与源码实现** | S08 锁定环境 `/media/imc/data/mot-data/envs/s08-camel`、S08 状态文件、S08 `T_SCRATCH` ckpt、`P_OFFICIAL` ckpt | 服务器核对 S08 资产 SHA256，检查 CAMELTrack Stage 1 配置与模型加载语义，编写并编译 `scripts/run_s09_camel.py` & `scripts/export_s09_report.py` | `source_audit.json`、`resolved_configs/`、`commands.sh`、`runtime_lock.json`、Targeted tests | 状态/ckpt SHA256 与 S08 一致，Stage 1 配置来源明确，新脚本单元测试通过 | 通过进 B；依赖或源码兼容阻断超过 2 次修复则停止回传 |
| **B：零重训阈值标定** | `dev8` GT-free 状态（7,565 帧）；`T_SCRATCH` @ `{0.46, 0.50, 0.519729}`；`P_OFFICIAL` @ `{0.50}`（另按 SHA 引用 S08 二者 @ `0.1`） | 服务器执行 4 次独立 Stage E 闭环推理与 TrackEval（`use_computed_sim_threshold=false`，显式设 `sim_threshold`） | 4 个新 run 的全精度汇总/逐视频指标、与 S08 `0.1` 对齐的总表 | 4×8=32 条逐视频 + 4 条 `COMBINED_SEQ` 完整导出，所有输入/ckpt SHA 校验一致 | 通过进 C；失败保留缺失行并尝试最多 2 次工程修复 |
| **C：Two-Stage 基线训练与评测** | `motion_train32` 训练，`motion_dev8` dev loss 与 GT-free 闭环 | 服务器执行 C1（3 个 TE 预训练，各自独立隔离工作目录）→ C2（联合训练 20 epochs，`last.ckpt`）→ C3（`dev8` 评测 `T_TWOSTAGE` @ `0.50` 与自身 `val/computed_sim_threshold`） | C1 三个 TE 曲线与 ckpt SHA；C2 20 epoch 曲线与 `last.ckpt` SHA；C3 2 组 `dev8` 全精度指标 | C1/C2 全程 0 NaN/Inf、有效梯度更新、新进程显式重载一致；C3 导出 2×8=16 条逐视频 + 2 条 `COMBINED_SEQ` | 通过进 D；OOM 不自动改 batch/精度，保存峰值/形状后停止 |
| **D：残余失败机理归因** | `T_TWOSTAGE` @ `0.50`（或最优参考阈值组，在运行前按规则确定并锁定为“C3 中 AssA 较高阈值或固定 0.50 组”，协议写明为 **`T_TWOSTAGE @ 0.50` 为主、`T_SCRATCH @ 0.519729` 作补充**） | 服务器实现并运行：<br>**D1**：对全部官方 IDSW 事件回放记录帧内相似度矩阵，导出 `s_pos`、`s_assigned`、`s_best_competing`、margin 及失联间隔；<br>**D2**：运行 GT 路由干净历史探针（不改权重、单列 `privileged_clean_history`），导出同 `dev8`（含 `0037`、`0044`）TrackEval 指标与 IDSW 变化 | `diagnostics.json`、D1 逐 IDSW 边际统计 CSV/JSON、`cases.jsonl`（D1/D2 对照样本最多 16 例）、D2 `per_sequence_metrics.csv`（标明特权诊断角色） | D1 覆盖 100% 目标条件官方 IDSW 事件；D2 完成全部 8 段且未泄露未来帧、未污染正常输出目录 | 通过进 E；若某帧无法解析 GT-tracklet 路由映射则单列 `unknown` 不得静默丢弃 |
| **E：导出标准包并停止** | A–D 全部权威产物与失败日志 | 运行 `scripts/export_s09_report.py` 生成标准包并核对哈希与文档更新 | `reports/diagnostics/s09_camel_calibration_and_attribution_20261126/` 全套文件；更新 `reports/README.md`、`docs/progress_log.md`、`docs/a2a/LATEST.md` | 所有清单、来源 SHA、覆盖检查通过 | **停止在 E，交由研究端决定 S10**；无授权不 commit/push |

- **预期规模计数：**
  - 完整跟踪闭环条件：Stage B 新增 4 个条件×1 seed×8 视频 + Stage C3 新增 2 个条件×1 seed×8 视频 = **6 组新在线闭环（48 条逐视频 + 6 条 `COMBINED_SEQ`）**，另加 S08 已有 2 组只读参考引用，以及 Stage D2 特权诊断 1 组（单列标注，不混入主表）。
  - 训练运行：Stage C1（3 个 TE）+ Stage C2（1 个联合模型）= 4 次独立训练运行（均为 seed 13）。
- **待解析项：**
  - CAMELTrack 仓库中 3 个单 cue TE 预训练默认超参（如 `max_epochs`、`lr`、`batch_size`、`samples_per_video`）。A 阶段从固定仓库 `/media/imc/data/mot-data/third_party/CAMELTrack` 配置文件直接读取并写入 `resolved_configs/stage_c1_*.yaml`；若仓库未提供单个 TE 独立 yaml，则采用 `cameltrack_train.yaml` 默认配置 + `gaffe._target_=cameltrack.architecture.Identity` + 单 encoder，并在 `run_manifest.json` 明确记录来源，若该途径也不可用则阻断 C 并回报。
- **预算与修复上限：**
  - 单张空闲物理 GPU 1–7（首选物理 GPU 1，**GPU 0 永久禁用**）；单卡并发数 1；
  - 总 GPU 预算 **≤ 16 卡时**（A ≤ 0.5h、B ≤ 1h、C ≤ 12h、D ≤ 2.5h）；墙钟 **≤ 48 小时**；主动工程 ≤ 2 个工作日；新增存储 ≤ 40 GiB 且数据盘剩余 ≥ 50 GiB；
  - 同一工程阻断最多 **2 次**有针对性兼容修复（新 `attempt` / `run_id`，保留失败日志）；第 3 次仍堵或 OOM/预算耗尽立即停止保存 `PARTIAL`。

## 回传和停止点

- **权威报告目录：** `reports/diagnostics/s09_camel_calibration_and_attribution_20261126/`（遵循 [report_bundle](../templates/report_bundle/README.md)，大型产物与 checkpoint 保留在 `/media/imc/data/mot-data/runs/s09_camel_calibration_and_attribution_20261126/`）。
- **必须包含的文件：** `README.md`、`run_manifest.json`、`checks.json`、`artifacts.json`、`aggregate_metrics.csv`（标明 `online_closed_loop` 与 `privileged_clean_history_probe` 区分列）、`per_sequence_metrics.csv`、`learning_curves.csv`（覆盖 C1 三个 TE 与 C2 20 epochs）、`diagnostics.json`、D1 分数分布摘要、`cases.jsonl` & `evidence/`、`source_audit.json`、`resolved_configs/`、`commands.sh`、`runtime_lock.json`。
- **研究端需要仅凭 Git 回答的三个问题：**
  1. 校准 `sim_threshold`（Stage B）与补齐 `train32` 两阶段 TE 预训练（Stage C）后，CAMELTrack 自训基线在 `dev8` 的真实 HOTA / AssA / IDSW 落在何处？
  2. 校准后残余的官方 IDSW 中，“低分强连”、“高分平局互换（`|margin| < 0.05`）”和“错配反超（`s_pos < s_neg - 0.05`）”各占多少比例？
  3. GT 路由干净历史探针（Stage D2）相对闭环 `T_TWOSTAGE` 挽回了多少 AssA 与多少 IDSW（特别是 `0037` / `0044`），从而决定 S10 立项 **A'（边际与拒识）**、**B'（抗闭环历史污染）** 还是 **C'（显式成对几何与方向遮挡偏置）**？
- **最后一个允许自动执行的步骤与下一轮不运行项：**
  - 最后一个自动步骤为 **Stage E（报告导出与本地文档同步）**。
  - **本轮明确不运行**：任何新增损失/新网络头/边偏置实验（S10 范围）、MOTIP/Flow Matching、多 seed 37/73、官方 `val`/`test` 或 SportsMOT 评测、自动 `git commit` / `git push`。
