# S10：CAMELTrack 纯拒写基准分解与抗闭环污染训练/一致性去污实验

> 状态：**设计已给出，服务器尚未执行，运行锁待服务器生成（`NOT_RUN`）**。
> 协议：[s10_camel_anticontamination_20261127.yaml](../../manifests/protocols/s10_camel_anticontamination_20261127.yaml)。
> 前置权威报告：[S09 报告](../../reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md) 与 [S09 复盘与设计说明](../research_status_review_20261127.md)。
> 规范依据：[研究端任务规范](../experiment_task_contract.md)、[任务书模板](../templates/server_experiment_task.md)、[服务器报告规范](../experiment_reporting.md)。

## 身份与决策

- **Task ID / 版本 / 日期：** `S10`（`s10_camel_anticontamination_20261127`） / v1 / 2026-11-27。
- **前置报告与停止范围：** S09（`s09_camel_calibration_and_attribution_20261126`）`COMPLETED`。既有 `P1/R1/R2/E0–E5/S03/S04/S06` 停止决定不变，不重启旧分支、不续跑 `S06 seeds 37/73`。
- **当前执行状态与自动推进依据：** `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）。用户交付服务器执行后，Stage A–E 在前置门通过且预算内自动顺序推进；到 Stage E 结束或遇到未解决阻断/预算上限时停止回传，不自动开启多 seed 确认或官方 `val`/`test` 提交。
- **唯一主问题：**
  1. 先剥离 S09 D2 `v3` 中“纯拒写污染”与“GT 补写漏配帧”的各自贡献（Stage B `D2_SKIP_ONLY`）；
  2. 在 `motion_train32`（同 C1 TE 初始化、同 20 epochs、seed 13、`sim_threshold=0.50`）下，相比**零重训置信度拒写对照（`C0`）**与**配平总交换比例的独立随机交换基线（`C1`）**，**连续异主接管污染训练（`C2`）**及**多线索时序一致性门控聚合（`C3`）**能否实质提升严格在线 `motion_dev8` 闭环 AssA / HOTA / IDF1，且不增加不可接受的碎片或 FN？
- **任务类别、允许主张与不允许主张：**
  - 类别：特权分解对照（Stage B）＋受控开发集方法探索（Stage C/D，单 seed `13`）。
  - 允许主张：`motion_dev8` 上同初始化、同数据预算、同推理阈值下的受控开发比较，以及一致性头对真实污染写入的离线检出诊断。
  - 不允许主张：不得用单 seed `motion_dev8` 结果宣称独立泛化或 SOTA；不得把 `D2_SKIP_ONLY` 当在线方法；若 `C0` 简单置信度拒写已解释 `C2/C3` 大部分收益或 `C3` 未胜过 `C1`，不得强行主张“一致性去污机制成立”。
- **结果如何决定下一步（S11）：**
  - 若 `C3`（或 `C2`）在 `dev8` 上**同时超过** `T_TWOSTAGE_TH050`、`C0_SIMPLE_WRITE_GATE` 与配平 `C1_MATCHED_IID_SWAP`（AssA/HOTA/IDF1 方向一致提升、且一致门控对真实污染有明显判别信号）→ 冻结机制并在 S11 扩展至多 seed（13/37/73）及未调参独立确认集（DanceTrack `val` + SportsMOT）；
  - 若 `C0_SIMPLE_WRITE_GATE` 已达到甚至超过 `C2/C3` → 停止复杂污染训练/门控头，分析为何简单分数拒写足够或残余单帧反转原因；
  - 若 `C2/C3` 均未显著优于 `C1` 或闭环无净收益（且 `D2_SKIP_ONLY` 显示漏配断档或单帧反转占主导）→ 终止当前纯 `B'` 扩展，转而设计 `A'`（分配边际与动态拒识）或 `C'`（成对几何与方向遮挡偏置）。

## 科学设计

- **可否定假设：**
  - `H_skip`：`D2_SKIP_ONLY`（仅阻止异主与 FP 污染写入，不做 `else:` GT 注入）仍能比闭环 `T_TWOSTAGE_TH050`（AssA `36.704`）明显改善 AssA，证明抑制写入污染本身具备真实价值。
  - `H_takeover`：保持总被交换历史帧占比与 `C1` 配平，保留时间连续的异主接管结构（`C2`）比独立随机交换（`C1`）更能改善 `dev8` 闭环 AssA。
  - `H_gate`：在训练期利用已知历史污染掩码监督多线索一致性权重 $w_t \in (0, 1)$ 并以 $\log(w_t + \epsilon)$ 软门控时序/跨目标聚合（`C3`），能够在不依赖 GT 的闭环推理中有效识别真实异主污染写入，并进一步优于 `C0` 简单分数拒写与 `C2`。
- **最强相关对照与唯一变化因素：**
  - **S09 既有基线参考（复用，不重训）**：`T_TWOSTAGE_TH050`（HOTA `54.443`, AssA `36.704`, IDF1 `53.629`, IDSW `786`）与 S09 `D2 v3`（`81.907` HOTA / `80.063` AssA）。
  - **`C0_SIMPLE_WRITE_GATE`（零重训强简单对照）**：完全使用 S09 `T_TWOSTAGE` ckpt 与 `sim_threshold=0.50`，预设两个固定写库阈值 `tau_write ∈ {0.60, 0.65}`（`C0_TAU060`, `C0_TAU065`）：当已存在轨迹 `trk.hits >= 3` 且当前匹配相似度 `sim_val < tau_write` 时，仍正常执行本次匹配与状态输出（保持匹配和 `time_wo_hits=0`），但**跳过将该检测加入 `trk.detections` gallery**（不改权重、不用 GT）。
  - **`C1_MATCHED_IID_SWAP`（等噪声量强训练对照）**：与 `C2` 完全共享 S09 C1 三个 TE 初始化、20 epochs 联合配置、batch/lr/seed 13，唯一区别是用独立随机帧交换替代连续接管，且**实际平均每条 tracklet 被交换帧比例与 `C2` 配平（实测差异 ≤ 5%）**。
  - **`C2_CONTIGUOUS_TAKEOVER`（连续接管训练分布因素）**：在 `motion_train32` 联合训练的 tracklet 变换中引入连续异主接管：从同视频空间重叠或邻近竞争轨迹（优先 `IoU >= 0.1`，不足时取最近邻 GT 轨迹）抽取连续 $L \in [2, 12]$ 帧观测替换当前 tracklet 历史段（约束接管前保留干净前缀长度 $\ge 2L$、接管段末端落在近当前帧窗口 `[-15, -1]` 内），**不改网络结构与损失**。
  - **`C3_CONSISTENCY_GATED_TE`（多线索一致性门控因素）**：在 `C2` 完全相同的训练数据增强与配置基础上，仅新增轻量因果一致性门控头（参数量预期 < 总参数 2%）及辅助 BCE 监督：
    - 读取训练变换精确产出的二值污染掩码 $m_t \in \{0, 1\}$（`1` 表示该历史帧为增强/接管注入的异主观测，`0` 表示干净原身份观测；**严格仅限训练期使用，推理期禁用**）；
    - 对每个 tracklet 历史时刻 $t$，结合三线索时序表征（或拼接后的观测 token 序列）输出一致性概率 $w_t = \sigma(g_\theta(h_t)) \in (0, 1)$，辅助损失 $\mathcal{L} = \mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{cons}} \mathcal{L}_{\text{BCE}}(w_t, 1 - m_t)$（固定 $\lambda_{\text{cons}} = 0.2$，不做网格搜索）；
    - 在推理与训练的 `TemporalEncoder` / `GAFFE` 轨迹池化聚合中，将 $\log(\text{clamp}(w_t, 10^{-4}, 1.0))$ 作为非侵入式注意力加权偏置（使低一致历史观测在生成轨迹表征时自动降权）；
    - 在 Stage D 另追加一个零重训推理切片 `C3_WITH_SOFT_WRITE_FILTER`：当最新写入观测（或上一步关联）预测一致概率 $w_{\text{last}} < 0.5$ 且 `hits >= 3` 时跳过持久化写库，评估“软注意力去污” vs. “软去污 + 写库过滤”。
- **实现责任（明确标注服务器待实现项）：**
  - 本地未编写执行代码；服务器须在 Stage A 先读取 `/media/imc/data/mot-data/third_party/CAMELTrack`（commit `bc1c2610`）中 `cameltrack/architecture/temporal_encoder.py`、`cameltrack/architecture/camel.py` 与 `cameltrack/train/transforms.py` 源码，确认精确张量接口与掩码传播方式，然后编写并单测 `scripts/run_s10_camel.py` 与 `scripts/export_s10_report.py`：
    1. 实现 `D2_SKIP_ONLY`（复用 S09 `CleanHistoryCAMELTrack` 结构并移除 `else:` GT 补写分支，验证 1-to-1 与不注入特征）；
    2. 实现 `C0_SIMPLE_WRITE_GATE`（`tau_write ∈ {0.60, 0.65}`）；
    3. 实现可返回精确 `corrupted_mask` 与控制平均交换帧率的 `ContiguousTakeoverTransform`（`C2`）及配平交换率的 `MatchedIidSwapTransform`（`C1`），并在 Stage A 离线抽样 200 个 batch 实测两者总交换帧率误差 `<= 5%`；
    4. 实现 `C3` 一致性门控头（必须严格保证时间因果性：`t` 帧一致性仅可见过去/当前观测与历史上下文，不得依赖未来帧或 GT）并通过梯度/形状/因果掩码单元测试。
- **数据来源、精确 split 与历史使用：**
  - 完全沿用 S08/S09 的 DanceTrack `train` 内部互斥划分（[split 清单](../../manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml)，SHA256 `dd0b7fefd353532e04fb36206c7a1cf624cc661054b198653173f23e320926ff`）：
    - `motion_train32`（32 段，34,231 帧）；`motion_dev8`（8 段，7,565 帧）。
  - 直接核验并复用 S08 生成的状态文件（`states_train32.pklz`、`states_dev8_train_ref.pklz`、`states_dev8_gt_free.pklz`）与 S09 C1 训练好的三个单线索 TE checkpoint（`app_encoder` SHA `e5d25a8a...`、`kp_encoder` SHA `d33e84bb...`、`bbox_encoder` SHA `fc3b50d8...`）以及 `T_TWOSTAGE` joint ckpt（SHA `20be35af...`）。
  - 声明：`motion_train32`/`motion_dev8` 为项目复用开发划分，官方 `val` 与 `test` 本轮均不碰。
- **GT 边界与共享推理政策：**
  - Stage B `D2_SKIP_ONLY` 为特权离线诊断（`gt_role: privileged_skip_only_not_deployable`），单列报告。
  - Stage C 所有在线条件（`C0/C1/C2/C3`）推理严格使用 `states_dev8_gt_free.pklz`，禁用任何 GT 与未来帧；统一共享推理政策：`min_det_conf=0.4`、`min_init_det_conf=0.9`、`min_num_hits=0`、`max_wo_hits=150`、`max_track_gallery_size=50`、`sim_threshold=0.50`、`use_computed_sim_threshold=false`、无插值后处理。
- **主指标、辅助指标与配对单位：**
  - 主指标：`COMBINED_SEQ` 官方 TrackEval HOTA、AssA、IDF1（0–100 分全精度）；次要与代价指标：DetA、MOTA、IDSW、FP、FN、Frag；配对单位：8 段视频（报告相对 `T_TWOSTAGE_TH050` 与 `C0`/`C1` 的逐视频胜/平/负与中位差）。
  - 训练全部固定 `seed=13`、`max_epochs=20`、`last.ckpt` 唯一选择，不按 `dev8` 挑 best-epoch 或扫阈值。

## 执行矩阵

| stage / ID | 输入与范围 | 操作 / 命令入口及实现责任 | 必须回传 | 通过条件 | 通过后 / 失败后 |
|---|---|---|---|---|---|
| **A：资产锁、污染率配平与源码单测** | S08 状态、S09 C1/C2 ckpt、`CAMELTrack@bc1c2610` | 服务器核对复用资产 SHA256，编写 `scripts/run_s10_camel.py` 与 `tests/test_s10_camel.py`，验证 `C1` vs `C2` 交换率差异 `<= 5%`、`C3` 因果/梯度单测 | `source_audit.json`、`runtime_lock.json`、`resolved_configs/`、`commands.sh`、单测结果 | SHA256 全匹配、交换率配平通过、单元测试全部通过 | 通过进 B；超过 2 次修复仍失败则停止 |
| **B：`D2_SKIP_ONLY` 与 `C0` 零重训基准** | `T_TWOSTAGE` ckpt (`20be35af...`)、`dev8` GT-free 状态 | 运行 `D2_SKIP_ONLY`（仅拒写、不注入 GT）1 次；运行 `C0_TAU060` 与 `C0_TAU065`（零重训简单写入门控）2 次，TrackEval 评测 | 3 组 `dev8` 全精度指标（`D2_SKIP_ONLY` 单列标注特权角色）、写入跳过次数统计 | 3×8=24 条逐视频与 3 条 `COMBINED_SEQ` 导出完整 | **通过自动进 C**（无论 `D2_SKIP_ONLY` 高低，结果仅作后续归因对照） |
| **C：受控训练三联组（`C1/C2/C3`）** | `motion_train32` + S09 C1 三个 TE ckpt；`dev8` dev loss 与 GT-free 闭环 | 均设 `seed=13`、20 epochs、`last.ckpt` 显式重载，依次独立运行：<br>1. `C1_MATCHED_IID_SWAP`；<br>2. `C2_CONTIGUOUS_TAKEOVER`；<br>3. `C3_CONSISTENCY_GATED_TE`；<br>完成后分别以 `sim_threshold=0.50` 评测 `dev8` | 3 条 20-epoch 完整 `learning_curves.csv`（含 `C3` train/val BCE 一致性 loss）、有效参数/梯度/重载 SHA、3×8=24 条逐视频 + 3 条 `COMBINED_SEQ` 指标 | 0 非有限梯度、新进程显式重载 diff=`0.0`、全部 24+3 指标行完整 | 通过进 D；OOM 保留峰值/形状停止，不自动改 batch/精度 |
| **D：机制验证与有界案例分析** | `C3` ckpt 与 `T_TWOSTAGE` 对比样本 | 1. 运行 `C3_WITH_SOFT_WRITE_FILTER` 1 次（评估软注意力 + 一致性写库过滤）；<br>2. 在 `T_TWOSTAGE_TH050` 真实闭环产生的 `1,108` 异主错配与合法匹配观测上，离线只读测算 `C3` 一致性头区分真实污染写入的 **AUROC / F1 / 检出率**（严禁 GT 进入模型）；<br>3. 导出最多 16 例对照案例（含 `0037`/`0044` 改善/退化/持平） | `C3_WITH_SOFT_WRITE_FILTER` 指标、`diagnostics.json`（真实污染写入判别 AUROC/F1、跳过率、其他身份影响）、`cases.jsonl` & `evidence/` | 覆盖全部 `dev8` 8 段与 100% 抽样条件事件，零未来/GT 泄漏 | 通过进 E；若一致头无法读取真实闭环历史序列则明确报缺口 |
| **E：导出标准报告并停止** | A–D 全部产物 | 运行 `scripts/export_s10_report.py` 生成报告并校验文件哈希，更新 `reports/README.md`、`docs/progress_log.md`、`docs/a2a/LATEST.md` | `reports/feasibility/s10_camel_anticontamination_20261127/` 完整包 | `checks.json` 与 `artifacts.json` 校验通过 | **停在 E，交由研究端决定是否进 S11 确认或转向 `A'/C'`**；无授权不 commit/push |

- **预期规模计数：**
  - 训练运行：3 次独立 20-epoch 联合训练（`C1`, `C2`, `C3`，均 seed 13，从 S09 C1 TE ckpt 启动）。
  - 在线闭环评测：6 组（`C0_TAU060`, `C0_TAU065`, `C1`, `C2`, `C3`, `C3_WITH_SOFT_WRITE_FILTER`，共 48 条逐视频 + 6 条 `COMBINED_SEQ`）＋ S09 `T_TWOSTAGE_TH050` 只读参考引用。
  - 特权诊断：1 组（`D2_SKIP_ONLY`，8 条逐视频 + 1 条 `COMBINED_SEQ`，严格隔离单列）。
- **待解析项：**
  - `CAMELTrack` 上游 `TemporalEncoder` 的 token 组织方式与掩码参数名（Stage A 检查 `/media/imc/data/mot-data/third_party/CAMELTrack/cameltrack/architecture/temporal_encoder.py` 后确定并写入 `resolved_configs/`）。若 `TemporalEncoder` 内部对 `src_key_padding_mask` / `attn_mask` 有不可绕过的形状限制，按“在池化汇总层做可微一致性加权 + 传递至 `GAFFE`”实施并在 `source_audit.json` 记录原因。
- **预算与安全约束：**
  - 物理 GPU 1–7 单卡（首选物理 GPU 1，**GPU 0 永久禁用**）；
  - 总 GPU 预算 **`≤ 12` 卡时**（A ≤ 0.2h、B ≤ 0.3h、C ≤ 9.0h、D ≤ 2.5h）；墙钟 **≤ 36 小时**；主动工程 ≤ 2 个工作日；新增存储 ≤ 25 GiB（数据盘保留 ≥ 50 GiB 空闲）；
  - 每项工程阻断最多 2 次有针对性兼容修复（保留失败目录与新 `run_id`）。

## 回传和停止点

- **权威报告目录：** `reports/feasibility/s10_camel_anticontamination_20261127/`（大型产物放 `/media/imc/data/mot-data/runs/s10_camel_anticontamination_20261127/`）。
- **研究端需要仅凭 Git 回答的三个问题：**
  1. 剥离 GT 漏配补写后，纯拒写基准 `D2_SKIP_ONLY` 在 `dev8` 上达到多少 AssA / HOTA / FN？
  2. 相比 `T_TWOSTAGE_TH050`、`C0_SIMPLE_WRITE_GATE` 与等量噪声对照 `C1_MATCHED_IID_SWAP`，`C2`（连续接管训练）与 `C3`（多线索一致性门控聚合）是否各自带来一致且代价可接受的 `dev8` 闭环提升？
  3. `C3` 的一致性门控头在真实闭环（无合成噪声）的 `1,108` 次异主写入上，是否确实学到了超越匹配分数阈值的真实污染判别力（AUROC / F1）？
- **最后一个允许自动执行的步骤与本轮不运行项：**
  - 最后自动步骤为 **Stage E（报告导出与索引同步）**。
  - **本轮明确不运行**：多 seed（37/73）、官方 DanceTrack `val`/`test`、SportsMOT、`A'`（边际/Dustbin 损失）或 `C'`（成对运动/遮挡边偏置）模块、以及未授权的 `git commit` / `git push`。
