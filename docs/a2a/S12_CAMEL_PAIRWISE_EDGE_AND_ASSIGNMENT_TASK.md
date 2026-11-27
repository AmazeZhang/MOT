# S12：CAMELTrack 成对运动/方向遮挡边偏置（C'）与分配对齐解歧（A'）受控实验

> 状态：**设计已给出，服务器尚未执行，运行锁待服务器生成（`NOT_RUN`）**。
> 协议：[s12_camel_pairwise_edge_and_assignment_20261127.yaml](../../manifests/protocols/s12_camel_pairwise_edge_and_assignment_20261127.yaml)。
> 前置权威报告：[S09 报告](../../reports/diagnostics/s09_camel_calibration_and_attribution_20261126/README.md)、[S10 报告](../../reports/feasibility/s10_camel_anticontamination_20261127/README.md)、[S11 报告](../../reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md)。
> 规范依据：[研究端任务规范](../experiment_task_contract.md)、[任务书模板](../templates/server_experiment_task.md)、[服务器报告规范](../experiment_reporting.md)。

## 身份与决策

- **Task ID / 版本 / 日期：** `S12`（`s12_camel_pairwise_edge_and_assignment_20261127`） / v1 / 2026-11-27。
- **前置报告与停止范围：** S11（`s11_camel_onset_and_conflict_gate_20261127`）`COMPLETED`，确认 **情况 2（写库侧条件拒写 `G1–G4` 全部劣于 `T_TWOSTAGE_TH050` 与 `T2_TH040`）**。既有 `P1/R1/R2/E0–E5/S03/S04/S06`、S10 `C2/C3` 合成接管训练及 S11 `G1–G4` 纯写库端门控停止决定维持不变，不再扩展写库侧过滤变体。
- **当前执行状态与自动推进依据：** `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）。用户交付服务器后，Stage A–E 在前置门通过且预算内自动顺序推进；到达 Stage E 结束或遇到未解决阻断/预算上限即停止回传，不自动启动多 seed（37/73）或官方 `val`/`test`。
- **唯一主问题：**
  在 `motion_train32` → `motion_dev8`（共享 S09 C1 TE 初始化、20 epochs、`seed=13`、主标定基线为 S11 **`T2_TH040`** `57.572` HOTA / `40.845` AssA / `60.117` IDF1 / `832` IDSW / `4,642` FN）上，针对 S11 Stage C 揭示的 **`149` 例真实 onset 切换**（`74.5%` 为 `gap==1`、`89.9%` 伴 `IoU>=0.30`、`63.1%` 真目标框缺检却以中位 `0.658` 抓错邻近框、`74.5%` 双向 `margin < 0.10` 且 `-margin` AUROC `0.9432`、`22.8%` 为 `2-cycle` 互换）：
  1. **零重训分配层解歧（`A'_infer`：`B1` Dual-Softmax 竞争校准 vs `B2` 边际/`2-cycle` 运动学解歧延迟）** 能否在不改权重下直接超越 `T2_TH040`？
  2. **训练端结构改进（`C'`：`M1` 成对运动学外推残差边偏置、`M2` 叠加 DOA 式方向遮挡与前后序几何边偏置）** 相比共享初始化与参数的 **单节点容量与属性对照（`M0_NODE_CAP_CTRL`）**，能否显著提升 `motion_dev8` 闭环 AssA / HOTA / IDF1？
  3. **分配对齐训练损失（`A'_train`：`M3` 在 `M2` 上加入双向竞争边际与未匹配防误抓校准损失）** 能否进一步减少真目标缺检时的误抓与低边际反转，并结合 Stage D 回放验证 `149` 例 onset 与 `31,791` 帧后切换污染尾段的实际消减比例？
- **任务类别、允许主张与不允许主张：**
  - 类别：零重训分配对比（Stage B）＋严格受控开发集单因子消融训练（Stage C/D，单 seed `13`）。
  - 允许主张：同数据（`train32`）、同初始化（S09 C1 TE）、同参数/容量对照（`M0`）下，成对运动/方向遮挡边偏置与分配对齐机制在 `motion_dev8` 上的受控机制因果增量与 onset 消减证据。
  - 不允许主张：严禁将单 seed `motion_dev8` 结论外推为未见集泛化或 SOTA；严禁与 `P_OFFICIAL`（含 `train+val` 预训练）直接相减宣称优劣；严禁把 Sinkhorn/Dustbin/Margin Loss 或方向遮挡本身宣称为“领域首创”（文献查重已记录 SLAck / HDST-GNN / MM-Tracker / DOA-SORT / VLA-ReID / C-TWiX 先例）；若 `M0` 达到或超过 `M2`，不得主张成对关系边结构有效；若 `B1/B2` 解释主要增益，必须诚实收缩训练侧贡献。
- **正/负/不确定结果如何决定 S13：**
  - **情况 A（强正向：`FREEZE_FOR_S13_CONFIRMATION`）**：`M2` 或 `M3`（含或不含 `A'_infer`）在 `motion_dev8 @ 0.40` 上**同时稳定胜过** `T2_TH040`（`57.572 / 40.845`）与 `M0_NODE_CAP_CTRL`（逐视频胜/平/负 `>= 5/8`，HOTA/AssA/IDF1 方向一致且无严重 `DetA/FN` 恶化），且 Stage D 解剖证明 onset 与后切换污染明显下降 → S13 冻结获胜结构，执行 multi-seed（`13/37/73`）并在未调参的官方 DanceTrack `val`（诚实披露 G1 基线历史）、`test` 及 SportsMOT 上完成确认评测。
  - **情况 B（部分/收缩正向：`CONTRACT_TO_MINIMAL_EFFECTIVE_MECHANISM`）**：若 `M1`（仅成对运动残差）与 `M2` 相当则舍弃方向遮挡分支；若 `B1/B2`（纯推理 `A'_infer`）已吸收绝大部分增益而训练端无额外优势则冻结轻量化推理分配改法；若 `M0` 胜过或持平 `M1/M2` 则否定成对边结构假设，转而评估纯节点特征/容量贡献。
  - **情况 C（整体负向/不确定：`STOP_CAMEL_EDGE_AND_REASSESS`）**：`B1/B2` 与 `M1–M3` 在 `dev8` 上均未相对 `T2_TH040` 产生明确正收益 → 停止在当前冻结检测/状态上做 GAFFE 边偏置或损失微调，审阅残余错误是否源于上游检测缺失（`94/149` 缺检）或需切换至端到端联合检测-关联范式。

## 科学设计

- **可否定假设：**
  - `H_A_infer`：利用双向竞争校准（`B1` Dustbin 增广 Dual-Softmax）或仅对极端低边际/空间冲突匹配做 `2-cycle` 运动学纠正与单帧延迟暂缓（`B2`），能在不饿死正常拥挤更新（克服 S11 `G1–G4` 的 `42%` 误拒写）的前提下降低 onset 错配并超越 `T2_TH040`（`57.572`）。
  - `H_C_kin`：在 GAFFE 的全局 token 注意力与输出相似度中注入**显式成对运动学外推残差边偏置**（`M1`），能克服单目标 `bbox_encoder`（`0.46%` 参数）的几何盲区，且效果优于等参数/同单节点信息的 `M0_NODE_CAP_CTRL`。
  - `H_C_doa`：在成对运动残差基础上进一步叠加**DOA 式方向遮挡（L/R/T/B）与地面底边前后序关系几何**（`M2`），能帮助 GAFFE 在 `89.9%` 空间重叠冲突（尤其 `63.1%` 背后目标缺检时前排遮挡物误抓）中动态降权受损外观线索，显著优于仅运动学 `M1`。
  - `H_A_train`：在 `M2` 基础上加入**分配对齐的双向竞争边际与未匹配防误抓校准损失**（`M3`），能直接拉大强竞争邻域的正负边际并压低无真目标轨的错抓分数，进一步减少 `74.5%` 低边际切换与 `22.8%` `2-cycle` 互换。

- **最强相关对照与唯一变化因素（全部预写固定，禁止看结果后调参）：**
  - **既有参考基线（直接引用，不重跑）**：S11 `T2_TH040`（`0.40`, `57.572` HOTA / `40.845` AssA / `60.117` IDF1 / `832` IDSW / `4,642` FN）、S09 `T_TWOSTAGE_TH050`（`0.50`, `54.443` / `36.704` / `53.629` / `786` / `5,615`）、S10 `D2_SKIP_ONLY`（`63.781` / `49.940`）。
  - **Stage B 零重训 `A'_infer` 两组（复用 `T_TWOSTAGE` ckpt，`sim_threshold=0.40`）：**
    1. **`B1_A_DUAL_SOFTMAX_TH040`**：在 `td_sim_matrix` $S \in \mathbb{R}^{T \times D}$ 上执行固定温度 $\tau_{\text{ds}} = 0.10$、固定 Dustbin 锚点 $s_0 = 0.40$ 的双向 Softmax 竞争校准：
       - 行增广 $\tilde{S}^{\text{row}}_i = [S_{i,1}, \dots, S_{i,D}, s_0]$，取 $P^{\text{row}}_{i,j} = \text{softmax}(\tilde{S}^{\text{row}}_i / \tau_{\text{ds}})_j$，Dustbin 概率为 $P^{\text{row}}_{i,\varnothing}$；
       - 列增广 $\tilde{S}^{\text{col}}_j = [S_{1,j}, \dots, S_{T,j}, s_0]$，取 $P^{\text{col}}_{i,j} = \text{softmax}(\tilde{S}^{\text{col}}_j / \tau_{\text{ds}})_i$，Dustbin 概率为 $P^{\text{col}}_{\varnothing, j}$；
       - 仅允许同时满足 $P^{\text{row}}_{i,j} \ge P^{\text{row}}_{i,\varnothing}$ 且 $P^{\text{col}}_{i,j} \ge P^{\text{col}}_{\varnothing, j}$ 的候选对参与匈牙利匹配（对非合法对置大惩罚），以校准相似度 $S^{\text{cal}}_{i,j} = \sqrt{P^{\text{row}}_{i,j} P^{\text{col}}_{i,j}}$ 作为分配收益（**无额外可调阈值**）。
    2. **`B2_A_MARGIN_2CYCLE_DEFER_TH040`**：先按原 `S >= 0.40` 跑匈牙利得到初步匹配，然后执行两步严格因果分配防卫：
       - **步骤 1（`2-Cycle` 因果运动学纠偏）**：对同帧任意两对成熟轨迹（`hits >= 3`）$(t_1 \to d_1), (t_2 \to d_2)$，若满足空间重叠冲突（$\text{IoU}(d_1, d_2) \ge 0.30$ 或 $\text{IoU}(b^{\text{last}}_1, b^{\text{last}}_2) \ge 0.30$）且至少一方双向 `assignment_margin < 0.10`，用各自因果匀速外推框 $\hat{b}_1, \hat{b}_2$ 计算交换前后 $\text{IoU}(\hat{b}, d)$ 之和；仅当互换 $(t_1 \to d_2, t_2 \to d_1)$ 使两对均满足 `S >= 0.30` 且 $\text{IoU}(\hat{b}_1, d_2) + \text{IoU}(\hat{b}_2, d_1) > \text{IoU}(\hat{b}_1, d_1) + \text{IoU}(\hat{b}_2, d_2) + 0.10$ 时执行去互换。
       - **步骤 2（极端歧义冲突单帧暂缓 `Coast`）**：对经过步骤 1 后的成熟活跃匹配（`hits >= 3` 且 `time_wo_hits == 0`），仅当同时满足 `assignment_margin < 0.05`（比 S11 `G2` 更窄，瞄准强竞争极窄边际）、`max_conflict_iou >= 0.30` 且 `sim_val < 0.65` 时，**取消本次配对本身**（令轨迹按正常 `unmatched` 保持 `time_wo_hits=1` 滑行而不切主、不写库，检测进入 `unmatched_dets` 但不低于 `min_init_det_conf=0.9` 才建新轨）。
  - **Stage C 受控四联组（共享 S09 C1 TE ckpt 初始化、`motion_train32`、`seed=13`、20 epochs、`last.ckpt`）：**
    1. **`M0_NODE_CAP_CTRL`（容量与单节点特征公平对照）**：保持 `GAFFE` 无成对 $(i,j)$ 边偏置；将与 `M2` 同源的**单节点**因果运动特征（轨迹匀速外推位置/速度/gap）和**单目标**方向遮挡密度特征（检测被同帧其他检测左/右/上/下遮挡率、`max_other_iou`、前排程度）分别通过零初始化输出层的残差 MLP 注入 `GAFFE` 输入的 tracklet token 与 detection token，且**保证 `M0` 新增可训练参数量 $\ge$ `M2` 新增可训练参数量**。
    2. **`M1_C_KIN_EDGE`（纯成对运动学边偏置）**：保持 `M0` 的 C1 TE 初始化与总训练设置，取消单节点额外头，仅在 `GAFFE` 注入**成对运动学外推残差边特征 $e^{\text{kin}}_{ij}$**（严格因果、无 GT）：
       - 由 `tracks.feats["bbox_ltwh"]` 与 `tracks.feats["age"]`（仅 `age >= 1` 有效历史框，过滤 `NaN`/零面积/dropout 槽）用近端加权最小二乘匀速外推至当前帧 `age = 0` 得 $\hat{b}_i = (\hat{c}_x, \hat{c}_y, \hat{w}, \hat{h})$（有效历史 `< 2` 时退化为最近观测框 $\vec{v}_i=0$，长 `gap` 采用指数衰减速度阻尼 $\alpha^{\Delta t-1}, \alpha=0.9$，数值稳健裁剪）；
       - 构造 $d_{\text{kin}} = 13$ 维尺度不变成对特征：$(\Delta \hat{c}_x/\hat{w}, \Delta \hat{c}_y/\hat{h}, \Delta c^{\text{last}}_x/w^{\text{last}}, \Delta c^{\text{last}}_y/h^{\text{last}}, \log(w_j/\hat{w}), \log(h_j/\hat{h}), \text{IoU}(\hat{b}_i, b_j), \text{GIoU}(\hat{b}_i, b_j), \text{IoU}(b^{\text{last}}_i, b_j), \cos(\vec{v}_i, \vec{c}_j-\vec{c}^{\text{last}}_i), \|\vec{v}_i\|/\hat{h}, \log(1+\Delta t_i), \mathbb{I}[\text{valid\_vel}])$；
       - 通过轻量 MLP 映射为两部分（均在输出层**严格零初始化**，使 step 0 等价于基准 CAMEL）：① 注入 `GAFFE` 每层自注意力 `tracklet<->detection` 块的逐头 logit 偏置 $B^{(h)}_{ij}$；② 输出端门控成对相似度残差 $\Delta S_{ij} = \gamma \tanh(\psi(e^{\text{kin}}_{ij}, z_i, z_j))$（$\gamma$ 初始为 `0` 或输出层零初始化）。
    3. **`M2_C_KIN_DOA_EDGE`（完整 `C'`：成对运动学 + DOA 式方向遮挡与前后序边偏置）**：在 `M1` 完全相同设置上，向成对边向量 $E_{ij} = [e^{\text{kin}}_{ij}, e^{\text{occ}}_{ij}]$ 拼接 $d_{\text{occ}} = 13$ 维**方向遮挡与前后序关系几何**（受 DOA-SORT 启发，但作用于多线索学习注意力）：
       - 地面底边前后序：$\Delta y_{2} = (y_{2,j} - \hat{y}_{2,i}) / \hat{h}_i$（正值表示检测 $j$ 在画面更下方/更靠前）、同帧其他检测 $k \neq j$ 在 $b_j$ 靠前（$y_{2,k} > y_{2,j}$）时对 $b_j$ 产生的**左/右/上/下四向边界遮挡覆盖率**（4 维）及 `det_max_other_iou(j)`（1 维）；
       - 轨迹外推框 $\hat{b}_i$ 相对检测 $j$ 的四向重叠侵入比例（左/右/上/下各 1 维，刻画“究竟是 $b_j$ 从一侧挡住了 $\hat{b}_i$ 还是中心对齐”）、以及 $\hat{b}_i$ 被所有更靠前检测 $k$（$y_{2,k} > \hat{y}_{2,i}$）的总体最大遮挡覆盖率与 `trk_max_other_iou(i)`（2 维，提示当前轨迹正处于背后重度遮挡区，应警惕“真目标缺检时误抓前排框”）。
       - 同时在 `GAFFE` 的 `det-det` 与 `trk-trk` 自注意力块注入简单的对称空间重叠偏置（`IoU` 与相对底边差），使场景组内竞争信息可沿空间邻接传播。
    4. **`M3_C_PRIME_PLUS_A_LOSS`（`C' + A'_train`）**：与 `M2` **架构、初始化、训练数据与步数完全一致**，唯一增量是在原 `CAMEL.compute_loss`（记为 $\mathcal{L}_{\text{base}}$）上增加固定系数的分配对齐损失：
       $$\mathcal{L} = \mathcal{L}_{\text{base}} + \lambda_{\text{margin}} \mathcal{L}_{\text{bidir\_margin}} + \lambda_{\text{rej}} \mathcal{L}_{\text{unmatched\_rej}}$$
       其中（默认超参固定 $\lambda_{\text{margin}} = 0.5$、$\lambda_{\text{rej}} = 0.25$、边际 $m_0 = 0.15$、拒识锚定 $\tau_{\text{neg}} = 0.35$、尺度 $\beta = 10.0$，不做 `dev8` 搜索）：
       - $\mathcal{L}_{\text{bidir\_margin}}$：对每个有效正匹配对 $(i, j^+)$，同时取行内最强负竞争者 $s^{\text{row\_neg}}_i = \max_{j \neq j^+} S_{ij}$ 与列内最强负竞争者 $s^{\text{col\_neg}}_{j^+} = \max_{i' \neq i} S_{i'j^+}$，施加平滑竞争边际损失 $\frac{1}{2\beta}[\text{softplus}(\beta(m_0 - (S_{ij^+} - s^{\text{row\_neg}}_i))) + \text{softplus}(\beta(m_0 - (S_{ij^+} - s^{\text{col\_neg}}_{j^+}) ))]$；
       - $\mathcal{L}_{\text{unmatched\_rej}}$：对所有有效但当前帧**无匹配 GT 检测**的轨迹 $i$（及无匹配 GT 轨迹的检测 $j$），提取其各自最高误抓负样本得分 $s^{\text{max\_false}}$（特别覆盖空间重叠邻近对），施加 $\frac{1}{\beta}\text{softplus}(\beta(s^{\text{max\_false}} - \tau_{\text{neg}}))$，直接压低 `94/149` 类“目标缺检时误抓邻近框”的得分并使其落到工作阈值 `0.40` 以下。

- **实现责任（明确标注服务器待实现项与 Stage A 待核对项）：**
  - 本地仅交付本任务书与协议；服务器须在 Stage A 先读取 `/media/imc/data/mot-data/third_party/CAMELTrack`（commit `bc1c2610`）源码中的 `cameltrack/architecture/camel.py`、`cameltrack/architecture/gaffe.py`、`cameltrack/train/losses.py`、`cameltrack/train/dataset.py`、`cameltrack/train/transforms.py`，核清并写入 `source_audit.json` 后再编写 `scripts/run_s12_camel.py`、`scripts/export_s12_report.py` 与 `tests/test_s12_camel.py`：
    1. 核清 `GAFFE` 内部注意力调用方式（是否用 `nn.TransformerEncoder` 或自定义 `MultiheadAttention`；如何注入 `attn_mask` / 逐层 `B_edge`，且保证 padding 掩码与空行零 NaN）；
    2. 核清 `CAMEL.compute_loss`（`SimLoss` 确切公式、温度、以及它对无匹配轨迹/检测的处理）；
    3. 核清 `DropoutFeatures` 对 `bbox_ltwh` 的置空/置 `NaN` 语义，并实测 200 个 `train32` batch 中“无匹配 GT 轨迹”与“空间 `IoU>=0.30` 重叠对”的实际占比（若天然无匹配轨迹占比 $< 3\%$，记录该统计并在 `M3` 中显式报告 $\mathcal{L}_{\text{unmatched\_rej}}$ 的有效触发样本数，**不擅自改变 `M3` 训练数据增强配比以免破坏与 `M2` 的同样本暴露公平性**）；
    4. 实现并单测共享的张量级因果外推函数 `compute_causal_extrapolation` 与方向遮挡几何函数 `compute_pairwise_kin_doa_features`，确保训练前向与在线推理调用**同一份函数**。

- **数据来源、精确 split 与 GT 边界：**
  - 全部沿用 [split 清单](../../manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml)（SHA256 `dd0b7fefd353532e04fb36206c7a1cf624cc661054b198653173f23e320926ff`）中的 `motion_train32`（32 段，34,231 帧）与 `motion_dev8`（8 段，7,565 帧），官方 `val`/`test` 本轮均不碰。
  - 状态文件与缓存：训练复用 `states_train32.pklz` / `states_dev8_train_ref.pklz`（及共享 DM 缓存）；所有闭环推理评测严格使用 `states_dev8_gt_free.pklz`，零 GT 访问、零未来帧访问。仅 Stage D 的 Onset 解剖回放与最终 TrackEval 允许离线只读加载 GT，且必须验证回放 MOT 与在线评测 MOT 逐字节一致。
  - 共享推理政策：`min_det_conf=0.4`、`min_init_det_conf=0.9`、`min_num_hits=0`、`max_wo_hits=150`、`max_track_gallery_size=50`、`use_computed_sim_threshold=false`、无插值后处理；主阈值 `0.40`，辅阈值 `0.50`（仅 Stage C 模型做双点无调参报告）。

- **主指标、辅助指标与配对单位：**
  - 主指标：`COMBINED_SEQ` 官方 TrackEval HOTA、AssA、IDF1（0–100 全精度）；次要与代价指标：DetA、MOTA、IDSW、FP、FN、Frag、以及有效参数增量与单帧推理耗时（ms/frame）。
  - 配对单位：8 段序列（报告相对 `T2_TH040` 与 `M0_NODE_CAP_CTRL` 的逐视频胜/平/负及中位差）。
  - 训练选模规则：固定 `seed=13`、`max_epochs=20`、`last.ckpt` 唯一最终评测（过程记录 `val/loss` 与 `val/sim_auroc` 曲线，不按 `dev8` 挑 best-epoch）。

## 执行矩阵

| stage / ID | 输入与范围 | 操作 / 命令入口及实现责任 | 必须回传 | 通过条件 | 通过后 / 失败后 |
|---|---|---|---|---|---|
| **A：源码核验、实现与单元测试** | S08/S09 资产、`CAMELTrack@bc1c2610` 源码 | 核验全部复用 SHA256；审计 `gaffe.py`/`losses.py`/`dataset.py` 细节及 batch 无匹配比例；编写并单测 `scripts/run_s12_camel.py` 与 `tests/test_s12_camel.py` | `source_audit.json`、`runtime_lock.json`、`resolved_configs/`、`commands.sh`、单测输出（含参数量与 step-0 等价验证） | SHA256 匹配、`M0_extra_params >= M2_extra_params`、step-0 零初始化等价、全部因果/梯度/无 NaN 单测 PASS | **通过自动进 B**；修复超 2 次仍失败则停止 |
| **B：零重训 `A'_infer` 分配解歧对照** | S09 `T_TWOSTAGE` ckpt (`20be35af...`) + `states_dev8_gt_free.pklz` | 固定 `sim_threshold=0.40` 独立运行 `B1_A_DUAL_SOFTMAX_TH040` 与 `B2_A_MARGIN_2CYCLE_DEFER_TH040` 闭环跟踪与 TrackEval | 2×8=16 条逐视频与 2 条 `COMBINED_SEQ` 指标 + 介入统计（纠偏次数/暂缓次数） | 零 GT 访问，16+2 行导出完整且导出误差 `0.0` | **通过自动进 C**（即使未超过 `T2_TH040` 也自动进 C 作为必须基线对照） |
| **C：受控四联组训练与 `dev8` 双点评测** | `motion_train32` + S09 C1 三个 TE ckpt；`states_dev8_gt_free.pklz` | 同 `seed=13`、20 epochs、`last.ckpt` 新进程显式重载，依次独立训练：<br>1. `M0_NODE_CAP_CTRL`；<br>2. `M1_C_KIN_EDGE`；<br>3. `M2_C_KIN_DOA_EDGE`；<br>4. `M3_C_PRIME_PLUS_A_LOSS`；<br>每组分别在 `0.40`（主）与 `0.50`（参考）评测 `dev8` | 4 条 20-epoch `learning_curves.csv`（含各分项 loss/梯度范数）、参数/更新 delta、重载 diff=`0.0`、4×2×8=64 条逐视频与 8 条 `COMBINED_SEQ` 指标 | 0 非有限梯度、重载 diff=`0.0`、64+8 行全精度完整且导出误差 `0.0` | **通过自动进 D**；OOM 保留峰值/形状停止，不自动改 batch/精度 |
| **D：协同检验、Onset 因果解剖与案例证据** | Stage C 获胜模型（`@0.40` HOTA/AssA 最优组）+ `dev8` | 1. 将 `B1` 与 `B2` 零重训分别叠加到 Stage C 最优模型上评测 `dev8`（2 组）；<br>2. 对最优模型（在 `@0.50` 与 `@0.40`）运行与 S11 Stage C 同口径的 **Onset & 污染尾段回放解剖**，直接对比 S11 `149` 例 onset（`94` 缺检误抓 / `55` 在场反转平局 / `34` `2-cycle`）与 `31,791` 污染尾段消减率；<br>3. 导出最多 16 例对照案例（含改善、持平与损伤） | 2×8+2 行协同评测指标、`diagnostics.json`（onset 消减对比表 + 耗时延迟）、`cases.jsonl` & `evidence/` | 回放 MOT 与在线 MOT 逐字节一致、零 GT 泄漏进推理、覆盖全部 `dev8` 8 段 | **通过自动进 E**；若某个推理协同失败则记录失败 attempt 并继续完成其余项与 E |
| **E：导出标准报告与同步交接** | A–D 全部产物 | 运行 `scripts/export_s12_report.py` 生成报告、校验哈希，更新 `reports/README.md`、`docs/progress_log.md`、`docs/a2a/LATEST.md` | `reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/` 完整包 | `checks.json` 与 `artifacts.json` 全校验通过 | **停在 E，交回研究端裁决 S13**；无明确授权不 commit/push |

- **预期规模计数：**
  - 训练运行：**`4` 次**独立 20-epoch Stage 2 联合训练（`M0/M1/M2/M3`，均 `seed=13`、共享 S09 C1 TE ckpt 初始化）。
  - 在线闭环评测：**`12` 组**（Stage B `2` 组 + Stage C `4 × 2 = 8` 组 + Stage D 协同 `2` 组；共 `96` 条逐视频 + `12` 条 `COMBINED_SEQ`）＋ S11 `T2_TH040` / S09 `T_TWOSTAGE_TH050` / S10 `D2_SKIP_ONLY` 只读参考引用。
  - 离线机制回放与解剖：**`2` 组**（Stage D 对最优 Stage C 模型在 `@0.50` 与 `@0.40` 的 Onset 解剖回放）。
- **待解析项（Stage A 解决并写入 `source_audit.json`）：**
  - `CAMELTrack@bc1c2610` 中 `GAFFE` 的内部自注意力实现方式（`nn.TransformerEncoder` vs 手写 attention）：若 `nn.TransformerEncoder` 仅接受 unified `mask`（形状 `[B*n_heads, L, L]`），则将成对边偏置构造为分块矩阵（`trk-det` / `det-trk` 填 $B_{\text{edge}}$，`trk-trk` / `det-det` 填对称空间邻接偏置，padding 槽位叠 `-1e9` 且对角保持 `0.0` 防 NaN），与 S10 `weighted_temporal_encode` 已验证的方式一致。
  - `CAMEL.compute_loss`（`SimLoss`）源码公式与默认温度 $\tau_{\text{sim}}$：在 `source_audit.json` 如实记录；若 `SimLoss` 内部已自带某种边际或温度，保持 `M0/M1/M2` 完全沿用原 `SimLoss`，仅 `M3` 在其之上叠加 $\mathcal{L}_{\text{bidir\_margin}} + \mathcal{L}_{\text{unmatched\_rej}}$。
  - `BBoxLinProj` / `DropoutFeatures` 在 `tracks.feats["bbox_ltwh"]` 遭遇 dropout 时的占位符编码（`NaN` vs `0`）：Stage A 读取源码确认后写入 `compute_causal_extrapolation` 掩码条件并单测覆盖。
- **预算与安全约束：**
  - 物理 GPU `1–7` 单卡（首选物理 GPU `1`，**GPU 0 永久禁用且严禁触碰**）；
  - 总 GPU 预算 **`≤ 9.0` 卡时**（A ≤ 0.3h、B ≤ 0.5h、C ≤ 6.5h、D ≤ 1.5h、E ≤ 0.2h）；墙钟 **≤ 24 小时**；主动工程 **≤ 2 个工作日**；新增数据盘存储 **`≤ 15 GiB`**（保留 ≥ 50 GiB 空闲）；
  - 每项工程阻断最多 **2 次**有针对性兼容修复，保留失败 `run_id` 目录。

## 回传和停止点

- **权威报告目录：** `reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/`（大型产物放 `/media/imc/data/mot-data/runs/s12_camel_pairwise_edge_and_assignment_20261127/`）。
- **研究端需要仅凭 Git 回答的四个核心问题：**
  1. 零重训分配层改法 `B1`（Dual-Softmax）与 `B2`（`2-cycle` 纠偏 + 极端歧义单帧暂缓）相对 `T2_TH040`（`57.572`）究竟能回收多少 HOTA/AssA/IDSW？是否证明了纯推理分配已足够？
  2. `M1`（仅成对运动残差）与 `M2`（运动残差 + DOA 方向遮挡与前后序偏置）相比同初始化同容量的 `M0_NODE_CAP_CTRL` 是否取得实质、逐视频一致的 AssA / HOTA / IDF1 提升？DOA 方向遮挡几何相对纯运动残差有多大净增量？
  3. `M3`（`C' + A'_train` 双向竞争边际 + 未匹配防误抓损失）相比 `M2` 是否进一步提升了闭环关联与 `dev8` 胜率？叠加 `A'_infer`（`B1/B2`）是否有协同增益？
  4. 最优 `C'/A'` 模型在真实闭环上是否实质消减了 S11 暴露的 `149` 例 onset（尤其是 `94` 例真目标缺检误抓与 `34` 例 `2-cycle` 互换）和 `31,791` 帧后切换污染尾段，从而接近甚至吃下 `D2_SKIP_ONLY`（`63.781` HOTA / `49.940` AssA）的显著空间？
- **最后一个允许自动执行的步骤与本轮不运行项：**
  - 最后自动步骤为 **Stage E（报告导出与索引同步）**。
  - **本轮明确不运行**：multi-seed（`37/73`）、官方 DanceTrack `val`/`test`、SportsMOT、S10 已否定的合成接管训练或写库过滤变体、`anchor_reid_dist` 新模型头、以及未授权的 `git commit` / `git push`。
