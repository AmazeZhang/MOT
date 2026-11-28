# S13：CAMELTrack 标尺校准、缺检拒识与早锚双时间尺度抗污染记忆（K-AnchorCAMEL）

> 状态：**设计已给出，服务器尚未执行，运行锁待服务器生成（`NOT_RUN`）**。
> 协议：[s13_camel_calibration_and_dual_memory_20261128.yaml](../../manifests/protocols/s13_camel_calibration_and_dual_memory_20261128.yaml)。
> 前置权威报告：[S10 报告](../../reports/feasibility/s10_camel_anticontamination_20261127/README.md)、[S11 报告](../../reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md)、[S12 报告](../../reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/README.md)。
> 规范依据：[AGENTS.md](../../AGENTS.md)、[研究端任务规范](../experiment_task_contract.md)、[任务书模板](../templates/server_experiment_task.md)、[服务器报告规范](../experiment_reporting.md)。

## 身份与决策

- **Task ID / 版本 / 日期：** `S13`（`s13_camel_calibration_and_dual_memory_20261128`） / v1 / 2026-11-28。
- **硕士论文总目标与防陷入定位：** 本轮严格服从 [AGENTS.md](../../AGENTS.md) 新增的硕士论文科研与防陷入规则——**不再针对小样本做无休止的微调诊断**，而是用“**1 次低成本 S12 矫正确认（Stage B）＋ 1 轮收敛型核心方法消融（Stage C/D）＋ 硬晋级/止损门（Stage E）**”直接回答：把 S10–S12 揭示的三大律（成对运动降 onset、缺检遇遮挡需拒识、早期无冲突锚点打破长尾污染）合为一体后，能否产生**足够支撑升级至多 seed（`13/37/73`）与完整 DanceTrack `val`/`test`（及后续 SportsMOT）**的实质性闭环跃升。
- **前置报告与永久停止分支：** S12（`s12_camel_pairwise_edge_and_assignment_20261127`）`COMPLETED`。维持对 `P1/R1/R2/E0–E5/S03/S04/S06`、S10 `C2/C3` 合成整帧替换、S11 `G1–G4` 纯写库端静止门控、以及 S12 已否定的未校准 `B1/B2 @ 0.40`、`M2` DOA 冗余自注意偏置、`M3` 无区分 `tau_neg=0.35` 强推损失的停止判定，本轮均不再重复探索。
- **当前执行状态与自动推进依据：** `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）。用户交付服务器后，Stage A–E 在前置门通过且预算内自动顺序推进；到达 Stage E 结束即停止回传，不自动开启 multi-seed（`37/73`）或官方 `val`/`test`。
- **唯一主问题：**
  1. **S12 矫正验证（Stage B 零重训）：** S12 `M1_C_KIN_EDGE` 因无约束 `delta_s` 将 `val/computed_sim_threshold` 推升至 `0.6869`（导致 `@0.40/@0.50` 几乎全盲、7/8 视频零差异、且 `M1+B1` 在 `dustbin=0.40` 下 `0` 次拒识）。在对齐其真实标尺的校准阈值（`0.55–0.68`）及同步校准 `dustbin` 的 `M1+B1` 下，`M1` 能否超越 `T2_TH040`（`57.572`）并消除 `dancetrack0024`（`-8.22`）/ `0053`（`-8.99`）的早期切错崩塌？
  2. **论文核心方法闭环检验（Stage C/D 收敛三联组）：** 在 `M1` 因果运动边偏置基础上，逐级叠加：
     - **`N1`（标尺零中心锚定 `C'`）**：消除 `delta_s` 标尺上移并按 `val` 零泄漏对齐决策阈值；
     - **`N2`（遮挡删帧抗误抓监督）**：仅对训练 batch 中“存在 `IoU>=0.25` 重叠干扰框且刻意丢弃其真值匹配框”的轨迹，专门教模型压低邻近干扰框分数（直击 S11/S12 中 `60%–63%` 真目标缺检时误抓邻框的主病因）；
     - **`N3`（早锚双时间尺度自纠偏记忆 `K-AnchorCAMEL`）**：在不饿死近期更新（克服 `G4`）、且保留干净参照打破自洽陷阱（克服 `C2/C3`）的前提下，为轨迹维护不被 FIFO 淘汰的早期低冲突锚点槽（`K=4`）并引入近期槽可控污染注入训练，能否同时显著削减 **`onset_cross`（`< 95`）与 `post_switch_cross` 长尾（`<< 40,570`）**，从而突破 `59.1+` HOTA 并收割 `D2_SKIP_ONLY`（`63.781`）抗污染空间？
- **任务类别、允许主张与不允许主张：**
  - 类别：零重训标尺矫正评估（Stage B）＋ 收敛型三臂受控机理消融与晋级判定（Stage C/D，单 seed `13`）。
  - 允许主张：在 `motion_train32 -> motion_dev8`（同 S09 C1 TE 初始化、20 epochs）严格对齐标尺下的机制消融证据；以及决定是否冻结该架构晋级 S14（多 seed + 完整 DanceTrack `val`/`test`）的定量门槛裁决。
  - 不允许主张：单 seed `dev8` 结果不得宣称为未见集 SOTA；不得与 `P_OFFICIAL`（含 `train+val`）直接作同数据差值比较；若 `N1/N2/N3` 未能清晰超过 Stage B 校准后的 `M1` 与 `T2_TH040`，不得强行归因为架构成功。
- **正 / 负 / 边界结果导致的明确 S14 路由（防陷入硬门槛）：**
  - **情况 1（`ESCALATE_TO_THESIS_CONFIRMATION`）**：Stage C 胜出组（或校准后方案）在 `motion_dev8` 上取得 **`HOTA >= 59.1`（`>= +1.5` vs `T2_TH040 57.572`）且 `AssA >= 42.85`（`>= +2.0` vs `40.845`）**，逐视频胜/平/负 `>= 5/8`，且 Stage D 回放证明 `post_switch_cross` 较 S11（`31,791`）与 S12（`40,570`）显著下降 → **停止在 `dev8` 上继续小修小补**，S14 直接冻结架构执行 multi-seed（`13/37/73`）＋ **完整 DanceTrack `train -> val`/`test`** ＋ **SportsMOT** 论文主实验验证！
  - **情况 2（`BORDERLINE_PIVOT_DECISION`）**：最优结果位于 `(57.572, 59.1)` 之间 → 停在当前局部关联加特征路线，不再开 S14 同类超参微调；由研究端依据 Stage D 案例决定：要么只补 1 轮 **Route B（严格在线因果交叉后轨迹级自纠偏/重锚定）**，要么直接切换 **Route C（更高天花板范式，如 `MOTIP` 端到端 ID 预测或实例可见性/遮挡序模型）**。
  - **情况 3（`STOP_CAMEL_LOCAL_PATCH_AND_PIVOT`）**：校准 Stage B 与 `N1–N3` 均无法超过 `T2_TH040`（`<= 57.572`）→ **彻底终止当前 CAMEL 局部边偏置与写端/训练抗污染分支**，记入 `innovation_registry.md`，直接转向 Route B/C 新赛道。

## 科学设计

- **可否定假设：**
  - `H_cal`（S12 矫正）：`M1 @ 0.40/0.50` 与 `T2_TH040` 之间 `-0.539` 的差距及 `0` 次 `B1` dustbin 拒识主要来源于 `delta_s` 引起的 `0.546 -> 0.687` 标尺上移；对齐到 `0.55–0.68`（及同步 `dustbin_score = sim_threshold`）后，`M1`（或 `M1+B1`）可恢复 `0024`/`0053` 掉分并至少追平/超过 `T2_TH040`（`57.572`）。
  - `H_scale`（`N1`）：仅对 `delta_s` 施加批内零均值中心化正则 $\lambda_{\text{center}} (\overline{\Delta S_{\text{valid}}})^2$（不破坏正负相对排序），即可将 `val/computed_sim_threshold` 锚定回 CAMEL 原生区间（`~0.52–0.58`）而不损失 `M1` 的 `-36.2%` onset 压降能力。
  - `H_occ_drop`（`N2`）：在训练期专对**存在 `IoU >= 0.25` 空间重叠干扰框**的正匹配随机移除真实检测框（`p_occ_drop = 0.20`），只对这批模拟“真目标被遮挡缺检”的轨迹施加局限于重叠邻框的最大分惩罚，能在**不伤害全局表征 AUROC**（克服 `M3` 坍塌）的前提下显著减少 `60%` 缺检误抓型 onset。
  - `H_dual_anchor`（`N3`）：同时向 GAFFE 注入 **永不被 50 帧滑窗淘汰的早期低冲突锚点一致性信号 `e_anc(i,j)`** 并在训练时对足够长轨迹的最近 `2–4` 槽注入局部重叠干扰（同时保证最老 `K_anchor=4` 锚点槽绝对干净），能打破“近期滑窗自洽漂移陷阱”，大幅缩短 `post_switch_cross` 尾段。

- **Stage B 零重训矫正设计（复用 S12 `c_m1_kin_edge_seed13_e20_v1/CAMEL/last.ckpt`，SHA256 `b3ca74bf...`）：**
  1. **`B_M1_CAL_GRID`**：在 `motion_dev8`（`states_dev8_gt_free.pklz`）上跑 4 点独立闭环评测：`sim_threshold` ∈ `{0.55, 0.60, 0.64, 0.68}`（零新训练，直接调用 `M1` 原模型）。
  2. **`B_M1_B1_CAL`**：修正 S12 `D_M1_PLUS_B1_TH040` 中 `dustbin_score` 写死 `0.40`（0 次拒识）的缺陷，设为 `dustbin_score = sim_threshold`、`tau_ds = 0.10`，跑 `{0.58, 0.62}` 两组，记录真实 `dustbin_rejections` 次数。
  3. **`B_ONSET_DIFF_0024_0053`**：从 S12 `M1@0.40` 与 Stage B 校准最佳点（及 `M1+B1`）的 `0024` 与 `0053` 上抽取并对照导致 `86.04`/`62.50` 掉分的精确 onset 帧上下文（帧号、`sim_val`、双向 `margin`、`max_conflict_iou`、是否缺检），写入 `diagnostics.json` 与 `evidence/`。

- **Stage C 收敛型三联组设计（共享 S09 C1 TE ckpts 初始化、`motion_train32`、`seed=13`、20 epochs、`last.ckpt`）：**
  - **共用校准评测规则（零信息泄漏）**：除固定在 `sim_threshold = 0.40`（向后可比基准）评测外，每一组训练结束后读取其自身在 `train32` 附带 `states_dev8_train_ref` 验证集 epoch 20 的 `val/computed_sim_threshold`，固定按 `th_cal = clip(round(val_th - 0.14, 2), 0.35, 0.65)`（复用 S11 `T_TWOSTAGE` 在 `0.546 - 0.146 = 0.40` 验证出的最优偏置量，**严禁按 `dev8` 闭环分数选阈值**；若某臂 `th_cal == 0.40` 则复用同一 run 不重复评测）同步报告闭环表现。
  - **组 1：`N1_KIN_SCALE_ANCHORED`（标尺零锚定 `C'` 基准组）**：
    - 完全保留 `M1` 的 `13` 维成对因果运动边偏置与注意力偏置（`+146,305` 参数）；
    - 修改输出与损失：将每样本有效对的 `delta_s` 均值惩罚纳入训练损失：
      $$\mathcal{L}_{N1} = \mathcal{L}_{\text{base\_sim\_delta}} + \lambda_{\text{center}} \frac{1}{B}\sum_{b=1}^{B} \left(\overline{\Delta S_b[\text{pair\_valid}]}\right)^2, \quad \lambda_{\text{center}} = 0.50$$
      （只约束 `delta_s` 平均漂移接近 `0`，不限制正负个体边对给出 `+0.3 / -0.3` 区分信号）。
  - **组 2：`N2_KIN_OCC_DROP_SUP`（`N1` + 遮挡删帧抗误抓监督）**：
    - **完全相同架构与参数量**（与 `N1` 严格同参）；
    - 在每个训练 batch 内（`training_step` 前向之前、且仅在训练集开启）：
      1. 对每条有效轨迹 `t`（对应 GT ID `y_t >= 0`）及其当前帧真值正检测 `d^+`（`y_d == y_t`），计算 `d^+` 与同帧其余有效检测的 `IoU`，标记满足 $\max_{d' \neq d^+} \text{IoU}(b_{d^+}, b_{d'}) \ge 0.25$ 的轨迹为**可遮挡冲突轨迹**；
      2. 在可遮挡冲突轨迹中以 `p_occ_drop = 0.25` 独立采样（若某样本不存在则跳过；确保不同样本内只删去被采样轨迹对应的真值检测 `dets.masks[b, d^+] = False`、`dets.targets[b, d^+] = -1`，不删掉同帧其他人的框），记这些轨迹集合为 $\mathcal{T}_{\text{occ}}$；
      3. 主对比损失 $\mathcal{L}_{\text{base}}$ 自动忽略已置掩的 `d^+`（不污染正常样本对比），同时**只对 $\mathcal{T}_{\text{occ}}$** 施加遮挡干扰框拒识损失（取其与原始 `d^+` 有 `IoU >= 0.25` 的邻近干扰检测集合 $\mathcal{N}_{\text{iou}}(t)$）：
         $$\mathcal{L}_{\text{occ}} = \text{mean}_{t \in \mathcal{T}_{\text{occ}}}\left[\frac{1}{\beta}\log\left(1 + \exp\left(\beta\left(\max_{d' \in \mathcal{N}_{\text{iou}}(t)} S_{t, d'} - \tau_{\text{occ}}\right)\right)\right)\right]$$
         固定 $\beta = 10.0, \tau_{\text{occ}} = 0.35, \lambda_{\text{occ}} = 0.15$（温和针对性监督，且记录每 epoch `occ_drop_count` 供核查）。
  - **组 3：`N3_DUAL_ANCHOR_CAMEL`（`N2` + 早锚双时间尺度记忆与近期污染鲁棒性，主创新方案）**：
    - **严格因果早锚记忆构造（训练与在线推理保持口径一致）**：
      - **训练期（`states_train32`）**：每条轨迹已有最多 $S \le 50$ 个有效历史槽（由 `age` 标记时间远近）。取其中**最老的至多 $K_{\text{anchor}} = 4$ 个有效槽**作为早期锚点参考 `anchor_slots`（代表轨迹较早期的干净观测），取**最新的至多 $K_{\text{recent}} = 8$ 个有效槽**作为近期参考 `recent_slots`；
      - **在线推理期（`Tracklet` 类拓展）**：在 `Tracklet` 上新增不可被 50 帧滑窗 FIFO 挤出的早锚缓存 `early_anchor_embs` / `early_anchor_vis`（最多存前 $K_{\text{anchor}} = 4$ 帧置信度 $\ge 0.7$ 且同帧无 `IoU >= 0.30` 冲突检测的更新；若建轨前 `8` 帧始终处于拥挤冲突中，则退而取最早期 `K_anchor=4` 帧低模糊度有效更新），保证长轨迹超过 50 帧后依然保有创生期的原初外观锚点（不跳过正常近期写入，零更新饥饿）。
    - **可控近期污染注入（`N3` 训练期仅作用于近期窗口、`p_hist_pollute = 0.25`）**：
      - 对有效槽数 $\ge 6$ 的轨迹（保持最老 $K_{\text{anchor}}$ 锚点槽 **100% 干净绝不污染**！），以概率 `p_hist_pollute = 0.25` 选取其**最近的 2 至 4 个槽（最小 `age`）**，用同帧或同 batch 中与该轨迹空间最邻近的另一个不同 GT ID 历史特征替换（模拟真实发生短暂错抓污染了近期滑窗的情景）；
      - 注意核心区别：在训练与推理中，同时用分区可见性加权余弦相似度计算 **`4` 维早锚一致性特征 `e_anc(i,j)`**：
        `[sim_anchor_reid(i,j), sim_recent_reid(i,j), sim_anchor_reid(i,j) - sim_recent_reid(i,j), anchor_recent_drift(i)]`（其中 `anchor_recent_drift(i) = 1 - cos(anchor_proto_i, recent_proto_i)` 刻画近期窗口是否已发生漂移污染），并将 `[e_kin (13D), e_anc (4D)]`（共 `17D`）送入与 `N1/N2` 同结构的 `edge_attn_td`、`edge_attn_dt`、`edge_sim_mlp`（新增 `4D` 对应线性权重零初始化以保 step-0 等价，额外参数增量仅约 `+10k`）。

- **实现责任（明确标注服务器待实现项与 Stage A 审计项）：**
  - 本地仅交付本任务书与协议；服务器须在 Stage A 先核对 `CAMELTrack@bc1c2610` 的 `cameltrack/cameltrack.py`（`Tracklet` 更新与特征提取顺序）与 `cameltrack/train/dataset.py`（`tracks.feats` 内槽位按 `age` 的排列方向及分区 `embeddings (6,128)` / `visibility_scores (6,)` 语义），将实测结论写入 `source_audit.json`，再实现 `scripts/run_s13_camel.py`、`scripts/export_s13_report.py` 与 `tests/test_s13_camel.py`。
  - Stage A 单测必须覆盖：① `states_train32` 中 `age` 槽序核验与在线 `Tracklet` 早锚槽超出 50 帧不淘汰不变量；② `occ_drop` 只遮罩选中真检测且不污染同帧其他 ID / 不产生 `NaN`；③ `hist_pollute` 严格只替换最近槽、最老锚点槽逐元素保持不变；④ `N1/N2/N3` step-0 零初始化与 `M1` 等价（或与基线零初始化等价）及全参数有限非零梯度验证。

- **数据来源、精确 split 与 GT 边界：**
  - 全程复用 `manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml`（SHA256 `dd0b7fef...`）中的 `motion_train32`（32 段、34,231 帧）与 `motion_dev8`（8 段、7,565 帧）；本轮不触碰官方 `val`/`test`。
  - 推理全程严格使用 `states_dev8_gt_free.pklz`，零 GT、零未来帧访问；仅 Stage D 回放与 TrackEval 允许只读加载 GT，并须校验回放 MOT 与在线 MOT 逐字节一致。

## 执行矩阵

| stage / ID | 输入与范围 | 操作 / 命令入口及实现责任 | 必须回传 | 通过条件 | 通过后 / 失败后 |
|---|---|---|---|---|---|
| **A：源码审计、实现与单测** | S09/S12 ckpts、`states_*`、`CAMELTrack@bc1c2610` | 校验全部复用资产 SHA256；审计 `cameltrack.py`（`Tracklet` gallery 与 `age` 槽序）；编写并单测 `run_s13_camel.py` / `test_s13_camel.py` | `source_audit.json`、`runtime_lock.json`、`resolved_configs/`、单测输出 | SHA256 一致、槽序对齐核清、所有早锚持久化/删帧/污染/梯度单测 PASS | **通过自动进 B**；修复 >2 次失败即停 |
| **B：S12 `M1` 标尺校准与错例靶向解剖（零训练）** | S12 `M1` `last.ckpt` (`b3ca74bf...`) + `states_dev8_gt_free.pklz` | 1. 扫 `M1` @ `{0.55, 0.60, 0.64, 0.68}`（4 组）；<br>2. 扫校准 `M1 + B1`（`dustbin = sim_th`）@ `{0.58, 0.62}`（2 组）；<br>3. 导出 `M1` 在 `0024`/`0053` 早期掉分 onset 上下文及与校准组对照 | 6×8=48 条逐视频 + 6 条 `COMBINED_SEQ` + `B1` dustbin 拒识数 + `0024/0053` 掉分帧诊断 | 零 GT 泄漏，导出误差 `0.0`，`M1+B1` 确认实际激活 dustbin 拒识 | **通过自动进 C**（无论 Stage B 胜负均进 C 以检验统一架构增量） |
| **C：收敛型三联组训练与零泄漏校准评测** | `motion_train32` + S09 C1 TE ckpts；`states_dev8_gt_free.pklz` | `seed=13`、20 epochs、`last.ckpt` 新进程显式重载：<br>1. `N1_KIN_SCALE_ANCHORED`；<br>2. `N2_KIN_OCC_DROP_SUP`；<br>3. `N3_DUAL_ANCHOR_CAMEL`；<br>每组测 `@0.40` 与其自身 `val`-派生 `th_cal`（若 `th_cal==0.40` 则复用） | 3 条 20ep `learning_curves.csv`（含 `val/computed_sim_threshold`、`occ_drop_count`、`pollute_count`）、`delta_s` 均值检查、各组 `th_cal` 及完整 `dev8` 指标 | `0` 非有限梯度、重载 diff `0.0`、`N1` 验证阈值回稳（`< 0.62`）、全部评测导出误差 `0.0` | **通过自动进 D**；OOM 保留记录停止 |
| **D：胜出组长尾解剖、协同校验与案例证据** | Stage C 最优组（按 `dev8` 最高 HOTA，平手看 AssA） | 1. 在 Stage C 最优组上按其 `th_cal` 运行 `B1_DUAL_SOFTMAX`（1 组）；<br>2. 运行 S11/S12 同口径 Onset & `post_switch_cross` 尾段回放解剖，核验 `onset_cross` 与 `post_switch_cross` 双指标变化；<br>3. 导出最多 16 例配对案例（覆盖 S11 `0024`/`0053`/`0037` 等关键转折帧） | 协同评测指标、`diagnostics.json`（含 S11/S12/S13 三代 `onset` 与 `post_switch` 对比表）、`cases.jsonl` & `evidence/` | 回放 MOT 与在线 MOT 逐字节一致、零 GT 泄漏 | **通过自动进 E** |
| **E：报告交付、晋级/止损裁决与交接** | A–D 全部产物 | 运行 `scripts/export_s13_report.py` 核验哈希并按硬门槛标定推荐路径（Case 1 晋级 S14 / Case 2 边界切换 / Case 3 停止本线），更新 `reports/README.md`、`docs/progress_log.md`、`docs/a2a/LATEST.md` | `reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/` 完整报告包 | `checks.json` 与 `artifacts.json` 全校验通过 | **停在 E，交回研究端按硬门槛拍板 S14**；无授权不 commit/push |

- **预期规模计数：**
  - 训练运行：**`3` 次**独立 20-epoch Stage 2 联合训练（`N1/N2/N3`，均 `seed=13`）。
  - 在线闭环评测：Stage B `6` 组 ＋ Stage C 至多 `6` 组 ＋ Stage D `1` 组 ＝ **至多 `13` 组**（至多 `104` 条逐视频 + `13` 条 `COMBINED_SEQ`）。
  - 离线回放解剖：**`1` 组**（Stage C 最优组 Onset & 尾段回放；另加 Stage B 只读对 `0024/0053` 的定向核查）。
- **预算与安全约束：**
  - 物理 GPU `1–7` 单卡（首选物理 GPU `1`，**物理 GPU 0 禁用且严禁触碰**）；
  - 总 GPU 预算 **`≤ 8.5` 卡时**（A `≤ 0.3h`、B `≤ 0.7h`、C `≤ 6.0h`、D `≤ 1.3h`、E `≤ 0.2h`）；墙钟 `≤ 24` 小时；新增数据盘存储 `≤ 15 GiB`（保留 `≥ 50 GiB` 空闲）；
  - 每项工程阻断最多 `2` 次兼容性修复。

## 回传和停止点

- **权威报告目录：** `reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/`（大型产物存 `/media/imc/data/mot-data/runs/s13_camel_calibration_and_dual_memory_20261128/`）。
- **研究端凭 Git 即可回答的三个核心裁决问题：**
  1. Stage B 校准 S12 `M1` 阈值标尺并激活 `B1` dustbin 拒识后，是否直接追平或超过 `T2_TH040`（`57.572`）？`0024`/`0053` 掉分是单纯阈值盲区引起，还是仍有真目标缺检级联发生？
  2. `N1`（标尺锚定）、`N2`（遮挡删帧抗误抓）与 `N3`（早锚双时间尺度抗污染记忆）在校准阈值下是否形成清晰的逐级增量，并同时压低 **`onset_cross`（`< 95`）与 `post_switch_cross` 长尾（`<< 40,570` / `31,791`）**？
  3. 本轮最优方案是否命中 **Case 1 硬晋级门槛（`HOTA >= 59.1` 且 `AssA >= 42.85`、`>= 5/8` 胜率）**？若命中则直推 S14（多 seed + 完整 DanceTrack `val`/`test` + SportsMOT）；若未命中，是否果断止损并转向 Route B/C？
- **最后一个允许自动执行的步骤与本轮明确不运行项：**
  - 最后自动步骤为 **Stage E（报告导出与交接文档更新）**。
  - **本轮明确不运行**：multi-seed（`37/73`）、官方 DanceTrack `val`/`test`、SportsMOT、S11 `G1–G4` 写端过滤、S12 `M2` DOA / `M3` `tau_neg`、未授权的 `git commit` / `git push`。
