# Experiment evidence snapshots

此目录只保存体积小、可审核、不可变的实验结果快照。原始数据、权重、检测缓存、完整 tracker 输出和可再生成的大型运行目录仍保存在仓库外。

2026-11-25 研究端补充审阅：[S07 复盘与方向决定](../docs/research_status_review_20261125.md)，附 [Git 导出核查记录](../docs/audits/s07_local_review_20261125/verification.json)。这是本地证据审阅，不是新 checkpoint 推理或跟踪实验；S07 原报告与停止判定未改写。

每个快照必须注明数据划分、运行范围、原始文件 SHA256 和限制；train 诊断不得写成 val 或新方法结果。

2026-11-25 起，研究端下发遵循 [任务规范](../docs/experiment_task_contract.md)，服务器按任务逐阶段回传。S08 CAMELTrack 有界基线闭环已于 2026-11-25 在物理 GPU 1 上完成 A–F 全阶段，权威报告见 [S08报告](baselines/s08_camel_baseline_20261125/README.md)（决定：`CONTINUE`，仅授权进入研究端下一轮单问题设计）。S09 阈值标定、Two-Stage 基线与失败归因已于 2026-11-26 在物理 GPU 1 上完成 A–E 全阶段并停止在 Stage E，权威报告见 [S09报告](diagnostics/s09_camel_calibration_and_attribution_20261126/README.md)。S10 CAMELTrack 纯拒写基准分解与抗闭环污染实验已于 2026-11-27 在物理 GPU 1 上完成 A–E 全阶段并停止在 Stage E，权威报告见 [S10报告](feasibility/s10_camel_anticontamination_20261127/README.md)。S11 零重训低阈值标定、149 例 Onset 解剖与冲突感知拒写实验已于 2026-11-27 在物理 GPU 1 上完成 A–E 全阶段并停止在 Stage E，权威报告见 [S11报告](feasibility/s11_camel_onset_and_conflict_gate_20261127/README.md)。

2026-11-23 起新增报告遵循 [报告规范 v1](../docs/experiment_reporting.md) 和 [报告包模板](../docs/templates/report_bundle/README.md)：完整精度汇总/逐视频指标随 Git 回传，负结果与未完成组也保留。历史 S01/S04 机器可读补齐尚待服务器执行，不代表本地已有原始数值。[当前交接与下一步](../docs/a2a/LATEST.md)。

当前快照：

- `feasibility/s11_camel_onset_and_conflict_gate_20261127/`：S11 CAMELTrack 零重训低阈值标定、149 例 Onset 解剖与冲突感知拒写（物理 GPU 1，0 训练，总计 `1.008 GPUh`、`0.086 GiB`）：Stage B `T2_TH040` (`0.40`) 将 `T_TWOSTAGE` 提升至 `57.572` HOTA / `40.845` AssA / `60.117` IDF1（回收 `973` FN，`5615 -> 4642`）；Stage C 与 S09 `T_TWOSTAGE_TH050` MOT 输出逐字节对齐，`149` 例 onset 中 `74.5%` 为连续帧、`89.9%` 伴 `IoU >= 0.30`、`74.5%` `margin < 0.10`（`-margin` 判别 onset AUROC `0.9432`）、`22.8%` 为 `2-cycle` 互换，且 `anchor_reid_dist` 将 `31,791` 帧后切换尾段判别 AUROC 从近期 `0.5251` 拉升至 `0.7062`；Stage D 证明写库端 `G1` 虽大幅优于 `C0_TAU065`（`48.686` vs `43.509`），但 `G1`–`G4` 均因拥挤区 `~42%` 误拒写致 `FN` 升至 `8,055–9,759`，确认 **Situation 2** 并路由 S12 主攻关联打分与分配层（`C' + A'`）。包含 `7` 行总体、`56` 行逐视频、`16` 例 onset 因果证据与 `47` 项产物哈希。
- `feasibility/s10_camel_anticontamination_20261127/`：S10 CAMELTrack 纯拒写分解与抗污染训练闭环（物理 GPU 1，总计 `4.072 GPUh`、`4.27 GiB`）：Stage B `D2_SKIP_ONLY`（`gt_injected_on_miss=0`）达 `63.781` HOTA / `49.940` AssA（相对 S09 `T_TWOSTAGE_TH050` `+9.338` HOTA / `+13.236` AssA），明确剥离纯拒写上限与 S09 `D2 v3` GT 漏配注入增益（`81.907`）；零重训 `C0` 标量门控严重掉点（`48.838`/`43.509`）；Stage C 等噪 `C1` (`51.780`) > `C3` (`49.781`) > `C2` (`48.210`)，三者均低于 S09 无污染基线 `54.443`；Stage D 软拒写 `49.313`，真实闭环跨 ID 写入一致性判别 AUROC `0.5002`（149 例首次切换 `0.6220` vs `1-sim` `0.8512`，而 `31,791` 例为漂移后自洽尾段）。包含 `7` 行总体、`56` 行逐视频、`474` 行曲线、`16` 例因果证据与 `54` 项产物哈希。
- `diagnostics/s09_camel_calibration_and_attribution_20261126/`：S09 CAMELTrack（commit `bc1c2610`）在物理 GPU 1 上完成 A–E 全阶段闭环：Stage B 零重训阈值标定（`T_SCRATCH` @ `0.46/0.50/0.519729` 与 `P_OFFICIAL` @ `0.50`）、Stage C `train32` 两阶段训练（3 个单 cue TE 20ep + 联合 GAFFE 20ep -> `T_TWOSTAGE` @ `0.50` 与 `0.5460`）、Stage D1 100% 官方 IDSW 相似度边际回放（`786` / `1,412` 例）及隔离单列的 Stage D2 1-to-1 GT 路由干净历史反事实探针（权威 `v3`，`AssA 36.704 -> 80.063`，`IDF1 53.629 -> 92.153`；保留 `v1`/`v2` 工程记录）；结论指向 S10 主线优先立项 `B'` 抗闭环历史污染（兼顾单帧反转抑制）。
- `baselines/s08_camel_baseline_20261125/`：S08 CAMELTrack（commit `bc1c2610`）有界基线训练与评测闭环；包含 A–F 全阶段证据、20 epoch `T_SCRATCH`（seed 13，train32）与 `P_OFFICIAL` 在 `motion_dev8` 的全精度 TrackEval 对比及 16 例抽样官方 IDSW 失败清单。
- `baselines/g1_ocsort_train_20261110/`：完整 DanceTrack train 诊断基线。
- `baselines/g1_ocsort_val_20261111/`：查看指标前冻结的完整 DanceTrack val G1 参考基线。
- `baselines/b0_b1_equivalence_motion_dev_20261112/`：冻结 motion-dev 上直接 B0 与关闭研究功能的 B1 适配器全量等价验收；float64 tracker 输出逐元素一致。
- `diagnostics/g2_train_failure_attribution_20261111/`：4 个 train 失败序列的冻结事件定义、20+20 对照抽样、描述统计和外部审阅图索引。
- `diagnostics/g2_candidate_contention_20261111/`：对 797 个父事件逐帧区分原始候选缺失与一对一匹配竞争的 train-only 探索性补充分解。
- `diagnostics/g2_d2_first_association_v2_20261111/`：只替换目标 track 当前预测框的 GT 特权首轮关联反事实；不写回 tracker，也不代表闭环指标提升。
- `diagnostics/g2_d1_history_correspondence_20261111/`：D1 v1 失败定义的透明记录；因 92/112 成功对照被错误判坏而废弃，不作为历史污染证据。
- `diagnostics/g2_d1_clean_gap_history_v2_20261111/`：D1 v2 探索性干净缺失段历史反事实；53 个身份改变失败恢复 0 个，对 112 个成功对照新增 1 个首轮正确并破坏 7 个。
- `diagnostics/g2_d1_score_margin_addendum_20261111/`：D1 v2 审计后总关联分补充；保持 165 个事件和父结果不变，记录 IoU+方向分数、门控与最佳竞争 track margin。
- `diagnostics/g2_d3_gt_detection_coordinate_20261112/`：只修正已有目标正检测坐标的 D3 特权首轮反事实；失败组新增正确 14/53，对照新增正确 1、破坏 7，并报告与 D2 的交并覆盖。
- `diagnostics/g2_d4_direction_cost_ablation_20261112/`：全局移除首轮方向项的冻结对照；失败组新增正确 0/53，成功对照破坏 3/112。
- `diagnostics/g2_d5_joint_d2_d3_20261112/`：把已知正确 pair 两侧同时设为同一 GT 框的关联一致性压力测试；失败组新增正确 52/53，但不代表可实现恢复上界或因果协同。
- `engineering/async_queue_smoke_20261112/`：GPU1/2/3 标签下的异步队列故障隔离验收；一个预期失败不影响两个 PASS，聚合器只读取 PASS。无 CUDA 张量、无研究指标。
- `engineering/p1_deterministic_overfit_smoke_20261112/`：在物理 GPU1 上完成的 128 样本确定性 GRU 过拟合 smoke；验证真实 CUDA、mask、归一化和 checkpoint，不是泛化或跟踪效果。
- `data/p1_motion_samples_v1_20261112/`：冻结 P1 严格在线 motion-train/dev 样本契约的构建与独立重复；大数组留在数据盘，只登记计数、内容哈希、来源和限制。
- `feasibility/p1_deterministic_offline_v1_20261112/`：三 seed 完整 motion-train→motion-dev 确定性预测可行性门；五项冻结条件通过并获准进入自然闭环，同时保留长 gap IoU 退化和 provisional WARN 审计。
- `diagnostics/e0_reassociation_output_chain_20261119/`：固定 831 事件的匹配、更新、输出和 GT 评价断点；424 个曾成熟输出轨迹在正检测更新后没有当帧输出。
- `feasibility/e1_mature_reemit_factorial_20261119/`：B0/CV × 原输出/U 输出四条件探索；CV+U 有 HOTA 增益，也增加 IDSW、Frag 和 FP。
- `feasibility/e2_reemit_guard_20261119/`：在已使用的 8 段 motion-dev 上比较分数与连续命中简单门控；同集风险标签影响阈值选择，provisional WARN。
- `feasibility/e3_short_gap_support_20261119/`：24 段 train-fit 的低分短缺失空间补回对照；G 相对 E 的 HOTA +0.050、FP +84，未过预写门槛，留出未运行；审计记录 GT 哈希时序偏离。
- `diagnostics/e4_oracle_association_headroom_20261119/`：24 段 train-fit 的一次 GT 特权正确配对五帧局部分支；454 个条件化错配事件、目标净增 1,426 正确帧、全锚点净增 1,643，headroom 门通过但不代表在线方法或官方完整序列指标。
- `feasibility/e5_nonoracle_appearance_local_20261119/`：24 段 train-fit 中 576 次配平的非 GT 外观/几何五帧局部分支；外观相对几何全锚点净帧 −6、其他锚点损伤 422/420，未过 fit 门，留出未运行。
- `diagnostics/e5_error_analysis_20261120/`：对 E5 的 47 次选择分歧、414 次共同其他锚点损伤及正常首轮配对归属做只读事后拆解；576 个事件连接复核，不构成新方法或新晋级门。
- `feasibility/s01_visual_association_20261120/`：固定检测的 G/Z/L/M 深度外观基线，24 段 train-fit 完整 TrackEval；S0 工程验收与同家族暂定完整性审阅均在此，不能外推为新方法或独立泛化。
- `diagnostics/s02_natural_identity_mechanism_20261121/`：在冻结 S01 轨迹上做 GT 辅助、只读的自然异主更新普查；事件标签和重复窗口均为诊断代理，预写门只授权计划受控记忆测试。
- `diagnostics/s03_single_memory_write_counterfactual_20261122/`：冻结 S01 M 轨迹上的单次记忆写入局部分支、严格对照重评和事后仪表补检；GT 特权机制诊断，不是在线方法或完整序列评测。
- `diagnostics/s04_bounded_memory_freeze_20261122/`：在与 S01/S03 fit 不重叠的八段 train 视频上，对比单次跳写和连续冻结目标视觉记忆的特权局部分支；记录停止门与独立审阅。
- `diagnostics/s05_visual_decision_coverage_20261123/`：补齐 S01/S04 机器可读证据，在复用的八段开发视频上完成 G/L/M 官方 TrackEval，并在 fit24 的 M 因果状态上普查合法 M/G/V 分量动作；41 个净正向代理分量跨 17 视频，资源门通过但不代表可学习或闭环增益。
- `feasibility/s06_component_selector_seed13_20261123/`：seed 13 前置特征仪表在空 tracker 边界两次工程失败；按冻结的一次修复上限停止。未训练、未评测、无科学结论，所有 GPU 均未使用。
- `feasibility/s06_component_selector_seed13_retry3_20261124/`：经透明 retry 完成 seed 13 七条件自然闭环；冻结 Context+risk 主门仅过 2/8，决定 `STOP_S06_NO_SEEDS_37_73`。Context-no-risk 有单 seed 描述性正结果，但不替换主门；报告含完整精度逐视频表、曲线、checkpoint、案例、偏离和独立审计。
- `diagnostics/s07_s06_checkpoint_postmortem_20261124/`：CPU-only 读取 S06 全部 12 个 checkpoint，在固定 train18/internal-dev6 上复算 g/h、排序和拒绝。后期模型降低标量 MSE，但 Context epoch30 的 internal-dev6 决策不胜 always-M；结果后 action-type 常数 MSE 更低。保留案例 attempt1 无效谱系、权威 attempt2、分母偏差和同家族暂定 WARN 审计；不授权新性能实验。
