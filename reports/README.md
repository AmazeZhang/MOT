# Experiment evidence snapshots

此目录只保存体积小、可审核、不可变的实验结果快照。原始数据、权重、检测缓存、完整 tracker 输出和可再生成的大型运行目录仍保存在仓库外。

每个快照必须注明数据划分、运行范围、原始文件 SHA256 和限制；train 诊断不得写成 val 或新方法结果。

当前快照：

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
