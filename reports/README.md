# Experiment evidence snapshots

此目录只保存体积小、可审核、不可变的实验结果快照。原始数据、权重、检测缓存、完整 tracker 输出和可再生成的大型运行目录仍保存在仓库外。

每个快照必须注明数据划分、运行范围、原始文件 SHA256 和限制；train 诊断不得写成 val 或新方法结果。

当前快照：

- `baselines/g1_ocsort_train_20261110/`：完整 DanceTrack train 诊断基线。
- `baselines/g1_ocsort_val_20261111/`：查看指标前冻结的完整 DanceTrack val G1 参考基线。
- `diagnostics/g2_train_failure_attribution_20261111/`：4 个 train 失败序列的冻结事件定义、20+20 对照抽样、描述统计和外部审阅图索引。
- `diagnostics/g2_candidate_contention_20261111/`：对 797 个父事件逐帧区分原始候选缺失与一对一匹配竞争的 train-only 探索性补充分解。
- `diagnostics/g2_d2_first_association_v2_20261111/`：只替换目标 track 当前预测框的 GT 特权首轮关联反事实；不写回 tracker，也不代表闭环指标提升。
- `diagnostics/g2_d1_history_correspondence_20261111/`：D1 v1 失败定义的透明记录；因 92/112 成功对照被错误判坏而废弃，不作为历史污染证据。
- `diagnostics/g2_d1_clean_gap_history_v2_20261111/`：D1 v2 探索性干净缺失段历史反事实；53 个身份改变失败恢复 0 个，对 112 个成功对照新增 1 个首轮正确并破坏 7 个。
- `diagnostics/g2_d1_score_margin_addendum_20261111/`：D1 v2 审计后总关联分补充；保持 165 个事件和父结果不变，记录 IoU+方向分数、门控与最佳竞争 track margin。
- `diagnostics/g2_d3_gt_detection_coordinate_20261112/`：只修正已有目标正检测坐标的 D3 特权首轮反事实；失败组新增正确 14/53，对照新增正确 1、破坏 7，并报告与 D2 的交并覆盖。
- `diagnostics/g2_d4_direction_cost_ablation_20261112/`：全局移除首轮方向项的冻结对照；失败组新增正确 0/53，成功对照破坏 3/112。
- `diagnostics/g2_d5_joint_d2_d3_20261112/`：把已知正确 pair 两侧同时设为同一 GT 框的关联一致性压力测试；失败组新增正确 52/53，但不代表可实现恢复上界或因果协同。
