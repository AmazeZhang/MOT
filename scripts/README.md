# Scripts

只放已经实现并有对应测试或验收记录的阶段入口。脚本负责参数解析和调用 `mot_research` 包，不在脚本中复制核心算法。

计划顺序：环境/数据检查 → TrackEval 人工例 → 检测导出 → 缓存重放 → 基线跟踪 → 诊断 → 模型训练。尚未实现的入口不创建空壳命令。

当前已验收入口：

- `validate_trackeval_metrics.py`：运行完美、漏检、身份交换三个人工评测例。
- `run_ocsort_smoke.py`：在显式安全 GPU 上执行短序列 YOLOX 检测、原图坐标缓存、OC-SORT 和缓存重放一致性检查。
- `replay_ocsort_cache.py`：CPU-only 地从固定检测缓存重放完整序列，拒绝覆盖旧输出，用于独立重复性检查。
- `render_tracking_overlay.py`：在原图上叠加 GT、固定检测和 tracker 输出，用于人工检查坐标和 ID。
- `analyze_g2_failure_events.py`：按冻结协议从 train 固定检测、GT 和基线输出提取“检测短时缺失后恢复”事件，生成身份保持/改变对照与代理特征；不读取 val，也不把代理写成因果真值。
- `render_g2_event_overlays.py`：为冻结抽样事件生成紧凑审阅图，固定展示缺失前、缺失段首尾和重新出现后的同一区域；图像只作为人工归因证据，不作为模型输入。
- `analyze_g2_candidate_contention.py`：在不改动父事件的前提下，逐缺失帧区分“无合格原始候选”和“候选存在但一对一匹配竞争”，结果仍是使用 GT 的诊断代理。
- `analyze_g2_d2_first_association.py`：完整重放基线并旁路捕获首轮关联；对每个已判定 train 事件独立注入当前 GT 位置框，只计算不写回的 D2 反事实关联空间。
- `analyze_g2_d1_history_correspondence.py`：完整重放基线，使用过去 30 帧已有噪声检测的 GT 对应解析 live track 的最近历史所有者；只重标记当前既有配对的语义身份，不改框、配对或 tracker 状态。
- `analyze_g2_d1_clean_gap_history.py`：D1 v2 探索性替代；从缺失前可信端点复制目标 track，在缺失段阻止其他 GT 的检测污染该副本，再以同一 Kalman 模型评估首轮关联，不写回基线。
- `analyze_g2_d1_score_margin_addendum.py`：在不改变 D1 v2 事件和结果的前提下，补记首轮正检测的原始 IoU+方向总分、竞争 track margin、门控和一对一快捷路径，并逐事件断言父结果一致。
- `analyze_g2_d3_detection_coordinate.py`：只把已有目标正检测的 XYXY 替换为当前 GT 框，保持分数、数量、顺序和全部 track 状态不变，记录首轮恢复、副作用、门控、总分 margin 与全局配对变化。
- `analyze_g2_final_counterfactual.py`：同一入口运行冻结的 D4 方向代价消融或 D5 同时 D2+D3 特权构造；两个模式使用独立原子输出，失败保留 `.failed/failure.json`，并登记 assignment backend/version。
