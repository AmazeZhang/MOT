# Scripts

只放已经实现并有对应测试或验收记录的阶段入口。脚本负责参数解析和调用 `mot_research` 包，不在脚本中复制核心算法。

计划顺序：环境/数据检查 → TrackEval 人工例 → 检测导出 → 缓存重放 → 基线跟踪 → 诊断 → 模型训练。尚未实现的入口不创建空壳命令。

当前已验收入口：

- `validate_trackeval_metrics.py`：运行完美、漏检、身份交换三个人工评测例。
- `run_ocsort_smoke.py`：在显式安全 GPU 上执行短序列 YOLOX 检测、原图坐标缓存、OC-SORT 和缓存重放一致性检查。
- `replay_ocsort_cache.py`：CPU-only 地从固定检测缓存重放完整序列，拒绝覆盖旧输出，用于独立重复性检查。
- `render_tracking_overlay.py`：在原图上叠加 GT、固定检测和 tracker 输出，用于人工检查坐标和 ID。
