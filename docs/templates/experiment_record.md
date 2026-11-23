# 实验记录模板

状态：待填写。每次运行复制一份，不覆盖旧记录。

本文件可作单次运行简记；2026-11-23 起完整 Git 回传使用 [报告规范](../experiment_reporting.md) 与 [报告包模板](report_bundle/README.md)。简记不能替代机器可读总体/逐视频结果和失败记录。

- run_id / 日期 / 阶段 / 对应假设：
- Git SHA / 是否干净 / 开发 diff：
- 环境版本 / 上游 SHA / 补丁：
- 数据 revision / split hash / 检测 hash / 权重 hash：
- 模型 / loss / 初始 checkpoint / 总更新数 / 样本暴露量：
- train seed / inference seed / K / S / bandwidth：
- 机器代号 / GPU 型号 / 精度 / 线程 / workers / batch：
- 完整执行命令与配置文件：
- 开始结束时间 / GPU 小时 / 峰值显存 / 失败重试次数：
- 输出与日志路径：
- 官方总体 HOTA / AssA / DetA / IDF1 / IDSW：未运行时写“未产生”
- 每视频结果与难例：
- 预测指标 / 实际关联指标 / 延迟 p50 p95：
- 与哪个 run 配对，除了研究因素还改变了什么：
- 是否使用 GT 特权信息 / 人工缺失 / 离线后处理：
- 已观察事实：
- 解释（与事实分开）：
- 异常与下一步：

不因结果不理想删除记录；不得仅记录最佳种子。
