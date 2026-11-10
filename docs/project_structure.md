# 项目目录与产物约定

本页定义仓库内代码、可提交元数据与仓库外大文件的边界。目录可随里程碑逐步填充，但不能用占位命令冒充已实现功能。

```text
MOT/
  src/mot_research/
    data/           # 数据契约、校验、划分和训练样本
    models/         # 共享编码器与 regression/Gaussian/MDN/CFM
    association/    # 候选兼容度、门控、dustbin 和训练代理
    tracking/       # OC-SORT 适配、在线状态和生命周期日志
    evaluation/     # TrackEval 调用与结果读取
    analysis/       # 事件归因、分桶、可视化和配对汇总
  scripts/          # 经实现和验收的阶段入口
  configs/          # 不含机器绝对路径的可提交配置
  manifests/        # 数据、权重、上游 SHA、检测缓存和划分清单
  reports/          # 体积小、不可变的实验结果与产物哈希快照
  tests/            # 坐标、因果性、梯度、缓存和评测关键用例
  docs/             # 方案、协议、决策与实验结论
  third_party/      # 固定 SHA 的外部源码；本地存在，Git 忽略
  local/            # 机器路径、环境快照和 smoke 结果；Git 忽略
```

## 仓库外数据根目录

本机当前使用 `/media/imc/data/mot-data`：

```text
mot-data/
  downloads/dancetrack/       # 原始 ZIP、revision 与 SHA256
  datasets/dancetrack/        # train/val 图像与 GT
  weights/dancetrack/         # 权重和本地来源记录
  cache/detections/           # 固定检测候选
  cache/motion/               # 轨迹样本与统计量
  manifests/                  # 完整的本机数据校验结果
  runs/<run_id>/              # 不覆盖的实验产物
```

代码、可复现元数据和轻量结果凭证进入 Git；图片、ZIP、权重、缓存、checkpoint、完整 tracker 输出和大日志不进入 Git。配置使用逻辑键或环境变量，不提交服务器绝对路径。

## Run 目录

每个真实运行使用唯一 `run_id`：

```text
runs/<run_id>/
  config/              # 解析后的完整配置
  provenance/          # Git SHA、环境、数据/权重/检测 hash
  logs/                # stdout/stderr 与时间记录
  checkpoints/         # 模型状态；若该 run 产生模型
  trackers/            # MOTChallenge 格式输出
  evaluation/          # TrackEval 原始输出
  summaries/           # 逐视频与总体轻量摘要
  artifacts/           # 可视化和失败样例
```

建议 `run_id` 采用 `<stage>_<method>_<split>_s<seed>_<YYYYMMDD-HHMMSS>`。run 必须对应预先登记的配置；失败运行也保留状态和错误原因。

## 第三方代码

`third_party/` 只保存本机 checkout，不嵌入本仓库历史。每个依赖的仓库 URL、commit、许可证、补丁和用途写入 `manifests/upstreams/`。正式运行不得跟随外部仓库浮动分支。

## GPU 约束

物理 GPU 0 永久保留给 Ubuntu 图形界面。项目任务必须显式选择物理 GPU 1–7，并在 run provenance 中记录 `CUDA_VISIBLE_DEVICES` 与物理编号。
