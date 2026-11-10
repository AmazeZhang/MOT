# 数据、检测缓存与评测

核查日期：2026-11-10。先准备 DanceTrack train/val；不需要立即下载 MOT17、MOT20、CrowdHuman 或 SportsMOT。读取官方仓库的检测器训练教程，不代表我们需要重训检测器。

## 1. 官方入口与第一批下载

- [DanceTrack 官方仓库](https://github.com/DanceTrack/DanceTrack)
- [官方仓库指向的数据镜像](https://huggingface.co/datasets/noahcao/dancetrack/tree/main)
- [官方仓库指向的权重镜像](https://huggingface.co/noahcao/dancetrack_models/tree/main/bytetrack_models)
- [官方 TrackEval](https://github.com/JonathonLuiten/TrackEval)

当前镜像可见 `train1.zip`（约 3.61 GB）、`train2.zip`（约 3.30 GB）、`val.zip`（约 4.21 GB）；第一批压缩包约 11.1 GB。`test1.zip`、`test2.zip` 暂缓。压缩包大小不是解压后占用；磁盘还需容纳权重、缓存和输出。

官方划分是 40 段 train、25 段 val、35 段 test，test 标签不公开。官方 README 的新公告将测试提交迁移到 CodaBench，但旧章节仍有 CodaLab 链接；后续按当前公告与实际比赛页面执行。本轮不提交 test。

### 下载命令示例

以下只负责下载，不自动解压，也不安装本项目模型。先选择服务器有配额且可写的绝对路径，替换 `/path/to/mot-data`。下载工具版本安装后记录到环境清单；它不属于训练算法依赖。

```bash
export MOT_DATA_ROOT=/path/to/mot-data
mkdir -p "$MOT_DATA_ROOT/downloads/dancetrack"
python -m pip install huggingface_hub
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
out = Path(os.environ['MOT_DATA_ROOT']) / 'downloads' / 'dancetrack'
repo = 'noahcao/dancetrack'
revision = HfApi().repo_info(repo_id=repo, repo_type='dataset').sha
files = ['train1.zip', 'train2.zip', 'val.zip']
(out / 'download_manifest.json').write_text(json.dumps(
    {'repo': repo, 'revision': revision, 'files': files}, indent=2))
snapshot_download(repo_id=repo, repo_type='dataset', revision=revision,
                  allow_patterns=files, local_dir=str(out))
print('downloaded snapshot:', revision)
PY
```

首次取到的 revision 要记录；断线续传时使用已记录的 revision，不每次切到新的 main。下载失败直接反馈原错误，不切换未经确认的数据源。HF 下载/本地目录机制见 [官方工具说明](https://huggingface.co/docs/huggingface_hub/guides/cli)。

### 解压前后检查

```bash
unzip -t "$MOT_DATA_ROOT/downloads/dancetrack/train1.zip"
unzip -t "$MOT_DATA_ROOT/downloads/dancetrack/train2.zip"
unzip -t "$MOT_DATA_ROOT/downloads/dancetrack/val.zip"
unzip -l "$MOT_DATA_ROOT/downloads/dancetrack/train1.zip" | head -n 25
sha256sum "$MOT_DATA_ROOT/downloads/dancetrack/"*.zip
```

先查看三个包的顶层目录，再决定解压目标，防止生成 `dancetrack/dancetrack/train` 或覆盖已有文件。保留每包校验结果、下载 revision 和实际组织方式。若发布方没有给 checksum，本地 SHA256 用于识别本次文件，不冒充独立来源真实性校验。

目标目录：

```text
MOT_DATA_ROOT/
  downloads/dancetrack/         # 原始 zip 与下载清单
  datasets/dancetrack/
    train/dancetrackXXXX/{img1/,gt/gt.txt,seqinfo.ini}
    val/dancetrackXXXX/{img1/,gt/gt.txt,seqinfo.ini}
  weights/dancetrack/           # 权重与来源清单
  cache/detections/<detector_id>/<split>/
  cache/motion/<dataset_version>/
  runs/<run_id>/
```

不能通过重编号把缺失帧“补齐”。校验实际图片数量、`seqLength`、图像大小、frameRate、帧号范围、GT 框正宽高、重复 frame/id、损坏图像和 seqmap。首尾帧及随机样本可视化；如果存在合法越界框，记录并按统一策略处理，不静默改变原 GT。

数据用于研究并遵循官方条款；本仓库只记录下载入口、清单和校验和，不上传图片、压缩包或权重。

## 2. 权重选择是一项独立验收

目前核实镜像中的文件为 `bytetrack_models/bytetrack_model.pth.tar`（约 793 MB），但**文件存在不等于已确认其训练划分和 OC-SORT 配置完全一致**。先记录来源、revision、SHA256、模型结构、输入尺寸、训练数据，再验证短视频结果。

优先使用能够确认仅用 train（及公开声明的外部数据）训练、未用 val 的 DanceTrack 检测权重。[OC-SORT Model Zoo](https://github.com/noahcao/OC_SORT/blob/master/docs/MODEL_ZOO.md) 和 [运行说明](https://github.com/noahcao/OC_SORT/blob/master/docs/GET_STARTED.md) 作为对照。两处权重是否相同要靠校验和与说明确认，不靠文件改名。

如训练来源不明确，可先跑工程 smoke test，但不能据此发布正式 val 比较结论。当前不要求你重训 YOLOX；把权重来源问题反馈后再决定。

## 3. 固定检测缓存

缓存必须来自检测器的原始候选，而非跟踪器输出：跟踪输出已经做过过滤、匹配与轨迹管理，不能当作公平检测池。

计划采用每序列 CSV：

```text
frame,det_id,x1,y1,x2,y2,score,class_id
```

约定：frame 从 1 开始，det_id 在 frame 内唯一；像素坐标为原图坐标，宽高采用连续 `x2-x1`、`y2-y1`，不混用 `+1`；空检测帧由序列元数据显式保留。原数据坐标原点是否需适配，必须按上游代码与可视化核实一次，并固化转换。缓存交给 OC-SORT 时避免重复缩放。

检测元数据记录：代码 commit、权重 hash、输入尺寸/resize/padding、精度、类别映射、置信度计算（YOLOX objectness×class）、导出阈值、NMS 阈值、候选保留策略。低分检测必须保留到所选跟踪器需要的下界，不能先按高分阈值删掉再做两阶段匹配。

缓存内容哈希与 manifest 作为全部 run 的共同输入标识；新阈值或 NMS 构成新版本。数据导出后按原图与框叠加图检查。GT 匹配标签单独保存，推理检测缓存中不携带 GT ID。

## 4. 训练划分与缺失构造

从 train 的视频 ID 创建 motion-train / motion-dev 清单，建议约 32/8，固定 split_seed=20261110；若同源内容成组，按组隔离。清单与 hash 提交 Git，暂不在未下载数据前虚构具体 ID。

归一化、模型拟合和任何数据驱动阈值只使用 motion-train。参数选择使用 motion-dev，最终方案锁定后可在完整 train 重训，再比较完整官方 val。主结果不在 val 上选出“最好看的种子”。

目标键使用 `(dataset, sequence, gt_id)`，不能跨视频按整数 ID 拼轨迹。样本记录历史帧号、真实观测掩码、时间间隔、目标帧、归一化版本。GT 只提供标签；历史输入使用实际检测，推理使用自身已关联历史。

缺失长度同时报告帧数与真实时间。合成事件先固定清单：序列、目标、起止帧、删除哪些对应检测、事件种子；全部模型使用同一清单。删除检测不等于删除图片，也不改变其他目标的候选；自然未扰动主表独立保留。避免把“每隔 4 帧采样”误写成“连续漏检 4 帧”。

## 5. TrackEval 的目录与命令

在实现阶段 clone 并固定上游 SHA；下面是服务器准备命令，不在本轮执行：

```bash
mkdir -p third_party
git clone https://github.com/JonathonLuiten/TrackEval.git third_party/TrackEval
git -C third_party/TrackEval rev-parse HEAD
```

首次冻结 SHA 后，各机器 checkout 同一个版本，保存补丁。`third_party/` 已被 Git 忽略；在后续来源清单中记录 SHA，不嵌套提交整个仓库。

seqmap 文本首行为 `name`，后面逐行序列名，不带 `.txt`。每个被评价序列都要有 tracker txt；无输出也需空文件。拟定结果布局：

```text
MOT_DATA_ROOT/runs/<run_id>/
  trackers/baseline/data/dancetrackXXXX.txt
  evaluation/
  seqmap.txt
```

tracker 文件每行 10 列：`frame,id,left,top,width,height,confidence,-1,-1,-1`。GT 文件的列含义与 tracker 输出不同，不把 GT 后三列直接拷成跟踪结果。帧号和 ID 均为整数；同帧同 ID 不可重复。

在仓库根目录、已验收的 `mot-eval` 环境中，设置真实路径后可使用：

```bash
export MOT_RUN_DIR="$MOT_DATA_ROOT/runs/your_run_id"
python third_party/TrackEval/scripts/run_mot_challenge.py \
  --BENCHMARK MOT17 --SPLIT_TO_EVAL val \
  --GT_FOLDER "$MOT_DATA_ROOT/datasets/dancetrack/val" \
  --SEQMAP_FILE "$MOT_RUN_DIR/seqmap.txt" \
  --SKIP_SPLIT_FOL True \
  --TRACKERS_FOLDER "$MOT_RUN_DIR/trackers" \
  --TRACKERS_TO_EVAL baseline --TRACKER_SUB_FOLDER data \
  --OUTPUT_FOLDER "$MOT_RUN_DIR/evaluation" \
  --METRICS HOTA CLEAR Identity \
  --USE_PARALLEL False --PLOT_CURVES False
```

`BENCHMARK MOT17` 在这里用于该通用 MOT 格式适配器的兼容配置，数据集仍然是显式路径中的 DanceTrack；结果登记必须写 DanceTrack。读取输出配置，确认 GT 路径、seqmap、帧数、类别与预处理设置。初期串行利于排错，验证后再并行。此目录参数依据 [TrackEval 适配器](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py) 和 DanceTrack 官方示例，仍需服务器实际验收。

保留 stdout、配置、per-sequence 结果和总体结果。主实验严格在线，不运行插值/平滑；带离线后处理的公开论文数值单列。序列聚合使用 TrackEval，不用自己简化的 HOTA 实现。
