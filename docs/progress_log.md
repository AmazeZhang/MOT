# 项目进展日志

本日志用于 Git 审查和阶段交接。只记录已经执行并核验的工作；计划、工程 smoke 指标和正式研究结果必须明确区分。后续工作按时间追加，不覆盖失败记录。

## 2026-11-10：仓库、环境和数据准备

### 已完成

- 将仓库同步到 `ed98b38`，远程使用 `git@github.com:AmazeZhang/MOT.git`。
- 写入 GPU 安全规则：物理 GPU 0 永久保留给 Ubuntu 图形界面，项目任务只能显式选择物理 GPU 1–7。
- 建立 `mot-core`：Python 3.10、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、NumPy 1.26.4、SciPy 1.13.1。
- 使用 `PYTHONNOUSERSITE=1` 隔离服务器已有的用户级 Python 包。
- 在物理 GPU 1 上通过矩阵反传、GRU 反传和 CUDA NMS 验收。
- 将 DanceTrack train/val 下载到 `/media/imc/data/mot-data`，固定数据 revision `a0ba42ac690c41e9850a20e76a0b9450a6fb6a47`。
- 三个 ZIP 完整性和 SHA256 检查通过；解压后 train 为 40 段/41,796 帧，val 为 25 段/25,508 帧。
- 数据结构、连续帧号、图像尺寸和 GT 基本约束检查通过，0 个错误。

### 验收与产物

- 本机环境摘要：`local/environment/setup-summary.md`（Git 忽略）。
- 可提交依赖锁：`configs/environments/locks/`，包含 `mot-core`、`mot-det`、`mot-eval` 的完整 `pip freeze` 快照。
- GPU 验收：`local/environment/gpu_smoke.json`（Git 忽略）。
- 可提交数据清单：`manifests/datasets/dancetrack.yaml`。
- 完整数据校验：`/media/imc/data/mot-data/manifests/dancetrack_validation.json`。

### 限制

- 本阶段只证明环境和数据可用，没有产生训练或方法效果结果。

## 2026-11-10：工程结构与 TrackEval

### 已完成

- 落地 `src/`、`scripts/`、`configs/`、`manifests/`、`tests/`、`third_party/` 和 `local/` 的职责边界。
- 固定官方 TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`，建立独立 `mot-eval` 环境。
- 保存 NumPy 标量别名兼容补丁；补丁不改变指标公式。
- 修复该固定 commit 的 CLI 对 `SEQMAP_FILE`、`OUTPUT_FOLDER` 两个路径参数的列表解析问题。
- 建立完美跟踪、一次漏检、身份交换三个 MOTChallenge 格式人工例并加入自动测试。

### 验收与产物

- 自动测试：`1 passed`。
- 人工例结果：完美例 HOTA/AssA/IDF1 为 1；漏检例 HOTA/AssA 为 0.5；交换例 IDSW 为 2。
- TrackEval 来源与补丁：`manifests/upstreams/trackeval.yaml`、`manifests/upstreams/patches/`。
- 本机结果：`local/evaluation/trackeval_sanity.json`（Git 忽略）。

### 限制

- 人工例只验证指标与适配器行为，不代表真实跟踪性能。

## 2026-11-10：OC-SORT、检测环境和 50 帧闭环

### 已完成

- 核对 OC-SORT 历史，固定论文时代 commit `bb2f6867a41082558d95ab47a854a3918de7831b`，避开 2026 Python tracker 重构。
- 建立独立 `mot-det` 环境，未编译当前流程不需要的 YOLOX C++ COCO 扩展。
- 从作者 Hugging Face 仓库 revision `6487fc12a2fe13db44506592324cea83270f5812` 下载 DanceTrack YOLOX/ByteTrack 权重。
- 权重 SHA256 为 `b8d1afba08f801f3fe2cb122faf3fc9af6c7856405a0da94cf91cdd5eb9b3321`；894 个模型张量以 strict 模式加载，缺失/多余键均为 0。
- 在物理 GPU 1 上完成单帧 FP32 推理和 CUDA NMS。
- 对 `dancetrack0027` 第 1–50 帧完成图像→YOLOX→固定检测 CSV→OC-SORT→缓存重放→TrackEval 闭环。
- GPU 0 拒绝测试通过。

### 验收与产物

- 固定检测：333 个；tracker 输出：296 行。
- 缓存重放的帧号和轨迹 ID 完全一致；序列化框最大误差约 `5e-6`，小于登记容差 `1e-5`。
- 50 帧 GT 子集 TrackEval：HOTA 76.38、AssA 71.583、IDF1 82.78、MOTA 86.944、IDSW 3。
- 运行摘要：`/media/imc/data/mot-data/runs/g1_ocsort_smoke_dancetrack0027_f1-50_20261110/summary.json`。
- 可提交配置与 provenance：
  - `configs/environments/mot-det.yaml`
  - `manifests/upstreams/oc_sort.yaml`
  - `manifests/weights/dancetrack_bytetrack.yaml`
  - `manifests/detections/ocsort_yolox_dance_smoke50.yaml`
  - `scripts/run_ocsort_smoke.py`

### 限制

- 以上指标来自单个 train 序列的前 50 帧，只是工程 smoke test，不是论文基线或研究结果。
- Hugging Face 权重与 OC-SORT Google Drive 文件是否完全相同、权重训练范围是否只包含 train，尚未完成独立核验；在此之前正式 val 主张保持阻断。
- 全序列检测缓存、三次重复性和完整 train/val 基线尚未运行。

## 2026-11-10：完整单序列基线与重复性

### 已完成

- 使用物理 GPU 1 对 `dancetrack0027` 完整 403 帧执行 FP32 YOLOX 检测，生成不可覆盖的原图坐标检测缓存。
- 从固定缓存运行 OC-SORT；直接运行与首次缓存重放的帧号和轨迹 ID 一致，最大框差约 `2.60e-6`。
- 新增 CPU-only `scripts/replay_ocsort_cache.py`，验证缓存 schema、帧范围、重复 det ID、框尺寸和分数，并拒绝覆盖旧输出。
- 从同一个检测缓存独立重放 OC-SORT 三次；三份 tracker 文件逐字节一致。
- 使用完整 403 帧 GT 和固定 TrackEval 评测该序列。
- 新增空检测输入测试，确认论文版 OC-SORT 的 `None` 会跳过时间推进，而形状 `[0,5]` 的空数组会正常推进 `frame_count` 和轨迹年龄。
- 生成第 1、100、200、300、403 帧叠加图并人工抽查；GT、固定检测和 tracker 框没有整体偏移或重复缩放。
- 从 OC-SORT Model Zoo 的官方 Google Drive 下载 `ocsort_dance_model.pth.tar`，与 Hugging Face `bytetrack_model.pth.tar` 完成逐字节比较。

### 验收与产物

- 固定检测：3,431 个；空检测帧：0；tracker 输出：3,082 行。
- 检测缓存 SHA256：`81e29bdfbbae5255d3ac9628fba0ee1311f4d356a03be56a5c25e80172cc1564`。
- 三次独立重放共同输出 SHA256：`a9d82dbc5e53c4b03a0e57d78c88ee7d53a01e0ab8ec4daf135f4ba04d5022a1`。
- 完整序列 TrackEval：HOTA 68.453、DetA 81.052、AssA 57.916、IDF1 72.284、MOTA 88.273、IDSW 13。
- 空帧语义自动测试：`1 passed`；原始 baseline 重放显式使用 `upstream_none`，未来研究接口必须单列 `advance_empty`。
- 可视化目录：`/media/imc/data/mot-data/runs/g1_ocsort_baseline_dancetrack0027_full_20261110/artifacts/overlays/`。
- 官方 Google Drive 与 Hugging Face 权重均为 792,835,731 bytes，SHA256 同为 `b8d1afba08f801f3fe2cb122faf3fc9af6c7856405a0da94cf91cdd5eb9b3321`，`cmp` 逐字节一致，894 个模型张量全部相等。
- 运行摘要：`/media/imc/data/mot-data/runs/g1_ocsort_baseline_dancetrack0027_full_20261110/summary.json`。
- 可提交 provenance：`manifests/detections/ocsort_yolox_dance_dancetrack0027_full.yaml`。

### 限制

- 这是一个 train 序列的完整闭环验收，不代表 DanceTrack val 总体基线，更不是新方法结果。
- 该序列没有自然空检测帧；空帧差异已由合成单元测试确认，但尚未在真实序列事件中触发。
- 权重等价性已解决。官方训练说明支持“DanceTrack train set”，但 checkpoint 内没有嵌入数据 manifest；正式报告必须保留此 provenance 限制。

## 2026-11-10：完整 train 固定检测与 G1 基线

### 已完成

- 新增 `scripts/export_dancetrack_split.py`：模型只加载一次，每段重新初始化 tracker；缓存、tracker 和摘要逐段原子落盘，支持已完成段哈希验签后续跑，拒绝覆盖、残缺成品和遗留 `.partial`。
- 新增 `scripts/validate_ocsort_split_replay.py` 与 `scripts/write_dancetrack_seqmap.py`，分别负责全 split 的 CPU 固定缓存重放和确定性 seqmap 生成。
- 使用 `CUDA_VISIBLE_DEVICES=1` 在物理 GPU 1 上一次完成全部 40 段 train 推理；物理 GPU 0 未使用，运行后 GPU 1 已释放。
- 对 40 个序列逐一校验图像连续性和尺寸、缓存与 tracker SHA256、设备字段和总计数；没有残留 `.partial`。
- 使用固定缓存在 CPU 上重放 40 段 OC-SORT，并逐行比较直接输出与重放输出的帧号、ID 和重建 XYXY。
- 使用固定 TrackEval commit 对全部 40 段 train GT 做 HOTA、CLEAR 和 Identity 评测。
- 运行 `experiment-audit` 独立语义审计；reviewer 因账号 usage limit 未返回判断，已按规则记录为 `REVIEW_UNAVAILABLE`，没有由执行者自行替代为 PASS。

### 验收与产物

- 完整 train：40 段、41,796 帧、355,231 个固定检测、331,420 行 tracker 输出、0 个自然空检测帧。
- 检测核心推理共 1,241.67 秒；端到端墙钟 4,180.65 秒。设备记录统一为物理 GPU 1 / `CUDA_VISIBLE_DEVICES=1` / 进程内 `cuda:0`。
- `dancetrack0027` 回归：批量缓存与原单序列缓存逐字节相同；批量 tracker 与原直接输出逐字节相同。
- 正式缓存重放：40/40 段、331,420 行的帧号和 ID 完全一致；最大 XYXY 误差 `3.9e-5` 像素，小于固定容差 `1e-4`。
- 两次 `1e-5` 尝试在 `dancetrack0015` 停止，观察误差分别为 `1.4e-5`（XYWH）和 `1.5e-5`（重建 XYXY）；失败目录保留。全量 `1e-3` 探针只用于定界，不作为正式验收。
- train TrackEval 总体：HOTA 62.203、DetA 85.665、AssA 45.212、IDF1 61.779、MOTA 93.945、IDSW 1,796。
- 低 HOTA/AssA 重点序列为 `dancetrack0044`、`dancetrack0012`、`dancetrack0037`；最高 IDSW 序列前三为 `dancetrack0044`、`dancetrack0020`、`dancetrack0012`。
- 主运行：`/media/imc/data/mot-data/runs/g1_ocsort_train_baseline_20261110/`。
- 正式重放：`/media/imc/data/mot-data/runs/g1_ocsort_train_replay_validation_1e-4_20261110/`。
- 可提交 provenance：`manifests/detections/ocsort_yolox_dance_train.yaml`。
- 可提交轻量证据：`reports/baselines/g1_ocsort_train_20261110/`，包含官方总体摘要、精简逐序列指标、缓存/tracker 哈希索引和重放验收索引。
- 审计状态：`EXPERIMENT_AUDIT.md`、`EXPERIMENT_AUDIT.json`；独立语义 verdict 不可用，确定性文件/哈希/范围检查已完成。
- 验证命令：新脚本 `py_compile` 通过；`tests/test_ocsort_empty_frame_semantics.py` 为 `1 passed`；`PYTHONPATH=src tests/test_trackeval_sanity.py` 为 `1 passed`；YAML/主 JSON 解析和 `git diff --check` 通过。

### 限制

- 这些是完整 **train** 基线诊断，不是 val 指标、泛化结果或新方法提升。
- 当前只有一个确定性 detector/tracker 配置，没有随机种子研究；高 DetA、低 AssA 只支持“关联是主要诊断方向”，不证明任何待实现方法有效。
- `1e-4` 是原图像素 XYXY 的重放数值容差；正式结果的帧号和 ID 是精确一致，不应把容差描述为身份层面的近似。
- 独立 `experiment-audit` reviewer 因 usage limit 不可用，完整性接受状态保持 unavailable；不得表述为独立审计 PASS。
- checkpoint 没有内嵌训练数据 manifest，仍需保留上游 provenance 限制。

## 下一步

1. 写入 val 冻结记录：沿用 train 的权重、FP32 输入、检测/NMS/OC-SORT 参数、`upstream_none` 语义、TrackEval 配置和一次性查看规则，不根据 val 调参。
2. 使用物理 GPU 1 一次生成 25 段 val 固定检测和直接 OC-SORT 输出，完成相同的哈希审计与 CPU 重放；继续禁止物理 GPU 0。
3. 一次性运行完整 val TrackEval，记录总体与逐序列结果；它将是 G1 官方验证基线，但仍不是新方法结果。
4. G1 val 完成后进入 G2，优先分析 train 中 `0044`、`0012`、`0020` 等失败段的检测漏失、近邻交叉和历史 ID 污染，不直接开始 Flow Matching。
