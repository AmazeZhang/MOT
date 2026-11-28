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

## 2026-11-11：冻结的一次性 val G1 基线

### 已完成

- 在查看 val 指标前写入 `configs/baselines/ocsort_dancetrack_g1.yaml` 和 `manifests/protocols/g1_dancetrack_val_freeze_20261111.yaml`，冻结权重、FP32 非融合检测、输入尺寸、检测/NMS、OC-SORT、空帧语义、TrackEval 和一次性查看规则。
- 冻结时配置 SHA256 为 `d967afa287e783b2004f6e313f76148f56ebd433b98b2a28dc5744099ef2b768`，协议 SHA256 为 `21c2336a409d61816ce5c821a4c9b10a4d104493b62db7a78c926f9d3ba0957b`；结果导出后复核未变化。
- 使用物理 GPU 1 一次完成 25 段 val 固定检测和直接 OC-SORT；没有工程失败、重试或结果后调参。物理 GPU 0 未使用，运行后 GPU 1 已释放。
- 对全部缓存、tracker、段摘要及其 SHA256 做完整审计；随后在 CPU 上从固定缓存独立重放并比较每一行帧号、ID 和原图 XYXY。
- 使用冻结的 TrackEval commit 和配置只运行一次完整 val 指标评测。
- 对照固定 OC-SORT 源码中的 `GET_STARTED.md`、`MODEL_ZOO.md` 和默认参数：tracker 默认参数一致，指标落在官方记录范围；同时保留运行精度/融合路径不同的限制。

### 验收与产物

- 完整 val：25 段、25,508 帧、238,084 个固定检测、202,711 行 tracker 输出、0 个自然空检测帧。
- 检测核心推理共 758.85 秒；整段墙钟 2,760.03 秒。设备记录统一为物理 GPU 1 / `CUDA_VISIBLE_DEVICES=1` / 进程内 `cuda:0`。
- 产物审计为 25/25 缓存、tracker 和段摘要全部存在且哈希匹配，无 `.partial`；缓存共 16,582,629 bytes，tracker 共 12,682,155 bytes。
- CPU 重放为 25/25 段、202,711 行帧号和 ID 精确一致；最大 XYXY 误差 `3.2e-5` 像素，小于冻结的 `1e-4` 容差。
- val TrackEval 总体：HOTA 52.059、DetA 77.400、AssA 35.130、IDF1 51.623、MOTA 87.265、IDSW 1,936。
- 上游文档给出的 HOTA 范围为 52.1--52.6，Model Zoo 为 HOTA 52.1、IDF1 51.6、AssA 35.3、MOTA 87.3、DetA 77.2；本次结果按展示精度复现该范围。上游命令示例使用 FP16+fuse，本次冻结运行使用 FP32+unfused，因此不能声称命令路径或输出逐比特相同。
- 主运行：`/media/imc/data/mot-data/runs/g1_ocsort_val_baseline_20261111/`；正式重放：`/media/imc/data/mot-data/runs/g1_ocsort_val_replay_validation_1e-4_20261111/`。
- 可提交 provenance：`manifests/detections/ocsort_yolox_dance_val.yaml`；可提交轻量证据：`reports/baselines/g1_ocsort_val_20261111/`。
- 验证结果：全部项目 YAML 可解析；4 个 val 主证据哈希、2 个冻结哈希、25 段产物索引、25 段重放索引和 26 行逐段/总体指标投影交叉检查通过；相关脚本 `py_compile`、`git diff --check` 通过；两项既有 pytest 各为 `1 passed`。

### 限制

- 这是冻结配置下的 G1 验证参考基线，不是新方法结果，也不证明 Flow Matching 或任何候选预测器有效。
- 当前只有一个确定性 detector/tracker 配置，不是随机种子研究，也不能支持 DanceTrack val 之外的鲁棒性主张。
- 逐序列 val 指标只为透明报告，不得用于方法、超参数或序列特定规则选择；后续失败归因和设计选择必须先在 train 内完成。
- checkpoint 没有嵌入训练数据 manifest；虽然权重与上游官方文件逐字节相同，正式报告仍需保留 provenance 限制。
- 独立语义审计此前因 reviewer usage limit 为 `REVIEW_UNAVAILABLE`；确定性哈希、范围与重放检查通过不等于独立审计 PASS。

## 2026-11-11：G2 train-only 检测缺失恢复事件提取

### 已完成

- 在派生统计前冻结 `manifests/protocols/g2_train_failure_attribution_20261111.yaml`，SHA256 为 `a825161d9800180559012b2b2cee5117e744e8d2aac31d1e2739d01a7dee6645`；只选用 G1 train 已登记的 `0012`、`0020`、`0037`、`0044`，未读取 val 做阈值或样本选择。
- 新增 `src/mot_research/analysis/failure_events.py` 和 `scripts/analyze_g2_failure_events.py`，提取同一 GT 连续有标注、检测一对一匹配短时不可用、随后重新匹配的有界事件；端点 tracker ID 分为身份改变、身份保持对照和无法判定。
- 检测候选固定为分数严格大于 0.6，GT--检测及 GT--tracker 的诊断匹配阈值固定为 IoU 0.5，缺失长度固定为 1--30 帧；这些事件不是官方 TrackEval IDSW 事件标签。
- 使用种子 20261111 从身份改变与身份保持事件各抽 20 个；不足时保留全部的规则已预先冻结，本次两层均无短缺。
- 新增 `scripts/render_g2_event_overlays.py`，为 40 个抽样事件生成同一裁剪区域的缺失前、缺失段首尾和重新出现后审阅图，并记录逐图 SHA256。
- 全阶段设置 `CUDA_VISIBLE_DEVICES=`，只用 CPU；物理 GPU 0 未使用。

### 验收与产物

- 四段共提取 797 个有界事件：身份改变 53、身份保持 112、端点无法判定 632。已判定端点中的身份改变比例为 53/165（32.1%），但这是刻意选择的困难 train 序列，不是总体失败率。
- 分段事件数：`0012` 为 168、`0020` 为 334、`0037` 为 89、`0044` 为 206；对应身份改变数分别为 16、12、10、15。
- 身份改变组的缺失长度中位数为 2 帧，身份保持组为 1 帧；这里只是描述统计，事件来自相同视频和身份，不能作为独立样本推断。
- 邻居框重叠代理在身份改变/保持组均为 100%；历史 ID 污染代理分别为 96.2%/89.3%。它们在两组都高度常见，不能据此声称是失败原因。
- 40 张审阅图均生成并哈希；4 张先用于渲染/坐标抽查，随后 40 张全部由 Codex 逐图审阅并记录可见性、候选类型、端点身份观测、画面模式和置信度。该记录不是人类专家标注，也不是因果真值。
- 事件审计通过：797 个 event ID 唯一，时序与缺失长度约束成立；20+20 样本可由冻结种子重建；4 段 GT、检测和 tracker 源文件哈希匹配，检测/tracker 哈希与 G1 train 凭证一致。
- 首轮合成测试为 `3 passed`；后续候选竞争用例加入后为 `4 passed`。分析和渲染脚本 `py_compile` 通过。分析环境为 `mot-eval`，渲染环境为已验收的 `mot-det`（OpenCV 4.10.0、SciPy 1.13.1）。
- 原始运行：`/media/imc/data/mot-data/runs/g2_train_failure_attribution_20261111/`；可提交轻量凭证：`reports/diagnostics/g2_train_failure_attribution_20261111/`。

### 限制

- “无一对一匹配检测”不等于原始候选池完全没有框；拥挤时可能是候选竞争。报告保留原始最佳 IoU，后续必须单列候选缺失与匹配竞争。
- 框重叠仅是公开几何代理，不是真实遮挡标注；代理共现不能建立因果关系。
- 632 个事件因一个或两个端点没有 tracker 匹配而无法判定，不能从主对照中静默删除；它们已完整计数并保留在数据盘。
- 这是 train-only 探索性诊断，未改变 tracker，也没有产生新方法、性能提升或 val 结论。

## 2026-11-11：G2 候选竞争分解与固定样本视觉审阅

### 已完成

- 在逐帧分解前冻结 `manifests/protocols/g2_candidate_contention_addendum_20261111.yaml`，SHA256 为 `193caf3552f33d9399ee3721e7c02cec3d9770b060365541fc39e445f698e867`；它引用父事件哈希并明确是事后探索性补充，不回改父协议。
- 新增 `scripts/analyze_g2_candidate_contention.py`，对每个父事件缺失帧重新执行相同的一对一匹配，并强制断言目标仍未匹配。
- 若没有 score > 0.6 且目标 IoU≥0.5 的原始候选，记为 `no_raw_candidate`；若存在候选但一对一分配后目标未匹配，记为 `candidate_contention`。
- 逐图审阅固定的 20 个身份改变和 20 个身份保持样本；`manual_review.csv` 明确记录 reviewer 为 Codex，并保留 5 个低置信度失败片段和 4 个低置信度对照片段。

### 验收与结果

- 797 个事件全部保留：385 个 `contention_only`、305 个 `absence_only`、107 个 `mixed`；共 2,367 个缺失帧，其中候选竞争 1,291 帧，原始候选缺失 1,076 帧。
- 身份改变事件中 45/53（84.9%）包含候选竞争；身份保持对照中 97/112（86.6%）包含候选竞争。两组都高度常见，候选竞争的存在不能单独解释身份改变。
- 固定 20+20 样本中，身份改变为 12 contention-only、4 mixed、4 absence-only；身份保持为 17 contention-only、3 absence-only。抽样没有 mixed 对照，但这是冻结抽样的结果，不能事后补选。
- Codex 视觉审阅中，失败/对照的目标可见性分别为：clear 4/4、partial 11/14、not-visually-resolvable 5/2。样本按 outcome 分层且来自 4 个困难视频，不能据此估计总体比例或统计显著性。
- 两组都以密集重叠、相似外观交叉、快速姿态变化或龙道具遮挡为主，没有出现可作为单一失败原因的清晰视觉分界。
- 分解审计通过：797 个父 event ID 一一对应，2,367 个逐帧分类与缺失长度总和一致，协议/父事件/GT/检测哈希匹配；运行全程 CPU，未使用物理 GPU 0。
- 原始运行：`/media/imc/data/mot-data/runs/g2_candidate_contention_20261111/`；轻量凭证：`reports/diagnostics/g2_candidate_contention_20261111/`。

### 限制

- 候选竞争是使用 GT 的诊断分类，线上 tracker 不知道候选属于哪个 GT；存在 IoU≥0.5 候选不等于检测身份正确或实际可关联。
- Codex 视觉审阅不是人类专家标注；`primary_visual_pattern` 只描述画面现象，不是遮挡、检测或运动的因果标签。
- 20+20 抽样为了对照而按 outcome 分层，不能用于估计真实事件发生率；同一视频和身份内的事件也不是独立样本。
- 结果继续反对直接跳到 Flow Matching：当前失败与成功都受到候选竞争和密集交互影响，尚未隔离纯运动预测的可改善空间。

## 下一步

## 2026-11-11：G2 D2 当前 GT 位置首轮关联反事实

### 已完成

- 初版协议 `g2_d2_gt_position_first_association_20261111.yaml` 在结果产生前冻结，但错误要求 CPU replay tracker 文本逐字节等同 GPU 直接输出。
- 前两次启动均未产生 D2 结果：第一次因误用不含 `filterpy` 的 `mot-eval` 环境停止；第二次在 `dancetrack0012` 后因初版过严验收停止，`.partial` 均自动删除。
- 未修改初版协议，新增替代协议 `g2_d2_gt_position_first_association_v2_20261111.yaml`，SHA256 为 `4f193c6ae3f3049a48c7a5ac4fec7a725a82dba1dc50f6092f1a0409d9776923`。唯一协议修正是沿用 G1 已接受的帧号/ID 精确一致、原图 XYXY 误差≤`1e-4`；D2 反事实定义和阈值未改。
- 新增 `src/mot_research/analysis/d2_first_association.py` 与 `scripts/analyze_g2_d2_first_association.py`，旁路捕获 OC-SORT 首轮关联输入/输出。
- 对每个已判定事件独立复制同一基线帧状态，只把缺失前原 track 的预测 XYXY 替换为当前 GT XYXY；检测池、分数、其他 track、速度/方向代价、Hungarian、阈值、二次匹配和生命周期均不变，oracle 结果不写回。

### 验收与结果

- 165 个已判定父事件全部适用：目标正检测存在，且缺失前原 track 在重新出现帧仍存活；632 个 unresolved 父事件按冻结协议不进入 D2。
- 四段 CPU 基线重放的帧号和 track ID 全部精确一致，最大 XYXY 误差 `8.0e-6`，小于 `1e-4`；对应 reference tracker 哈希与 G1 train 凭证一致。
- 所有事件中，基线首轮正匹配为 112/165，GT 位置 oracle 为 132/165；新增可恢复 22，受损 2。这里只是独立首轮关联计数，不是净跟踪性能。
- 53 个身份改变事件：基线正匹配 1，oracle 正匹配 23，22 个新增可恢复（41.5%），30 个仍未恢复。
- 112 个身份保持对照：基线正匹配 111，oracle 正匹配 109；2 个原本正确的事件被 oracle 全局分配破坏，另有 1 个基线/Oracle 均未正匹配。
- 22 个可恢复失败按候选类型为 contention-only 15、mixed 4、absence-only 3；它们的缺失长度中位数为 2 帧。
- 30 个 oracle 未恢复失败中，正检测全部被其他 track 占用；原 track 24 次未匹配、6 次匹配错误检测，且 25/30 的全局匹配对集合没有因理想位置框改变。
- 这表明完美当前位置对部分事件有明显首轮关联空间，但多数剩余失败已受历史身份和全局竞争约束，不能仅靠更强单轨运动预测解决。
- 原始运行：`/media/imc/data/mot-data/runs/g2_d2_first_association_v2_20261111/`；轻量凭证：`reports/diagnostics/g2_d2_first_association_v2_20261111/`。

### 限制

- GT 当前框是推理时不可用的特权信息；D2 只估计机制空间，不是方法。
- oracle 决策不写回，未测量后续轨迹状态、ID 恢复持续时间或 TrackEval 指标；不得把 22 个事件写成 22 个闭环修复。
- 保留了基线速度方向项和全局 Hungarian，因此两个成功事件被破坏；这是隔离“只改位置框”的真实副作用，不应事后删除。
- 没有同时做 D1 历史修正或 D3 检测位置修正，也没有读取 val。

## 下一步

1. 优先冻结并实现 D1 历史对应诊断：针对 D2 未恢复的 30 个失败，确认正检测被哪个 GT/track 占用，并只纠正用于关联的历史身份对应，不改检测坐标或生命周期。
2. D1 需要同时报告对 112 个成功对照的副作用；不能只在失败事件上运行或只保留改善案例。
3. 随后独立实现 D3：只替换已有一对一正检测的坐标为 GT 框，保持检测数量、分数和身份缺失不变，以分离观测定位误差。
4. 当前证据支持先做一个最小确定性运动接口，但不支持直接上 Flow Matching：D2 仅覆盖 41.5% 的失败首轮关联空间，且多数未恢复事件属于其他 track 抢占正检测。

## 2026-11-11：D2 结果到主张审查门

### 已完成

- 在继续 D1/D3 前执行 `result-to-claim` 审查，并将结果写入 `CLAIMS_FROM_RESULTS.md` 与 `findings.md`。
- 新鲜 GPT-5.6-Sol reviewer 给出 `partial`、中等置信度判断；由于是同模型家族，接受状态仅为 provisional，不等同独立审稿通过。
- 本地 trace 位于 `.aris/traces/result-to-claim/2026-11-11_run01/`，包含请求、原始响应、reviewer 元数据和事件记录；`.aris/` 按设计不提交 Git。
- 项目既有的文件哈希、范围、重放和计数交叉检查通过；通用 canonical evidence checker 未能定位，已明确记录为 `tool_unresolved`，没有用项目自检替代语义审查。

### 结论与约束

- 当前只支持有边界的陈述：D2 在冻结的困难 train 事件中暴露 22/53 的当前位置首轮关联空间，但 30/53 仍被历史/全局竞争约束，并对 2/112 成功对照产生副作用。
- 当前不支持 learned uncertainty、Flow Matching、可部署恢复、闭环身份恢复或 HOTA/AssA/IDF1 改善。
- 审查路线为补充实验：先做相同事件/对照上的 D1、D3 和竞争 track 成本/间隔分解；随后才决定是否进入确定性、Gaussian、MDN，再决定是否值得做 Flow Matching。

## 2026-11-11：G2 D1 历史对应诊断与完整性审计

### 已完成

- 在任何 D1 结果产生前冻结 v1 协议 `g2_d1_history_correspondence_20261111.yaml`，SHA256 为 `9ec265b63d80b9140c2de3cf378094dad71bacdc677420a220ff783dbe3870a1`。
- v1 保持基线配对不变，只按过去 30 帧最近 GT 映射观测重标接收正检测的 track。结果为失败组新增正确 0/53、成功对照被语义指标判坏 92/112。
- 一致性审计发现，这 92 个对照的 owner 全部来自 `after_frame-1` 的 gap 内污染观测；v1 把污染观测反当成 track 真身份，方向与预期相反。因此完整保留 v1 运行并标为 failed operationalization，不将其解释为历史污染因果证据。
- 查看 v1 控制失败后，单独冻结探索性替代协议 `g2_d1_clean_gap_history_v2_20261111.yaml`，SHA256 为 `a3cd7c82c47cc7dd30b0199f7c0fea5dd6e2a8a613282ab2b84ef8ccc266fcca`；协议明确记录事后设计，不伪装成独立预注册确认。
- v2 在 `before_frame` 基线更新后复制目标 track；gap 每帧使用相同 Kalman `predict` 后 `update(None)`，阻止任何 gap 检测写入副本；在 `after_frame` 只替换该目标的预测框、速度和 previous observation，检测池、其他 track、求解器、阈值和基线状态不变，结果不写回。
- 两次正式运行均显式设置空 `CUDA_VISIBLE_DEVICES`。现场快照显示物理 GPU0 为 413 MiB/0%，GPU1--7 均为 18 MiB/0%；但 raw summary 未保存运行时环境，因此只能把原始设备声明视为命令/现场记录，不能说已由结果文件独立认证。

### v2 验收与结果

- 165/165 事件适用；四段 replay 帧号/track ID 精确一致，最大 XYXY 误差 `8.0e-6`，低于冻结 `1e-4`。
- 124/165 个事件的目标 track 在 gap 内收到至少一次检测更新；104/165 至少一次更新的一对一 GT owner 为其他身份。总共移除 242 次更新，其中 212 次映射到其他 GT、30 次无法映射。
- 53 个身份改变事件：基线首轮正匹配 1，v2 仍为 1；新增正确 0、受损 0。严格表述应为“新增首轮正确 0/53”，不是所有 53 个都从错误变正确失败。
- 112 个身份保持对照：基线首轮正匹配 111，v2 为 105；1 个新增正确、7 个被破坏。
- D2 仍未恢复的 30 个失败，v2 新增恢复 0；但其中只有 9 个存在可清除的目标 gap 更新，21 个没有受到非零历史干预。
- 只有 12/53 个失败存在目标 gap 更新，41/53 的 clean 与 baseline 预测完全相同；全局首轮配对在失败组仅改变 2/53，在对照组改变 8/112。
- 锚点审计：165 个 `before_frame` tracker 端点全部满足冻结父匹配；164 个 tracker 观测与目标正检测 IoU≥0.99。`dancetrack0012:gt4:f106-108` 的最近观测映射到 GT9，与 GT4 正检测 IoU 为 0.616；按协议保留，且配对变化为 0、既不恢复也不受损。
- 原始运行：`/media/imc/data/mot-data/runs/g2_d1_history_correspondence_20261111/` 与 `/media/imc/data/mot-data/runs/g2_d1_clean_gap_history_v2_20261111/`；轻量凭证位于同名 `reports/diagnostics/` 目录。

### 完整性审计

- 新鲜 reviewer 对全部源文件和原始表执行对抗式审查，并独立 CPU 内存重放：重建 797 个父事件与 contention 分类、165 个 D1 事件，通过 3,795 个 v1 字段比较和 2,640 个 v2 字段比较；检查 48,726 行 replay，所有帧/ID 精确一致。
- Verdict 为 `WARN`，same-family/provisional。未发现假 GT、自归一化活动指标、幽灵结果、数字不符、静默删样本或隐藏负效应。
- WARN 原因：v2 是看过 v1 后的探索性替代；“清理历史”仅指抑制目标 gap 更新，覆盖不足以支持一般历史因果结论；一个锚点身份不干净；原始运行元数据不完整。
- 冻结 v2 协议还承诺报告关联总分 margin，但当前 CSV 只有 IoU、像素变化和配对变化。这是明确的协议/输出偏差；原协议和原运行保持不变，后续只能以单列 addendum 补充。

### 限制与下一步

- D1 v2 只说明窄的 gap-update 抑制没有修复所选失败；不能声称历史无关，也不能把对照受损写成闭环轨迹受损。
- 先补 D1 总关联分 margin addendum，并在后续运行中保存命令、设备、源码与 `seqinfo.ini` 哈希。
- 然后冻结并执行 D3：只替换已有目标正检测的坐标为 GT 框，保持数量、分数和身份缺失不变。
- D3 完成前不进入 learned predictor；即使进入，也先做确定性、Gaussian、MDN，再依据闭环证据决定是否实现 Flow Matching。

## 2026-11-11：D1 总关联分 margin addendum

### 已完成

- 为补齐 D1 v2 冻结协议承诺但原 CSV 漏记的总关联分，单独冻结 `g2_d1_score_margin_addendum_20261111.yaml`，SHA256 为 `ca0a778d82a67e6d1eaceba7e6f97d230072148e20e6d95bec8494c0296ead71`。该 addendum 不修改父协议、父运行或事件集合。
- 新增 `scripts/analyze_g2_d1_score_margin_addendum.py`，独立重放相同 165 个事件，并逐事件断言父 D1 的匹配、IoU、预测变化、正检测 owner、恢复/受损和配对变化全部一致。
- 记录上游 OC-SORT 活动首轮分数 `IoU + detection-score-weighted velocity-direction consistency`，不归一化；同时记录最终 `IoU≥0.3` 接受、快捷路径 `IoU>0.3`、最佳其他 track 和最佳 eligible 竞争 margin。
- 新运行保存完整 argv、Python/NumPy、空 `CUDA_VISIBLE_DEVICES`、OC-SORT commit，以及脚本、helper、上游源码、四段 `seqinfo.ini`、GT、检测、baseline、父结果和协议哈希。

### 验收与结果

- 165/165 父事件精确复现；四段 baseline replay 帧号/ID 精确一致，最大 XYXY 误差 `7.85e-6`，低于 `1e-4`。
- 165 个事件帧在 baseline/clean 两侧均没有触发一对一 IoU 快捷路径，全部进入全局 Hungarian 分数路径。
- 53 个身份改变事件：目标总分提高 11、相同 41、降低 1；获得门控 3、丢失门控 0；配对集合改变 2，但新增首轮正确仍为 0。
- 53/53 失败事件的正检测都有至少一个 IoU-eligible 竞争 track。clean 后 52/53 的目标分数仍低于最佳 eligible 竞争者；唯一正 margin 是原本就首轮正确的事件。
- D2 未恢复的 30 个事件：目标分数提高 9、相同 21；3 个获得门控、2 个改变配对集合，但 30/30 相对最佳 eligible 竞争者的 clean margin 仍为负，中位数 `-0.2871`，范围 `[-0.6387, -0.0421]`。
- 112 个成功对照：分数提高 31、降低 81，1 个丢失门控。这是局部分数描述，不是闭环轨迹损害。
- 原始运行：`/media/imc/data/mot-data/runs/g2_d1_score_margin_addendum_20261111/`；轻量凭证：`reports/diagnostics/g2_d1_score_margin_addendum_20261111/`。
- 运行时 `CUDA_VISIBLE_DEVICES` 已在原始 `runtime.json` 记录为空；结束后物理 GPU0 仍为 413 MiB/0%，GPU1--7 为 18 MiB/0%。

### 独立完整性审计

- 新鲜 same-family reviewer verdict 为 provisional `WARN`，WARN 仅来自继承的 train-only、四困难序列、事后探索性范围；实现、数字、lineage、哈希和 replay 检查全部通过。
- reviewer 独立重放比较 5,610 个父字段和 5,610 个 addendum 字段，所有数值差异为 0；捕获 324 个上游 Hungarian 矩阵、151,784 个元素，addendum 原始分数与求解器负代价逐元素 bitwise 相等。
- 轻量 `outcome_summary.csv` 的每一格、四段 48,726 行 tracker replay、所有输入/源码/seqinfo 哈希均复核通过；没有假 GT、自归一化、幽灵指标、漏事件或事后筛选。

### 解释边界与下一步

- margin 结果与强分数竞争一致，但不能说竞争因果决定失败；Hungarian 优化全局矩阵，且有两个成功对照在目标 pairwise margin 为负时仍正确分配。
- D2 未恢复事件中仍只有 9/30 受到非零 D1 干预；addendum 关闭的是测量缺口，不会把 D1 升级为确认性或闭环证据。
- D1 现阶段交付闭合。下一步冻结 D3 检测坐标修正协议，继续使用相同 165 个事件和成功对照。

## 2026-11-12：G2 D3 已有检测坐标修正

### 已完成

- 在任何 D3 结果产生前冻结 `g2_d3_gt_detection_coordinate_20261111.yaml`，SHA256 为 `b68621e82337abc9c2663c44d06a5edf1d98487fa19533c5337d9651f08567fb`。
- 新增 `src/mot_research/analysis/d3_detection_coordinate.py` 和 `scripts/analyze_g2_d3_detection_coordinate.py`。每个事件独立复制当前首轮检测矩阵，只把已有 target-positive detection 的 XYXY 换成当前目标 GT；ID、分数、数量、顺序、其他检测和所有 tracker 输入保持不变。
- D3 决策不写回；同时保存 raw IoU+方向总分、门控、最佳 eligible 竞争者 margin、快捷路径、全局配对变化、完整 argv/环境和输入/源码/seqinfo 哈希。
- 使用空 `CUDA_VISIBLE_DEVICES` 在 CPU 上完成；raw `runtime.json` 记录为空。结束后物理 GPU0 为 413 MiB/0%，GPU1--7 为 18 MiB/0%。

### 验收与结果

- 165/165 事件适用；四段 baseline replay 帧号/ID 精确一致，最大 XYXY 误差 `7.85e-6`，低于 `1e-4`。
- 53 个身份改变事件：基线首轮正匹配 1，D3 为 15；新增正确 14、受损 0。目标分数提高 34、降低 19；门控新增 2、丢失 1；17 个事件改变全局配对集合。
- 112 个成功对照：基线首轮正匹配 111，D3 为 105；新增正确 1、受损 7。完美目标检测坐标仍可能通过全局分配产生副作用。
- D2 未恢复的 30 个失败中，D3 新增正确 9；对应分数提高 22、降低 8，12 个改变全局配对集合。
- D2 新增正确 22、D3 新增正确 14，交集 5，D2-only 17、D3-only 9，并集 31。其余 22 是“二者都没有新增正确”：其中 21 个在二者后仍首轮错误，1 个原本就首轮正确。
- 165 个 baseline/D3 事件均未触发一对一 IoU 快捷路径，全部进入全局 Hungarian 路径。
- 原始运行：`/media/imc/data/mot-data/runs/g2_d3_gt_detection_coordinate_20261112/`；轻量凭证：`reports/diagnostics/g2_d3_gt_detection_coordinate_20261112/`。

### 完整性审计

- 新鲜 same-family reviewer verdict 为 provisional `WARN`；WARN 来自四个预选 train 视频和解释边界，非数字或实现错误。
- 独立核验 165 次替换均只改变一个检测的 XYXY；检测分数/顺序/其他行、序列化 tracker 状态和全部原始关联输入逐字节不变。
- 6,435 个原始 CSV 单元格差异为 0；324 个活动求解矩阵、48,726 行 baseline replay 和 40 项 provenance/hash 检查通过；四段 GT 与归档 zip 成员逐字节一致。
- 原运行没有记录实际选择的 assignment backend 或 `lap` 版本；审计环境为 `lap 0.5.12`，不能反推原运行版本。后续运行必须补记。
- D2/D3 只能称为不相同的独立首轮 headroom，不得称为互补因果机制；31/53 只是 separate-counterfactual 集合并集。

## 2026-11-12：D1/D2/D3 结果到主张门

### Verdict

- 新鲜 `result-to-claim` reviewer 判定：对“运动/分布不足是主导可行动原因”这一预期主张为 `no`，置信度 medium；这是未建立 dominance，不是证明假设为假。
- 最强可辩护结果仅为：四个预选 DanceTrack train 视频的 165 个 resolved 自然 gap 事件上，D2/D3 分别提供 22/14 个新增首轮正确，交集 5；D1 窄 gap-update 抑制新增 0，但只实际干预 12/53 个失败。
- C4 保持 `NOT_SUPPORTED_BY_PROVENANCE`；learned uncertainty、闭环恢复、HOTA/AssA/IDF1 和一般化均无支持。

### 路由与停止条件

- 最小确定性 motion interface：有条件 GO，仅作为 B1 等价后的小规模 P1 工程探索。
- Gaussian：HOLD，等 P1 的自然闭环结果可信后再做同编码器对照。
- MDN 与 Flow Matching：当前 NO-GO，不得因 oracle headroom 直接启动。
- 下一步先对 21 个 D2/D3 后仍首轮错误事件冻结两个窄诊断：方向代价消融，以及真正同时注入 D2+D3 的组合反事实；它们必须与当前 separate-counterfactual 并集分开报告。
- 随后冻结 train/dev 视频划分并完成 B0/B1 等价，再在完整自然 dev 视频上测固定检测、官方指标、恢复持续性、对照副作用和延迟；同时保留简单 observation/association comparator。
- 若确定性预测只改善误差/oracle 而不改善真实选择或闭环身份稳定，停止扩大运动模型；若简单观测/关联对照解释收益，转向该机制。
- Gaussian/MDN 只改善 best-of-K 或预测分数而不改善闭环时，不得升级分布建模主张；FM 在相同预算/延迟下不优于 MDN 时，放弃 FM 作为核心贡献。

### 审查状态

- D1、D1 addendum、D3 均有 same-family provisional `WARN` 完整性审查和确定性独立重放；早期 D2/G2 语义完整性接受仍 unavailable。
- generic `evidence_check.py` 仍无法解析；项目专用哈希/重放检查不能替代该 generic precheck。
- result-to-claim 本身为 same-family/provisional，trace 保存在本地 `.aris/`。

## 2026-11-12：G2 最终诊断 D4/D5

### 已完成

- 在任何结果产生前分别冻结 D4 方向代价消融和 D5 同时 D2+D3 协议，SHA256 为 `ab329426deffcefcb50d38803fc78481cb03402194650041d04d34d8955dad7b` 与 `e47a740857929815e01bdc9a3af66b4ba4cb79d52cd138af5d72ccb747adf6fd`。
- 新增 `scripts/analyze_g2_final_counterfactual.py`。D4 只把首轮方向权重从 0.2 设为 0；D5 同时把 intended track 与已有 positive detection 的 XYXY 设为当前 target GT。两个作业独立、原子落盘，失败会保留独立 `.failed/failure.json`，互不删除结果。
- 两项使用空 `CUDA_VISIBLE_DEVICES` 并行执行，未暴露任何 CUDA 设备；物理 GPU0 结束后仍为 413 MiB/0%，GPU1--7 均为 18 MiB/0%。

### 验收与结果

- 两项均覆盖 165/165 事件与预注册的 21 个 baseline/D2/D3 均负失败；四段 baseline replay 帧号/ID 精确一致，最大 XYXY 误差 `7.85e-6`，低于 `1e-4`。
- D4：53 个失败中新增正确 0、破坏 0；112 个成功对照中新增正确 0、破坏 3。21 个焦点失败新增正确 0。该结果只否定“全局去掉方向项即可修复”的窄操作，不证明方向或历史无关。
- D5：53 个失败从 1 个正匹配变为 53 个，新增正确为 52；112 个对照新增正确 1、破坏 0；21 个焦点失败全部变为正匹配。
- D5 的 intended pair IoU 对全部 165 个事件都是按定义构造出的 1。它只证明在把已知正确 pair 两侧设为同一 GT 框后，保留的方向项与全局分配没有阻止该 pair；不得表述为可实现恢复上界、D2/D3 因果协同或模型收益。
- 原始运行：`/media/imc/data/mot-data/runs/g2_d4_direction_cost_ablation_20261112/` 与 `/media/imc/data/mot-data/runs/g2_d5_joint_d2_d3_20261112/`；轻量凭证在对应 `reports/diagnostics/` 目录。
- 运行明确记录 `lap.lapjv 0.5.12`、NumPy 1.26.4、SciPy 1.13.0、空 CUDA 可见性和 OC-SORT commit。相关 8 个事件分析测试通过；全仓组合测试另有两个既有环境失败：`mot-eval` 缺 `filterpy`，且直接运行未加载 TrackEval NumPy 兼容层。这两项不冒充本轮通过。

### 完整性审计与限制

- 新鲜 GPT-6.1-Sol ultra same-family reviewer 给出 provisional `WARN`；实现、数值和重放通过，WARN 来自四个预选 train 视频、632 个 unresolved 父事件未进入、D5 构造优势和冻结时间只能由可变 mtime/自记录时间相符而不能独立证明。
- reviewer 独立复现 D4/D5 全部 15,180 个 CSV 单元、489 个求解矩阵/231,063 个元素、48,726 行 baseline replay、797 个父事件、contention 输出和 52 项输入哈希；最大数值差为 0。
- G2 计划中的最终窄诊断至此关闭。它没有改变 C4 dominance verdict=`no`，也不授权直接开始 Gaussian、MDN 或 Flow Matching。

### 精确下一步

1. 冻结视频级 train/dev 划分，把已检查的四个视频视为开发证据，不再用于无约束选择。
2. 建立 B0 原始 wrapper 与 B1 研究接口，并证明关闭研究功能时逐帧/逐 ID 等价。
3. 从一开始加入 GPU1--7 异步作业结构、GPU0 硬拒绝、独立状态/日志/checkpoint、原子结果和只聚合 PASS 作业。
4. 运行最小确定性 P1：自然完整 dev 视频、固定检测、真实闭环写回、官方指标、gap 后身份持续性、对照副作用、延迟和简单 observation/association comparator。

## 2026-11-12：冻结 motion 划分与 B0/B1 等价

### 已完成

- 冻结 `manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml`，SHA256 为 `dd0b7fefd353532e04fb36206c7a1cf624cc661054b198653173f23e320926ff`。
- 四个已检查 G2 困难视频显式作为 motion-dev anchors；从其余 36 段按 `SHA256("20261110:<sequence>")` 升序取前四段，最终 motion-train/dev 为 32/8。
- 新增 split validator，验证 40 段全集覆盖、无交集、成员 hash、选择规则、GT/seqinfo、帧数和首/中/末帧逐字节重复。motion-train 为 34,231 帧，motion-dev 为 7,565 帧，sentinel duplicate group 为 0，官方 val 未读取。
- 新增 `FirstRoundAssociationAdapter`：只在单次 update 内接管首轮 association，输入只读、进程内串行、`finally` 恢复上游符号；二轮、旧观测重匹配、KF/ORU 和生命周期不改。
- 在任何等价结果产生前冻结 `b0_b1_motion_dev_equivalence_20261112.yaml`，SHA256 为 `c529f253de741b65998364477739fc463111b0b24ca401cb9767a2a6b777d171`。

### 验收与产物

- 全部 8 段、7,565 帧完成直接 B0 与 B1-off 独立重放，共 73,479 行 tracker 输出和 7,565 次 B1 首轮调用。
- B0/B1 frame/track ID 精确一致，原图 float64 XYXY 逐元素完全一致；两者与登记 G1 输出的 frame/ID 精确一致，最大序列化 XYXY 误差 `7.85e-6 < 1e-4`。
- CPU-only 运行记录空 `CUDA_VISIBLE_DEVICES`、`lap.lapjv 0.5.12` 和固定 OC-SORT commit；物理 GPU0 未使用。
- split、adapter 和既有事件分析测试累计 `12 passed`；新增源码 `py_compile` 通过。
- 原始运行：`/media/imc/data/mot-data/runs/b0_b1_equivalence_motion_dev_20261112/`；轻量凭证：`reports/baselines/b0_b1_equivalence_motion_dev_20261112/`。

### 限制与下一步

- motion-dev 中四段是刻意选择的困难诊断视频，只能作为开发证据；检测器也可能见过官方 train。正式泛化仍需锁定方法后使用官方 val。
- 本阶段只证明 B1 关闭研究功能时不改变 B0；尚未验证 P1 模型、研究分数、训练、状态写回或指标提升。
- 下一步建立 GPU1--7 异步运行层和失败隔离，然后登记并启动最小 P1 数据/训练 smoke test；GPU0 继续硬拒绝。

## 2026-11-12：GPU1--7 异步作业层

### 已完成

- 新增 `src/mot_research/experiments/async_queue.py` 与 `scripts/experiment_queue.py`，提供异步 launch/status/wait/aggregate、独立 attempt、原子 JSON 状态和 artifact 哈希。
- manifest 只接受物理 GPU1--7，GPU0 立即拒绝；同一并发批次禁止重复 GPU，job 无权自行设置 `CUDA_VISIBLE_DEVICES`。启动前保存 GPU inventory，并默认拒绝已有 compute process 的选中 GPU。
- 每个 job 使用独立 `stdout.log`、`stderr.log`、`resolved_job.json`、`status.json` 和 `artifacts.partial`。成功原子改名为 `artifacts/`；失败保留为 `artifacts.failed/`；重试只能新建编号 attempt。
- aggregate 只选择每个 job 最新 PASS attempt，失败、运行中和缺失 job 单列，不因部分成功伪造完整结果。

### 端到端验收

- 工程 smoke 同时启动 GPU1 PASS、GPU2 预期 FAIL、GPU3 PASS 三个独立进程；这些作业不创建 CUDA 张量，只核对映射与故障隔离。
- GPU2 非零退出后，GPU1/GPU3 均独立 PASS；失败日志和 metrics 保留在 `artifacts.failed/`，两个成功结果在各自 `artifacts/`。
- 聚合器只收录 GPU1/GPU3，并明确列出 GPU2 `latest_state=FAIL`。物理 GPU0 始终未暴露；结束快照仍为 GPU0 413 MiB/0%，GPU1--7 18 MiB/0%。
- queue、adapter、split 与事件测试累计 `15 passed`，新增源码 `py_compile` 通过。
- 原始运行：`/media/imc/data/mot-data/runs/async_queue_smoke_20261112/`；轻量记录：`reports/engineering/async_queue_smoke_20261112/`。

### 限制与下一步

- 这是调度与故障隔离验收，不代表训练稳定性、速度、显存或方法效果；GPU 标签被正确隔离，但 smoke 未申请 CUDA tensor。
- 下一步冻结 P1 样本契约，先做数据构建/严格在线性检查和极小样本过拟合；这些通过后才用 GPU1--7 异步启动正式 seed/job。

## 2026-11-12：P1 严格在线运动样本

### 已完成

- 在任何 P1 样本结果产生前冻结 `manifests/protocols/p1_motion_samples_v1_20261112.yaml`，SHA256 为 `9348f0b551daef54492e393be787a29cbb154dfc564bbfaf9824d8b2331c3690`。
- 新增 `motion_samples.py` 和 `build_p1_motion_samples.py`：按帧顺序先发出目标样本、再把当前匹配检测追加到历史，确保当前/未来检测不会进入输入；每个历史来自过往 GT 匹配的真实检测框，长度 8、至少 2 个观测、最大 gap 31 帧。
- 状态固定为 `[cx/W, cy/H, log(w/W), log(h/H)]`，监督目标为当前 GT 状态及其相对最后观测检测状态的 delta。训练对应使用 GT--检测 Hungarian IoU `>=0.5`，首轮检测池保持 score `>0.6`。
- 规范化统计只用 32 段 motion-train 拟合，8 段 motion-dev 复用且不重新拟合；官方 DanceTrack val 未读取。

### 验收与产物

- motion-train 为 266,854 个样本，其中 261,069 个目标帧存在当前正检测、5,785 个不存在；motion-dev 为 80,458 个样本，其中 75,668/4,790 个分别存在/不存在当前正检测。
- train/dev 样本键交集为 0；全部历史帧严格早于目标，至少两个有效历史，记录的 frame gap 与最后历史帧一致，状态有限且 sample key 唯一。状态编解码最大绝对误差为 `2.22e-15 < 1e-10`。
- 两个独立原子构建目录的 summary、normalization、input hashes、NPZ SHA256 和数组内容哈希完全一致；按缺失 IoU 的 `NaN==NaN` 语义逐数组比较也完全一致。train/dev 内容哈希分别为 `2d5c8b35546f18d66f306f771eabb294cdca4d93f1e628fc0b946b9385aec711` 和 `e055e17ce0dc2e24bf2fd899ed24a266359aecd34f78f33d3f1ac8828e68d141`。
- 新增 motion 样本、split、adapter 和 queue 相关测试共 `10 passed`；构建脚本与 helper `py_compile` 通过。两次构建均使用空 `CUDA_VISIBLE_DEVICES`，物理 GPU0 结束快照仍为 413 MiB/0%。
- 原始目录：`/media/imc/data/mot-data/runs/p1_motion_samples_v1_20261112/` 与 `/media/imc/data/mot-data/runs/p1_motion_samples_v1_repeat_20261112/`；轻量凭证：`reports/data/p1_motion_samples_v1_20261112/`。

### 限制与精确下一步

- GT 只用于监督目标和训练期历史对应，但这些历史是 GT 对齐的检测观测，不是 tracker 自身闭环关联出的历史；四个 G2 困难视频还是有意选择的 dev anchors。样本数量与数据可重复性不是预测质量或跟踪提升。
- 下一步实现最小确定性 GRU（固定输入/损失/seed），先在极小训练子集做过拟合 smoke，验证 batch、mask、归一化、checkpoint 和 GPU 队列真实 CUDA 路径；通过后才启动少量 seed 的 motion-train 训练与完整 motion-dev 离线误差/自然闭环对照。

## 2026-11-12：P1 确定性 GRU 真实 CUDA 过拟合 smoke

### 已完成

- 新增最小 `DeterministicMotionGRU`：每个有效历史观测输入四维 motion-train 规范化状态和 `log1p((target_frame-history_frame)/fps)`，先压紧左填充有效项再送入单层 GRU，输出规范化四维 delta。
- 新增稳定 SHA256 sample-key 子集选择、训练入口和三项单测，覆盖子集确定性、padding 值不影响输出、delta 规范化与 target state 重建。模型/P1 数据/队列相关组合测试为 `13 passed`；此前六个现有相关测试文件组合为 `19 passed`。
- 在结果前冻结 `p1_deterministic_overfit_smoke_20261112.yaml`，SHA256 为 `b9c339caf5fdf29a0d0368c6917358ec1be01dcf6b6f2f7517b8372391003aa9`；源码、样本、规范化、128 样本选择、seed、800 步优化和 acceptance 均预先登记。

### 验收与产物

- 队列在物理 GPU1 启动一个独立作业，进程记录 `CUDA_VISIBLE_DEVICES=1`、`cuda:0`、单一可见设备、PyTorch 2.6.0+cu124/CUDA 12.4 和 NVIDIA GeForce RTX 4090 D；GPU0 没有暴露，结束快照仍为 413 MiB/0%。
- 初始 normalized MSE 为 `0.7186536193`，800 步后为 `0.0001083946`，final/initial 为 `0.0001508301`；分别通过预冻结的 `<=0.05` 与 `<=0.1` 阈值。
- checkpoint 重载后的 128 个预测与保存前最大绝对差为 0；作业 return code 0，四项预期 artifact 齐全，aggregate 为 1/1 PASS。
- 原始队列与 checkpoint：`/media/imc/data/mot-data/runs/p1_deterministic_overfit_smoke_20261112/`；轻量凭证：`reports/engineering/p1_deterministic_overfit_smoke_20261112/`。checkpoint SHA256 为 `e0638947d69ab72f98e43f1dc7e36f050d3e4c6379bfbbece035d966b9db2814`，不进入 Git。

### 限制与精确下一步

- 训练和评估刻意使用同一 128 样本，这只证明实现管线和模型容量可用，不能说明 motion-dev 泛化、关联改善、闭环身份稳定或跟踪指标提升。
- 下一步先冻结完整 motion-train 训练和 motion-dev 离线评价：少量独立 seed 分配到 GPU1--3，并与零 delta、最后速度外推等无学习 comparator 在相同样本上比较。只有 dev 误差、gap 分层和 seed 稳定性达到预注册门槛，才把预测接入 B1 做自然闭环；否则停止扩大模型。

## 2026-11-12：P1 三 seed 离线预测可行性门

### 冻结设计与执行

- 在任何完整训练或 motion-dev 模型结果前冻结 `p1_deterministic_offline_v1_20261112.yaml`，SHA256 为 `5f918470aac7220073744d947e939ecb754cf7072756b9028168b51523057aea`。
- 三个固定 seed `20261112/13/14` 分别使用物理 GPU1/2/3，训练完整 266,854 个 motion-train 样本 10 epoch；无 dev 早停、无最佳 checkpoint/seed 选择。GPU0 未暴露，结束快照仍为 413 MiB/0%。
- 完整 80,458 样本 motion-dev 同时评价 learned GRU、零 delta 和最后两观测恒速度；保存逐样本预测、整体/序列/gap/当前正检测存在性指标。三个作业均 return code 0、artifact 齐全并独立 PASS。
- 预注册门要求：平均宏 MSE 比最佳对照 `<=0.98`、每个 seed 不差于最佳对照、gap2--31 平均 MSE 比 `<=0.95`、宏 IoU 下降不超过 0.005、seed MSE 相对极差 `<=0.10`；五项必须全部通过。

### 聚合停止与透明修正

- 原聚合器在写 summary 前安全停止，因为它要求不同进程独立计算的 comparator 整棵 JSON 字典 bitwise 相等。定位到的差异只有 IoU mean/median 浮点叶，最大为 zero delta `3.33e-16`、恒速度 `2.22e-16`；结构、计数、非浮点字段和定义一致。
- 在读取 learned gate 结果前冻结 `p1_deterministic_offline_v1_1_aggregation_addendum_20261112.yaml`，SHA256 为 `773024f3e9f10f363ef03b89da07ba1425db347d97742b26698b9e09e76e890d`。它只允许 comparator 浮点叶绝对误差 `<=1e-12`，结构/计数仍精确，learned 分支不改，原五项阈值不变。
- 审计确认技术修正忠实；但“结果前未检查 learned 分支”和冻结先后只能由自记录与可变文件时间支持，无法事后独立认证，因此保留 chronology WARN。

### 离线结果

- 决策为 `GO_TO_NATURAL_CLOSED_LOOP`，五项条件全部通过。三个 learned 宏 MSE 为 `1.380908/1.385514/1.383229`，最佳零-delta 为 `1.510670`；均值比值 `0.915631`，三个 seed 各自在 8/8 序列上更低。
- gap2--31 learned 平均 MSE `12.047239`，零 delta `13.015015`，比值 `0.925642`；learned 宏 IoU `0.850501`，零 delta `0.843628`，差 `+0.006873`；seed 相对极差 `0.003330`。
- 不利分层必须保留：650 个 gap16--31 样本上，MSE 比零 delta改善约 9.28%，但三个 seed IoU 均更低，平均差 `-0.008110`。motion-dev 又以 gap1 为主（75,428/80,458），不能写成所有长 gap 一致改善。
- 原始队列：`/media/imc/data/mot-data/runs/p1_deterministic_offline_v1_20261112/`；原始 summary：`/media/imc/data/mot-data/runs/p1_deterministic_offline_v1_summary_20261112/`；轻量凭证：`reports/feasibility/p1_deterministic_offline_v1_20261112/`。

### 完整性审计与精确下一步

- `experiment-audit` 新鲜 same-family reviewer 给出 provisional `WARN`。它独立核验 347,312 个 target state 对原始 GT 最大误差 0、40 个 GT 对归档一致、全部 80,458 个 dev 预测键/目标、50 项 provenance/artifact、120 项 GT/seqinfo/检测哈希和所有指标；独立复算最大差 `3.33e-16`，五项门全部复现。
- WARN 原因是 GT 对齐历史、4/8 已选困难 anchor、gap16--31 IoU 退化、没有闭环官方指标和冻结 chronology 不能独立认证；不存在伪造 GT、自归一化或 phantom result 发现。
- 当前只支持“通过冻结离线开发门并获准做自然闭环”。不支持关联/身份恢复、HOTA/AssA/IDF1、部署收益、统一长 gap 改善、正式泛化或扩展 Gaussian/MDN/FM。
- 下一步冻结 B1 自然闭环协议：使用 tracker 自身历史、固定检测、三个 seed 全部运行；同时保留 B0 和可实现简单 comparator，评价官方 HOTA/AssA/IDF1、gap 后身份持续性、成功对照副作用和延迟。针对 gap16--31 预注册单独安全门；不选择最佳 seed。

## 2026-11-12：已完成实验的本地复盘与后续排程修订

### 完成内容

- 基于 `bcbd572` 的代码、冻结协议、轻量结果与既有审计，复盘 G1/G2、B0/B1-off、P1 样本与三 seed 离线结果；新增 `docs/experiment_review_20261112.md`。
- 结论：保留整体研究问题，下一轮仅做 P1 自然闭环，复用现有三个 checkpoint；暂缓 Gaussian/MDN/FM 和 24 组矩阵，不继续扩展泛化的 oracle 诊断。
- 建议最小 6 个方法级运行：B0/KF、同接口零位移、同接口恒速各一次，P1 三 seed 各一次，全部覆盖现有 8 段 motion-dev。这是待冻结的下一轮建议，不是已执行实验或新训练结果。
- 补充真实历史契约、首轮预测框替换边界、共同事件分母、长 gap 安全评价及结果分支。明确确定性闭环失败并不逻辑上否定概率模型，但当前证据不足以直接扩大模型复杂度。
- 更新 README 入口、手册执行顺序和研究监管入口；澄清实际 P1 hidden=64/MSE/三 seed 与初始草案配置的区别。创新台账 C1–C5 未升级，历史冻结协议及结果未改写。

### 验证证据

- 按 P1 `source_hashes.yaml` 重新计算本地 9 个协议/配置/代码 SHA256，9/9 匹配。
- 从已提交 `gate_summary.json` 复算 MSE 三 seed 均值 `1.3832172289519844`、对照比值 `0.9156314457034193`、gap2–31 比值 `0.9256415991645001`、宏 IoU 差 `0.006872605823983857`、seed 相对极差 `0.003329592132010729`，与轻量摘要一致。
- 阅读 D2/D3/D5 outcome CSV、B0/B1 逐视频表及样本/模型/指标/接口实现；保留 D2、D3 对成功对照的损伤和 gap16–31 IoU 退化，未将首轮 oracle 匹配计数写成持续身份恢复。
- 对本次五份新增/更新入口与复盘文档检查 36 个本地 Markdown 链接，全部存在；`git diff --check` 通过。此次仅改文档，没有新增 CUDA 作业或重新运行训练/TrackEval。

### 限制与精确下一步

- 本次未访问服务器，未读取原始 NPZ/checkpoint/全量预测，因此不是既有审计的独立复现；哈希一致不能认证冻结时间。已有审计的 provisional WARN 保留。
- 下一步起草并冻结 `P1 natural closed-loop v1`：填实真实历史规则、数值增益/副作用/长 gap/延迟门、6 个主运行和预算；随后实现并验收启用路径，在 motion-dev 上执行。完整结果揭晓前固定决策规则，暂不用官方 val 调参。
- 服务器仍使用 `/home/imc/yzy/MOT`，大型产物保存在 `/media/imc/data/mot-data`；未来 CUDA 作业显式排除物理 GPU0，启动前重新检查 GPU1–7 占用。
- 修改留在本地工作树，未创建 Git commit 或 push。

## 2026-11-12：P1 自然历史首轮替换接口 smoke

### 实现与冻结

- 新增 `natural_motion_policy.py` 与 `run_p1_natural_closed_loop.py`：按稳定 tracker ID 保存真实匹配检测历史，只在完整帧更新结束后追加；轨迹删除即清理，重生使用新 ID。新轨迹出生时，上游没有填写 `last_observation`，因此以更新后新 KF 状态唯一匹配同帧高分检测；歧义或无法映射时不猜测并计数。
- 空/单观测、gap>31、NaN/Inf 或非法预测均保留原 KF 首轮框并显式计数。替代框只传入首轮 `associate`，不写回 KF 状态；方向项、阈值、旧观测重匹配、第二轮和生命周期保持上游实现。
- 首轮请求改用显式数据集帧号，避免上游 `None` 空检测帧不递增内部 `frame_count` 时压缩真实 gap；`None` 帧不重复追加旧观测。motion-dev 8 段当前均无整帧检测缓存为空，但该边界已用真实 OC-SORT 测试覆盖。
- 在运行结果前冻结工程协议 `p1_natural_closed_loop_interface_smoke_v1_20261112.yaml`，SHA256 为 `f77a2a65f5d9a6c58915ca0debaacb47d3a8ad7983e8522aa5cc73847eb93b74`。固定 `dancetrack0012` 1--200 帧和 seed 20261112 checkpoint；该 seed 仅因编号最小用于接口验收，不做优选。

### 验收与环境记录

- `mot-core` 的 adapter/natural-policy 测试为 `7 passed, 2 skipped`（按环境设计无 FilterPy）；`mot-det` 为 `9 passed`，含真实 OC-SORT upstream 逐元素等价、出生检测记录和空帧真实时间轴测试。脚本 `py_compile`、YAML 解析和 `git diff --check` 通过。
- attempt_001 在执行 tracker 前因 `mot-det` 缺 PyYAML 失败；失败日志/状态保留，未覆盖。补装并声明 `PyYAML==6.0.2`，同步 `mot-det.yaml` 与完整 pip lock 后，新建 attempt_002。
- 启动前 GPU0 有图形界面进程；GPU1--7 无 compute process。attempt_002 仅暴露物理 GPU1，并记录进程 `cuda:0`、PyTorch 2.6.0+cu124/CUDA 12.4；GPU0 未使用。
- attempt_002 工程门全部通过：200 帧/200 次首轮调用，2,560 个 tracker rows 中 2,516 个 eligible 且实际替换，44 个因历史不足回退；记录 2,060 个真实观测和 17 个出生观测，17/17 唯一映射，unresolved/ambiguous/非有限/非法预测均为 0。
- tracker-only 延迟在固定 20 帧 warm-up 后，180 帧 p50 `4.7521 ms`、p95 `7.0502 ms`、mean `5.1281 ms`。这不含检测器，且尚未与其它方法做同协议比较。
- 原始队列：`/media/imc/data/mot-data/runs/p1_natural_closed_loop_interface_smoke_v1_20261112/`；轻量记录：`reports/engineering/p1_natural_closed_loop_interface_smoke_v1_20261112/`。checkpoint、1,945 行 tracker 输出及逐帧调用日志均留在数据盘，不进入 Git。

### 边界与精确下一步

- 该结果仅证明自然历史启用路径的时序、映射、数值和运行契约可用；没有读取 GT、没有计算 HOTA/AssA/IDF1，也不能说明关联或身份改善。
- 下一步在完整结果揭晓前实现共同分母的恢复/持续/损伤评价，并冻结六个主运行的完整协议与数值决策门；随后才运行 8 段 motion-dev 的 B0/KF、零位移、恒速和三个 learned seed。Gaussian/MDN/FM、gap>=16 次要回退和官方 val 仍暂停。
- 本阶段修改尚未 commit 或 push。

## 2026-11-12：P1 闭环共同恢复事件分母

### 定义与实现

- 新增 `closed_loop_events.py` 与 `build_p1_closed_loop_event_set.py`，在任何完整候选方法输出产生前，只用 motion-dev GT、固定检测和已登记 B0 定义共同事件；候选方法不能按自身失败重选分母。
- 事件固定要求：GT 连续可见，gap 前有 IoU>=0.5 的全局一对一正检测，随后 1--31 帧无正检测，再出现正检测，且 B0 在 gap 前有匹配 tracker。重现成功定义为方法自己的 gap 前 ID 在 after 帧接回；持续成功定义为该 ID 从 after 起连续 5 个 GT 可见帧保持，边界不足 5 帧单列右删失。
- gap 分层固定为 1、2--3、4--7、8--15、16--31；另保留四段困难 anchor/其余四段和 B0 success-control/recovery-failure/censored 分层。方法 gap 前无匹配按共同事件失败记录，不从分母删除。
- 在计算事件数前冻结 `p1_closed_loop_common_events_v1_20261112.yaml`，SHA256 为 `ddbd280e57119aa030b99b9e8c0d893c34fd86836cde3a637b3454b3d34c47aa`；官方 val 与候选方法输出均未读取。

### 构建结果与边界

- 固定事件共 831 个：anchor 604、non-anchor 227；gap1/2--3/4--7/8--15/16--31 分别为 398/264/118/34/17。另有 254 个因 B0 gap 前无匹配排除，8 个因 gap>31 排除。
- B0 在重现帧接回 gap 前 ID 为 159/831。连续 5 帧持续性可评价 780 个，其中 B0 成功 81、失败 699；51 个右删失。后续所有方法必须使用同一 831 个 event ID，并分别报告 reconnect 与 persistence。
- gap16--31 只有 17 个，B0 reconnect 和 persistent success control 都为 0。该层无论均值如何都标记小样本/安全性不确定；完整协议只能预注册与简单对照的观察性不退化条件，不能宣称普遍长 gap 安全。
- 事件模块测试在两个环境均通过；组合结果分别为 `mot-core 10 passed, 2 skipped`、`mot-det 12 passed`。成功 events CSV SHA256 为 `25c2602e9f5f2a6352b9bef3fc6ffdff0879b68016f0c06359c46b1ec867d2fc`。
- 第一次构建在事件计算后因 summary 中将 Python `False` 拼成 `false` 而失败，`.failed` 完整保留；修正后使用新目录 retry1 成功，未覆盖失败证据。原始成功目录：`/media/imc/data/mot-data/runs/p1_closed_loop_common_events_v1_retry1_20261112/`；轻量记录：`reports/data/p1_closed_loop_common_events_v1_20261112/`。

### 精确下一步

- 实现对固定 event CSV 的逐方法评价和官方 TrackEval 八段聚合入口，先用 B0 自评复现 159 reconnect、81/780 persistence 和已登记 TrackEval 基线作为验收。
- 验收通过后，再冻结六个主运行、最小 HOTA/AssA 增益、DetA/harm/长 gap/延迟限制和只揭晓一次的预算；随后才启动完整闭环。当前没有完整新方法指标，Gaussian/MDN/FM 与官方 val 仍暂停。

## 2026-11-13：P1 自然闭环六方法完整实验

### 冻结设计与执行

- 在完整方法结果揭晓前冻结 `p1_natural_closed_loop_v1_20261113.yaml`，SHA256 为 `48c69f27aac2da9eac6c69314a4bba1616280895cc08cd78c566ed7c7a982534`。固定 B0、零 delta、恒速和三个既有 learned seed，覆盖 motion-dev 全部 8 段/7,565 帧，不重新训练、不挑 seed、不使用官方 val。
- 新增共同事件评价、官方 TrackEval API 评价和一次性门聚合入口。B0 验收先精确复现 831 个事件、159 次 reconnect、81/780 次 persistence，以及登记的八段 TrackEval 基线。
- B0/zero/CV 在独立 CPU 原子目录执行；learned seed `20261112/13/14` 分别只暴露物理 GPU1/2/3 并映射到进程 `cuda:0`。所有 tracker、评价和聚合作业均 PASS，GPU0 未暴露。

### 官方指标与恢复结果

- B0 HOTA/AssA 为 `50.059599/31.662792`；零 delta 为 `47.664771/28.677977`；恒速为 `51.056654/32.623675`。
- 三个 learned seed HOTA 为 `49.052182/49.205286/49.555429`，AssA 为 `30.296771/30.460473/30.866921`。均值相对 B0 为 HOTA `-0.788633`、AssA `-1.121403`，相对恒速为 `-1.785689/-2.082287`。
- 固定 699 个 B0 persistence failures 上，zero/CV/learned 的恢复数为 `22/23/19/17/17`；固定 81 个 B0 success controls 上损伤数为 `23/20/21/20/19`。三个 learned seed 的恢复减损伤均为负。
- 17 个 gap16--31 事件上所有方法 persistence success 都为 0；该层样本小且没有 B0 成功对照，不能把共同为零解释成长 gap 安全。
- Learned 三次运行最大分序列 tracker-update p95 为 `10.9287/10.5127/11.0010 ms`，均低于 15 ms；出生映射 `342/345/346`，无 unresolved/ambiguous、非有限或非法预测。

### 决策、审计与记录

- 冻结门最终 `NO_GO`：16 项中 8 项失败。失败项包括相对最强对照的 HOTA/AssA、seed 胜出、最差 seed 安全、恢复增益、恢复减损伤和 harm 限制；通过的方向性/DetA/延迟/数值等检查不能覆盖主指标失败。
- fresh same-family `experiment-audit` 为 provisional `WARN`，原因是选定开发范围、仅 17 个长 gap 事件和可变本地 chronology；未发现伪造 GT、归一化欺诈、phantom result、漏报运行或干预泄漏。审计独立重建 831 个事件、4,986 个方法事件结果，重跑六个 TrackEval 并复现全部 16 项门。
- `result-to-claim` 给出 `claim_supported=no`、`route=pivot`、`confidence=high`。离线预测门通过没有转化为稳健自然闭环身份关联；不进入 Gaussian、MDN、FM 或官方 val。
- 原始 tracker：`/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_20261113/`；learned 队列：`/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_learned_20261113/`；评价：`/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_evaluation_20261113/`；聚合：`/media/imc/data/mot-data/runs/p1_natural_closed_loop_v1_summary_20261113/`。
- 轻量报告：`reports/feasibility/p1_natural_closed_loop_v1_20261113/`。大文件、checkpoint、逐帧输出仍只在数据盘，不进入 Git。本阶段尚未 commit 或 push。

### 精确下一步

- 先冻结一个有界、前瞻的机制诊断：在相同 B0 历史与候选集上并列 KF、zero、恒速和全部 learned seed，记录中心/尺寸误差、IoU、正确候选是否存在、门控、竞争者排名、Hungarian 配对及其与闭环 harm 的关系。
- 恒速必须保留为强简单对照。只有诊断识别出可区分且可行动的机制时，才允许最多一个只用 motion-train 的确定性修正；若没有机制或修正再次失败，停止/转向 learned-motion 分支。

## 2026-11-13：P1 同 B0 状态机制诊断与分支停止

### 冻结设计与执行

- 新增纯分析模块、3 个单测和 `analyze_p1_common_state_mechanism.py`。在任何全量诊断结果产生前冻结 `p1_common_state_mechanism_v1_20261113.yaml`，SHA256 为 `48e431925f92212583e3182b0498122f17364b5855be0ec367f93329df11622a`。
- 诊断固定使用既有 831 个共同事件，只在登记的重新出现帧旁路评价 B0/KF、zero、恒速和三个 frozen learned checkpoint。六方法共享同一 B0 live tracks、真实检测历史、检测候选、方向项、门控和 Hungarian backend；替代结果不写回 tracker/KF/history。
- 预注册四类 learned 相对恒速劣势：目标 IoU `>=0.01`、归一化中心误差 `>=0.005`、过门率 `>=0.02`、即时正匹配率 `>=0.02`，均要求至少两个 seed 同向。无触发时直接停止 learned-motion 分支；只有触发后事件级复核才可能授权最多一次修正。
- 唯一一次完整运行 CPU-only，`CUDA_VISIBLE_DEVICES` 为空，未使用任何 GPU。输出原子写入 `/media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113/`。

### 结果

- 831 × 6 = 4,986 个 event-method 行完整；829 个事件对所有方法共同 applicable，两个事件的预期 B0 track 已死亡并对所有方法保留。829 个共同事件全部具有可用历史，五个替代方法都实际应用预测。
- B0/CV/三个 learned 的目标 IoU 均值分别为 `0.625057/0.522368/0.617239/0.617172/0.615925`；归一化中心误差为 `0.015774/0.022220/0.016212/0.016217/0.016266`。
- CV 过门/即时正匹配为 `692/509`；三个 learned 为 `771/555`、`774/562`、`768/559`。三 seed 均在 601/829 个事件上同时比 CV 有更高目标 IoU，至少两个 seed 更高的事件为 628/829；三 seed 均匹配而 CV 失败为 81 个，反向为 28 个。
- learned 相对 CV 的四个注册“劣势”为 `-0.094411/-0.005988/-0.095296/-0.059912`，一致劣势 seed 数全部为 0，四项均不触发。负值意味着 learned 在这些同状态单步聚合量上优于 CV。
- 该结果没有推翻自然闭环 `NO_GO`。相反，它形成单步更好、实际闭环更差的反转，但不能据此因果归因为反馈/history/lifecycle。B0 的 575 个即时首轮正匹配中只有 158 个 reconnect，说明即时匹配不能替代闭环身份指标。

### 审计与结果路由

- fresh same-family `experiment-audit` 为 provisional `WARN`，只因选定开发范围、单步边界、17 个长 gap、描述性 outcome join 和可变 chronology。审计独立重放 7,565 帧/73,479 tracker rows，逐字节复现八段 B0，并以零差异复现 4,986 行/189,468 个 CSV 单元；749 次事件帧旁路均未改变 tracker/KF/history，831 次 GT 读取均在六方法输出固定后发生。
- 未发现假 GT、预测自归一化、phantom result、漏事件或 writeback。原始 CSV SHA256 为 `d08df0baec806d732fa04b463d073b0d0b9bb6ac5b3954fa1249fae77354a73e`，summary 为 `d2e9d4138701de0105750cf0d6e836bb5164068358ac5a0cf65df9d73b4a998c`。
- `result-to-claim` 给出 `claim_supported=no`、`route=pivot`、`confidence=high`。当前获准确定性修正数为 0；不允许 Gaussian、MDN、FM 或 official val。不能将“注册劣势未触发”写成“反馈/history 机制不存在”。
- 轻量记录：`reports/diagnostics/p1_common_state_mechanism_v1_20261113/`；本地 audit/claim trace 位于被忽略的 `.aris/traces/`。大体积 replay 和逐事件 CSV 留在数据盘。

### 精确下一步

- 按冻结规则关闭当前 learned-motion 分支，不放宽阈值、不事后选择 subgroup 或 seed、不追加一个无机制依据的修正。
- 下一阶段不是继续 P1 诊断，而是进行研究路线选择：从与当前负结果正交的新问题中选择候选，先建立文献/provenance 和最小可证伪协议再运行。
- 若未来重新研究 feedback/history/lifecycle，必须作为单独授权的新分支和新冻结协议，且不得声称本轮已识别该机制。
- 当前修改尚未 commit 或 push；Git 中不包含 checkpoint、raw replay 或逐事件大文件。

## 2026-11-13：自然闭环与机制诊断结果的本地复盘

### 完成内容

- 本地初始为 `57d4aa2`、工作树干净。发现 GitHub 已有新增实验后，以 `git pull --ff-only` 快进同步至 `1e94b52`；未创建新 commit 或 push。
- 阅读六方法自然闭环、同 B0 状态机制诊断的冻结协议、轻量结果、既有审计和相关事件/诊断代码，新增 `docs/experiment_review_20261113.md`。
- 同意停止当前冻结 GRU 首轮框替换分支，保留 KF 与恒速双对照；明确单步 learned 优于 CV 不等于优于 KF，更不等于闭环收益。无触发是注册诊断的结果与资源停止规则，不是运动学习/概率模型/FM 被普遍否定。
- 记录干预范围与诊断范围的差异、内部首轮匹配与输出 reconnect 的差异、两种 gap 定义不同，以及恒速成功对照损伤 20/81 的限制。
- 更新 README、验证路线、研究监管入口与当前状态的同步说明，避免重复执行已完成闭环/机制诊断。未改代码、冻结协议、阈值或历史实验结果。

### 验证证据

- 两个新增实验协议 SHA256 与轻量结果一致；机制审计 Markdown/JSON 与记录哈希一致，共 4/4。
- 从闭环 `gate_summary.json` 重算三 seed HOTA/AssA/IDF1/DetA 均值，与记录一致。HOTA 相对 B0/CV 为 `-0.7886331538/-1.7856887400` 点；AssA 为 `-1.1214034223/-2.0822867506` 点。
- 由 81 个 B0 成功事件与恢复/损伤计数，推算固定 780 个可评价事件的持续成功数：B0=81、zero=80、CV=84、learned=79/78/79。该推算仅核对摘要算术，不是重新从 tracker 输出评价。
- 检查本次新增/更新文档的本地 Markdown 链接均存在，`git diff --check` 通过。仅文档修改，未运行训练或 CUDA 作业。

### 限制与精确下一步

- 本次未访问训练服务器、未读取完整逐事件 CSV/预测/checkpoint、未重跑 TrackEval；原始产物复现依据既有 same-family provisional WARN 审计。哈希与摘要复算不能替代独立全量复现。
- 下一步做一轮有边界的研究问题选择与窄范围先例核查，比较少数候选的可观察输入、竞争解释、最小可证伪实验和成本，再选择一个新分支。关联干预与损伤控制只是待审候选，尚无新方法或创新结论。
- 当前 P1 停止判定保留，不追加 P1 修正、Gaussian/MDN/FM 或 official-val 运行。本次文档修改留在工作树，未创建 commit 或 push。

## 2026-11-13：停止 P1 后的研究路线选择建议

### 完成内容

- 使用 hypothesis-generation 的证据/假设/竞争解释/区分实验框架，完成 17 条机制导向网页检索，保存查询边界和原始来源链接于 `docs/research_route_proposal_20261113.md`。
- 对比继续运动生成、观测可靠性、关联改动收益/损伤学习三类路线；优先建议第三类的有限可行性试验，不声称其已具备创新或有效性。
- 核查 Learning to Track、MARLMOT、DeconfuseTrack、UTrack 相关方法段落；TrackTrack、Sentinel、OATrack 的索引材料明确显示自适应关联、分解混淆、状态保护已有先例。将各源的全文/片段访问等级与限制写入文档。
- 形成 R1 草案：KF/CV 可执行匹配提案、完整分歧分量、同状态双分支短程标签、在线特征轻量选择；主要区分未来身份监督、即时标签与简单规则/减少干预次数的竞争解释。
- 建议先 200 个 fit 样本的工程与标签验证，再视信号决定扩至最多 20,000 个分量和完整闭环。24/8 训练内划分、H=10、预算与开发门均为待冻结建议，不是已执行或已预注册设置。
- 在创新台账登记未获支持的候选 R1，新增 README 入口；当前 P1 停止状态和 C1–C5 状态保持。

### 验证与限制

- 文献结论仅引用原始论文、作者/会议页面、出版社/PMC 索引；未向搜索引擎发送项目原始数据或结果数值。
- 检索不是穷尽的新颖性检查；部分 2026 论文直接全文受访问限制，本次未检查官方代码。已有风险/决策学习先例意味着“添加门控”不足以支持新贡献。
- 新方案、README 和创新台账中的 26 个本地 Markdown 链接检查通过；文档查询清单为 17 条，与本轮检索记录一致；`git diff --check` 通过。
- 本阶段仅提出方法与实验设计，未连接服务器、未构建分支标签、未安装新环境、未训练或启动 GPU 作业；没有新增研究结果。
- 下一步为 R1 阶段 A 的全文/代码差异核查、冻结最小标签协议和 200 样本双分支验收；先判断标签及可学习性，再决定是否进入完整方法研发。不自动恢复 P1 或开展概率模型。
- 新增产物与入口留在本地工作树，未创建 Git commit 或 push。

## 2026-11-18：R1 阶段 A 同状态双分支工程验收

### 完成内容

- 在已同步的 `f5c88f0` 工作树上阅读 R1 路线和现有 OC-SORT 首轮接口；核查 Learning to Track、Event-aware Learning、DeconfuseTrack 等原始来源，发现“未来事件监督”和“分解混淆关联”均有明确先例。将可确认差异及访问限制写入 `docs/r1_stage_a_literature_and_protocol_20261118.md`，不作新颖性结论。
- 在读取本轮标签前冻结 `manifests/protocols/r1_stage_a_20261118.yaml`，SHA256 `b698941dbbea9a34c3c9b5e6b9f3cb4952941736c143fcfb6af964f3eacd6562`。原 32 段 motion-train 固定拆为 24 fit / 8 calibration；fit 排名前四为 `0087/0098/0072/0001`。原 motion-dev 与官方 val 未读取。
- 新增整分量切换模块 `src/mot_research/analysis/r1_components.py`、CPU 回放器 `scripts/run_r1_stage_a.py` 和相关测试。A0 为原 B0/KF 首轮关联，A1 仅切换一个完整分歧分量到同状态恒速提案；后续 9 帧均回到 B0。用 `t−1` 的 GT—输出锚点产生 10 帧身份正确数的正/负代理标签。
- 第一次调用 `mot-core` 因缺少 `filterpy` 在首帧失败，现场留在 `/media/imc/data/mot-data/runs/r1_stage_a_20261118.failed/`。改用已有 `mot-det` 环境重试，成功目录为 `/media/imc/data/mot-data/runs/r1_stage_a_20261118_retry1/`；没有安装新依赖或覆盖失败现场。
- 使用 `PYTHONNOUSERSITE=1`、`CUDA_VISIBLE_DEVICES=''`、单线程 BLAS/OMP 完成 4 段 CPU 回放。物理 GPU 0 及其他 GPU 均未使用；未训练模型。

### 验证与产物

- 四段共 4,112 帧，A0 父轨迹逐帧与发现阶段输出完全一致；所有已执行分量的一对一匹配、A0 未来分支完全一致、父状态字节不变检查通过。
- 从 219 个满足完整窗口的分歧分量中按冻结哈希每段取最多 50 个，共执行 164 个（50/50/50/14）。另有 6 个尾部窗口不足 10 帧而排除。代理净标签正 34、负 48、零 82；受益锚点 43、受损锚点 58。四段均有正负例，达到预定的阶段 A 探索性工程门。
- 本次总墙钟约 35.49 秒，进程峰值 RSS 458,884 KiB。原始逐样本 JSONL、逐段摘要、输入哈希和设备记录均在上述数据盘成功目录；协议与代码在 Git 工作树中。
- `PYTHONPATH=src:scripts` 下 13 项分量合法性、身份标签、首轮适配器及自然历史回归测试通过；`git diff --check` 通过。第三方 OC-SORT 的 NumPy 标量转换产生 4 条既有弃用警告。
- 新鲜同系列 `experiment-audit` 为 provisional `WARN`，记录在 `docs/audits/r1_stage_a_20261118/`。审阅重算四个 JSONL 的计数、排序、标签算术、GT 可见性和 12 个输入哈希均一致；未发现假 GT、自归一化或虚构结果。

### 限制与精确下一步

- 审计指出 164 个动作中有 64 个分量没有任何 `t−1` 可锚定的分量内目标，且这 64 个净标签全为零；真正至少覆盖一个分量内锚点的为 100 个。所有 164 个虽有某些 GT 锚点，但不应称为 164 个直接受监督的动作。分支逐帧输出框未归档，审计不能独立重算每个标签。
- 结果是四段 fit 视频、一次确定性工程 smoke、自定义短程代理；不说明选择器可学习、闭环有效、HOTA/AssA 改善或研究新颖性。协议本轮仍未提交，Git 尚不能独立证明其冻结时间。
- **下一步：** 先把分量内锚点覆盖率、不可评价动作处理及每视频上限写进阶段 B 的新冻结协议，完成 Event-aware Learning 等近邻全文/源码比较，并预先固定留出集评判门和预算；然后才可扩样、训练线性选择器与即时标签/简单规则对照。本轮不启用 GPU、不恢复 P1、未创建 commit 或 push。

## 2026-11-18：R1 阶段 B 训练内留出诊断与停止

### 完成内容

- 在标签运行前固定 `manifests/protocols/r1_stage_b_20261118.yaml`，SHA256 `6b601deb6e5aebe3d1cc7449170c00367505b4d5c1dca25cc2a36eba3bb43fde`：24 fit / 8 留出沿用阶段 A，逐帧最多一个分量、每段最多 600、全局最多 20,000，H=10；零分量内锚点动作保留并单列；25% 同覆盖率比较与八项阶段 C 门槛预定。
- 补读 Event-aware Learning 与 DeconfuseTrack 正文相关方法段落，补充到 `docs/r1_stage_a_literature_and_protocol_20261118.md`。近邻已经使用模拟 tracker、事件监督和分解式关联，本轮不宣称 R1 新颖性；可核对源码不足的限制也明确写出。
- 新增 `scripts/run_r1_stage_b_labels.py`，在固定检测缓存上对 32 段 motion-train 执行真实 B0 状态与单分量 CV 动作的成对回放，同时记录未来 10 帧/当前帧标签、分量内锚点覆盖率和仅来自当前/过去的 13 个特征。每视频按哈希抽存 5 组成对分支的完整输出，便于独立重标。
- 新增 `scripts/analyze_r1_stage_b.py`，仅在 24 fit 段拟合归一化、Ridge 净收益和 Logistic 损伤模型；对八段留出按视频固定 25% 覆盖比较未来标签、即时标签、简单 IoU 差值与 20 个固定随机种子。没有调参搜索或留出阈值调整。
- 32 段共 34,231 帧，1,519 个原始分歧分量，12 个窗口尾部排除，1,426 个候选帧均执行（fit 1,106、留出 320）；931 个动作有分量内 GT 锚点，495 个无。标签采集墙钟 374.03 秒，峰值 RSS 458,884 KiB。使用现有 `mot-det` 的 CPU，`CUDA_VISIBLE_DEVICES=''`，GPU0 和其他 GPU 均未使用。

### 结果与核验

- 留出 320 个候选中，各方法按视频共选择 83 次：未来标签线性 +168、即时标签线性 +60、简单 IoU 规则 +239、20 种随机选择均值 +18.85（均为重叠短窗口的身份正确帧数差之和，不是完整闭环指标）。未来方法的受益/受损锚点 35/12，简单规则 51/18。
- 阶段 C 八项预定检查中六项通过、两项失败：未来方法相对简单规则的每动作净差为 `−71/83≈−0.8554`，未达到 `+0.1`；仅在 1/8 段留出视频严格胜过简单规则，未达到 6/8。**阶段 C 门槛 FAIL；不运行自然闭环。** 逐视频数字、口径和路径见 `docs/r1_stage_b_report_20261118.md`。
- 原始 JSONL、逐段摘要与审计分支存档：`/media/imc/data/mot-data/runs/r1_stage_b_labels_20261118/`；模型/排序/门槛：`/media/imc/data/mot-data/runs/r1_stage_b_linear_20261118/analysis.json`。协议、脚本及报告留在 Git 工作树，数据与分支输出未放入仓库。
- 新鲜同系列 `experiment-audit` 为 provisional `WARN`，见 `docs/audits/r1_stage_b_20261118/`。审阅独立核对 1,426 行标签、96 个输入哈希、拟合权重、八段方法及门槛；抽存的 160 组成对分支重新用原始 GT 标注，未来/即时标签均 0 差异。未发现会逆转门槛失败的错误。
- 阶段 B 两脚本编译检查通过；阶段 A 相关 13 项分量、标签和适配器回归测试此前通过。`git diff --check` 通过。

### 限制与精确下一步

- 这是固定动作集与线性模型的一次内部留出诊断；83 个被选择动作的窗口可能重叠，不能将净帧数相加解释为在线同时干预的收益。8 段留出仍来自已使用的 motion-train，且部分视频仅有十多个候选。未触碰 motion-dev 或官方 val。
- 1,266 组未抽存逐帧分支输出，需重跑才能独立重标；运行摘要的代码/依赖版本未嵌入，但本次报告单列脚本 SHA256。没有第三方源码级完整新颖性认证。
- **下一步：** 按原 R1 停止规则关闭当前“未来标签线性选择器→完整闭环”路线，不扩大到 MLP、不放宽门槛、不事后挑视频。保留分量动作和成对标签基础设施；下一轮先基于当前失败原因与文献近邻重新选择研究问题并冻结独立协议。本轮未创建 Git commit 或 push。

## 2026-11-18：同步远端并独立复核 R1 阶段 B 证据（分析轮）

### 完成内容

- `git pull --ff-only origin main`：`f5c88f0` → `62bad22`（1 个提交，17 个文件，+1353/−4），同步后工作树干净。
- 通读新增的阶段 A 先例核查、阶段 B 报告、两份实验完整性审阅、两份冻结协议，以及 `scripts/run_r1_stage_b_labels.py`、`scripts/analyze_r1_stage_b.py`、`src/mot_research/analysis/r1_components.py` 与新增测试。
- 本副本运行在 Windows，`/media/imc/data/mot-data/` 不可访问；未连接服务器、未读取原始运行目录、未重跑实验。

### 验证证据

- 复核协议与脚本哈希：本机直接计算的 SHA256 与记录不一致；按 LF 规范化（去除 `\r`）后四个文件全部与记录吻合——阶段 A 协议 `b698941dbbea…`、阶段 B 协议 `6b601deb6e5a…`、`run_r1_stage_b_labels.py` `7de6b3bc3115…`、`analyze_r1_stage_b.py` `188dbc169cd7…`。原因是本机 `core.autocrlf=true` 且仓库没有 `.gitattributes`，检出时 LF→CRLF；文件内容本身未变，但 Windows 检出下脚本内的协议哈希自检会失败。
- 按阶段 B 协议八项门槛与报告逐视频表复算：未来标签选择器 `+168/83`、简单 IoU 规则 `+239/83`、即时标签 `+60/83`、随机 `+18.85/83`；逐动作净差 `−71/83 = −0.8554`，严格胜出 `1/8`；八项检查六 true、两 false，与报告一致。
- 追加稳健性检查（只使用报告表中的逐视频汇总，不接触原始标签）：按选中数加权、以 8 段视频为重采样单位自助抽样 20,000 次（seed 20261118），逐动作净差均值 `−0.835`，95% 区间 `[−1.537, −0.061]`，`P(差 ≥ +0.1) ≈ 1.1%`；逐一剔除单段视频后仍在 `−0.57` 到 `−1.05`。`dancetrack0023` 占未来选择器 `+168` 的 55.4%、占 83 个选中动作的 32.5%，但门槛失败不是该视频单独造成的。
- 本机测试：`tests/test_r1_components.py` 3/3 通过；`tests/test_r1_stage_a_label.py` 因 `scripts/run_r1_stage_a.py` 顶层 `import resource`（Unix-only）在 Windows 无法收集；其余测试依赖外部 OC-SORT 检出与 TrackEval，未运行。

### 限制与精确下一步

- 复核范围仅限 Git 内协议、脚本、报告与逐视频汇总；1,426 行标签、`analysis.json`、1,266 组未存档分支无法在本副本独立重算，审阅中的输入哈希只能与 LF 规范化内容比对。
- 自助检查是对已发布汇总的二次分析，不替代原始标签重算；短窗口重叠、固定 25% 覆盖率与 8 段留出的解释限制不变。
- 未修改协议、门槛、结果或既有文档，仅追加本条目；未创建 Git commit 或 push。
- **下一步：** 先决定新研究问题（台账要求有独立机制依据），决定前不扩大 R1、不放宽门槛、不事后挑视频。若要恢复跨平台可复现性，建议新增 `.gitattributes`（`* text=auto eol=lf` 或按类型指定）、在运行摘要中写入代码与依赖版本，并把 `resource` 导入改为平台保护；这些改动需另行确认后执行。

## 2026-11-18：DiffuTrack 全文核查与候选路线削减

### 完成内容

- 读取用户提供的 `s41598-026-50731-8.pdf`（Scientific Reports 16:19960，13 页）全文，核对方法、消融、诊断与局限。
- 该文即创新台账已登记的 DiffuTrack：MDM 用条件扩散在 4D 归一化框状态上生成多假设（L=8、DDIM 10 步），关联代价为 `1 − (1/S)Σ IoU(样本, 检测)`（Eq. 10），再与时间感知外观原型代价融合后做匈牙利分配。
- 结论：此前设想的 M2“解析混合 → 期望 IoU 打分”在机制上与该文 Eq. 10 相同（蒙特卡洛期望 IoU），不能作为方法创新；DiffMOT 亦已覆盖扩散运动预测。本轮未修改任何协议、门槛或结果。

### 关键读法与限制

- 该文未隔离“打分通道”：表 10 中纯生成输出与融合输出的 AssA 同为 50.1，差异在 DetA（75.4 vs 80.8）；表 11 显示 fallback 续轨触发于 31.6% 轨迹、平均 2.7 帧、73.9% 再关联；作者正文亦称 continue-under-uncertainty 是主要通路。故“更宽支撑集改善关联打分”未被直接证据隔离。
- 组件消融基线是 KF+CE（DanceTrack val 48.5 HOTA / 35.1 AssA），不是 OC-SORT（test 55.1 / 38.0），MDM 的 +9.7 HOTA 是对弱基线测得，且 TPCL 依赖外观。
- 覆盖率对比以扩散经验分位数 vs KF 模型协方差，KF CR@95%=42.1%，不是同口径校准比较；主表为单次运行、无置信区间；未提供代码链接。

### 对路线的影响

- 降级：M2（分布 → 期望 IoU）改为对照条件 S1，不再作为贡献；M3（学习分布 + 联合训练）先例密集、成本高，不作为主线。
- 保留并收窄：同一组假设下的**决策规则**对照（单样本 IoU、均值 IoU、最大 IoU、混合似然、边际感知分配），以及用成对分支归因“打分 vs 续轨”两条通道。这是该文未做的消融。
- 本仓库设置无外观、固定检测缓存，不能声称复现或超过 DiffuTrack，只能作为机制层面的受控研究。

### 精确下一步

- 执行方案已写入 [决策规则对照执行草案](next_branch_decision_rule_plan_20261118.md)（2026-11-18 19:23 记录）：P0 服务器事实核查 → P1 离线重排 → P2 闭环 → P3 通道归因；其中数值门槛为草案，运行前须另写 YAML 协议并冻结。

## 2026-11-18：R1 回放脚本的可复现性小修（服务器工作树）

### 完成内容

- 新增 `.gitattributes`，对 Git 识别的文本文件固定 LF。检查全部已跟踪工作树文件，CRLF 文件数为 0，因此无需批量改写或重新归一化。
- 新增 `src/mot_research/analysis/runtime_record.py`：未来 R1 阶段 A/B 标签运行和阶段 B 分析在启动时记录 Git HEAD、Python 可执行文件/版本、NumPy/SciPy/FilterPy/lap/PyYAML 版本及所用源码 SHA256；输出前再次核对源码哈希，若运行中被修改则失败。标签运行还记录 OC-SORT 与适配器源码哈希。
- 运行摘要中的峰值 RSS 改由平台保护函数提供：不支持 `resource` 的平台返回 `null`；macOS 的字节单位换算成 KiB；Linux 保持 KiB。此改动不涉及动作、标签、模型或门槛计算。
- 旧阶段 A/B 的原始摘要和已审计结果保持原样；新字段只在后续新运行中出现。此前报告保存的脚本哈希对应当时已执行代码，不能用本次修改后的文件哈希替代。
- 拉取远端 `d1f4e6b` 前备份本地本条日志，快进后重新追加；三个脚本、`.gitattributes`、运行记录工具及测试文件的拉取前后 SHA256 均相同。

### 验证与限制

- 15 项 R1 分量/标签/适配器/运行记录测试通过；模拟 `resource` 不可用、macOS 单位换算及运行中源码变更均已覆盖。三个脚本及新工具编译检查通过。
- `git check-attr` 确认 README、Python、YAML 的 `text=auto`、`eol=lf`；已跟踪文件 CRLF 数为 0。实际 `mot-det` 环境快照可采到 Python 3.10.22 和五个关键依赖版本，并核对 12 个标签运行相关源码哈希。
- 没有重跑研究实验或生成新研究结论；GPU0 与其他 GPU 均未使用。历史审计中的未存档分支与旧摘要 provenance 限制仍在。
- **下一步：** 按远端新增草案先做 R2 的 P0 事实核查，不运行 tracker；依据候选分数是否已存决定 P1 能否直接离线重排。当前工程修复仍留在工作树，未创建 commit 或 push。

## 2026-11-18：拉取远端决策规则草案并完成 R2 P0 事实核查

### 完成内容

- `git fetch` 后确认远端新增 `d1f4e6b`：`README.md`、`docs/progress_log.md` 和新建 `docs/next_branch_decision_rule_plan_20261118.md`。与本地未提交工程修复仅在进度日志的追加位置重叠；先备份该段日志，再 `git pull --ff-only origin main` 快进 `62bad22 → d1f4e6b`，最后恢复本地日志。脚本、新工具、测试和 `.gitattributes` 的拉取前后 SHA256 均保持不变。
- 通读新增草案：新候选不是重训生成器，而是在同一运动假设集下比较 S0 点 IoU、S1 均值 IoU、S2 最大 IoU、S3 混合似然及 S4 边际感知分配；计划次序为 P0 事实核查、P1 离线重排、P2 自然闭环、P3 打分/续轨通道归因。草案门槛尚未冻结，不能直接运行。
- 执行草案 §6 的 P0 **只读核查**，记录在 `reports/engineering/r2_recon_20261118/README.md`。旧机制诊断 CSV 为 38 列、4,986 行（831 事件 × 6 方法），只含注册目标对与最强竞争轨迹的摘要，没有完整候选分数矩阵、所有候选框或 KF 协方差，故 P1 不能从旧 CSV 直接重排；需先冻结协议并做八段 motion-dev 的 CPU 候选集重放。

### 验证与限制

- 旧 CSV SHA256 `d08df0baec806d732fa04b463d073b0d0b9bb6ac5b3954fa1249fae77354a73e` 与其 `summary.json` 一致；八段检测缓存逐文件 SHA256 与旧 `input_hashes.json` 一致。三份 P1 checkpoint 仍在且各自 SHA256 与旧登记一致；OC-SORT checkout 为登记的 `bb2f6867a41082558d95ab47a854a3918de7831b`。
- 旧事件分层为 699 个 recovery failure、81 个 success control、51 个 persistence censored；831 中有 2 个 B0 目标轨迹死亡而不可评价。新协议必须分开登记总事件、各层与可评价分母，不能以 699+81 充当总数。
- P0 只读文件和环境元数据，未启动 tracker、训练或 GPU 作业；没有产生新研究结果。远端草案中的 DiffuTrack 解释由其文献核查给出，本轮只确认新执行次序及本机输入事实，不认证创新性。
- **下一步：** 在开始 P1 前写并冻结 R2 重放/打分协议，明确重放捕获的完整候选字段、S0–S4 公式、随机种子、分母、预算、验收与停止条件；然后才可 CPU 重放并离线重排。当前不运行 P1/P2/P3 或官方 val，本地工程修复与 P0 报告均未 commit/push。

## 2026-11-18：完成 R2 P1 事件帧采集与离线规则门槛

### 完成内容

- 先落盘冻结 `manifests/protocols/r2_decision_rules_p1_20261118.yaml`（SHA256 `1c69a19b5a3f8429a436700aed8942afdd6f798430ffa5b7d0a2424254ccef3d`），登记 831 个固定事件、8 段 motion-dev、KF 20 样本+2 转向模式、S0–S4 精确规则、随机种子、分母和 P1/P2 门槛。执行只用 CPU，`CUDA_VISIBLE_DEVICES=''`；GPU0 及其他 GPU 均未使用。
- 新增 `scripts/capture_r2_decision_states.py`，B0 独占轨迹状态，旁路取得 CV/三个 frozen checkpoint 的预测；捕获 749 个不同事件帧，覆盖全部 831 事件。逐视频核对 B0 回放帧/ID 与参考一致、坐标误差 ≤1e-4；全量状态与重放轨迹在 `/media/imc/data/mot-data/runs/r2_capture_p1_20261118/`。
- 新增 `scripts/rescore_r2_decision_rules.py`，按冻结规则生成 3,745 条事件帧×变体完整候选矩阵与 20,775 条事件×变体×规则排序结果，原始数据在 `/media/imc/data/mot-data/runs/r2_rescore_p1_20261118/`。主指标为注册正检测的目标轨迹是否排首位，只是离线代理，不是匈牙利分配、闭环恢复率或 TrackEval。
- 在 CV 主变体的 829 个可评价事件中，S1 成功 545、失败 284。S2 救回 57/伤害 38，S3 为 91/63，S4 为 64/39；三者救回均超过 10%，均在 7/8 视频有救回，但对 S1 成功事件的新增伤害为 7.0%、11.6%、7.2%，全部高于预定 2% 上限，**P1 晋级门槛 FAIL，不运行 P2**。逐视频表和解释边界写在 `reports/feasibility/r2_p1_decision_rules_20261118/README.md`。
- 新鲜同系列 `experiment-audit` 只读复算 831/749/20,775/3,745 的覆盖、完整矩阵的获胜轨迹与门槛数字，结论 provisional `WARN`；原结果中的 `winner_margin` 与协议要求的目标 margin 不同。新增 `scripts/derive_r2_target_margin.py`，从原矩阵另存 20,775 行正确的次要 margin 至 `/media/imc/data/mot-data/runs/r2_rescore_p1_20261118_secondary_margin/`，不改原始文件或主门槛。审阅详见 `docs/audits/r2_p1_20261118/`。

### 验证、限制与精确下一步

- 三个新脚本编译通过；`git diff --check` 通过。独立核对 20,775 个唯一结果键、3,745 个矩阵尺寸及 B0/CV 的 S1–S4 完全一致。两次主运行记录了 Git HEAD `d1f4e6b`、Python/依赖版本及源码哈希；采集约 170 秒、评分约 81 秒。脚本运行中的源码哈希复核通过。
- B0/CV 的 S1–S4 使用同一组 KF+转向假设，CV 点框只影响 S0；不能把 S1–S4 的变化归因给 CV。指标是基于已固定 GT 锚定事件的离线排序代理，不能推断闭环 AssA 或官方 val。协议尚未提交 Git，哈希不能独立证明落盘先于结果；峰值 RSS 与精确 CPU 核时未采集。补算 margin 是次要字段，不能改变 P1 门槛结论。
- **下一步：** 按预定停止规则保留本次阴性晋级结果，不进行 P2 或事后调参。下一轮研究决策可评估“关联打分与续轨贡献归因”，但执行前需另立并冻结 P3 的对照协议。当前没有创建 Git commit 或 push，工作树保留供用户审阅与记录。

## 2026-11-18：归档并推送 R1 工程修复与 R2 P1 记录

- 用户确认本轮没有待跑的实验后，复核 15 个待提交文件：仅包含 `.gitattributes`、R1 运行来源记录修复、R2 协议与脚本、P0/P1 报告、审计记录和进度日志。权重、检测缓存、原始运行 JSONL/CSV 与数据集均不在提交清单，仍位于数据盘或被 `.gitignore` 排除。
- `origin/main` 与提交前本地 `main` 同步；以 `AmazeZhang <1109600167@qq.com>` 配置提交 `5b0bd15`（`research: record R2 P1 decision-rule gate and provenance fixes`），已推送至 `origin/main`。本条为提交后的归档补记。
- 提交前 `git diff --cached --check` 通过，R1 相关 6 项测试通过；R2 三个脚本编译及协议、采集、评分、次要 margin 附录的哈希/状态核对通过。没有重新运行研究实验。
- **下一步：** 保留 P1 门槛失败的停止决定。若转向“打分与续轨贡献归因”，先评估研究问题与可执行对照，再冻结新的协议；GPU0 继续禁止使用。

## 2026-11-18：毕业设计阶段状态核对

> 历史记录说明（2026-11-19）：本节描述仓库更新前 `f5c88f0` 的检查；当时未见 R1 结果。之后工作区已同步到 `502d7df`，R1/R2 完成情况以上文实验记录及下文最新复盘为准，不能将本节当作当前阶段状态。

### 已完成核对

- 用户报告已在服务器开始执行 R1 建议。本次只读取本地仓库和 GitHub 默认分支进展日志，未连接服务器或操作运行任务。
- 本地 HEAD 为 `f5c88f0`；核对时本地与 GitHub `docs/progress_log.md` 的 Git blob SHA 均为 `e49a615482087950c3a9410e3b05a435509c804b`，最新记录仍为 2026-11-13 路线建议。该一致性只证明日志一致，不证明服务器没有新工作。
- 本地 `reports/`、协议及实现文件清单中未找到 R1 的已完成运行凭证；R1 状态记为“用户报告已启动，运行阶段与效果待核验”，不能将建议中的 200/20,000 样本等数字当成已执行配置。
- 重读 P1 自然闭环 `gate_summary.json`，确认 KF/CV/learned 三 seed 的官方指标及恢复损伤计数与既有复盘一致。

### 当前成果与边界

- 可复用工程成果：环境、数据和检测缓存、官方基线与评测、真实历史接口、共同事件、作业和结果记录。
- 已有研究证据：G2 的有限首轮位置相关空间；P1 离线改善；P1 自然闭环负结果；同 B0 状态单步相对 CV 更好而完整闭环更差。最后一项仍未建立具体因果机制。
- 尚无已证明有效且具备明确创新的新跟踪方法。FM/Gaussian/MDN 没有在本项目中完成方法效果验证；它们不是被当前 P1 试验普遍否定。
- 现有开发结果可作为毕业设计的实验平台、问题分析和对照材料；核心方法贡献与确认实验仍待完成。

### 精确下一步

- 获取服务器 R1 实际协议、运行摘要与失败记录，先核验双分支隔离和标签定义，再检查有效样本数、收益/损伤/无变化分布以及相同干预量下的简单规则对照；只有随后出现自然闭环收益，才升级为方法有效证据。
- 在报告同步前，不推断 R1 已通过工程门、完成训练或产生提升。本次仅追加状态交接记录，未启动任务、修改代码、创建 commit 或 push。

## 2026-11-19：工作区更新复盘与下一步重心

- **完成工作：** 基于 HEAD `502d7df` 阅读 R1 阶段 B、R2 P1 报告、冻结协议、审计摘要和相关实现；核对新增决策规则草案。未找到独立标注的 DeepSeek 原始对话，本次评价针对落盘建议。
- **验证证据：** R1 留出等量 83 次动作，未来标签线性 +168，简单 IoU +239，原门槛失败；R2 S2/S3/S4 救回 57/91/64、伤害 38/63/39，超过 2% 上限，保留停止结论。上述为报告与实现核对，未在本机复算服务器原始产物。
- **关键发现：** 固定事件中 829/831 个目标轨迹仍存活；既有 B0 575 次首轮正匹配中只有 158 次重连。在线读取固定提交的 OC-SORT 源码，并核对本地运行器和事件评价代码：默认 `min_hits=3`、缺失后连续命中重置及输出门，构成“正确匹配但未输出”的具体候选机制。尚未逐事件测得其占比，不将 417 次差额全部归因于输出抑制。
- **解释修正：** R1/R2 的失败是特定晋级规则失败，不是所有价值学习或分布规则无效；fallback 统计不能证明续轨贡献；三条件不足以分解完整二因素交互；汇总 HOTA 不能直接由汇总 DetA/AssA 平方根乘积作严格贡献分解。
- **产物：** `docs/experiment_review_20261119.md`，含事件链路核对 E0、B0/CV × 原输出/成熟轨迹即时输出的完整四条件 E1、后续分支与预算。同步 README 当前状态，在旧草案及更新前状态检查处增加历史说明，保留原文与本次之前的未提交内容。
- **限制：** 本次无服务器连接、无实验重跑、无模型训练；Nature 网页后续表格访问遇登录重定向，未独立重核先前 PDF 全文数字。E0/E1 是建议，尚未冻结、实现、运行；没有新的方法有效性或新颖性结论。
- **精确下一步：** 服务器先冻结新的探索性 E0 协议，零改动重放全部 8 段 B0，记录内部匹配、更新、输出和 GT 评价之间的断点；有输出抑制实例才执行预先冻结的 E1。合计实施预算 2–3 个工作日；CPU、`CUDA_VISIBLE_DEVICES=''`，产物留数据盘，不重启 FM/R1/R2 大矩阵。
- **本地检查：** `git diff --check` 通过；README、最新复盘及旧草案三份文档的本地链接均可解析；复算 R1 每动作差 −0.8554、R2 净救回 19/28/25、伤害比例 6.97%/11.56%/7.16%、B0 首轮正确但未重连差额 417，与报告一致。另读取固定版本 TrackEval 源码确认 HOTA 按阈值计算。复盘形成时仅修改文档，尚未创建 commit 或 push；随后用户明确要求提交并推送本次安排更新。

## 2026-11-19：同步远端并理解 E0/E1 下一轮实验

- 本地工作树原本干净，`git fetch` 后确认远端仅有一项新增提交 `a6dd87d`，以 `git pull --ff-only origin main` 从 `502d7df` 快进至 `a6dd87d`。新增 `docs/experiment_review_20261119.md`，并更新 README、旧 R2 草案状态说明和进度日志。
- 新路线聚焦“观测缺失后的内部匹配、状态更新、实际输出、GT 评价之间的断点”。先冻结探索性 E0 协议，对 8 段 motion-dev 的 B0 零改动回放覆盖 831 事件及其后至少 5 帧；逐事件记录存活、入池、首轮/最终配对、更新、输出门与 GT 对实际输出的评价。若证实存在匹配后输出抑制，再冻结并执行 E1 的 B0/CV × 原输出/成熟轨迹即时输出完整 2×2。E0/E1 建议合计 2–3 个工作日、CPU-only，`CUDA_VISIBLE_DEVICES=''`；旧 R2 P2/P3 不沿用，也不启动 FM/R1/R2 大矩阵。
- 从数据盘原始 `event_method_mechanism.csv` 和固定 `events.csv` 独立核对：831 事件中 B0 首轮正确匹配 575 次；这 575 次中 158 次满足重连指标，另有 1 次重连不属于首轮正确匹配集合。`src/mot_research/evaluation/closed_loop_events.py` 的重连指标要求重现帧 GT 对实际输出匹配到原 ID；`third_party/OC_SORT/.../ocsort.py` 的正常更新路径默认 `min_hits=3`，输出要求当帧有更新且 `hit_streak` 达门槛（启动期例外）。这些事实只支持 E0 的候选解释，不能把 417 次差额全部归因于输出门。
- 本轮仅同步与分析，没有冻结新协议、实现 E0/E1、重跑 tracker、运行评测或使用任何 GPU。远端新文档是执行建议，未证明新方法收益。**下一步：** 若开始实施，先把 E0 的事件分母、互斥断点分类、GT 匹配口径、零改动回放验收、固定抽样规则、预算和输出哈希写成冻结协议，再进行 B0 CPU 回放；只有 E0 观察到输出抑制实例才进入 E1。此本地补记未提交或推送。

## 2026-11-19：E0 事件链路与 E1 成熟轨迹输出规则四条件实验

### 已完成

- 按 11 月 19 日复盘，先落盘 E0 协议 `manifests/protocols/e0_reassociation_output_chain_20261119.yaml`（SHA256 `e587c07b4aa47f8f14f6b59a3837395a397920cbb5111ad4b45ce211db03eca7`）。B0 对 8 段 motion-dev 完整零改动回放，逐段与历史轨迹帧/ID 精确一致、框误差 ≤1e-4；GT 只在回放结束后用于标签。831 事件的互斥链路类别为：目标不存活 2、正检测未入池 0、未由正检测更新 242、更新后未输出 425、输出后 GT 未分配原 ID 4、无断点重连 158；重连总数仍为 159（另 1 个事件由其他检测输出而重连）。425 个更新后未输出中 424 个曾成熟输出，达到预定 E1 触发条件。运行 `/media/imc/data/mot-data/runs/e0_reassociation_output_chain_20261119/`，轻量报告 `reports/diagnostics/e0_reassociation_output_chain_20261119/README.md`。
- 在 E1 运行前落盘四条件协议 `manifests/protocols/e1_mature_reemit_factorial_20261119.yaml`（SHA256 `0aba57b6214b45b6071cbc75d5fefda6be6425f696697b424aa23e264e00b249`）。A/B 共用一次 B0 执行，C/D 共用一次 CV 执行；U 只为过去确认输出且当帧真实高分检测更新、原输出门省略的轨迹多发布检测框与原 ID，逐帧核对生成 U 不改变序列化轨迹状态。A/C 的 16 段输出均与历史 B0/CV 参考一致；B/D 分别新增 2,595/2,310 行。轨迹与来源 `/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119/`。
- 固定版本 TrackEval 的 `COMBINED_SEQ`：A(B0 原) HOTA 50.060 / AssA 31.663 / IDF1 49.554；B(B0 U) 50.549 / 31.649 / 49.710；C(CV 原) 51.057 / 32.624 / 49.606；D(CV U) 51.534 / 32.659 / 49.803。B−A 的 HOTA +0.489、AssA −0.014，不符合继续判据；D−C 的 HOTA +0.477、AssA +0.035、IDF1 +0.197，符合“下一轮简单基线”判据。但 B−A/D−C 的 FP +343/+328、IDSW +130/+91、Frag +102/+174，必须与收益一同报告。固定事件的五帧持续成功 A/B/C/D 为 81/328/84/328；全部逐视频数字见 `reports/feasibility/e1_mature_reemit_factorial_20261119/README.md`，原始评测在 `/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119_eval/`。
- 新鲜同系列 `experiment-audit` 主审阅 provisional `WARN`：主门槛、官方汇总、831/3324 行事件与参考回放均可复算，范围仅 8 段已使用的开发视频。审阅发现 E1 主运行漏存协议中的内部首轮配对辅助计数；补做只读 CV 回放及 CV 自身 ID 标签附录并经独立复核通过。固定事件 B0 首轮正配对 575/831；CV 缺失前自身 ID 有 GT 锚点 806/831，其中首轮正配对 494/806。机械比较 CV 与 B0 数字 ID 的 102/831 不是 CV 正确率。附录在 `/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119_internal_pairs_addendum/` 与 `/media/imc/data/mot-data/runs/e1_mature_reemit_factorial_20261119_internal_pair_labels/`，审阅记录在 `docs/audits/e0_e1_20261119/`。

### 验证、限制与精确下一步

- 新脚本编译检查通过；E0/E1 全部 CPU-only，`CUDA_VISIBLE_DEVICES=''`，GPU0 与其他 GPU 均未使用。E0 墙钟约 370 秒，E1 轨迹约 670 秒、评测约 30 秒；运行摘要均记录 Python/依赖/源码哈希、墙钟、CPU 时间与峰值 RSS。输出文件留数据盘，仓库只有协议、代码、轻量报告与审计。
- E0 分类是“首断点”而非重连失败分区；不能把 425 个无输出或 575−158 的差额都解释为同一因果机制。U 不改内部配对，HOTA 增益不能声称内部关联改善；CV 的 AssA 增益仅 +0.035 点，FP/IDSW/Frag 代价明显。8 段训练内开发集已经被先前工作使用，未做官方 val、独立种子或新颖性核查。补录内部配对是事后描述，不改变冻结 E1 判据。
- **下一步：** 保留 CV+U 作为有代价的简单强化基线。若研究下一方法，应先围绕“成熟轨迹再输出时如何降低误恢复/IDSW”做专项文献重合检查与可观察风险分析，再冻结一个最小反事实实验；不直接把 U 包装为核心创新，也不自动启动 FM/扩散/MDN 或官方 val。工作树保持未提交，未 push，供用户通过 Git 审阅记录。

## 2026-11-19：E2 再输出风险分析与两个简单门控对照

### 完成内容

- 对 E1 已冻结的全部额外输出做行级风险分析，数据盘 `/media/imc/data/mot-data/runs/e2_reemit_risk_20261119/`。CV+U 新增 2,310 行的固定 GT IoU 0.5 代理口径为 2,061 TP/249 FP；其中 `score < 0.8` 的 1,390 行含 227 FP，`score >= 0.8` 的 920 行含 22 FP。该阈值是看过同一开发集标签后选择的，属于探索。
- 核查 OC-SORT、Deep SORT、ByteTrack 官方实现，确认连续命中确认、置信度分层与再激活均有成熟先例；E2 不提出新方法主张。随后冻结 `manifests/protocols/e2_reemit_guard_20261119.yaml`（SHA256 `479e2e4203ac31be867620b9a7bb1dac7fe495e90051de9dfa4265f369dc131f`），只筛选 E1 CV+U 新增行：E 为分数≥0.8，F 为当前连续命中≥2，C 原行全保留，不重跑或改变 tracker 内部状态。
- 官方 TrackEval 聚合：C 原输出 HOTA/AssA/IDF1 为 51.057/32.624/49.606，IDSW/FP 为 725/427；D 无门控为 51.534/32.659/49.803，816/755；E 分数门控为 51.356/32.688/49.761，764/469；F 连续命中门控为 51.312/32.647/49.708，774/527。E/F 均达到预写的开发集效用门：相对 C 三指标增加，相对 D IDSW/FP 减少；但 E/F 的 HOTA 低于 D，且相对 C 仍有额外 IDSW/FP。固定 831 事件五帧持续成功 C/D/E/F 为 84/328/167/89。完整逐视频和其它指标在 `reports/feasibility/e2_reemit_guard_20261119/README.md`；原始轨迹与评测在 `/media/imc/data/mot-data/runs/e2_reemit_guard_20261119/`。
- `experiment-audit` 新鲜同系列只读审阅 provisional `WARN`：73,658 行 C 完整保留，E/F 新增 920/1,056 行与规则精确一致；官方汇总和 831 事件复算一致。警告针对同一开发集 GT 风险标签参与阈值选择、缺少独立确认，以及未提交协议的冻结时间只能由文件时间/哈希支持。审阅记录 `docs/audits/e2_reemit_guard_20261119/`，trace 在 Git 忽略的 `.aris/traces/experiment-audit/2026-11-19_run01/`。

### 验证、限制与精确下一步

- 两个新脚本编译通过；E2 CPU-only，`CUDA_VISIBLE_DEVICES=''`，GPU0 与所有其他 GPU 均未使用。E2 输出摘要 SHA256 `8cad3a38720d8bb40df5c65a8fa092cc35e3356b88dab20facb30b396d80627f`，风险行表 SHA256 `b8887cf3589ad29a768c8e199365e4fb3111fb54c8883346a467b9dc8de73273`。数据盘产物 11 MB + 308 KB；未复制权重或数据到 Git。`git diff --check` 通过。
- 本轮只是一组同视频、同检测的输出门控探索；新增行 TP/FP 是自定义代理，不能与官方 FP 对号入座。不能声称内部匹配改善、阈值最优、独立验证或新颖性。E 保留为下一轮简单基线，当前 U 无门控仍有更高 HOTA；两者形成明确风险收益对照。
- **下一步：** 停止在这 8 段上继续调输出阈值。若要继续做毕业设计方法，先用 train-fit 定义一个针对重现身份可靠性的机制和最小反事实，冻结 C/D/E 这几个强简单对照与停止门，再在未参与该规则选择的范围做确认；不自动启动大模型训练或官方 val。当前工作树保留未提交、未推送，供 Git 审阅记录。

## 2026-11-19：E3 短缺失空间支持 fit 实验，未通过晋级门

### 已完成

- 按 E2 下一步，核对 R1 已固定的 24 段 train-fit / 8 段 train-heldout、检测缓存和 CV 回放接口；检索 ConfTrack、ByteTrack 等置信度与空间关联先例，明确本轮不是新颖性主张。先冻结 `manifests/protocols/e3_short_gap_support_20261119.yaml`，SHA256 `5bcb1ea1d12548bc253ed32deb1824cd3eb109bf80c206c256e2db94cf67a86d`，再运行 24 段 fit。规则 G 在 E 的分数≥0.8 基线外，仅补当前分数 0.6–0.8、距同 ID 上次原输出≤5 帧、检测框与上次原输出框 IoU≥0.5 的成熟轨迹；C 原输出、D 无门控 U、E 分数门控同台。所有条件共用一次 CV tracker 状态，只改变输出。
- 24 段共 25,328 帧、C 原输出 182,933 行；D/E/G 分别新增 2,854/1,458/2,045 行。固定 TrackEval 汇总 HOTA C/D/E/G 为 59.790/60.110/60.011/60.061；AssA 为 41.218/41.286/41.282/41.285；IDF1 为 58.368/58.497/58.471/58.491；IDSW 为 928/1001/977/979；官方 FP 为 415/745/502/586。G−E HOTA +0.050 低于冻结的 +0.100 最小收益，FP +84 高于冻结的 +50 上限；尽管 21/24 段 HOTA 略升、AssA/IDF1 略增，**fit 晋级门 FAIL**。按停止规则未运行 8 段 heldout、未改阈值。逐视频表和边界见 `reports/feasibility/e3_short_gap_support_20261119/README.md`，原始产物在 `/media/imc/data/mot-data/runs/e3_short_gap_support_fit_20261119/`。
- 新鲜 `experiment-audit` 同系列只读审阅 provisional `WARN`：源/输出哈希、96 份四条件轨迹精确嵌套、2,854 条新增行规则、官方数值与门槛均独立复算。审计发现脚本在每段输出后、全部输出完成前对 GT 文件计算 SHA256，违反冻结协议 `GT_read_only_after_outputs_complete` 的字面时间要求。没有 GT 标签解析或规则泄漏证据，但偏离必须保留；详见 `docs/audits/e3_short_gap_support_20261119/`。已执行脚本不事后改写，未来同类协议应在全部轨迹落盘后才计算 GT 哈希。

### 验证、限制与精确下一步

- 脚本编译通过，协议与来源哈希核对通过；fit 墙钟约 1,199 秒，CPU 核时约 0.333 小时，峰值 RSS 458,884 KiB。运行摘要 SHA256 `24aa6db797fc3eef9a52e8513624c1f9e5fa8fe7f74a2117d22d4e9ef8752507`。`CUDA_VISIBLE_DEVICES=''`，GPU0 与其他 GPU 均未使用。heldout 目录不存在；`git diff --check` 通过。
- 24/8 划分先前用于 R1，E 的 0.8 门槛也来自已看过 GT 风险标签的 motion-dev。E3 只说明这条固定 G 规则在 fit 上未满足预写收益/代价门，不能否定所有时空证据或确认 E 的泛化。G 只改输出，不修复内部匹配；文献重合高，不可包装为毕业设计方法贡献。
- **下一步：** 停止 E3 这一输出补回候选并保留阴性结果。下一阶段若继续研究身份问题，应围绕剩余未更新/错匹配事件定义一个直接干预关联或状态写入的最小反事实，先限定动作与先例，再冻结与 C/D/E 的对照；不要在相同 fit 上调本次阈值、启动 E3 heldout 或自动使用官方 val。工作树未提交、未推送，供用户通过 Git 审阅记录。

## 2026-11-19：E0–E3 主张收缩与 E4 正确再关联特权反事实

### 完成内容

- 使用 `result-to-claim` 新鲜同系列审阅 E0–E3，结论 provisional `claim_supported=no`：成熟轨迹即时输出能改善开发集可见连续性与部分 HOTA，但按构造不改变内部配对；E2 的分数门控有代价，E3 的短缺失空间规则未过 fit 门。U/G 作为身份方法候选停止。决策写入 `docs/e0_e3_claim_gate_20261119.md`、`findings.md` 与 `CLAIMS_FROM_RESULTS.md`。
- 为回答剩余错配是否有值得研究的空间，冻结 E4 GT 特权局部五帧反事实：从 E3 C 原输出与 GT 选出成熟轨迹重现缺口，每段按固定 SHA256 最多取 50 个候选；正常 CV+E 与一次强制正确一对一配对从同一预帧状态分叉，后四帧恢复原关联和 E 输出。这个动作仅用于构造性 headroom，不是可部署方法，也不产生官方完整序列 TrackEval 指标。
- E4 首次、`retry1`、`retry2` 分别在工程验收中因 OC-SORT 原地修改共享检测数组、全局出生 ID 计数器串扰、评价锚点误用 E 而失败；各失败目录保留错误、部分进度及当时执行脚本。修复分别写入新协议，最终 `manifests/protocols/e4_oracle_association_headroom_20261119_retry3.yaml`（SHA256 `1e286bee37b6fc0aec90b865f23df8aecab56f568be1218774adff2647208068`）完整运行，未覆盖失败产物。
- E4 retry3 覆盖 24 段 train-fit：1,903 个候选缺口、固定选中 1,073 个；617 个原回放已由正检测更新、2 个不符合成熟定义，余下 454 个有效错配事件。GT 特权配对后的五帧目标同 ID 正确输出净增 1,426 帧（每事件 +3.141），全锚点净增 1,643 帧（每事件 +3.619）；目标受损 9/454，其他锚定身份受损事件 37/454。五项预写 headroom 门均 `PASS`，完整逐视频表见 `reports/diagnostics/e4_oracle_association_headroom_20261119/README.md`。成功运行 `/media/imc/data/mot-data/runs/e4_oracle_association_headroom_20261119_retry3/`，摘要 SHA256 `628e979e86a0478f15f69df2a040192986d84fc0044b83b8e801bc71e772f7dd`。
- `experiment-audit` 新鲜同系列 provisional `WARN`：原始事件哈希、1,903/1,073/454 分母、目标/全锚点计数、正常分支逐帧回放与门槛均复核；独立重放一个 oracle 事件吻合。454 个五帧窗口对应 2,270 次暴露、1,750 个不同视频帧，其中 387 个帧被多次使用；全部 oracle 轨迹未逐帧归档，逐项独立复算仍需重跑。审阅见 `docs/audits/e4_oracle_association_headroom_20261119/`。随后 `result-to-claim` 给出的 provisional 判定仅为条件化 GT 特权短程 headroom `yes`、信心中等，见 `docs/e4_claim_gate_20261119.md`；不能升级为在线算法、完整序列指标或新颖性。

### 验证、限制与精确下一步

- E4 脚本编译通过；成功运行 24 段父回放与 E3 C/E 逐帧帧号/ID 精确、框误差≤1e-4，五帧正常分支同样一致。454 个事件键唯一，24 份 JSONL 哈希与摘要相符，独立加总重得 +1,426/+1,643、9/37；`git diff --check` 通过。retry3 墙钟约 1,528 秒、CPU 核时约 0.424 小时、峰值 RSS 463,212 KiB；`CUDA_VISIBLE_DEVICES=''`，GPU0 与其他 GPU 均未使用。
- 分母先由 GT 与 C 定义，再条件化到正常回放失败；强制配对本身读 GT，且只测相互可重叠的局部窗口。不能把 454/1,903 当自然错误率、把每事件收益外推成 HOTA、把“headroom”当数学最优上界，或推断线上可观察特征足以选对。未跑 8 段 train-heldout、motion-dev 或官方 val，也未训练新模型。
- **下一步：** 只设计并运行一个锁定的**非 GT 信息源**实验：从当前/过去图像裁剪提取固定廉价外观表征，在全部在线可观察候选中选择至多一个一对一纠正，与 CV+E 和配平干预预算的简单 IoU/margin+检测分数规则比较；24 段 fit 定唯一阈值，8 段 train-heldout 锁定后一次评测。若留出集全锚点净收益和损伤不能胜过最强简单对照，停止身份关联候选；不在留出集反复调参，不启动 FM/MDN 或官方 val。工作树保留未提交、未推送，供用户通过 Git 审阅。

## 2026-11-20：E5 非 GT 外观选择器配平实验，fit 门失败

### 已完成

- 按 E4 的下一步，先固定 `manifests/protocols/e5_nonoracle_appearance_local_20261119.yaml`（SHA256 `62b3b98f875cad1d3610bdc28f91e359edfe3b53ffc3585ec82e87ec5c8a3942`），再运行 `scripts/run_e5_nonoracle_appearance_local.py --partition fit`。候选只依赖原 CV 的 C 输出、检测缓存和当前/过去图像；每段按固定 SHA256 最多取 30 个首次可观察缺口。HSV 裁剪外观与简单 IoU/检测分数对照在同一事件、同一预帧状态上各强制一次一对一首轮配对，后四帧恢复 CV+E；GT 仅在全部动作及原始分支轨迹落盘后用于评分。
- 24 段 train-fit 共有 1,515 个可观察首次缺口，固定选中 664 个；87 个未过外观门槛，1 个目标不成熟或不存活，**576 次配平动作**，其中 47 次选出不同检测。五帧全锚点净正确帧相对原 CV+E：几何 −1,317、外观 −1,323，外观−几何 **−6**；其他锚点受损事件几何 420、外观 422。目标有历史 GT 锚点 529 次，当帧有目标 GT 正检测 233 次，选中正检测几何 102、外观 99。预写 fit 门要求外观−几何每动作至少 +0.10 帧且其他锚点损伤不更多，两项均失败；动作量与分歧量门槛通过。**fit 研究门 FAIL**，按协议未运行 8 段 train-heldout、未调阈值。
- 完整逐视频表、方法和范围见 `reports/feasibility/e5_nonoracle_appearance_local_20261119/README.md`；原始产物及两支五帧逐事件输出在 `/media/imc/data/mot-data/runs/e5_nonoracle_appearance_local_fit_20261119/`，摘要 SHA256 `7ec8d4dde1b2d041f100bb150abf99289386553b34093e2c5074e8762a3ebdbb`。新鲜同系列 `experiment-audit` provisional `WARN`：24 份原始/评分文件及来源哈希吻合，全部 576 个事件由另一套 IoU/匈牙利实现独立重算，0 处不一致；WARN 针对拟合集、重叠局部分支和无完整序列指标。审计见 `docs/audits/e5_nonoracle_appearance_local_20261119/`。`result-to-claim` 新鲜同系列审阅给当前外观优于简单规则主张 `no`，拟合门失败判定信心高；决策见 `docs/e5_claim_gate_20261120.md`、`findings.md`、`CLAIMS_FROM_RESULTS.md`。

### 验证、限制与精确下一步

- 执行脚本编译通过；父回放逐帧与 E3 C/E 帧/ID 一致、框误差≤1e-4。24 段原始/评分文件和运行摘要哈希通过独立审计，留出目录不存在；`git diff --check` 通过。fit 墙钟约 1,420 秒、CPU 核时约 0.395 小时、峰值 RSS 481,312 KiB。`CUDA_VISIBLE_DEVICES=''`，物理 GPU0 与其他 GPU 均未使用。
- 只对哈希选中的 664 个可观察事件尝试分支，且动作触发由外观置信度决定；576 个窗口可重叠，不能当成独立视频重复。两条强制规则都明显伤害原 CV+E，不能据此概括所有外观特征无用、所有几何方法更优，或推断完整序列 HOTA/AssA/IDF1。E4 的 GT 特权收益不能转移为 E5 的非 GT 收益。协议落盘和运行时间有本机记录，但未提交协议无法独立证明预注册时间。审阅为同系列 provisional；通用 `evidence_check.py` 不可用。
- **精确下一步：** 停止当前固定 HSV 选择器，不运行它的 heldout、不在同一 fit 调门槛或启动 FM/MDN。先对 47 个两规则分歧动作与双方共同损伤做只读机制分析，区分正确目标正检测是否存在、原配对是否已正确、颜色相似导致的混淆与强制动作本身的损伤；只有找到可在线观测且与当前规则实质不同的机制，才拟定新协议和新 fit 门。若没有新机制，暂停这条身份修复分支。工作树保留未提交、未推送，供用户通过 Git 审阅记录。

## 2026-11-20：E5 失误只读拆解与正常首轮配对补录

### 完成内容

- 在查看全量事后分组前登记 `manifests/protocols/e5_readonly_error_analysis_20261120.yaml`（注明已看过一个样例行），对 E5 原始 576 个动作进行只读拆解；`scripts/analyze_e5_errors.py` 复核 E5 各段原始/评分文件哈希和汇总，并将 47 个分歧、414 个两规则均损伤其他锚点的事件索引保存在数据盘。529 次同选检测的两支五帧输出逐项相同；47 次分歧的外观−几何正确帧差为 14 次正合计 +47、13 次负合计 −53、20 次零，净 −6。
- 为消除原始 E5 未保存首轮配对的限制，另登记 `manifests/protocols/e5_baseline_pair_trace_addendum_20261120.yaml`（SHA256 `287a29d01bf91414480dd331db9da9e75f689b9b74e6f77a3fe7c19670185e33`），用 `scripts/derive_e5_baseline_pair_trace.py` 对 24 段原 CV+E 做无 GT、无强制动作的只读重放，逐帧与 E3 C/E 的 ID/框核对，再对 576 个事件补录正常首轮配对。事后连接与标签复核脚本为 `scripts/summarize_e5_pair_trace.py`、`scripts/classify_e5_error_labels.py`。
- 576 次中 47 次无历史目标 GT 锚点；另有 107 次锚定目标当帧不在 GT、64 次当前无 IoU≥0.5 的高分检测、125 次有重叠检测但匈牙利全局匹配分给别的 GT。故“无被分配目标正检测”的 343 次不是 343 次检测器漏检。余下 233 次有已分配正检测：正常首轮已配给目标 121、配给别的轨迹 105、未配对 7。两规则选中的检测在正常首轮已属于其他轨迹分别为几何 567/576、外观 566/576；在 414 次共同其他锚点损伤中，两个分支各有 400 次覆盖一个与检测 GT 身份的历史 C 锚点一致的原配对，交集 398、并集 402。这是动作冲突的直接证据，但不能独自解释全部五帧损益。
- 完整表、定义、限制与下一步见 `reports/diagnostics/e5_error_analysis_20261120/README.md`。机器产物位于 `/media/imc/data/mot-data/runs/e5_readonly_error_analysis_20261120/` 和 `/media/imc/data/mot-data/runs/e5_baseline_pair_trace_addendum_20261120/`，后者原回放 `summary.json` SHA256 `c3795d00fa85ab0fdae90fb2d8fc88c987bc40f1c6eb8057f88d4cddb51488f0`、连接摘要 SHA256 `b1feb884612d188e6299ab5edb5ee7feb15e0396a697e013273648bcfd4e6fd0`。新鲜同系列 `experiment-audit` provisional `WARN`：576 个唯一事件连接、来源哈希和五帧评分独立复算一致；警告针对拟合集事后范围与因果措辞，见 `docs/audits/e5_error_analysis_20261120/`。

### 验证、限制与精确下一步

- 四个诊断脚本均编译通过；正常父回放 24 段逐帧与 E3 C/E 匹配，全部 576 个事件键、所选检测 ID 与保存的 E5 文件相同；独立抽样 CPU 回放核对了一例事件全部 7 个首轮请求框/分数与检测 CSV。细分标签文件 `positive_label_and_displacement_check.json` SHA256 `e8d67b290494938ad79732280cdfab7b241fd2ded61dba288eae959cf6b6ad66`。首轮补录墙钟约 1,153 秒、CPU 核时约 0.320 小时、峰值 RSS 485,052 KiB。所有执行均设 `CUDA_VISIBLE_DEVICES=''`，物理 GPU0 和其他 GPU 均未使用；E5 heldout 仍不存在。`git diff --check`、Markdown 本地链接和审计 JSON 解析通过。
- 576 个五帧窗口可能重叠；“其他锚点受损”两支可能是不同 GT，审计在 414 次共同受损中发现 26 次受损身份集合无交集。`zip(高分 det_id, 全部缓存行)` 的索引写法假设高分行在低分行前；本轮 576 个动作帧的高分 ID 顺序均经直接 CSV 检查，261 帧含低分行且无反例；未来若换缓存需先修复此输入假设。首轮请求框只在一例独立重放中全量核对，不声称 576 次请求框都被逐项复核。GT 事后标签不能变成在线触发规则，覆盖已有正确首轮配对与五帧损伤也不等于完整因果归因。
- **精确下一步：** E5 当前选择器和留出集继续停止；不在同一 fit 根据上述 GT 分组调阈值，不凭此启动 FM/MDN 或官方 val。当前没有已验证的在线安全触发信号，身份关联分支暂缓。若后续要继续，先从独立来源提出实质不同的可观察触发或允许拒绝动作的机制，并单独冻结新协议、简单对照和 fit 门；未来复用检测缓存代码前修复高低分行索引假设。本轮工作树保留未提交、未推送，供 Git 审阅记录。

## 2026-11-20：两周整体进展与方向收束复盘

- **完成工作：** 在本地 HEAD `cdac367`、初始干净工作树上，核对 E0–E5 报告、协议、主张决策、审计摘要及 E5 候选/选择和 E4 强制配对实现；结合既有 P1/R1/R2 评估整体研究进展。没有连接服务器、重跑原始产物或新开实验。
- **积极成果：** E0 将匹配更新后无输出与真实错配区分；CV 是较强简单对照；CV+分数门控 E 在 8 段 motion-dev 与不重叠的 24 段 fit 上分别有约 +0.300/+0.221 HOTA，同时保留 IDSW/FP 代价。E4 支持 GT 条件化的一次正确关联有五帧收益，但不支持可部署方法。工程与诊断平台可复用，尚无已确认的新方法主线。
- **关键解释：** E5 排除当前已与目标配对的检测，外观门通过后强制选择替代检测；121/576 次正常首轮已正确，529/576 次两个选择器同选，47 次分歧中仅 5 次注册正检测在替代池内。共同动作损伤大于 HSV 与几何差异，不能扩大为外观信息或关联学习普遍无效。旧 STOP 结论不变。
- **核对数字：** 529/576=91.84%；121/576=21.01%；567/576=98.44%；E4 全锚点净帧 1643/454=3.619；E3 E−C 由报告舍入值复算约 +0.221 HOTA。这些只是已报告数据的算术核对，不是本机独立实验复现。
- **产物：** `docs/research_status_review_20261120.md`；README 增加整体复盘入口。文档列出成果层级、负结果边界、E4→E5 的触发条件差异、开发与确认的区分，以及最多三个待文献核查的不同研究问题。
- **限制：** 本轮是现有证据复盘，未进行新一轮文献检索、创新审查或模型验证；候选问题不能当作新颖性承诺。E5 heldout 此次未运行不代表该划分从未被 R1 使用；现有开发数据不能重新声称独立确认。
- **精确下一步：** 收束当前局部规则链，先为最多三个机制不同的候选核查近邻工作、公开代码、信息来源、强简单对照和可证伪小实验，再选一条实现；不默认追加 E5 修补或 FM 训练。按要求不提交、不推送，保留可审阅文档变更。

## 2026-11-20：结合实验负结果的文献发散与最小验证设计

- **完成工作：** 采用 paper-search 与 hypothesis-generation 工作流，运行三组关键词、六源 API 检索，逐项阅读去重结果的可用摘要；另定向查阅原始论文及作者仓库。整理 3 项 API 保留记录与 15 项影响选题的补检材料，核查 MOTIP、Deep OC-SORT、BUSCA 文档及 LA-MOTR 代码入口。只发送通用研究关键词，没有上传实验数据。
- **实质发现：** MOTIP 已有历史遮挡/身份交换增强；MOTIP2 已加入空间相关身份交换；2019 exposure-bias 与 2022 event-aware 工作已有自身错误或模拟候选训练。HMP、LiAM-SAM、SAM-Deep-EIoU 分别限制了泛化的可靠记忆、生命周期记忆及选择性掩码验证创新主张。故没有把“真实错误训练”“置信度写入”直接认定为新颖。
- **建议：** 首先验收固定检测下的深度外观对照与 MOTIP 官方推理，再验证竞争身份污染的结构化训练假设；允许保持原配对的冲突集合决策为低成本备用，可靠记忆/分割与 FM 暂缓。明确同检测比较、完整序列评价、其他身份损伤、配平增强、数据/预训练泄漏、视频级重复与开发预算；不撤销旧 STOP。
- **产物：** `调研/研究方向发散_2026-11-20/方向评估与最小验证方案.md`、同目录 `allinone.md`；README 增加入口。原始 API JSON/错误日志仅在忽略目录 `local/research_20261120/`。MOTIP 训练文档 blob `63933e23ebc3fb9b6ed515cdd48a8c1a0a17a5c4`，BUSCA README blob `0c9b1686c78fae619857d84425f731f3d78f3677`，不混称仓库版本或模型哈希。
- **验证证据：** API 原始 12 条、去重 6 条、语义保留 3 条；API 有限流/解析失败，错误原文已归档。代码核查仅到文档/链接层：BUSCA 训练章节仍 Coming soon，LA-MOTR 入口返回 404，MOTIP2 未核实可用官方代码/权重。未下载权重、安装环境、运行推理或训练，没有使用任何 GPU。文档本地链接与 `git diff --check` 经检查。
- **限制：** 这不是系统综述或完整新颖性审查，2026 部分工作仅预印本；外部论文指标均不是本项目复现。训练日程是规划预算，显存/时长未测。官方训练权重可能见过本项目内部划分，不能用于宣称内部留出泛化。没有已经证实的新方法。
- **精确下一步：** 冻结 S0/S1 执行协议：选定官方特征/模型版本，审计权重训练数据，固定检测与输出政策，列出单视频验收和同检测深度外观对照；随后再配置隔离环境并开展服务器推理。根据新基线自然错误决定是否训练候选 A，不能直接把调研候选转成完整训练清单。本轮保留工作树，未提交、未推送。

## 2026-11-20：服务器执行就绪性检查

- **完成与证据：** 逐项检查方向评估文档，确认其为研究设计，尚缺第三方 commit/权重锁定、输入清单、完整对照配置、适配脚本、运行命令和可执行验收条件；在文首明确标记“尚不能直接作为服务器实验手册”。
- **限制：** 本轮未补齐上述实现，也没有执行环境安装、模型下载或服务器实验；不能把调研中列出的 S0/S1 日程当成已验收操作步骤。仅修改 Markdown，`git diff --check` 通过。
- **精确下一步：** 先完成 S0/S1 专用执行手册、冻结配置和必要适配脚本，区分本地可核验部分与必须在服务器完成的 CUDA/权重验收，再交付逐步命令。用户当前可盘点 GPU 占用、磁盘、驱动和现有数据路径。未提交、未推送。

## 2026-11-20：S0/S1 A2A 任务书、固定设计与外观核心代码

- **用户分工：** 本地仅补核心方法/代码与任务定义，服务器 agent 实现详细脚本和环境；通过 GitHub 仓库中的文档、配置、代码和结果交接。没有启动其他 agent、发送远程消息或访问服务器。
- **完成内容：** 新增 `docs/a2a/S01_SERVER_TASK.md`、`docs/a2a/s01_server_return.md`（NOT_RUN 模板）、`manifests/protocols/s01_visual_association_handoff_20261120.yaml`。设计锁定 G=CV+E、Z=接入但权重零、L=最近深度特征、M=EMA 深度特征，L/M 权重 0.25、EMA 旧记忆权重 0.9；固定首轮注入，保持检测/输出/后续关联不变。不训练 A/B 或 FM、不跑旧 heldout 或 val/test。运行前完整 commit、权重/预处理哈希仍由服务器按规则补齐，不冒称运行协议已全部验收。
- **核心代码：** `src/mot_research/association/appearance_core.py` 提供同掩码筛选检测 ID/框、无效特征显式回退、零权重恒等分数融合、严格 begin/finish 帧顺序的 last/EMA 记忆；记忆按真实最终匹配（含出生）写入，删除清除、缺失保留。没有实现图像网络、全量回放器或新训练方法。已通过 GitHub 原始源码核查固定 OC-SORT 的分数插入点，来源 blob `78111b75419b761d63f35dc1b2301230e7e209c5`。
- **验证证据：** `tests/test_appearance_core.py` 的 11 项 unittest 全部通过，覆盖高低分交错 ID、重复/非法 ID、无效与超大特征、零权重、竞争配对排序、无效边、因果记忆、原子失败/重试、删除/缺失、EMA 与空输入。使用本机捆绑 Python 3.12.14 / NumPy 2.3.5，`CUDA_VISIBLE_DEVICES=''`；另验证 Python 3.10 AST 语法兼容，不声称已在服务器 Python 3.10 运行。PyYAML 6.0.2 仅安装到忽略的 `local/s01_validation_deps` 用于配置检查，没有改变项目依赖。
- **协议核查：** R1 划分文件哈希仍为 `6b601deb6e5aebe3d1cc7449170c00367505b4d5c1dca25cc2a36eba3bb43fde`，24 个唯一 fit 视频；按新协议固定哈希选得 pilot 为 `dancetrack0051/0016/0072/0052`。协议 SHA256 `f99c994a2ab622f3e6f7b1156a1b94410ab53c1e79275aa3076e6e57700ffa5c`；YAML/组配置断言、相对文档链接与 `git diff --check` 通过。
- **限制：** 无实际检测/图像、权重提取、GPU、完整 tracker 或 TrackEval 验收；单元测试是工程证据，没有新研究指标。官方 MOTIP 与固定检测外观分支分开验收，一个依赖失败不阻塞另一个。当前外观公式是简单基线，不是 Deep OC-SORT 全系统复现，也不代表候选 A/B 创新实现。完整命令和集成测试由服务器 agent 补充。
- **精确下一步：** 用户将本批工作树同步到 GitHub 后，让服务器 agent 从任务书开始，自主固定运行身份、配置隔离环境、实现约定的提取/回放/汇总入口并完成 S0 工程门；随后按协议运行四段 pilot 与全部 24 段固定开发对照，将完整结果填入回传模板和进展日志。CPU/GPU 预算耗尽保留 partial 并回传，不自动扩展研究范围。遵循 AGENTS.md，本轮未提交、未推送，未声称 GitHub 已同步。

## 2026-11-20：拉取 S0/S1 远端安排并核对服务器就绪性

- **完成内容：** 工作树原本干净，执行 `git fetch origin main` 与 `git pull --ff-only origin main`，从 `cdac367` 快进到 AmazeZhang 的 `7ac7261`。阅读 S0/S1 任务书、协议、回传模板、研究状态评估、核心代码与测试；修正任务书中已过时的“尚未推送”交接描述。当前本地修改仅为任务书这一说明和本进展记录，未提交、未推送。
- **安排理解：** S0 先锁定运行版本、输入与官方 Deep OC-SORT DanceTrack 外观提取器/权重来源、SHA256、预处理及环境；用 `dancetrack0027` 第 1–200 帧验证 G/Z 与旧 E3 E 的工程等价，MOTIP 官方单视频推理单独验收。S1 在同一固定检测与 CV+E 输出下比较 G（无外观）、Z（零权重接入）、L（最近特征）、M（EMA 特征）：先四段 pilot，再 24 段 train-fit，按协议回传逐视频与汇总指标、受影响配对及成本。此次不训练新方法，不跑旧 E5 heldout 或官方 val/test；物理 GPU0 始终禁用。
- **验证与证据：** 当前 `HEAD=7ac72619515fcd30c4dc5fe29facf5197585bcbf`；协议 `manifests/protocols/s01_visual_association_handoff_20261120.yaml` SHA256 `f99c994a2ab622f3e6f7b1156a1b94410ab53c1e79275aa3076e6e57700ffa5c`；R1 划分哈希与协议一致，24 段 fit 和 pilot `dancetrack0051/0016/0072/0052` 核对一致；检测清单、E3 fit 参照与 `dancetrack0027` 图像/检测缓存均存在。以 `CUDA_VISIBLE_DEVICES=''` 和现有 `mot-det` Python 3.10 环境运行 `test_appearance_core.py`，11 项全部通过。OC-SORT 构造函数会重置全局出生 ID 计数器，和逐视频独立回放的要求一致。未使用任何 GPU。
- **限制与精确下一步：** 运行身份锁文件、特征提取器、集成回放与评测入口仍待实现，回传模板仍是 `NOT_RUN`；以上检查不等于 S0/S1 实验完成。下一步按任务书先完成 S0 的来源/权重/环境锁定与最小工程门，确认 G/Z 逐帧等价后再启动 S1 pilot 和 24 段 fit。所有大型产物放 `/media/imc/data/mot-data`，只在空闲的物理 GPU1–7 上做 CUDA 工作。

## 2026-11-20：S01 S0 单视频工程验收

- **完成内容：** 官方 Deep-OC-SORT 与 MOTIP 固定 commit 并检出到数据盘，下载各自 DanceTrack 权重并记录 SHA256；创建隔离外观/MOTIP Python 3.12 环境，保留现有 Python 3.10 回放环境。新增外观缓存入口、S01 固定对照回放入口、OC-SORT 源 det_id 窄补丁及四项集成测试；运行锁放 `manifests/runtime/`，大型产物放 `/media/imc/data/mot-data/runs/s01_visual_association_20261120/`。核心与集成共 15 项测试通过。完整表及来源见 `reports/feasibility/s01_visual_association_20261120/README.md`。
- **工程验收：** `s0_smoke2` 的 200 帧 1,542 条特征有效，前 100 键缓存/直接重算一致；`s0_visual_smoke1` 中 G 与归档 E3 E 的 1,504 行帧/ID 一致、框差≤1e-4，Z 与 G 的首轮配对/最终更新/输出逐帧一致；L/M 均实际进入非快路径加分并发生各 2 帧自身基准首轮配对变化。MOTIP 官方 `submit` 在同一 train 视频只读视图上跑完 200 帧，1,673 行合法输出，峰值监测 1,339 MiB。两分支均未加载 GT 计算研究指标，CUDA 仅显式使用物理 GPU1，GPU0 未使用。
- **失败与限制：** 首次缓存直接检查因单张与批量 FP32 数值差异超容差停止，首次 G 回放因无记忆模式回调标记错误提前停止，失败目录保留；修复均以新 run_id 重跑。MOTIP 首次运行自动下载的 ResNet50 权重已迁移数据盘，第二次锁定后重跑。官方高维梯度全套测试未完成，只记录通过的前向与前五组梯度。S0 是工程门，S1 pilot/24 fit 与 TrackEval 尚未运行，不能主张外观提升或新方法有效。
- **精确下一步：** 为四段固定 pilot 创建完整输入/代码/权重运行锁并提取全视频特征，四组独立跟踪回放；核对 G/Z、特征记录和资源预算后继续全部 24 fit。轨迹与 trace 先落哈希，GT 只在后续独立评测阶段读取。当前工作树未提交、未推送。

## 2026-11-21：S01 四段 pilot 完成，继续全部 fit

- **完成与证据：** 固定四段 `dancetrack0051/0016/0072/0052` 共 5,772 帧，GPU1 上顺序提取 32,942 条有效特征，0 无效，每段前 100 键缓存/直接重算通过；CPU 独立回放 G/Z/L/M 全部完成。G 与历史 E3 E、Z 与 G 的工程门在四段均通过。轨迹和 trace 的 32 个哈希先落 `s1_pilot_evaluation1/frozen_inputs.json`（SHA256 `122e4be6e18db1915980a1017ceff3748f5d85e4a5d896292c291e949f312aa4`），之后 TrackEval 完成；摘要 SHA256 `997202cdd6646d56faf1f8bcf3a0e8b5844cb85fd42464885366a8277a083e71`。
- **pilot 描述值：** COMBINED_SEQ HOTA G/Z 52.78、L 56.12、M 61.48；AssA 31.58/35.72/43.21，IDF1 49.56/53.85/59.21（G/Z、L、M）；L/M 各有 22/24 帧自身无外观首轮配对变化。完整逐视频与成本见 `reports/feasibility/s01_visual_association_20261120/README.md` 和数据盘 `s1_pilot_evaluation1/summary.json`。这是复用 train-fit 的描述性基线，权重训练重叠未排除，不能据此主张泛化或新方法有效。
- **资源与下一步：** pilot 特征抽取约 168 秒墙钟，四组跟踪 CPU 核时合计约 1.14 小时，远低于 24 GPU 小时/96 CPU 核小时预算。根据预写规则，工程门通过后无论 pilot 指标方向都继续全部 24 段；已预登记其余 20 段的 19,556 帧输入哈希，并启动 `s1_remaining20_features1` 与独立 `s1_full_tracking1` 队列。每段状态和日志单独保存，失败不会覆盖其他段；完整 24 段评测尚未运行。GPU0 继续禁用，未提交、未推送。

## 2026-11-21：S01 全部 24 段 fit 完成、重评与完整性审阅

- **完成与证据：** 固定 24 段 DanceTrack train-fit 共 25,328 帧，物理 GPU1 顺序提取 186,676 条有效 FP32 外观特征、0 无效；四组 CPU 闭环均通过工程门。24 段 G 轨迹与归档 E3 E 逐字节相同，24 段 Z 与 G 逐字节相同。L/M 分别在 132/138 帧实际改变首轮配对集合。权威评测 `s1_full_evaluation2/summary.json` SHA256 `b3585deb2eebc5132ac898478a69d8bdb803f9b989c0bdaf8c88ea0bf6ddabb0`，`frozen_inputs.json` SHA256 `ed50243913084d3df2840423feb2bb3ffb3b010667dbc8cf4ebf8d483340fcc2`，评测运行锁 `manifests/runtime/s01_visual_association_s1_full_eval2_20261121.json` SHA256 `473a092e59f06033419f2bf2f145e5476e871f9fdbc923c2b721c9d0b7fab460`。192 个轨迹/trace 文件和 24 个特征缓存哈希先冻结，再读 GT 评测；重评锁还核对 TrackEval 本地补丁、46 个 Python 源文件、投影脚本及 24 个 GT 文件。原始结果均在 `/media/imc/data/mot-data/runs/s01_visual_association_20261120/`。
- **描述性结果：** TrackEval COMBINED_SEQ HOTA G/Z `60.01`、L `60.74`、M `62.47`；AssA `41.28/42.32/44.60`，IDF1 `58.47/59.33/61.03`（G/Z、L、M），M−G IDSW `−75`。M 对 G 的逐视频 HOTA 18 段上升、6 段下降。pilot M−G HOTA `+8.70` 点在完整 24 段收缩为 `+2.46` 点。完整逐视频表、来源与限制见 `reports/feasibility/s01_visual_association_20261120/README.md`；S0/S1 完成事实、环境和路径已填入 `docs/a2a/s01_server_return.md`。
- **重评原因与完整性：** 首次全量评测 `s1_full_evaluation1` 锁定了 TrackEval commit，但遗漏本地兼容补丁及投影辅助脚本的完整哈希；首评保留。补锁后从相同轨迹重评，results/deltas/costs/视频顺序与首评精确相同。新上下文 GPT-6.1-Sol ultra 独立审阅给出同家族暂定 **WARN**：GT、分数和原始结果核对通过；范围是复用 train-fit，权重训练重叠未知，且历史 pilot 特征锁中一个不参与提取的回放脚本哈希过时。详见 `reports/feasibility/s01_visual_association_20261120/EXPERIMENT_AUDIT.md` 和 `.json`。审阅 trace 保留本地 `.aris/traces/experiment-audit/2026-11-21_run01/`。
- **成本、验证与限制：** 24 段特征抽取墙钟累计约 742 秒、模型调用约 373 秒，PyTorch 峰值分配约 710 MB；四组跟踪合计约 5.14 CPU 核时，首评 TrackEval 约 74 秒墙钟。L/M 回放包含额外无外观反事实关联和 trace 写入，不能解释为部署 FPS。核心/集成 15 项测试再次通过，`git diff --check` 与 OC-SORT 补丁检查通过。没有独立 heldout、val/test、随机种子实验或新方法训练；不能主张泛化或新方法有效。物理 GPU0 全程未参与。本轮未提交、未推送。
- **精确下一步：** 根据全量自然错误与上述限制，审议候选 A 的最小训练/验证协议，预写成功和停止条件，并先解决权重训练数据重叠的解释问题；任何新实验另建运行锁和范围，不直接将本轮开发集描述值当方法结论。

## 2026-11-21：S01 主张门与 S02 自然身份更新只读诊断

- **主张门：** 使用 `result-to-claim` 新上下文 GPT-6.1-Sol ultra 复核 S01 原始摘要、96 个条件/视频 trace 和完整性审阅；对“已证明竞争身份记忆污染且应直接训练候选 A”的主张判 `no`，高置信度认为**当前证据不足以支持该机制或直接训练**，同家族暂定且 S01 完整性状态 `WARN`。S01 仍支持固定操作点上视觉历史有描述性信息。结论已写 `CLAIMS_FROM_RESULTS.md` 与 `findings.md`，审阅 trace 在本地 `.aris/traces/result-to-claim/2026-11-21_run01/`。通用 `evidence_check.py` 帮助程序不可用，S01 另有独立原始文件/官方 TrackEval 审阅。
- **S02 预写与执行：** 在查看 S02 GT 诊断结果前创建 `manifests/protocols/s02_natural_identity_mechanism_20261121.yaml`（SHA256 `39c7c147734dedfca28a148ae576f49488545f9060e9fda8f1a1409fb60084ea`）和运行锁；新增 `scripts/analyze_s02_natural_identity_mechanism.py`，用冻结 S01 G/L/M 的真实更新、24 段 GT 与高分检测做一对一 IoU≥0.5 离线身份代理，定义三次同 GT 更新的稳定历史、每轨首次异主更新、正确高分检测可见与活跃竞争轨迹代理、后续 30 帧错误更新。所有 S01 输出在 GT 读取前已冻结；S02 先写源输入哈希，再读 GT。全程 CPU、`CUDA_VISIBLE_DEVICES=''`，GPU0 未用；不训练或改 tracker。
- **结果与限制：** 权威 retry3 共 24 段、25,328 帧；M 的稳定历史后异主更新 732 次，每轨首次符合定义 278 次，其中 86 次同时有原 GT 正确高分检测和另一活跃外来 GT 锚轨迹，分布 22 段；75 次在至多 30 帧内至少又有一次原 GT 高分检测可见时的异主更新。预写投入门 20/6/10 通过，仅授权计划受控记忆写入测试。G/L 对应 89/88 个定义内事件。86 个 M 事件仅在 66 个不同视频帧，后续窗口有重复；未验证正确检测对目标轨迹的首轮合法边，不能称可修复事件或因果污染。原始结果与逐视频表见 `reports/diagnostics/s02_natural_identity_mechanism_20261121/README.md`。
- **失败、重评与审阅：** run1 源码锁未显式列 GT 对齐辅助脚本；原脚本字节快照留数据盘。retry2 在任何输出前因 `PYTHONNOUSERSITE=1` 下 SciPy 实际 1.13.1、锁误记用户站点 1.13.0 而失败，日志保留。retry3 补齐主脚本、两个辅助源码及 Python/NumPy/SciPy/PyYAML 版本校验，事件和逐视频文件与 run1 逐字节相同。权威 `summary.json` SHA256 `5cdb942ab76cd0b705fabeac8aca5db9608a784f800951441e4bf8b20aeb0f38`，`frozen_inputs.json` SHA256 `e5a3713010d90650fb2e49a1ffcc673792a14f61a3ca647603ef0a6064423a8c`，运行锁 SHA256 `99d760268e15109b519dc4450825931ab2cfedcf14d15c3309a9edcff3740a78`。新上下文同家族独立审阅重新实现全部 24 段 × G/L/M、843 个事件，537 项哈希无差异，余弦最大误差 `4.44e-16`，结论暂定 **WARN**（代理定义、重复窗口、非因果范围；无阻断数值 bug）。审计文件位于 S02 报告目录，trace 留本地 `.aris/traces/experiment-audit/2026-11-21_run02/`。
- **精确下一步：** 已写 `docs/a2a/S03_CONTROLLED_MEMORY_TASK.md` 设计草案。下一轮先冻结可复现的 M 状态克隆和合法候选筛选，再在自然错误帧只抑制一次错误视觉记忆写入，与配平的干净写入抑制对照比较 30 帧闭环的全冲突集合净收益；这是 GT 条件化诊断，不是在线算法或候选 A 训练。当前工作树未提交、未推送。

## 2026-11-22：S03 单次视觉记忆写入受控分支、严格对照重评与主张门

- **完成内容：** 使用 `CUDA_VISIBLE_DEVICES=''` 在 CPU 上先做 S03 单视频 M 状态克隆工程 smoke，再按冻结协议对 24 段 S01 train-fit M 输入执行完整回放，并在 S02 的全部 86 个定义内异主更新及 86 个配平干净更新处各抑制一次目标视觉记忆写入，闭环观察未来至多 30 帧。所有分支先于 GT 评分冻结，GT 仅用于离线事件/对照选择和事后评分。首版对照排除只覆盖 86 个 actionable 首次事件，40/86 个对照接近其他 M first-onset；保留首版后新增独立 retry2 协议及运行锁，改为避开同视频全部 278 个 M first-onset 的 ±30 帧。40 个对照更换，86 个错误事件及分支内容不变。retry2 在已知首版零结果后进行，未冒称盲化独立复现。原始运行在数据盘 `s03_single_memory_write_counterfactual_20261121/`；权威重评在 `/media/imc/data/mot-data/runs/s03_single_memory_write_counterfactual_20261122_retry2/`，摘要 SHA256 `12293779fa5ceb1930eb9f59d0c99f07ee670d5bcfb3a38cbf6e8c4c7cd472bc`，选择 SHA256 `5c71eafe1e3d11e4a021298b2f3490a6bcf238927434b809cd1babc01965bccb`，分支冻结索引 SHA256 `19320a027e1cd43b2604008eab0e53cf6bae2494b38d4c0a4ad3525eb8c9214b`。协议、运行锁、代码与完整方法见 `reports/diagnostics/s03_single_memory_write_counterfactual_20261122/README.md`。
- **结果与解释：** retry2 错误/干净组各 86 个，事件帧目标正检测合法边 70/85；172/172 个分支的未来输出行都无变化，目标、其他前帧输出锚点和全部锚点的正确帧净差均为零。172/172 次目标记忆向量扰动非零，实际后续融合分数探针有变化，故空结果不是未执行抑制或恒零评分。5,106 次未来帧暴露仅对应 3,562 个不同视频帧，93 对窗口重叠、最高 6 重，5 个窗口被终点截断。新对照成本中位数约 0.387、最大约 3.130，无平衡保证。此结果只支持冻结 M 下单次写入的局部观察空效应；不否定多次写入、较长期或错误 tracker 状态机制，也不是在线方法或完整序列 TrackEval。
- **独立审阅与决策：** 新上下文 GPT-6.1-Sol ultra `experiment-audit` 给同家族暂定 **WARN**，339 次哈希核对、24 视频 25,328 帧落盘 replay、172 个事实分支和独立 IoU/Hungarian 计分全部匹配；无假 GT、分数造高或幽灵结果，范围/依赖和非穷尽内部状态核验保留限制。报告见同目录 `EXPERIMENT_AUDIT.md/.json`，原始审阅 trace 在本地 `.aris/traces/experiment-audit/2026-11-22_run01/`。新上下文 `result-to-claim` 给目标主张 **no**、当前训练停止判断高信心（同家族暂定、完整性 WARN）；决策见 `docs/s03_claim_gate_20261122.md`、`CLAIMS_FROM_RESULTS.md`、`findings.md`，审阅原文 SHA256 `f3e7e0bb8035ae0429c5d9a23d32a15cfefa48daf7e8decfd4e29a057646ee74`。通用 `evidence_check.py` 不可用，另有上述确定性审阅。
- **验证与精确下一步：** 相关 `appearance_core`/S01 集成/natural motion policy 测试 `22 passed`；四个 S03 脚本 `py_compile`、审计 JSON 解析、`git diff --check` 通过。全程物理 GPU0 未用，未提交、未推送。**停止以“单次错写是主要后续 ID 错误原因”为依据的候选 A 训练；无需继续在相同 86 个事件上扩窗或调阈值。** 如继续身份关联方向，先设计另一个可证伪且在线可观测的机制，事前冻结其干预、干净更新代价、其他身份损伤和视频级成功/停止门；只有出现独立可复现的受控机制收益，再另拟配平训练和验证。

## 2026-11-23：推送 S01–S03 进度，完成 S04 有界记忆冻结实验与主张门

- **Git 记录：** 按用户本轮要求，先复核待推文件、忽略规则、文件体量和账号，再将 S01–S03 的轻量代码、协议、锁、报告与记录提交并推送至 `origin/main`：`c4c5fb8 research: record S01-S03 visual memory diagnostics`。没有将权重、原始数据或大型运行结果推入仓库。之后产生的 S04 代码和文档保持未提交、未推送。
- **S04 设计与数据：** 在提取新特征前锁定 `manifests/protocols/s04_bounded_memory_freeze_20261122.yaml`（SHA256 `b6e40f6a24d7c8f79a41e43e89ce21de403bc43f2872e01597c81285d04fd90f`）。八段 DanceTrack train 共 8,903 帧，虽与 S01/S03 fit 不重叠，但曾用于 R1，并非完全独立验证。运行根目录 `/media/imc/data/mot-data/runs/s04_bounded_memory_freeze_20261122/`。特征锁 `manifests/runtime/s04_heldout_features1_20261122.json`，8 段共 76,123 条有效 FP32 特征、0 无效、800 次直接/缓存核对通过；每视频独立状态、日志和输出保留。完整 M 基线 CPU 回放八段均通过。权威分支锁为 `manifests/runtime/s04_bounded_memory_freeze_retry4_20261122.json`（SHA256 `16ac218cf4122938b1fa032e4895233b6bfb1dda1a0aacad249b0626462869cb`）。此前三份分支预检锁因环境版本或协议输出路径字段错误在 GT 读取/输出目录创建前停止，锁与失败日志均保留，没有产出研究结果。
- **S04 原始结果：** 从八段 M 中记录 67 个首次异主更新；按预写条件选中 15 个错误事件（五视频）及 15 个同视频干净对照。每例从同一预帧状态跑事实、单次跳写、连续冻结三臂；连续冻结实际抑制错误组 10–31 次、干净组 26–31 次目标写入。30/30 例三臂输出逐帧相同，所有目标/其他/全部前帧输出锚点正确帧差为 0；900 次未来帧暴露对应 604 个不同视频帧。稀疏门通过，收益门失败，摘要决策 `NO_SIGNAL_STOP_CURRENT_RATIONALE`。原始 `branches/summary.json` SHA256 `589323a38141cb9964d5a0ee4eef6ee1514896fed64e097223a3de070920e128`；详情见 `reports/diagnostics/s04_bounded_memory_freeze_20261122/README.md`。
- **独立审阅与解释：** 新上下文同家族 GPT-6.1-Sol ultra `experiment-audit` 暂定 **WARN**：9,371 次哈希比较、8,903 帧独立基线回放、全部 30 个三臂分支和计分复核匹配；没有假 GT、造高分或死计分。30/30 个实际 runner 探针证实目标记忆和实际 LAP 分数变化；严格 owner-positive 竞争且连续相对单次分数变化只覆盖错误组 **14/15 例、四视频、133 帧**，不能把 15/15 合法边说成 15 次决定性机制机会。审阅在 `EXPERIMENT_AUDIT.md/.json`，本地 trace 在 `.aris/traces/experiment-audit/2026-11-22_run02/`。新上下文 `result-to-claim` 同家族暂定 **no / stop**、对当前训练停止判断信心 high，见 `docs/s04_claim_gate_20261123.md`、`CLAIMS_FROM_RESULTS.md`、`findings.md`；原文 SHA256 `ce0d3c69e4e7cf93f8c4e6b0528ed22f3441594c6490befedf95c141a5911ee5`。通用 `evidence_check.py` 不可用，另有上述确定性核验。
- **验证、硬件与限制：** S04 两项新单测通过；`appearance_core`/S01 集成/natural motion policy/S04 合计 **24 passed**，四项既有 NumPy 弃用提示；新脚本 `py_compile`、审计 JSON 解析、`git diff --check` 通过。仅特征抽取使用 `CUDA_VISIBLE_DEVICES=1` 对应物理 GPU1；基线、分支和审阅使用 `CUDA_VISIBLE_DEVICES=''` 的 CPU。**物理 GPU0 全程未使用**。这是 GT 特权局部诊断，不是在线方法、独立验证或官方完整序列评测；重叠窗口、对照平衡、锚点覆盖和保留的错误 tracker/CV 状态限制主张。
- **精确下一步：** 保留 S04 原始产物和审阅；按预写停止规则，不启动候选 A 训练，不在这 15 例上扩窗、扫阈值或筛选找正例。若继续身份方向，先提出与当前冻结不同的在线可观测机制，事前锁定真实身份竞争/assignment margin、干净更新代价、其他身份损伤、视频级门和独立数据；有受控机制收益后才另拟配平训练及验证。

## 2026-11-23：研究端接收 S01–S04，复盘积极结果与下一步重心

- **检查范围：** 从本地干净 HEAD `5f93d2a` 阅读 S01 回传、S01–S04 实验报告/审计、冻结协议与主张门，检查 S01 特征与分数注入、S03 状态分叉、S04 连续抑制实现。没有连接服务器或重跑原始实验，也没有新增文献检索。
- **主要判断：** S01 的深度外观 EMA 相对 G 在 24 段开发集上有 +2.46 HOTA、+3.32 AssA、+2.56 IDF1、IDSW −75，是值得保留的简单视觉关联基线；新颖性、训练数据重叠与泛化尚未解决。S03 172 个分支、S04 30 个三臂事件输出不变，支持停止当前记忆冻结理由；不支持否定首次错配前的视觉决策学习。138 个 M 变化帧的参照是其自身同状态去外观，不是独立 G 全程轨迹。
- **本地验证：** 四份权威运行锁 SHA256 与报告吻合；S02/S03/S04 主脚本与依赖共 27 次比较、13 个唯一项目源码文件均匹配，4 次第三方文件引用本地不存在而未核。四份审计 JSON 解析正常。S01 舍入表的指标差算术复核一致。原始数据盘 `/media/imc/data/mot-data` 不在本地，未独立重算指标或分支；服务器审计中的回放/哈希数量仅引用，不计为本轮执行。
- **产物：** `docs/research_status_review_20261123.md`；README 更新最新阶段、将 11 月 20 日标为历史；S01 回传增加后续状态导航，不改写原始实验结论。局部文档链接与 `git diff --check` 检查通过。
- **精确下一步：** 建议另立有界协议：利用 S04 现成特征/M 轨迹补八段 G/L/M 完整指标（开发稳定性检查，非独立测试）；用已有 S01/S02 状态拆分候选缺失、运动门控、外观误判、全局竞争及后续关联阶段问题，再决定合法分量决策或学习关联的最小对照。不要延窗/扫冻结参数或直接启动 A/FM。下一次 Git 回传补小体量机器可读 summary 与完整精度逐视频指标，便于研究端核查，不需要重新运行既有审计。本轮没有新实验、提交或推送。

## 2026-11-23：下一轮候选与实验报告规范 v1

- **完成内容：** 按用户要求在 AGENTS.md 增加每轮实验报告及 Git 交接规则；新增 `docs/experiment_reporting.md`、`docs/templates/report_bundle/`（两份说明、五个 JSON 骨架、三张 CSV 表头）和 `docs/a2a/LATEST.md`。规范要求完整精度总体/逐视频表、计划与实际覆盖、代码/输入/权重来源、失败/重试、GT 边界、案例及训练曲线；工程和纯诊断可按实际范围省略不适用指标并说明，不制造研究结果。模板是格式契约，不是已经实现的导出/验证程序。
- **下一轮建议：** `docs/a2a/NEXT_ROUND_20261123.md` 将 S05 限定为补 S01/S04 原始小型证据、八段 G/L/M 全序列指标与一次合法候选覆盖普查；S06 条件化设计为整组匹配的新增正确/损伤预测，小型 Local/Context 两模型 × 两种固定风险策略，含简单固定权重对照、数据使用/选模/多 seed/实际收益门及预算。未实现或执行 S05/S06，没有新的训练或性能结果；V1/V2 登记为 `NOT_SUPPORTED_BY_PROVENANCE`，不恢复旧停止路线。
- **先例核查：** 定向读取 Deep OC-SORT、DeconfuseTrack、TrackTrack、CAMELTrack 原始来源及 UCLouvain 2026 硕士论文公开摘要。CAMELTrack 已覆盖多线索与目标间关系学习，不能把泛化的上下文网络称为创新；硕士论文 PDF 尚 embargo，仅引用摘要机制线索。CVF HTML 两次 403 后使用可读的 arXiv/会议 PDF。具体链接和检索边界见建议文档；未做穷尽性新颖性审查或官方代码适配验收。
- **验证证据：** 五个 JSON 模板解析通过且均为 TEMPLATE，三张 CSV 均只有合法且不重复的表头，两张指标表列一致；当前改动/新增 Markdown 的 97 个本地链接全部可定位；按原 24 fit 清单核得 18/6 的拟议内部划分无交集。`git diff --check` 通过。仅文档与格式验证，没有访问服务器原始数据、使用 GPU、训练、重评或声称独立复现。
- **产物与限制：** 更新 README、报告索引、研究治理入口、创新台账及旧实验简记模板导航；保留此前未提交的 S01–S04 复盘修改。服务器回传历史补齐仍未执行，报告模板不含真实指标；现有开发集复用和 ReID 训练重叠未知的限制不变。
- **精确下一步：** 用户审阅并同步本批文件后，服务器 agent 从 `docs/a2a/LATEST.md` 进入，先补历史机器可读结果并将 S05 转成锁定配置/命令，完成 S05 后按规范回传；再根据覆盖及八段结果决定是否实现 S06。没有提交、推送或发送远程消息。

## 2026-11-23：完成 S05 稳定性/合法动作覆盖，冻结 S06 seed 13 协议

- **历史补齐与协议：** 从权威原始文件导出 S01 G/Z/L/M 的 4 行完整精度 aggregate、96 行逐视频指标，以及 S04 的 30 个事件结果；未从 Markdown 舍入值重建。S05 在新跟踪前冻结 `manifests/protocols/s05_visual_decision_coverage_20261123.yaml`（SHA256 `b35aef525d6c6b43564d036ea090dd9edf6f6c868c8f2b45c29a6cd18d20cbc3`）。
- **S05-A：** 在复用且非独立的八段 train development 上完成 G/L，并经哈希条件复用 M。官方 combined G/L/M HOTA 为 `68.90273986139066 / 68.81504013879633 / 69.51869387798489`，AssA 为 `52.82377438598485 / 52.77110310801603 / 53.56446597941373`，IDSW 为 `270 / 250 / 235`。M 相对 G pooled HOTA/AssA `+0.615954/+0.740692`，但只赢 5/8 视频，最差视频 HOTA `-7.1388`，不支持一致增益。首轮因缺 PYTHONPATH 在跟踪前失败，第二 immutable attempt 仅修环境并完成。
- **S05-B：** 遍历 fit24 共 25,328 帧；GT-free 同状态 M/G/V 生成器找到 239 个差异帧、242 个 union 分量，126 个可完整代理标注。41 个合法去重 G/V 动作满足 `g-h>0`，覆盖 40 帧、17 视频，刚过预写 40/4 资源门；46 个分量存在任一 `g>0`，其中 5 个同时损伤 M 正确配对。69 是涉及的 video-track ID，不是恢复身份数；实际有 43 个唯一 video-track ID 新增代理正确配对。`dancetrack0027` 完整确定性重放的候选 trace 和 tracker 输出哈希均一致。
- **审计与主张：** 新上下文同家族 GPT-6.1-Sol ultra 暂定 `WARN`，独立读取 152 JSON、335,407 tracker 行、25,328 trace 和 242 分量，并重跑官方 TrackEval，27 行逐项一致；未见假 GT、归一化造高、幽灵结果、重复膨胀或 GPU0 使用。限制是复用开发数据、离线 owner 代理、外观预训练重叠未知及范围。`result-to-claim` 只支持资源可行性，路由为 `implement_and_run_seed13_only`。
- **报告和硬件：** 权威小型报告在 `reports/diagnostics/s05_visual_decision_coverage_20261123/`；大产物在 `/media/imc/data/mot-data/runs/s05_visual_decision_coverage_20261123/`。全 S05 使用 `CUDA_VISIBLE_DEVICES=''`，没有使用任何 GPU，物理 GPU0 明确未用。保留一个无科学输出的 `source_snapshot` glob launcher 错误。没有 commit 或 push。
- **S06 冻结：** 在任何 S06 专用提取、训练或结果前冻结 `manifests/protocols/s06_component_selector_seed13_20261123.yaml`（SHA256 `290a95e3e36e96dbc62cd20e7661a6b1c80b1627a2849586cb37c412f2afdf51`）。只允许 seed 13；37/73 阻断。seed 13 必须以 Context+risk 相对 M 同时达到 HOTA/AssA 各 `+0.5`、IDSW 不增加、至少 5/8 视频 HOTA 为正，并在 pooled HOTA/AssA 同时胜 V 与 Local+risk，才允许后续 seeds。
- **精确下一步：** 实现 S06 因果特征仪表并先验证 instrumented M 输出哈希等价；训练 Local/Context seed 13，按内部 dev6 官方 AssA→HOTA→早 epoch 选 checkpoint，再一次性跑 evaluation8 全七条件闭环。开始任何 CUDA 工作前重新查卡，只用物理 GPU1–7 中空闲单卡并显式记录映射；物理 GPU0 始终禁用。

## 2026-11-23：S06 seed 13 在训练前工程门停止

- **冻结与实现：** 在 S06 专用提取前冻结 `manifests/protocols/s06_component_selector_seed13_20261123.yaml`（SHA256 `290a95e3e36e96dbc62cd20e7661a6b1c80b1627a2849586cb37c412f2afdf51`），固定 train18/internal-dev6/evaluation8、Local/Context 表示、两个风险政策、seed 13 全数值门和一次工程修复上限。新增因果方向分、EMA cosine、竞争 margin 原语与测试，以及 instrumented M runner/immutable lock 生成器。
- **失败尝试：** attempt1 在 `dancetrack0027` 首个无 tracker 帧因空 velocity/observation 形状校验停止，未形成完整 trace 或 tracker 结果。修复后用新目录 attempt2 重试，但 upstream OC-SORT 的空 `k_observations` 实际为一维 `(0,)`，仍在同一首帧停止。源码只读核查定位到空 tracker list 的 `np.array([...])` 构造。协议只允许一次修复，故没有 attempt3。两个目录没有覆盖，但 preparer 把 attempt2 内部 run ID 也硬编码成 attempt1；该元数据缺陷原样保留并写入报告。
- **后置代码状态：** 工作树已补 `(0,)` observation 回归测试，现有相关测试 20 passed；该后置修正没有作为运行执行，不能算 S06 等价或科学证据。报告为 `reports/feasibility/s06_component_selector_seed13_20261123/`。
- **科学状态与硬件：** `FAILED_ENGINEERING / NO_SCIENTIFIC_DECISION`。没有样本导出、训练、checkpoint、闭环 tracker 或 TrackEval；seed 13/37/73 都是科学 `NOT_RUN`。两次均 `CUDA_VISIBLE_DEVICES=''`，物理 GPU0 及所有 GPU 完全未用。没有 commit/push。
- **精确下一步：** 若用户仍要继续 S06，先另立 versioned retry 协议，显式继承已知失败并把空 tracker 两种形状的单测作为前置门；随后才允许 attempt3-equivalent 的新 run ID。未获新授权前不再执行当前 S06。

## 2026-11-24：S06 retry3 完成 seed 13，冻结主门失败并停止

- **恢复与等价门：** 用户明确授权后冻结 retry2；修复空 tracker `(0,)` 表示和唯一 run ID。24/24 fit 视频、25,328 帧的 instrumented M tracker 哈希与权威 M 完全一致，候选投影逐帧一致，242 个差异分量和所有特征有限性检查通过。导出 train18 199 分量/104 可标注/208 动作/33 profitable，internal-dev6 43/22/44/8；未标注样本保留 null，不当负例。
- **训练与选择：** retry2 首个训练锁因相对协议路径在锁前停止；随后 GPU1 训练在首次 optimizer step 前因缺确定性 CuBLAS 环境停止。新冻结 retry3（SHA256 `5762feb586e341fb5cca6583baaea00f7ecbedfe87c7ef1ec96f2b580ceb0014`），使用 `CUDA_VISIBLE_DEVICES=1`、逻辑 `cuda:0→物理 GPU1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。Local/Context 各 25,794 参数、seed13、30 epochs、12 checkpoint 完成；internal-dev6 官方 AssA→HOTA→早 epoch 选择 Local epoch5、Context epoch30，evaluation8 未参与选择。
- **一次性 evaluation8：** 40/40 新闭环任务和七条件官方 TrackEval 完成。M HOTA/AssA/IDSW `69.518694/53.564466/235`；V `70.436750/55.035986/246`；Local-risk 与 M 相同；Context-risk `69.613263/53.707202/239`。Context-risk 相对 M 仅 `+0.094569/+0.142736`、IDSW `+4`、仅 1/8 视频 HOTA 为正，相对 V `−0.823486/−1.328785`；主门仅过 2/8，决定 `STOP_S06_NO_SEEDS_37_73`。
- **次要条件：** 预定义 Context-no-risk 为 `70.998795/55.768069/226`，相对 M `+1.480101/+2.203603/−9 IDSW`，相对 V `+0.562045/+0.732083/−20`；4 视频正、2 负、2 平。这是单 seed 复用开发描述，不得事后替换 Context-risk 主门或作为机制/泛化主张。
- **审计与偏离：** 新上下文 GPT-6.1-Sol ultra 同家族暂定 `WARN`，独立复现 693 个最终 TrackEval 单元格（最大差 0）和 378 个代理动作标签（零差）。发现协议写 128 components、实际按 128+80 action examples 累积且末批仍 `/128`；Local 另有 8,192 个上下文输入权重结构性不活跃，故有效容量未严格匹配。上述在看结果后发现，原样报告，不授权修补重跑。历史 GPU 分配没有独立快照；锁/代码/运行记录显示训练只用物理 GPU1、其余 CPU-only，物理 GPU0 未暴露。
- **主张门与报告：** result-to-claim 给 `claim_supported=no`、confidence high、精确路由 stop。权威报告为 `reports/feasibility/s06_component_selector_seed13_retry3_20261124/`，含 7 行 aggregate、56 行逐视频、60 行曲线、54 个案例/对照、checkpoint、gate、审计和 artifact hashes。大型根目录约 retry2 279MB、retry3 202MB，留在数据盘。未 commit/push。
- **精确下一步：** 停止 S06，不运行 seeds37/73，不以偏离为由事后修复同一门。只允许对现有 artifact 做 CPU-only postmortem。若未来研究 Context-no-risk，必须另立新问题/协议、修正 component batching、加入有效容量对照并规划未触碰评估；不自动授权新的性能运行。

## 2026-11-24：研究端复核 S05/S06 与次要正向信号

- **完成内容：** 审阅 `54d03cc` 中 S05/S06 协议、完整精度汇总/逐视频表、训练曲线、选模记录、案例、审计和实际模型/训练/回放代码；新增 `docs/research_status_review_20261124.md`、只读脚本 `scripts/review_s06_git_evidence.py` 与 `docs/audits/s05_s06_local_review_20261124/derived_summary.json`。同步 README、LATEST 和创新台账导航，保留原报告与停止决定。
- **结果判断：** Context-risk 原主门仍仅 2/8 通过，S06 停止、不续跑 seeds37/73。计划内 Context 无额外风险相对 M 有 +1.480101 HOTA/+2.203603 AssA/−9 IDSW，相对固定 V 有 +0.562045/+0.732083/−20；但对 V 仅 2 胜/2 负/4 平，优势主要集中 0074。以上为已回传开发数据的复算，不是本轮新跑指标或独立泛化结论。
- **新解释限制：** train18 只有 104 个可标注分量/208 个动作，执行每 epoch 两步；风险策略选模打平导致 Local epoch5（约10步）、Context epoch30（约60步）。Local 在导出案例中对理论上 g=h=0 的 M 动作仍预测约 0.4，不能将低动作率等同于可靠风险识别。Local 已含竞争 margin；加上实际容量不匹配，尚不能将差异归因于上下文。案例缺配对/合法边/逐身份后续证据，artifact ID 仍有未定义项，列为补充交付缺口。
- **实际验证：** CPU-only 标准库脚本检查 7 个总体/56 个逐视频条目、与 gate JSON 的 441 个共同指标单元格精确一致、S05/S06 清单中 12 个 Git 文件 SHA256、8 条门槛及案例/样本计数；临时目录重复执行得到完全一致 JSON，已有输出拒绝覆盖。72 个本地文档链接与 git diff --check 通过。服务器报告的 693 次官方指标复算/378 个标签复现仅引用，未在本机重新执行。没有 GPU 工作、模型训练、服务器访问或原始 TrackEval 重评。
- **精确下一步：** 建议服务器在一个工作日/8 CPU 核小时内，对现有全部 checkpoint 在固定 train18/internal-dev6 样本导出标签/预测/排序、M 零点误差和常数参照，并补 0074/0032/0049/0023 变化决策的配对和后续身份证据、可解析来源引用；不重训、不扫风险阈值、不根据 evaluation8 重选 checkpoint。体检后再决定是否另立无额外风险/相对效用的新问题；当前不升级 V1/V2。未提交、未推送。

## 2026-11-24：S07 完成既有 checkpoint CPU 学习质量体检

- **冻结与范围：** 在运行前新增 `manifests/protocols/s07_s06_checkpoint_postmortem_20261124.yaml`（SHA256 `c6039c138507081783b233010928f917422d4748a5bf9eb4bacd481a630cacef`），固定只读取 S06 Local/Context epoch5–30 共 12 个 checkpoint，在既有 train18/internal-dev6 可标注样本上推理；禁止梯度更新、调参、重选 checkpoint、evaluation8 选模、新跟踪和 GPU 工作。权威 run ID 为 `s07_s06_checkpoint_postmortem_cpu_attempt1`。
- **完成结果：** 导出 3,024 行全精度预测。train18 为 104 分量/208 动作；internal-dev6 为 22/44，实际 5 个视频有标签。Context epoch30 internal-dev6 视频等权 mean MSE `0.06909482306121303`，低于 Local epoch30 `0.08368197978582628` 和冻结 train-mean 常数 `0.09706207694768555`；但无额外风险决定为 `13/22`，低于 always-M `14/22`，risk 为 `14/22` 且全部选 M。train18 无额外风险为 `80/104`，显示训练到开发的决策差距。
- **结果后补检：** 协议全局声明视频等权，但 decision accuracy 实现按分量 pooled；保留原结果并新增两种分母补充，不覆盖。结果后探索性 M=0+共享 non-M train 均值在 internal-dev6 MSE `0.06755007982129707`，M/G/V train 均值为 `0.05587726359924876`，均低于 Context epoch30；这些不是冻结对照，只说明原 action-agnostic 常数不足证明 component-specific learning。
- **案例谱系：** 首次导出误把 trace 的 G proposal 配对标成 baseline M，独立审计发现后保留为 `case_evidence_attempt1_invalid*`。权威 attempt2 覆盖指定 4 段全部 15 个改动，明确为同状态 G-vs-applied；独立重建 868 项一致。但旧 trace 不含同状态 M 候选与精确合法 mask，4 个选 G 案例相对 G 差异为空，14 条配对 owner 未解析，不能建立单动作因果或官方 IDSW 证据。
- **验证与审阅：** 临时目录重复推理的 predictions/CSV 逐字节一致；独立复算 1,104 个指标/常数/差值最大误差约 `1.67e-16`，107 个哈希引用/70 个唯一文件无不匹配，不同 Torch 版本 CPU 重推理最大差约 `5.96e-08`。result-to-claim 为 `partial`、confidence high；完整性审计同家族暂定 `WARN`，未见假 GT、自归一化或幽灵结果。
- **资源与产物：** 全过程 `CUDA_VISIBLE_DEVICES=''`、8 CPU 线程，物理 GPU 列表为空、GPU0 未使用；主推理 1.5301 秒，CPU 核时上界约 0.0034。权威小报告为 `reports/diagnostics/s07_s06_checkpoint_postmortem_20261124/`，大产物在 `/media/imc/data/mot-data/runs/s07_s06_checkpoint_postmortem_20261124/`。实验完成时保持可审阅工作树；2026-11-25 用户随后明确授权提交并推送本次交接。
- **精确下一步：** 保持 `STOP_S06_NO_SEEDS_37_73`，不自动启动任何新性能实验。研究端先选择转向成熟关联基线失败研究，或另立新相对效用问题；新协议必须包含 M 解析零点、action-type 强常数、排序/regret/拒绝校准、公平 component batching/有效容量及未触碰最终评估，并重新获得执行授权。

## 2026-11-25：S07 研究端复盘与路线收束

- **完成内容：** 审阅 `92b0802` 的 S07 协议、24 条 checkpoint/split 结果、强常数与分母补充、15 个权威案例、审计、执行代码；新增 `docs/research_status_review_20261125.md` 与 `docs/audits/s07_local_review_20261125/{reproduce.py,verification.json}`。更新 README、报告索引、LATEST 和研究治理/创新台账，不改写权威实验结果。
- **判断变化：** Context epoch30 的 internal-dev6 MSE 为 0.069095，低于同 epoch Local，但高于结果后动作类型常数 0.055877；最终无额外风险决定 13/22，always-M 为 14/22，二分类分解 TP2/FP3/FN6/TN11。视频等权后仍无推进优势。保留 S06 已有单 seed 闭环正向数值，但降低当前稀疏双头选择器的研究优先级；不将小样本一例差异或 MSE 对照解释为普遍否定上下文学习。停止同批补充诊断，不续 seeds37/73。
- **实际验证：** 本地只读标准库脚本核对 13 个 Git 报告文件 SHA256/大小、5 个执行源码哈希及协议哈希，CSV/JSON 共 456 个数值单元格一致；复算 48 条分母记录的视频均值，从 Git 已有 S05 标签重建动作类型均值和四个强常数 MSE（误差<1e-12）。15 个案例均缺同状态 M 与精确合法 mask；原 CSV 的 runtime 哈希可由 Git 文件重建 CRLF 精确复现，属于换行差异。84 个本地 Markdown 链接、JSON 格式和 git diff --check 通过。
- **范围与限制：** 没有服务器访问、模型推理、训练、CUDA 或新 TrackEval；S07 的 3,024 行推理重复/1,104 项独立复算仅引用已有审计。本次只基于 Git 结果做算术/来源/逻辑核查。重新读取 CAMELTrack 作者论文作为平台方向依据，没有新核验仓库训练可用性或承诺运行耗时。V1/V2 与 C1–C5 不升级。
- **精确下一步：** 建议改为准备成熟学习关联基线（优先 CAMELTrack）的有界执行任务书，固定版本/数据/权重来源、历史划分使用、官方最小训练与推理验收、资源预算和报告要求；先建立可信闭环，再据自然失败选一个具体方法问题。当前不启动新训练、调阈值或重复 S07，不把基线复现当创新。未提交、未推送。

## 2026-11-25：规范研究端任务下发并准备 S08 基线闭环

- **完成内容：** 新增 `docs/experiment_task_contract.md`、`docs/templates/server_experiment_task.md`，在 AGENTS.md 加入研究端指导职责；规范问题/对照/数据历史/选模/更新步数/预算、自动继续与停止、实现责任、回传文件和本地审阅。更新报告规范、README、研究治理入口、旧服务器手册与报告索引；LATEST 改为单一当前任务入口，历史停止范围明确限于原分支。
- **具体任务：** 新增 `docs/a2a/S08_CAMEL_BASELINE_TASK.md` 与 `manifests/protocols/s08_camel_baseline_20261125.yaml`。A来源环境锁→B预训练在线验收→C训练/恢复smoke→D train32随机初始化seed13单阶段20epoch→E两条件dev8完整评测→F最多16例自然失败摘要。前置工程门通过且预算内自动继续，最后停止回传；总≤24GPU卡时、单张物理GPU1–7、GPU0禁用。沿用已使用的motion train32/dev8，官方val/test不在本轮，P预训练与T随机初始化只作描述性参考，不称受控方法优势。
- **来源核查：** 通过GitHub连接器读取CAMELTrack官方 `bc1c261075b0beacfe71c2896453be83643153c8` 的README、pyproject、主/训练/模块配置与模型/训练数据代码。任务明确论文两阶段train+val与本轮单阶段train32不同；处理默认global权重、0.5/0.1阈值覆盖、缓存划分警告、当前目录last.ckpt自动恢复和重载返回值未赋回的风险。依赖版本、资产内容/训练重叠和真实运行兼容性留在A阶段锁定，不编造已安装/已复现。
- **实际验证：** 使用本地既有PyYAML解析新协议，逐项核对train32/dev8列表与原split完全一致、32/8互斥且合计40、A–F六阶段、16条逐视频/2条总体预期覆盖、24GPU卡时分项、GPU0排除、20epoch固定schedule和随机初始化字段；本轮文档链接检查与git diff --check通过。检查只覆盖设计结构和导航，不证明服务器命令可执行或科学方法有效。
- **范围与限制：** 只有研究侧文档/协议修改和官方源码阅读，没有服务器操作、安装/下载训练资产、训练、CUDA或新TrackEval。实际launcher、状态过滤、指标导出由服务器实现；协议状态 `DESIGN_READY_RUNTIME_LOCK_PENDING` / `NOT_RUN`，冻结时间为空，运行前须生成实际执行快照。原有S07本地复盘工作树保留；未commit/push。
- **精确下一步：** 用户将S08任务交给服务器后，从A阶段落实依赖/资产/划分/命令锁并顺序执行有界流程，返回 `reports/baselines/s08_camel_baseline_20261125/` 标准证据包。研究端据训练可用性、完整指标及失败证据决定一个后续方法问题；不自动续S06、换MOTIP、做两阶段扩展或新增模块。

## 2026-11-25：S08 CAMELTrack 有界基线闭环完成（A–F）

- **阶段 A（环境/来源/划分锁）：** 在 `/media/imc/data/mot-data/third_party/CAMELTrack` 固定官方 commit `bc1c261075b0beacfe71c2896453be83643153c8`，通过 `uv` 在 `/media/imc/data/mot-data/envs/s08-camel` 安装锁定依赖（Python 3.12.13、`torch 2.6.0`、`tracklab 1.3.17`、`sn-trackeval 0.4.0`）。从 HuggingFace revision `6759b7c27b99d8f2b8d8a13be69c6dbbb9cc40ff` 下载并校验 `camel_bbox_app_kps_dancetrack.ckpt`（SHA256 `cecce022...`）和 `states/dancetrack-train.pklz`（SHA256 `3c6f7ba8...`）。通过 `scripts/prepare_s08_camel_assets.py` 核验 40 段划分互斥且帧/检测维度完整（bbox 4 / app 6x128 / kps 17x3），生成 split 隔离状态文件与 GT-free 推理状态文件，导出 `source_audit.json`、`split_audit.json`、`resolved_configs/`、`runtime_lock.json` 和 `commands.sh`。
- **阶段 B（在线边界与确定性验收）：** 修复首轮 `b_p_official_online_check_v1` 中上游 `Tracklet.count` 跨次推理不重置问题并关闭多余渲染（`visualization=none`），保留 `v1` 失败记录。权威运行 `b_p_official_online_check_v2` 在物理 GPU 1 上对 `dancetrack0027` 完成 1–400 正常、1–200 未来截断、1–400 去 GT 及重复推理 4 次运行：重跑 SHA256 一致、去 GT 后 SHA256 一致、未来截断前缀输出逐行完全一致（400 帧 3,228 行，200 帧 1,572 行）。
- **阶段 C（随机初始化训练/恢复 smoke）：** 修复 `v1–v4` 工程探针与 RNG 状态回调（Lightning 默认不保存 RNG 且 `-inf` 掩码相减为 `NaN`），保留全部失败 attempt。权威运行 `c_t_scratch_smoke_v5` 完成 100 步训练：梯度全部有限，GAFFE（51/51）与三个 encoder（55/56，仅未用 `occluded_token` 未变）有效更新，固定无增强探针 loss 从 `2.8315` 降至 `0.8089`；step 50 新进程恢复后 `global_step`、`lr`、scheduler、优化器状态、PyTorch/CUDA/NumPy/Python RNG 及探针输出均精确一致；step 100 checkpoint 新进程重载输出逐元素一致（最大有限差异 `0.0`）。
- **阶段 D（20 epochs 单阶段从头训练）：** 独立目录 `d_t_scratch_seed13_e20_v1` 与隔离数据缓存（seed 13，FP32 batch 8，43,475,200 可训练参数）完成 20 epochs / 8,000 步训练，耗时 `14,962.28 s`（4.156 GPU 卡时），峰值显存 `2,801.79 MiB`，全过程 0 非有限梯度。`train/loss_epoch` 末轮为 `0.0357`，`val/loss` 从 epoch 1 的 `1.0132` 降至 epoch 20 的 `0.2021`（`val/sim_acc = 0.9532`、`val/sim_auroc = 0.9922`）。保存 epoch 2 工程检查点（SHA256 `d50f53a7...`）与唯一正式 epoch 20 `last.ckpt`（SHA256 `fb86a65e...`）。
- **阶段 E（`motion_dev8` 闭环追踪与官方 TrackEval）：** 修正 `tracklab.utils.wandb.init(cfg)` 全局变量初始化（保留 `e_p_official_dev8_v1` 失败记录），在物理 GPU 1 上分别以新进程显式重载并核验 SHA256 后完成 `e_p_official_dev8_v2`（`P_OFFICIAL`）与 `e_t_scratch_dev8_v1`（`T_SCRATCH`）的 GT-free 闭环追踪与 TrackEval。`COMBINED_SEQ`（7,565 帧）：`T_SCRATCH` HOTA `53.980593`、DetA `81.017950`、AssA `36.096358`、IDF1 `56.694987`、MOTA `90.621339`、IDSW `1451`、FP `1721`、FN `4434`、Frag `1428`；`P_OFFICIAL` 平台参考 HOTA `66.992740`、DetA `82.523905`、AssA `54.515523`、IDF1 `71.762874`、MOTA `91.590525`、IDSW `638`、FP `1646`、FN `4536`、Frag `1397`。
- **阶段 F（有界自然失败摘要与报告交付）：** 通过官方 TrackEval `MotChallenge2DBox` + `CLEAR` 匹配回放精准定位 `T_SCRATCH` 的全部 `1,451` 个官方 IDSW 事件（逐视频数与官方 CSV 完全一致），其中 76.71%（1,113/1,451）存在 IoU≥0.3 竞争 GT 框、40.73%（591/1,451）发生于匹配中断 >1 帧之后、两者重叠 33.91%（492/1,451）。按固定 SHA256 每视频最多抽样 2 例共 16 例导出至 `failure_inventory.csv`、`cases.jsonl`、`evidence/` 与 `diagnostics.json`。全流程权威运行总 GPU 耗时 `4.465` 卡时（GPU 0 禁用且未使用），标准报告包位于 `reports/baselines/s08_camel_baseline_20261125/`。
- **精确下一步：** 停止在阶段 F，交由研究端基于 `T_SCRATCH` 在 `dev8` 的闭环指标与 16 例官方 IDSW 证据确定下一轮唯一聚焦的方法研究问题；未执行 `git commit` / `git push`。

## 2026-11-26：拉取 S08 报告、文献窄查重与制定 S09 标定归因任务

- **完成内容：** 拉取远端 `d662023`，审阅 S08 A–F 全部结果、配置与 16 例官方 IDSW 证据；针对离线 `val/sim_auroc = 0.9922` 与闭环 `AssA = 36.096`（IDSW 1,451，76.71% 伴竞争 GT、59.27% 为 `gap==1` 连续帧交换）的反常落差开展 2025–2026 文献窄查重与候选细化，形成 [2026-11-26 复盘与路线审查](research_status_review_20261126.md)，并起草符合规范的 [S09 任务书](a2a/S09_CAMEL_CALIBRATION_AND_ATTRIBUTION_TASK.md) 与 [S09 版本化协议](../manifests/protocols/s09_camel_calibration_and_attribution_20261126.yaml)。同步更新 [LATEST](a2a/LATEST.md)、[创新台账](research_governance/innovation_registry.md) 与 [README](../README.md)。
- **关键核查：** 确认 S08 Stage E 强制 `sim_threshold=0.1` 而 `T_SCRATCH` 验证推荐阈值为 `0.519729`，且 S08 仅执行单阶段训练（缺失 Stage 1 单线索 TE 预训练）；窄查重确认 Sinkhorn+Dustbin+Margin 已有 SLAck / HDST-GNN / MM-Tracker 先例，曝光偏差与可靠记忆已有 MOTIP2 / MOTRv3 / HMP / LiAM-SAM / TCEI 先例，而在多线索 Transformer（CAMELTrack GAFFE）中注入显式成对运动残差与方向遮挡偏置（受 DOA-SORT / HyperSSM 启发）尚存在机制交叉空间。
- **范围与限制：** 本轮仅做本地文件审阅、联网文献查重与下一阶段任务设计；未连接服务器、未安装环境、未执行训练/推理/TrackEval 或使用任何 GPU。S09 状态为 `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`），各候选方向维持 `NOT_SUPPORTED_BY_PROVENANCE`。工作树保持未提交、未推送。
- **精确下一步：** 用户审阅 [2026-11-26 复盘](research_status_review_20261126.md) 与 [S09 任务书](a2a/S09_CAMEL_CALIBRATION_AND_ATTRIBUTION_TASK.md) 后，将 S09 交由服务器从 Stage A 起执行（阈值标定 + `train32` 两阶段基线 + 相似度边际/干净历史归因），凭实证结果在 A'/B'/C' 中单选 S10 创新方向。

## 2026-11-26：S09 完成阈值标定、Two-Stage 基线与闭环失败归因（A–E 全阶段）

- **完成与环境锁（Stage A）：** 核验 S08 6 项复用资产 SHA256 与上游 `CAMELTrack@bc1c2610` 源码一致；确认无独立 Stage 1 YAML（按 README 固定 `camel.yaml` 默认超参 + `max_epochs=20` + `gaffe._target_=cameltrack.architecture.Identity` + 单 encoder，`train_add_val=false`）。在 `scripts/run_s09_camel.py` 中对 PyTorch 2.6 `weights_only=False` 加载与 `CAMELDataset._load_pickle` 实例级缓存做零语义变更适配（`tests/test_s09_camel.py` 3/3 单元测试通过，验证逐元素一致、Stage 2 部分加载与 D2 1-to-1 拷贝路由）。全流程仅使用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`，`NVIDIA GeForce RTX 4090 D`），物理 GPU 0 禁用且未使用；总计 `8.486 GPUh`（budget `<= 16h`），新增数据盘占用 `6.30 GiB`（budget `<= 40 GiB`，剩余 `1.3 TiB`）。
- **阈值标定（Stage B，`dev8` GT-free 7,565 帧）：**
  - `T_SCRATCH_TH046` (`0.46`)：HOTA `53.348458`、DetA `81.014127`、AssA `35.260194`、IDF1 `55.168037`、MOTA `90.691624`、IDSW `1337`。
  - `T_SCRATCH_TH050` (`0.50`)：HOTA `55.215470`、DetA `80.690036`、AssA `37.899583`、IDF1 `56.203730`、MOTA `90.157708`、IDSW `1333`。
  - `T_SCRATCH_TH_COMP` (`0.519729`)：HOTA `54.443861`、DetA `80.653048`、AssA `36.875286`、IDF1 `53.868511`、MOTA `89.636124`、IDSW `1412`。
  - `P_OFFICIAL_TH050` (`0.50`)：HOTA `63.873840`、DetA `82.656548`、AssA `49.398883`、IDF1 `64.332257`、MOTA `90.770540`、IDSW `606`。阈值标定仅将 `T_SCRATCH` AssA 提升至多 `+1.80`（vs S08 `@0.1` `36.096`），不是核心瓶颈。
- **Two-Stage `train32` 训练与评测（Stage C）：**
  - C1 三个单 cue TE 各随机初始化训练 20 epochs（隔离缓存、`0` 非有限梯度、新进程重载 diff `0.0`）：`app_encoder` (`val/loss 1.1271`, `sim_auroc 0.9615`)、`kp_encoder` (`0.5223`, `0.9771`)、`bbox_encoder` (`0.4647`, `0.9863`)。
  - C2 联合 GAFFE + C1 权重训练 20 epochs（`last.ckpt` SHA256 `20be35af...`，GAFFE `51/51` 及 TE `55/56` 张量有效更新，`0` 非有限梯度、`val/loss 0.2391`、`val/sim_auroc 0.99225`、`epoch20_val_computed_sim_threshold = 0.545996`）。Stage C 总耗时 `7.477 GPUh`（budget `<= 12h`）。
  - C3 `dev8` 闭环：`T_TWOSTAGE_TH050` (`0.50`) 取得 HOTA `54.442514`、DetA `80.968530`、AssA `36.704320`、IDF1 `53.629050`、MOTA `90.373494`、IDSW **`786`**（较 `T_SCRATCH_TH050` 减半，-41%，但全局 AssA 仍仅 `36.704`）；`T_TWOSTAGE_TH_COMP` (`0.5460`) 取得 HOTA `51.712428`、AssA `33.318555`、IDSW `889`。
- **失败机理归因与隔离探针（Stage D）：**
  - **D1 相似度边际回放（100% 覆盖且 MOT SHA256 逐字节一致）：** `T_TWOSTAGE_TH050` 全部 `786` 例官方 IDSW 中，错配反超 (`margin < -0.05`) 占 **52.29%** (`411/786`)、高分平局 (`|margin| < 0.05`) 占 **25.95%** (`204/786`)、未匹配新建轨迹占 **17.43%** (`137/786`)、正向高分但匈牙利 1-to-1 竞争落败占 **4.33%** (`34/786`)；`60.05%` 发生于 `gap > 1`，`76.08%` 存在 `IoU >= 0.3` 竞争 GT，中位 `margin = -0.0755`。`T_SCRATCH_TH_COMP` (`1412/1412`) 分别为反超 `57.72%` (`815`)、平局 `29.18%` (`412`)、未匹配 `11.76%` (`166`)、匈牙利竞争 `1.35%` (`19`)。
  - **D2 特权 1-to-1 GT 路由干净历史探针（隔离非部署，保留 `v1`/`v2` 工程记录，权威 `v3`）：** `v1` 因未浅拷贝共享 `clean_det` 且广播到同 GT 兄弟轨迹导致 `.age` 多次累加与 `dancetrack0037` 轨迹克隆爆炸（`168,380` 更新/`576` IDs/`2,281` IDSW，保留并标记作废）；`v2` 因赋值 `age=0` `int` 触发 `TypeError`；权威 `v3`（`copy.copy` + `.age.clone().zero_()` + 严格 1-to-1 路由，`76,509` 更新 `<= 77,689` 预测匹配，拦截 `1,108` 异主与 `1,319` FP 入库污染）将 `T_TWOSTAGE_TH050` 闭环指标拉升至 HOTA **`81.907414`** (`+27.465`)、AssA **`80.063201`** (**`+43.359`**)、IDF1 **`92.152928`** (`+38.524`)、MOTA `91.673140`、IDSW `830`，其中 `dancetrack0037` AssA `21.796 -> 84.008` (`+62.212`)、`dancetrack0044` AssA `24.757 -> 78.349` (`+53.592`)。
- **报告导出与 S10 路线判断（Stage E）：** 通过 `scripts/export_s09_report.py` 导出权威报告包 `reports/diagnostics/s09_camel_calibration_and_attribution_20261126/`（`aggregate_metrics.csv` 7 行、`per_sequence_metrics.csv` 56 行、`learning_curves.csv` 476 行、`cases.jsonl` & `evidence/` 16 例、`checks.json` 13/13 全通过、`0.0` TrackEval 导出差异）。实证指向 **S10 首选 `B'`（抗闭环历史污染 / 自愈 gallery 更新）** 为主，同时诚实保留干净历史下仍触发 `1,108` 次单帧异主预测（`830` CLEAR 局部跳变）作为次级因素。
- **精确下一步：** 停止在 Stage E，交由研究端审阅 `reports/diagnostics/s09_camel_calibration_and_attribution_20261126/` 并锁定 S10 协议（优先评估 `B'`）；未执行 `git commit` / `git push`。

## 2026-11-27：S09 复盘审阅与 S10 抗污染实验任务设计

- **完成内容：** 拉取并审阅 `f8c6dc3` S09 完整报告包及 `scripts/run_s09_camel.py` 源码，形成 [2026-11-27 S09 复盘与设计说明](research_status_review_20261127.md)，并起草符合规范的 [S10 任务书](a2a/S10_CAMEL_ANTICONTAMINATION_TASK.md) 与 [S10 版本化协议](../manifests/protocols/s10_camel_anticontamination_20261127.yaml)。同步更新 [LATEST](a2a/LATEST.md)、[创新台账](research_governance/innovation_registry.md) 与 [README](../README.md)。
- **关键核查发现：** S09 Stage D2 `v3`（AssA `80.063`）除跳过 `1,108` 异主与 `1,319` FP 写入外，在 `CleanHistoryCAMELTrack.process` 第 1215–1224 行 `else:` 分支还对漏配/错配帧强制注入了 GT 正检测（至少 `1,247+` 次）。故 S10 Stage B 先用 `D2_SKIP_ONLY` 剥离“纯拒写”上限，并在 Stage C/D 以 `C0` 零重训简单写入门控与 `C1` 等交换比例 i.i.d. 对照严格评估 `C2`（连续接管训练）及 `C3`（多线索一致性门控聚合 + 真实污染判别 AUROC）。
- **范围与限制：** 本轮仅做本地复盘与 S10 实验任务设计，未连接服务器、未执行训练/推理/TrackEval 或使用任何 GPU。S10 状态为 `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`），各新机制维持 `NOT_SUPPORTED_BY_PROVENANCE`。
- **精确下一步：** 提交并推送本次 S10 设计，由服务器按 [LATEST](a2a/LATEST.md) 领取 **S10 任务**从 Stage A 开始执行并返回 `reports/feasibility/s10_camel_anticontamination_20261127/` 报告包。

## 2026-11-27：S10 完成 CAMELTrack 纯拒写分解与抗污染训练闭环（A–E 全阶段）

- **阶段 A（环境锁、等噪标定与单元测试）：** 新增 `scripts/run_s10_camel.py`、`tests/test_s10_camel.py` 与 `scripts/export_s10_report.py`。核验全部 8 项 S08/S09 复用资产 SHA256 与隔离拷贝共享数据缓存；以向量化 `ContiguousTakeoverTransform`（`p_takeover=0.55`, `L∈[2,12]`）和 `MatchedIidSwapTransform` 在 200 batches 标定得 `p_iid = 0.076880`（`C2` 替换率 `0.071268` vs `C1` `0.071080`，相对偏差 `0.26% <= 5%`）。`C3` 因果 `ConsistencyGateHead` 参数量 `382,785`（占全模型 `0.83% < 2%`）。3 项单元测试全通过（D2 零补写不变量、`corrupted_mask` 连续范围、全 1 权重等价与零因果泄漏/有限梯度）。全阶段仅使用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`），物理 GPU 0 禁用且未使用；总 GPU 耗时 **`4.072 GPUh`**（预算 `<= 12.0h`），新增数据盘存储 **`4.27 GiB`**（预算 `<= 25.0 GiB`）。
- **阶段 B（纯拒写上限分解与 `C0` 零重训基线，`motion_dev8`）：**
  - **`D2_SKIP_ONLY`（`b_d2_skip_only_v2`，隔离探针，`gt_injected_on_miss = 0`）：** 阻止 `2,170` 次跨 ID 与 `1,136` 次 FP 入库后获得 HOTA **`63.781`**、DetA `81.568`、AssA **`49.940`**、IDF1 **`63.186`**、MOTA `88.958`、IDSW `936`。相比 S09 `T_TWOSTAGE_TH050`（`54.443 / 36.704`）纯停写即可恢复 **`+9.338` HOTA（`+13.236` AssA）**；并与含 miss GT 注入的 S09 `D2 v3`（`81.907`）清晰分离出 `+18.126` HOTA 的漏配重构部分。
  - **`C0_SIMPLE_WRITE_GATE`（零重训标量相似度门控）：** `tau=0.60` 掉至 `48.838` HOTA / `29.847` AssA / `1,225` IDSW（跳过 `3,208` 次更新）；`tau=0.65` 掉至 `43.509` HOTA / `23.925` AssA / `1,666` IDSW（跳过 `7,475` 次更新），表明静态标量门控误伤正常高速运动目标的模板刷新。
- **阶段 C（严格受控 `train32` 训练 + `dev8` `@0.50` 闭环评测）：** 3 组训练均完成 20 epochs（各约 `0.98 GPUh`，`0` 非有限梯度，新进程重载 diff `0.0`）：
  - `C1_MATCHED_IID_SWAP`（`c1_matched_iid_seed13_e20_v2` → `c_eval_c1_th050_v1`）：HOTA `51.780`、AssA `33.256`、IDF1 `49.964`、MOTA `89.796`、IDSW `888`。
  - `C2_CONTIGUOUS_TAKEOVER`（`c2_contiguous_takeover_seed13_e20_v1` → `c_eval_c2_th050_v1`）：HOTA `48.210`、AssA `28.936`、IDF1 `45.176`、MOTA `89.025`、IDSW `1228`。
  - `C3_CONSISTENCY_GATED_TE`（`c3_consistency_gated_seed13_e20_v1` → `c_eval_c3_th050_v1`）：HOTA `49.781`、AssA `30.765`、IDF1 `46.604`、MOTA `89.507`、IDSW `1110`。虽在同分布接管训练下相比 `C2` 回升 `+1.570` HOTA / `+1.829` AssA，但所有合成污染训练均劣于无污染 S09 `T_TWOSTAGE_TH050`（`54.443`）。
- **阶段 D（软拒写滤网与真实闭环污染检出 AUROC）：**
  - `C3_WITH_SOFT_WRITE_FILTER`（`d_c3_soft_write_filter_w050_v2`，`w_last < 0.50`）：跳过 `750/74,826` 次写入，得 HOTA `49.313`、AssA `30.201`、IDF1 `45.506`、IDSW `1,106`（无增益）。
  - `d_real_pollution_auroc_v1`：重放 `T_TWOSTAGE_TH050` 在 `dev8` 的 `76,658` 次写入（`42,847` clean、`31,940` cross-ID、`1,350` FP、`521` unanchored）：虽然 `C3` 在合成接管验证集上 `val/cons_acc = 0.9896`，但真实跨 ID 写入全局判别 `AUROC = 0.5002`（`F1 = 0.0259`，成熟轨 `hits>=3` 为 `0.5032`）。描述性拆解发现 `31,791/31,940` 次跨 ID 帧属于切主后的漂移自洽尾段（`AUROC = 0.4997`）；即便是 **`149` 例首次切换 onset** 帧，`C3` 仅得 `AUROC = 0.6220`（`>=0.5` 检出率 `4.70%`），明显弱于原始相似度 `1 - sim`（`AUROC = 0.8512`），暴露合成邻域借框接管与真实闭环切换间的显著 sim-to-real 鸿沟。
- **失败保留与报告交付（Stage E）：** 保留 3 次工程失败前驱目录（`b_d2_skip_only_v1`、`c1_matched_iid_seed13_e20_v1`、`d_c3_soft_write_filter_w050_v1`），不影响权威结果。已导出完整报告包 `reports/feasibility/s10_camel_anticontamination_20261127/`（`aggregate_metrics.csv` 7 行、`per_sequence_metrics.csv` 56 行、`learning_curves.csv` 474 行、`cases.jsonl` & `evidence/` 16 例、`checks.json` 14/14 全通过、TrackEval 导出差 `0.0`、`artifacts.json` 54 项）。
- **精确下一步：** 停止在 Stage E，不扩展 multi-seed 37/73 或继续当前 `C2/C3` 合成交换路线；交由研究端审阅 S10 完整报告包并决定后续真实错配采样（来自自然轨迹而非合成同帧替框）或关联机制方向。未执行 `git commit` / `git push`。

## 2026-11-27：S10 审阅与起草 S11 零重训 Onset 解剖及冲突感知拒写任务

- **完成内容：** 拉取远端 `238b517`，审阅 S10 完整报告包（`reports/feasibility/s10_camel_anticontamination_20261127/`）与训练曲线/案例日志；新增 [S11 任务书](a2a/S11_CAMEL_ONSET_AND_CONFLICT_GATE_TASK.md) 与 [S11 版本化协议](../manifests/protocols/s11_camel_onset_and_conflict_gate_20261127.yaml)，同步更新 [LATEST](a2a/LATEST.md)、[创新主张台账](research_governance/innovation_registry.md) 与 [README](../README.md)。
- **关键决策：** 维持 S10 对 `C2/C3` 合成接管与局部滑窗一致性的停止判定；针对 S10 Stage D 揭示的“`31,791` 例自洽漂移尾段由 `149` 例 onset 引爆且 `1-sim` 在 onset 达 `0.8512` AUROC、而无差别 `C0` 门控因误杀孤立快动目标致 `FN` 飙升”现象，设计纯推理零重训的 S11 任务（`<= 2.0 GPUh`、物理 GPU 1、GPU 0 禁用）：Stage B 补测 `T_TWOSTAGE` 在 `0.10/0.30/0.40` 低阈值表现，Stage C 对 `149` 例真实 onset 做 100% 病因解剖（`gap`/竞争 `IoU`/双向 `margin`/`2-cycle`）并离线测试“创生早期干净锚点 vs 近期滑窗” `anchor_reid_dist` 判别力，Stage D 固定 `@0.50` 验证 4 组冲突感知条件拒写与早锚保留闭环基线（`G1–G4`）。
- **范围与限制：** 本轮仅做本地审阅、文档与协议起草；未连接服务器、未执行推理或 TrackEval、未修改任何模型或运行训练。S11 状态为 `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`），零重训启发式门控仅作可部署基线与机制桥接，不宣称为架构创新。
- **精确下一步：** 提交并推送 S11 设计，交由服务器按 [LATEST](a2a/LATEST.md) 执行 Stage A–E 并返回 `reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/`，随后进入 S12（`C'` / `A'`）路线定案。

## 2026-11-27：S11 完成零重训低阈值标定、149 例 Onset 解剖与冲突感知拒写闭环（A–E 全阶段）

- **阶段 A（资产核验、实现与单元测试）：** 新增 `scripts/run_s11_camel.py`、`tests/test_s11_camel.py` 与 `scripts/export_s11_report.py`。校验全部 5 项 S09/S10 资产 SHA256，确认 `states_dev8_gt_free.pklz` 内 `det.embeddings` 为 `(6, 128)` 非归一化分区特征与 `visibility_scores` `(6,)` 并锁定分区加权余弦距离。3 项单元测试全通过（竞争 `IoU` 排除自身、双向 `margin` 及 `+inf` 边界、`2-cycle` 检测、`G1–G3` 跳写保库与 `G4` 80 步 `len<=50` 前 5 槽早期锚点固定）。全流程 0 训练，仅用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`，GPU 0 禁用且未使用），总 GPU 耗时 **`1.008 GPUh`**（预算 `<= 2.0h`），新增数据盘占用 **`0.086 GiB`**（预算 `<= 5.0 GiB`）。
- **阶段 B（`T_TWOSTAGE` 低阈值补充标定，`motion_dev8`）：**
  - **`T2_TH040` (`0.40`)**：HOTA **`57.572`**、DetA `81.394`、AssA **`40.845`**、IDF1 **`60.117`**、MOTA `91.184`、IDSW `832`、FP `1,676`、FN **`4,642`**（相比 `@0.50` 回收 `973` 漏检 `FN` 且 HOTA `+3.129`、AssA `+4.141`、IDF1 `+6.488`）。
  - `T2_TH030` (`0.30`)：`56.595` HOTA / `39.537` AssA / `876` IDSW / `4,659` FN；`T2_TH010` (`0.10`)：`56.594` HOTA / `39.537` AssA / `877` IDSW / `4,659` FN。
- **阶段 C（149 例 Onset 解剖与早锚漂移离线探针，`c_onset_anatomy_v2`，保留 `c_onset_anatomy_v1` 失败记录）：**
  - 回放 8/8 序列 MOT 输出与 S09 `T_TWOSTAGE_TH050` 逐字节一致（零 GT 泄漏），覆盖 `76,658` 次写入（`42,847` clean、`149` onset、`31,791` post_switch、`1,350` fp、`521` unanchored）。
  - `149` 例 onset 解剖：`74.5%`（`111`）为 `gap == 1` 连续切换、`89.9%`（`134`）存在 `IoU >= 0.30` 空间竞争（中位 `0.556`）、`74.5%`（`111`）`assignment_margin < 0.10`（中位 `0.0456`）、`22.8%`（`34`）为 `2-cycle` 互换；在场的 `55` 例真目标框中 `34` 例强反转（`< -0.05`）、`16` 例平局、`5` 例匈牙利竞争落败；`94` 例真目标框缺检。
  - 判别力比较：`onset_vs_clean` 上 **`-assignment_margin` 达 `0.9432` AUROC**（5% FPR 下召回 `73.8%`），`max_conflict_iou` `0.8666`，`1-sim` `0.8512`；`post_switch_vs_clean`（`31,791` 帧）上 **`anchor_reid_dist` 达 `0.7062` AUROC**（全量 cross-ID `0.7058`），打破近期滑窗自洽陷阱（`recent_reid_dist` `0.5251`、`1-sim` `0.5198`）。
- **阶段 D（`G1–G4` 零重训冲突拒写与早锚固定 `@0.50` 闭环，结论命中 情况 2）：**
  - `G1_CONFLICT_IOU030_TAU065`：`48.686` HOTA / `29.715` AssA / `45.128` IDF1 / `1,338` IDSW / `8,055` FN（跳过 `4,733` 次）；相对 S10 无条件 `C0_TAU065`（`43.509`）大幅挽回 `+5.177` HOTA（`-1,039` FN）。
  - `G2_MARGIN_M010_TAU070`：`48.139` HOTA / `29.249` AssA / `1,480` IDSW / `8,877` FN（跳过 `5,883` 次）。
  - `G3_HYBRID_GATE_NO_PIN`：`44.442` HOTA / `25.088` AssA / `1,734` IDSW / `9,588` FN（跳过 `7,703` 次）。
  - `G4_HYBRID_GATE_EARLY_PIN5`：`43.380` HOTA / `23.918` AssA / `1,698` IDSW / `9,759` FN（跳过 `7,857` 次，分配 `2,839` 个早锚槽）。
  - 全部 `G1–G4` 仍大幅劣于 `T_TWOSTAGE_TH050` (`54.443`) 与 `T2_TH040` (`57.572`)，因密集场景约 `42%` 正常成熟匹配伴随 `IoU>=0.30` 冲突导致 `>98%` 误跳过（`4,733–7,857` 拒写 vs 仅 `~82` 捕获 onset），在遮挡区饿死运动与外观更新。
- **阶段 E（报告交付与 S12 路由）：** 已导出 `reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/`（`7` 行总体、`56` 行逐视频、`16` 例 onset 证据、`checks.json` 10/10 全通过、`artifacts.json` 47 项全校验、`tests/test_s11_camel.py` `3 passed`）。**定案进入 情况 2**：写库端启发式刹车不足以止损，S12 直接从关联打分与分配层立项 **`C'`（GAFFE 成对运动/方向遮挡边偏置）+ `A'`（边际/`2-cycle`/延迟解歧）**，并以 `T2_TH040`（`57.572`）与 `anchor_reid_dist` 漂移信号作配套参照。未执行 `git commit` / `git push`。

## 2026-11-27：S11 复盘审阅与制定 S12（`C' + A'`）受控实验任务

- **完成内容：** 审阅 S11 完整报告包（`reports/feasibility/s11_camel_onset_and_conflict_gate_20261127/`：`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`diagnostics.json`、`cases.jsonl`）并结合 2025–2026 文献查重（DOA-SORT、HyperSSM、VLA-ReID、C-TWiX、SLAck、HDST-GNN、MM-Tracker、MOTIP2），起草 [S12 任务书](a2a/S12_CAMEL_PAIRWISE_EDGE_AND_ASSIGNMENT_TASK.md) 与 [S12 版本化协议](../manifests/protocols/s12_camel_pairwise_edge_and_assignment_20261127.yaml)，同步更新 [LATEST](a2a/LATEST.md) 与 [创新主张台账](research_governance/innovation_registry.md)。
- **关键科学设计：** 针对 S11 揭示的 `149` 例真实 onset 主因（`74.5%` `gap==1`、`89.9%` `IoU>=0.30`、`63.1%` 真目标缺检却误抓邻框、`74.5%` 双向 `margin < 0.10`、`-margin` AUROC `0.9432`、`22.8%` `2-cycle` 互换），设计零重训分配对照（Stage B `B1` Dustbin 增广 Dual-Softmax 与 `B2` `2-cycle` 纠偏+极端歧义单帧暂缓）、同 S09 C1 TE 初始化与等参数单节点容量对照（Stage C `M0_NODE_CAP_CTRL` vs `M1_C_KIN_EDGE` vs `M2_C_KIN_DOA_EDGE` vs `M3_C_PRIME_PLUS_A_LOSS`）、以及 Stage D 协同与 S11 同口径 onset/尾段消减解剖。
- **范围与限制：** 本轮仅做本地研究复盘、联网文献查重与 S12 任务/协议设计，未连接服务器、未执行训练/推理/TrackEval 或使用任何 GPU；S12 状态为 `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`），各待验机制保持 `NOT_SUPPORTED_BY_PROVENANCE`。工作树保持未提交、未推送。
- **精确下一步：** 用户将工作树提交/交付服务器后，服务器按 [LATEST](a2a/LATEST.md) 领取 **S12 任务**从 Stage A（核验 `gaffe.py`/`losses.py`/`dataset.py` 源码接口与单测）顺序执行至 Stage E 并返回 `reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/`，随后由研究端根据结果决定是否进入 S13 多 seed 与官方 `val`/`test` 确认。

## 2026-11-28：S12 完成 CAMELTrack 成对运动/方向遮挡边偏置 (`C'`) 与分配对齐 (`A'`) 受控实验（A–E 全阶段）

- **阶段 A（源码核验、共享接口实现与单测）：** 核验 `CAMELTrack@bc1c2610`（`GAFFE` 4 层 `nn.TransformerEncoder`，序列 `[dets, tracks, cls]`；`NTXentLoss` 温度 `0.07` 余弦相似度；`DropoutFeatures` 设 `drop_bbox=1` 而非 NaN；`norm_coords='positive'` 前缩放）及全部 11 项 S08–S11 复用资产 SHA256。200-batch `train32` 审计确认无匹配轨迹占比 `7.59%`（`3,794/49,986` >= `3%`）、`IoU>=0.30` 重叠对 `7.20%`、`drop_bbox` `9.59%`。新增 `scripts/run_s12_camel.py`、`tests/test_s12_camel.py`（`3 passed`，含因果恒速外推/dropout/NaN 防护、`M0_extra (266,240) >= M2_extra (151,105) > M1_extra (146,305) > 0`、step-0 零初始化等价及 `edge_sim_mlp`/`edge_attn_td`/`node_t_mlp` 非零有限梯度验证）与 `scripts/export_s12_report.py`。全流程严格单卡运行于物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`，物理 GPU 0 禁用且未触碰）；总计 **`8.871 GPUh`**（预算上限 `<= 9.0h`；B `0.313h`、C `7.936h` [20ep 联合训练四臂共 `6.810h` + 8 次评测 `1.126h`]、D `0.623h`）、新增数据盘占用 **`4.097 GiB`**（`<= 15.0 GiB`）。
- **阶段 B（零重训 `A'_infer` 对照，复用 S09 `T_TWOSTAGE` @ `0.40`）：**
  - `B1_A_DUAL_SOFTMAX_TH040`（`b1_dual_softmax_th040_v1`，dustbin=0.40 拒识 `10` 次）：HOTA `55.919`、DetA `80.999`、AssA `38.718`、IDF1 `58.154`、MOTA `90.967`、IDSW `931`、FN `4,713`（`58.73 ms/f`，相对 S11 `T2_TH040` `57.572` 下降 `-1.653` HOTA）。
  - `B2_A_MARGIN_2CYCLE_DEFER_TH040`（`b2_margin_2cycle_defer_th040_v1`，`2-cycle` 交换 `220` 次、暂缓 `1,244` 次）：HOTA `54.356`、DetA `81.950`、AssA `36.173`、IDF1 `52.843`、MOTA `90.163`、IDSW `1,218`、FN `5,239`（`85.62 ms/f`，下降 `-3.215` HOTA）。
- **阶段 C（受控四联组 20-epoch 训练 + `dev8` `@0.40`/`@0.50` 闭环评测，`seed=13`、`last.ckpt`）：** 4 组均完成 20 epochs（`8,000` 步）、`0` 次非有限梯度、新进程重载 diff `0.0`，全部目标/新增参数有效更新：
  - `M0_NODE_CAP_CTRL`（`266,240` 新增参数）：`@0.40` HOTA `54.179` / AssA `36.244` / IDF1 `55.354` / IDSW `879` / FN `4,733`（`@0.50` HOTA `53.577`）。
  - **`M1_C_KIN_EDGE`（`146,305` 新增参数，Stage C 按规则获胜组）**：`@0.40` HOTA **`57.033`** / AssA **`40.275`** / IDF1 **`60.417`** / MOTA `91.079` / IDSW `957` / FN `4,555` / `64.93 ms/f`（`@0.50` HOTA `57.033` / AssA `40.276` / IDSW `956`）。相比容量对照 `M0` 净增 **`+2.854` HOTA / `+4.031` AssA / `+5.063` IDF1**；高出 S09 `T_TWOSTAGE_TH050` (`54.443`) `+2.590` HOTA，且 `IDF1` (`60.417` vs `60.117`) 与 `FN` (`4,555` vs `4,642`) 略优于 `T2_TH040`，但总体仍低于 S11 `T2_TH040`（`57.572`）`-0.539` HOTA（`+125` IDSW，逐视频胜平负 `3/0/5`）。
  - `M2_C_KIN_DOA_EDGE`（`151,105` 新增参数）：`@0.40` HOTA `56.670` / AssA `39.762` / IDF1 `59.318` / MOTA `91.218` / IDSW `903` / FN `4,529`（`@0.50` 完全相同）；DOA 相对 `M1` 微降 IDSW（`903` vs `957`）与 FN（`4,529` vs `4,555`），总体 `-0.363` HOTA。
  - `M3_C_PRIME_PLUS_A_LOSS`（`151,105` 新增参数）：`@0.40` HOTA `52.025` / AssA `33.586` / IDF1 `54.434` / IDSW `954` / FN `4,744`（`@0.50` `52.545`），较 `M2` 显著退化 `-4.645` HOTA；训练期 8,000 步及全部 `train/*` 指标严格有限，仅 `val/unmatched_rej_loss` 在第 [2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 19, 20] epoch 记录 `nan`（无未匹配项时的日志聚合）。
- **阶段 D（`M1` 协同评测与 Onset/污染尾段解剖回放）：**
  - `D_M1_PLUS_B1_TH040`：`57.009` HOTA / `40.322` AssA / `59.952` IDF1 / `980` IDSW / `4,566` FN（`0` dustbin 拒识，`-0.024` vs `M1`）。
  - `D_M1_PLUS_B2_TH040`：`52.627` HOTA / `34.239` AssA / `54.849` IDF1 / `1,002` IDSW / `4,549` FN（`257` 次 `2-cycle` 互换与 `40` 次暂缓导致 `-4.406` HOTA 退化）。
  - Stage D 解剖回放（`M1 @0.50` 与 `@0.40` 均与在线评测 MOT 逐字节一致、`unanchored=0`）：**`onset_cross` 相比 S11 基准（`149`）大幅下降 `-36.2%` 至 `95` 例**（`gap==1` `111 -> 82`、`IoU>=0.30` `134 -> 83`、真目标缺检误抓 `94 -> 57`、真目标在场 `55 -> 38`、`2-cycle` 互换 `34 -> 20`）；但 `post_switch_cross` 从 `31,791` 增至 `40,568/40,570`（`clean_match` `42,847 -> 35,844`），表明残留切换引发了更长持续污染尾段，亦解释了 `M1` (`57.033`) 与 `D2_SKIP_ONLY` (`63.781`) 之间 `6.748` HOTA 的剩余空间。
- **阶段 E（报告导出与交接）：** 已导出 `reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/`（`aggregate_metrics.csv` `12` 行、`per_sequence_metrics.csv` `96` 行、`s11_reference_comparison.csv`、`learning_curves.csv`、`diagnostics.json`、`cases.jsonl` & `evidence/` `16` 例配对样本、`commands.sh`、`checks.json` 全校验通过、`artifacts.json` `67` 项哈希）。工作树保持未提交、未推送。
- **精确下一步：** 停止在 Stage E，交由研究端审阅 `reports/feasibility/s12_camel_pairwise_edge_and_assignment_20261127/`，针对“成对运动边偏置大幅优于同容量单节点控制 (`+2.854` HOTA) 且削减 `36.2%` onset (`149 -> 95`)、但仍未能超过无新增边头基线 `T2_TH040` (`57.572`) 且残留切主污染尾段扩大 (`31,791 -> 40,570`)”的实证裁决 S13 方向。未执行 `git commit` / `git push`。

## 2026-11-28：S12 深度复盘、核心科研总纲固化与 S13 任务设计

- **完成内容（研究端本地审阅与文档建设）：**
  1. 审阅 S12 完整报告包（`README.md`、`aggregate_metrics.csv`、`per_sequence_metrics.csv`、`learning_curves.csv`、`diagnostics.json`、`cases.jsonl`、`evidence/*.json`、`checks.json`）与 `scripts/run_s12_camel.py` 源码：确认 `M1`（`57.033`）相对 `M0`（`54.179`）取得 `+2.854 HOTA / +4.031 AssA` 且压低 `onset_cross -36.2%`（`149 -> 95`），但暴露两大关键机制原因——① `delta_s` 无零均值约束将 `val/computed_sim_threshold` 从 `0.546` 推高至 `0.6869`，使 `@0.40/@0.50` 落在决策盲区（`7/8` 视频完全一致且 `M1+B1` 零拒识）；② 残余 `95` 例 onset 中仍有 `60.0%`（`57/95`）发生在真目标缺检瞬间，且一旦错连引发 `40,570` 帧超长尾污染（如 `0024` 仅 `6` IDSW 即掉 `-8.22`，而 `M1+B1` 可立即恢复至 `94.292`）。
  2. 按用户要求将**研究生毕业论文总目标（原创方法创新 + 权威全量基准实质提点 + 多数据集/多 seed 验证）与防陷入死循环规则（1–2 轮硬止损与路线切换门控）**写入项目核心治理文档：[AGENTS.md](../AGENTS.md)、[experiment_task_contract.md](experiment_task_contract.md)、[README.md](../README.md) 与 [创新主张台账](research_governance/innovation_registry.md)。
  3. 设计整合型 **[S13 任务书](a2a/S13_CAMEL_CALIBRATION_AND_DUAL_MEMORY_TASK.md)** 与 **[S13 版本化协议](../manifests/protocols/s13_camel_calibration_and_dual_memory_20261128.yaml)**，并更新 [LATEST](a2a/LATEST.md)：Stage B 零重训完成 S12 `M1` 标尺对齐扫阈（`0.55–0.68`）、同步 `dustbin` 校准版 `M1+B1` 及 `0024/0053` 掉分核查；Stage C/D 进行收敛三联组消融（`N1` 零中心标尺锚定 → `N2` 遮挡删帧抗误抓监督 → `N3` 永不淘汰早锚双时间尺度记忆 `K-AnchorCAMEL`），设定明确硬门槛（`HOTA >= 59.1` 且 `AssA >= 42.85` 才晋级 S14 多 seed 与完整 `val`/`test`，否则止损切向轨迹级自纠偏或更高天花板范式）。
- **范围与限制：** 本轮仅完成本地审阅、治理规范修订与 S13 任务/协议编写；未连接服务器、未执行训练/推理/TrackEval、未使用 GPU。S13 状态为 `DESIGN_READY_RUNTIME_LOCK_PENDING`（`NOT_RUN`）；未执行 `git commit` / `git push`。
- **精确下一步：** 用户审阅本轮文档更新与 S13 方案并在授权提交推送后，交由服务器按 [LATEST](a2a/LATEST.md) 领取 **S13** 从 Stage A 顺序执行至 Stage E 并返回 `reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/`。

## 2026-11-28：S13 完成 CAMELTrack 标尺校准、缺检拒识与早锚双时间尺度抗污染记忆（`K-AnchorCAMEL`，A–E 全阶段）

- **阶段 A（源码时序审计、接口实现、单测与实 batch 验证）：** 审计 `CAMELTrack@bc1c2610` `Tracklet.detections` FIFO 裁剪（`[-max_gallery_size:]`）与 `Tracklet.padded_features` 逆序堆叠（index `0` 最新、`last` 最老），并与 `CAMELDataset`（`age = img_id - image_id`，`sort_values("age", ascending=True)`）在 `100` 个 `train32` 实 batch 逐轨迹比对：**`24,957` 条有效轨迹 `monotone_violations = 0`**（`>=6` 槽占比 `99.19%`、`occ` 重叠 `IoU>=0.25` 触发率 `43.54%`）。校验 8 项复用资产 SHA256，新增 [`run_s13_camel.py`](../scripts/run_s13_camel.py)、[`test_s13_camel.py`](../tests/test_s13_camel.py)（`3 passed`：50+ 帧早锚不被淘汰、`occ_drop`/`hist_pollute` 不变量、`-inf` 填充态 step-0 等价与 2 步有限更新）及 [`export_s13_report.py`](../scripts/export_s13_report.py)，并验证 `5+5` real train/val batch 在 `N1/N2/N3` 严格有限。全流程仅用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`，`NVIDIA GeForce RTX 4090 D`），物理 GPU 0 严格禁用；权威净计算 **`7.264 GPUh`**（`<= 8.5h` 上限：B `0.909h`、C `6.028h`、D `0.328h`），含透明保留之失败/重跑维修记录 `v1/v2` 的总毛耗时 **`10.011 GPUh`**，额外数据盘占用 **`5.063 GiB`**（`<= 15.0 GiB`）。
- **阶段 B（零重训 S12 `M1` 标尺扫阈与 `0024/0053` 定向审计，`motion_dev8`）：**
  - **`M1` 扫阈**：`B_M1_CAL_055` (`0.55`: `57.001` HOTA / `40.229` AssA / `60.359` IDF1 / `960` IDSW)、`B_M1_CAL_060` (`0.60`: `56.994`)、`B_M1_CAL_064` (`0.64`: `56.208`)、**`B_M1_CAL_068`** (`0.68`: **`57.079`** HOTA / `40.270` AssA / `59.972` IDF1 / `945` IDSW，仅比 `M1@0.40` `57.033` 高 `+0.046`，仍落后 `T2_TH040` `-0.492` HOTA），**证伪 `H_cal`**。
  - **校准 `dustbin=sim_threshold` 组合 `B1`**：`B_M1_B1_CAL_058` (`0.58`: `55.953` HOTA、`0` 次拒识)、`B_M1_B1_CAL_062` (`0.62`: `56.167` HOTA、`4` 次拒识)；在 `0024/0053` 定向解剖中（两子集回放均过 ID 双射 + 帧/框/分逐行相等校验），`b_audit_0024_0053_b1_th062_v1` 将 `post_switch_cross` 从 `3,893` 压至 `3,507`（`clean 6,423 -> 6,809`），但全局闭环仍降 `-0.912` HOTA。两例定向审计同时揭示 **`8/10` (`80%`) onset 发生于真目标框当帧缺检（`true_det_absent`）** 且 `10/10` 存在 `IoU >= 0.30` 竞争。
- **阶段 C（受控三联组 20-epoch 训练 + `@0.40` 与零泄漏 `th_cal` 闭环评测，`seed=13`、`last.ckpt`）：** 3 组权威训练均完成 `20` epochs（`8,000` 步）、`0` 非有限梯度、新进程重载 diff `0.0`，合并 936 行曲线无 `NaN`（保留 `N1 v1/v2` 与 `N2 v1` 工程修理存档）：
  - `N1_KIN_SCALE_ANCHORED`（`c_n1_scale_anchored_seed13_e20_v3`，`146,305` 新增参数）：有效居中 `delta_s`（train `|delta_s| = 0.00986`、val `0.08641`），但 `epoch20_val_th = 0.6804` 仍 `>= 0.62`（`th_cal = 0.54`），闭环仅 `@0.40` `54.461` HOTA / `36.593` AssA（`@0.54`: `54.246` HOTA / `36.321` AssA）。
  - `N2_KIN_OCC_DROP_SUP`（`c_n2_occ_drop_seed13_e20_v2`，`146,305` 新增参数，`train/occ_drop_count = 38.39`，`val_th = 0.6398`，`th_cal = 0.50`）：相对 `N1 @0.40` 大幅挽回 **`+2.733` HOTA / `+4.471` AssA**，达 `@0.40` **`57.194` HOTA / `41.063` AssA / `60.528` IDF1**（`dev8` 逐视频 `6/0/2` 胜 `N1`；`@0.50`: `55.470` HOTA / `38.597` AssA）。
  - **`N3_DUAL_ANCHOR_CAMEL`**（`c_n3_dual_anchor_seed13_e20_v1`，`147,073` 新增参数，`train/hist_pollute_count = 62.09`，`val_th = 0.6209`，`th_cal = 0.48`，**Stage C 获胜组**）：`@0.40` 取得 **`57.519` HOTA / `81.365` DetA / `40.787` AssA / `59.550` IDF1 / `1,187` IDSW / `4,485` FN / `75.84 ms/f`**（相对 `N2` 再增 `+0.325` HOTA、相对 S12 `M1_TH040` 增 `+0.486` HOTA、逐视频 `5/0/3` 胜 `M1`）；`@0.48` 取得 `57.453` HOTA / `40.831` AssA / `59.674` IDF1 / `4,367` FN（逐视频 `5/0/3` 胜 `T2_TH040`）。总体较 S11 无新增头基线 `T2_TH040`（`57.572`）仍微差 **`-0.052` HOTA**（`4/0/4`）。
- **阶段 D（`N3+B1` 协同与全 `dev8` 解剖回放）：**
  - `N3_PLUS_B1_THCAL`（`d_winner_plus_b1_thcal_v1`，`@0.48`、`dustbin=0.48`）：`57.467` HOTA / `40.680` AssA / **`60.405` IDF1** / `1,114` IDSW（`dustbin_rejected = 7`，注：wrapper 汇总计数器 `tracks_with_early_anchors` 未递增自检，实际帧级已执行 `_record_early_anchor`/`_build_anchor_override`）。
  - 全 `dev8` 解剖回放（`d_winner_anatomy_dev8_v1`，与在线 MOT `8/8` 逐字节一致）：`clean_match = 34,578`、`onset_cross = 104`（`80` 例 `gap==1`、`97` 例 `IoU>=0.30`、`76` 例真目标缺检）、`post_switch_cross = 42,454`（高于 S11 `31,791` 与 S12 `40,570`）、`fp_pollution = 1,094`（低于 S11 `1,350` 与 S12 `1,649`）。
- **阶段 E（报告导出与毕业论文晋级/止损裁决，`CASE_3`）：** 已导出 [`reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/`](../reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/README.md)（`13` 行总体、`104` 行逐视频、`936` 行曲线、`16` 例配对样本与 `evidence/*.json`、`checks.json` 全校验通过、`0.0` TrackEval 差、`artifacts.json` `66` 项哈希）。**裁决命中 `CASE_3_STOP_CAMEL_LOCAL_PATCH_AND_PIVOT`**：虽然 `N2`（`p_occ_drop` 拒识监督 `+2.733`）与 `N3`（`K=4` 永久早锚 `+0.325`）在 CAMEL 受控消融内部成立，但完整 `K-AnchorCAMEL`（`57.519`）未能超越 `T2_TH040`（`57.572`）且距 `59.1` 晋级门差 `-1.581` HOTA、尾段 `post_switch_cross` 反升至 `42,454`，依 `AGENTS.md` 硬止损纪律正式终止 CAMEL 局部边/槽记忆微调分支，不扩展 multi-seed `37/73` 或官方 `val`/`test`。
- **精确下一步：** 停止在 Stage E（未执行 `git commit` / `git push`），交由研究端审阅 [S13 报告包](../reports/feasibility/s13_camel_calibration_and_dual_memory_20261128/README.md) 并正式决策切换至 Route B（轨迹级交叉后漂移检测与全局图重连）或 Route C（更高天花板端到端关联骨干如 MOTIP2）。
