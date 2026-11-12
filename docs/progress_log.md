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
