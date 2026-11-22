# S0/S1 服务器 agent 任务书

日期：2026-11-20。用途：通过本 Git 仓库交接研究设计、核心代码、服务器实现与结果。**可交给服务器 agent 实施，不是用户直接复制即可运行的训练脚本。**

## 1. 任务边界与完成定义

本地研究端负责问题、固定对照、核心计算与结果审阅；服务器 agent 负责环境、第三方版本/权重验收、提取器、跟踪接入、缓存、批量运行、评测和真实结果回传。

此次仅实施 S0/S1：建立深度外观对照，并单独验收 MOTIP 官方推理。**不训练候选 A/B，不启动 FM，不恢复 E5 heldout，不跑官方 val/test。** 已有 STOP 保留。环境排障与必要适配由服务器 agent 自主完成；缺依赖或权重拿不到时记录具体问题，不自行更换研究任务或宣称通过。

完成的含义是：返回环境与来源记录、工程门结果、四组固定对照的整段结果、完整逐视频表和限制。性能不提升也属于有效完成；不允许为了“通过”在同一轮调权重/阈值。

必读顺序：

1. 仓库根 [AGENTS.md](../../AGENTS.md)。
2. [本轮协议](../../manifests/protocols/s01_visual_association_handoff_20261120.yaml)，作为实验语义的唯一配置来源。
3. [核心代码](../../src/mot_research/association/appearance_core.py)与[测试](../../tests/test_appearance_core.py)。
4. [方向评估](../../调研/研究方向发散_2026-11-20/方向评估与最小验证方案.md)理解背景，不把后续 A/B 候选当本轮待执行事项。
5. [回传模板](s01_server_return.md)。

## 2. GitHub 交接约定

- **交接状态（2026-11-20）：** 本任务书已随 `7ac7261` 推送到 `origin/main`，服务器工作区已快进拉取。执行前仍须核对实际 `HEAD` 和工作树状态；仅看到旧 `cdac367` 不代表已收到交接。
- 开始前保存 `git status --short`、`git rev-parse HEAD`；若工作树有修改，保存相关补丁和哈希，不 reset/覆盖。协议、核心代码与服务器实现都记录实际 SHA256。
- 如创建分支，使用 `codex/s01-visual-baselines` 或同前缀的唯一名称。不自动建立 PR、远程评论或消息。本任务书不覆盖 AGENTS.md 的“提交/推送需用户明确要求”；有用户授权后再同步 Git。
- 仓库回传代码、协议锁、简短报告、评测摘要与原始文件哈希。数据、特征、权重、大日志、逐帧轨迹留数据盘。连接信息及机器私有记录放忽略的 `local/`。
- 每次回传更新 `docs/progress_log.md` 和 `docs/a2a/s01_server_return.md`，分别列明完成、失败、未做、下一步。用 `NOT_RUN / ENGINEERING_PASS / ENGINEERING_FAIL / COMPLETE_EXPLORATORY / BLOCKED_DEPENDENCY`，不把工程 PASS 称为方法有效。
- 研究端下一次依据 Git 中报告与哈希分析；**不假设 GitHub 会自动唤醒另一个 agent**。由用户让相应 agent 同步并继续即可。

## 3. 版本与环境：服务器补齐后再运行

协议的设计已固定，运行身份尚待服务器锁定。服务器应新增 `manifests/runtime/s01_visual_association_<run_id>.json`，记录协议列出的全部字段，并填写非占位值。代码/权重不明时不得开始正式特征提取或对照；工程排障允许先做最小加载测试并标注 smoke。

已固定的依赖：OC-SORT `bb2f6867a41082558d95ab47a854a3918de7831b`、TrackEval `12c8791b303e0a0b50f753af204249e622d0281a` 及本仓库已有评测补丁。新增依赖从 [Deep OC-SORT 官方仓库](https://github.com/GerardMaggiolino/Deep-OC-SORT)、[MOTIP 官方仓库](https://github.com/MCG-NJU/MOTIP)选取可用版本，**先固定完整 commit，再读取对应版本文档和权重配置**，不能运行到中途更新 main。

外观提取器选择规则：使用 Deep OC-SORT 的 DanceTrack 路径实际使用的模型和官方权重，不自行换 CLIP、DINO、SAM 或随机网络。记录模型架构、权重文件名、官方来源 URL、实际 SHA256、训练数据披露状态、许可证和 crop/resize/颜色通道/归一化细节。未披露的训练数据写 unknown，不写 no overlap。若该权重不可获得，返回依赖阻塞，提供替代建议但不混入本实验。

MOTIP 单独固定官方 DanceTrack 配置及兼容权重。用本任务指定的 train 单视频做工程验收，不拿其指标与现有 YOLOX 管线比较。不把训练数据中视频上的结果称作泛化。禁止冒用官方 test 指标作为本机验收数值。两个分支分别锁定各自依赖；MOTIP 依赖失败不阻塞固定检测的外观分支，回传时分别标注状态。

本项目 `pyproject.toml` 限定 Python >=3.10,<3.11；MOTIP 推荐环境不同。**保持现有回放环境；特征提取与 MOTIP 使用独立环境**，通过缓存/文件交换。无需为兼容一个第三方仓库全局放宽本项目 Python 约束。第三方代码放数据盘；可维护本地链接，但不要把整库、权重加入 Git。

服务器 repo：`/home/imc/yzy/MOT`。数据/下载/第三方/缓存/输出：`/media/imc/data/mot-data`。所有 CUDA 工作显式选择当前空闲的物理 GPU1–7；GPU0 禁用，记录 GPU UUID、物理编号与 `CUDA_VISIBLE_DEVICES`。回放和 CPU 测试用 `CUDA_VISIBLE_DEVICES=''`。环境创建和实际运行命令由服务器 agent 填入回传文档，不使用猜测的统一安装脚本。

## 4. 输入与缓存契约

使用既有 DanceTrack train 检测缓存，路径和哈希入口已在协议中固定。源 CSV 字段为 `frame,det_id,x1,y1,x2,y2,score,class_id`，坐标为原图连续 xyxy。源检测框和检测分数不修改；crop 时的边界裁剪另记 `crop_xyxy`。

每个特征记录以 `(sequence, frame, det_id)` 为主键，保存原框、分数、裁剪框、FP32 特征、valid、invalid_reason。可以每序列 NPZ + JSON 元数据；加载时 `allow_pickle=False`。存储顺序不保证按分数排列，不能用“高分 ID 列表 zip 全部框”连接。使用 `select_high_detections` 同时筛选 ID 和数据，再按 ID 取特征。

提取器 `eval()`、无梯度、FP32，保持官方单框预处理；不得使用 E5 的 HSV 内部 70% crop。每个高分检测必须有一条记录；无效 crop/非有限/零特征用零向量和明确无效原因，不能删除行。推理和记忆代码通过零向量识别缺失，加载器必须落实 `valid=False => zero vector`。保留特征提取耗时与缓存读取耗时，缓存回放速度不等于端到端速度。

直接提取与缓存读取对照：取按 sequence/frame/det_id 排序的前 100 个键（不足则全部），逐项比较 valid、ID、框、分数；特征 `atol=1e-5, rtol=1e-4`。范围内无效特征比例超过 1% 先排查提取器。这是工程门，不是筛数据规则；不能删除难例后继续。

## 5. 核心算法与插入点

四组固定对照：

| 组 | 首轮关联 | 外观记忆 | 用途 |
|---|---|---|---|
| G | 原 CV+E | 无 | 复核历史基线 |
| Z | 接入新路径，外观权重 0 | EMA，alpha=0.9 | 检查接入是否改变基线 |
| L | 原首轮分数 + 0.25 × cosine | 最近一次真实匹配特征 | 简单深度视觉对照 |
| M | 同 L | 归一化 EMA，旧记忆权重 0.9 | 多时刻聚合对照 |

0.25/0.9 是本轮固定的初始操作点，不是论文参数复现或已知最优值。本轮不网格搜索。负结果只适用于这个提取器、注入位置和操作点，不概括全部视觉方法。

**分数插入点已按固定 OC-SORT 源码核查：** `trackers/ocsort_tracker/association.py::associate` 的非简单匹配分支原本调用 `linear_assignment(-(iou_matrix + angle_diff_cost))`。服务器在该位置使用 `fuse_appearance_score`，保持原 LAP 后端、原一对一 IoU fast path、原匹配后 IoU 拒绝和后续重关联流程。外观只对 `IoU >= 0.3` 且两端特征有效的边加分；这个 mask 是加分范围，**不是新增 hard gating**。不要先删掉其它边或自行改成带 dummy 的新优化器。

上述源文件 [固定版本链接](https://github.com/noahcao/OC_SORT/blob/bb2f6867a41082558d95ab47a854a3918de7831b/trackers/ocsort_tracker/association.py)，blob `78111b75419b761d63f35dc1b2301230e7e209c5`。核查过原文，不把整个第三方文件复制进核心模块。服务器保留短适配补丁与来源/许可证记录。

G/Z/L/M 均沿用现有 CV 首轮预测框接口及 E 输出政策；输出政策随各自自然轨迹执行，不复用 G 的输出身份集合。完整跟踪器状态、运动历史、视觉记忆、ID 出生计数在每组每视频独立重置。**不能把 G 的历史作为 L/M 自己的历史**。相机运动补偿、插值、新 KF、Deep OC-SORT 自适应权重等额外组件均不加入。

记忆顺序示意（服务器需实现下列占位接口，不是可直接运行脚本）：

```python
# Current frame image features may be read, but track history is through t-1.
past, written_frames = memory.begin_frame(t, valid_tracker_ids_in_request_order)
# Within the exact upstream non-fast-path association branch:
score, used = fuse_appearance_score(
    iou_matrix + angle_diff_cost, current_det_features, past,
    iou_matrix >= 0.3, weight=condition_weight,
)
# Keep upstream solver, IoU rejection, subsequent rounds, births and E output.
tracker.update_with_instrumented_first_round(...)
# Only after all rounds: actual detector-to-track matches, including births.
memory.finish_frame(t, live_track_ids, final_track_to_det_id,
                    current_feature_det_ids, current_features)
```

服务器接入必须解决：

- `valid_tracker_ids_in_request_order` 与请求中 tracker 矩阵逐列一致；核查上游预测 NaN 后的轨迹剔除，不能假设原列表永远不变。
- 在所有实际 `update(det)` 与轨迹出生位置传递源 `det_id`；最终映射由真实赋值产生。禁止依赖最近框 IoU 反查 ID，重复框时会错。
- 只写当前真实高分检测，不写 KF 预测、插值、伪观测或 GT。错配的真实检测也照写，这是自然基线；不能用 GT 清洗。
- 若一个帧没有首轮调用，仍推进 begin/finish；空帧、出生、删除、后续轮重新匹配均有 trace。缺失保持原记忆，删除清除。`begin_frame` 一帧只能一次，不能每次回调重复调用。
- 特征维度固定；缓存与 tracker ID 不混用。快路径不使用外观也要计数，防止“零干预”被误判为外观无效。
- G/Z 无需改变上游任何状态。零权重除了分数等价，还必须验证首轮匹配、最终真实配对与输出逐帧等价。

核心模块没有求解器、图像网络或 GT 入口；其中的有效性/时序检查是防接入错误，不是新的置信度方法。缺失特征边不加分可能产生覆盖差异，必须报告有记忆轨迹比例、可用边比例与损伤，不声称已控制所有关联偏置。

## 6. 执行次序与验收

### S0：环境、身份映射与单视频

1. 盘点服务器、锁定依赖与数据、保存运行身份文件。
2. 在回放环境运行核心测试。下面是本仓库已经存在的真实测试入口：

   ```bash
   cd /home/imc/yzy/MOT
   CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m unittest discover -s tests -p test_appearance_core.py -v
   ```

3. 用 `dancetrack0027` 第 1–200 帧提取特征和 G/Z 回放。用于工程，不能据这些帧选择 λ 或作研究结果。G 与相同范围历史 E3 E 对齐；Z 与 G 的配对、输出必须一致。
4. 同范围做 MOTIP 官方单视频推理验收；如果官方脚本只接受数据集子集，建立数据盘上的只读链接视图，记录选帧清单，不修改原数据。确认模型加载、输出合法、ID 唯一、NaN/Inf 为零，记录显存/速度。无评测分数门。

### S1：固定对照的完整视频

1. 从 R1 的 24 个 fit 视频按协议哈希规则选 4 个 pilot；本地已核得依次为 `dancetrack0051, dancetrack0016, dancetrack0072, dancetrack0052`。服务器复算并将名字清单/哈希写入运行身份。先四组全跑这四段，检查映射、工程门与资源开销。
2. 工程通过且预算允许，就完成全部 24 段四组结果，**不根据 pilot 指标正负决定是否扩展到其余 fit**。pilot 已完成的原配置结果可复用，不能择优重跑。
3. GT 在推理轨迹和 trace 哈希落盘后由独立评测阶段读取。运行本项目固定 TrackEval，报告 COMBINED_SEQ 与逐视频；不要把逐视频 HOTA 简单平均当整体 HOTA。
4. 完成后回传，不自动跑旧 8 段、motion-dev、val/test，不启动训练。任何修复保留失败目录，以新 run_id 重跑受影响范围；如果改变研究算法/参数，另建修订协议，不能悄悄覆盖原设计。

工程门：核心测试、det_id 对齐、特征缓存一致、G 对历史 E 的帧/ID 精确且 xyxy 误差≤1e-4、Z 对 G 完全一致、无重复匹配、无时序泄漏、未使用 GPU0。无法通过时返回工程失败；不继续生成混淆的研究结论。

本轮是基线信息探索，不设置事后“胜出”阈值。报告 L−G、M−G、M−L 的 AssA/IDF1/HOTA、IDSW/FP/FN/Frag 和逐视频方向，结合实际改变关联的比例解释。即使分数增加，仍不能称新方法；若外观几乎没参与，结论是验证覆盖不足。后续是否投入训练由这些证据与文献差异共同决定。

## 7. 必须实现和回传的文件

建议由服务器新增以下入口（**当前尚不存在，名称是接口约定，不是可执行承诺**）：

- `scripts/extract_s01_appearance.py`：特征提取/缓存与直接提取核对。
- `scripts/run_s01_visual_baselines.py`：G/Z/L/M 自然闭环、状态隔离、真实更新 trace。
- `scripts/summarize_s01_visual_baselines.py`：输入哈希、工程门、官方评测与逐视频汇总。
- 对实际接入添加集成测试：重复框不同 det_id、高低分交错、空帧、出生、轨迹删除/NaN 剔除、后续轮更新、快路径、Z 等价。

数据盘每次运行至少保存：

```text
<run_id>/
  runtime_lock.json
  environment.json
  input_hashes.json
  feature_manifest.json
  engineering_checks.json
  trackers/{G,Z,L,M}/<sequence>.txt
  traces/{G,Z,L,M}/<sequence>.jsonl
  evaluation/                 # official combined + detailed outputs
  summary.json
  logs/
```

trace 至少含 sequence/frame/condition、请求 det_ids/tracker_ids、原/最终首轮配对、最终实际更新 det_id（含出生）、记忆最后写入帧、使用外观边数、fast-path 状态和输出 ID。大矩阵可只在 smoke 保存，后续保留计数与变化边；不要省略验证 Z 所需信息。

`summary.json` 必须有协议/实现/缓存/权重哈希、完成与失败视频清单、各组全局和逐视频指标、L−G/M−G/M−L、实际外观参与率、无效/缺失特征数、配对变化数、GPU/CPU 时间、显存、工程门与失败原因。结果报告保留全部组和逐视频表，不只报最优。

本地已通过核心单元测试；这不替代服务器 GPU、提取器、跟踪适配和实际数据验收。Git 回传中应包含真实命令、版本、摘要位置/哈希与下一步，且没有凭证、权重或原始数据。
