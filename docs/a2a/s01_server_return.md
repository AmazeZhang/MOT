# S01 服务器回传：S0 工程门与 S1 固定外观基线

日期：2026-11-21。状态：**COMPLETE_EXPLORATORY**。按既定协议完成 S0 和 S1；未训练候选方法，未使用 heldout 或官方 val/test。完整逐视频表见 [结果报告](../../reports/feasibility/s01_visual_association_20261120/README.md)。所有原始产物在 `/media/imc/data/mot-data/runs/s01_visual_association_20261120/`，未放进 Git 仓库。

## 运行身份和环境

| 项目 | 固定值或证据 |
|---|---|
| 项目与协议 | HEAD `7ac72619515fcd30c4dc5fe29facf5197585bcbf`，本次代码尚未提交；[协议](../../manifests/protocols/s01_visual_association_handoff_20261120.yaml) SHA256 `f99c994a2ab622f3e6f7b1156a1b94410ab53c1e79275aa3076e6e57700ffa5c`；各运行锁记录源码、补丁和输入哈希 |
| 外观来源 | 官方 Deep-OC-SORT `6bb51d027b137233f5c520b6fcc4f2ae387a6ba9`，DanceTrack `dance_sbs_S50.pth` SHA256 `004beb12f0dd9dcf355bc57f5b8cc66b6904fc6d5599fb41ee9403c4c659ab29`；训练集与本次 train-fit 重叠状态 unknown / possible |
| MOTIP 独立分支 | 官方 `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`，DanceTrack 权重 SHA256 `c642793c1f3d526aa8537d59bc6a3d201b44b16099076519ffad5ec379050800` |
| OC-SORT / TrackEval | `bb2f6867a41082558d95ab47a854a3918de7831b` / `12c8791b303e0a0b50f753af204249e622d0281a`；源 det_id 窄补丁 [文件](../../manifests/upstreams/patches/s01-ocsort-det-id-provenance.patch) SHA256 `b4b7bc7ae4822a01660db4bb801b02dad82e1a32571e0f5e1753f5ed147644d7` |
| 环境 | `/media/imc/data/mot-data/envs/s01-appearance`、`/media/imc/data/mot-data/envs/s01-motip`；CPU 回放与 TrackEval 使用 `/home/imc/anaconda3/envs/mot-det`（Python 3.10.22，`lap.lapjv` 0.5.12）。逐运行依赖版本见数据盘 runtime lock。|
| 数据与 GPU | DanceTrack train 在 `/media/imc/data/mot-data/datasets/dancetrack/train`；检测缓存在 `/media/imc/data/mot-data/cache/detections/ocsort_yolox_dance_bb2f686_b8d1/train`。特征和 MOTIP 仅用物理 GPU1，CPU 评测显式 `CUDA_VISIBLE_DEVICES=''`；物理 GPU0 未参与。|

## 实现与工程验收

新增 [外观提取入口](../../scripts/extract_s01_appearance.py)、[四组回放入口](../../scripts/run_s01_visual_baselines.py)、[评测入口](../../scripts/summarize_s01_visual_baselines.py)、[det_id 集成测试](../../tests/test_s01_visual_integration.py)；运行锁位于 `manifests/runtime/`。外观仅进入首轮合法高分边；Z 权重为零；L 最近特征，M 旧记忆权重 0.9 的 EMA，L/M 加分权重均 0.25。没有改检测、后续关联或输出政策。

核心 11 项、det_id 集成 4 项测试通过。S0 `dancetrack0027` 200 帧特征 1,542 条，前 100 键直接重算通过；G 对归档 E3 E 帧/ID 精确且框误差 ≤1e-4，Z 对 G 首轮配对、真实更新与输出一致，L/M 各有 2 帧首轮配对改变。MOTIP 官方 `submit` 200 帧输出 1,673 行、13 个 ID、0 重复/非法/非有限，物理 GPU1 监测峰值 1,339 MiB；未用 GT 计算 MOTIP 指标。

保留失败：`s0_smoke1` 因单张和逐帧批量 FP32 数值差异超容差而失败，`s0_smoke2` 在相同批量形状下通过；`s0_replay1` 因 G 回调状态遗漏失败，`s0_replay2` 修复通过；`s0_motip1` 自动下载的辅助 ResNet50 已移至数据盘并在 `s0_motip2` 锁定重跑。官方 deformable attention 前向和前五组梯度检查通过，最后两组高维检查主动中止，不能称全套通过。修复没有改变协议参数。

## S1 结果及可追溯性

预选 pilot `0051/0016/0072/0052` 共 5,772 帧、32,942 条特征。工程门通过后继续完整 24 段，共 **25,328 帧、186,676 条有效 FP32 特征、0 无效**。24 段 G 轨迹与归档 E3 E 逐字节相同，Z 与 G 逐字节相同；全部工程门通过。L/M 分别在 132/138 帧改变首轮配对集合，占全部帧 0.52%/0.54%。

| 组 | COMBINED_SEQ HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| G | 60.01 | 87.31 | 41.28 | 58.47 | 977 | 502 | 6,738 |
| Z | 60.01 | 87.31 | 41.28 | 58.47 | 977 | 502 | 6,738 |
| L | 60.74 | 87.24 | 42.32 | 59.33 | 902 | 501 | 6,698 |
| M | 62.47 | 87.57 | 44.60 | 61.03 | 902 | 501 | 6,680 |

M−G 的 HOTA/AssA/IDF1 为 `+2.46/+3.32/+2.56` 个百分点，IDSW `−75`；M 对 G 逐视频 HOTA 上升 18 段、下降 6 段。pilot M−G HOTA `+8.70` 点，完整 24 段为 `+2.46` 点。逐视频和完整指标、L−G/M−L 见 [结果报告](../../reports/feasibility/s01_visual_association_20261120/README.md)。

**权威重评**：`s1_full_evaluation2/summary.json` SHA256 `b3585deb2eebc5132ac898478a69d8bdb803f9b989c0bdaf8c88ea0bf6ddabb0`；`frozen_inputs.json` SHA256 `ed50243913084d3df2840423feb2bb3ffb3b010667dbc8cf4ebf8d483340fcc2`；[评测运行锁](../../manifests/runtime/s01_visual_association_s1_full_eval2_20261121.json) SHA256 `473a092e59f06033419f2bf2f145e5476e871f9fdbc923c2b721c9d0b7fab460`。锁覆盖 TrackEval 本地补丁、46 个 Python 源文件、指标投影脚本及 24 个 GT 文件；192 个轨迹/trace 文件和 24 个特征缓存哈希在 GT 读取前冻结。首次 `s1_full_evaluation1` 因评测源码锁不完整保留作历史记录；重评的 results/deltas/costs/视频顺序与首评精确相同。

24 段特征抽取累计约 742 秒 GPU1 墙钟、模型调用约 373 秒，PyTorch 峰值分配约 710 MB；四组跟踪合计约 5.14 CPU 核时，首次 TrackEval 约 74 秒墙钟。回放为诊断额外计算无外观配对并写 trace，不能把该计时称为部署 FPS。

## 解释与下一步

这是一组固定操作点、复用 DanceTrack train-fit 的描述性基线。外观权重训练数据重叠未排除；pilot 包含于 24 段，非独立重复；没有独立 heldout、val/test、多随机种子、参数搜索或新方法训练。结果支持“在当前固定检测和简单首轮外观接入下，完整 fit 集的关联指标方向为正，同时部分视频为负”，不支持泛化收益或新方法有效性主张。完整性审阅见 [实验审计](../../reports/feasibility/s01_visual_association_20261120/EXPERIMENT_AUDIT.md)。

下一步根据全量自然错误和审计限制，审议候选 A 的最小可行性实验与预写成功/停止条件；任何新训练或独立划分评测另立协议。当前工作树保留供 Git 审查，未提交、未推送。
