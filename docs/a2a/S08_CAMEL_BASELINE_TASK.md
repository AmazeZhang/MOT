# S08：CAMELTrack 有界基线训练与评测闭环

日期：2026-11-25；任务版本 v1；状态：**设计已给出，服务器未执行，运行锁待服务器生成**。

协议：[s08_camel_baseline_20261125.yaml](../../manifests/protocols/s08_camel_baseline_20261125.yaml)。前置：[S07 研究端复盘](../research_status_review_20261125.md)。遵循 [任务规范](../experiment_task_contract.md) 与 [报告规范](../experiment_reporting.md)。用户将本任务交给服务器后，A–F 在前置门通过且预算内自动顺序执行，结束后统一回传；本地这次只编写指导，没有启动服务器。

## 1. 这轮要得到什么

问题是：**能否在单张 4090 上建立可训练、可重载、严格在线、可导出官方指标的成熟学习关联平台，并获得第一份自然失败分布？**

本轮只做 CAMELTrack，不追加新模块。成功产物是我们能继续改方法的基线、可复现训练入口和可用误差证据。工程通过与基线分数分别报告；不要求达到论文分数，不把复现或较低 loss 当创新，不按“是否超过旧 M/V”决定平台是否可运行。

S06 选择器、seeds37/73、FM、风险阈值搜索继续停止。S08 不执行跨数据集、多种新方法、多 seed 确认、论文完整两阶段复现或 test 提交。后续只能根据本轮结果另立一个具体问题。

## 2. 已核验的官方路径与实现责任

固定 [CAMELTrack commit bc1c261075b0beacfe71c2896453be83643153c8](https://github.com/TrackingLaboratory/CAMELTrack/tree/bc1c261075b0beacfe71c2896453be83643153c8)。本地实际读过此版本的 README、pyproject、主/训练/模块配置及训练和加载代码，没有安装运行。关键来源：

- [官方 README](https://github.com/TrackingLaboratory/CAMELTrack/blob/bc1c261075b0beacfe71c2896453be83643153c8/README.md)：支持预计算 Tracker States、推理与训练；论文流程是 TE 预训练再联合训练，并使用 train+val。**本任务采用单阶段 train32 训练，是平台基线，不是论文设置复现。**
- [模块配置](https://github.com/TrackingLaboratory/CAMELTrack/blob/bc1c261075b0beacfe71c2896453be83643153c8/cameltrack/configs/modules/track/camel.yaml)：默认三个 cue、batch8、单阶段默认2 epochs；默认 checkpoint 是 global，推理 override 的 similarity threshold 为0.5，与模型配置0.1不同。不能依赖隐含默认。
- [训练配置](https://github.com/TrackingLaboratory/CAMELTrack/blob/bc1c261075b0beacfe71c2896453be83643153c8/cameltrack/configs/cameltrack_train.yaml)：默认读取 `dancetrack-train.pklz`/`dancetrack-val.pklz`，与 README 的简化命名不同。自定义 split 必须同时改 dataset 元数据和 state 映射。
- [训练入口](https://github.com/TrackingLaboratory/CAMELTrack/blob/bc1c261075b0beacfe71c2896453be83643153c8/cameltrack/cameltrack.py#L276)：从当前目录 `CAMEL/last.ckpt` 自动恢复；选择 checkpoint 后 `load_from_checkpoint` 的返回未赋回 `self.CAMEL`。因此正式闭环必须在新进程显式加载选定文件并核对 SHA，不能只信训练末尾日志。
- [数据缓存代码](https://github.com/TrackingLaboratory/CAMELTrack/blob/bc1c261075b0beacfe71c2896453be83643153c8/cameltrack/train/dataset.py#L395)：已有缓存配置差异仅警告；S08 使用独立、包含 split/config hash 的缓存目录，禁止误复用别的划分。

服务器 agent 实现薄包装脚本、split/state 过滤适配、配置、日志/指标导出和针对性测试；不需要本地先替服务器装环境。代码进入本项目 `scripts/` / `configs/`，第三方改动以补丁和版本锁交付，不复制整个外部仓库。官方已知入口如下，**不是可以不加配置直接批量执行的 S08 脚本**：

```text
uv run tracklab -cn cameltrack dataset=dancetrack
uv run tracklab -cn cameltrack_train dataset=dancetrack
```

实际命令须由 A 阶段实现并解析配置后写入 `commands.sh`，包含每阶段入口、GPU 环境、工作目录和运行 ID。不要声称本任务已经提供可运行的 launcher。

## 3. 固定数据、初始化和配置

沿用 [motion_train32 / motion_dev8](../../manifests/splits/dancetrack_motion_train_dev_v1_20261112.yaml) 的逐视频列表，协议复制完整列表以便交叉核对。train32 全帧训练，dev8 全帧开发评估；新训练的 dev loss 使用同一个 dev8。全部来自官方 train，且过去已参与项目研究，**不是新留出集**。它与 S06 的 train18/internal-dev6/evaluation8 划分不同，禁止把其范围混在一个比较表里。官方 val 已有历史评测，本轮不用；test 不下载、不评测、不提交。

优先使用官方发布的 DanceTrack **train** Tracker States，从中拆出 train32/dev8，避免重跑检测/外观/姿态。只允许 bbox+appearance+keypoints 三 cue 完整状态；A 阶段记录来源 revision、文件 SHA、检测器/外观/姿态来源、维度、每视频帧和检测覆盖。缺失来源写 unknown，不能宣称与旧缓存一致或没有预训练重叠。状态若不是逐帧因果特征、缺必要 cue 或无法确认视频映射，停止回传阻断点，不默默更换数据源/重建全库。

训练与推理的 state 角色必须分开：GT 可构造训练目标/离线评价；推理只用逐帧检测和特征，清除缓存 tracker 输出/track ID，不把缓存的已有配对当模型预测。B 阶段验证去除 GT 后输出一致、截断未来后前缀输出一致。不能只因 TrackLab 引擎叫 offline 就断定离线，也不能因论文称 online 就跳过检查。

| 条件 | 初始化/用途 | 配置与边界 |
|---|---|---|
| `P_OFFICIAL` | 官方 `camel_bbox_app_kps_dancetrack.ckpt`；平台参考 | 固定权重及 SHA；报告其训练范围，不假定没见过 dev8。不可作为相同训练数据下的受控对照 |
| `T_SCRATCH` | 所有关联 TE/GAFFE 随机初始化，seed13；我们的可继续开发基线 | 总 checkpoint、各 TE、GAFFE checkpoint 都为 null；冻结检测/外观/姿态；正常训练样本，不限 S05 的104冲突分量 |

共享 inference 政策固定：`min_det_conf=0.4`、`min_init_det_conf=0.9`、`sim_threshold=0.1`、`use_computed_sim_threshold=false`；显式设模型值及 checkpoint override，关闭自动阈值。其余生命周期按 pinned config：min_num_hits0、max_wo_hits150、gallery50。不插值，不后处理补框，不调阈值。P/T 使用相同 dev8 检测/特征/输出政策/评测器，但初始化和关联训练数据可能不同，只作系统描述。

T 主训练固定：官方三 cue 架构/损失/增强，AdamW 等实际优化器以 pinned 源码为准，配置学习率1e-4、weight_decay1e-4、FP32、batch8、accumulate1、num_samples32、samples_per_video100、max_length50、train_add_val=false、multi_dataset_training=false、20 epochs、seed13。不因 GPU 多而开启 DDP。A 阶段导出全部有效配置、实际 optimizer/scheduler 和参数数目；不得仅记录本段摘要。

## 4. 执行顺序与通过条件

| 阶段 | 具体工作 | 通过后动作 / 失败处理 | 本阶段回传 |
|---|---|---|---|
| A：来源和执行锁 | 独立环境；锁官方 commit、TrackLab/KPR 等实际解析版本；下载/核验 P 权重和 train states；实现 train32/dev8 适配和命令 | 数据成员互斥且40段完整、状态映射/特征维度可用、全部路径落数据盘；生成运行前快照后进 B。缺资产/依赖则在预算内修复，仍失败就停止 | `source_audit.json`、`split_audit.json`、`resolved_configs/`、`commands.sh`、`runtime_lock.json`、安全源码补丁 |
| B：预训练推理与在线边界 | P 在 train 的 dancetrack0027 第1–400帧推理；另跑1–200前缀及去GT的1–400；同输入重复1–400一次 | MOT格式、有限数值、单帧ID唯一、输出非空且全帧覆盖记录；重复输出一致，前缀一致、去GT一致。原始帧有零检测允许，但须核覆盖；通过进 C | 4个工程run记录、输出哈希、首尾/空帧统计、因果/GT访问证据、时延/显存。无需把400帧 HOTA 当研究结果 |
| C：训练/恢复 smoke | train32 中0027前400帧，dev仅0012前200；随机初始化，seed13，100 optimizer steps，上限一个GPU卡时；在step50保存并新进程恢复继续至100 | loss/梯度有限、有效参数有更新、恢复后参数/优化器/调度器/step/RNG正确；保存后新进程加载输出与保存时eval输出一致。固定无增强训练探针的step0/100 loss均回传，下降仅诊断，不按数值调参；通过进 D | loss/step/暴露量曲线、分模块有效参数/梯度、恢复核对、checkpoint SHA和加载证据 |
| D：正常训练 | **新run从头**训练T train32；20 epochs，完整开发 loss 每epoch；epoch2为工程检查点，同一20epoch schedule继续 | epoch2无NaN、数据范围正确且剩余预算可容纳则继续，不因loss/开发表现重选超参。epoch20的last是唯一正式checkpoint；达到上限保留PARTIAL，不将提前停止当20epoch完成 | 20行以上真实曲线、每epoch步数/暴露/有效样本、学习率、参数/初始化证据、全部失败、epoch2/20权重清单 |
| E：完整闭环 | P和T最后checkpoint分别独立在线跟踪dev8全部8段，再跑官方TrackEval；同一套输入/评测 | 完成2条件×1种子标记×8视频=16条逐视频、2条COMBINED；不要用P挑选T。T seed13；P训练seed未知写NA并说明。通过进 F，失败保留缺失行 | 全精度aggregate/per_sequence；HOTA/AssA/DetA/IDF1/IDSW/FP/FN/Frag；覆盖、源结果摘要、每阶段速度与显存 |
| F：有界自然失败摘要 | 只分析T这8段已有轨迹；先从官方IDSW事件做描述，最多16个例子，不加新模型/反事实分支 | 导出后停止本轮。做不到可靠映射就标未知，不无限扩建诊断框架 | `failure_inventory.csv`、`cases.jsonl`、`diagnostics.json`、一页结论及下一步建议 |

C 使用可控步数包装并记录相较官方入口的工程改动。D **一开始就设 max_epochs=20**，不要先设2再改20导致学习率计划改变；epoch2在同一训练中存档检查，需暂停时保留完整状态。每次独立attempt用新工作目录，禁止自动捡到旧 `CAMEL/last.ckpt`；合法恢复须标父run并复制/指向指定SHA。C的权重、优化器和样本缓存不能流入D。

E 的主输出是基线完整度与官方指标，不设“必须比P/旧M高”的研究门，不据dev8挑epoch或换阈值。P可能训练过dev8，P−T不解释为方法差异。旧M/V不在本轮执行矩阵，不从S06八段结果拼接差值。T分数很低也如实交付，研究端再区分实现问题、训练不足和平台适配问题。

## 5. F 阶段的最小失败证据

用官方评价关联/IDSW记录定位事件；每视频按 `SHA256("13:<sequence>:<frame>:<gt_id>")` 升序最多取2个，全部候选数量也回传。没有事件的视频记录0。不能重定义官方IDSW；若事件细节无法导出，只报告已有总体/逐视频指标及明确缺口，不制造自定义IDSW替代。

`failure_inventory.csv` 每个采样事件给出：sequence/frame/GT ID、事件来源、前后预测身份映射、前一个匹配帧间隔、当前是否存在IoU≥0.5的合格检测（不是tracker合法边证明）、当前附近是否有其他GT框IoU≥0.3的竞争身份、其他受影响身份、unknown原因、证据引用。缺失和拥挤可重叠，不把标签解释为原因。报告所有事件数和抽样分母，不从最多16例估总体类型比例。案例给帧号/坐标/邻近窗口的小证据，不强求原视频上传。

研究端据此判断下一轮优先“观测中断后重关联”还是“相似外观竞争”，若证据不足明确不选；不能预先宣布某一模块有创新，也不为了候选方向补无限诊断。

## 6. 预算、修复与环境

- 单张空闲4090，先查占用，只选物理1–7；GPU0永久禁用，不能使用/终止其他任务。记录CUDA_VISIBLE_DEVICES和逻辑映射；CPU任务设空值。
- 仓库 `/home/imc/yzy/MOT`；第三方工作副本、环境缓存、权重、states和输出均在 `/media/imc/data/mot-data`（已有环境可复用但记录）；本项目私有机器记录在ignored `local/`。
- 总GPU预算 **24卡时**，其中B≤1、C≤1、D≤20、E≤2；A若需要CUDA也计入总预算，不能挤占后续到超限。总执行墙钟≤72小时（含下载等待）；主动工程投入≤2个工作日；新增资产/缓存≤100GiB且保留至少50GiB空闲，超过先停止回传。均为投入上限，不是耗时预测。
- 同一阻断最多2次有针对性的工程修复后重试，每次新attempt；限路径/API/依赖、无语义变化的split适配/日志/恢复问题。不能把普通空输入错误当科学失败。第三次仍堵或预算耗尽即交付PARTIAL/FAILED。
- FP32 batch8若OOM，先核对泄漏与非法超长样本；仍不适配则停止并回传峰值/形状，**本轮不自动改batch、精度、梯度累积或架构**。未来任务可据证据给合适配置。
- 不开公网W&B上传；`use_wandb=false`在上游仍创建offline logger，显式禁联网/设offline目录并计入产物。密钥不进Git。

## 7. 必须回传的包与最终判断

权威报告：`reports/baselines/s08_camel_baseline_20261125/`，未完成也沿用同目录并诚实状态；从 [report_bundle](../templates/report_bundle/README.md) 开始。除其README/manifest/checks/artifacts/曲线/指标/诊断/案例外，必须有A表列出的来源/划分/配置/命令/运行锁，以及 `failure_inventory.csv`（不能导出则omissions解释）。Git保存轻量源指标摘要和证据；大checkpoint/states/轨迹/日志留服务器，清单列SHA/大小/来源。

`checks.json` 至少逐项记录：split与实际loader成员、随机初始化、缓存隔离、GT-free与前缀、重跑一致性、有效梯度/恢复、指定checkpoint重载、两组输入一致性、指标与源值一致性、16+2覆盖、资源/GPU安全。未做写NOT_RUN，不生成“全通过”。环境通过、指标正确、方法有效是不同结论。

最终回信回答：①完成到哪一步、平台是否可继续开发；②T的训练和真实dev8指标如何、P参考的可比性限制；③自然失败证据支持优先研究什么、仍缺什么。决定代码沿用报告规范：A–C通过但D/E未完整为 `ENGINEERING_ONLY` 或 `INCONCLUSIVE`；D/E可信完成可 `CONTINUE`（仅表示可设计下一轮，不自动执行）；技术不可行则 `INCONCLUSIVE` 并指出阻断，不声称CAMEL方法无效。

完成F或任一停止条件后更新 `reports/README.md`、`docs/progress_log.md`、`docs/a2a/LATEST.md`，保留任务和权威报告链接。无明确授权不commit/push。**下一轮的新方法、两阶段训练或更换MOTIP均不自动启动。**
