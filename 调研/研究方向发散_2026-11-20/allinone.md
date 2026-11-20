# MOT 方向发散：检索记录与证据目录

检索截止：2026-11-20。目标：结合本项目 P1/R1/R2/E0–E5，筛选适合研究生毕业设计的下一轮验证。主报告：[方向评估与最小验证方案](方向评估与最小验证方案.md)。本记录不是系统综述，也不是新颖性证明。

## 多源 API 结果

查询：`multi-object tracking appearance temporal`、`multi-object tracking association uncertainty`、`multi-object tracking motion state`；年份 2024–2026，程序传入截止日 2026-11-20；每源每查询最多 3 条。这是浅层入口，主要结论另由定向搜索和原文核查支持。

原始命中：arXiv=9，DBLP=0，OpenAlex=0，OpenReview=0，Semantic Scholar=0，Crossref=3；共 12 条、去重后 6 条（合并 6 条重复）。逐条读标题及可用摘要，语义排除 3 条与本轮单摄像头 RGB 人体 MOT 不直接相关的结果，保留 3 条，保持程序排名。摘要缺失的条目保守保留，不用标题推断其贡献。

| 原排名 | 论文 | 日期／状态 | 引用数 | 词汇匹配分 | 来源 | 摘要审读与用途 |
|---|---|---|---|---|---|---|
| 1 | [Multi-Object Tracking with Integrated Appearance and Mamba-Based Motion Features](https://doi.org/10.21203/rs.3.rs-7318459/v1) | 日期缺失；预印本标识，出版状态未核实 | 未获得 | 7 | Crossref | 摘要提出运动与外观耦合；只能作为相关线索，不能作强结果依据 |
| 2 | [MambaMOT: State-Space Model as Motion Predictor for Multi-Object Tracking](https://arxiv.org/abs/2403.10826) | 2024-03-16；arXiv | 未获得 | 5 | arXiv | 学习运动替换 KF；说明单换运动架构已有先例 |
| 4 | [Tracking the Traffic: Multi-Vehicle Tracking Using Motion Estimation and Object Association](https://doi.org/10.2139/ssrn.5281652) | 日期与出版状态未核实 | 未获得 | 5 | Crossref | 无摘要，保留待核；不参与方法推荐 |

分数是程序关键词相关度，不是质量或创新评分。原始返回中的引用数 0 不被解释成真实零引用；未核实日期的条目不能确认满足时间窗。未过滤记录与摘要在忽略目录 `local/research_20261120/paper_search_results.json`，可用于恢复筛除项；这些原始材料不提交 Git。

检索错误：DBLP 返回解析错误，OpenAlex 与 Semantic Scholar 返回 HTTP 429，Crossref 两个查询返回 HTTP 429。OpenReview 返回 0 且没有记录错误，不能据此推断该领域没有论文。CLI 与 skill 文档不一致，`--json` 被拒绝，之后改用已检查的 `search_papers` 程序接口完成导出。逐条 stderr 原文附在文末。

## Model Knowledge

0 条未经联网核查的记忆补充。本轮没有把模型回忆当作来源；补充工作均在以下原始论文页面或作者仓库核查。旧论文用于追溯近邻，明确不属于 API 的 2024–2026 时间窗。

## 定向补检与原始来源核查

定向关键词覆盖 learnable association、history corruption、exposure bias、rollout、reliable memory、selective appearance/mask、Flow Matching、conflict association。以下 15 项是影响推荐的保留材料；编号仅供查阅，不是科学价值排名。

| 编号 | 原始论文／年份 | 已核查内容 | 对本项目的作用与限制 |
|---|---|---|---|
| W1 | [Multiple Object Tracking as ID Prediction / MOTIP，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) | 正文训练增强与关联设计；作者仓库安装、训练、模型表 | 首选可学习关联入口；随机遮挡/身份交换已存在 |
| W2 | [MOTIP2: Spatial Priors for End-to-End Multi-Object Tracking，2026](https://arxiv.org/abs/2610.10391) | HTML 方法、训练附录；作者标注 BMVC 2026 accepted | 空间交换、损失与表征先例；本轮未核实官方可用代码/权重 |
| W3 | [LA-MOTR: End-to-End Multi-Object Tracking by Learnable Association，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.html) | 会议页面摘要与方法说明 | 支持关联学习路线；代码地址访问 404，不能当已验收底座 |
| W4 | [Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification，2023](https://arxiv.org/abs/2302.11813) | 摘要；官方 README | 低成本深度外观对照；默认命令含后处理，严格在线需关闭 |
| W5 | [Lost and Found: Overcoming Detector Failures in Online Multi-Object Tracking / BUSCA，ECCV 2024](https://arxiv.org/abs/2407.10151) | 论文与官方 README 的候选、在线范围、权重和训练章节 | 学习候选选择已有先例；推理可用线索充分，训练章节尚未提供完整教程 |
| W6 | [SAM-Deep-EIoU: Selective Mask Propagation for Multi-Object Tracking，2026 预印本](https://arxiv.org/abs/2606.13033) | HTML 的选择性传播与比较设置 | 与“不确定时引入掩码验证”高度接近；未核实代码和各设置因果时序 |
| W7 | [Hierarchical Multi-Prototype Appearance Memory: A Plug-and-Play Module for Identity-Stable Online Multi-Object Tracking，Electronics 2026](https://www.mdpi.com/2079-9292/15/11/2357) | 摘要与作者对消融的说明，网页部分加载受限 | 可靠写入、多原型和保守恢复已有先例；未独立复核作者实验 |
| W8 | [MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking，ICCV 2023](https://arxiv.org/abs/2307.15700) | 论文方法概要与作者仓库入口 | 长期记忆、时序聚合不是空白；本轮未运行代码 |
| W9 | [LiAM-SAM: Lifecycle-Aware Memory for Robust SAM2-Based Multi-Object Tracking，2026 预印本](https://arxiv.org/abs/2609.28078) | 摘要 | 生命周期与可靠长期记忆近邻；摘要级信息，不据此推断部署成本 |
| W10 | [Eliminating Exposure Bias and Metric Mismatch in Multiple Object Tracking，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Maksai_Eliminating_Exposure_Bias_and_Metric_Mismatch_in_Multiple_Object_Tracking_CVPR_2019_paper.pdf) | 正文引言与训练问题定义 | 自身错误训练及评价代理已有历史；不能宣称首次闭环训练 |
| W11 | [Systematized event-aware learning for multi-object tracking，UAI 2022](https://proceedings.mlr.press/v180/lee22a.html) | 官方论文摘要与书目信息 | 模拟跟踪候选与事件感知关联训练先例 |
| W12 | [DeconfuseTrack: Dealing with Confusion for Multi-Object Tracking，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf) | 正文混淆分解设计 | 对选择性处理竞争事件构成近邻；尚需逐模块比对 |
| W13 | [Focusing on Tracks for Online Multi-Object Tracking / TrackTrack，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) | 会议摘要和方法概要 | 轨迹视角分配与初始化已有先例；不能把换匹配顺序当创新 |
| W14 | [VLA-ReID: Video-Level Association for Re-Identification in Multi-Object Tracking with Highly Similar Objects，2026 预印本](https://arxiv.org/abs/2607.17157) | 摘要 | 视频级候选关联表征近邻；蜜蜂场景不能直接外推 DanceTrack |
| W15 | [MeMoSORT: Memory-Assisted Filtering and Motion-Adaptive Association Metric for Multi-Person Tracking，2025 预印本](https://arxiv.org/abs/2508.09796) | HTML 方法概要 | 记忆补偿运动和自适应匹配已有先例；不是恢复 FM 的证据 |

未纳入核心证据的搜索结果包括跨相机 3D、RGB-T、语言指代、需 IMU 的跟踪器，以及仅从二次网站看到且未核实的论文。它们可提示后续关键词，但不用于确认本项目新颖性。遇到标注晚于截止日的出版月份，不据其搜索摘要作当前已发表证据。未从搜索结果数量推断研究热度。

## 可复现性核查

| 入口 | 本轮已确认 | 仍需验收 |
|---|---|---|
| [MOTIP](https://github.com/MCG-NJU/MOTIP) | README、INSTALL、GET_STARTED、MODEL_ZOO 可读取；存在训练配置与预训练/完整模型链接 | 实际下载、许可证归档、目标服务器编译与推理；版本与历史权重兼容；训练数据污染 |
| [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT) | 官方 README、外观与缓存说明、权重链接、默认后处理命令 | 旧依赖兼容、权重下载、固定本项目检测输入的适配 |
| [BUSCA](https://github.com/lorenzovaquero/BUSCA) | 官方 README、5 个跟踪器适配、权重链接、GPLv3 标注；训练章节 Coming soon | 权重实际可用性、DanceTrack 适配、训练代码完整性；不照抄全部依赖安装 |
| [LA-MOTR 论文给出的仓库](https://github.com/PenK1nG/LA-MOTR) | 论文给出地址 | GitHub 连接器请求 README/仓库返回 404；可能是访问或发布问题，不能断言永远无代码 |

GitHub 文件 blob 核查：MOTIP `docs/GET_STARTED.md` 为 `63933e23ebc3fb9b6ed515cdd48a8c1a0a17a5c4`；BUSCA README 为 `0c9b1686c78fae619857d84425f731f3d78f3677`。这不是整个仓库 commit，也不是权重哈希。没有运行或下载模型。

## Summary of all searched results

### 1. Overview

API 为 2024–2026 的三组通用关键词，返回 6 条唯一记录、保留 3 条；另保留 15 项经原始页面核查的定向材料，总计 18 项。补检包含 2019/2022/2023 的基础近邻。覆盖视觉关联、训练历史、冲突处理、可靠记忆；不足以保证查全。

### 2. Trends

本样本能看到从人工匹配与深度外观到直接学习关联、再到结构化训练先验的路线。MOTIP/MeMOTR 出自同一 MCG-NJU 作者团队，但不能用这个小型、偏向性样本判断整个领域的产出趋势。2026 材料中多篇仅为预印本；会议接收状态、作者实验报告和独立复现需区分。

### 3. Key themes

- 学习关联：W1/W2/W3。选底座先查可运行性，再谈方法改进。
- 训练历史与错误暴露：W1/W10/W11。真实错误训练已有先例，需比较匹配预算和结构差异。
- 选择性候选与冲突处理：W5/W12/W13。必须把不动作与他人损伤纳入评价。
- 记忆与额外视觉信息：W6/W7/W8。新信息是否有效与怎样安全使用是两项独立问题。
- 多帧表征与运动学习：W14/W15/API 第 2 项。跨对象、跨数据集的证据不能直接转成我们的收益。

### 4. Keywords frequency

仅统计三个 API 保留标题，统一大小写及连字符，按“含该概念的标题数”计数；不把手工补检和摘要混入。小样本并列项选技术相关者，不代表领域热度。

| Keyword | Count |
|---|---|
| Multi-object / Multi-vehicle | 3 |
| Tracking | 3 |
| Motion | 3 |
| Mamba | 2 |
| Appearance | 1 |

### 5. Most cited by accepted paper

| Rank | Title | Year | Citations |
|---|---|---|---|
| 不可计算 | API 未取得可靠引用数据；不以缺失值 0 排名 | — | 未获得 |

### 6. Most cited by first author

| Rank | Author | Papers in set | Total citations |
|---|---|---|---|
| 不可计算 | — | — | 未获得 |

### 7. Recommendations for reading

1. W10：先明确 exposure bias 不是新概念，避免重新发现旧问题。
2. W1：理解可学习身份关联与现成训练增强，选可操作底座。
3. W2：检查空间结构增强对候选 A 的直接重合。
4. W5：比较可部署的候选选择、未匹配处理与在线限制。
5. W7：检查可靠记忆方向的重合，再决定是否暂缓。

## 对选题的实际判断

优先做“学习型视觉关联底座 + 自然错误机制验证”；候选 A 研究竞争身份污染的结构化训练，B 研究包含原方案的合法冲突集合决策，C 可靠记忆/分割验证暂缓。A/B 都有重要先例，不宣称新颖。先导试验若只证实强基线有效，收获就是可靠底座与问题收缩，不能包装成新方法结果。具体输入、对照、否定条件和分阶段安排见主报告。

## 检索错误原文

完整 stderr 在本地 `local/research_20261120/paper_search_api_stderr.txt`；以下按实际记录附入，不将失败来源计为成功无命中。

```text
[dblp] Error on query 'multi-object tracking appearance temporal': Expecting value: line 1 column 1 (char 0)
[dblp] Error on query 'multi-object tracking association uncertainty': Expecting value: line 1 column 1 (char 0)
[dblp] Error on query 'multi-object tracking motion state': Expecting value: line 1 column 1 (char 0)
[open_alex] Error on query 'multi-object tracking appearance temporal': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=multi-object+tracking+appearance+temporal&filter=publication_year%3A2024-2026&sort=relevance_score%3Adesc&page=1&per-page=3
[open_alex] Error on query 'multi-object tracking association uncertainty': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=multi-object+tracking+association+uncertainty&filter=publication_year%3A2024-2026&sort=relevance_score%3Adesc&page=1&per-page=3
[open_alex] Error on query 'multi-object tracking motion state': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=multi-object+tracking+motion+state&filter=publication_year%3A2024-2026&sort=relevance_score%3Adesc&page=1&per-page=3
[semantic_scholar] Error on query 'multi-object tracking appearance temporal': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=multi-object+tracking+appearance+temporal&offset=0&limit=3&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2024-2026
[semantic_scholar] Error on query 'multi-object tracking association uncertainty': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=multi-object+tracking+association+uncertainty&offset=0&limit=3&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2024-2026
[semantic_scholar] Error on query 'multi-object tracking motion state': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=multi-object+tracking+motion+state&offset=0&limit=3&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2024-2026
[crossref] Error on query 'multi-object tracking appearance temporal': 429 Client Error: Too Many Requests for url: https://api.crossref.org/works?query=multi-object+tracking+appearance+temporal&offset=0&rows=3&filter=from-pub-date%3A2024%2Cuntil-pub-date%3A2026&sort=relevance&order=desc
[crossref] Error on query 'multi-object tracking association uncertainty': 429 Client Error: Too Many Requests for url: https://api.crossref.org/works?query=multi-object+tracking+association+uncertainty&offset=0&rows=3&filter=from-pub-date%3A2024%2Cuntil-pub-date%3A2026&sort=relevance&order=desc
```

CLI 初次调用的参数错误：

```text
usage: search_papers.py [-h] [--query QUERY] [--queries QUERIES] --start-year
                        START_YEAR --end-year END_YEAR
                        [--max-papers MAX_PAPERS]
                        [--sources {arxiv,dblp,open_alex,openreview,semantic_scholar,crossref} [{arxiv,dblp,open_alex,openreview,semantic_scholar,crossref} ...]]
                        [--no-parallel] [--start-date START_DATE]
                        [--end-date END_DATE] [--raw] [--min-score MIN_SCORE]
search_papers.py: error: unrecognized arguments: --json local/research_20261120/paper_search_results.json
```
