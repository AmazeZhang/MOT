# 六源检索归档与语义初筛

检索日期：2026-11-03。36 条 API 原始记录，按规范化题名组成 30 组。题名分组仅用于显示，不将同名不同 DOI 强行认定为同一出版物。摘要缺失时保留待核实，不能仅按关键词判无关。以下包含全部组及中文摘要概述；主报告所用的定向补检论文另见核查记录。引用数缺失记为未获取。

原始查询：`occlusion robust multi object tracking motion association`；`multi object tracking generative motion`；`flow matching association aware motion prediction`。年份 2022–2026，每源每查询最多 6 条。

## 1. MambaMOT: State-Space Model as Motion Predictor for Multi-Object Tracking

- 状态：保留／直接相关线索。摘要描述学习式运动预测，可为确定性模型对照；尚未核查正文。
- 来源：arxiv；API 年份：2024；venue：arXiv；引用数：0。
- 作者：Hsiang-Wei Huang, Cheng-Yen Yang, Wenhao Chai, Zhongyu Jiang, Jenq-Neng Hwang。
- 链接：[记录 1](http://arxiv.org/abs/2403.10826v2)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

以状态空间模型学习 MOT 运动预测，针对传统线性预测器难处理的复杂运动。对本项目的主要价值是提供学习式确定性运动建模的比较方向。


本组包含 2 条原始返回；全部版本在 search_results.json。

## 2. Awesome Multi-modal Object Tracking

- 状态：排除本轮核心证据。摘要研究多模态视觉目标跟踪综述，未指向检测集合的身份关联或运动分布。
- 来源：arxiv；API 年份：2024；venue：arXiv；引用数：0。
- 作者：Chunhui Zhang, Li Liu, Hao Wen, Xi Zhou, Yanfeng Wang。
- 链接：[记录 1](http://arxiv.org/abs/2405.14200v2)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

汇总不同传感模态的目标跟踪资料与问题，重点是多模态视觉信息融合。任务中的 multi-modal 与本项目的多目标身份分配并不等价。


本组包含 2 条原始返回；全部版本在 search_results.json。

## 3. GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture

- 状态：保留／邻近任务。遮挡与泛化有关，但摘要对象为通用目标跟踪，不能直接当作 MOT 关联证据。
- 来源：arxiv；API 年份：2026；venue：arXiv；引用数：0。
- 作者：Shih-Fang Chen, Jun-Cheng Chen, I-Hong Jhuo, Yen-Yu Lin。
- 链接：[记录 1](http://arxiv.org/abs/2602.14771v5)
- DOI：10.1109/TCSVT.2026.3675005

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

将联合嵌入预测架构用于通用目标跟踪，讨论模型适配和遮挡处理。可以作为遮挡表征学习的邻近线索，不能据摘要推断其解决检测集合关联。
## 4. A Robust Deep Networks based Multi-Object MultiCamera Tracking System for City Scale Traffic

- 状态：保留／外围 MOT。多摄像机车辆跟踪与 ReID；与固定检测的单摄像机 FM 关联较远。
- 来源：arxiv；API 年份：2025；venue：arXiv；引用数：0。
- 作者：Muhammad Imran Zaman, Usama Ijaz Bajwa, Gulshan Saleem, Rana Hammad Raza。
- 链接：[记录 1](http://arxiv.org/abs/2505.00534v1)
- DOI：10.1007/s11042-023-16243-7

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

构建城市交通多摄像机车辆检测、跟踪与重识别系统，结合检测网络、外观特征和轨迹关联，并在 AI City 数据上评价。
## 5. eMoE-Tracker: Environmental MoE-based Transformer for Robust Event-guided Object Tracking

- 状态：排除本轮核心证据。摘要核心是事件/图像模态融合和环境专家，不是多目标运动分布与分配。
- 来源：arxiv；API 年份：2024；venue：arXiv；引用数：0。
- 作者：Yucheng Chen, Lin Wang。
- 链接：[记录 1](http://arxiv.org/abs/2406.20024v3)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

结合图像与事件信息，使用环境属性专家及对比关系建模，提高不同环境条件下的目标跟踪鲁棒性。重点是模态与模板特征交互。


本组包含 2 条原始返回；全部版本在 search_results.json。

## 6. Unified Sequence-to-Sequence Learning for Single- and Multi-Modal Visual Object Tracking

- 状态：排除本轮核心证据。摘要核心为单/多模态视觉目标框序列预测，未涉及本轮身份竞争问题。
- 来源：arxiv；API 年份：2023；venue：arXiv；引用数：0。
- 作者：Xin Chen, Ben Kang, Jiawen Zhu, Dong Wang, Houwen Peng, Huchuan Lu。
- 链接：[记录 1](http://arxiv.org/abs/2304.14394v3)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

将目标框跟踪表述为序列生成，并通过统一接口和任务提示扩展到多模态输入。主要讨论目标跟踪网络表示。


本组包含 2 条原始返回；全部版本在 search_results.json。

## 7. Omni Survey for Multimodality Analysis in Visual Object Tracking

- 状态：排除本轮核心证据。摘要为多模态 VOT 综述，不能混同多目标 MOT。
- 来源：arxiv；API 年份：2025；venue：arXiv；引用数：0。
- 作者：Zhangyong Tang, Tianyang Xu, Xuefeng Zhu, Hui Li, Shaochuan Zhao, Tao Zhou, Chunyang Cheng, Xiaojun Wu, Josef Kittler。
- 链接：[记录 1](http://arxiv.org/abs/2508.13000v2)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

综述多模态视觉目标跟踪的数据、对齐、模型和评价，讨论辅助模态并非总能带来收益，以及现有数据类别分布的局限。
## 8. Sea You Later: Metadata-Guided Long-Term Re-Identification for UAV-Based Multi-Object Tracking

- 状态：保留／外围 MOT。利用 UAV 元数据做运动与长期 ReID；数据条件不同。
- 来源：arxiv；API 年份：2023；venue：arXiv；引用数：0。
- 作者：Cheng-Yen Yang, Hsiang-Wei Huang, Zhongyu Jiang, Heng-Cheng Kuo, Jie Mei, Chung-I Huang, Jenq-Neng Hwang。
- 链接：[记录 1](http://arxiv.org/abs/2311.03561v2)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

针对海事 UAV 跟踪的相似外观、相机运动和长期重识别问题，利用 GPS、高度与相机姿态等元数据连接轨迹。
## 9. Environment-aware Motion Matching

- 状态：排除本轮核心证据。摘要研究角色动画 motion matching，非检测轨迹身份关联。
- 来源：arxiv；API 年份：2025；venue：arXiv；引用数：0。
- 作者：Jose Luis Ponton, Sheldon Andrews, Carlos Andujar, Nuria Pelechano。
- 链接：[记录 1](http://arxiv.org/abs/2510.22632v1)
- DOI：10.1145/3763334

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

研究环境条件下的动画运动匹配，属于角色运动合成语境，不能仅凭 motion matching 词组纳入身份关联研究。
## 10. Topological Flow Matching

- 状态：排除本轮核心证据。摘要研究流场拓扑约束，非框运动分布；关键词重合不足。
- 来源：arxiv；API 年份：2026；venue：arXiv；引用数：0。
- 作者：Kacper Wyrwal, İsmail İlkan Ceylan, Alexander Tong。
- 链接：[记录 1](http://arxiv.org/abs/2606.15897v1)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

将拓扑性质引入生成流场的建模，研究生成过程中的结构约束。当前摘要未提供与 MOT 框或身份匹配直接对应的机制。
## 11. Alignment Matters Inside and Out in Equivariant Graph Flow Matching

- 状态：保留／方法学远邻。等变图与 FM 对齐可能对未来集合生成有借鉴；不作为直接撞题证据。
- 来源：arxiv；API 年份：2026；venue：arXiv；引用数：0。
- 作者：Moritz Piening, Christian Wald。
- 链接：[记录 1](http://arxiv.org/abs/2608.26961v2)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

讨论等变图生成中的内部及外部对齐对 flow matching 的影响。若后续研究集合或联合目标生成，可能有方法借鉴意义。
## 12. Flow Matching for Scalable Simulation-Based Inference

- 状态：保留／方法学背景。条件分布推断与密度估计；非 MOT 实证。
- 来源：arxiv；API 年份：2023；venue：arXiv；引用数：0。
- 作者：Maximilian Dax, Jonas Wildberger, Simon Buchholz, Stephen R. Green, Jakob H. Macke, Bernhard Schölkopf。
- 链接：[记录 1](http://arxiv.org/abs/2305.17161v2)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

使用 flow matching 学习用于基于模拟的推断的条件分布，关注可扩展概率推断。可作为概率建模背景，非跟踪实证。
## 13. AdaOcc: Adaptive Forward View Transformation and Flow Modeling for 3D Occupancy and Flow Prediction

- 状态：排除本轮核心证据。摘要的 flow 是占据运动流预测，不是所研究的 FM 生成框关联。
- 来源：arxiv；API 年份：2024；venue：arXiv；引用数：0。
- 作者：Dubing Chen, Wencheng Han, Jin Fang, Jianbing Shen。
- 链接：[记录 1](http://arxiv.org/abs/2407.01436v1)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

以自适应视图变换和流建模进行三维占据及运动预测。这里的运动流任务与条件生成框分布不是同一个问题。
## 14. Simulation-free Schrödinger bridges via score and flow matching

- 状态：保留／方法学背景。score/flow matching 与模拟无关训练基础；不作 MOT 性能证据。
- 来源：arxiv；API 年份：2023；venue：arXiv；引用数：0。
- 作者：Alexander Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, Yoshua Bengio。
- 链接：[记录 1](http://arxiv.org/abs/2307.03672v3)
- DOI：未获取

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

研究通过 score matching 和 flow matching 构造无需模拟训练路径的薛定谔桥方法。作为生成建模方法学背景保留。
## 15. Occlusion-Aware Motion Modeling for Robust Multi-Object Tracking: A Comparative Analysis

- 状态：保留／二级资料线索。比较分析，技术主张仍应回到原始方法；不作为本报告结论主证据。
- 来源：crossref；API 年份：2025；venue：Applied and Computational Engineering；引用数：1。
- 作者：Haoyu Zhang。
- 链接：[记录 1](https://doi.org/10.54254/2755-2721/2026.tj30603)
- DOI：10.54254/2755-2721/2026.tj30603

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

比较遮挡感知运动建模与 MOT 鲁棒性相关方法。属于二级分析线索，主报告仍以被讨论方法的原始论文为证据。
## 16. Occlusion-Robust Multi-Object Tracking with Adaptive Feature Management and Motion Compensation

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：2025；venue：2025 IEEE International Conference on Advanced Visual and Signal-Based Systems (AVSS)；引用数：0。
- 作者：Jin Hong, Yoojin Han, Junseok Kwon。
- 链接：[记录 1](https://doi.org/10.1109/avss65446.2025.11149928)
- DOI：10.1109/avss65446.2025.11149928

**摘要**

未获取。

## 17. Occlusion-Robust Multi-Object Tracking against viewpoint changing for UAV

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Yue Zhai, Ding Lu, Jianjun Li, Yong Wang。
- 链接：[记录 1](https://doi.org/10.2139/ssrn.5436573)
- DOI：10.2139/ssrn.5436573

**摘要**

未获取。

## 18. Finite Memory Observation-Centric Sort for Occlusion-Robust Multi-Object Tracking

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Vitaliy Kim, Young-Eun Kim, Seong-Whan Lee。
- 链接：[记录 1](https://doi.org/10.2139/ssrn.4838053)
- DOI：10.2139/ssrn.4838053

**摘要**

未获取。

## 19. A Robust Anti-Occlusion Multi-Object Tracking Method for Infrared Ship Tracking

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Liqiong Chen, Huanxin Bai, Yu Wu, Zhaobing Qiu, Huanxian Liu, Renhui Wei, Feng Huang。
- 链接：[记录 1](https://doi.org/10.2139/ssrn.5392546)
- DOI：10.2139/ssrn.5392546

**摘要**

未获取。

## 20. Tracking the Traffic: Multi-Vehicle Tracking Using Motion Estimation and Object Association

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：NIKESHA PATEL, Dr. Keyur  N. Brahmbhatt。
- 链接：[记录 1](https://doi.org/10.2139/ssrn.5281652)
- DOI：10.2139/ssrn.5281652

**摘要**

未获取。

本组包含 2 条原始返回；全部版本在 search_results.json。

## 21. Multi-Object Tracking with Integrated Appearance and Mamba-Based Motion Features

- 状态：保留／直接相关线索。摘要为外观与 Mamba 运动融合，需核查实现及协议。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Changhao Zhou, Xun Duan, Dafu Zu, Guangqian Kong。
- 链接：[记录 1](https://doi.org/10.21203/rs.3.rs-7318459/v1)
- DOI：10.21203/rs.3.rs-7318459/v1

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

融合外观特征与基于 Mamba 的运动特征进行多目标跟踪。可用于查找学习式运动模型和融合对照。
## 22. RM3DMOT: Roadside Monocular 3D Multi-Object Tracking with Motion-Appearance Optimization

- 状态：保留／邻近 3D MOT。摘要涉及运动与外观优化，领域和观测条件不同。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Yuyang Yao, Ping Wang, Zuxing Li, Chao Wang, Can Tian, Shuo Yang。
- 链接：[记录 1](https://doi.org/10.36227/techrxiv.176419309.95056409/v1)
- DOI：10.36227/techrxiv.176419309.95056409/v1

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

面向路侧单目三维多目标跟踪，通过运动与外观信息优化关联。观测条件和预测空间与本项目不同。
## 23. JDECMC: Improving JDE based multi-object tracking with Camera Motion Compensation

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：2024；venue：Displays；引用数：8。
- 作者：Melikamu Liyih Sinishaw, Shu Liu。
- 链接：[记录 1](https://doi.org/10.1016/j.displa.2024.102682)
- DOI：10.1016/j.displa.2024.102682

**摘要**

未获取。

## 24. Towards Accurate State Estimation: Kalman Filter Incorporating Motion Dynamics for 3D Multi-Object Tracking

- 状态：保留／邻近 3D MOT。摘要涉及 KF 动力学，可用于概率基线思路，尚未核查正文。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Mohamed Nagy, Naoufel Werghi, Bilal Hassan, Jorge Dias, Majid Khonji。
- 链接：[记录 1](https://doi.org/10.2139/ssrn.6506777)
- DOI：10.2139/ssrn.6506777

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

将运动动力学纳入三维 MOT 的卡尔曼状态估计，试图改善运动模型适应性。需要读正文才能判断与概率基线的具体关系。
## 25. MotionFormer: multi-object tracking with motion information

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：2023；venue：Third International Conference on Control and Intelligent Robotics (ICCIR 2023)；引用数：0。
- 作者：Cheng Fu, Lihui Cen。
- 链接：[记录 1](https://doi.org/10.1117/12.3011539)
- DOI：10.1117/12.3011539

**摘要**

未获取。

## 26. SDFM: Spatial-Dominated Flow Matching for Stochastic Human Motion Prediction

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：2026；venue：ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)；引用数：0。
- 作者：Yangzheng Zhou, Yubin Xiao, Pengzhong Li。
- 链接：[记录 1](https://doi.org/10.1109/icassp55912.2026.11460874)
- DOI：10.1109/icassp55912.2026.11460874

**摘要**

未获取。

## 27. Uncertainty-Aware Transformer Framework for Robust Robotic Motion Prediction

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：2025；venue：Proceedings of the 1st International Conference on Computational and Cognitive Systems Science；引用数：0。
- 作者：Geetha K., . Srisailanath, Aparna Sikarwar。
- 链接：[记录 1](https://doi.org/10.5220/0014945800005024)
- DOI：10.5220/0014945800005024

**摘要**

未获取。

## 28. Review for "Melt flow index prediction for batch polypropylene based on temporal feature matching"

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：2025；venue：未获取；引用数：0。
- 作者：未获取。
- 链接：[记录 1](https://doi.org/10.1002/cjce.70407/v1/review1)；[记录 2](https://doi.org/10.1002/cjce.70407/v1/review2)
- DOI：10.1002/cjce.70407/v1/review1；10.1002/cjce.70407/v1/review2

**摘要**

未获取。

本组包含 2 条原始返回；全部版本在 search_results.json。

## 29. FLOWR.ROOT – A flow matching-based foundation model for joint multi-purpose structure-aware 3D ligand generation and affinity prediction

- 状态：排除本轮核心证据。摘要明确是结构感知配体生成，和身份关联研究无直接关系。
- 来源：crossref；API 年份：2026；venue：Nature Communications；引用数：8。
- 作者：Julian Cremer, Tuan Le, Mohammad M. Ghahremanpour, Emilia Sługocka, Filipe Menezes, Djork-Arné Clevert。
- 链接：[记录 1](https://doi.org/10.1038/s41467-026-74130-9)
- DOI：10.1038/s41467-026-74130-9

**摘要概述（基于 API 摘要，未自动视为正文已核查）**

研究基于 flow matching 的三维配体生成及亲和力预测。对象和目标均不涉及多目标视频跟踪。
## 30. Hierarchical Prediction-Uncertainty-Aware Motion Planning for Autonomous Driving in Lane-Changing Scenarios

- 状态：保留／待核实。API 未提供摘要；仅有题名和元数据，不能据此确认相关或排除，不用于支持任何技术结论。
- 来源：crossref；API 年份：None；venue：未获取；引用数：0。
- 作者：Ruoyu Yao, Xiaotong Sun。
- 链接：[记录 1](https://doi.org/10.2139/ssrn.4764726)
- DOI：10.2139/ssrn.4764726

**摘要**

未获取。

## 来源错误原文

以下是工具记录，不代表对应来源没有论文。OpenReview 返回 0 条，未记录错误；未推断其覆盖完备性。

```text
[dblp] Error on query 'occlusion robust multi object tracking motion association': Expecting value: line 1 column 1 (char 0)
[dblp] Error on query 'multi object tracking generative motion': Expecting value: line 1 column 1 (char 0)
[dblp] Error on query 'flow matching association aware motion prediction': Expecting value: line 1 column 1 (char 0)
[open_alex] Error on query 'occlusion robust multi object tracking motion association': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=occlusion+robust+multi+object+tracking+motion+association&filter=publication_year%3A2022-2026&sort=relevance_score%3Adesc&page=1&per-page=6
[open_alex] Error on query 'multi object tracking generative motion': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=multi+object+tracking+generative+motion&filter=publication_year%3A2022-2026&sort=relevance_score%3Adesc&page=1&per-page=6
[open_alex] Error on query 'flow matching association aware motion prediction': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=flow+matching+association+aware+motion+prediction&filter=publication_year%3A2022-2026&sort=relevance_score%3Adesc&page=1&per-page=6
[semantic_scholar] Error on query 'occlusion robust multi object tracking motion association': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=occlusion+robust+multi+object+tracking+motion+association&offset=0&limit=6&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2022-2026
[semantic_scholar] Error on query 'multi object tracking generative motion': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=multi+object+tracking+generative+motion&offset=0&limit=6&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2022-2026
[semantic_scholar] Error on query 'flow matching association aware motion prediction': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=flow+matching+association+aware+motion+prediction&offset=0&limit=6&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2022-2026
```
