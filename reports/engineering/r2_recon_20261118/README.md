# R2 P0：服务器事实核查（2026-11-18）

状态：**已完成只读核查；未运行 tracker、未调用 CUDA。** 对应[远端执行草案](../../../docs/next_branch_decision_rule_plan_20261118.md) §6 P0。本文判断数据能否直接用于 P1 离线重排，不评估新方法。

## 决策

`/media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113/event_method_mechanism.csv` 有 38 列、4,986 行，即 831 个事件 × 6 种方法。它保存了注册正检测与目标轨迹的 IoU/总分、同一检测上最强的合格竞争轨迹及最终匹配结果；**没有完整候选分数矩阵、所有候选框、每条轨迹的预测分布或 KF 协方差**。`prediction_xyxy` 仅是目标轨迹的单个预测框。`live_track_count` 和 `first_round_detection_count` 只是数量，不足以恢复逐候选信息。

因此 P1 **不能从现有 CSV 直接重排 S0–S4**。须先另写并冻结 CPU 重放协议，再按固定 831 事件重建首轮完整候选集、轨迹状态、假设与分数矩阵；重放结果用独立目录保存。既有 `scripts/analyze_p1_common_state_mechanism.py:176-203,305-410` 已在内存中计算过完整 IoU/总分矩阵，但写出的行只保留目标对及最强竞争者摘要，可复用其捕获接口，不应把旧 CSV 误称为完整候选数据。

## 输入、来源与可用性

| 项目 | 核查结果 |
| --- | --- |
| 旧事件 CSV | 4,986 行 / 38 列 / 831 事件 / 6 方法；SHA256 `d08df0baec806d732fa04b463d073b0d0b9bb6ac5b3954fa1249fae77354a73e`，与旧 `summary.json` 一致 |
| 旧运行摘要 | `/media/imc/data/mot-data/runs/p1_common_state_mechanism_v1_20261113/summary.json`；SHA256 `d2e9d4138701de0105750cf0d6e836bb5164068358ac5a0cf65df9d73b4a998c` |
| 旧输入哈希表 | 同目录 `input_hashes.json`；SHA256 `5f0e4fb762aeae8bc1911f705947a3cdd3e563d41a9bf62af3d81883e82b21fc` |
| 检测缓存 | 八段 motion-dev 的 CSV 均存在于 `/media/imc/data/mot-data/cache/detections/ocsort_yolox_dance_bb2f686_b8d1/train/`，逐文件 SHA256 与旧 `input_hashes.json` 全部一致；来源见 `manifests/detections/ocsort_yolox_dance_train.yaml`：YOLOX/DanceTrack 权重 SHA256 `b8d1afba08f801f3fe2cb122faf3fc9af6c7856405a0da94cf91cdd5eb9b3321`、FP32、输入 800×1440、检测置信度 0.1、NMS IoU 0.7；跟踪高分阈值 0.6 |
| OC-SORT | `third_party/OC_SORT` 当前 HEAD `bb2f6867a41082558d95ab47a854a3918de7831b`，与 `manifests/upstreams/oc_sort.yaml` 一致 |
| P1 checkpoint | `20261112/13/14` 三个 `checkpoint.pt` 均存在于旧 `p1_deterministic_offline_v1_20261112/jobs/.../artifacts/`，逐文件 SHA256 与旧 `input_hashes.json` 的三个登记值一致；仅可作为草案所说的可选独立模式，不能因为存在就默认加入主比较 |
| 环境 | `PYTHONNOUSERSITE=1` 的现有 `mot-det`：Python 3.10.22、NumPy 1.26.4、SciPy 1.13.1、FilterPy 1.4.5、lap 0.5.12、PyYAML 6.0.2。P0 只读核查，没有启动 GPU 作业 |

完整文件哈希和逐段缓存哈希已在旧运行 `input_hashes.json`；后续重放协议仍必须重新验证实际文件。

## 38 列字典

| 列 | 含义 |
| --- | --- |
| `event_id`, `sequence`, `after_frame`, `gt_id`, `gap_bin`, `diagnostic_anchor`, `baseline_stratum`, `method_id` | 固定事件、视频/帧、GT 目标、间隔分层、诊断标记、B0 结果分层及六种方法标识 |
| `status`, `non_applicable_reason` | 该方法是否可计算及原因；B0 中 829/831 可计算，2 个目标轨迹已死亡 |
| `positive_detection_id`, `intended_track_id` | 已注册的正检测和目标 B0 轨迹 ID |
| `live_track_count`, `first_round_detection_count` | 首轮池的轨迹/检测数量；不是 ID 列表或候选框 |
| `history_observations`, `history_last_frame`, `history_target_gap`, `history_eligible`, `prediction_applied` | 目标轨迹的真实观测历史长度/最后帧/间隔、资格和是否替换预测框 |
| `prediction_xyxy`, `target_iou`, `center_error_normalized`, `log_width_abs_error`, `log_height_abs_error` | **目标轨迹单个**预测框及其相对 GT 的几何诊断；不含其他轨迹的框 |
| `intended_positive_iou`, `intended_positive_total_score`, `intended_gate_eligible` | 正检测—目标轨迹这一对的 IoU、方向融合总分和 IoU 门控资格 |
| `eligible_other_track_count`, `best_eligible_other_track_id`, `best_eligible_other_score`, `intended_minus_best_eligible_other` | 同一个正检测上其他合格轨迹的数量、最强竞争者及与目标分数之差；不含所有竞争者 |
| `positive_detection_assigned_track_id`, `intended_assigned_detection_id`, `immediate_positive_match`, `association_pairs_changed_vs_b0` | 首轮最终匹配与相对 B0 的边变化数；不含完整匹配边表 |
| `closed_loop_persistence_censored`, `closed_loop_persistence_success`, `closed_loop_effect` | 引用已有自然闭环结果的描述性结果字段，不是本次反事实重排标签 |

旧事件分层为 `recovery_failure=699`、`success_control=81`、`persistence_censored=51`，合计 831。后续冻结协议必须分别登记总事件分母、可评价事件分母和每种新规则的适用性，不能把 699+81 当成全部 831，也不能事后删除 51 个删失事件。

## 精确下一步

先冻结 `manifests/protocols/r2_*.yaml`：规定 831 事件定义与分母、重放时保存的完整候选/轨迹状态字段、S0–S4 公式和随机种子、P1/P2 数值门、预算及失败保留策略。随后在 CPU、`CUDA_VISIBLE_DEVICES=''` 下重放八段 motion-dev，仅为事件帧采集完整候选信息并做 B0 输出一致性检查；在这些输入固定后才进行 P1 重排。当前不运行 P1/P2/P3 或官方 val。
