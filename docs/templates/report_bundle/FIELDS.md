# 模板字段说明

这些 JSON 是填写骨架，不是 JSON Schema 或运行结果。新增字段允许，公共指标列与单位不能改。删除未使用的可选模板，并在 `omissions` 登记；`case_template.json` 仅供生成 `cases.jsonl`，真实报告不把该空对象当一个案例。

manifest 的数组元素需要填下列内容，不能用空数组表示“已完成但不想回传”：

| 数组 | 每个元素的字段 |
|---|---|
| `scopes` | `scope_id, dataset_split, sequences, frame_ranges, purpose, previous_use, weight_overlap_status, evaluator_config_artifact_id` |
| `conditions` | `condition, description, config_artifact_id, baseline_condition, changed_factors, fixed_factors` |
| `planned_runs` | `run_id, condition, seed, scope_id, expected_sequences`；取消另附原因与协议版本 |
| `attempts` | `run_id, attempt_id, status, started_at, ended_at, command, cwd, exit_code, failure_reason, parent_attempt_id, supersedes, reused_artifacts` |
| `authoritative_evaluations` | `run_id, eval_id, scope_id, status, output_artifact_id` |
| `resources` | `run_id, machine_alias, gpu_model, physical_gpu_ids, CUDA_VISIBLE_DEVICES, logical_to_physical, wall_seconds, cpu_core_hours, gpu_device_hours, peak_vram_mib, measurement_method, phase` |
| `inputs.weights` | `name, source_url, sha256, training_data, overlap_audit, artifact_id` |
| `omissions` | `file_or_field, reason` |
| `protocol.deviations` | `changed_at, old_value, new_value, reason, results_seen_before_change` |

`training` 为 null 时必须在 omissions 说明未训练；否则填结构配置、参数量、可训练参数量、优化器/学习率/调度、batch、精度、种子、数据清单、训练暴露量、选模规则及每个 seed 的 checkpoint ID/步数/SHA256。环境文件保存安全的版本摘要即可，完整机器私有记录仍在 ignored local。

artifacts 的 `files` 每项为 `artifact_id, location, path, bytes, sha256, role`，location 是 `git / server / external`。Git 路径相对报告目录；清单不记录自己的哈希，避免递归。目录以文件索引文件表示。清单在所有结果文件最终写完后生成，更新任何文件须重生成对应条目。

diagnostics 的 `statistics` 每项为 `name, definition, value, unit, numerator, denominator, scope, gt_role, source`；`per_sequence_statistics` 使用同一结构并补 sequence。计数要有分母或明确为何不适用；分母为 0 时比例为 null。source 引用 artifact ID 与 locator。多窗口重叠统计明确 unique 与 exposure。

CSV 只有表头是待填写模板。真实 `COMPLETED` 评测必须包含计划内所有 condition/seed/scope/sequence 的数据；缺失/失败行必须保留并写原因。aggregate 使用 COMBINED_SEQ，不能填视频均值。曲线使用长表，每个 metric 一行，不伪造没有观测的 epoch。
