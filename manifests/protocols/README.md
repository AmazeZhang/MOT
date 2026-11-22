# Frozen experiment protocols

此目录保存查看验证集或测试集结果之前冻结的配置和决策规则。协议一旦开始运行不得原地修改；工程失败需要重跑时，参数保持不变并追加失败记录。

本轮最终 G2 诊断使用两个彼此独立的冻结协议：

- `g2_d4_direction_cost_ablation_20261112.yaml`：首轮速度方向代价消融。
- `g2_d5_joint_d2_d3_20261112.yaml`：真正同时注入 D2+D3 的特权坐标反事实；不把独立 D2/D3 结果并集冒充联合结果。

S01/S02 当前链条：

- `s01_visual_association_handoff_20261120.yaml`：固定深度外观 G/Z/L/M 工程与 train-fit 探索协议。
- `s02_natural_identity_mechanism_20261121.yaml`：S01 冻结轨迹上的 GT 辅助自然错误普查初版；保留完整原始运行。
- `s02_natural_identity_mechanism_retry2_20261121.yaml`：补依赖源码锁的第一次重评，环境版本校验在输出前停止，保留失败日志。
- `s02_natural_identity_mechanism_retry3_20261121.yaml`：补齐依赖源码和实际 Python/SciPy 等版本的权威重评，定义与初版相同。
- `s03_memory_clone_smoke_20261121.yaml`：单视频 M 全序列回放与事件状态克隆工程门。
- `s03_single_memory_write_counterfactual_20261121.yaml`：S03 首版；保留原始运行和对照排除口径偏差记录。
- `s03_single_memory_write_counterfactual_retry2_20261122.yaml`：对照明确排除同视频全部 278 个 M 首次事件的 ±30 帧，独立重跑。
- `s03_instrumentation_addendum_20261122.yaml`：对首版分支的记忆向量变化与未来 trace 做只读事后补检，不改变主结果。
