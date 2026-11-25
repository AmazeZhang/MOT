# Result-to-claim

- `claim_supported: partial`
- `confidence: high`
- `review_independence: same-family`
- `acceptance_status: provisional`
- `routing_action: preserve S06 STOP; no new run authorized`

支持的窄结论：固定 seed13 样本上，后期 checkpoint 相对 action-agnostic 常数降低 g/h 回归误差；Context epoch30 的 internal-dev6 MSE 低于同 epoch Local。

不支持：可迁移动作排序、可靠非 M 恢复、component-specific utility learning、Context 机制优越、跟踪性能或泛化。Context epoch30 无额外风险在 train18 为 80/104，但 internal-dev6 为 13/22，低于 always-M 的 14/22；risk 等于 always-M。结果后 action-type 常数还能取得更低 MSE，进一步限制回归误差解释。

案例 attempt2 只补充 G-vs-applied 的描述性证据；同状态 M 配对、精确合法 mask 和因果身份归因仍缺失。
