# D2 engineering attempts

No D2 result was produced by the first two starts.

1. `mot-eval` start: stopped before event computation because the environment lacks OC-SORT's `filterpy` dependency; the `.partial` directory was automatically removed.
2. `mot-det` start under protocol v1: completed the first sequence replay, then correctly stopped because v1 required byte-identical tracker text. G1 had already established the proper CPU replay contract as exact frame/ID plus XYXY tolerance `1e-4`; no result directory was retained.
3. `mot-det` under frozen replacement protocol v2: completed successfully. The counterfactual definition and thresholds were unchanged; only replay acceptance was corrected to the pre-existing G1 contract.

Protocol v1 remains immutable at `manifests/protocols/g2_d2_gt_position_first_association_20261111.yaml`. Protocol v2 records both failures and the replacement rationale.
