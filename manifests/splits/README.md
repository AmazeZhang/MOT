# Split manifests

motion-train/motion-dev 必须按视频或同源视频组划分。清单冻结后记录生成脚本版本、seed 和内容 hash。

- `dancetrack_motion_train_dev_v1_20261112.yaml`：32/8 视频级冻结划分；四个已检查 G2 视频显式作为 dev anchors，其余四个 dev 视频由 seed 20261110 的 SHA256 排序规则确定。
