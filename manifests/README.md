# Manifests

这里保存可提交的轻量 provenance：

- `upstreams/`：第三方仓库 URL、固定 commit、许可证和补丁；
- `datasets/`：官方来源、revision、文件清单和校验规则，不含数据本体；
- `weights/`：权重来源、训练范围声明、SHA256 和兼容配置，不含权重；
- `splits/`：motion-train/motion-dev 的冻结序列清单；
- `detections/`：检测器、权重、阈值和缓存内容 hash。

只有真实取得并核验的信息才能写入 manifest；未知字段明确标为 `unknown`，不能根据文件名猜测。
