# 服务器与环境配置

核查日期：2026-11-10。以下为 Linux/NVIDIA 服务器候选方案；没有在你的服务器执行。先反馈盘点信息，再选择安装分支。Mac 本地只用于编辑文档、代码和轻量检查。

## 1. 硬件与系统需求

| 项目 | 起步建议 | 说明 |
|---|---|---|
| GPU | 1 张 4090；3060 可做小批量检查 | 不要求首轮使用全部卡 |
| CPU | 每个任务先分配 4–8 核 | 依数据加载与评测实测调整 |
| RAM | 单任务希望有 32 GB 可用 | 规划值，不是经过实测的最低需求 |
| 存储 | 优先本地 SSD，建议预留约 150–250 GB | 包含压缩包、解压图像、缓存、权重和输出；不是数据集实际大小声明 |
| 系统 | 优先现有可用 Ubuntu 22.04/24.04 x86_64 | 不要求为了本项目重装；其他 Linux 先反馈 |
| 管理 | Conda/Mamba 独立环境 | 已有 Docker/集群容器规范时沿用，不要求 sudo |

不能把 8×24GB 当成单任务自动拥有 192GB 显存。8 卡是否同机、是否共享、磁盘是否网络盘，会直接影响并发方案。

以下只读命令可以现在执行并反馈；路径字段按实际填写，输出保存在仓库外或 `local/`，不要上传包含其他用户进程的完整系统转储：

```bash
cat /etc/os-release
uname -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
lscpu
free -h
df -h
df -i
command -v conda
command -v mamba
command -v python3
command -v nvcc
```

命令不存在也属于有用反馈。先不要升级驱动、CUDA 或系统 Python。

## 2. 三个环境分工

| 环境 | 任务 | 当前决定 |
|---|---|---|
| `mot-core` | 新运动模型、缓存读取、训练与关联研究 | 使用下面的候选组合做验收 |
| `mot-det` | 上游 YOLOX/OC-SORT 检测导出、必要的旧依赖 | 等上游版本与服务器反馈后锁定，不能直接宣称兼容 |
| `mot-eval` | 官方 TrackEval，CPU 即可 | 独立安装、人工例子验证，再冻结版本 |

这是一种隔离依赖的安排，数据文件在环境之间共享。新模型不需要为了读取缓存安装整套 YOLOX。`mot-det` 和 `mot-core` 最终能共用同一组合也可以，但先分别验收。DiffMOT 若后续需要旧版本，另建复现环境。

OC-SORT 的安装文档仍注明测试 Python 3.8；TrackEval 的 `requirements.txt` 含较旧的 NumPy/SciPy 固定版本。不能把二者的 requirements 直接覆盖到新环境。[OC-SORT 安装说明](https://github.com/noahcao/OC_SORT/blob/master/docs/INSTALL.md)、[TrackEval 依赖](https://github.com/JonathonLuiten/TrackEval/blob/master/requirements.txt)。

## 3. `mot-core` 候选组合

采用可明确配对的历史版本，作为兼容性起点，不宣称是最新版本或最佳性能配置：Python 3.10、PyTorch 2.6.0、torchvision 0.21.0、CUDA 12.4 wheel。此配对来自 [PyTorch 官方历史版本表](https://pytorch.org/get-started/previous-versions/)。

系统 NVIDIA 驱动、wheel 自带的 CUDA runtime、系统 `nvcc` 是三件事。`nvidia-smi` 显示的 CUDA Version 不是当前 Python 环境的 runtime 版本。普通 wheel 推理/训练通常不需要另装完整 Toolkit；编译第三方 CUDA 扩展时才需要核对 `nvcc` 与构建工具。

NVIDIA 对 CUDA 12.x 给出驱动兼容范围，但小版本兼容有功能限制；不要仅凭一个驱动主版本号断言所有算子可运行。反馈驱动后选择 cu124 或经核查的其他组合，并做实际验收。[NVIDIA 兼容文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

下面命令供审核后在服务器运行；本轮未运行。假设已有 Conda，且 cu124 分支已确认适用：

```bash
conda create -n mot-core python=3.10 pip -y
conda activate mot-core
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==1.26.4 scipy==1.13.1 PyYAML==6.0.2 tqdm==4.67.1 pytest==8.3.5
python -m pip check
```

后面一行是项目拟定依赖，不是已经测试过的锁文件。先不安装 torchaudio、MMCV、Detectron2、TensorRT、FlashAttention、分布式训练框架或大型模型库。画图、表格、视频包等在实现需要时添加并冻结。

### 最小 CUDA 验收

只使用获分配的 GPU；例如当前 shell 已按调度器设置 `CUDA_VISIBLE_DEVICES`。以下小张量检查不会验证完整训练性能：

```bash
python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('runtime:', torch.version.cuda, 'visible devices:', torch.cuda.device_count())
assert torch.cuda.is_available(), 'CUDA unavailable'
device = torch.device('cuda:0')
print('gpu:', torch.cuda.get_device_name(device))
x = torch.randn(32, 16, device=device, requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device)
scores = torch.tensor([0.9, 0.8], device=device)
assert nms(boxes, scores, 0.5).numel() == 1
torch.cuda.synchronize()
print('forward/backward and torchvision CUDA NMS passed')
PY
```

验收失败时保留完整错误、版本和命令；不要在多个环境里连续降级到“碰巧能 import”为止。`torch.cuda.is_available()` 为真也不足以代替算子与反传检查。

## 4. 旧检测环境与评测环境怎样落地

`mot-det` 按以下顺序实施：选定上游提交 → 读取对应 requirements/setup → 独立环境解析依赖 → 单图推理/NMS → 短视频 → 检测缓存重放。可能遇到 NumPy 弃用别名、Cython/COCO 编译、PyTorch checkpoint 加载行为变化。只做最小兼容补丁并记录差异，不在解决依赖时顺带改算法。

`mot-eval` 候选为 Python 3.10＋NumPy 1.26.4＋SciPy 1.13.1，其他 import 依赖按锁定的 TrackEval 提交补齐。不盲目执行其历史 requirements。若遇到弃用 API，用有版本记录的兼容补丁或独立旧环境；必须通过指标人工例子与真实短序列再冻结。评测公式不自行重写。

PyTorch 新版本加载旧 checkpoint 报错时，先确认来源、格式和上游加载方式，不把任意未知对象反序列化当作通用修复。官方权重只在记录其来源后用于对应模型。

## 5. 环境验收后的归档

在仓库根目录、对应环境中执行，`local/` 已被忽略：

```bash
mkdir -p local/environment
python -m pip freeze > local/environment/mot-core-pip-freeze.txt
conda env export --from-history > local/environment/mot-core-history.yml
python -m torch.utils.collect_env > local/environment/mot-core-torch-env.txt
```

`pip freeze` 是已安装包快照，不是包含平台/驱动/下载哈希的完整可复现锁。实现阶段补充平台标签、torch wheel 源、上游 commit、必要补丁、权重与数据校验和。审查后提交不含主机敏感信息的依赖清单。

随机性控制覆盖 Python、NumPy、torch、DataLoader worker、采样器；记录 TF32、cuDNN、AMP 与确定性开关。相同 seed 不保证跨版本、跨平台完全一致，见 [PyTorch 2.6 可复现说明](https://docs.pytorch.org/docs/2.6/notes/randomness.html)。首轮数值调试用 FP32；确认后再比较 AMP，概率密度和 log-sum-exp 等关键计算保留 FP32。

## 6. 并发与测速

训练任务通过调度器或明确的 GPU 分配运行；不在同一张卡同时塞多个用于速度比较的任务。正式延迟比较固定一张同型号卡、精度和线程数，先预热，CUDA 同步后计时，分开记录模型、关联和 I/O，报告视频人数与 p50/p95。记录一次峰值显存，不能只靠 `nvidia-smi` 的瞬时截图估计。

CPU 线程和 DataLoader worker 从小值开始，避免 8 个任务各自开启几十线程。数据缓存共享只读，输出按 run_id 隔离。第一轮不需要 NCCL 多卡验收；将来确实需要 DDP 时再测试互联和通信。
