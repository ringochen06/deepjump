# 华为云复现 DeepJump(接近论文全数据)可执行方案

> 目标:在华为云 **NVIDIA GPU** 上,用本仓库 `huawei-cloud-fullscale` 分支,复现接近论文全数据版本的
> DeepJump(H=128,δ∈{1,10,100},5 温度 × 5 replica,25 Å 全原子 Vector-Map loss,~500k 步)。
> 策略(已与你确认):**管线按全量设计,先用 ~1000 domain 子集跑通验证,再决定是否拉满全量。**

---

## 0. "接近论文"的准确含义(先对齐预期)

我们对齐的是**规模 + 训练配方 + 评估口径**,不是逐比特架构:
- ✅ 对齐:H=128、Nh=4、6+6 层、crop 256、全局 batch 128、Adam lr 5e-3→3e-3、grad-clip 0.1、
  ~500k 步、δ 分模型(1/10/100)、5 温度 × 5 replica、25 Å 全原子 Vector-Map loss、分布式多卡。
- ⚠️ 仍有差异(诚实声明):等变层是手写 GVP 风格(非 e3nn 球谐 l=2)、纯 ODE x₁ 预测。这些是
  DeepJump-lite 的既有简化;是否要补 e3nn/l=2 是另一档工作(见 §9)。
- ❌ 无法复现:fast-folder 主表格(stationary JSD / ΔG / MFPT / ab initio folding),因为无 DESRES 数据。
  我们用 **mdCATH 上的 TICA 分布式 JSD** 作为可得的分布式评估。

---

## 1. 代码侧已就绪(本分支做的改动)

| 改动 | 文件 | 作用 |
|---|---|---|
| 可扩展数据管线 | `src/deepjump/data/mdcath.py` | 不在 init 打开 5398 文件;manifest 供帧数;每 worker LRU 惰性句柄;紧凑轨迹索引(内存 ~MB);pickle/fork 安全(num_workers>0) |
| manifest 构建 | `scripts/build_manifest.py` | 扫描一次 → `manifest.json`,训练秒级启动 |
| DDP 训练器 | `scripts/train_ddp.py` | torchrun/NCCL、DistributedSampler、AMP(bf16/fp16)、梯度累积到有效 batch、warmup+线性衰减、rank0 val/日志/断点、`--resume` |
| 共享 loss/调度 | `src/deepjump/training.py` | pairwise + 25 Å 全原子 loss;`lr_at` warmup+decay |
| 论文档配置 | `configs/paper_h128_d{1,10,100}.yaml` | H=128、全温度/replica、crop 256、全原子 loss、500k 步 |
| 云脚本 | `cloud/{setup_env,download_data,run_ddp}.sh` | 环境 / 数据 / 启动 |

---

## 2. 实例与存储选型

**计算(二选一):**
- **A. ECS 裸机 GPU 实例(推荐,DDP 控制最直接)**:1 台 **8×NVIDIA A100/A800 80GB**(或 8×V100 32GB 降配),
  NVLink 优先。单节点 8 卡即可 `torchrun --nproc_per_node=8`。
- **B. ModelArts 训练作业**:用专属资源池(GPU),把本仓库打成训练镜像,启动命令即 `run_ddp.sh`;
  好处是与 OBS 原生集成、可排队;代价是镜像/作业配置一次性成本。

> A800 80GB / A100 80GB 是首选:crop 256 + 25 Å 全原子 loss 显存吃紧,80GB 能放下 per-GPU batch 2-4。
> V100 32GB 可行但要把 per-GPU batch 降到 1、grad_accum 提到 16,并用 `amp_dtype: fp16`。

**存储(三层):**
1. **OBS 桶**:放原始 mdCATH(~2-3 TB)+ checkpoint 归档。便宜、持久、跨实例。
2. **实例本地 NVMe SSD**:训练时数据的**随机读**热盘。把当前要训的子集从 OBS 同步到 `/data/mdcath`。
   (dataloader 是逐帧随机读 h5,必须本地 SSD,不能直接读 OBS。)
3. (多节点才需)**SFS Turbo**:多机共享 `/data`,省去每节点各同步一份。

---

## 3. 一次性环境搭建

```bash
git clone https://github.com/ringochen06/deepjump.git && cd deepjump
git checkout huawei-cloud-fullscale
# CUDA 版本按 nvidia-smi 调整 TORCH_CUDA(cu121 覆盖多数 A100/A800)
TORCH_CUDA=cu121 bash cloud/setup_env.sh
conda activate deepjump
```

---

## 4. 数据准备

```bash
# 验证阶段:先下 1000 个最小 domain(~几百 GB)到本地 NVMe,并建 manifest
MODE=subset N=1000 ROOT=/data/mdcath bash cloud/download_data.sh

# 全量阶段(确认要拉满后):~2-3 TB,建议先落 OBS 再 obsutil sync 到本地
MODE=full ROOT=/data/mdcath bash cloud/download_data.sh
# 或:obsutil sync obs://<your-bucket>/mdcath /data/mdcath && \
#     python scripts/build_manifest.py --root /data/mdcath --out /data/mdcath/manifest.json
```

manifest 建好后训练启动是秒级的(不再逐个开文件)。

---

## 5. 训练

```bash
# 验证跑(1000 domain,δ=1);先把 max_steps 改小(如 50000)确认曲线与吞吐
CONFIG=configs/paper_h128_d1.yaml bash cloud/run_ddp.sh

# 断点续训
CONFIG=configs/paper_h128_d1.yaml RESUME=runs/paper_h128_d1/last.ckpt bash cloud/run_ddp.sh

# 三个 δ 各一个模型(论文是分 δ 训练):依次或分机跑 d1 / d10 / d100
```

**有效 batch**:`batch_size × world_size × grad_accum`。配置里 `2 × 8 × 8 = 128`(对齐论文)。
换 GPU 数就调 `grad_accum` 保持 128。显存不够先降 `batch_size` 再升 `grad_accum`。

**监控**:rank0 打印 `it/s`、`loss`、`lr`,并写 `runs/.../history.json`(含 τ=0 诚实 val 与 no-op 基线)。
可选:`tensorboard --logdir runs/`(脚本已装 tensorboard,按需接入)。

**断点**:每 `ckpt_every`(默认 10000)步存 `ckpt_<step>.pt` + `last.ckpt`(含 optimizer/step),
保留最近 `keep_last_k` 个。**务必定期把 `last.ckpt` 传回 OBS**(实例回收不丢)。

---

## 6. 从子集到全量的切换

代码/配置无需改结构,只需:
1. `MODE=full` 重新下载 + 重建 manifest;
2. 配置里 `data.root`/`data.manifest` 指向全量;`val_fraction: 0.02`(全量下约 100 个 held-out domain);
3. 视吞吐把 `max_steps` 拉到 500000;
4. 多节点则设 `NNODES/NODE_RANK/MASTER_ADDR` 跑 `run_ddp.sh`。

---

## 7. 评估

```bash
# 单步 τ-sweep(诊断 field accuracy)
python scripts/diagnose_tau.py --ckpt runs/paper_h128_d1/last.ckpt
# 分布式:DeepJump 原生随机 ODE 集合采样的 TICA JSD(核心指标)
python scripts/tica_eval.py --ckpt runs/paper_h128_d1/last.ckpt --gen conditional --K 8
```
关注 **conditional-ensemble TICA JSD 是否随规模持续逼近/跨过 no-dynamics 地板**——这是"规模是否
真的把分布关上"的判据(小规模已给出 0.56→0.35 的单调趋势)。

---

## 8. 成本与时间(务必先看,别直接拉满)

- **吞吐(粗估,8×A100 80GB, H=128, crop 256, 全原子 loss, 有效 batch 128)**:约 **0.3–1 optimizer-step/s**
  (全原子 O(M²) loss 是瓶颈,M=crop×14≈3584)。→ **500k 步 ≈ 6–20 天/单个 δ 模型**。三个 δ 更久。
- **建议路径(省钱省时,且科学上足够)**:
  1. 先 **1000 domain + δ=1 + 50–100k 步**(~1–3 天):确认吞吐、曲线、以及 TICA JSD 是否较 lite(0.35)进一步下降。
  2. 看结果再决定是否**全量 + 500k**。多数情况下,子集 + 更长训练已能证明"规模关上分布"这一结论。
- **降本优化(可选,建议全量前做)**:当前 25 Å 全原子 loss 会**先物化 [B,M,M,3] 再 mask**,显存/算力浪费大。
  改成**邻居列表/分块(neighbor-list, chunked)只算 25 Å 内的对**,可大幅提速降显存——这是全量前最值得做的一处工程优化(我可以帮你实现)。

---

## 9. 风险与应对

| 风险 | 应对 |
|---|---|
| 全原子 loss @ crop256 显存/算力爆 | 先降 batch 升 accum;根治用 neighbor-list 分块 loss(§8) |
| 实例被回收 / 抢占 | `last.ckpt` 定期传 OBS;`--resume` 无缝续训 |
| dataloader 成瓶颈(逐帧随机读) | 数据放本地 NVMe(非 OBS);`num_workers: 12`;`max_open_files` 按 ulimit 调 |
| bf16 数值(等变层范数/LayerNorm) | 已默认 bf16(A100 稳);异常则 `amp: false` 或 `amp_dtype: fp16`+scaler |
| δ=100 部分轨迹帧数不足 | 数据集自动跳过 `num_frames ≤ 100` 的轨迹(compact index 已处理) |
| 架构差异(无 e3nn l=2) | 属既定简化;若要更保真,单开一档接 e3nn(工作量大,非本方案范围) |
| 成本失控 | 严格按 §8 分阶段;子集验证达标再拉满 |

---

## 10. 一页速查

```bash
# 环境
TORCH_CUDA=cu121 bash cloud/setup_env.sh && conda activate deepjump
# 数据(验证)
MODE=subset N=1000 ROOT=/data/mdcath bash cloud/download_data.sh
# 训练(8 卡, δ=1)
CONFIG=configs/paper_h128_d1.yaml bash cloud/run_ddp.sh
# 续训 / 评估
CONFIG=configs/paper_h128_d1.yaml RESUME=runs/paper_h128_d1/last.ckpt bash cloud/run_ddp.sh
python scripts/tica_eval.py --ckpt runs/paper_h128_d1/last.ckpt --gen conditional --K 8
```

配置改这些即可对齐/缩放:`data.{root,manifest,temperatures,replicas,delta_frames,val_fraction}`、
`train.{batch_size,grad_accum,max_steps,amp_dtype,num_workers}`、`model.hidden`。
