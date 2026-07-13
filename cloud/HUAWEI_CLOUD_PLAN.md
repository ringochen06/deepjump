# Reproducing DeepJump on Huawei Cloud (near-paper scale) — Executable Plan

> **Goal**: on Huawei Cloud NVIDIA GPUs, using this repo's `cloud-fullscale` branch, train a
> near-paper-scale DeepJump (H=128, 5 temperatures × 5 replicas, ~500k-step recipe) and evaluate
> it distributionally (TICA JSD).
> **Strategy** (agreed): the data pipeline is built for the *full* 5398-domain dataset, but we
> **validate on a ~1000-domain subset first**, then decide whether to scale to full.

---

## 0. Reality check: what hardware is actually available (2026)

Verified against the current Huawei Cloud purchase pages:
- **No A100 / A800** in Beijing-4, Ulanqab-1, Guiyang-1, Shanghai-1, or Guangzhou via self-service.
  Getting A100 requires a **support ticket / whitelist** (ModelArts dedicated pool or BMS).
- **Beijing-4** offers T4 / P4 / V100; the largest is **8× V100 16 GB**.
  - Correct flavor for 8× V100 16 GB = **P2v `p2v.16xlarge.8`** (64 vCPU, 512 GiB RAM),
    ~**¥131.50/hour ≈ ¥63,118/month** (compute only, excl. storage/network).
  - `p2v.2xlarge.8` = **1× V100 16 GB** (validation).
  - `P2s` is V100 **32 GB** but max **2 cards** — not usable for 8-way DDP.
- Ulanqab-1: max 2× V100 32 GB, pay-as-you-go currently unavailable.
- Account quota: 200 ECS / 800 vCPU / ~1.6 TiB RAM (all unused) — plenty for `p2v.16xlarge.8`
  (64 vCPU). **GPU-card quota is not shown; file a ticket to confirm 8× V100 can be provisioned
  before committing.**

**Consequence for training** (this drives all the config choices below):
- **V100 = Volta → fp16 only (NO bf16).** Use `amp_dtype: fp16` (GradScaler; already wired).
- **16 GB/card (not 80).** So **crop 128** (not 256), **per-GPU batch 1**, high grad-accum.
- The paper's **25 Å all-atom Vector-Map loss materialises an O(M²) `[B,M,M,3]` tensor** and **OOMs
  at 16 GB**. The V100 config therefore uses the cheaper **heavy-atom offset loss** by default. To
  train the *faithful* all-atom loss on 16 GB you need a **chunked / neighbor-list implementation**
  (§8, item 1) — not yet in the repo.

---

## 1. What "near-paper" means here (expectation alignment)

Aligned: H=128, Nh=4, 6+6 layers, global batch 128, Adam lr 5e-3→3e-3, grad-clip 0.1, long
schedule, per-δ models (1/10/100), 5 temps × 5 replicas, DDP.
Compromised on V100: **crop 128** (vs 256), **offset loss** (vs 25 Å all-atom, until chunked),
**fp16** (vs bf16), **bounded step count** (cost — §8).
Still out of scope: fast-folder headline numbers (JSD/ΔG/MFPT/ab-initio) — no DESRES data; we use
**mdCATH TICA distributional JSD** as the reachable distributional metric.

---

## 2. Code already in place (this branch)

| Component | File | Purpose |
|---|---|---|
| Scale-safe data pipeline | `src/deepjump/data/mdcath.py` | no file opens at init; manifest for frame counts; per-worker LRU lazy handles; compact per-trajectory index (~MB RAM); fork/spawn-safe |
| Manifest builder | `scripts/build_manifest.py` | scan once → `manifest.json`; instant training startup |
| DDP trainer | `scripts/train_ddp.py` | torchrun/NCCL, DistributedSampler, AMP (fp16/bf16), grad-accum to target batch, warmup + linear LR decay, rank-0 val/log/checkpoint, `--resume` |
| Shared loss/schedule | `src/deepjump/training.py` | pairwise + offset + 25 Å all-atom losses; `lr_at` |
| Configs | `configs/v100_h128_d1.yaml` (V100), `configs/paper_h128_d{1,10,100}.yaml` (A100 template) | ready to run |
| Cloud scripts | `cloud/{setup_env,download_data,run_ddp}.sh` | env / data / launch |

---

## 3. What to buy & which services to enable (in order)

1. **VPC + subnet** — defaults are fine.
2. **Security group** — inbound allow **TCP 22 (SSH)** from **your own public IP only** (not 0.0.0.0/0).
3. **Key pair** — Console → Key Pairs → create → download the `.pem` (use keys, not passwords).
4. **OBS bucket** — for raw mdCATH + checkpoint archival (cheap, durable, survives instance deletion).
   Note the **AK/SK** for `obsutil`.
5. **ECS instance**:
   - Flavor: **validation** = `p2v.2xlarge.8` (1× V100 16 GB) or a cheaper T4 (Pi2); **formal** =
     `p2v.16xlarge.8` (8× V100 16 GB). **Pay-as-you-go.**
   - **Image**: a **GPU-accelerated public image with the Tesla driver + CUDA pre-installed**
     (Ubuntu 20.04/22.04) — avoids manual driver setup.
   - **Disks**: 100 GB system disk **plus a data disk — EVS "Extreme SSD" 500 GB–1 TB — mounted at
     `/data`** as the dataloader's random-read hot store. (V100 ECS flavors generally have no local
     NVMe, so use a fast EVS volume.)
   - Attach an **EIP** (for data download + SSH), select the **key pair** and **security group**.
6. *(Only if you insist on A100)* ModelArts / support ticket for a whitelisted pool — not
   self-service; don't block on it.

---

## 4. Connecting & preparing the instance

```bash
chmod 600 key.pem
ssh -i key.pem root@<EIP>            # or ubuntu@<EIP>, depending on the image
nvidia-smi                            # confirm N x V100 are visible
# mount the data disk at /data
lsblk                                 # find the data disk (e.g. /dev/vdb)
mkfs.ext4 /dev/vdb && mkdir -p /data && mount /dev/vdb /data
# OBS access (install obsutil, configure AK/SK) — for data sync + checkpoint upload
```

---

## 5. Environment (once per instance)

```bash
git clone https://github.com/ringochen06/deepjump.git && cd deepjump
git checkout cloud-fullscale
# choose the CUDA wheel matching nvidia-smi's driver; V100 -> cu118 or cu121
TORCH_CUDA=cu118 bash cloud/setup_env.sh && conda activate deepjump
```

---

## 6. Data

```bash
# validation: 1000 smallest domains to the local EVS disk, then build the manifest
MODE=subset N=1000 ROOT=/data/mdcath bash cloud/download_data.sh

# full (only after deciding to scale): ~2-3 TB; prefer OBS then sync to local
MODE=full ROOT=/data/mdcath bash cloud/download_data.sh
# or:  obsutil sync obs://<bucket>/mdcath /data/mdcath && \
#      python scripts/build_manifest.py --root /data/mdcath --out /data/mdcath/manifest.json
```

Once the manifest exists, training startup is instant (no per-file opens).

---

## 7. Training

```bash
# STEP A - single-GPU smoke: temporarily set max_steps=2000 in the config, confirm it does not OOM
# and measure it/s.
CONFIG=configs/v100_h128_d1.yaml bash cloud/run_ddp.sh

# STEP B - 8-GPU formal run (run_ddp.sh auto-uses all visible GPUs)
CONFIG=configs/v100_h128_d1.yaml bash cloud/run_ddp.sh

# resume (after preemption / restart)
CONFIG=configs/v100_h128_d1.yaml RESUME=runs/v100_h128_d1/last.ckpt bash cloud/run_ddp.sh
```

- **Effective batch** = `batch_size × world_size × grad_accum`. The V100 config uses
  `1 × 8 × 16 = 128` (paper's global batch). If a GPU OOMs, keep batch 1 and raise grad_accum;
  if memory is spare, try batch 2 / grad_accum 8, or crop 160.
- **Monitoring**: rank-0 prints `it/s`, `loss`, `lr` and writes `runs/.../history.json` (honest τ=0
  val + no-op baseline). Optional: `tensorboard --logdir runs/`.
- **Checkpoints**: every `ckpt_every` steps → `ckpt_<step>.pt` + `last.ckpt` (model+optimizer+step),
  keeping the last `keep_last_k`. **Periodically upload `last.ckpt` to OBS** so a reclaimed instance
  doesn't lose progress.

---

## 8. Cost & time — read before scaling up

- **Measure real throughput in STEP A first** — do not burn 8 GPUs on an estimate. Rough guide:
  crop 128 / H=128 / fp16 ≈ 1–2 it/s per card; with effective batch 128 and grad_accum 16, one
  optimizer step processes 16 micro-batches/card → **~8–16 s/step**, so 100k steps can take **days**.
- `p2v.16xlarge.8` ≈ **¥131.5/hour ≈ ¥3,156/day**. Therefore: **run δ=1 only, with a bounded step
  budget (e.g. 50k–100k)** — do **not** launch 500k × 3 δ (tens of thousands of RMB).
- **Stop/delete the GPU instance when not training** (keep the EVS data disk + OBS) — GPU-hours are
  by far the biggest cost.
- **Recommended optimization before any serious run — chunked/neighbor-list 25 Å all-atom loss.**
  The current all-atom loss builds the full `[B,M,M,3]` then masks (M=crop×14). A chunked version
  that only computes pairs within 25 Å would **fit the faithful all-atom loss into 16 GB** and speed
  things up. This is the single most valuable code change for a faithful V100 reproduction.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| All-atom loss OOMs at 16 GB | V100 config uses offset loss; for faithful all-atom, implement chunked loss (§8) |
| fp16 numerics (equivariant norms / LayerNorm → NaN) | GradScaler is wired; if NaN persists, keep sensitive modules in fp32 autocast, or lower the initial scale |
| Instance reclaimed / preempted | upload `last.ckpt` to OBS regularly; `--resume` continues seamlessly |
| Dataloader is the bottleneck (per-frame random reads) | data on local EVS SSD (not OBS); `num_workers: 8`; tune `max_open_files` vs ulimit |
| δ=100 trajectories too short | dataset auto-skips trajectories with `num_frames ≤ 100` (compact index handles it) |
| GPU quota / availability | file a support ticket to confirm 8× V100 before committing |
| Architecture gap (no e3nn l=2) | a deliberate lite simplification; faithful l=2 would be a separate track |
| Runaway cost | strictly phase per §8; scale to full only after the subset run passes |

---

## 10. One-page cheat sheet

```bash
# env
TORCH_CUDA=cu118 bash cloud/setup_env.sh && conda activate deepjump
# data (validation subset)
MODE=subset N=1000 ROOT=/data/mdcath bash cloud/download_data.sh
# train (8x V100, delta=1)
CONFIG=configs/v100_h128_d1.yaml bash cloud/run_ddp.sh
# resume / evaluate
CONFIG=configs/v100_h128_d1.yaml RESUME=runs/v100_h128_d1/last.ckpt bash cloud/run_ddp.sh
python scripts/tica_eval.py --ckpt runs/v100_h128_d1/last.ckpt --gen conditional --K 8
```

Knobs to align/scale: `data.{root,manifest,temperatures,replicas,delta_frames,val_fraction}`,
`train.{batch_size,grad_accum,max_steps,amp_dtype,num_workers}`, `model.hidden`.
The `configs/paper_h128_d*.yaml` files keep the A100 template (crop 256, bf16, all-atom loss) for
when 80 GB GPUs become available.
