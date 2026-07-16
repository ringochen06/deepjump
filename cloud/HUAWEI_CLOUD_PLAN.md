# Reproducing DeepJump on Huawei Cloud (near-paper scale) — Executable Plan

> **Goal**: on Huawei Cloud NVIDIA GPUs, using this repo's `cloud-fullscale` branch, train a
> near-paper-scale DeepJump (H=128, 5 temperatures × 5 replicas, ~500k-step recipe) and evaluate
> it distributionally (TICA JSD).
> **Strategy** (agreed): the data pipeline is built for the *full* 5398-domain dataset, but we
> **validate on a ~1000-domain subset first**, then decide whether to scale to full.

---

## 0. Hardware selection (generic)

> ⚠️ **Flavors, prices, regional stock, and account quotas change often — verify on the Huawei
> Cloud console before committing.** (Availability last cross-checked against the Beijing-4 console
> in **2026-07**; live pricing/quota/purchase status is tracked separately, not in this repo.)

- **NVIDIA GPUs only** for this plan (the code is CUDA + DDP). A100/A800 are frequently **not
  self-service** and may require a support ticket / whitelist (ModelArts dedicated pool or BMS).
- A practical target is **8× V100 16 GB** on the **P2v** family (`p2v.16xlarge.8`); the single-GPU
  validation flavor is `p2v.2xlarge.8`. (`P2s` is V100 32 GB but caps at 2 cards — not for 8-way DDP.)
- Before provisioning: confirm the **GPU-card quota** for your target flavor (vCPU/RAM quota alone is
  not enough), and check current price and regional stock on the console.

**Consequence for training on 16 GB cards** (drives the config choices below):
- **V100 = Volta → fp16 only (NO bf16).** Use `amp_dtype: fp16` (GradScaler; already wired).
- **16 GB/card (not 80).** The formal config `configs/v100_paper_d1.yaml` runs **crop 256, per-GPU
  batch 16, grad_accum 1** → effective batch `16 × 8 = 128` (the paper's global batch).
- The paper's **25 Å all-atom Vector-Map loss fits 16 GB at this setting** — **measured on 1× V100
  16 GB: ~9.7 GB allocated / ~10.6 GB reserved, ~156 ms/step, no NaN/OOM** (real 257-residue domain,
  batch 16). So the formal V100 config uses the **faithful all-atom loss** (`w_allatom=1`), not the
  offset loss. (`configs/v100_h128_d1.yaml` keeps a lower-memory crop-128 / offset-loss fallback for
  cards under memory pressure.)

---

## 1. What "near-paper" means here (expectation alignment)

Aligned: H=128, Nh=4, 6+6 layers, global batch 128, **crop 256**, **25 Å all-atom loss** (fits 16 GB,
measured), Adam lr 5e-3→3e-3, grad-clip 0.1, per-δ models (1/10/100), 5 temps × 5 replicas, DDP.
Compromised on V100: **fp16** (vs bf16), **bounded step count** (cost — §8).
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
| Configs | `configs/v100_smoke.yaml` (1-GPU smoke), `configs/v100_ddp_smoke.yaml` (8-GPU 100-step smoke, same footprint as formal), `configs/v100_paper_d1.yaml` (**V100 formal**: crop 256, all-atom loss), `configs/v100_h128_d1.yaml` (crop-128/offset-loss fallback), `configs/paper_h128_d{1,10,100}.yaml` (A100 template) | ready to run |
| Frozen subset | `configs/subset_1000.txt` | reproducible 1000-domain id list (306 GB) so staging + manifest match exactly |
| Cloud scripts | `cloud/{setup_env,download_data,run_ddp}.sh` + `cloud/{stage_to_obs,sync_from_obs,ckpt_to_obs}.sh` | env / data / launch + OBS staging & ckpt archival |

---

## 3. What to buy & which services to enable (in order)

1. **VPC + subnet** — defaults are fine.
2. **Security group** — inbound allow **TCP 22 (SSH)** from **your own public IP only** (not 0.0.0.0/0).
3. **Key pair** — Console → Key Pairs → create → download the `.pem` (use keys, not passwords).
4. **OBS bucket** — for raw mdCATH + checkpoint archival (cheap, durable, survives instance deletion).
   Note the **AK/SK** for `obsutil`. **Create it in the SAME region as the GPU instance** so the
   later `obsutil sync` to the instance is an intra-region (fast, near-free) transfer, not a public
   download. **Stage the data into OBS from a cheap CPU box BEFORE renting the 8-GPU instance** — see
   §6. Frozen subset sizes (measured from the HF listing, 2026-07): **1000-domain subset = 306 GB**
   (smallest domains, ≤0.7 GB each; 196–364 MB per file), **full dataset = 3.61 TB** (5398 files).
5. **ECS instance**:
   - Flavor: **validation** = `p2v.2xlarge.8` (1× V100 16 GB) or a cheaper T4 (Pi2); **formal** =
     `p2v.16xlarge.8` (8× V100 16 GB). **Pay-as-you-go.**
   - **Image**: a **GPU-accelerated public image with the Tesla driver + CUDA pre-installed**
     (Ubuntu 20.04/22.04) — avoids manual driver setup.
   - **Disks**: 100 GB system disk **plus a data disk — EVS "Extreme SSD" 500 GB–1 TB — mounted at
     `/data`** as the dataloader's random-read hot store. (V100 ECS flavors generally have no local
     NVMe, so use a fast EVS volume.) The 1000-domain subset is **306 GB**, so **500 GB EVS** is
     enough for the validation run; size up only for the full 3.61 TB dataset.
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
obsutil config -i=<AK> -k=<SK> -e=<obs-endpoint>
BUCKET=obs://<your-bucket> bash cloud/obs_roundtrip_test.sh   # 2-file up/down sanity check
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

## 6. Data — OBS-first staging (do this BEFORE renting the 8-GPU instance)

The 8× V100 instance bills by the GPU-hour, so it must **never** pull data over the public internet
from HuggingFace. Instead: stage the subset into OBS once from a **cheap CPU box**, then the GPU
instance only does an **intra-region `obsutil sync`** (fast, near-free, no HF).

```bash
# --- (A) on a CHEAP CPU box in the same region (or your laptop), NO GPU rented yet ---
# 1. freeze the exact subset ids (already committed as configs/subset_1000.txt; re-run to refresh)
python scripts/select_subset.py --n 1000 --max-gb 0.7 --out configs/subset_1000.txt   # 1000 doms, 306 GB
# 2. HF download that subset -> build manifest -> upload both to OBS (obsutil configured with AK/SK)
BUCKET=obs://<your-bucket> SUBSET=configs/subset_1000.txt bash cloud/stage_to_obs.sh
#    (full dataset instead:  BUCKET=obs://<your-bucket> MODE=full bash cloud/stage_to_obs.sh)

# --- (B) later, ON the 8-GPU instance, right after boot ---
BUCKET=obs://<your-bucket> bash cloud/sync_from_obs.sh          # OBS -> /data (intra-region, no HF)
```

Fallback (skip OBS, download straight to the instance — burns GPU-hours, not recommended for 8-GPU):
```bash
MODE=subset N=1000 ROOT=/data/mdcath bash cloud/download_data.sh
```

Once the manifest exists (staging builds it for you), training startup is instant (no per-file opens).

---

## 7. Training

```bash
# STEP A - single-GPU SMOKE (~15 domains, 60 steps): confirms no OOM and measures peak GPU
# memory + ms/step. Uses the dedicated smoke config (do NOT run the 100k formal config here).
python scripts/train_ddp.py --config configs/v100_smoke.yaml     # direct python is fine for 1 GPU

# STEP B - 8-GPU 100-step DDP SMOKE (~minutes): confirms all 8 ranks sync over NCCL, per-GPU
# memory fits 16 GB, DistributedSampler shards the real 5-temp x 5-replica data, and a mid-run
# checkpoint is written + rotated. Verify BEFORE committing to the long run. See checklist below.
CONFIG=configs/v100_ddp_smoke.yaml bash cloud/run_ddp.sh

# STEP C - 8-GPU formal run (run_ddp.sh auto-uses all visible GPUs). Archive ckpts to OBS in bg.
RUN_DIR=runs/v100_paper_d1 BUCKET=obs://<your-bucket> bash cloud/ckpt_to_obs.sh &
CONFIG=configs/v100_paper_d1.yaml bash cloud/run_ddp.sh

# resume (after preemption / restart): pull the ckpt dir back from OBS, then --resume
obsutil sync obs://<your-bucket>/ckpts/v100_paper_d1 runs/v100_paper_d1
CONFIG=configs/v100_paper_d1.yaml RESUME=runs/v100_paper_d1/last.ckpt bash cloud/run_ddp.sh
```

**STEP B (100-step DDP smoke) — what to confirm before the long run** (same crop 256 / batch 16 /
all-atom footprint as the formal run, so its peak-memory reading is the real one):
- **8-card sync**: `nvidia-smi` shows all 8 GPUs busy; rank-0 log advances steps; `torchrun` spawns 8 procs.
- **NCCL**: no `NCCL error` / hang at init; the run reaches step 10+ (first all-reduce succeeded). If it
  hangs at start, set `NCCL_DEBUG=INFO` and check the security group allows intra-node loopback.
- **Memory**: rank-0 `peakGPU(cum)` should land near the measured ~10.6 GB, safely under 16 GB. If a GPU
  OOMs, drop `train.batch_size` 16→8 (halves memory) or `data.crop_length` 256→192.
- **Checkpoint**: `runs/v100_ddp_smoke/` has `ckpt_50.pt`, `ckpt_100.pt` + `last.ckpt` after the run
  (mid-run write at step 50 works; written atomically). Optionally test resume: `RESUME=runs/v100_ddp_smoke/last.ckpt`.

- **Effective batch** = `batch_size × world_size × grad_accum`. The formal config uses
  `16 × 8 × 1 = 128` (paper's global batch); on a single card that line prints `effective_batch=16`,
  not 128. If a GPU OOMs, lower batch/crop; if memory is spare, raise batch or crop — decide from the
  smoke's measured peak.
- **Monitoring**: rank-0 prints `loss`, `lr`, `it/s`, `ms/step`, `peakGPU(cum)` (whole-run high-water
  mark) and `data %` (dataloader-wait fraction), and writes `runs/.../history.json` (honest τ=0 val +
  no-op baseline). Optional: `tensorboard --logdir runs/`.
- **Checkpoints**: every `ckpt_every` steps → `ckpt_<step>.pt` + `last.ckpt` (model+optimizer+step,
  written atomically via `.tmp` + `os.replace`), keeping the last `keep_last_k`. `cloud/ckpt_to_obs.sh`
  syncs them to OBS every 10 min so a reclaimed instance doesn't lose progress.

---

## 8. Cost & time — read before scaling up

- **Measure real throughput in STEP A first** — do not burn 8 GPUs on an estimate. `ms/step` from the
  smoke × `grad_accum` × `max_steps` gives the wall-clock; multiply by the current hourly price (check
  the console — it changes) for the cost. Both peak memory and step time are **to be measured on the
  instance**, not assumed here.
- Because per-optimizer-step cost is high (`grad_accum` micro-batches each), **run δ=1 only with a
  bounded step budget** (e.g. 50k–100k) rather than 500k × 3 δ. Scale only after the subset run passes.
- **Stop/delete the GPU instance when not training** (keep the EVS data disk + OBS) — GPU-hours are
  by far the biggest cost.
- **Optional future optimization (NOT required) — chunked/neighbor-list 25 Å all-atom loss.** The
  current all-atom loss builds the full `[B,M,M,3]` then masks (M=crop×14); it already fits 16 GB at
  crop 256 / batch 16 (~10.6 GB measured). A chunked version computing only pairs within 25 Å would
  free memory headroom for a larger crop/batch and could speed the step up — a nice-to-have, not a
  blocker for the V100 run.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| All-atom loss memory at 16 GB | fits at crop 256 / batch 16 (~10.6 GB measured); if a card OOMs, drop batch 16→8 or crop 256→192, or use the `v100_h128_d1.yaml` offset-loss fallback |
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
# --- BEFORE renting the 8 GPUs: stage 1000-domain subset (306 GB) into OBS from a cheap box ---
python scripts/select_subset.py --n 1000 --max-gb 0.7 --out configs/subset_1000.txt
BUCKET=obs://<your-bucket> bash cloud/obs_roundtrip_test.sh              # verify OBS access first
BUCKET=obs://<your-bucket> SUBSET=configs/subset_1000.txt bash cloud/stage_to_obs.sh
# --- then rent 8x V100, on the instance: ---
TORCH_CUDA=cu118 bash cloud/setup_env.sh && conda activate deepjump
BUCKET=obs://<your-bucket> bash cloud/sync_from_obs.sh          # OBS -> /data (no HF, no GPU waste)
# 1-GPU smoke (peak mem + ms/step), then 8-GPU 100-step DDP smoke (sync/NCCL/mem/ckpt)
python scripts/train_ddp.py --config configs/v100_smoke.yaml
CONFIG=configs/v100_ddp_smoke.yaml bash cloud/run_ddp.sh
# then the bounded formal run (8x V100, delta=1) with ckpt archival to OBS
RUN_DIR=runs/v100_paper_d1 BUCKET=obs://<your-bucket> bash cloud/ckpt_to_obs.sh &
CONFIG=configs/v100_paper_d1.yaml bash cloud/run_ddp.sh
# resume / evaluate
CONFIG=configs/v100_paper_d1.yaml RESUME=runs/v100_paper_d1/last.ckpt bash cloud/run_ddp.sh
python scripts/tica_eval.py --ckpt runs/v100_paper_d1/last.ckpt --gen conditional --K 8
```

Knobs to align/scale: `data.{root,manifest,temperatures,replicas,delta_frames,val_fraction}`,
`train.{batch_size,grad_accum,max_steps,amp_dtype,num_workers}`, `model.hidden`.
The `configs/paper_h128_d*.yaml` files keep the A100 template (crop 256, bf16, all-atom loss) for
when 80 GB GPUs become available.
