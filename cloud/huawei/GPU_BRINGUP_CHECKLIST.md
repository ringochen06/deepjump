# DeepJump 8×V100 Bring-up Checklist

This is the execution checklist for the length-proportional 1000-domain run. It does not
authorize starting, resizing, restarting, or deleting cloud resources, or starting formal
training. Obtain explicit user approval at each required gate.

## Frozen inputs

- Branch: `cloud-fullscale`
- Repository commit before local preparation changes: `3f9f7f715034d7124508addcca64960a061129f0`
- OBS: `obs://deepjump-mdcath-cn4-ringochen/mdcath_length_proportional_1000/`
- H5 files: `1000`
- H5 bytes: `668131379559`
- Manifest: `1000` domains / `25000` trajectories
- Subset SHA256: `39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734`
- Strategy / seed: `length-proportional` / `20260715`
- Planned EVS data disk: a new blank `2048 GiB` volume; initialization and mounting are pending.

## Before purchase and immediately after creation

- [ ] Record the live 8×V100 hourly price, billing mode, voucher balance, and budget cap.
- [ ] Record the planned maximum powered-on duration and hard stop time.
- [ ] Confirm `deepjump-v100-8gpu-20260716` is in `cn-north-4` and has 8×V100 16 GB.
- [ ] Confirm the new 2048 GiB EVS is attached and the OBS bucket remains in `cn-north-4`.
- [ ] Confirm the security group exposes TCP 22 only from the user's current public IP.
- [ ] Do not modify or delete the OBS staging prefix until GPU-side readback passes.
- [ ] Obtain explicit approval before clicking the final purchase action that creates the GPU instance.

## Powered-on time, cost, and recovery envelope

These are planning estimates, not measured GPU results:

- Infrastructure audit and blank-disk initialization: approximately 15–30 minutes if the image
  and new EVS device are healthy.
- 668 GB OBS-to-EVS sync: approximately 45–120 minutes. The prior reverse-direction upload measured
  42m29s at about 250 MB/s, but this does not guarantee GPU-instance read speed.
- Tests, strict audit, and 100-step smoke: approximately 15–30 minutes plus any environment repair.
- Initial powered-on envelope through smoke: approximately 1.25–3 hours.
- Expected cost formula: `live 8×V100 hourly price × 1.25–3 hours`. Do not substitute a stale price.

Before start, fill in the live hourly price, maximum authorized spend, and hard stop time. Stop and
report if OBS sync makes no byte progress for 10 minutes, projected sync time exceeds the authorized
window, any gate fails, or the spend cap is approached. `obsutil sync` is incremental, so a stopped
sync can be resumed. No formal-training checkpoint is needed before smoke; smoke writes atomic
checkpoints at steps 50 and 100. Calibration writes every 250 steps. Formal checkpoint cadence and
recovery commands must be approved after calibration.

## Gate A — read-only infrastructure audit immediately after start

Capture all output in the run log before changing the filesystem:

```bash
date -Is
hostnamectl
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
nvidia-smi topo -m
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,UUID
findmnt /data || true
df -hT /data || true
df -i /data || true
ip -brief address
```

Stop if the GPU count is not 8 or the new data device is ambiguous. On a new instance, `/data` and
the repository are not expected to exist yet; do not infer a device name from examples.

## Gate B — initialize and mount the new blank data disk

First use `lsblk -f` and `blkid` to prove that the 2048 GiB device has no filesystem, mountpoint, or
existing data. Stop and ask for review if any filesystem signature, partition, or unexpected device
appears. Only for a confirmed blank whole device (for example `/dev/vdb`), initialize and mount it:

```bash
sudo mkfs.ext4 /dev/vdb
sudo mkdir -p /data
sudo mount /dev/vdb /data
sudo blkid /dev/vdb
```

Record the UUID and add an `UUID=... /data ext4 defaults,nofail 0 2` entry to `/etc/fstab`, then run
`sudo mount -a`. Re-run `lsblk -f`, `findmnt`, `df -hT /data`, and `df -i /data`. Require approximately
2 TiB visible capacity before syncing 668 GB. The device path above is illustrative; never run
`mkfs` until the actual blank device has been unambiguously identified.

## Gate C — code and environment

Do not copy credentials or `.obsutilconfig`. Credential entry is performed by the user.

```bash
cd /data/deepjump
source /opt/conda/etc/profile.d/conda.sh
conda activate deepjump
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
pytest -q
bash -n cloud/huawei/*.sh
```

Clone or deploy only the reviewed commit into `/data/deepjump`; record its branch, full SHA, and
clean `git status --short`. Require 8 CUDA devices and a passing test suite.

## Gate D — OBS access, sync, and strict local audit

First run the existing OBS roundtrip check. Then sync into a new local directory so the old short
subset remains intact:

```bash
cd /data/deepjump
BUCKET=obs://deepjump-mdcath-cn4-ringochen bash cloud/huawei/obs_roundtrip_test.sh

BUCKET=obs://deepjump-mdcath-cn4-ringochen \
OBS_PREFIX=mdcath_length_proportional_1000 \
ROOT=/data/mdcath \
EXPECTED_H5=1000 \
EXPECTED_BYTES=668131379559 \
EXPECTED_SUBSET_SHA256=39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734 \
EXPECTED_TRAJECTORIES=25000 \
EXPECTED_STRATEGY=length-proportional \
EXPECTED_SEED=20260715 \
EXPECTED_COMMIT=3f9f7f7 \
AUDIT_SAMPLES=5 \
bash cloud/huawei/sync_from_obs.sh
```

The strict audit must print `"status": "PASS"`. Record sync duration and throughput. Do not modify
or delete the OBS staging prefix until this GPU-side readback passes.

## Gate E — 8-GPU 100-step smoke only

Before launch, report expected runtime and cost from the live hourly price. The smoke stop conditions
are any NCCL error/hang, world size other than 8, NaN/non-finite loss, OOM, unexpected data audit
change, missing validation, or missing/unreadable atomic checkpoint.

```bash
cd /data/deepjump
source /opt/conda/etc/profile.d/conda.sh
conda activate deepjump
mkdir -p runs/v100_ddp_smoke
CONFIG=configs/v100_ddp_smoke.yaml bash cloud/huawei/run_ddp.sh 2>&1 | tee runs/v100_ddp_smoke/console.log
```

Require and record:

- `world=8` and effective batch `128`;
- all eight GPUs active and NCCL initialized without error;
- finite loss, validation loss, RMSD, and no-op RMSD;
- peak memory below 16 GB on every GPU;
- measured steps/s, ms/step, data-wait fraction, and GPU utilization;
- `ckpt_50.pt`, `ckpt_100.pt`, and `last.ckpt` exist and load successfully;
- resume from `last.ckpt` is tested with a separate bounded configuration or explicit stop limit;
- smoke artifacts are copied to OBS and read back.

## After smoke

Do not start `configs/v100_paper_d1.yaml`. First update `STATUS.md`, `RUN_LOG.md`, and
`REPORT_NOTES.md`, calculate the calibrated 100k-step duration and cost, define the checkpoint
interval/recovery command, and obtain explicit approval before running the prepared bounded
calibration config:

```bash
CONFIG=configs/v100_ddp_calibration.yaml bash cloud/huawei/run_ddp.sh
```

Only after calibration review may formal training be proposed. Formal training always requires a
separate explicit approval.
