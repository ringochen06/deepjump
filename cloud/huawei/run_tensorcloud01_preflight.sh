#!/usr/bin/env bash
# Bounded, fail-closed TensorCloud01 cloud preflight. This script never starts
# calibration or formal training. It requires an explicit shutdown policy and
# powers the authorized GPU instance off on every exit path.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set EXPECTED_REPO_COMMIT to the reviewed deployed SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set EXPECTED_HOSTNAME to the authorized GPU instance hostname}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1 for the authorized bounded run}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OBS_RUN_PREFIX=${OBS_RUN_PREFIX:-deepjump-preflight/tensorcloud01}

OVERFIT_CONFIG=configs/v100_tensorcloud01_overfit.yaml
SMOKE_CONFIG=configs/v100_tensorcloud01_d1_smoke.yaml
OVERFIT_DIR="$REPO/runs/v100_tensorcloud01_overfit"
SMOKE_DIR="$REPO/runs/v100_tensorcloud01_d1_smoke"
RUN_DIR="$REPO/runs/tensorcloud01_preflight_$RUN_ID"
READBACK_DIR="$RUN_DIR/obs_readback"
OBS_DST="$BUCKET/$OBS_RUN_PREFIX/$RUN_ID"

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || {
  printf 'refusing unbounded run: SHUTDOWN_ON_EXIT must be 1\n' >&2
  exit 2
}
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  printf 'hostname mismatch: actual=%s expected=%s\n' "$(hostname)" "$EXPECTED_HOSTNAME" >&2
  exit 2
}

shutdown_on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    printf 'preflight failed; best-effort OBS evidence archive start=%s\n' "$(date -Is)"
    [[ -d "$RUN_DIR" ]] && timeout 90s obsutil sync "$RUN_DIR" "$OBS_DST/failure/audit"
    [[ -d "$OVERFIT_DIR" ]] && timeout 90s obsutil sync "$OVERFIT_DIR" "$OBS_DST/failure/overfit"
    [[ -d "$SMOKE_DIR" ]] && timeout 90s obsutil sync "$SMOKE_DIR" "$OBS_DST/failure/smoke"
    set -e
  fi
  printf 'preflight exit=%s; requesting immediate shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || printf 'ERROR: immediate shutdown command failed\n' >&2
  exit "$code"
}
trap shutdown_on_exit EXIT

# Defense in depth: the absolute timer survives a hung child process. The EXIT
# trap requests an earlier shutdown after either success or failure.
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

cd "$REPO"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/preflight.log") 2>&1

printf 'run_id=%s start=%s hard_stop_minutes=%s obs_dst=%s\n' \
  "$RUN_ID" "$(date -Is)" "$HARD_STOP_MINUTES" "$OBS_DST"

command -v obsutil >/dev/null
[[ -x "$PYTHON" ]] || { printf 'missing python: %s\n' "$PYTHON" >&2; exit 2; }
[[ -x "$TORCHRUN" ]] || { printf 'missing torchrun: %s\n' "$TORCHRUN" >&2; exit 2; }
[[ ! -e "$OVERFIT_DIR" ]] || { printf 'refusing to overwrite %s\n' "$OVERFIT_DIR" >&2; exit 2; }
[[ ! -e "$SMOKE_DIR" ]] || { printf 'refusing to overwrite %s\n' "$SMOKE_DIR" >&2; exit 2; }

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'deployed commit mismatch: actual=%s expected=%s\n' \
    "$actual_commit" "$EXPECTED_REPO_COMMIT" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty; refusing cloud preflight\n' >&2
  exit 2
}

gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$gpu_count" == 8 ]] || { printf 'GPU count %s != 8\n' "$gpu_count" >&2; exit 2; }
findmnt /data
df -hT /data
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'
"$PYTHON" -m pytest -q | tee "$RUN_DIR/pytest.log"

printf 'gate=data_audit start=%s\n' "$(date -Is)"
"$PYTHON" scripts/audit_mdcath_staging.py \
  --root "$DATA_ROOT" \
  --expected-h5 1000 \
  --expected-bytes 668131379559 \
  --expected-subset-sha256 39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734 \
  --expected-trajectories 25000 \
  --expected-strategy length-proportional \
  --expected-seed 20260715 \
  --expected-commit 3f9f7f7 \
  --samples 5 | tee "$RUN_DIR/data_audit.json"

printf 'gate=single_domain_overfit start=%s\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=2m 12m \
  "$PYTHON" scripts/train.py --config "$OVERFIT_CONFIG" --fast-dev \
  --fast-dev-max-loss-ratio 0.25 --fast-dev-max-rmsd-ratio 0.50 \
  2>&1 | tee "$RUN_DIR/overfit.log"
"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "PASS", r' \
  "$OVERFIT_DIR/fast_dev.json"

printf 'gate=eight_gpu_smoke start=%s\n' "$(date -Is)"
mkdir "$SMOKE_DIR"
timeout --signal=TERM --kill-after=2m 6m \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$SMOKE_CONFIG" \
  2>&1 | tee "$SMOKE_DIR/console.log"
grep -q 'world=8 ' "$SMOKE_DIR/console.log"
grep -q 'effective_batch=128 ' "$SMOKE_DIR/console.log"
grep -Eq 'peakGPU_by_rank\(cum\) \[([0-9]+\.[0-9]+,){7}[0-9]+\.[0-9]+\]GB' \
  "$SMOKE_DIR/console.log"
peak_csv=$(grep -Eo 'peakGPU_by_rank\(cum\) \[[^]]+\]GB' "$SMOKE_DIR/console.log" \
  | tail -n 1 | sed -E 's/.*\[([^]]+)\]GB/\1/')
awk -F, '{
  if (NF != 8) exit 1
  for (i = 1; i <= NF; i++) if ($i <= 0 || $i >= 16.0) exit 1
}' <<<"$peak_csv"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$SMOKE_DIR/ckpt_10.pt" \
  --history "$SMOKE_DIR/history.json" \
  --expected-step 10 \
  --expected-world-size 8 \
  --output "$RUN_DIR/local_checkpoint_gate.json"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$SMOKE_DIR/last.ckpt" \
  --history "$SMOKE_DIR/history.json" \
  --expected-step 10 \
  --expected-world-size 8 \
  --output "$RUN_DIR/local_last_checkpoint_gate.json"

printf 'gate=checkpoint_resume_readback start=%s\n' "$(date -Is)"
timeout --signal=TERM --kill-after=2m 5m \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$SMOKE_CONFIG" --resume "$SMOKE_DIR/last.ckpt" \
  2>&1 | tee "$RUN_DIR/resume_readback.log"
grep -q 'resumed from .*last.ckpt at step 10' "$RUN_DIR/resume_readback.log"

printf 'gate=obs_archive_and_readback start=%s\n' "$(date -Is)"
obsutil sync "$OVERFIT_DIR" "$OBS_DST/overfit"
obsutil sync "$SMOKE_DIR" "$OBS_DST/smoke"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST/smoke" "$READBACK_DIR"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$READBACK_DIR/ckpt_10.pt" \
  --history "$READBACK_DIR/history.json" \
  --expected-step 10 \
  --expected-world-size 8 \
  --output "$RUN_DIR/obs_checkpoint_gate.json"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$READBACK_DIR/last.ckpt" \
  --history "$READBACK_DIR/history.json" \
  --expected-step 10 \
  --expected-world-size 8 \
  --output "$RUN_DIR/obs_last_checkpoint_gate.json"
local_sha=$(sha256sum "$SMOKE_DIR/ckpt_10.pt" | awk '{print $1}')
readback_sha=$(sha256sum "$READBACK_DIR/ckpt_10.pt" | awk '{print $1}')
[[ "$local_sha" == "$readback_sha" ]] || {
  printf 'OBS checkpoint SHA256 mismatch: local=%s readback=%s\n' \
    "$local_sha" "$readback_sha" >&2
  exit 2
}
local_last_sha=$(sha256sum "$SMOKE_DIR/last.ckpt" | awk '{print $1}')
readback_last_sha=$(sha256sum "$READBACK_DIR/last.ckpt" | awk '{print $1}')
[[ "$local_last_sha" == "$readback_last_sha" ]] || {
  printf 'OBS last.ckpt SHA256 mismatch: local=%s readback=%s\n' \
    "$local_last_sha" "$readback_last_sha" >&2
  exit 2
}

printf '{"status":"PASS","run_id":"%s","commit":"%s","checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$actual_commit" "$local_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
printf 'TensorCloud01 preflight PASS; formal training was not started.\n'
