#!/usr/bin/env bash
# Independent bounded probes for TensorCloud01 FP16/LR stability. Completing
# this matrix never authorizes calibration or formal training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set EXPECTED_REPO_COMMIT to the reviewed deployed SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set EXPECTED_HOSTNAME to the authorized GPU instance hostname}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1 for the authorized bounded run}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-35}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OBS_RUN_PREFIX=${OBS_RUN_PREFIX:-deepjump-preflight/tensorcloud01-numerics}

RUN_DIR="$REPO/runs/tensorcloud01_numerics_$RUN_ID"
READBACK_ROOT="/tmp/tensorcloud01_readback_$RUN_ID"
OBS_DST="$BUCKET/$OBS_RUN_PREFIX/$RUN_ID"
STATUS_FILE="$RUN_DIR/status.tsv"

PROBE_LABELS=(fp32_lr5e3 fp16_warmup20 fp16_lr5e4)
PROBE_CONFIGS=(
  configs/v100_tensorcloud01_fp32_lr5e3_probe.yaml
  configs/v100_tensorcloud01_fp16_warmup20_probe.yaml
  configs/v100_tensorcloud01_fp16_lr5e4_probe.yaml
)
PROBE_DIRS=(
  "$REPO/runs/v100_tensorcloud01_fp32_lr5e3_probe"
  "$REPO/runs/v100_tensorcloud01_fp16_warmup20_probe"
  "$REPO/runs/v100_tensorcloud01_fp16_lr5e4_probe"
)
PROBE_STEPS=(3 30 30)

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || {
  printf 'refusing unbounded run: SHUTDOWN_ON_EXIT must be 1\n' >&2
  exit 2
}
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  printf 'hostname mismatch: actual=%s expected=%s\n' "$(hostname)" "$EXPECTED_HOSTNAME" >&2
  exit 2
}

archive_evidence() {
  command -v obsutil >/dev/null || return 0
  set +e
  [[ -d "$RUN_DIR" ]] && timeout 90s obsutil sync "$RUN_DIR" "$OBS_DST/audit"
  for i in "${!PROBE_LABELS[@]}"; do
    [[ -d "${PROBE_DIRS[$i]}" ]] \
      && timeout 90s obsutil sync "${PROBE_DIRS[$i]}" "$OBS_DST/${PROBE_LABELS[$i]}"
  done
  set -e
}

shutdown_on_exit() {
  code=$?
  trap - EXIT
  archive_evidence
  printf 'numerics exit=%s; requesting immediate shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || printf 'ERROR: immediate shutdown command failed\n' >&2
  exit "$code"
}
trap shutdown_on_exit EXIT

sudo -n shutdown -h "+$HARD_STOP_MINUTES"

cd "$REPO"
mkdir -p "$RUN_DIR"
: >"$STATUS_FILE"
exec > >(tee -a "$RUN_DIR/numerics.log") 2>&1

printf 'run_id=%s start=%s hard_stop_minutes=%s obs_dst=%s\n' \
  "$RUN_ID" "$(date -Is)" "$HARD_STOP_MINUTES" "$OBS_DST"

command -v obsutil >/dev/null
[[ -x "$PYTHON" ]] || { printf 'missing python: %s\n' "$PYTHON" >&2; exit 2; }
[[ -x "$TORCHRUN" ]] || { printf 'missing torchrun: %s\n' "$TORCHRUN" >&2; exit 2; }
actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'deployed commit mismatch: actual=%s expected=%s\n' \
    "$actual_commit" "$EXPECTED_REPO_COMMIT" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty; refusing numerical probes\n' >&2
  exit 2
}
gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$gpu_count" == 8 ]] || { printf 'GPU count %s != 8\n' "$gpu_count" >&2; exit 2; }
for dir in "${PROBE_DIRS[@]}"; do
  [[ ! -e "$dir" ]] || { printf 'refusing to overwrite %s\n' "$dir" >&2; exit 2; }
done

"$PYTHON" scripts/audit_mdcath_staging.py \
  --root /data/mdcath \
  --expected-h5 1000 \
  --expected-bytes 668131379559 \
  --expected-subset-sha256 39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734 \
  --expected-trajectories 25000 \
  --expected-strategy length-proportional \
  --expected-seed 20260715 \
  --expected-commit 3f9f7f7 \
  --samples 3 | tee "$RUN_DIR/data_audit.json"

run_probe() {
  label=$1
  config=$2
  out_dir=$3
  expected_step=$4
  log="$RUN_DIR/${label}.log"

  printf 'probe=%s config=%s start=%s\n' "$label" "$config" "$(date -Is)"
  setsid timeout --signal=TERM --kill-after=30s 6m \
    "$TORCHRUN" --standalone --nproc_per_node=8 \
    scripts/train_ddp.py --config "$config" >"$log" 2>&1 &
  probe_pid=$!
  detected_failure=0
  while kill -0 "$probe_pid" 2>/dev/null; do
    if grep -Eq 'FloatingPointError|CUDA out of memory|NCCL[^[:space:]]*.*(error|Error)' "$log"; then
      detected_failure=1
      kill -TERM -- "-$probe_pid" 2>/dev/null || true
      break
    fi
    sleep 2
  done
  set +e
  wait "$probe_pid"
  code=$?
  set -e
  cat "$log"

  if (( code == 0 )); then
    "$PYTHON" scripts/validate_training_checkpoint.py \
      --checkpoint "$out_dir/last.ckpt" \
      --history "$out_dir/history.json" \
      --expected-step "$expected_step" \
      --expected-world-size 8 \
      --output "$RUN_DIR/${label}_checkpoint_gate.json"
    if grep -Eq 'scaler_skips [1-9]' "$log"; then
      printf 'probe %s had scaler skips\n' "$label" >&2
      code=3
    fi
  fi

  printf '%s\t%s\t%s\t%s\n' "$label" "$code" "$detected_failure" "$(date -Is)" \
    >>"$STATUS_FILE"
}

for i in "${!PROBE_LABELS[@]}"; do
  run_probe "${PROBE_LABELS[$i]}" "${PROBE_CONFIGS[$i]}" \
    "${PROBE_DIRS[$i]}" "${PROBE_STEPS[$i]}"
done

archive_evidence
mkdir -p "$READBACK_ROOT"
for i in "${!PROBE_LABELS[@]}"; do
  label=${PROBE_LABELS[$i]}
  out_dir=${PROBE_DIRS[$i]}
  expected_step=${PROBE_STEPS[$i]}
  if [[ -s "$out_dir/last.ckpt" ]]; then
    mkdir "$READBACK_ROOT/$label"
    obsutil sync "$OBS_DST/$label" "$READBACK_ROOT/$label"
    "$PYTHON" scripts/validate_training_checkpoint.py \
      --checkpoint "$READBACK_ROOT/$label/last.ckpt" \
      --history "$READBACK_ROOT/$label/history.json" \
      --expected-step "$expected_step" \
      --expected-world-size 8 \
      --output "$RUN_DIR/${label}_obs_checkpoint_gate.json"
    local_sha=$(sha256sum "$out_dir/last.ckpt" | awk '{print $1}')
    readback_sha=$(sha256sum "$READBACK_ROOT/$label/last.ckpt" | awk '{print $1}')
    [[ "$local_sha" == "$readback_sha" ]] || {
      printf 'OBS SHA256 mismatch for %s\n' "$label" >&2
      exit 2
    }
  fi
done

printf '{"status":"COMPLETE","run_id":"%s","commit":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$actual_commit" "$OBS_DST" "$(date -Is)" | tee "$RUN_DIR/summary.json"
archive_evidence
printf 'TensorCloud01 numerical matrix complete; calibration/formal training was not started.\n'
