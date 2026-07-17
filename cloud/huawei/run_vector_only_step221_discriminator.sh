#!/usr/bin/env bash
# Run the preregistered two-arm numerical discriminator for the vector-only
# TensorCloud01 step-221 FP16/high-LR failure. This never authorizes calibration
# or formal training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set EXPECTED_HOSTNAME to the authorized GPU instance hostname}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1 for the authorized bounded run}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-20}

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || {
  printf 'refusing unbounded run: SHUTDOWN_ON_EXIT must be 1\n' >&2
  exit 2
}
[[ "$HARD_STOP_MINUTES" == 20 ]] || {
  printf 'refusing changed hard stop: HARD_STOP_MINUTES must be 20\n' >&2
  exit 2
}
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  printf 'hostname mismatch: actual=%s expected=%s\n' "$(hostname)" "$EXPECTED_HOSTNAME" >&2
  exit 2
}

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ -n "${RUN_DIR:-}" ]] && command -v obsutil >/dev/null \
    && declare -F archive_evidence >/dev/null; then
    set +e
    archive_evidence
    set -e
  fi
  printf 'numerical discriminator exit=%s; requesting immediate shutdown at %s\n' \
    "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: immediate shutdown command failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

# Defense in depth before any run-specific input is required.
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set EXPECTED_REPO_COMMIT to the reviewed deployed SHA}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OBS_RUN_PREFIX=${OBS_RUN_PREFIX:-deepjump-preflight/vector-only-step221}
RUN_DIR="$REPO/runs/vector_only_step221_discriminator_$RUN_ID"
READBACK_ROOT="/tmp/vector_only_step221_readback_$RUN_ID"
OBS_DST="$BUCKET/$OBS_RUN_PREFIX/$RUN_ID"
STATUS_FILE="$RUN_DIR/status.tsv"

LABELS=(fp32_highlr fp16_lowlr)
CONFIGS=(
  configs/v100_tensorcloud01_vector_only_fp32_highlr_step230.yaml
  configs/v100_tensorcloud01_vector_only_fp16_lowlr_step230.yaml
)
OUT_DIRS=(
  "$REPO/runs/v100_tensorcloud01_vector_only_fp32_highlr_step230"
  "$REPO/runs/v100_tensorcloud01_vector_only_fp16_lowlr_step230"
)

archive_evidence() {
  command -v obsutil >/dev/null || return 0
  if [[ -d "$RUN_DIR" ]]; then
    timeout 90s obsutil sync "$RUN_DIR" "$OBS_DST/audit" \
      || printf 'WARNING: best-effort audit archive failed\n' >&2
  fi
  for i in "${!LABELS[@]}"; do
    if [[ -d "${OUT_DIRS[$i]}" ]]; then
      timeout 90s obsutil sync "${OUT_DIRS[$i]}" "$OBS_DST/${LABELS[$i]}" \
        || printf 'WARNING: best-effort arm archive failed: %s\n' "${LABELS[$i]}" >&2
    fi
  done
  return 0
}

cd "$REPO"
mkdir -p "$RUN_DIR"
: >"$STATUS_FILE"
exec > >(tee -a "$RUN_DIR/discriminator.log") 2>&1

printf 'run_id=%s start=%s hard_stop_minutes=%s obs_dst=%s\n' \
  "$RUN_ID" "$(date -Is)" "$HARD_STOP_MINUTES" "$OBS_DST"
command -v obsutil >/dev/null
[[ -x "$PYTHON" ]] || { printf 'missing python: %s\n' "$PYTHON" >&2; exit 2; }
[[ -x "$TORCHRUN" ]] || { printf 'missing torchrun: %s\n' "$TORCHRUN" >&2; exit 2; }
if pgrep -af '[s]cripts/train_ddp.py'; then
  printf 'another train_ddp.py process already exists; refusing duplicate run\n' >&2
  exit 2
fi

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'deployed commit mismatch: actual=%s expected=%s\n' \
    "$actual_commit" "$EXPECTED_REPO_COMMIT" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty; refusing numerical discriminator\n' >&2
  exit 2
}
gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$gpu_count" == 8 ]] || { printf 'GPU count %s != 8\n' "$gpu_count" >&2; exit 2; }
for dir in "${OUT_DIRS[@]}"; do
  [[ ! -e "$dir" ]] || { printf 'refusing to overwrite %s\n' "$dir" >&2; exit 2; }
done

"$PYTHON" -m pytest -q tests/test_training_gates.py tests/test_cloud_configs.py
"$PYTHON" scripts/audit_mdcath_staging.py \
  --root "$DATA_ROOT" \
  --expected-h5 1000 \
  --expected-bytes 668131379559 \
  --expected-subset-sha256 39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734 \
  --expected-trajectories 25000 \
  --expected-strategy length-proportional \
  --expected-seed 20260715 \
  --expected-commit 3f9f7f7 \
  --samples 3 | tee "$RUN_DIR/data_audit.json"

run_arm() {
  label=$1
  config=$2
  out_dir=$3
  log="$RUN_DIR/${label}.log"
  status=FAIL

  printf 'arm=%s config=%s start=%s\n' "$label" "$config" "$(date -Is)"
  setsid timeout --signal=TERM --kill-after=30s 8m \
    "$TORCHRUN" --standalone --nproc_per_node=8 \
    scripts/train_ddp.py --config "$config" >"$log" 2>&1 &
  arm_pid=$!
  while kill -0 "$arm_pid" 2>/dev/null; do
    if grep -Eq 'FloatingPointError|CUDA out of memory|NCCL[^[:space:]]*.*(error|Error)' "$log"; then
      kill -TERM -- "-$arm_pid" 2>/dev/null || true
      break
    fi
    sleep 2
  done
  set +e
  wait "$arm_pid"
  code=$?
  set -e
  cat "$log"

  if (( code == 0 )) \
    && grep -q 'world=8 params=4,038,240 effective_batch=128' "$log" \
    && grep -q 'done. artifacts in' "$log" \
    && ! grep -Eq 'scaler_skips [1-9][0-9]*' "$log" \
    && ! grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$log"
  then
    set +e
    "$PYTHON" scripts/validate_training_checkpoint.py \
      --checkpoint "$out_dir/ckpt_230.pt" \
      --history "$out_dir/history.json" \
      --expected-step 230 \
      --expected-world-size 8 \
      --expected-delta 1 \
      --require-vector-only \
      --output "$RUN_DIR/${label}_ckpt_gate.json"
    gate_code=$?
    "$PYTHON" scripts/validate_training_checkpoint.py \
      --checkpoint "$out_dir/last.ckpt" \
      --history "$out_dir/history.json" \
      --expected-step 230 \
      --expected-world-size 8 \
      --expected-delta 1 \
      --require-vector-only \
      --output "$RUN_DIR/${label}_last_gate.json"
    last_gate_code=$?
    gate_code=$((gate_code | last_gate_code))
    set -e
    if (( gate_code == 0 )); then
      status=PASS
    else
      code=$gate_code
    fi
  fi

  printf '%s\t%s\t%s\t%s\n' "$label" "$status" "$code" "$(date -Is)" \
    | tee -a "$STATUS_FILE"
  archive_evidence

  if [[ "$status" == PASS ]]; then
    mkdir -p "$READBACK_ROOT/$label"
    (
      cd "$out_dir"
      sha256sum ckpt_230.pt last.ckpt history.json config.json
    ) >"$RUN_DIR/${label}_sha256.txt"
    obsutil sync "$OBS_DST/$label" "$READBACK_ROOT/$label"
    (
      cd "$READBACK_ROOT/$label"
      sha256sum -c "$RUN_DIR/${label}_sha256.txt"
    )
    "$PYTHON" scripts/validate_training_checkpoint.py \
      --checkpoint "$READBACK_ROOT/$label/last.ckpt" \
      --history "$READBACK_ROOT/$label/history.json" \
      --expected-step 230 \
      --expected-world-size 8 \
      --expected-delta 1 \
      --require-vector-only \
      --output "$RUN_DIR/${label}_obs_last_gate.json"
  fi
}

for i in "${!LABELS[@]}"; do
  run_arm "${LABELS[$i]}" "${CONFIGS[$i]}" "${OUT_DIRS[$i]}"
done

[[ "$(wc -l < "$STATUS_FILE" | tr -d ' ')" == 2 ]] || {
  printf 'numerical matrix did not record exactly two arms\n' >&2
  exit 2
}
printf '{"status":"MATRIX_COMPLETE","scope":"numerical_discriminator","run_id":"%s","commit":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$actual_commit" "$OBS_DST" "$(date -Is)" | tee "$RUN_DIR/summary.json"
archive_evidence
printf 'Vector-only step-221 numerical matrix complete; calibration/formal training was not started.\n'
