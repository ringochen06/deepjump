#!/usr/bin/env bash
# Run exactly one fresh-init, 1000-step vector-only TensorCloud01 delta=1
# calibration on eight V100s.
# This fail-closed diagnostic never starts formal training and powers the instance
# off on every exit path.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
# Always import the reviewed deployment, never a stale site-packages checkout.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set EXPECTED_HOSTNAME to the authorized GPU instance hostname}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1 for the authorized bounded run}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-30}

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || {
  printf 'refusing unbounded run: SHUTDOWN_ON_EXIT must be 1\n' >&2
  exit 2
}
[[ "$HARD_STOP_MINUTES" == 30 ]] || {
  printf 'refusing changed hard stop: HARD_STOP_MINUTES must be 30\n' >&2
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
  if [[ "$code" != 0 ]] && [[ -n "${RUN_DIR:-}" ]] && command -v obsutil >/dev/null; then
    set +e
    printf 'calibration failed; best-effort evidence archive start=%s\n' "$(date -Is)"
    [[ -d "$RUN_DIR" ]] && timeout 120s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -n "${CALIBRATION_DIR:-}" ]] && [[ -d "$CALIBRATION_DIR" ]] && \
      timeout 120s obsutil sync "$CALIBRATION_DIR" "${OBS_DST:-}/failure/calibration"
    set -e
  fi
  printf 'calibration delta=%s exit=%s; requesting immediate shutdown at %s\n' \
    "${DELTA:-unknown}" "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: immediate shutdown command failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

# Defense in depth: this independent timer survives a hung child process.
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

# Require all run-specific inputs only after the independent hard stop is armed.
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set EXPECTED_REPO_COMMIT to the reviewed deployed SHA}
DELTA=${DELTA:?set DELTA to exactly 1}
TRAIN_TIMEOUT_MINUTES=${TRAIN_TIMEOUT_MINUTES:-24}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OBS_RUN_PREFIX=${OBS_RUN_PREFIX:-deepjump-calibration/tensorcloud01}
LR_PROFILE=${LR_PROFILE:-reference}

[[ "$DELTA" == 1 ]] || {
  printf 'unsupported DELTA=%s; vector-only calibration is frozen to delta=1\n' "$DELTA" >&2
  exit 2
}
[[ "$TRAIN_TIMEOUT_MINUTES" -le 24 ]] || {
  printf 'refusing training timeout above 24 minutes\n' >&2
  exit 2
}
case "$LR_PROFILE" in
  reference)
    CONFIG=configs/v100_tensorcloud01_vector_only_d1_calibration.yaml
    CALIBRATION_DIR="$REPO/runs/v100_tensorcloud01_vector_only_d1_calibration"
    ;;
  lowlr)
    CONFIG=configs/v100_tensorcloud01_vector_only_d1_lowlr_calibration.yaml
    CALIBRATION_DIR="$REPO/runs/v100_tensorcloud01_vector_only_d1_lowlr_calibration"
    ;;
  *)
    printf 'unsupported LR_PROFILE=%s; expected reference or lowlr\n' "$LR_PROFILE" >&2
    exit 2
    ;;
esac

RUN_DIR="$REPO/runs/tensorcloud01_d${DELTA}_calibration_audit_$RUN_ID"
READBACK_DIR="/tmp/tensorcloud01_d${DELTA}_calibration_readback_$RUN_ID"
OBS_DST="$BUCKET/$OBS_RUN_PREFIX/$LR_PROFILE/delta$DELTA/$RUN_ID"

cd "$REPO"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/calibration.log") 2>&1

printf 'run_id=%s delta=%s lr_profile=%s start=%s hard_stop_minutes=%s train_timeout_minutes=%s obs_dst=%s\n' \
  "$RUN_ID" "$DELTA" "$LR_PROFILE" "$(date -Is)" "$HARD_STOP_MINUTES" "$TRAIN_TIMEOUT_MINUTES" "$OBS_DST"

command -v obsutil >/dev/null
[[ -x "$PYTHON" ]] || { printf 'missing python: %s\n' "$PYTHON" >&2; exit 2; }
[[ -x "$TORCHRUN" ]] || { printf 'missing torchrun: %s\n' "$TORCHRUN" >&2; exit 2; }
[[ ! -e "$CALIBRATION_DIR" ]] || {
  printf 'refusing to overwrite calibration directory: %s\n' "$CALIBRATION_DIR" >&2
  exit 2
}
if pgrep -af '[s]cripts/train_ddp.py'; then
  printf 'another train_ddp.py process already exists; refusing duplicate run\n' >&2
  exit 2
fi

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'deployed commit mismatch: actual=%s expected=%s\n' "$actual_commit" "$EXPECTED_REPO_COMMIT" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty; refusing cloud calibration\n' >&2
  exit 2
}

gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$gpu_count" == 8 ]] || { printf 'GPU count %s != 8\n' "$gpu_count" >&2; exit 2; }
findmnt /data
df -hT /data
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 107374182400 )) || {
  printf 'less than 100 GiB available on /data: %s bytes\n' "$available_bytes" >&2
  exit 2
}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'

printf 'gate=tests start=%s\n' "$(date -Is)"
timeout --signal=TERM --kill-after=30s 8m \
  "$PYTHON" -m pytest -q \
  tests/test_tensor_cloud01.py \
  tests/test_cloud_configs.py \
  tests/test_training_gates.py \
  tests/test_select_calibration_checkpoints.py \
  tests/test_scientific_evaluation_gate.py \
  tests/test_transition_robustness_eval.py \
  tests/test_audit_mdcath_staging.py \
  2>&1 | tee "$RUN_DIR/pytest.log"

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

printf 'gate=eight_gpu_delta%s_calibration start=%s\n' "$DELTA" "$(date -Is)"
timeout --signal=TERM --kill-after=2m "${TRAIN_TIMEOUT_MINUTES}m" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CONFIG" \
  2>&1 | tee "$RUN_DIR/train.log"

grep -q 'world=8 params=4,038,240 effective_batch=128' "$RUN_DIR/train.log"
grep -q 'done. artifacts in' "$RUN_DIR/train.log"
if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$RUN_DIR/train.log"; then
  printf 'training log contains a fatal numerical/runtime signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train.log"; then
  printf 'training log contains one or more GradScaler skipped updates\n' >&2
  exit 2
fi

printf 'gate=local_checkpoints start=%s\n' "$(date -Is)"
for step in 250 500 750 1000; do
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$CALIBRATION_DIR/ckpt_${step}.pt" \
    --history "$CALIBRATION_DIR/history.json" \
    --expected-step "$step" \
    --expected-world-size 8 \
    --expected-delta 1 \
    --require-vector-only \
    --history-mode contains \
    --output "$RUN_DIR/local_ckpt_${step}_gate.json"
done
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$CALIBRATION_DIR/last.ckpt" \
  --history "$CALIBRATION_DIR/history.json" \
  --expected-step 1000 \
  --expected-world-size 8 \
  --expected-delta 1 \
  --require-vector-only \
  --output "$RUN_DIR/local_last_gate.json"
"$PYTHON" scripts/select_calibration_checkpoints.py \
  --history "$CALIBRATION_DIR/history.json" \
  --config "$CALIBRATION_DIR/config.json" \
  --expected-delta "$DELTA" --require-vector-only --count 2 \
  --output "$RUN_DIR/local_checkpoint_selection.json"

printf 'gate=resume_readback start=%s\n' "$(date -Is)"
timeout --signal=TERM --kill-after=1m 3m \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CONFIG" --resume "$CALIBRATION_DIR/last.ckpt" \
  2>&1 | tee "$RUN_DIR/resume_readback.log"
grep -q 'resumed from .*last.ckpt at step 1000' "$RUN_DIR/resume_readback.log"

(
  cd "$CALIBRATION_DIR"
  sha256sum ckpt_250.pt ckpt_500.pt ckpt_750.pt ckpt_1000.pt last.ckpt history.json config.json
) > "$RUN_DIR/artifact_sha256.txt"

printf 'gate=obs_archive_and_readback start=%s\n' "$(date -Is)"
obsutil sync "$CALIBRATION_DIR" "$OBS_DST/calibration"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST/calibration" "$READBACK_DIR"
(
  cd "$READBACK_DIR"
  sha256sum -c "$RUN_DIR/artifact_sha256.txt"
)
for step in 250 500 750 1000; do
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$READBACK_DIR/ckpt_${step}.pt" \
    --history "$READBACK_DIR/history.json" \
    --expected-step "$step" \
    --expected-world-size 8 \
    --expected-delta 1 \
    --require-vector-only \
    --history-mode contains \
    --output "$RUN_DIR/obs_ckpt_${step}_gate.json"
done
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$READBACK_DIR/last.ckpt" \
  --history "$READBACK_DIR/history.json" \
  --expected-step 1000 \
  --expected-world-size 8 \
  --expected-delta 1 \
  --require-vector-only \
  --output "$RUN_DIR/obs_last_gate.json"
"$PYTHON" scripts/select_calibration_checkpoints.py \
  --history "$READBACK_DIR/history.json" \
  --config "$READBACK_DIR/config.json" \
  --expected-delta "$DELTA" --require-vector-only --count 2 \
  --output "$RUN_DIR/obs_checkpoint_selection.json"
cmp "$RUN_DIR/local_checkpoint_selection.json" "$RUN_DIR/obs_checkpoint_selection.json"

final_sha=$(sha256sum "$CALIBRATION_DIR/ckpt_1000.pt" | awk '{print $1}')
printf '{"status":"PASS","scope":"bounded_calibration","run_id":"%s","commit":"%s","lr_profile":"%s","delta_frames":%s,"steps":1000,"checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$actual_commit" "$LR_PROFILE" "$DELTA" "$final_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
printf 'TensorCloud01 delta=%s lr_profile=%s bounded calibration PASS; formal training was not started.\n' \
  "$DELTA" "$LR_PROFILE"
