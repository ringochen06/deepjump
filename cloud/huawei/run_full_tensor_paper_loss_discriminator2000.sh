#!/usr/bin/env bash
# Train the full-tensor TensorCloud01 candidate from fresh initialization to
# step 2000 in two exact-resume stages, run the frozen H20 discriminator,
# archive/read back all artifacts, and always power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-135}

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 135 ]] || { printf 'HARD_STOP_MINUTES must be 135\n' >&2; exit 2; }

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 120s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -d "${CALIBRATION_DIR:-}" ]] && timeout 180s obsutil sync "$CALIBRATION_DIR" "${OBS_DST:-}/failure/calibration"
    [[ -d "${CONTINUATION_DIR:-}" ]] && timeout 180s obsutil sync "$CONTINUATION_DIR" "${OBS_DST:-}/failure/continuation"
    set -e
  fi
  printf 'bounded full-tensor discriminator exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -c 2>/dev/null || true
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown command failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
STAGE1_TIMEOUT_MINUTES=${STAGE1_TIMEOUT_MINUTES:-25}
STAGE2_TIMEOUT_MINUTES=${STAGE2_TIMEOUT_MINUTES:-25}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOMAIN_LIST=${DOMAIN_LIST:-configs/dev_20_length_proportional_seed0.txt}
DOMAIN_LIST_SHA256=${DOMAIN_LIST_SHA256:-4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af}
VECTOR_BASELINE_OBS_PREFIX=${VECTOR_BASELINE_OBS_PREFIX:-$BUCKET/deepjump-calibration/vector-only-paper-loss-continuation2000/20260718T032252Z/audit/rollouts}
VECTOR_BASELINE_SHA256=${VECTOR_BASELINE_SHA256:-35b73f0d3f0889201fb192735114a7e818e30df41259edf6f4a6f8f8479755ff}
CALIBRATION_CONFIG=configs/v100_tensorcloud01_full_d1_fp32_calibration.yaml
CONTINUATION_CONFIG=configs/v100_tensorcloud01_full_d1_fp32_continuation2000.yaml
CALIBRATION_DIR="$REPO/runs/v100_tensorcloud01_full_d1_fp32_calibration"
CONTINUATION_DIR="$REPO/runs/v100_tensorcloud01_full_d1_fp32_continuation2000"
RUN_DIR="$REPO/runs/full_tensor_paper_loss_discriminator2000_$RUN_ID"
ROLLOUT_DIR="$RUN_DIR/rollouts"
VECTOR_BASELINE_DIR="$RUN_DIR/vector_baseline"
READBACK_DIR="/tmp/full_tensor_paper_loss_discriminator2000_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-calibration/full-tensor-paper-loss-discriminator2000/$RUN_ID"

[[ "$STAGE1_TIMEOUT_MINUTES" -le 25 ]] || { printf 'stage-1 timeout exceeds 25 minutes\n' >&2; exit 2; }
[[ "$STAGE2_TIMEOUT_MINUTES" -le 25 ]] || { printf 'stage-2 timeout exceeds 25 minutes\n' >&2; exit 2; }
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$CALIBRATION_DIR" "$CONTINUATION_DIR"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$ROLLOUT_DIR"
mkdir "$VECTOR_BASELINE_DIR"
exec > >(tee -a "$RUN_DIR/discriminator.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty\n' >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
command -v obsutil >/dev/null
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || { printf 'runtime missing\n' >&2; exit 2; }
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
df -hT /data
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 107374182400 )) || { printf 'less than 100 GiB free\n' >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'

"$PYTHON" -m pytest -q \
  tests/test_tensor_cloud01.py \
  tests/test_cloud_configs.py \
  tests/test_training_gates.py \
  tests/test_paper_loss_continuation.py \
  tests/test_rollout_robustness_eval.py \
  tests/test_audit_mdcath_staging.py | tee "$RUN_DIR/pytest.log"

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

obsutil sync "$VECTOR_BASELINE_OBS_PREFIX" "$VECTOR_BASELINE_DIR"
mapfile -t baseline_candidates < <(
  find "$VECTOR_BASELINE_DIR" -type f -name rollout_2000.json -print
)
[[ "${#baseline_candidates[@]}" == 1 ]] || {
  printf 'expected exactly one frozen vector-only baseline; found %s\n' "${#baseline_candidates[@]}" >&2
  exit 2
}
VECTOR_BASELINE_PATH=${baseline_candidates[0]}
baseline_rel=$(realpath --relative-to="$RUN_DIR" "$VECTOR_BASELINE_PATH")
baseline_sha=$(sha256sum "$VECTOR_BASELINE_PATH" | awk '{print $1}')
[[ "$baseline_sha" == "$VECTOR_BASELINE_SHA256" ]] || {
  printf 'frozen vector-only baseline SHA256 mismatch\n' >&2
  exit 2
}

timeout --signal=TERM --kill-after=2m "${STAGE1_TIMEOUT_MINUTES}m" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CALIBRATION_CONFIG" \
  2>&1 | tee "$RUN_DIR/train_stage1.log"

grep -q 'world=8 params=4,840,032 effective_batch=128' "$RUN_DIR/train_stage1.log"
grep -q 'done. artifacts in' "$RUN_DIR/train_stage1.log"
if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$RUN_DIR/train_stage1.log"; then
  printf 'stage-1 training log contains a fatal signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train_stage1.log"; then
  printf 'stage-1 training log contains skipped optimizer updates\n' >&2
  exit 2
fi

SOURCE_CHECKPOINT="$CALIBRATION_DIR/ckpt_1000.pt"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --history "$CALIBRATION_DIR/history.json" \
  --expected-step 1000 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --history-mode contains \
  --output "$RUN_DIR/source_checkpoint_gate.json"
source_checkpoint_sha=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')

timeout --signal=TERM --kill-after=2m "${STAGE2_TIMEOUT_MINUTES}m" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CONTINUATION_CONFIG" --resume "$SOURCE_CHECKPOINT" \
  2>&1 | tee "$RUN_DIR/train_stage2.log"

grep -q 'world=8 params=4,840,032 effective_batch=128' "$RUN_DIR/train_stage2.log"
grep -q 'resumed from .* at step 1000' "$RUN_DIR/train_stage2.log"
grep -q 'done. artifacts in' "$RUN_DIR/train_stage2.log"
if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$RUN_DIR/train_stage2.log"; then
  printf 'stage-2 training log contains a fatal signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train_stage2.log"; then
  printf 'stage-2 training log contains skipped optimizer updates\n' >&2
  exit 2
fi

for step in $(seq 1100 100 2000); do
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$CONTINUATION_DIR/ckpt_${step}.pt" \
    --history "$CONTINUATION_DIR/history.json" \
    --expected-step "$step" --expected-world-size 8 --expected-delta 1 \
    --require-full-tensor --history-mode contains \
    --output "$RUN_DIR/local_ckpt_${step}_gate.json"
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 5m \
    "$PYTHON" scripts/rollout_robustness_eval.py \
    --ckpt "$CONTINUATION_DIR/ckpt_${step}.pt" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --domains 3 --starts 2 --steps 20 --methods mean,ode_1 \
    --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
    --output "$ROLLOUT_DIR/rollout_${step}.json" \
    > "$ROLLOUT_DIR/rollout_${step}.log" 2>&1
done

"$PYTHON" scripts/adjudicate_full_tensor_discriminator.py \
  --history "$CONTINUATION_DIR/history.json" \
  --rollout-dir "$ROLLOUT_DIR" \
  --vector-baseline "$VECTOR_BASELINE_PATH" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/decision.json"

source_checkpoint_sha_after=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
[[ "$source_checkpoint_sha_after" == "$source_checkpoint_sha" ]] || {
  printf 'source checkpoint changed during continuation\n' >&2
  exit 2
}

(
  cd "$CALIBRATION_DIR"
  sha256sum config.json history.json last.ckpt ckpt_*.pt
) > "$RUN_DIR/calibration_sha256.txt"
(
  cd "$CONTINUATION_DIR"
  sha256sum config.json history.json last.ckpt ckpt_*.pt
) > "$RUN_DIR/continuation_sha256.txt"
(
  cd "$RUN_DIR"
  sha256sum decision.json source_checkpoint_gate.json data_audit.json rollouts/*.json "$baseline_rel"
) > "$RUN_DIR/audit_sha256.txt"

obsutil sync "$CALIBRATION_DIR" "$OBS_DST/calibration"
obsutil sync "$CONTINUATION_DIR" "$OBS_DST/continuation"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST/calibration" "$READBACK_DIR/calibration"
obsutil sync "$OBS_DST/continuation" "$READBACK_DIR/continuation"
obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(
  cd "$READBACK_DIR/calibration"
  sha256sum -c "$RUN_DIR/calibration_sha256.txt"
)
(
  cd "$READBACK_DIR/continuation"
  sha256sum -c "$RUN_DIR/continuation_sha256.txt"
)
(
  cd "$READBACK_DIR/audit"
  sha256sum -c "$RUN_DIR/audit_sha256.txt"
)
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$READBACK_DIR/continuation/last.ckpt" \
  --history "$READBACK_DIR/continuation/history.json" \
  --expected-step 2000 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor \
  --output "$RUN_DIR/obs_last_gate.json"

decision_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")
final_sha=$(sha256sum "$CONTINUATION_DIR/ckpt_2000.pt" | awk '{print $1}')
printf '{"status":"%s","scope":"bounded_full_tensor_discriminator_only","formal_training_authorized":false,"run_id":"%s","commit":"%s","source_checkpoint_sha256":"%s","final_checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$decision_status" "$RUN_ID" "$actual_commit" "$source_checkpoint_sha" "$final_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
printf 'Bounded full-tensor discriminator complete with decision=%s; formal training was not started.\n' "$decision_status"
