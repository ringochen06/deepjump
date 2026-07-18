#!/usr/bin/env bash
# Resume the exact audited FP32 vector-only step-1000 checkpoint to step 2000,
# evaluate every 100 steps, archive/read back artifacts, and always power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-90}

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 90 ]] || { printf 'HARD_STOP_MINUTES must be 90\n' >&2; exit 2; }

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 120s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -d "${CONTINUATION_DIR:-}" ]] && timeout 180s obsutil sync "$CONTINUATION_DIR" "${OBS_DST:-}/failure/continuation"
    set -e
  fi
  printf 'bounded continuation exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
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

CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the frozen step-1000 checkpoint}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?set exact checkpoint SHA256}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
TRAIN_TIMEOUT_MINUTES=${TRAIN_TIMEOUT_MINUTES:-25}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOMAIN_LIST=${DOMAIN_LIST:-configs/dev_20_length_proportional_seed0.txt}
DOMAIN_LIST_SHA256=${DOMAIN_LIST_SHA256:-4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af}
CONFIG=configs/v100_tensorcloud01_vector_only_d1_fp32_continuation2000.yaml
CONTINUATION_DIR="$REPO/runs/v100_tensorcloud01_vector_only_d1_fp32_continuation2000"
RUN_DIR="$REPO/runs/vector_only_paper_loss_continuation2000_$RUN_ID"
ROLLOUT_DIR="$RUN_DIR/rollouts"
READBACK_DIR="/tmp/vector_only_paper_loss_continuation2000_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-calibration/vector-only-paper-loss-continuation2000/$RUN_ID"

[[ "$TRAIN_TIMEOUT_MINUTES" -le 25 ]] || { printf 'training timeout exceeds 25 minutes\n' >&2; exit 2; }
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
[[ ! -e "$RUN_DIR" ]] || { printf 'refusing to overwrite %s\n' "$RUN_DIR" >&2; exit 2; }
[[ ! -e "$CONTINUATION_DIR" ]] || {
  printf 'refusing to overwrite continuation directory %s\n' "$CONTINUATION_DIR" >&2
  exit 2
}
mkdir -p "$ROLLOUT_DIR"
exec > >(tee -a "$RUN_DIR/continuation.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty\n' >&2
  exit 2
}
[[ -f "$CHECKPOINT" ]] || { printf 'checkpoint missing\n' >&2; exit 2; }
source_checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$source_checkpoint_sha" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch\n' >&2
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

"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --history "$(dirname "$CHECKPOINT")/history.json" \
  --expected-step 1000 --expected-world-size 8 --expected-delta 1 \
  --require-vector-only --history-mode contains \
  --output "$RUN_DIR/source_checkpoint_gate.json"

timeout --signal=TERM --kill-after=2m "${TRAIN_TIMEOUT_MINUTES}m" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CONFIG" --resume "$CHECKPOINT" \
  2>&1 | tee "$RUN_DIR/train.log"

grep -q 'world=8 params=4,038,240 effective_batch=128' "$RUN_DIR/train.log"
grep -q 'resumed from .* at step 1000' "$RUN_DIR/train.log"
grep -q 'done. artifacts in' "$RUN_DIR/train.log"
if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$RUN_DIR/train.log"; then
  printf 'training log contains a fatal signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train.log"; then
  printf 'training log contains skipped optimizer updates\n' >&2
  exit 2
fi

for step in $(seq 1100 100 2000); do
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$CONTINUATION_DIR/ckpt_${step}.pt" \
    --history "$CONTINUATION_DIR/history.json" \
    --expected-step "$step" --expected-world-size 8 --expected-delta 1 \
    --require-vector-only --history-mode contains \
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

"$PYTHON" scripts/adjudicate_paper_loss_continuation.py \
  --history "$CONTINUATION_DIR/history.json" \
  --rollout-dir "$ROLLOUT_DIR" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/decision.json"

source_checkpoint_sha_after=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$source_checkpoint_sha_after" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  printf 'source checkpoint changed during continuation\n' >&2
  exit 2
}

(
  cd "$CONTINUATION_DIR"
  sha256sum config.json history.json last.ckpt ckpt_*.pt
) > "$RUN_DIR/continuation_sha256.txt"
(
  cd "$RUN_DIR"
  sha256sum decision.json source_checkpoint_gate.json data_audit.json rollouts/*.json
) > "$RUN_DIR/audit_sha256.txt"

obsutil sync "$CONTINUATION_DIR" "$OBS_DST/continuation"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST/continuation" "$READBACK_DIR/continuation"
obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
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
  --require-vector-only \
  --output "$RUN_DIR/obs_last_gate.json"

decision_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")
final_sha=$(sha256sum "$CONTINUATION_DIR/ckpt_2000.pt" | awk '{print $1}')
printf '{"status":"%s","scope":"bounded_paper_loss_continuation_only","formal_training_authorized":false,"run_id":"%s","commit":"%s","source_checkpoint_sha256":"%s","final_checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$decision_status" "$RUN_ID" "$actual_commit" "$source_checkpoint_sha" "$final_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
printf 'Bounded continuation complete with decision=%s; formal training was not started.\n' "$decision_status"
