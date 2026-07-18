#!/usr/bin/env bash
# Run the bounded same-domain feedback-adaptation discriminator and always power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:?set audited step-5000 checkpoint}
SOURCE_CHECKPOINT_SHA256=${SOURCE_CHECKPOINT_SHA256:?set audited checkpoint SHA256}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-120}
TRAIN_TIMEOUT_MINUTES=${TRAIN_TIMEOUT_MINUTES:-90}
HARD_STOP_UNIT="deepjump-feedback-adapt-hard-stop-$RUN_ID-$$"

CONFIG=configs/v100_tensorcloud01_full_d1_unroll3_adapt250.yaml
DOMAIN_LIST=configs/tiny_overfit_domain_1a0hA01.txt
DOMAIN_LIST_SHA256=3da0d5c44e5d1a68aa5e99d01acf296725f7e8f94ca990547991b0d87bcf0d9d
TRAIN_DIR="$REPO/runs/v100_tensorcloud01_full_d1_unroll3_adapt250"
RUN_DIR="$REPO/runs/full_tensor_feedback_adapt250_$RUN_ID"
READBACK_DIR="/tmp/full_tensor_feedback_adapt250_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/full-tensor-feedback-adapt250/$RUN_ID"

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" -le 120 ]] || { printf 'hard stop exceeds 120 minutes\n' >&2; exit 2; }
[[ "$TRAIN_TIMEOUT_MINUTES" -le 90 ]] || { printf 'training timeout exceeds 90 minutes\n' >&2; exit 2; }

shutdown_on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 180s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -d "${TRAIN_DIR:-}" ]] && timeout 300s obsutil sync "$TRAIN_DIR" "${OBS_DST:-}/failure/training"
    set -e
  fi
  printf 'feedback adaptation exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || true
  exit "$code"
}
trap shutdown_on_exit EXIT

sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$TRAIN_DIR" "$READBACK_DIR"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { printf 'tracked worktree is dirty\n' >&2; exit 2; }
[[ "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" == "$SOURCE_CHECKPOINT_SHA256" ]] || {
  printf 'source checkpoint SHA256 mismatch\n' >&2
  exit 2
}
[[ "$(sha256sum "$DOMAIN_LIST" | awk '{print $1}')" == "$DOMAIN_LIST_SHA256" ]] || {
  printf 'domain list SHA256 mismatch\n' >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
command -v obsutil >/dev/null
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || { printf 'runtime missing\n' >&2; exit 2; }
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 107374182400 )) || { printf 'less than 100 GiB free\n' >&2; exit 2; }

printf 'RUN_ID=%s\nCOMMIT=%s\nCONFIG=%s\nSOURCE_CHECKPOINT=%s\n' \
  "$RUN_ID" "$actual_commit" "$CONFIG" "$SOURCE_CHECKPOINT"
findmnt /data
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -m pytest -q \
  tests/test_tensor_cloud01.py \
  tests/test_cloud_configs.py \
  tests/test_unroll_feedback.py \
  tests/test_rollout_robustness_eval.py \
  tests/test_feedback_adaptation.py | tee "$RUN_DIR/pytest.log"

"$PYTHON" scripts/audit_mdcath_staging.py \
  --root "$DATA_ROOT" --expected-h5 1000 --expected-bytes 668131379559 \
  --expected-subset-sha256 39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734 \
  --expected-trajectories 25000 --expected-strategy length-proportional \
  --expected-seed 20260715 --expected-commit 3f9f7f7 --samples 5 \
  | tee "$RUN_DIR/data_audit.json"

timeout --signal=TERM --kill-after=2m "${TRAIN_TIMEOUT_MINUTES}m" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CONFIG" --warm-start "$SOURCE_CHECKPOINT" \
  2>&1 | tee "$RUN_DIR/train.log"

grep -q 'world=8 params=4,840,032 effective_batch=128 .*train_domains=1 val_domains=1' "$RUN_DIR/train.log"
grep -q 'warm-started model .* fresh optimizer/step 0' "$RUN_DIR/train.log"
grep -q 'done. artifacts in' "$RUN_DIR/train.log"
if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$RUN_DIR/train.log"; then
  printf 'training log contains a fatal signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train.log"; then
  printf 'training log contains skipped optimizer updates\n' >&2
  exit 2
fi

CHECKPOINT="$TRAIN_DIR/ckpt_250.pt"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$CHECKPOINT" --history "$TRAIN_DIR/history.json" \
  --expected-step 250 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --history-mode contains --output "$RUN_DIR/local_ckpt_gate.json"
checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 10m \
  "$PYTHON" scripts/rollout_robustness_eval.py \
  --ckpt "$CHECKPOINT" --domain-list "$DOMAIN_LIST" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" --domains 1 --starts 5 --steps 6 \
  --methods mean --teacher-forced-mean --seed 20260718 \
  --integrator euler --tau-max 1.0 --drift-anchor state \
  --output "$RUN_DIR/rollout.json" > "$RUN_DIR/rollout.log" 2>&1

"$PYTHON" scripts/adjudicate_feedback_adaptation.py \
  --result "$RUN_DIR/rollout.json" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$checkpoint_sha" \
  --output "$RUN_DIR/decision.json"

(
  cd "$TRAIN_DIR"
  sha256sum config.json history.json last.ckpt ckpt_*.pt
) > "$RUN_DIR/training_sha256.txt"
(
  cd "$RUN_DIR"
  sha256sum pytest.log data_audit.json train.log local_ckpt_gate.json rollout.json decision.json
) > "$RUN_DIR/audit_sha256.txt"

obsutil sync "$TRAIN_DIR" "$OBS_DST/training"
obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR"
obsutil sync "$OBS_DST/training" "$READBACK_DIR/training"
obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(cd "$READBACK_DIR/training" && sha256sum -c "$RUN_DIR/training_sha256.txt")
(cd "$READBACK_DIR/audit" && sha256sum -c "$RUN_DIR/audit_sha256.txt")

decision_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")
printf '{"status":"%s","scope":"single_domain_250_step_feedback_adaptation_only","formal_training_authorized":false,"run_id":"%s","commit":"%s","checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$decision_status" "$RUN_ID" "$actual_commit" "$checkpoint_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
sha256sum "$RUN_DIR/summary.json" > "$RUN_DIR/summary_sha256.txt"
obsutil cp -f "$RUN_DIR/summary.json" "$OBS_DST/audit/summary.json"
obsutil cp -f "$RUN_DIR/summary_sha256.txt" "$OBS_DST/audit/summary_sha256.txt"
printf 'Feedback adaptation complete with decision=%s; formal training was not started.\n' "$decision_status"
