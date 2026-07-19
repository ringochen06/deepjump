#!/usr/bin/env bash
# Run the bounded first-party source-law discriminator and always power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-135}
TRAIN_TIMEOUT_MINUTES=${TRAIN_TIMEOUT_MINUTES:-35}
HARD_STOP_UNIT="deepjump-source-law-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 240s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -d "${TRAIN_DIR:-}" ]] && timeout 360s obsutil sync "$TRAIN_DIR" "${OBS_DST:-}/failure/training"
    set -e
  fi
  printf 'source-law discriminator exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 135 ]] || { printf 'HARD_STOP_MINUTES must be 135\n' >&2; exit 2; }
[[ "$TRAIN_TIMEOUT_MINUTES" -le 35 ]] || { printf 'training timeout exceeds 35 minutes\n' >&2; exit 2; }

sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n systemctl stop 'deepjump-recovery-hard-stop-*.timer' 2>/dev/null || true
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
CONFIG=configs/v100_tensorcloud01_full_d1_first_party_source_law1000.yaml
CONFIG_SHA256=4af2c12e7db9b77f58faf376da800f088ba65521d927c6a4582dc4b9c03c53d3
DOMAIN_LIST=configs/tiny_overfit_domain_1a0hA01.txt
DOMAIN_LIST_SHA256=3da0d5c44e5d1a68aa5e99d01acf296725f7e8f94ca990547991b0d87bcf0d9d
TRAIN_DIR="$REPO/runs/v100_tensorcloud01_full_d1_first_party_source_law1000"
RUN_DIR="$REPO/runs/first_party_source_law1000_$RUN_ID"
READBACK_DIR="/tmp/first_party_source_law1000_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/first-party-source-law1000/$RUN_ID"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$TRAIN_DIR" "$READBACK_DIR"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { printf 'tracked worktree is dirty\n' >&2; exit 2; }
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || {
  printf 'config SHA256 mismatch\n' >&2
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

findmnt /data
df -hT /data
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 107374182400 )) || { printf 'less than 100 GiB free\n' >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'

"$PYTHON" -m pytest -q \
  tests/test_sampling_integrators.py \
  tests/test_source_law_adjudication.py \
  tests/test_tensor_cloud01.py \
  tests/test_cloud_configs.py \
  tests/test_training_gates.py \
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

timeout --signal=TERM --kill-after=2m "${TRAIN_TIMEOUT_MINUTES}m" \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CONFIG" \
  2>&1 | tee "$RUN_DIR/train.log"

grep -q 'world=8 params=4,840,032 effective_batch=128 .*train_domains=1 val_domains=1' "$RUN_DIR/train.log"
grep -q 'done. artifacts in' "$RUN_DIR/train.log"
if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$RUN_DIR/train.log"; then
  printf 'training log contains a fatal signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train.log"; then
  printf 'training log contains skipped optimizer updates\n' >&2
  exit 2
fi

CHECKPOINT="$TRAIN_DIR/ckpt_1000.pt"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$CHECKPOINT" --history "$TRAIN_DIR/history.json" \
  --expected-step 1000 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --history-mode contains \
  --output "$RUN_DIR/local_ckpt_gate.json"
checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 25m \
  "$PYTHON" scripts/rollout_robustness_eval.py \
  --ckpt "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 1 --starts 5 --steps 6 --methods ode_150 \
  --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
  --output "$RUN_DIR/h6.json" > "$RUN_DIR/h6.log" 2>&1

"$PYTHON" scripts/adjudicate_source_law_candidate.py \
  --h6 "$RUN_DIR/h6.json" --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$checkpoint_sha" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/decision.json"
status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")

if [[ "$status" == ADVANCE_SOURCE_LAW_H20 ]]; then
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 60m \
    "$PYTHON" scripts/rollout_robustness_eval.py \
    --ckpt "$CHECKPOINT" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --domains 1 --starts 5 --steps 20 --methods ode_150 \
    --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
    --output "$RUN_DIR/h20.json" > "$RUN_DIR/h20.log" 2>&1
  "$PYTHON" scripts/adjudicate_source_law_candidate.py \
    --h6 "$RUN_DIR/h6.json" --h20 "$RUN_DIR/h20.json" \
    --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$checkpoint_sha" \
    --domain-list-sha256 "$DOMAIN_LIST_SHA256" --output "$RUN_DIR/decision.json"
  status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")
fi

(
  cd "$TRAIN_DIR"
  sha256sum config.json history.json last.ckpt ckpt_*.pt
) > "$RUN_DIR/training_sha256.txt"
printf '{"status":"%s","scope":"single_domain_first_party_source_law_only","formal_training_authorized":false,"run_id":"%s","commit":"%s","checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$status" "$RUN_ID" "$actual_commit" "$checkpoint_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
(
  cd "$RUN_DIR"
  sha256sum decision.json h6.json h6.log data_audit.json local_ckpt_gate.json \
    pytest.log summary.json
  [[ ! -f h20.json ]] || sha256sum h20.json h20.log
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 5m obsutil sync "$TRAIN_DIR" "$OBS_DST/training"
timeout --signal=TERM --kill-after=30s 3m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir "$READBACK_DIR"
timeout --signal=TERM --kill-after=30s 5m obsutil sync "$OBS_DST/training" "$READBACK_DIR/training"
timeout --signal=TERM --kill-after=30s 3m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(
  cd "$READBACK_DIR/training"
  sha256sum -c "$RUN_DIR/training_sha256.txt"
)
(
  cd "$READBACK_DIR/audit"
  sha256sum -c "$RUN_DIR/audit_sha256.txt"
)
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$READBACK_DIR/training/ckpt_1000.pt" \
  --history "$READBACK_DIR/training/history.json" \
  --expected-step 1000 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --history-mode contains \
  --output "$RUN_DIR/obs_ckpt_gate.json"
timeout --signal=TERM --kill-after=30s 2m \
  obsutil cp "$RUN_DIR/obs_ckpt_gate.json" "$OBS_DST/audit/obs_ckpt_gate.json"
mkdir -p "$READBACK_DIR/final"
timeout --signal=TERM --kill-after=30s 2m \
  obsutil cp "$OBS_DST/audit/obs_ckpt_gate.json" "$READBACK_DIR/final/obs_ckpt_gate.json"
cmp "$RUN_DIR/obs_ckpt_gate.json" "$READBACK_DIR/final/obs_ckpt_gate.json"
printf 'Source-law discriminator complete with decision=%s; formal training was not started.\n' "$status"
