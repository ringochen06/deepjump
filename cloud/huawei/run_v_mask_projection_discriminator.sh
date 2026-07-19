#!/usr/bin/env bash
# Run the bounded same-checkpoint masked-V inference discriminator and power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-75}
HARD_STOP_UNIT="deepjump-v-mask-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  printf 'masked-V discriminator exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 75 ]] || { printf 'HARD_STOP_MINUTES must be 75\n' >&2; exit 2; }
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
CHECKPOINT=${CHECKPOINT:?set the existing source-law ckpt_1000.pt path}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the frozen checkpoint SHA256}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOMAIN_LIST=configs/tiny_overfit_domain_1a0hA01.txt
DOMAIN_LIST_SHA256=3da0d5c44e5d1a68aa5e99d01acf296725f7e8f94ca990547991b0d87bcf0d9d
RUN_DIR="$REPO/runs/v_mask_projection_$RUN_ID"
READBACK_DIR="/tmp/v_mask_projection_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/v-mask-projection/$RUN_ID"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$READBACK_DIR"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { printf 'tracked worktree is dirty\n' >&2; exit 2; }
[[ "$(sha256sum "$DOMAIN_LIST" | awk '{print $1}')" == "$DOMAIN_LIST_SHA256" ]] || {
  printf 'domain list SHA256 mismatch\n' >&2
  exit 2
}
[[ -f "$CHECKPOINT" ]] || { printf 'checkpoint missing\n' >&2; exit 2; }
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch\n' >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
df -hT /data
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
"$PYTHON" -c \
  'import sys; from scripts.adjudicate_source_law_candidate import _verify_checkpoint_source_law; _verify_checkpoint_source_law(sys.argv[1])' \
  "$CHECKPOINT"

"$PYTHON" -m pytest -q \
  tests/test_sampling_integrators.py \
  tests/test_rollout_robustness_eval.py \
  tests/test_source_law_adjudication.py \
  tests/test_v_mask_projection_adjudication.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"

run_eval() {
  steps=$1
  variant=$2
  timeout_minutes=25
  [[ "$steps" == 1 ]] && timeout_minutes=8
  output="$RUN_DIR/${variant}_h${steps}.json"
  extra=()
  [[ "$variant" == masked ]] && extra+=(--project-v-atom-mask)
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s "${timeout_minutes}m" \
    "$PYTHON" scripts/rollout_robustness_eval.py \
    --ckpt "$CHECKPOINT" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --domains 1 --starts 5 --steps "$steps" --methods ode_150 \
    --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
    "${extra[@]}" --output "$output" > "$RUN_DIR/${variant}_h${steps}.log" 2>&1
}

run_eval 1 current
run_eval 1 masked
"$PYTHON" -m scripts.adjudicate_v_mask_projection \
  --current "$RUN_DIR/current_h1.json" --masked "$RUN_DIR/masked_h1.json" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" --steps 1 \
  --output "$RUN_DIR/decision_h1.json"
status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision_h1.json")

if [[ "$status" == ADVANCE_MASKED_H6 ]]; then
  run_eval 6 current
  run_eval 6 masked
  "$PYTHON" -m scripts.adjudicate_v_mask_projection \
    --current "$RUN_DIR/current_h6.json" --masked "$RUN_DIR/masked_h6.json" \
    --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
    --domain-list-sha256 "$DOMAIN_LIST_SHA256" --steps 6 \
    --h1-decision "$RUN_DIR/decision_h1.json" \
    --output "$RUN_DIR/decision_final.json"
  status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision_final.json")
else
  cp "$RUN_DIR/decision_h1.json" "$RUN_DIR/decision_final.json"
fi

printf '{"status":"%s","scope":"same_checkpoint_masked_v_inference_only","twenty_domain_authorized":false,"second_seed_authorized":false,"confirmation_authorized":false,"formal_training_authorized":false,"run_id":"%s","commit":"%s","checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$status" "$RUN_ID" "$actual_commit" "$CHECKPOINT_SHA256" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
(
  cd "$RUN_DIR"
  sha256sum current_h1.json current_h1.log masked_h1.json masked_h1.log \
    decision_h1.json decision_final.json pytest.log summary.json
  [[ ! -f current_h6.json ]] || sha256sum current_h6.json current_h6.log masked_h6.json masked_h6.log
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/audit"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(
  cd "$READBACK_DIR/audit"
  sha256sum -c "$RUN_DIR/audit_sha256.txt"
)
printf 'Masked-V discriminator complete with decision=%s; training was not started.\n' "$status"
