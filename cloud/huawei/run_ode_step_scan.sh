#!/usr/bin/env bash
# Run the bounded same-checkpoint H1 ODE step-count discriminator and power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}
HARD_STOP_UNIT="deepjump-ode-step-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  printf 'ODE step scan exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 45 ]] || { printf 'HARD_STOP_MINUTES must be 45\n' >&2; exit 2; }
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
CHECKPOINT=${CHECKPOINT:?set the existing source-law ckpt_1000.pt path}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the frozen checkpoint SHA256}
REFERENCE_ODE150=${REFERENCE_ODE150:?set the frozen current H1 ode_150 result path}
REFERENCE_ODE150_SHA256=${REFERENCE_ODE150_SHA256:?set the frozen current H1 result SHA256}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOMAIN_LIST=configs/tiny_overfit_domain_1a0hA01.txt
DOMAIN_LIST_SHA256=3da0d5c44e5d1a68aa5e99d01acf296725f7e8f94ca990547991b0d87bcf0d9d
METHODS=(mean ode_1 ode_2 ode_5 ode_10 ode_20 ode_40 ode_75 ode_150)
RUN_DIR="$REPO/runs/ode_step_scan_$RUN_ID"
READBACK_DIR="/tmp/ode_step_scan_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/ode-step-scan/$RUN_ID"

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
[[ -f "$REFERENCE_ODE150" ]] || { printf 'reference ode_150 result missing\n' >&2; exit 2; }
[[ "$(sha256sum "$REFERENCE_ODE150" | awk '{print $1}')" == "$REFERENCE_ODE150_SHA256" ]] || {
  printf 'reference ode_150 SHA256 mismatch\n' >&2
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
  tests/test_ode_step_scan_adjudication.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"

result_args=()
for method in "${METHODS[@]}"; do
  output="$RUN_DIR/${method}.json"
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 6m \
    "$PYTHON" scripts/rollout_robustness_eval.py \
    --ckpt "$CHECKPOINT" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --domains 1 --starts 5 --steps 1 --methods "$method" \
    --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
    --output "$output" > "$RUN_DIR/${method}.log" 2>&1
  result_args+=(--result "$method=$output")
done

"$PYTHON" -m scripts.adjudicate_ode_step_scan \
  "${result_args[@]}" \
  --reference-ode150 "$REFERENCE_ODE150" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/decision.json"
status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")

printf '{"status":"%s","scope":"same_checkpoint_h1_ode_step_scan","twenty_domain_authorized":false,"second_seed_authorized":false,"confirmation_authorized":false,"formal_training_authorized":false,"run_id":"%s","commit":"%s","checkpoint_sha256":"%s","reference_ode150_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$status" "$RUN_ID" "$actual_commit" "$CHECKPOINT_SHA256" \
  "$REFERENCE_ODE150_SHA256" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
(
  cd "$RUN_DIR"
  for method in "${METHODS[@]}"; do
    sha256sum "${method}.json" "${method}.log"
  done
  sha256sum decision.json pytest.log summary.json
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/audit"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(
  cd "$READBACK_DIR/audit"
  sha256sum -c "$RUN_DIR/audit_sha256.txt"
)
printf 'ODE step scan complete with decision=%s; training was not started.\n' "$status"
