#!/usr/bin/env bash
# Run the bounded 20-domain clean-source H1 endpoint development gate.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-75}
HARD_STOP_UNIT="deepjump-dev20-endpoint-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 240s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    set -e
  fi
  printf 'Dev20 endpoint gate exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
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
DOMAIN_LIST=configs/dev_20_length_proportional_seed0.txt
DOMAIN_LIST_SHA256=4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af
RUN_DIR="$REPO/runs/dev20_endpoint_gate_$RUN_ID"
READBACK_DIR="/tmp/dev20_endpoint_gate_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/dev20-endpoint-gate/$RUN_ID"

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
[[ "$(wc -l < "$DOMAIN_LIST" | tr -d ' ')" == 20 ]] || { printf 'domain count != 20\n' >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { printf 'checkpoint missing\n' >&2; exit 2; }
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch\n' >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval|endpoint_grid_eval|endpoint_panel_eval).py'; then
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
"$PYTHON" -c \
  'import pathlib,sys,torch; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False)["cfg"]["data"]; assert d["domains"]==["1a0hA01"]; assert pathlib.Path(d["root"]).resolve()==pathlib.Path(sys.argv[2]).resolve()' \
  "$CHECKPOINT" "$DATA_ROOT"

"$PYTHON" -m pytest -q \
  tests/test_shapes.py \
  tests/test_sampling_integrators.py \
  tests/test_rollout_robustness_eval.py \
  tests/test_source_law_adjudication.py \
  tests/test_endpoint_grid_adjudication.py \
  tests/test_endpoint_panel_adjudication.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"

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

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 55m \
  "$PYTHON" scripts/endpoint_panel_eval.py \
  --ckpt "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --starts 3 --runtime-probe-output "$RUN_DIR/runtime_probe.json" \
  --output "$RUN_DIR/panel.json" > "$RUN_DIR/panel.log" 2>&1

"$PYTHON" -m scripts.adjudicate_endpoint_panel \
  --result "$RUN_DIR/panel.json" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/decision.json"
status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN_DIR/decision.json")

printf '{"status":"%s","scope":"dev20_5x5_clean_endpoint_gate","domains":20,"primary_unseen_domains":19,"cells":500,"starts":1500,"domain_list_sha256":"%s","second_seed_authorized":false,"untouched_confirmation_authorized":false,"recursive_evaluation_authorized":false,"formal_training_authorized":false,"run_id":"%s","commit":"%s","checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$status" "$DOMAIN_LIST_SHA256" "$RUN_ID" "$actual_commit" "$CHECKPOINT_SHA256" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
(
  cd "$RUN_DIR"
  sha256sum runtime_probe.json panel.json panel.log decision.json data_audit.json pytest.log summary.json
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/audit"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(
  cd "$READBACK_DIR/audit"
  sha256sum -c "$RUN_DIR/audit_sha256.txt"
)
printf '{"status":"OBS_READBACK_PASS","run_id":"%s","audit_sha256":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$(sha256sum "$RUN_DIR/audit_sha256.txt" | awk '{print $1}')" "$(date -Is)" \
  > "$RUN_DIR/readback_completion.json"
(
  cd "$RUN_DIR"
  sha256sum readback_completion.json > readback_completion.sha256
)
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/completion"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/completion"
(
  cd "$READBACK_DIR/completion"
  sha256sum -c "$RUN_DIR/readback_completion.sha256"
)
printf 'Dev20 endpoint gate complete with decision=%s; training was not started.\n' "$status"
