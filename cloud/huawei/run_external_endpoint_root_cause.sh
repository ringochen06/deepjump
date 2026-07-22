#!/usr/bin/env bash
# Run the frozen no-training external endpoint root-cause discriminator.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TRAINING_DATA_ROOT=${TRAINING_DATA_ROOT:-/data/mdcath}
EXTERNAL_DATA_ROOT=${EXTERNAL_DATA_ROOT:-/data/mdcath_external20}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-75}
HARD_STOP_UNIT="deepjump-external-root-cause-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 240s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    set -e
  fi
  printf 'External endpoint root-cause exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
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
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" --on-active=75m /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
CHECKPOINT=${CHECKPOINT:?set the frozen multidomain ckpt_1000.pt path}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the frozen checkpoint SHA256}
REFERENCE_PANEL=${REFERENCE_PANEL:?set the frozen external panel.json path}
REFERENCE_PANEL_SHA256=6b904ca244242987e28dcc3598a8ad877501f45e9ca4acba26a3e6dedc683b25
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
TRAINING_DOMAIN_LIST=configs/subset_1000_length_proportional.txt
TRAINING_DOMAIN_LIST_SHA256=39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734
EXTERNAL_DOMAIN_LIST=configs/external_dev_20_length_proportional_seed20260721.txt
EXTERNAL_DOMAIN_LIST_SHA256=9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245
CONTEXT_DOMAIN_LIST=configs/external_context_9_root_cause.txt
CONTEXT_DOMAIN_LIST_SHA256=7ec4af135d80c94764099c201ed1e3283f8bf17579fba34aa208e477bb484573
RUN_DIR="$REPO/runs/external_endpoint_root_cause_$RUN_ID"
READBACK_DIR="/tmp/external_endpoint_root_cause_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/external-endpoint-root-cause/$RUN_ID"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$READBACK_DIR"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

[[ "$(git rev-parse HEAD)" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { printf 'tracked worktree is dirty\n' >&2; exit 2; }
[[ "$(sha256sum "$TRAINING_DOMAIN_LIST" | awk '{print $1}')" == "$TRAINING_DOMAIN_LIST_SHA256" ]]
[[ "$(sha256sum "$EXTERNAL_DOMAIN_LIST" | awk '{print $1}')" == "$EXTERNAL_DOMAIN_LIST_SHA256" ]]
[[ "$(sha256sum "$CONTEXT_DOMAIN_LIST" | awk '{print $1}')" == "$CONTEXT_DOMAIN_LIST_SHA256" ]]
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA256" ]]
[[ "$(sha256sum "$REFERENCE_PANEL" | awk '{print $1}')" == "$REFERENCE_PANEL_SHA256" ]]
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
if pgrep -af '[s]cripts/(train_ddp|external_endpoint_panel_eval|external_endpoint_root_cause).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
df -hT /data
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
"$PYTHON" -m pytest -q \
  tests/test_external_endpoint_root_cause.py \
  tests/test_external_endpoint_panel_adjudication.py \
  tests/test_tensor_cloud01.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"

"$PYTHON" scripts/audit_mdcath_staging.py \
  --root "$TRAINING_DATA_ROOT" --expected-h5 1000 --expected-bytes 668131379559 \
  --expected-subset-sha256 "$TRAINING_DOMAIN_LIST_SHA256" --expected-trajectories 25000 \
  --expected-strategy length-proportional --expected-seed 20260715 --expected-commit 3f9f7f7 \
  --samples 5 | tee "$RUN_DIR/training_data_audit.json"
"$PYTHON" scripts/audit_external_mdcath.py \
  --root "$EXTERNAL_DATA_ROOT" --domain-list "$EXTERNAL_DOMAIN_LIST" \
  --domain-list-sha256 "$EXTERNAL_DOMAIN_LIST_SHA256" --expected-bytes 13778143616 \
  --output "$RUN_DIR/external_data_audit.json"

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 45m \
  "$PYTHON" scripts/external_endpoint_root_cause.py \
  --ckpt "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --training-data-root "$TRAINING_DATA_ROOT" \
  --training-domain-list "$TRAINING_DOMAIN_LIST" \
  --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
  --external-data-root "$EXTERNAL_DATA_ROOT" \
  --external-domain-list "$EXTERNAL_DOMAIN_LIST" \
  --external-domain-list-sha256 "$EXTERNAL_DOMAIN_LIST_SHA256" \
  --context-domain-list "$CONTEXT_DOMAIN_LIST" \
  --context-domain-list-sha256 "$CONTEXT_DOMAIN_LIST_SHA256" \
  --reference-panel "$REFERENCE_PANEL" --reference-panel-sha256 "$REFERENCE_PANEL_SHA256" \
  --output "$RUN_DIR/root_cause.json" > "$RUN_DIR/root_cause.log" 2>&1

"$PYTHON" -m scripts.adjudicate_external_endpoint_root_cause \
  --result "$RUN_DIR/root_cause.json" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --reference-panel "$REFERENCE_PANEL" --reference-panel-sha256 "$REFERENCE_PANEL_SHA256" \
  --output "$RUN_DIR/decision.json"
"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$EXPECTED_REPO_COMMIT" "$CHECKPOINT_SHA256" "$OBS_DST" <<'PY'
import datetime, json, sys
decision = json.load(open(sys.argv[1]))
summary = {
    "status": decision["status"],
    "context_status": decision["context_status"],
    "outlier_status": decision["outlier_status"],
    "formal_training_authorized": False,
    "run_id": sys.argv[3],
    "commit": sys.argv[4],
    "checkpoint_sha256": sys.argv[5],
    "obs": sys.argv[6],
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(sys.argv[2], "w").write(json.dumps(summary, separators=(",", ":")) + "\n")
print(json.dumps(summary, indent=2))
PY
(
  cd "$RUN_DIR"
  sha256sum root_cause.json root_cause.log decision.json training_data_audit.json \
    external_data_audit.json pytest.log summary.json
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/audit"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/audit"
(cd "$READBACK_DIR/audit" && sha256sum -c "$RUN_DIR/audit_sha256.txt")
printf '{"status":"OBS_READBACK_PASS","run_id":"%s","audit_sha256":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$(sha256sum "$RUN_DIR/audit_sha256.txt" | awk '{print $1}')" "$(date -Is)" \
  > "$RUN_DIR/readback_completion.json"
(cd "$RUN_DIR" && sha256sum readback_completion.json > readback_completion.sha256)
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/completion"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$READBACK_DIR/completion"
(cd "$READBACK_DIR/completion" && sha256sum -c "$RUN_DIR/readback_completion.sha256")
printf 'Root-cause probe complete; no training was started.\n'
