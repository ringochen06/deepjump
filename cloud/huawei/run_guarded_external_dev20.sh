#!/usr/bin/env bash
# Download, audit, and evaluate the one frozen fresh external guarded panel.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TRAINING_DATA_ROOT=${TRAINING_DATA_ROOT:-/data/mdcath}
EXTERNAL_DATA_ROOT=${EXTERNAL_DATA_ROOT:-/data/mdcath_guarded_external20_seed20260722}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-150}
HARD_STOP_UNIT="deepjump-guarded-external-dev20-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 240s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    set -e
  fi
  printf 'Guarded external-dev20 exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || [[ "$code" != 0 ]] || code=$?
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 150 ]] || { printf 'HARD_STOP_MINUTES must be 150\n' >&2; exit 2; }
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
CHECKPOINT=${CHECKPOINT:?set frozen full-tensor step2000 checkpoint path}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set frozen checkpoint SHA256}
PREREQUISITE_DECISION=${PREREQUISITE_DECISION:?set guarded training-dev20 decision path}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
TRAINING_DOMAIN_LIST=configs/subset_1000_length_proportional.txt
TRAINING_DOMAIN_LIST_SHA256=39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734
PRIOR_EXTERNAL_DOMAIN_LIST=configs/external_dev_20_length_proportional_seed20260721.txt
PRIOR_EXTERNAL_DOMAIN_LIST_SHA256=9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245
UNTOUCHED_DOMAIN_LIST=configs/confirmation_100_length_proportional_seed20260717.txt
UNTOUCHED_DOMAIN_LIST_SHA256=e56ed7de735db542f4e20fb73f2654a6c1bcf67f3082849f63f0ab74f4208c38
EXTERNAL_DOMAIN_LIST=configs/guarded_external_dev_20_length_proportional_seed20260722.txt
EXTERNAL_DOMAIN_LIST_SHA256=9bae11fa0e6336e7451c372efa25ca55af77aa9cb27f91e1fd241612531a920f
EXTERNAL_EXPECTED_BYTES=13354825648
EXPECTED_CHECKPOINT_SHA256=f3b5965303794e14059f2b67b6b81a538fadb1303c44e1d7c640af44ea690222
PREREQUISITE_DECISION_SHA256=b234f31db96c2f461ea0abd056aa6e724d2d94aa52930bbc990c43cfc302000b
RUN_DIR="$REPO/runs/guarded_external_dev20_$RUN_ID"
READBACK_DIR="/tmp/guarded_external_dev20_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/guarded-external-dev20/$RUN_ID"

[[ "$CHECKPOINT_SHA256" == "$EXPECTED_CHECKPOINT_SHA256" ]]
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
for pair in \
  "$TRAINING_DOMAIN_LIST:$TRAINING_DOMAIN_LIST_SHA256" \
  "$PRIOR_EXTERNAL_DOMAIN_LIST:$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
  "$UNTOUCHED_DOMAIN_LIST:$UNTOUCHED_DOMAIN_LIST_SHA256" \
  "$EXTERNAL_DOMAIN_LIST:$EXTERNAL_DOMAIN_LIST_SHA256" \
  "$CHECKPOINT:$CHECKPOINT_SHA256" \
  "$PREREQUISITE_DECISION:$PREREQUISITE_DECISION_SHA256"; do
  path=${pair%%:*}; digest=${pair##*:}
  [[ -f "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]]
done
[[ "$(wc -l < "$EXTERNAL_DOMAIN_LIST" | tr -d ' ')" == 20 ]]
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" ]]; command -v obsutil >/dev/null
if pgrep -af '[s]cripts/(train_ddp|guarded_endpoint_panel_eval|external_endpoint_panel_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2; exit 2
fi
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
required_bytes=$((EXTERNAL_EXPECTED_BYTES + 10 * 1024 * 1024 * 1024))
[[ "$available_bytes" -ge "$required_bytes" ]] || { printf 'insufficient /data free space\n' >&2; exit 2; }

"$PYTHON" -m pytest -q \
  tests/test_select_subset.py tests/test_guarded_sampling.py \
  tests/test_guarded_endpoint_panel_adjudication.py \
  tests/test_external_endpoint_panel_adjudication.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"
"$PYTHON" scripts/audit_mdcath_staging.py \
  --root "$TRAINING_DATA_ROOT" --expected-h5 1000 --expected-bytes 668131379559 \
  --expected-subset-sha256 "$TRAINING_DOMAIN_LIST_SHA256" --expected-trajectories 25000 \
  --expected-strategy length-proportional --expected-seed 20260715 \
  --expected-commit 3f9f7f7 --samples 5 | tee "$RUN_DIR/training_data_audit.json"

mkdir -p "$EXTERNAL_DATA_ROOT"
timeout --signal=TERM --kill-after=30s 60m "$PYTHON" scripts/download_mdcath.py \
  --root "$EXTERNAL_DATA_ROOT" --domains-file "$EXTERNAL_DOMAIN_LIST" --retries 5 \
  | tee "$RUN_DIR/download.log"
"$PYTHON" scripts/audit_external_mdcath.py \
  --root "$EXTERNAL_DATA_ROOT" --domain-list "$EXTERNAL_DOMAIN_LIST" \
  --domain-list-sha256 "$EXTERNAL_DOMAIN_LIST_SHA256" \
  --expected-bytes "$EXTERNAL_EXPECTED_BYTES" --output "$RUN_DIR/external_data_audit.json"

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 55m \
  "$PYTHON" scripts/guarded_endpoint_panel_eval.py \
  --ckpt "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --training-data-root "$TRAINING_DATA_ROOT" \
  --training-domain-list "$TRAINING_DOMAIN_LIST" \
  --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
  --panel-kind fresh-external --panel-data-root "$EXTERNAL_DATA_ROOT" \
  --prior-external-domain-list "$PRIOR_EXTERNAL_DOMAIN_LIST" \
  --prior-external-domain-list-sha256 "$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
  --untouched-domain-list "$UNTOUCHED_DOMAIN_LIST" \
  --untouched-domain-list-sha256 "$UNTOUCHED_DOMAIN_LIST_SHA256" \
  --prerequisite-decision "$PREREQUISITE_DECISION" \
  --prerequisite-decision-sha256 "$PREREQUISITE_DECISION_SHA256" \
  --domain-list "$EXTERNAL_DOMAIN_LIST" --domain-list-sha256 "$EXTERNAL_DOMAIN_LIST_SHA256" \
  --runtime-probe-output "$RUN_DIR/runtime_probe.json" \
  --output "$RUN_DIR/panel.json" > "$RUN_DIR/panel.log" 2>&1

"$PYTHON" -m scripts.adjudicate_guarded_endpoint_panel \
  --result "$RUN_DIR/panel.json" --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --training-domain-list "$TRAINING_DOMAIN_LIST" \
  --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
  --panel-kind fresh-external \
  --prior-external-domain-list "$PRIOR_EXTERNAL_DOMAIN_LIST" \
  --prior-external-domain-list-sha256 "$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
  --untouched-domain-list "$UNTOUCHED_DOMAIN_LIST" \
  --untouched-domain-list-sha256 "$UNTOUCHED_DOMAIN_LIST_SHA256" \
  --prerequisite-decision "$PREREQUISITE_DECISION" \
  --prerequisite-decision-sha256 "$PREREQUISITE_DECISION_SHA256" \
  --domain-list "$EXTERNAL_DOMAIN_LIST" --domain-list-sha256 "$EXTERNAL_DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/decision.json"
"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$actual_commit" "$CHECKPOINT_SHA256" "$OBS_DST" <<'PY'
import datetime, json, sys
decision = json.load(open(sys.argv[1]))
summary = {
    "status": decision["status"],
    "second_seed_authorized": decision["second_seed_authorized"],
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "run_id": sys.argv[3], "commit": sys.argv[4],
    "checkpoint_sha256": sys.argv[5], "obs": sys.argv[6],
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(sys.argv[2], "w").write(json.dumps(summary, separators=(",", ":")) + "\n")
print(json.dumps(summary, indent=2))
PY
(
  cd "$RUN_DIR"
  sha256sum runtime_probe.json panel.json panel.log decision.json training_data_audit.json \
    external_data_audit.json download.log pytest.log summary.json
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
printf 'Guarded external-dev20 complete; no training was started.\n'
