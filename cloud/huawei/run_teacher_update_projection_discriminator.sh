#!/usr/bin/env bash
# Run the frozen scalar step-2000 teacher-update projection discriminator without training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
RUN_ROOT=${RUN_ROOT:-/data/deepjump-runs}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}
HARD_STOP_UNIT="deepjump-teacher-projection-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
HARD_STOP_EVIDENCE="/tmp/${HARD_STOP_UNIT}.evidence.log"
OBS_PREFIX_EMPTY_VERIFIED=0

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 && "$OBS_PREFIX_EMPTY_VERIFIED" == 1 && \
    -n "${RUN_DIR:-}" && -d "${RUN_DIR:-}" && -n "${OBS_DST:-}" ]]; then
    set +e
    timeout --signal=TERM --kill-after=30s 2m obsutil sync "$RUN_DIR" "$OBS_DST/failure/audit"
    set -e
  fi
  printf 'Teacher projection discriminator exit=%s; requesting shutdown at %s\n' \
    "$code" "$(date -Is)"
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
{
  sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
    --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
  sudo -n systemctl is-active "$HARD_STOP_UNIT.timer"
  sudo -n systemctl show "$HARD_STOP_UNIT.timer" \
    -p ActiveState -p SubState -p TriggerUSec
  sudo -n systemctl cat "$HARD_STOP_UNIT.service" \
    | grep -F 'ExecStart="/usr/bin/systemctl" "poweroff"'
} | tee "$HARD_STOP_EVIDENCE"
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
CHECKPOINT=${CHECKPOINT:?set frozen scalar step2000 checkpoint path}
H20_RESULT=${H20_RESULT:?set independently read-back H20 result.json path}
H20_DECISION=${H20_DECISION:?set independently read-back H20 decision.json path}
H20_COMPLETION=${H20_COMPLETION:?set independently read-back H20 completion path}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

CHECKPOINT_SHA256=fc5f1e7b5188af4911e518ac0e3d44c2aba4a22431360bde704465c9c1889a73
H20_RESULT_SHA256=bacf07bdd93119a0b793b67335a520c468c5749d5d9da71d887d5a5fe8aa7753
H20_DECISION_SHA256=03a953b4bda5e45391f7a06311eceeb84485a1a4b4a54f01edcd8aa7aea2609d
H20_COMPLETION_SHA256=70a84d0e6e1bb4491ce51d89bcaf7fccab090ded2dc0415a267792e659794512
DOMAIN_LIST=configs/dev_20_length_proportional_seed0.txt
DOMAIN_LIST_SHA256=4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af
RUN_DIR="$RUN_ROOT/teacher_update_projection_$RUN_ID"
RUNNER_LOG="$RUN_ROOT/teacher_update_projection_$RUN_ID.launcher.log"
READBACK_ONE="/tmp/teacher_update_projection_readback_one_$RUN_ID"
READBACK_TWO="/tmp/teacher_update_projection_readback_two_$RUN_ID"
FINAL_READBACK_ONE="/tmp/teacher_update_projection_final_one_$RUN_ID"
FINAL_READBACK_TWO="/tmp/teacher_update_projection_final_two_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/teacher-update-projection/$RUN_ID"

[[ "$BUCKET" == "obs://deepjump-mdcath-cn4-ringochen" ]] || {
  printf 'unexpected OBS bucket\n' >&2; exit 2;
}
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  printf 'RUN_ID must be UTC basic timestamp YYYYMMDDTHHMMSSZ\n' >&2; exit 2;
}
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$RUNNER_LOG" "$READBACK_ONE" "$READBACK_TWO" \
  "$FINAL_READBACK_ONE" "$FINAL_READBACK_TWO"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
cp "$HARD_STOP_EVIDENCE" "$RUN_DIR/hard_stop_evidence.log"
exec 3>&1 4>&2
exec > >(tee -a "$RUNNER_LOG") 2>&1
TEE_PID=$!

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { printf 'worktree is dirty\n' >&2; exit 2; }
for spec in \
  "$CHECKPOINT:$CHECKPOINT_SHA256:checkpoint" \
  "$H20_RESULT:$H20_RESULT_SHA256:H20_result" \
  "$H20_DECISION:$H20_DECISION_SHA256:H20_decision" \
  "$H20_COMPLETION:$H20_COMPLETION_SHA256:H20_completion"; do
  path=${spec%%:*}
  rest=${spec#*:}
  expected=${rest%%:*}
  label=${rest##*:}
  [[ -f "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
    printf '%s SHA256 mismatch\n' "$label" >&2; exit 2;
  }
done
[[ "$(sha256sum "$DOMAIN_LIST" | awk '{print $1}')" == "$DOMAIN_LIST_SHA256" ]] || {
  printf 'domain list SHA256 mismatch\n' >&2; exit 2;
}
[[ "$(wc -l < "$DOMAIN_LIST" | tr -d ' ')" == 20 ]] || {
  printf 'frozen dev panel count mismatch\n' >&2; exit 2;
}
mapfile -t domain_ids < "$DOMAIN_LIST"
[[ "${domain_ids[0]}" == 1gxlA02 && "${domain_ids[9]}" == 2dgmA02 && \
  "${domain_ids[19]}" == 4i9cA01 ]] || {
  printf 'frozen spread-domain identities mismatch\n' >&2; exit 2;
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
printf 'Verified hostname=%s commit=%s clean_tree=true checkpoint=%s H20=%s domains=%s\n' \
  "$(hostname)" "$actual_commit" "$CHECKPOINT_SHA256" "$H20_DECISION_SHA256" \
  '1gxlA02,2dgmA02,4i9cA01'
timeout 30s obsutil ls "$OBS_DST/" -limit=1 | tee "$RUN_DIR/obs_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py "$RUN_DIR/obs_prefix_preflight.log"
OBS_PREFIX_EMPTY_VERIFIED=1
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval|teacher_update_projection_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
df -hT /data
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'

# Maximum declared success-path timeout envelope is 40.5 minutes:
# preflight 0.5 + pytest 4 + evaluation 24 + six OBS transfers at 2 minutes.
timeout --signal=TERM --kill-after=30s 4m "$PYTHON" -m pytest -q \
  tests/test_teacher_update_projection.py \
  tests/test_rollout_robustness_eval.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"

cp "$H20_RESULT" "$RUN_DIR/h20_result.json"
cp "$H20_DECISION" "$RUN_DIR/h20_decision.json"
cp "$H20_COMPLETION" "$RUN_DIR/h20_readback_completion.json"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 24m \
  "$PYTHON" scripts/teacher_update_projection_eval.py \
  --ckpt "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 3 --starts 2 --steps 20 --calibration-domain 1gxlA02 \
  --output "$RUN_DIR/result.json" > "$RUN_DIR/result.log" 2>&1

"$PYTHON" -m scripts.adjudicate_teacher_update_projection \
  --result "$RUN_DIR/result.json" --checkpoint "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" --h20-result "$RUN_DIR/h20_result.json" \
  --h20-decision "$RUN_DIR/h20_decision.json" \
  --h20-completion "$RUN_DIR/h20_readback_completion.json" \
  --output "$RUN_DIR/decision.json"

"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$actual_commit" "$CHECKPOINT_SHA256" "$OBS_DST" <<'PY'
import datetime, json, sys
decision = json.load(open(sys.argv[1]))
summary = {
    "status": decision["status"],
    "scope": decision["scope"],
    "h20_result_sha256": decision["h20_result_sha256"],
    "h20_decision_sha256": decision["h20_decision_sha256"],
    "h20_completion_sha256": decision["h20_completion_sha256"],
    "external_development_authorized": False,
    "second_seed_authorized": False,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "run_id": sys.argv[3],
    "deployed_commit": sys.argv[4],
    "checkpoint_sha256": sys.argv[5],
    "obs": sys.argv[6],
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(sys.argv[2], "w").write(json.dumps(summary, separators=(",", ":")) + "\n")
print(json.dumps(summary, indent=2))
PY

printf 'Freezing runtime evidence before exact OBS archive at %s\n' "$(date -Is)"
exec 1>&3 2>&4
wait "$TEE_PID"
exec 3>&- 4>&-
cp "$RUNNER_LOG" "$RUN_DIR/runtime_evidence.log"
exec > >(tee -a "$RUNNER_LOG") 2>&1
(
  cd "$RUN_DIR"
  sha256sum hard_stop_evidence.log obs_prefix_preflight.log pytest.log \
    runtime_evidence.log h20_result.json h20_decision.json \
    h20_readback_completion.json result.json result.log decision.json summary.json
) > "$RUN_DIR/audit_sha256.txt"
timeout --signal=TERM --kill-after=30s 2m obsutil sync "$RUN_DIR" "$OBS_DST/audit"

verify_readback() {
  target=$1
  phase=$2
  report=${3:-}
  mkdir -p "$target/audit"
  timeout --signal=TERM --kill-after=30s 2m obsutil sync "$OBS_DST/audit" "$target/audit"
  command=("$PYTHON" scripts/verify_teacher_update_projection_readback.py
    --root "$target/audit" --checkpoint "$CHECKPOINT"
    --domain-list "$DOMAIN_LIST" --phase "$phase"
    --expected-run-id "$RUN_ID" --expected-deployed-commit "$actual_commit"
    --expected-obs "$OBS_DST")
  if [[ -n "$report" ]]; then
    "${command[@]}" | tee "$report"
  else
    "${command[@]}"
  fi
}

verify_readback "$READBACK_ONE" initial "$RUN_DIR/initial_readback_one.json"
verify_readback "$READBACK_TWO" initial "$RUN_DIR/initial_readback_two.json"
"$PYTHON" scripts/verify_teacher_update_projection_readback.py \
  --root "$READBACK_ONE/audit" --root-two "$READBACK_TWO/audit" \
  --checkpoint "$CHECKPOINT" --domain-list "$DOMAIN_LIST" --phase initial \
  --expected-run-id "$RUN_ID" --expected-deployed-commit "$actual_commit" \
  --expected-obs "$OBS_DST" | tee "$RUN_DIR/initial_readback_pair.json"

"$PYTHON" - "$RUN_DIR/readback_completion.json" "$RUN_ID" "$actual_commit" \
  "$RUN_DIR/audit_sha256.txt" "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" \
  "$RUN_DIR/initial_readback_one.json" "$RUN_DIR/initial_readback_two.json" \
  "$RUN_DIR/initial_readback_pair.json" <<'PY'
import datetime, hashlib, json, sys
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
decision = json.load(open(sys.argv[5]))
completion = {
    "status": "OBS_DOUBLE_READBACK_PASS",
    "decision_status": decision["status"],
    "run_id": sys.argv[2],
    "commit": sys.argv[3],
    "audit_manifest_sha256": sha(sys.argv[4]),
    "archived_decision_sha256": sha(sys.argv[5]),
    "archived_summary_sha256": sha(sys.argv[6]),
    "recomputed_decision_sha256": sha(sys.argv[5]),
    "initial_readback_one_sha256": sha(sys.argv[7]),
    "initial_readback_two_sha256": sha(sys.argv[8]),
    "initial_readback_pair_sha256": sha(sys.argv[9]),
    "independent_readbacks_verified": 2,
    "external_development_authorized": False,
    "second_seed_authorized": False,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(sys.argv[1], "w").write(json.dumps(completion, separators=(",", ":")) + "\n")
PY
(
  cd "$RUN_DIR"
  sha256sum initial_readback_one.json initial_readback_two.json \
    initial_readback_pair.json readback_completion.json
) > "$RUN_DIR/completion_sha256.txt"
timeout --signal=TERM --kill-after=30s 2m obsutil sync "$RUN_DIR" "$OBS_DST/audit"

verify_readback "$FINAL_READBACK_ONE" completion
verify_readback "$FINAL_READBACK_TWO" completion
"$PYTHON" scripts/verify_teacher_update_projection_readback.py \
  --root "$FINAL_READBACK_ONE/audit" --root-two "$FINAL_READBACK_TWO/audit" \
  --checkpoint "$CHECKPOINT" --domain-list "$DOMAIN_LIST" --phase completion \
  --expected-run-id "$RUN_ID" --expected-deployed-commit "$actual_commit" \
  --expected-obs "$OBS_DST"
printf 'Teacher update projection discriminator complete; training was not started.\n'
