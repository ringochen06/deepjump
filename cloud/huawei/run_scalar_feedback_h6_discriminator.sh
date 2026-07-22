#!/usr/bin/env bash
# Run the frozen scalar step-2000 H1-H6 feedback discriminator without training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
RUN_ROOT=${RUN_ROOT:-/data/deepjump-runs}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}
HARD_STOP_UNIT="deepjump-scalar-feedback-h6-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
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
  printf 'Scalar feedback H6 discriminator exit=%s; requesting shutdown at %s\n' \
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
    | grep -F 'ExecStart=/usr/bin/systemctl poweroff'
} | tee "$HARD_STOP_EVIDENCE"
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
CHECKPOINT=${CHECKPOINT:?set the frozen scalar step2000 checkpoint path}
SOURCE_TRAINING_EVIDENCE=${SOURCE_TRAINING_EVIDENCE:?set the sealed scalar training_evidence.json path}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

CHECKPOINT_SHA256=fc5f1e7b5188af4911e518ac0e3d44c2aba4a22431360bde704465c9c1889a73
SOURCE_EVIDENCE_SHA256=63c4e1027bd03722ab335bd61cff458aa7d3c562ee6adee22c75bbff461691da
SOURCE_COMMIT=9af7125cab0badd3b4e3ef94de37d8a996d4c532
DOMAIN_LIST=configs/dev_20_length_proportional_seed0.txt
DOMAIN_LIST_SHA256=4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af
RUN_DIR="$RUN_ROOT/scalar_feedback_h6_$RUN_ID"
RUNNER_LOG="$RUN_ROOT/scalar_feedback_h6_$RUN_ID.launcher.log"
READBACK_ONE="/tmp/scalar_feedback_h6_readback_one_$RUN_ID"
READBACK_TWO="/tmp/scalar_feedback_h6_readback_two_$RUN_ID"
FINAL_READBACK_ONE="/tmp/scalar_feedback_h6_final_one_$RUN_ID"
FINAL_READBACK_TWO="/tmp/scalar_feedback_h6_final_two_$RUN_ID"
OBS_DST="$BUCKET/deepjump-diagnostics/scalar-feedback-h6/$RUN_ID"

[[ "$BUCKET" == "obs://deepjump-mdcath-cn4-ringochen" ]] || {
  printf 'unexpected OBS bucket\n' >&2; exit 2;
}
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  printf 'RUN_ID must be UTC basic timestamp YYYYMMDDTHHMMSSZ\n' >&2; exit 2;
}
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$READBACK_ONE" "$READBACK_TWO" \
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
[[ -f "$CHECKPOINT" && "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch\n' >&2; exit 2;
}
[[ -f "$SOURCE_TRAINING_EVIDENCE" && \
  "$(sha256sum "$SOURCE_TRAINING_EVIDENCE" | awk '{print $1}')" == "$SOURCE_EVIDENCE_SHA256" ]] || {
  printf 'source training evidence SHA256 mismatch\n' >&2; exit 2;
}
[[ "$(sha256sum "$DOMAIN_LIST" | awk '{print $1}')" == "$DOMAIN_LIST_SHA256" ]] || {
  printf 'domain list SHA256 mismatch\n' >&2; exit 2;
}
[[ "$(head -n 1 "$DOMAIN_LIST")" == 1gxlA02 ]] || {
  printf 'frozen selected domain mismatch\n' >&2; exit 2;
}
[[ "$(wc -l < "$DOMAIN_LIST" | tr -d ' ')" == 20 ]] || {
  printf 'frozen dev panel count mismatch\n' >&2; exit 2;
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
printf 'Verified hostname=%s commit=%s clean_tree=true checkpoint=%s source_commit=%s domain=%s\n' \
  "$(hostname)" "$actual_commit" "$CHECKPOINT_SHA256" "$SOURCE_COMMIT" 1gxlA02
timeout 30s obsutil ls "$OBS_DST/" -limit=1 | tee "$RUN_DIR/obs_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py "$RUN_DIR/obs_prefix_preflight.log"
OBS_PREFIX_EMPTY_VERIFIED=1
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
df -hT /data
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
"$PYTHON" - "$SOURCE_TRAINING_EVIDENCE" "$SOURCE_COMMIT" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload["commit"] == sys.argv[2]
assert payload["candidate_checkpoint_sha256"] == "fc5f1e7b5188af4911e518ac0e3d44c2aba4a22431360bde704465c9c1889a73"
PY

# Maximum declared success-path timeout envelope is 31.5 minutes:
# preflight 0.5 + pytest 4 + evaluation 15 + six OBS transfers at 2 minutes.
timeout --signal=TERM --kill-after=30s 4m "$PYTHON" -m pytest -q \
  tests/test_rollout_robustness_eval.py \
  tests/test_scalar_feedback_h6_adjudication.py \
  tests/test_training_gates.py | tee "$RUN_DIR/pytest.log"

cp "$SOURCE_TRAINING_EVIDENCE" "$RUN_DIR/source_training_evidence.json"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 15m \
  "$PYTHON" scripts/rollout_robustness_eval.py \
  --ckpt "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 1 --starts 5 --steps 6 --methods mean --teacher-forced-mean \
  --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
  --output "$RUN_DIR/result.json" > "$RUN_DIR/result.log" 2>&1

"$PYTHON" -m scripts.adjudicate_scalar_feedback_h6 \
  --result "$RUN_DIR/result.json" \
  --checkpoint "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" \
  --source-evidence "$RUN_DIR/source_training_evidence.json" \
  --output "$RUN_DIR/decision.json"

"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$actual_commit" "$CHECKPOINT_SHA256" "$SOURCE_COMMIT" "$OBS_DST" <<'PY'
import datetime, json, sys
decision = json.load(open(sys.argv[1]))
summary = {
    "status": decision["status"],
    "scope": decision["scope"],
    "external_development_authorized": False,
    "second_seed_authorized": False,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "run_id": sys.argv[3],
    "deployed_commit": sys.argv[4],
    "checkpoint_sha256": sys.argv[5],
    "checkpoint_source_commit": sys.argv[6],
    "obs": sys.argv[7],
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
  sha256sum hard_stop_evidence.log obs_prefix_preflight.log pytest.log runtime_evidence.log \
    source_training_evidence.json \
    result.json result.log decision.json summary.json
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 2m obsutil sync "$RUN_DIR" "$OBS_DST/audit"

verify_readback() {
  target=$1
  phase=$2
  report=${3:-}
  mkdir -p "$target/audit"
  timeout --signal=TERM --kill-after=30s 2m obsutil sync "$OBS_DST/audit" "$target/audit"
  command=("$PYTHON" scripts/verify_scalar_feedback_h6_readback.py
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
"$PYTHON" scripts/verify_scalar_feedback_h6_readback.py \
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
"$PYTHON" scripts/verify_scalar_feedback_h6_readback.py \
  --root "$FINAL_READBACK_ONE/audit" --root-two "$FINAL_READBACK_TWO/audit" \
  --checkpoint "$CHECKPOINT" --domain-list "$DOMAIN_LIST" --phase completion \
  --expected-run-id "$RUN_ID" --expected-deployed-commit "$actual_commit" \
  --expected-obs "$OBS_DST"
printf 'Scalar feedback H1-H6 discriminator complete; training was not started.\n'
