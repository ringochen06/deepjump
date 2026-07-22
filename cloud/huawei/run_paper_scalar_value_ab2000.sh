#!/usr/bin/env bash
# Run the preregistered fresh scalar-value architecture arm against the sealed
# vector-only seed-0 control. This runner never consumes external or untouched
# data and always powers off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-420}
HARD_STOP_UNIT="deepjump-paper-scalar-value-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
OBS_DST_OWNED=0

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 && "$OBS_DST_OWNED" == 1 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 300s obsutil sync \
      "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -d "${CANDIDATE_DIR:-}" ]] && timeout 300s obsutil sync \
      "$CANDIDATE_DIR" "${OBS_DST:-}/failure/candidate"
    set -e
  fi
  printf 'Paper scalar-value A/B exit=%s; requesting shutdown at %s\n' \
    "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || {
  printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2;
}
[[ "$HARD_STOP_MINUTES" == 420 ]] || {
  printf 'HARD_STOP_MINUTES must be 420\n' >&2; exit 2;
}
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n systemctl show "$HARD_STOP_UNIT.timer" \
  --property=ActiveState,SubState,NextElapseUSecRealtime --no-pager
sudo -n systemctl show "$HARD_STOP_UNIT.service" --property=ExecStart --no-pager \
  | grep -Fq '/usr/bin/systemctl poweroff'
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$BUCKET" == "obs://deepjump-mdcath-cn4-ringochen" ]] || {
  printf 'unexpected OBS bucket\n' >&2; exit 2;
}
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  printf 'RUN_ID must be UTC basic timestamp YYYYMMDDTHHMMSSZ\n' >&2; exit 2;
}

BASELINE_SOURCE_RUN_ID=20260722T051048Z
BASELINE_SOURCE_COMMIT=fd92112c9ab7c3e941138a95b136f51c29558353
BASELINE_SOURCE_OBS="$BUCKET/deepjump-calibration/paper-vector-ab2000/$BASELINE_SOURCE_RUN_ID"
BASELINE_CHECKPOINT_SHA256=19d960826938419e1bf494701a09b395ece729e1c0dc2c8a5d1e6bf36d73053b
BASELINE_HISTORY_SHA256=36f8850ba4e9c094526850370b22371d10df76765eead3e39adf051e68d0d80e
BASELINE_DECISION_SHA256=0816f94b01bf8b434086677d59c913193a70aa8b802f79b46378590f772af7bf
BASELINE_FINAL_DECISION_SHA256=1ceb092102c4c0ad608289a19d924a60e7f55df4fe226a21f8fd27895ab1bac6
BASELINE_FINAL_STATUS=STOP_PAPER_VECTOR_ABSOLUTE_GATE
CANDIDATE_CONFIG=configs/v100_tensorcloud01_vector_scalar_value_d1_fp32_paper_horizon500k_2000.yaml
CANDIDATE_CONFIG_SHA256=2b6d96b647d386fe942bcfdc85dac29f6428b6fe685ce5b10a3f9117cdc48832
TRAINING_DOMAIN_LIST=configs/subset_1000_length_proportional.txt
TRAINING_DOMAIN_LIST_SHA256=39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734
DOMAIN_LIST=configs/dev_20_length_proportional_seed0.txt
DOMAIN_LIST_SHA256=4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af

RUN_DIR="$REPO/runs/paper_scalar_value_ab2000_$RUN_ID"
CANDIDATE_DIR="$REPO/runs/v100_tensorcloud01_vector_scalar_value_d1_fp32_paper_horizon500k_2000"
SOURCE_READBACK="/tmp/paper_scalar_value_source_$RUN_ID"
BASELINE_DIR="$SOURCE_READBACK/candidate"
READBACK_ONE="/tmp/paper_scalar_value_readback_one_$RUN_ID"
READBACK_TWO="/tmp/paper_scalar_value_readback_two_$RUN_ID"
FINAL_READBACK_ONE="/tmp/paper_scalar_value_final_one_$RUN_ID"
FINAL_READBACK_TWO="/tmp/paper_scalar_value_final_two_$RUN_ID"
OBS_DST="$BUCKET/deepjump-calibration/paper-scalar-value-ab2000/$RUN_ID"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  printf 'hostname mismatch\n' >&2; exit 2;
}
cd "$REPO"
for path in "$RUN_DIR" "$CANDIDATE_DIR" "$SOURCE_READBACK" \
  "$READBACK_ONE" "$READBACK_TWO" "$FINAL_READBACK_ONE" \
  "$FINAL_READBACK_TWO"; do
  [[ ! -e "$path" ]] || {
    printf 'refusing to overwrite %s\n' "$path" >&2; exit 2;
  }
done
mkdir -p "$RUN_DIR"
exec 3>&1 4>&2
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1
TEE_PID=$!

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'commit mismatch\n' >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || {
  printf 'worktree is dirty or has untracked files\n' >&2; exit 2;
}
[[ "$(sha256sum "$CANDIDATE_CONFIG" | awk '{print $1}')" == \
  "$CANDIDATE_CONFIG_SHA256" ]]
for pair in \
  "$TRAINING_DOMAIN_LIST:$TRAINING_DOMAIN_LIST_SHA256" \
  "$DOMAIN_LIST:$DOMAIN_LIST_SHA256"; do
  path=${pair%%:*}; digest=${pair##*:}
  [[ -f "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]]
done
[[ "$(wc -l < "$DOMAIN_LIST" | tr -d ' ')" == 20 ]]
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || {
  printf 'GPU count != 8\n' >&2; exit 2;
}
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || {
  printf 'runtime missing\n' >&2; exit 2;
}
command -v obsutil >/dev/null
timeout 30s obsutil ls "$OBS_DST/" -limit=1 \
  | tee "$RUN_DIR/obs_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py \
  "$RUN_DIR/obs_prefix_preflight.log"
OBS_DST_OWNED=1
if pgrep -af '[s]cripts/(train_ddp|guarded_endpoint_panel_eval|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 107374182400 )) || {
  printf 'less than 100 GiB free\n' >&2; exit 2;
}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
timeout --signal=TERM --kill-after=30s 12m "$PYTHON" -m pytest -q \
  tests/test_cloud_configs.py \
  tests/test_training_gates.py \
  tests/test_paper_vector_ab.py \
  tests/test_paper_scalar_value_ab.py \
  tests/test_guarded_sampling.py \
  tests/test_guarded_endpoint_panel_adjudication.py \
  tests/test_rollout_robustness_eval.py \
  tests/test_tensor_cloud01.py \
  tests/test_ddp_sync.py | tee "$RUN_DIR/pytest.log"
timeout --signal=TERM --kill-after=30s 15m \
  "$PYTHON" scripts/audit_mdcath_staging.py \
  --root "$DATA_ROOT" \
  --expected-h5 1000 \
  --expected-bytes 668131379559 \
  --expected-subset-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
  --expected-trajectories 25000 \
  --expected-strategy length-proportional \
  --expected-seed 20260715 \
  --expected-commit 3f9f7f7 \
  --samples 5 | tee "$RUN_DIR/data_audit.json"

# Recover and verify the exact completed vector-only source run. This binds the
# baseline checkpoint, history, and absolute decision to its immutable OBS
# manifests before any candidate result is computed.
mkdir -p "$SOURCE_READBACK/baseline" "$SOURCE_READBACK/candidate" \
  "$SOURCE_READBACK/audit"
timeout --signal=TERM --kill-after=30s 12m obsutil sync \
  "$BASELINE_SOURCE_OBS/baseline" "$SOURCE_READBACK/baseline" \
  | tee "$RUN_DIR/source_baseline_sync.log"
timeout --signal=TERM --kill-after=30s 12m obsutil sync \
  "$BASELINE_SOURCE_OBS/candidate" "$SOURCE_READBACK/candidate" \
  | tee "$RUN_DIR/source_candidate_sync.log"
timeout --signal=TERM --kill-after=30s 12m obsutil sync \
  "$BASELINE_SOURCE_OBS/audit" "$SOURCE_READBACK/audit" \
  | tee "$RUN_DIR/source_audit_sync.log"
"$PYTHON" scripts/verify_paper_vector_readback.py \
  --root "$SOURCE_READBACK" --phase completion \
  --expected-final-decision-sha256 "$BASELINE_FINAL_DECISION_SHA256" \
  --expected-final-status "$BASELINE_FINAL_STATUS" \
  --output "$RUN_DIR/source_readback_gate.json"
"$PYTHON" - "$SOURCE_READBACK/audit/summary.json" \
  "$SOURCE_READBACK/audit/readback_completion.json" \
  "$SOURCE_READBACK/audit/candidate_decision.json" \
  "$SOURCE_READBACK/audit/decision.json" \
  "$BASELINE_SOURCE_RUN_ID" "$BASELINE_SOURCE_COMMIT" \
  "$BASELINE_CHECKPOINT_SHA256" "$BASELINE_FINAL_DECISION_SHA256" \
  "$BASELINE_FINAL_STATUS" <<'PY'
import hashlib, json, sys
summary_path, completion_path, candidate_decision_path, decision_path = sys.argv[1:5]
run_id, commit, checkpoint_sha, final_decision_sha, final_status = sys.argv[5:10]
summary = json.load(open(summary_path))
completion = json.load(open(completion_path))
candidate_decision = json.load(open(candidate_decision_path))
decision = json.load(open(decision_path))
if summary.get("run_id") != run_id or completion.get("run_id") != run_id:
    raise SystemExit("sealed vector source run_id mismatch")
if summary.get("commit") != commit or completion.get("commit") != commit:
    raise SystemExit("sealed vector source commit mismatch")
for payload in (summary, completion, candidate_decision, decision):
    if payload.get("candidate_checkpoint_sha256", payload.get("checkpoint_sha256")) != checkpoint_sha:
        raise SystemExit("sealed vector source checkpoint mismatch")
if not (
    summary.get("status")
    == completion.get("scientific_status")
    == decision.get("status")
    == final_status
):
    raise SystemExit("sealed vector source scientific status mismatch")
decision_sha = hashlib.sha256(open(decision_path, "rb").read()).hexdigest()
if decision_sha != final_decision_sha:
    raise SystemExit("sealed vector source final decision identity mismatch")
if completion.get("decision_sha256") != final_decision_sha:
    raise SystemExit("sealed vector source decision SHA mismatch")
PY
[[ "$(sha256sum "$BASELINE_DIR/ckpt_2000.pt" | awk '{print $1}')" == \
  "$BASELINE_CHECKPOINT_SHA256" ]]
[[ "$(sha256sum "$BASELINE_DIR/history.json" | awk '{print $1}')" == \
  "$BASELINE_HISTORY_SHA256" ]]
[[ -n "$BASELINE_DECISION_SHA256" ]]
[[ "$(sha256sum "$SOURCE_READBACK/audit/candidate_decision.json" | awk '{print $1}')" == \
  "$BASELINE_DECISION_SHA256" ]]
cp "$SOURCE_READBACK/audit/candidate_decision.json" \
  "$RUN_DIR/sealed_baseline_decision.json"
cp "$CANDIDATE_CONFIG" "$RUN_DIR/candidate_config.yaml"
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$BASELINE_DIR/ckpt_2000.pt" \
  --history "$BASELINE_DIR/history.json" \
  --expected-step 2000 --expected-world-size 8 --history-mode contains \
  --expected-delta 1 --require-vector-only \
  --expected-lr-horizon-steps 500000 \
  --output "$RUN_DIR/baseline_checkpoint_gate.json"

"$PYTHON" - "$CANDIDATE_CONFIG" <<'PY'
import sys
from deepjump.config import load_config
cfg = load_config(sys.argv[1])
assert cfg.model.tensor_cloud01 is True
assert cfg.model.tensor_cloud01_vector_only_attention is True
assert cfg.model.tensor_cloud01_vector_only_scalar_value is True
assert cfg.train.max_steps == 2000
assert cfg.train.lr_horizon_steps == 500000
assert cfg.train.resume == ""
assert cfg.train.batch_size * 8 * cfg.train.grad_accum == 128
assert cfg.train.amp is False
PY

timeout --signal=TERM --kill-after=2m 60m \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
  scripts/train_ddp.py --config "$CANDIDATE_CONFIG" 2>&1 \
  | tee "$RUN_DIR/train_candidate.log"
grep -q 'world=8 params=4,443,744 effective_batch=128' \
  "$RUN_DIR/train_candidate.log"
grep -q 'done. artifacts in' "$RUN_DIR/train_candidate.log"
if grep -Eiq 'resumed from|warm-started|FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' \
  "$RUN_DIR/train_candidate.log"; then
  printf 'candidate training log contains a forbidden signature\n' >&2
  exit 2
fi
if grep -Eq 'scaler_skips [1-9][0-9]*' "$RUN_DIR/train_candidate.log"; then
  printf 'candidate training log contains skipped optimizer updates\n' >&2
  exit 2
fi
for step in 1000 2000; do
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$CANDIDATE_DIR/ckpt_${step}.pt" \
    --history "$CANDIDATE_DIR/history.json" \
    --expected-step "$step" --expected-world-size 8 --history-mode contains \
    --expected-delta 1 --require-vector-scalar-value \
    --expected-lr-horizon-steps 500000 \
    --output "$RUN_DIR/candidate_checkpoint_${step}_gate.json"
done

BASELINE_CHECKPOINT="$BASELINE_DIR/ckpt_2000.pt"
CANDIDATE_CHECKPOINT="$CANDIDATE_DIR/ckpt_2000.pt"
CANDIDATE_CHECKPOINT_SHA256=$(sha256sum "$CANDIDATE_CHECKPOINT" | awk '{print $1}')
[[ "$BASELINE_CHECKPOINT_SHA256" != "$CANDIDATE_CHECKPOINT_SHA256" ]]

run_training_panel() {
  arm=$1
  profile=$2
  checkpoint=$3
  checkpoint_sha=$4
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 60m \
    "$PYTHON" scripts/guarded_endpoint_panel_eval.py \
    --ckpt "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
    --checkpoint-profile "$profile" \
    --training-data-root "$DATA_ROOT" \
    --training-domain-list "$TRAINING_DOMAIN_LIST" \
    --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --runtime-probe-output "$RUN_DIR/${arm}_runtime_probe.json" \
    --output "$RUN_DIR/${arm}_panel.json" \
    > "$RUN_DIR/${arm}_panel.log" 2>&1
  "$PYTHON" -m scripts.adjudicate_guarded_endpoint_panel \
    --result "$RUN_DIR/${arm}_panel.json" \
    --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
    --checkpoint-profile "$profile" \
    --training-domain-list "$TRAINING_DOMAIN_LIST" \
    --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --output "$RUN_DIR/${arm}_decision.json"
}
run_training_panel baseline paper-horizon-vector-only-500k \
  "$BASELINE_CHECKPOINT" "$BASELINE_CHECKPOINT_SHA256"
run_training_panel candidate paper-horizon-vector-scalar-value-500k \
  "$CANDIDATE_CHECKPOINT" "$CANDIDATE_CHECKPOINT_SHA256"

CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 12m \
  "$PYTHON" scripts/rollout_robustness_eval.py \
  --ckpt "$CANDIDATE_CHECKPOINT" \
  --checkpoint-sha256 "$CANDIDATE_CHECKPOINT_SHA256" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 3 --starts 2 --steps 20 --methods mean,ode_1 \
  --seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state \
  --output "$RUN_DIR/candidate_h20.json" \
  > "$RUN_DIR/candidate_h20.log" 2>&1

"$PYTHON" - "$RUN_DIR/training_evidence.json" "$RUN_ID" "$actual_commit" \
  "$CANDIDATE_CONFIG_SHA256" "$RUN_DIR/sealed_baseline_decision.json" \
  "$RUN_DIR/baseline_decision.json" "$RUN_DIR/candidate_decision.json" \
  "$BASELINE_DIR/history.json" \
  "$CANDIDATE_DIR/history.json" "$RUN_DIR/candidate_h20.json" \
  "$BASELINE_CHECKPOINT_SHA256" "$CANDIDATE_CHECKPOINT_SHA256" \
  "$TRAINING_DOMAIN_LIST_SHA256" "$DOMAIN_LIST_SHA256" <<'PY'
import hashlib, json, sys
(
    output, run_id, commit, config_sha, baseline_decision,
    baseline_replay_decision, candidate_decision, baseline_history,
    candidate_history, candidate_h20, baseline_checkpoint_sha,
    candidate_checkpoint_sha, training_sha, domain_sha,
) = sys.argv[1:15]
sha = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
evidence = {
    "schema": "deepjump.scalar_value_training_evidence.v1",
    "run_id": run_id,
    "commit": commit,
    "candidate_config_sha256": config_sha,
    "baseline_decision_sha256": sha(baseline_decision),
    "baseline_replay_decision_sha256": sha(baseline_replay_decision),
    "candidate_decision_sha256": sha(candidate_decision),
    "baseline_history_sha256": sha(baseline_history),
    "candidate_history_sha256": sha(candidate_history),
    "candidate_h20_sha256": sha(candidate_h20),
    "baseline_checkpoint_sha256": baseline_checkpoint_sha,
    "candidate_checkpoint_sha256": candidate_checkpoint_sha,
    "training_domain_list_sha256": training_sha,
    "domain_list_sha256": domain_sha,
}
open(output, "w").write(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
PY

"$PYTHON" scripts/adjudicate_paper_scalar_value_ab.py \
  --baseline-decision "$RUN_DIR/sealed_baseline_decision.json" \
  --baseline-replay-decision "$RUN_DIR/baseline_decision.json" \
  --candidate-decision "$RUN_DIR/candidate_decision.json" \
  --baseline-history "$BASELINE_DIR/history.json" \
  --candidate-history "$CANDIDATE_DIR/history.json" \
  --candidate-h20 "$RUN_DIR/candidate_h20.json" \
  --evidence-manifest "$RUN_DIR/training_evidence.json" \
  --output "$RUN_DIR/decision.json"

"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$actual_commit" "$BASELINE_CHECKPOINT_SHA256" \
  "$CANDIDATE_CHECKPOINT_SHA256" "$OBS_DST" "$BASELINE_SOURCE_RUN_ID" \
  "$BASELINE_SOURCE_COMMIT" <<'PY'
import datetime, json, sys
decision = json.load(open(sys.argv[1]))
summary = {
    "status": decision["status"],
    "external_development_scientifically_eligible": decision[
        "external_development_scientifically_eligible"
    ],
    "external_development_authorized": False,
    "second_seed_scientifically_eligible": False,
    "second_seed_authorized": False,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "run_id": sys.argv[3],
    "commit": sys.argv[4],
    "baseline_checkpoint_sha256": sys.argv[5],
    "candidate_checkpoint_sha256": sys.argv[6],
    "obs": sys.argv[7],
    "baseline_source_run_id": sys.argv[8],
    "baseline_source_commit": sys.argv[9],
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(sys.argv[2], "w").write(json.dumps(summary, separators=(",", ":")) + "\n")
print(json.dumps(summary, indent=2))
PY

# Freeze runner.log before constructing the exact audit manifest.
printf 'Freezing runner log before exact archive at %s\n' "$(date -Is)"
exec 1>&3 2>&4
wait "$TEE_PID"
exec 3>&- 4>&-

(
  cd "$BASELINE_DIR"
  sha256sum config.json history.json last.ckpt ckpt_1000.pt ckpt_2000.pt
) > "$RUN_DIR/baseline_sha256.txt"
(
  cd "$CANDIDATE_DIR"
  sha256sum config.json history.json last.ckpt ckpt_1000.pt ckpt_2000.pt
) > "$RUN_DIR/candidate_sha256.txt"
(
  cd "$RUN_DIR"
  find . -maxdepth 1 -type f \
    ! -name '*sha256*' -print0 | sort -z | xargs -0 sha256sum
) > "$RUN_DIR/audit_sha256.txt"
(
  cd "$RUN_DIR"
  sha256sum baseline_sha256.txt candidate_sha256.txt audit_sha256.txt
) > "$RUN_DIR/readback_manifests.sha256"

timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$BASELINE_DIR" "$OBS_DST/baseline"
timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$CANDIDATE_DIR" "$OBS_DST/candidate"
timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$RUN_DIR" "$OBS_DST/audit"

verify_readback() {
  target=$1
  mkdir -p "$target/baseline" "$target/candidate" "$target/audit"
  timeout --signal=TERM --kill-after=30s 8m obsutil sync \
    "$OBS_DST/baseline" "$target/baseline"
  timeout --signal=TERM --kill-after=30s 8m obsutil sync \
    "$OBS_DST/candidate" "$target/candidate"
  timeout --signal=TERM --kill-after=30s 8m obsutil sync \
    "$OBS_DST/audit" "$target/audit"
  "$PYTHON" scripts/verify_paper_scalar_value_readback.py \
    --root "$target" --phase initial
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$target/baseline/ckpt_2000.pt" \
    --history "$target/baseline/history.json" --expected-step 2000 \
    --expected-world-size 8 --expected-delta 1 --require-vector-only \
    --expected-lr-horizon-steps 500000
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$target/candidate/ckpt_2000.pt" \
    --history "$target/candidate/history.json" --expected-step 2000 \
    --expected-world-size 8 --expected-delta 1 --require-vector-scalar-value \
    --expected-lr-horizon-steps 500000
}
verify_readback "$READBACK_ONE"
verify_readback "$READBACK_TWO"
"$PYTHON" scripts/verify_paper_scalar_value_readback.py \
  --root "$READBACK_ONE" --root-two "$READBACK_TWO" --phase initial

"$PYTHON" - "$READBACK_TWO/audit/decision.json" \
  "$READBACK_TWO/audit/audit_sha256.txt" "$RUN_DIR/readback_completion.json" \
  "$RUN_ID" "$actual_commit" "$BASELINE_CHECKPOINT_SHA256" \
  "$CANDIDATE_CHECKPOINT_SHA256" <<'PY'
import datetime, hashlib, json, sys
decision_path, audit_path, output_path = sys.argv[1:4]
run_id, commit, baseline_sha, candidate_sha = sys.argv[4:8]
decision = json.load(open(decision_path))
for key in (
    "external_development_authorized", "second_seed_authorized",
    "untouched_confirmation_authorized", "formal_training_authorized",
):
    if decision.get(key) is not False:
        raise SystemExit(f"readback decision must keep {key}=false")
completion = {
    "status": "OBS_PRECOMPLETION_DOUBLE_READBACK_PASS",
    "run_id": run_id,
    "commit": commit,
    "scientific_status": decision["status"],
    "decision_sha256": hashlib.sha256(open(decision_path, "rb").read()).hexdigest(),
    "audit_manifest_sha256": hashlib.sha256(open(audit_path, "rb").read()).hexdigest(),
    "baseline_checkpoint_sha256": baseline_sha,
    "candidate_checkpoint_sha256": candidate_sha,
    "external_development_authorized": False,
    "second_seed_authorized": False,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(output_path, "w").write(json.dumps(completion, separators=(",", ":")) + "\n")
PY
(cd "$RUN_DIR" && sha256sum readback_completion.json readback_manifests.sha256 \
  > final_marker.sha256)
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
verify_final_readback() {
  target=$1
  mkdir -p "$target/baseline" "$target/candidate" "$target/audit"
  timeout --signal=TERM --kill-after=30s 4m obsutil sync \
    "$OBS_DST/baseline" "$target/baseline"
  timeout --signal=TERM --kill-after=30s 4m obsutil sync \
    "$OBS_DST/candidate" "$target/candidate"
  timeout --signal=TERM --kill-after=30s 4m obsutil sync \
    "$OBS_DST/audit" "$target/audit"
  "$PYTHON" scripts/verify_paper_scalar_value_readback.py \
    --root "$target" --phase completion
}
verify_final_readback "$FINAL_READBACK_ONE"
verify_final_readback "$FINAL_READBACK_TWO"
"$PYTHON" scripts/verify_paper_scalar_value_readback.py \
  --root "$FINAL_READBACK_ONE" --root-two "$FINAL_READBACK_TWO" \
  --phase completion
printf 'Scalar-value seed0 A/B complete; external/seed1/untouched/formal were not started.\n'
