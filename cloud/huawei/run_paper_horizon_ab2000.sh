#!/usr/bin/env bash
# Run the preregistered matched fresh-continuous LR-horizon A/B, evaluate both
# fixed step-2000 checkpoints on training-dev20, conditionally run the frozen
# paper-horizon external20 comparison, archive twice, and power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-600}
HARD_STOP_UNIT="deepjump-paper-horizon-ab-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "${RUN_DIR:-}" ]] && timeout 300s obsutil sync "$RUN_DIR" "${OBS_DST:-}/failure/audit"
    [[ -d "${BASELINE_DIR:-}" ]] && timeout 300s obsutil sync "$BASELINE_DIR" "${OBS_DST:-}/failure/baseline"
    [[ -d "${CANDIDATE_DIR:-}" ]] && timeout 300s obsutil sync "$CANDIDATE_DIR" "${OBS_DST:-}/failure/candidate"
    set -e
  fi
  printf 'Paper-horizon A/B exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 600 ]] || { printf 'HARD_STOP_MINUTES must be 600\n' >&2; exit 2; }
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
BASELINE_CONFIG=configs/v100_tensorcloud01_full_d1_fp32_horizon_ab_baseline2000.yaml
CANDIDATE_CONFIG=configs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000.yaml
BASELINE_CONFIG_SHA256=309a35f4bf5b510466fb99a9ae5107a5c89ce921b3ecf53675e201eec5006fd8
CANDIDATE_CONFIG_SHA256=506237167a3921bdf4dfe795fccee1b79bd8628aa89f3c28b5677301097a9898
TRAINING_DOMAIN_LIST=configs/subset_1000_length_proportional.txt
TRAINING_DOMAIN_LIST_SHA256=39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734
DOMAIN_LIST=configs/dev_20_length_proportional_seed0.txt
DOMAIN_LIST_SHA256=4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af
PRIOR_EXTERNAL_DOMAIN_LIST=configs/external_dev_20_length_proportional_seed20260721.txt
PRIOR_EXTERNAL_DOMAIN_LIST_SHA256=9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245
PRIOR_FRESH_EXTERNAL_DOMAIN_LIST=configs/guarded_external_dev_20_length_proportional_seed20260722.txt
PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256=9bae11fa0e6336e7451c372efa25ca55af77aa9cb27f91e1fd241612531a920f
UNTOUCHED_DOMAIN_LIST=configs/confirmation_100_length_proportional_seed20260717.txt
UNTOUCHED_DOMAIN_LIST_SHA256=e56ed7de735db542f4e20fb73f2654a6c1bcf67f3082849f63f0ab74f4208c38
PAPER_EXTERNAL_DOMAIN_LIST=configs/paper_horizon_external_dev_20_length_proportional_seed20260723.txt
PAPER_EXTERNAL_DOMAIN_LIST_SHA256=9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb
PAPER_EXTERNAL_EXPECTED_BYTES=14236836972
EXTERNAL_DATA_ROOT=${EXTERNAL_DATA_ROOT:-/data/mdcath_paper_horizon_external20_seed20260723}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
BASELINE_DIR="$REPO/runs/v100_tensorcloud01_full_d1_fp32_horizon_ab_baseline2000"
CANDIDATE_DIR="$REPO/runs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000"
RUN_DIR="$REPO/runs/paper_horizon_ab2000_$RUN_ID"
OBS_DST="$BUCKET/deepjump-calibration/paper-horizon-ab2000/$RUN_ID"
READBACK_ONE="/tmp/paper_horizon_ab2000_readback_one_$RUN_ID"
READBACK_TWO="/tmp/paper_horizon_ab2000_readback_two_$RUN_ID"
COMPLETION_READBACK="/tmp/paper_horizon_ab2000_completion_$RUN_ID"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
for path in "$RUN_DIR" "$BASELINE_DIR" "$CANDIDATE_DIR" \
  "$READBACK_ONE" "$READBACK_TWO" "$COMPLETION_READBACK"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { printf 'worktree is dirty or has untracked files\n' >&2; exit 2; }
[[ "$(sha256sum "$BASELINE_CONFIG" | awk '{print $1}')" == "$BASELINE_CONFIG_SHA256" ]]
[[ "$(sha256sum "$CANDIDATE_CONFIG" | awk '{print $1}')" == "$CANDIDATE_CONFIG_SHA256" ]]
[[ "$(sha256sum "$TRAINING_DOMAIN_LIST" | awk '{print $1}')" == "$TRAINING_DOMAIN_LIST_SHA256" ]]
[[ "$(sha256sum "$DOMAIN_LIST" | awk '{print $1}')" == "$DOMAIN_LIST_SHA256" ]]
for pair in \
  "$PRIOR_EXTERNAL_DOMAIN_LIST:$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
  "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST:$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256" \
  "$UNTOUCHED_DOMAIN_LIST:$UNTOUCHED_DOMAIN_LIST_SHA256" \
  "$PAPER_EXTERNAL_DOMAIN_LIST:$PAPER_EXTERNAL_DOMAIN_LIST_SHA256"; do
  path=${pair%%:*}; digest=${pair##*:}
  [[ -f "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]]
done
[[ "$(wc -l < "$DOMAIN_LIST" | tr -d ' ')" == 20 ]]
[[ "$(wc -l < "$PAPER_EXTERNAL_DOMAIN_LIST" | tr -d ' ')" == 20 ]]
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || { printf 'runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
timeout 30s obsutil ls "$OBS_DST/" -limit=1 \
  | tee "$RUN_DIR/obs_prefix_preflight.log"
"$PYTHON" - "$RUN_DIR/obs_prefix_preflight.log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r"Object number\s*(?:is)?\s*:\s*([0-9]+)", text)
if not match:
    raise SystemExit("OBS prefix preflight did not return a parseable object count")
if int(match.group(1)) != 0:
    raise SystemExit("refusing to reuse non-empty OBS evidence prefix")
PY
if pgrep -af '[s]cripts/(train_ddp|guarded_endpoint_panel_eval|external_endpoint_panel_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
df -hT /data
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 161061273600 )) || { printf 'less than 150 GiB free\n' >&2; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
timeout --signal=TERM --kill-after=30s 10m "$PYTHON" -m pytest -q \
  tests/test_cloud_configs.py \
  tests/test_training_gates.py \
  tests/test_paper_horizon_ab.py \
  tests/test_guarded_sampling.py \
  tests/test_guarded_endpoint_panel_adjudication.py \
  tests/test_tensor_cloud01.py | tee "$RUN_DIR/pytest.log"

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

"$PYTHON" - "$BASELINE_CONFIG" "$CANDIDATE_CONFIG" <<'PY'
from dataclasses import asdict
import sys
from deepjump.config import load_config
from deepjump.training import lr_at

left, right = map(load_config, sys.argv[1:])
assert asdict(left.data) == asdict(right.data)
assert asdict(left.model) == asdict(right.model)
left_train, right_train = asdict(left.train), asdict(right.train)
assert left_train.pop("lr_horizon_steps") == 1000
assert right_train.pop("lr_horizon_steps") == 500000
left_train.pop("out_dir"); right_train.pop("out_dir")
assert left_train == right_train
assert left.train.max_steps == right.train.max_steps == 2000
assert left.train.resume == right.train.resume == ""
assert lr_at(1999, left) == 3.0e-3
assert abs(lr_at(1999, right) - 0.004992801120448179) < 1e-15
PY

run_arm() {
  arm=$1
  config=$2
  output_dir=$3
  horizon=$4
  log="$RUN_DIR/train_${arm}.log"
  timeout --signal=TERM --kill-after=2m 60m \
    "$TORCHRUN" --standalone --nproc_per_node=8 \
    scripts/train_ddp.py --config "$config" 2>&1 | tee "$log"
  grep -q 'world=8 params=4,840,032 effective_batch=128' "$log"
  grep -q 'done. artifacts in' "$log"
  if grep -Eiq 'resumed from|warm-started|FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$log"; then
    printf '%s training log contains a forbidden signature\n' "$arm" >&2
    exit 2
  fi
  if grep -Eq 'scaler_skips [1-9][0-9]*' "$log"; then
    printf '%s training log contains skipped optimizer updates\n' "$arm" >&2
    exit 2
  fi
  for step in 1000 2000; do
    "$PYTHON" scripts/validate_training_checkpoint.py \
      --checkpoint "$output_dir/ckpt_${step}.pt" \
      --history "$output_dir/history.json" \
      --expected-step "$step" --expected-world-size 8 --history-mode contains \
      --expected-delta 1 --require-full-tensor \
      --expected-lr-horizon-steps "$horizon" \
      --output "$RUN_DIR/${arm}_checkpoint_${step}_gate.json"
  done
}

run_arm baseline "$BASELINE_CONFIG" "$BASELINE_DIR" 1000
run_arm candidate "$CANDIDATE_CONFIG" "$CANDIDATE_DIR" 500000

BASELINE_CHECKPOINT="$BASELINE_DIR/ckpt_2000.pt"
CANDIDATE_CHECKPOINT="$CANDIDATE_DIR/ckpt_2000.pt"
BASELINE_CHECKPOINT_SHA256=$(sha256sum "$BASELINE_CHECKPOINT" | awk '{print $1}')
CANDIDATE_CHECKPOINT_SHA256=$(sha256sum "$CANDIDATE_CHECKPOINT" | awk '{print $1}')
[[ "$BASELINE_CHECKPOINT_SHA256" != "$CANDIDATE_CHECKPOINT_SHA256" ]]

run_panel() {
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
    --output "$RUN_DIR/${arm}_panel.json" > "$RUN_DIR/${arm}_panel.log" 2>&1
  "$PYTHON" -m scripts.adjudicate_guarded_endpoint_panel \
    --result "$RUN_DIR/${arm}_panel.json" \
    --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
    --checkpoint-profile "$profile" \
    --training-domain-list "$TRAINING_DOMAIN_LIST" \
    --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
    --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --output "$RUN_DIR/${arm}_decision.json"
}

run_panel baseline paper-horizon-ab-baseline1000 \
  "$BASELINE_CHECKPOINT" "$BASELINE_CHECKPOINT_SHA256"
run_panel candidate paper-horizon-500k \
  "$CANDIDATE_CHECKPOINT" "$CANDIDATE_CHECKPOINT_SHA256"

"$PYTHON" scripts/adjudicate_paper_horizon_ab.py \
  --baseline-decision "$RUN_DIR/baseline_decision.json" \
  --candidate-decision "$RUN_DIR/candidate_decision.json" \
  --baseline-history "$BASELINE_DIR/history.json" \
  --candidate-history "$CANDIDATE_DIR/history.json" \
  --output "$RUN_DIR/training_ab_decision.json"

TRAINING_AB_DECISION_SHA256=$(sha256sum "$RUN_DIR/training_ab_decision.json" | awk '{print $1}')
training_ab_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$RUN_DIR/training_ab_decision.json")

if [[ "$training_ab_status" == ADVANCE_PAPER_HORIZON_EXTERNAL20 ]]; then
  mkdir -p "$EXTERNAL_DATA_ROOT"
  timeout --signal=TERM --kill-after=30s 90m \
    "$PYTHON" scripts/download_mdcath.py \
    --root "$EXTERNAL_DATA_ROOT" --domains-file "$PAPER_EXTERNAL_DOMAIN_LIST" --retries 5 \
    | tee "$RUN_DIR/external_download.log"
  timeout --signal=TERM --kill-after=30s 10m \
    "$PYTHON" scripts/audit_external_mdcath.py \
    --root "$EXTERNAL_DATA_ROOT" --domain-list "$PAPER_EXTERNAL_DOMAIN_LIST" \
    --domain-list-sha256 "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" \
    --expected-bytes "$PAPER_EXTERNAL_EXPECTED_BYTES" \
    --output "$RUN_DIR/external_data_audit.json"

  run_external_panel() {
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
      --panel-kind paper-horizon-external --panel-data-root "$EXTERNAL_DATA_ROOT" \
      --prior-external-domain-list "$PRIOR_EXTERNAL_DOMAIN_LIST" \
      --prior-external-domain-list-sha256 "$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
      --prior-fresh-external-domain-list "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST" \
      --prior-fresh-external-domain-list-sha256 "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256" \
      --untouched-domain-list "$UNTOUCHED_DOMAIN_LIST" \
      --untouched-domain-list-sha256 "$UNTOUCHED_DOMAIN_LIST_SHA256" \
      --prerequisite-decision "$RUN_DIR/training_ab_decision.json" \
      --prerequisite-decision-sha256 "$TRAINING_AB_DECISION_SHA256" \
      --candidate-checkpoint-sha256 "$CANDIDATE_CHECKPOINT_SHA256" \
      --domain-list "$PAPER_EXTERNAL_DOMAIN_LIST" \
      --domain-list-sha256 "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" \
      --runtime-probe-output "$RUN_DIR/external_${arm}_runtime_probe.json" \
      --output "$RUN_DIR/external_${arm}_panel.json" \
      > "$RUN_DIR/external_${arm}_panel.log" 2>&1
    "$PYTHON" -m scripts.adjudicate_guarded_endpoint_panel \
      --result "$RUN_DIR/external_${arm}_panel.json" \
      --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
      --checkpoint-profile "$profile" \
      --training-domain-list "$TRAINING_DOMAIN_LIST" \
      --training-domain-list-sha256 "$TRAINING_DOMAIN_LIST_SHA256" \
      --panel-kind paper-horizon-external \
      --prior-external-domain-list "$PRIOR_EXTERNAL_DOMAIN_LIST" \
      --prior-external-domain-list-sha256 "$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
      --prior-fresh-external-domain-list "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST" \
      --prior-fresh-external-domain-list-sha256 "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256" \
      --untouched-domain-list "$UNTOUCHED_DOMAIN_LIST" \
      --untouched-domain-list-sha256 "$UNTOUCHED_DOMAIN_LIST_SHA256" \
      --prerequisite-decision "$RUN_DIR/training_ab_decision.json" \
      --prerequisite-decision-sha256 "$TRAINING_AB_DECISION_SHA256" \
      --candidate-checkpoint-sha256 "$CANDIDATE_CHECKPOINT_SHA256" \
      --domain-list "$PAPER_EXTERNAL_DOMAIN_LIST" \
      --domain-list-sha256 "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" \
      --output "$RUN_DIR/external_${arm}_decision.json"
  }

  run_external_panel baseline paper-horizon-ab-baseline1000 \
    "$BASELINE_CHECKPOINT" "$BASELINE_CHECKPOINT_SHA256"
  run_external_panel candidate paper-horizon-500k \
    "$CANDIDATE_CHECKPOINT" "$CANDIDATE_CHECKPOINT_SHA256"
  "$PYTHON" scripts/adjudicate_paper_horizon_ab.py \
    --panel-kind paper-horizon-external \
    --baseline-decision "$RUN_DIR/external_baseline_decision.json" \
    --candidate-decision "$RUN_DIR/external_candidate_decision.json" \
    --baseline-history "$BASELINE_DIR/history.json" \
    --candidate-history "$CANDIDATE_DIR/history.json" \
    --output "$RUN_DIR/decision.json"
  printf '{"status":"EXECUTED_PAPER_HORIZON_EXTERNAL20","completed_at":"%s"}\n' \
    "$(date -Is)" > "$RUN_DIR/external_status.json"
else
  cp "$RUN_DIR/training_ab_decision.json" "$RUN_DIR/decision.json"
  printf '{"status":"SKIPPED_PAPER_HORIZON_EXTERNAL20","reason":"training_ab_did_not_advance","completed_at":"%s"}\n' \
    "$(date -Is)" > "$RUN_DIR/external_status.json"
fi

"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$actual_commit" "$BASELINE_CHECKPOINT_SHA256" "$CANDIDATE_CHECKPOINT_SHA256" "$OBS_DST" <<'PY'
import datetime, json, sys
decision = json.load(open(sys.argv[1]))
summary = {
    "status": decision["status"],
    "external_development_authorized": decision["external_development_authorized"],
    "second_seed_scientifically_eligible": decision.get(
        "second_seed_scientifically_eligible", False
    ),
    "second_seed_authorized": False,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "run_id": sys.argv[3],
    "commit": sys.argv[4],
    "baseline_checkpoint_sha256": sys.argv[5],
    "candidate_checkpoint_sha256": sys.argv[6],
    "obs": sys.argv[7],
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(sys.argv[2], "w").write(json.dumps(summary, separators=(",", ":")) + "\n")
print(json.dumps(summary, indent=2))
PY

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
    ! -name runner.log ! -name '*sha256*' -print0 | sort -z | xargs -0 sha256sum
) > "$RUN_DIR/audit_sha256.txt"

timeout --signal=TERM --kill-after=30s 8m obsutil sync "$BASELINE_DIR" "$OBS_DST/baseline"
timeout --signal=TERM --kill-after=30s 8m obsutil sync "$CANDIDATE_DIR" "$OBS_DST/candidate"
timeout --signal=TERM --kill-after=30s 8m obsutil sync "$RUN_DIR" "$OBS_DST/audit"

verify_readback() {
  target=$1
  mkdir -p "$target/baseline" "$target/candidate" "$target/audit"
  timeout --signal=TERM --kill-after=30s 8m obsutil sync "$OBS_DST/baseline" "$target/baseline"
  timeout --signal=TERM --kill-after=30s 8m obsutil sync "$OBS_DST/candidate" "$target/candidate"
  timeout --signal=TERM --kill-after=30s 8m obsutil sync "$OBS_DST/audit" "$target/audit"
  (cd "$target/baseline" && sha256sum -c "$RUN_DIR/baseline_sha256.txt")
  (cd "$target/candidate" && sha256sum -c "$RUN_DIR/candidate_sha256.txt")
  (cd "$target/audit" && sha256sum -c "$RUN_DIR/audit_sha256.txt")
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$target/baseline/ckpt_2000.pt" \
    --history "$target/baseline/history.json" --expected-step 2000 \
    --expected-world-size 8 --expected-delta 1 --require-full-tensor \
    --expected-lr-horizon-steps 1000
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$target/candidate/ckpt_2000.pt" \
    --history "$target/candidate/history.json" --expected-step 2000 \
    --expected-world-size 8 --expected-delta 1 --require-full-tensor \
    --expected-lr-horizon-steps 500000
}
verify_readback "$READBACK_ONE"
verify_readback "$READBACK_TWO"

"$PYTHON" - "$READBACK_TWO/audit/decision.json" "$RUN_DIR/audit_sha256.txt" \
  "$RUN_DIR/authorization.json" "$RUN_DIR/readback_completion.json" "$RUN_ID" \
  "$actual_commit" "$BASELINE_CHECKPOINT_SHA256" "$CANDIDATE_CHECKPOINT_SHA256" <<'PY'
import datetime, hashlib, json, sys

decision_path, audit_path, authorization_path, completion_path = sys.argv[1:5]
run_id, commit, baseline_sha, candidate_sha = sys.argv[5:9]
decision = json.load(open(decision_path))
decision_sha = hashlib.sha256(open(decision_path, "rb").read()).hexdigest()
audit_sha = hashlib.sha256(open(audit_path, "rb").read()).hexdigest()
for key in (
    "second_seed_authorized",
    "untouched_confirmation_authorized",
    "formal_training_authorized",
):
    if decision.get(key) is not False:
        raise SystemExit(f"readback decision must keep {key}=false")
scientifically_eligible = decision.get("second_seed_scientifically_eligible")
if scientifically_eligible not in {True, False}:
    raise SystemExit("readback decision has non-boolean scientific eligibility")
eligible = bool(
    decision.get("status") == "PASS_PAPER_HORIZON_EXTERNAL20"
    and scientifically_eligible is True
)
common = {
    "run_id": run_id,
    "commit": commit,
    "scientific_status": decision["status"],
    "decision_sha256": decision_sha,
    "audit_manifest_sha256": audit_sha,
    "baseline_checkpoint_sha256": baseline_sha,
    "candidate_checkpoint_sha256": candidate_sha,
    "second_seed_scientifically_eligible": eligible,
    "untouched_confirmation_authorized": False,
    "formal_training_authorized": False,
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
authorization = {
    **common,
    "status": "SECOND_SEED_AUTHORIZED" if eligible else "SECOND_SEED_NOT_AUTHORIZED",
    "second_seed_authorized": eligible,
    "authorization_requires_independent_readback": True,
}
completion = {
    **common,
    "status": "OBS_DOUBLE_READBACK_PASS",
    "second_seed_authorized": False,
}
open(authorization_path, "w").write(
    json.dumps(authorization, separators=(",", ":")) + "\n"
)
open(completion_path, "w").write(
    json.dumps(completion, separators=(",", ":")) + "\n"
)
PY
(cd "$RUN_DIR" && sha256sum authorization.json readback_completion.json \
  > final_markers.sha256)
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$COMPLETION_READBACK"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$OBS_DST/audit" "$COMPLETION_READBACK"
(cd "$COMPLETION_READBACK" && sha256sum -c "$RUN_DIR/final_markers.sha256")
printf 'Paper-horizon A/B complete; seed1/untouched/formal training was not started.\n'
