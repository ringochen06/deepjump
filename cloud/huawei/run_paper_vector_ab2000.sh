#!/usr/bin/env bash
# Run the preregistered third arm: a fresh vector-only paper-horizon model
# against the sealed full-tensor paper-horizon control. Conditionally consume
# the frozen external20 once, archive twice, and always power off.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
OBS_CLAIM_PYTHON=${OBS_CLAIM_PYTHON:-/data/venvs/obs-claim/bin/python}
DATA_ROOT=${DATA_ROOT:-/data/mdcath}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-660}
HARD_STOP_UNIT="deepjump-paper-vector-ab-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
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
  printf 'Paper-vector A/B exit=%s; requesting shutdown at %s\n' \
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
[[ "$HARD_STOP_MINUTES" == 660 ]] || {
  printf 'HARD_STOP_MINUTES must be 660\n' >&2; exit 2;
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

BASELINE_SOURCE_RUN_ID=20260722T012922Z
BASELINE_SOURCE_COMMIT=dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b
BASELINE_SOURCE_OBS="$BUCKET/deepjump-calibration/paper-horizon-ab2000/$BASELINE_SOURCE_RUN_ID/candidate"
BASELINE_SOURCE_AUDIT_OBS="$BUCKET/deepjump-calibration/paper-horizon-ab2000/$BASELINE_SOURCE_RUN_ID/audit"
BASELINE_SOURCE_DECISION_SHA256=2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38
BASELINE_CHECKPOINT_SHA256=fb12d776b106867ca14a8f56476daf776a6296b6dca640f03c2188a75a69bb47
BASELINE_HISTORY_SHA256=868e3e44386163e61e61f6c0da60c160e3cb9f282e20c3ba7a9198208c64fa3f
CANDIDATE_CONFIG=configs/v100_tensorcloud01_vector_only_d1_fp32_paper_horizon500k_2000.yaml
CANDIDATE_CONFIG_SHA256=dea3690f81d542c03289e79f0f37cfa4083beb9390aa0eeb68baa82e75ca5e4a
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
EXTERNAL_DATA_ROOT=/data/mdcath_paper_vector_external20_seed20260723
OBS_ENDPOINT=https://obs.cn-north-4.myhuaweicloud.com
OBS_BUCKET=deepjump-mdcath-cn4-ringochen
EXTERNAL_CLAIM_KEY="deepjump-governance/external-panel-claims/v1/$PAPER_EXTERNAL_DOMAIN_LIST_SHA256/claim.json"
EXTERNAL_CLAIM_OBS="$BUCKET/${EXTERNAL_CLAIM_KEY%/claim.json}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}

RUN_DIR="$REPO/runs/paper_vector_ab2000_$RUN_ID"
CANDIDATE_DIR="$REPO/runs/v100_tensorcloud01_vector_only_d1_fp32_paper_horizon500k_2000"
BASELINE_DIR="/tmp/paper_vector_baseline_$RUN_ID"
SOURCE_AUDIT_ONE="/tmp/paper_vector_source_audit_one_$RUN_ID"
SOURCE_AUDIT_TWO="/tmp/paper_vector_source_audit_two_$RUN_ID"
READBACK_ONE="/tmp/paper_vector_ab_readback_one_$RUN_ID"
READBACK_TWO="/tmp/paper_vector_ab_readback_two_$RUN_ID"
FINAL_READBACK_ONE="/tmp/paper_vector_ab_final_one_$RUN_ID"
FINAL_READBACK_TWO="/tmp/paper_vector_ab_final_two_$RUN_ID"
CLAIM_READBACK_ONE="/tmp/paper_vector_external_claim_one_$RUN_ID.json"
CLAIM_READBACK_TWO="/tmp/paper_vector_external_claim_two_$RUN_ID.json"
OBS_DST="$BUCKET/deepjump-calibration/paper-vector-ab2000/$RUN_ID"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  printf 'hostname mismatch\n' >&2; exit 2;
}
cd "$REPO"
for path in "$RUN_DIR" "$CANDIDATE_DIR" "$BASELINE_DIR" \
  "$SOURCE_AUDIT_ONE" "$SOURCE_AUDIT_TWO" "$READBACK_ONE" "$READBACK_TWO" \
  "$FINAL_READBACK_ONE" "$FINAL_READBACK_TWO" "$CLAIM_READBACK_ONE" \
  "$CLAIM_READBACK_TWO"; do
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
  "$DOMAIN_LIST:$DOMAIN_LIST_SHA256" \
  "$PRIOR_EXTERNAL_DOMAIN_LIST:$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
  "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST:$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256" \
  "$UNTOUCHED_DOMAIN_LIST:$UNTOUCHED_DOMAIN_LIST_SHA256" \
  "$PAPER_EXTERNAL_DOMAIN_LIST:$PAPER_EXTERNAL_DOMAIN_LIST_SHA256"; do
  path=${pair%%:*}; digest=${pair##*:}
  [[ -f "$path" && "$(sha256sum "$path" | awk '{print $1}')" == "$digest" ]]
done
[[ "$(wc -l < "$DOMAIN_LIST" | tr -d ' ')" == 20 ]]
[[ "$(wc -l < "$PAPER_EXTERNAL_DOMAIN_LIST" | tr -d ' ')" == 20 ]]
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || {
  printf 'GPU count != 8\n' >&2; exit 2;
}
[[ -x "$PYTHON" && -x "$TORCHRUN" && -x "$OBS_CLAIM_PYTHON" ]] || {
  printf 'runtime missing\n' >&2; exit 2;
}
"$OBS_CLAIM_PYTHON" -c \
  'import importlib.metadata,obs; assert importlib.metadata.version("esdk-obs-python") == "3.26.6"'
command -v obsutil >/dev/null
timeout 30s obsutil ls "$OBS_DST/" -limit=1 \
  | tee "$RUN_DIR/obs_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py \
  "$RUN_DIR/obs_prefix_preflight.log"
OBS_DST_OWNED=1
timeout 30s obsutil ls "$EXTERNAL_CLAIM_OBS/" -limit=1 \
  | tee "$RUN_DIR/external_claim_preflight_initial.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py \
  "$RUN_DIR/external_claim_preflight_initial.log"
if pgrep -af '[s]cripts/(train_ddp|guarded_endpoint_panel_eval|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

findmnt /data
available_bytes=$(df --output=avail -B1 /data | tail -n 1 | tr -d ' ')
(( available_bytes >= 161061273600 )) || {
  printf 'less than 150 GiB free\n' >&2; exit 2;
}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
timeout --signal=TERM --kill-after=30s 12m "$PYTHON" -m pytest -q \
  tests/test_cloud_configs.py \
  tests/test_training_gates.py \
  tests/test_paper_vector_ab.py \
  tests/test_guarded_sampling.py \
  tests/test_guarded_endpoint_panel_adjudication.py \
  tests/test_rollout_robustness_eval.py \
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

# Prove from two independent exact OBS readbacks that the authoritative source
# run stopped before consuming this external20.
mkdir -p "$SOURCE_AUDIT_ONE" "$SOURCE_AUDIT_TWO"
timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$BASELINE_SOURCE_AUDIT_OBS" "$SOURCE_AUDIT_ONE" \
  | tee "$RUN_DIR/source_audit_one_sync.log"
timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$BASELINE_SOURCE_AUDIT_OBS" "$SOURCE_AUDIT_TWO" \
  | tee "$RUN_DIR/source_audit_two_sync.log"
"$PYTHON" scripts/verify_paper_vector_source_stop.py \
  --readback-one "$SOURCE_AUDIT_ONE" --readback-two "$SOURCE_AUDIT_TWO" \
  --source-runner cloud/huawei/run_paper_horizon_ab2000.sh \
  --output "$RUN_DIR/source_proof.json"
SOURCE_PROOF_SHA256=$(sha256sum "$RUN_DIR/source_proof.json" | awk '{print $1}')

# Recover the exact sealed full-tensor paper-horizon arm from its immutable
# source prefix. Its checkpoint and history identities were fixed before this
# third-arm implementation.
mkdir -p "$BASELINE_DIR"
timeout --signal=TERM --kill-after=30s 12m obsutil sync \
  "$BASELINE_SOURCE_OBS" "$BASELINE_DIR"
[[ "$(sha256sum "$BASELINE_DIR/ckpt_2000.pt" | awk '{print $1}')" == \
  "$BASELINE_CHECKPOINT_SHA256" ]]
[[ "$(sha256sum "$BASELINE_DIR/history.json" | awk '{print $1}')" == \
  "$BASELINE_HISTORY_SHA256" ]]
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$BASELINE_DIR/ckpt_2000.pt" \
  --history "$BASELINE_DIR/history.json" \
  --expected-step 2000 --expected-world-size 8 --history-mode contains \
  --expected-delta 1 --require-full-tensor \
  --expected-lr-horizon-steps 500000 \
  --output "$RUN_DIR/baseline_checkpoint_gate.json"

"$PYTHON" - "$CANDIDATE_CONFIG" <<'PY'
from dataclasses import asdict
import sys
from deepjump.config import load_config
cfg = load_config(sys.argv[1])
assert cfg.model.tensor_cloud01 is True
assert cfg.model.tensor_cloud01_vector_only_attention is True
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
grep -q 'world=8 params=4,038,240 effective_batch=128' \
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
    --expected-delta 1 --require-vector-only \
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
run_training_panel baseline paper-horizon-500k \
  "$BASELINE_CHECKPOINT" "$BASELINE_CHECKPOINT_SHA256"
run_training_panel candidate paper-horizon-vector-only-500k \
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

"$PYTHON" scripts/adjudicate_paper_vector_ab.py \
  --baseline-decision "$RUN_DIR/baseline_decision.json" \
  --candidate-decision "$RUN_DIR/candidate_decision.json" \
  --baseline-history "$BASELINE_DIR/history.json" \
  --candidate-history "$CANDIDATE_DIR/history.json" \
  --candidate-h20 "$RUN_DIR/candidate_h20.json" \
  --output "$RUN_DIR/training_ab_decision.json"
TRAINING_AB_DECISION_SHA256=$(sha256sum \
  "$RUN_DIR/training_ab_decision.json" | awk '{print $1}')
training_status=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$RUN_DIR/training_ab_decision.json")

if [[ "$training_status" == ADVANCE_PAPER_VECTOR_EXTERNAL20 ]]; then
  # The panel was frozen before the predecessor A/B and has never been
  # consumed. Refuse any pre-existing local root so a stale/peeked copy cannot
  # enter this one-shot external evaluation.
  [[ ! -e "$EXTERNAL_DATA_ROOT" ]] || {
    printf 'paper-vector external root already exists; refusing possible reuse\n' >&2
    exit 2
  }
  timeout 30s obsutil ls "$EXTERNAL_CLAIM_OBS/" -limit=1 \
    | tee "$RUN_DIR/external_claim_preflight_final.log"
  "$PYTHON" scripts/verify_obsutil_empty_prefix.py \
    "$RUN_DIR/external_claim_preflight_final.log"
  "$PYTHON" - "$RUN_DIR/external_claim.json" "$RUN_ID" "$actual_commit" \
    "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" "$BASELINE_SOURCE_DECISION_SHA256" \
    "$SOURCE_PROOF_SHA256" "$TRAINING_AB_DECISION_SHA256" \
    "$BASELINE_CHECKPOINT_SHA256" \
    "$CANDIDATE_CHECKPOINT_SHA256" <<'PY'
import datetime, json, sys
output, run_id, commit, panel_sha, source_decision_sha, source_proof_sha, training_decision_sha, baseline_sha, candidate_sha = sys.argv[1:10]
claim = {
    "schema": "deepjump.external_panel_claim.v1",
    "status": "CLAIMED_FOR_SINGLE_USE",
    "run_id": run_id,
    "commit": commit,
    "panel_sha256": panel_sha,
    "panel_count": 20,
    "expected_total_bytes": 14236836972,
    "source_stop_decision_sha256": source_decision_sha,
    "source_proof_sha256": source_proof_sha,
    "training_ab_decision_sha256": training_decision_sha,
    "baseline_checkpoint_sha256": baseline_sha,
    "candidate_checkpoint_sha256": candidate_sha,
    "prior_authoritative_run_consumed": False,
    "claimed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
open(output, "w").write(json.dumps(claim, sort_keys=True, separators=(",", ":")) + "\n")
PY
  claim_sha=$(sha256sum "$RUN_DIR/external_claim.json" | awk '{print $1}')
  timeout --signal=TERM --kill-after=30s 2m \
    "$OBS_CLAIM_PYTHON" scripts/claim_external_panel.py \
    --endpoint "$OBS_ENDPOINT" --bucket "$OBS_BUCKET" \
    --key "$EXTERNAL_CLAIM_KEY" \
    --claim-json "$RUN_DIR/external_claim.json" \
    --readback-one "$CLAIM_READBACK_ONE" \
    --readback-two "$CLAIM_READBACK_TWO" \
    --output "$RUN_DIR/external_claim_readback.json"
  [[ "$(sha256sum "$CLAIM_READBACK_ONE" | awk '{print $1}')" == "$claim_sha" ]]
  [[ "$(sha256sum "$CLAIM_READBACK_TWO" | awk '{print $1}')" == "$claim_sha" ]]
  mkdir -p "$EXTERNAL_DATA_ROOT"
  timeout --signal=TERM --kill-after=30s 90m \
    "$PYTHON" scripts/download_mdcath.py \
    --root "$EXTERNAL_DATA_ROOT" --domains-file "$PAPER_EXTERNAL_DOMAIN_LIST" \
    --retries 5 | tee "$RUN_DIR/external_download.log"
  timeout --signal=TERM --kill-after=30s 10m \
    "$PYTHON" scripts/audit_external_mdcath.py \
    --root "$EXTERNAL_DATA_ROOT" --domain-list "$PAPER_EXTERNAL_DOMAIN_LIST" \
    --domain-list-sha256 "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" \
    --expected-bytes "$PAPER_EXTERNAL_EXPECTED_BYTES" \
    --output "$RUN_DIR/external_data_audit.json"
  "$PYTHON" scripts/write_external_download_manifest.py \
    --root "$EXTERNAL_DATA_ROOT" \
    --domain-list "$PAPER_EXTERNAL_DOMAIN_LIST" \
    --domain-list-sha256 "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" \
    --expected-bytes "$PAPER_EXTERNAL_EXPECTED_BYTES" \
    --audit "$RUN_DIR/external_data_audit.json" \
    --claim "$RUN_DIR/external_claim.json" --claim-sha256 "$claim_sha" \
    --run-id "$RUN_ID" --commit "$actual_commit" \
    --output "$RUN_DIR/external_download_manifest.json"
  EXTERNAL_DOWNLOAD_MANIFEST_SHA256=$(sha256sum \
    "$RUN_DIR/external_download_manifest.json" | awk '{print $1}')

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
      --panel-kind paper-vector-external \
      --panel-data-root "$EXTERNAL_DATA_ROOT" \
      --prior-external-domain-list "$PRIOR_EXTERNAL_DOMAIN_LIST" \
      --prior-external-domain-list-sha256 "$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
      --prior-fresh-external-domain-list "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST" \
      --prior-fresh-external-domain-list-sha256 "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256" \
      --untouched-domain-list "$UNTOUCHED_DOMAIN_LIST" \
      --untouched-domain-list-sha256 "$UNTOUCHED_DOMAIN_LIST_SHA256" \
      --prerequisite-decision "$RUN_DIR/training_ab_decision.json" \
      --prerequisite-decision-sha256 "$TRAINING_AB_DECISION_SHA256" \
      --baseline-checkpoint-sha256 "$BASELINE_CHECKPOINT_SHA256" \
      --candidate-checkpoint-sha256 "$CANDIDATE_CHECKPOINT_SHA256" \
      --external-claim "$RUN_DIR/external_claim.json" \
      --external-claim-sha256 "$claim_sha" \
      --external-download-manifest "$RUN_DIR/external_download_manifest.json" \
      --external-download-manifest-sha256 "$EXTERNAL_DOWNLOAD_MANIFEST_SHA256" \
      --source-proof "$RUN_DIR/source_proof.json" \
      --source-proof-sha256 "$SOURCE_PROOF_SHA256" \
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
      --panel-kind paper-vector-external \
      --prior-external-domain-list "$PRIOR_EXTERNAL_DOMAIN_LIST" \
      --prior-external-domain-list-sha256 "$PRIOR_EXTERNAL_DOMAIN_LIST_SHA256" \
      --prior-fresh-external-domain-list "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST" \
      --prior-fresh-external-domain-list-sha256 "$PRIOR_FRESH_EXTERNAL_DOMAIN_LIST_SHA256" \
      --untouched-domain-list "$UNTOUCHED_DOMAIN_LIST" \
      --untouched-domain-list-sha256 "$UNTOUCHED_DOMAIN_LIST_SHA256" \
      --prerequisite-decision "$RUN_DIR/training_ab_decision.json" \
      --prerequisite-decision-sha256 "$TRAINING_AB_DECISION_SHA256" \
      --baseline-checkpoint-sha256 "$BASELINE_CHECKPOINT_SHA256" \
      --candidate-checkpoint-sha256 "$CANDIDATE_CHECKPOINT_SHA256" \
      --external-claim "$RUN_DIR/external_claim.json" \
      --external-claim-sha256 "$claim_sha" \
      --external-download-manifest "$RUN_DIR/external_download_manifest.json" \
      --external-download-manifest-sha256 "$EXTERNAL_DOWNLOAD_MANIFEST_SHA256" \
      --source-proof "$RUN_DIR/source_proof.json" \
      --source-proof-sha256 "$SOURCE_PROOF_SHA256" \
      --domain-list "$PAPER_EXTERNAL_DOMAIN_LIST" \
      --domain-list-sha256 "$PAPER_EXTERNAL_DOMAIN_LIST_SHA256" \
      --output "$RUN_DIR/external_${arm}_decision.json"
  }
  run_external_panel baseline paper-horizon-500k \
    "$BASELINE_CHECKPOINT" "$BASELINE_CHECKPOINT_SHA256"
  run_external_panel candidate paper-horizon-vector-only-500k \
    "$CANDIDATE_CHECKPOINT" "$CANDIDATE_CHECKPOINT_SHA256"
  "$PYTHON" scripts/adjudicate_paper_vector_ab.py \
    --panel-kind paper-vector-external \
    --baseline-decision "$RUN_DIR/external_baseline_decision.json" \
    --candidate-decision "$RUN_DIR/external_candidate_decision.json" \
    --baseline-history "$BASELINE_DIR/history.json" \
    --candidate-history "$CANDIDATE_DIR/history.json" \
    --output "$RUN_DIR/decision.json"
  printf '{"status":"EXECUTED_PAPER_VECTOR_EXTERNAL20","completed_at":"%s"}\n' \
    "$(date -Is)" > "$RUN_DIR/external_status.json"
else
  cp "$RUN_DIR/training_ab_decision.json" "$RUN_DIR/decision.json"
  printf '{"status":"SKIPPED_PAPER_VECTOR_EXTERNAL20","reason":"training_ab_did_not_advance","completed_at":"%s"}\n' \
    "$(date -Is)" > "$RUN_DIR/external_status.json"
fi

"$PYTHON" - "$RUN_DIR/decision.json" "$RUN_DIR/summary.json" "$RUN_ID" \
  "$actual_commit" "$BASELINE_CHECKPOINT_SHA256" \
  "$CANDIDATE_CHECKPOINT_SHA256" "$OBS_DST" "$BASELINE_SOURCE_RUN_ID" \
  "$BASELINE_SOURCE_COMMIT" <<'PY'
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
  "$PYTHON" scripts/verify_paper_vector_readback.py \
    --root "$target" --phase initial
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$target/baseline/ckpt_2000.pt" \
    --history "$target/baseline/history.json" --expected-step 2000 \
    --expected-world-size 8 --expected-delta 1 --require-full-tensor \
    --expected-lr-horizon-steps 500000
  "$PYTHON" scripts/validate_training_checkpoint.py \
    --checkpoint "$target/candidate/ckpt_2000.pt" \
    --history "$target/candidate/history.json" --expected-step 2000 \
    --expected-world-size 8 --expected-delta 1 --require-vector-only \
    --expected-lr-horizon-steps 500000
}
verify_readback "$READBACK_ONE"
verify_readback "$READBACK_TWO"
"$PYTHON" scripts/verify_paper_vector_readback.py \
  --root "$READBACK_ONE" --root-two "$READBACK_TWO" --phase initial

"$PYTHON" - "$READBACK_TWO/audit/decision.json" \
  "$READBACK_TWO/audit/audit_sha256.txt" "$RUN_DIR/readback_completion.json" \
  "$RUN_ID" "$actual_commit" "$BASELINE_CHECKPOINT_SHA256" \
  "$CANDIDATE_CHECKPOINT_SHA256" <<'PY'
import datetime, hashlib, json, sys
decision_path, audit_path, output_path = sys.argv[1:4]
run_id, commit, baseline_sha, candidate_sha = sys.argv[4:8]
decision = json.load(open(decision_path))
for key in ("second_seed_authorized", "untouched_confirmation_authorized", "formal_training_authorized"):
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
  "$PYTHON" scripts/verify_paper_vector_readback.py \
    --root "$target" --phase completion
}
verify_final_readback "$FINAL_READBACK_ONE"
verify_final_readback "$FINAL_READBACK_TWO"
"$PYTHON" scripts/verify_paper_vector_readback.py \
  --root "$FINAL_READBACK_ONE" --root-two "$FINAL_READBACK_TWO" \
  --phase completion
printf 'Paper-vector A/B complete; seed1/untouched/formal training was not started.\n'
