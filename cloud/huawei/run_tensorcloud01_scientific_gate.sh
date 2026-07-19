#!/usr/bin/env bash
# Run the frozen 20-domain transition and geometry development gate in parallel.
# This evaluates one reviewed checkpoint, never trains, and powers off on every
# exit path. A scientific FAIL is archived as a completed experiment.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to one selected calibration checkpoint}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?set the reviewed checkpoint SHA256}
EXPECTED_CHECKPOINT_STEP=${EXPECTED_CHECKPOINT_STEP:?set the selected step}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set the deployed reviewed commit}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set the authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1}
DELTA=${DELTA:?set DELTA to exactly one of 1, 10, or 100}
EVAL_ID=${EVAL_ID:?set a stable evaluation identifier shared with the calibration}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-30}
DOMAIN_LIST=${DOMAIN_LIST:-configs/dev_20_length_proportional_seed0.txt}
DOMAIN_LIST_SHA256=${DOMAIN_LIST_SHA256:-4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af}
RUN_DIR="$REPO/runs/tensorcloud01_d${DELTA}_step${EXPECTED_CHECKPOINT_STEP}_scientific_$EVAL_ID"
READBACK_DIR="/tmp/tensorcloud01_scientific_readback_$EVAL_ID"
OBS_DST="$BUCKET/deepjump-evaluation/tensorcloud01-scientific/delta$DELTA/step$EXPECTED_CHECKPOINT_STEP/$EVAL_ID"

case "$DELTA" in
  1|10|100) ;;
  *) printf 'unsupported DELTA=%s\n' "$DELTA" >&2; exit 2 ;;
esac
[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 30 ]] || { printf 'HARD_STOP_MINUTES must be 30\n' >&2; exit 2; }
[[ "$EXPECTED_CHECKPOINT_STEP" =~ ^(250|500|750|1000)$ ]] || {
  printf 'checkpoint step must be one of 250, 500, 750, or 1000\n' >&2
  exit 2
}
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }

shutdown_on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "$RUN_DIR" ]] && timeout 120s obsutil sync "$RUN_DIR" "$OBS_DST/failure"
    set -e
  fi
  printf 'scientific evaluation delta=%s step=%s exit=%s; requesting shutdown at %s\n' \
    "$DELTA" "$EXPECTED_CHECKPOINT_STEP" "$code" "$(date -Is)"
  sudo -n shutdown -h now || printf 'ERROR: shutdown command failed\n' >&2
  exit "$code"
}
trap shutdown_on_exit EXIT
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

cd "$REPO"
[[ ! -e "$RUN_DIR" ]] || { printf 'refusing to overwrite %s\n' "$RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/scientific_gate.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  printf 'tracked worktree is dirty\n' >&2
  exit 2
}
[[ -f "$CHECKPOINT" ]] || { printf 'missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 2; }
actual_checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$actual_checkpoint_sha" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch\n' >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
command -v obsutil >/dev/null
findmnt /data
df -hT /data
if pgrep -af '[s]cripts/(train_ddp|transition_robustness_eval|geometry_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

"$PYTHON" -m pytest -q \
  tests/test_evaluation_protocol.py \
  tests/test_transition_robustness_eval.py \
  tests/test_evaluation_integration_gate.py \
  tests/test_scientific_evaluation_gate.py | tee "$RUN_DIR/pytest.log"

printf 'starting parallel 20-domain gates at %s\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 18m \
  "$PYTHON" scripts/transition_robustness_eval.py \
  --ckpt "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 20 --starts 1 --draws 4 --methods mean \
  --real-frames 500 --max-features 512 \
  --lag 10 --tica-components 4 --clusters 32 --msm-lag 1 \
  --seed 20260717 --output "$RUN_DIR/transition_20domains.json" \
  > "$RUN_DIR/transition_20domains.log" 2>&1 &
transition_pid=$!

CUDA_VISIBLE_DEVICES=1 timeout --signal=TERM --kill-after=30s 18m \
  "$PYTHON" scripts/geometry_robustness_eval.py \
  --ckpt "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 20 --starts 1 --steps 20 --methods mean \
  --reference-frames 500 --calibration-draws 10000 \
  --alpha 0.01 --seed 20260717 \
  --output "$RUN_DIR/geometry_20_20domains.json" \
  > "$RUN_DIR/geometry_20_20domains.log" 2>&1 &
geometry20_pid=$!

CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=30s 24m \
  "$PYTHON" scripts/geometry_robustness_eval.py \
  --ckpt "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 20 --starts 1 --steps 100 --methods mean \
  --reference-frames 500 --calibration-draws 10000 \
  --alpha 0.01 --seed 20260717 \
  --output "$RUN_DIR/geometry_100_20domains.json" \
  > "$RUN_DIR/geometry_100_20domains.log" 2>&1 &
geometry100_pid=$!

wait "$transition_pid"
wait "$geometry20_pid"
wait "$geometry100_pid"
printf 'parallel 20-domain gates finished at %s\n' "$(date -Is)"

set +e
"$PYTHON" scripts/validate_scientific_evaluation.py \
  --transition "$RUN_DIR/transition_20domains.json" \
  --geometry-20 "$RUN_DIR/geometry_20_20domains.json" \
  --geometry-100 "$RUN_DIR/geometry_100_20domains.json" \
  --expected-checkpoint "$CHECKPOINT" \
  --expected-step "$EXPECTED_CHECKPOINT_STEP" \
  --expected-domains 20 --expected-delta "$DELTA" \
  --expected-domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/local_scientific_gate.json"
scientific_status=$?
set -e
[[ "$scientific_status" == 0 || "$scientific_status" == 1 ]] || {
  printf 'unexpected scientific validator exit=%s\n' "$scientific_status" >&2
  exit 2
}

(
  cd "$RUN_DIR"
  sha256sum transition_20domains.json geometry_20_20domains.json \
    geometry_100_20domains.json local_scientific_gate.json > SHA256SUMS
)
obsutil sync "$RUN_DIR" "$OBS_DST"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST" "$READBACK_DIR"
(
  cd "$READBACK_DIR"
  sha256sum -c SHA256SUMS
)
set +e
"$PYTHON" scripts/validate_scientific_evaluation.py \
  --transition "$READBACK_DIR/transition_20domains.json" \
  --geometry-20 "$READBACK_DIR/geometry_20_20domains.json" \
  --geometry-100 "$READBACK_DIR/geometry_100_20domains.json" \
  --expected-checkpoint "$CHECKPOINT" \
  --expected-step "$EXPECTED_CHECKPOINT_STEP" \
  --expected-domains 20 --expected-delta "$DELTA" \
  --expected-domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --output "$RUN_DIR/obs_scientific_gate.json"
readback_scientific_status=$?
set -e
[[ "$readback_scientific_status" == "$scientific_status" ]] || {
  printf 'local and OBS scientific decisions disagree\n' >&2
  exit 2
}

if [[ "$scientific_status" == 0 ]]; then
  decision=PASS
else
  decision=FAIL
fi
printf '{"status":"PASS","scope":"completed_scientific_evaluation","scientific_decision":"%s","eval_id":"%s","commit":"%s","checkpoint_step":%s,"checkpoint_sha256":"%s","delta_frames":%s,"domains":20,"cells_per_domain":25,"obs":"%s","completed_at":"%s"}\n' \
  "$decision" "$EVAL_ID" "$actual_commit" "$EXPECTED_CHECKPOINT_STEP" \
  "$actual_checkpoint_sha" "$DELTA" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST"
printf 'Scientific evaluation completed; decision=%s. Formal training was not started.\n' "$decision"
