#!/usr/bin/env bash
# Strict 5x5 model-level integration gate for the TensorCloud01 checkpoint.
# This is deliberately small in domains/starts/draws but never relaxes the
# complete mdCATH temperature/replica grid. It does not start training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the reviewed TensorCloud01 checkpoint}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?set the reviewed checkpoint SHA256}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set EXPECTED_REPO_COMMIT to the deployed SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set EXPECTED_HOSTNAME to the authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-35}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOMAIN_LIST=${DOMAIN_LIST:-configs/dev_20_length_proportional_seed0.txt}
DOMAIN_LIST_SHA256=${DOMAIN_LIST_SHA256:-4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af}
RUN_DIR="$REPO/runs/tensorcloud01_eval_integration_$RUN_ID"
READBACK_DIR="/tmp/tensorcloud01_eval_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-evaluation/tensorcloud01-integration/$RUN_ID"

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }

shutdown_on_exit() {
  code=$?
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "$RUN_DIR" ]] && timeout 90s obsutil sync "$RUN_DIR" "$OBS_DST/failure"
    set -e
  fi
  printf 'evaluation integration exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -h now || printf 'ERROR: shutdown command failed\n' >&2
  exit "$code"
}
trap shutdown_on_exit EXIT
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

cd "$REPO"
[[ ! -e "$RUN_DIR" ]] || { printf 'refusing to overwrite %s\n' "$RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/integration.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { printf 'tracked worktree is dirty\n' >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { printf 'missing checkpoint: %s\n' "$CHECKPOINT" >&2; exit 2; }
actual_checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$actual_checkpoint_sha" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch: actual=%s expected=%s\n' \
    "$actual_checkpoint_sha" "$EXPECTED_CHECKPOINT_SHA256" >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
command -v obsutil >/dev/null

"$PYTHON" -m pytest -q \
  tests/test_evaluation_protocol.py \
  tests/test_tica_robustness_eval.py \
  tests/test_transition_robustness_eval.py \
  tests/test_evaluation_integration_gate.py | tee "$RUN_DIR/pytest.log"

printf 'gate=transition_5x5 start=%s\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 10m \
  "$PYTHON" scripts/transition_robustness_eval.py \
  --ckpt "$CHECKPOINT" \
  --domain-list "$DOMAIN_LIST" \
  --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
  --domains 1 --starts 1 --draws 2 --methods mean \
  --real-frames 128 --max-features 128 \
  --lag 10 --tica-components 4 --clusters 32 --msm-lag 1 \
  --seed 20260717 --output "$RUN_DIR/transition_5x5.json" \
  > "$RUN_DIR/transition_5x5.log" 2>&1

for steps in 20 100; do
  printf 'gate=geometry_5x5_%s start=%s\n' "$steps" "$(date -Is)"
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 12m \
    "$PYTHON" scripts/geometry_robustness_eval.py \
    --ckpt "$CHECKPOINT" \
    --domain-list "$DOMAIN_LIST" \
    --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --domains 1 --starts 1 --steps "$steps" --methods mean \
    --reference-frames 128 --calibration-draws 200 \
    --alpha 0.01 --seed 20260717 \
    --output "$RUN_DIR/geometry_${steps}_5x5.json" \
    > "$RUN_DIR/geometry_${steps}_5x5.log" 2>&1
done

"$PYTHON" scripts/validate_full_grid_evaluation.py \
  --transition "$RUN_DIR/transition_5x5.json" \
  --geometry-20 "$RUN_DIR/geometry_20_5x5.json" \
  --geometry-100 "$RUN_DIR/geometry_100_5x5.json" \
  --expected-checkpoint "$CHECKPOINT" --expected-step 30 --expected-domains 1 \
  --output "$RUN_DIR/local_gate.json"

(
  cd "$RUN_DIR"
  sha256sum transition_5x5.json geometry_20_5x5.json geometry_100_5x5.json local_gate.json \
    > SHA256SUMS
)
obsutil sync "$RUN_DIR" "$OBS_DST"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST" "$READBACK_DIR"
(
  cd "$READBACK_DIR"
  sha256sum -c SHA256SUMS
)
"$PYTHON" scripts/validate_full_grid_evaluation.py \
  --transition "$READBACK_DIR/transition_5x5.json" \
  --geometry-20 "$READBACK_DIR/geometry_20_5x5.json" \
  --geometry-100 "$READBACK_DIR/geometry_100_5x5.json" \
  --expected-checkpoint "$CHECKPOINT" --expected-step 30 --expected-domains 1 \
  --output "$RUN_DIR/obs_gate.json"

printf '{"status":"PASS","scope":"integration_only","run_id":"%s","commit":"%s","checkpoint_sha256":"%s","cells_per_domain":25,"evaluated_domains":1,"obs":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$actual_commit" "$actual_checkpoint_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST"
printf 'Strict 5x5 model-level integration PASS; scientific calibration/training was not started.\n'
