#!/usr/bin/env bash
# Bounded inference-only discriminator for deterministic mean versus ODE sampling.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the frozen step1000 checkpoint}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?set checkpoint SHA256}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?set SHUTDOWN_ON_EXIT=1}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-10}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
DOMAIN_LIST=${DOMAIN_LIST:-configs/dev_20_length_proportional_seed0.txt}
DOMAIN_LIST_SHA256=${DOMAIN_LIST_SHA256:-4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af}
RUN_DIR="$REPO/runs/vector_only_sampling_discriminator_$RUN_ID"
READBACK_DIR="/tmp/vector_only_sampling_discriminator_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-evaluation/vector-only-sampling-discriminator/$RUN_ID"

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 10 ]] || { printf 'HARD_STOP_MINUTES must be 10\n' >&2; exit 2; }

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && command -v obsutil >/dev/null; then
    set +e
    [[ -d "$RUN_DIR" ]] && timeout 60s obsutil sync "$RUN_DIR" "$OBS_DST/failure"
    set -e
  fi
  printf 'sampling discriminator exit=%s; requesting shutdown at %s\n' "$code" "$(date -Is)"
  sudo -n shutdown -c 2>/dev/null || true
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 ]]; then
    printf 'ERROR: shutdown command failed with exit=%s\n' "$shutdown_code" >&2
    [[ "$code" != 0 ]] || code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT
sudo -n shutdown -h "+$HARD_STOP_MINUTES"

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
[[ ! -e "$RUN_DIR" ]] || { printf 'refusing to overwrite %s\n' "$RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/discriminator.log") 2>&1

actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || { printf 'commit mismatch\n' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { printf 'tracked worktree is dirty\n' >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { printf 'checkpoint missing\n' >&2; exit 2; }
actual_checkpoint_sha=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[[ "$actual_checkpoint_sha" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  printf 'checkpoint SHA256 mismatch\n' >&2
  exit 2
}
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 8 ]] || { printf 'GPU count != 8\n' >&2; exit 2; }
command -v obsutil >/dev/null
if pgrep -af '[s]cripts/(train_ddp|rollout_robustness_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi

"$PYTHON" -m pytest -q \
  tests/test_sampling_integrators.py \
  tests/test_rollout_robustness_eval.py | tee "$RUN_DIR/pytest.log"

for anchor in state conditioner; do
  printf 'anchor=%s start=%s\n' "$anchor" "$(date -Is)"
  CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 4m \
    "$PYTHON" scripts/rollout_robustness_eval.py \
    --ckpt "$CHECKPOINT" \
    --domain-list "$DOMAIN_LIST" \
    --domain-list-sha256 "$DOMAIN_LIST_SHA256" \
    --domains 1 --starts 1 --steps 20 \
    --methods mean,ode_1,ode_5,ode_20 \
    --seed 20260718 --integrator euler --tau-max 1.0 \
    --drift-anchor "$anchor" \
    --output "$RUN_DIR/${anchor}.json" \
    > "$RUN_DIR/${anchor}.log" 2>&1
done

"$PYTHON" - "$RUN_DIR/state.json" "$RUN_DIR/conditioner.json" \
  "$CHECKPOINT" "$DOMAIN_LIST_SHA256" "$RUN_DIR/local_gate.json" <<'PY'
import json
import math
import sys

state_path, conditioner_path, checkpoint, domain_sha, output = sys.argv[1:]
expected_methods = {"noop", "mean", "ode_1", "ode_5", "ode_20"}
results = {}
for anchor, path in (("state", state_path), ("conditioner", conditioner_path)):
    result = json.load(open(path))
    assert result["checkpoint"] == checkpoint
    assert result["checkpoint_step"] == 1000
    assert result["domain_panel"]["sha256"] == domain_sha
    assert result["domain_panel"]["evaluated_count"] == 1
    assert result["settings"]["domains"] == 1
    assert result["settings"]["starts"] == 1
    assert result["settings"]["steps"] == 20
    assert result["settings"]["drift_anchor"] == anchor
    assert set(result["summary"]) == expected_methods

    def check_finite(value):
        if isinstance(value, dict):
            for child in value.values():
                check_finite(child)
        elif isinstance(value, list):
            for child in value:
                check_finite(child)
        elif isinstance(value, float):
            assert math.isfinite(value)

    check_finite(result)
    results[anchor] = result

assert results["state"]["domains"][0]["methods"]["mean"] == (
    results["conditioner"]["domains"][0]["methods"]["mean"]
)
report = {
    "status": "PASS",
    "scope": "inference_mechanism_probe_only",
    "checkpoint_step": 1000,
    "evaluated_domains": 1,
    "starts": 1,
    "rollout_steps": 20,
    "anchors": ["state", "conditioner"],
    "methods": sorted(expected_methods),
}
open(output, "w").write(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

(
  cd "$RUN_DIR"
  sha256sum state.json conditioner.json local_gate.json > SHA256SUMS
)
obsutil sync "$RUN_DIR" "$OBS_DST"
mkdir "$READBACK_DIR"
obsutil sync "$OBS_DST" "$READBACK_DIR"
(
  cd "$READBACK_DIR"
  sha256sum -c SHA256SUMS
)

printf '{"status":"PASS","scope":"inference_mechanism_probe_only","run_id":"%s","commit":"%s","checkpoint_sha256":"%s","obs":"%s","completed_at":"%s"}\n' \
  "$RUN_ID" "$actual_commit" "$actual_checkpoint_sha" "$OBS_DST" "$(date -Is)" \
  | tee "$RUN_DIR/summary.json"
obsutil sync "$RUN_DIR" "$OBS_DST"
printf 'Sampling discriminator PASS; no training or scientific gate was run.\n'
