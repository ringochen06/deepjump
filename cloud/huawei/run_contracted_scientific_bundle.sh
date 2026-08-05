#!/usr/bin/env bash
# Run one contracted scientific evaluation bundle. Never starts training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-240}
HARD_STOP_UNIT="deepjump-scientific-bundle-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR=
OBS_PREFIX=

shutdown_on_exit() {
  code=$?
  trap - EXIT
  set +e
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" && -n "$OBS_PREFIX" ]] \
    && command -v obsutil >/dev/null; then
    timeout 8m obsutil sync "$RUN_DIR" "$OBS_PREFIX/failure-audit"
  fi
  sudo -n shutdown -h now
  shutdown_code=$?
  set -e
  [[ "$code" != 0 ]] || code=$shutdown_code
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { echo 'SHUTDOWN_ON_EXIT must be 1' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 240 ]] || { echo 'HARD_STOP_MINUTES must remain 240' >&2; exit 2; }
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n systemctl show "$HARD_STOP_UNIT.timer" \
  --property=ActiveState,SubState,NextElapseUSecRealtime --no-pager
sudo -n systemctl show "$HARD_STOP_UNIT.service" --property=ExecStart --no-pager \
  | grep -Fq '/usr/bin/systemctl poweroff'
sudo -n shutdown -c 2>/dev/null || true

EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set reviewed commit}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
PROTOCOL=${PROTOCOL:?set scientific protocol JSON}
PROTOCOL_SHA256=${PROTOCOL_SHA256:?set exact protocol SHA256}
EVALUATOR_SOURCE_SHA256=${EVALUATOR_SOURCE_SHA256:?set reviewed evaluator source SHA256}
SESSION=${SESSION:?set exact scientific session JSON}
SESSION_SHA256=${SESSION_SHA256:?set exact scientific session SHA256}
PREREQUISITE=${PREREQUISITE:?set exact v2 prerequisite JSON}
PREREQUISITE_SHA256=${PREREQUISITE_SHA256:?set exact prerequisite SHA256}
CHECKPOINT=${CHECKPOINT:?set contracted checkpoint}
CONTRACT=${CONTRACT:?set full-data contract}
PANEL_FILE=${PANEL_FILE:?set frozen panel file}
DATA_ROOT=${DATA_ROOT:?set qualified read-only data root}
BUCKET=${BUCKET:?set OBS bucket}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
QUALIFICATION_MODE=${QUALIFICATION_MODE:-bundle}

[[ "$EXPECTED_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
for digest in "$PROTOCOL_SHA256" "$EVALUATOR_SOURCE_SHA256" \
  "$SESSION_SHA256" "$PREREQUISITE_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
case "$QUALIFICATION_MODE" in bundle|delta1_oracle|runtime) ;; *) exit 2 ;; esac
[[ "$BUCKET" == obs://* ]] || exit 2
for path in "$PROTOCOL" "$SESSION" "$PREREQUISITE" "$CHECKPOINT" \
  "$CONTRACT" "$PANEL_FILE" "$DATA_ROOT"; do
  [[ "$path" == /* ]] || { echo "path must be absolute: $path" >&2; exit 2; }
done
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { echo 'hostname mismatch' >&2; exit 2; }

cd "$REPO"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$REPO/src"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REPO_COMMIT" ]] || exit 2
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
  echo 'worktree is dirty or has untracked files' >&2
  exit 2
}
actual_evaluator_sha=$("$PYTHON" - "$REPO/scripts/contracted_scientific_bundle_eval.py" <<'PY'
import hashlib
import os
import stat
import sys

flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(sys.argv[1], flags)
with os.fdopen(fd, "rb") as handle:
    before = os.fstat(handle.fileno())
    raw = handle.read()
    after = os.fstat(handle.fileno())
if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
    raise SystemExit("evaluator source changed during hash")
print(hashlib.sha256(raw).hexdigest())
PY
)
[[ "$actual_evaluator_sha" == "$EVALUATOR_SOURCE_SHA256" ]] || {
  echo 'evaluator source SHA256 mismatch' >&2
  exit 2
}
for path in "$PROTOCOL" "$SESSION" "$PREREQUISITE" "$CHECKPOINT" \
  "$CONTRACT" "$PANEL_FILE"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "non-regular input: $path" >&2; exit 2; }
done

mount_options=$(findmnt -T "$DATA_ROOT" -n -o OPTIONS)
case ",$mount_options," in
  *,ro,*) ;;
  *) echo 'DATA_ROOT mount must be read-only' >&2; exit 2 ;;
esac
gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$gpu_count" == 8 ]] || { echo "GPU count $gpu_count != 8" >&2; exit 2; }
if pgrep -af '[s]cripts/(train|train_ddp|contracted_.*eval).py'; then
  echo 'conflicting training/evaluation process exists' >&2
  exit 2
fi
for service in deepjump-mdcath-download.service deepjump-mdcath-hash.service \
  deepjump-mdcath-copy.service; do
  ! systemctl is-active --quiet "$service" || {
    echo "data mutation service is active: $service" >&2
    exit 2
  }
done

RUN_DIR="$REPO/runs/contracted_scientific_bundle_$RUN_ID"
OBS_PREFIX="$BUCKET/deepjump-scientific/contracted-bundle/$RUN_ID"
RAW_OUTPUT="$RUN_DIR/raw.json"
DECISION_OUTPUT="$RUN_DIR/decision.json"
STATE_ARCHIVE_OUTPUT="$RUN_DIR/state_archive.npz"
RUNTIME_PROBE_OUTPUT="$RUN_DIR/evidence/runtime_probe.json"
READBACK_ONE="/tmp/contracted_scientific_readback_one_$RUN_ID"
READBACK_TWO="/tmp/contracted_scientific_readback_two_$RUN_ID"
READBACK_THREE="/tmp/contracted_scientific_readback_three_$RUN_ID"
for path in "$RUN_DIR" "$READBACK_ONE" "$READBACK_TWO" "$READBACK_THREE"; do
  [[ ! -e "$path" ]] || { echo "refusing existing path: $path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR/evidence"
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

# Read the session only through the same hash-bound, no-follow loader used by
# the evaluator. Bind every authority-bearing output before any one-shot claim.
"$PYTHON" - "$PROTOCOL" "$PROTOCOL_SHA256" "$SESSION" "$SESSION_SHA256" \
  "$PREREQUISITE" "$PREREQUISITE_SHA256" "$EXPECTED_REPO_COMMIT" \
  "$RUNTIME_PROBE_OUTPUT" "$RAW_OUTPUT" "$DECISION_OUTPUT" \
  "$STATE_ARCHIVE_OUTPUT" "$OBS_PREFIX" \
  "$RUN_DIR/evidence/session_binding.json" "$QUALIFICATION_MODE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts.contracted_scientific_bundle_eval import (
    ORACLE_UNRESOLVED,
    load_delta1_oracle_artifact,
    load_bundle_prerequisites_for_mode,
    load_protocol,
    load_runtime_feasibility_artifact,
    load_scientific_prerequisite,
    load_session,
    validate_bundle_bindings,
)

(protocol_path, protocol_sha, session_path, session_sha, prerequisite_path,
 prerequisite_sha, commit, runtime_output, raw_output, decision_output,
 state_archive_output, obs_prefix, report_path, qualification_mode) = sys.argv[1:]
protocol = load_protocol(protocol_path, protocol_sha)
session = load_session(session_path, session_sha)
prerequisite = load_scientific_prerequisite(
    prerequisite_path, prerequisite_sha, session=session
)
validate_bundle_bindings(
    protocol, session, prerequisite, expected_repo_commit=commit
)
expected = {
    "runtime_probe_output": str(Path(runtime_output).resolve()),
    "raw_output": str(Path(raw_output).resolve()),
    "decision_output": str(Path(decision_output).resolve()),
    "state_archive_output": str(Path(state_archive_output).resolve()),
    "obs_prefix": obs_prefix,
}
if any(session[key] != value for key, value in expected.items()):
    raise SystemExit("scientific session output/OBS binding mismatch")
oracle, runtime = load_bundle_prerequisites_for_mode(
    protocol, session, qualification_mode
)
report = {
    "schema": "deepjump.contracted_scientific_runner_binding.v1",
    "session_sha256": session["sha256"],
    "authorization_id": session["authorization_id"],
    "phase": session["phase"],
    "panel_sha256": session["panel_sha256"],
    **expected,
    "oracle_artifact_sha256": None if oracle is None else oracle["sha256"],
    "runtime_feasibility_sha256": None if runtime is None else runtime["sha256"],
    "formal_training_authorized": False,
}
from scripts.contracted_scientific_bundle_eval import _write_new_json
_write_new_json(report_path, report)
PY

timeout 8m obsutil ls "$OBS_PREFIX/" -limit=1 \
  | tee "$RUN_DIR/evidence/obs_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py \
  "$RUN_DIR/evidence/obs_prefix_preflight.log"

timeout --signal=TERM --kill-after=30s 12m \
  "$PYTHON" -m pytest -q \
  tests/test_contracted_scientific_bundle_eval.py \
  tests/test_adjudicate_contracted_scientific_bundle.py \
  tests/test_contracted_scientific_bundle_runner.py \
  tests/test_evaluation_consumption.py \
  tests/test_full_training_data_contract.py \
  2>&1 | tee "$RUN_DIR/evidence/pytest.log"

# Refuse to consume the one-shot authorization unless both numerical kernels
# declare their reviewed implementation boundaries ready.
"$PYTHON" scripts/contracted_scientific_bundle_eval.py --implementation-status \
  | tee "$RUN_DIR/evidence/evaluator_implementation.json"
"$PYTHON" scripts/adjudicate_contracted_scientific_bundle.py --implementation-status \
  | tee "$RUN_DIR/evidence/adjudicator_implementation.json"
"$PYTHON" - "$RUN_DIR/evidence/evaluator_implementation.json" \
  "$RUN_DIR/evidence/adjudicator_implementation.json" <<'PY'
import json
import sys

from scripts.contracted_scientific_bundle_eval import _read_stable_regular_bytes

evaluator = json.loads(_read_stable_regular_bytes(sys.argv[1], "evaluator status")[0])
adjudicator = json.loads(_read_stable_regular_bytes(sys.argv[2], "adjudicator status")[0])
if evaluator.get("numerical_kernel_implemented") is not True:
    raise SystemExit("scientific evaluator kernel is not implemented; authorization preserved")
if adjudicator.get("independent_numerical_recomputation_implemented") is not True:
    raise SystemExit("scientific adjudicator kernel is not implemented; authorization preserved")
PY

# Qualification artifacts can only be created by the two real measurement
# CLIs in this reviewed evaluator source. Both commands reject pre-existing raw
# and decision outputs; no runner path accepts caller-supplied raw evidence.
if [[ "$QUALIFICATION_MODE" == delta1_oracle ]]; then
  [[ ! -e "$RAW_OUTPUT" && ! -e "$DECISION_OUTPUT" ]] || exit 2
  timeout --signal=TERM --kill-after=2m 150m \
    "$PYTHON" scripts/contracted_scientific_bundle_eval.py \
    --measure-delta1-oracle \
    --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA256" \
    --session "$SESSION" --expected-session-sha256 "$SESSION_SHA256" \
    --prerequisite-decision "$PREREQUISITE" \
    --expected-prerequisite-decision-sha256 "$PREREQUISITE_SHA256" \
    --checkpoint "$CHECKPOINT" --contract "$CONTRACT" \
    --panel-file "$PANEL_FILE" --data-root "$DATA_ROOT" \
    --expected-repo-commit "$EXPECTED_REPO_COMMIT" \
    --raw-output "$RAW_OUTPUT" --decision-output "$DECISION_OUTPUT"
  [[ -s "$RAW_OUTPUT" && -s "$DECISION_OUTPUT" ]] || exit 2
  timeout 8m obsutil sync "$RUN_DIR" "$OBS_PREFIX/qualification-audit"
  exit 0
fi
if [[ "$QUALIFICATION_MODE" == runtime ]]; then
  PROBE_PLAN=${PROBE_PLAN:?set frozen runtime probe-plan JSON}
  PROBE_PLAN_SHA256=${PROBE_PLAN_SHA256:?set exact runtime probe-plan SHA256}
  [[ "$PROBE_PLAN" == /* && "$PROBE_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
  [[ ! -e "$RAW_OUTPUT" && ! -e "$DECISION_OUTPUT" ]] || exit 2
  timeout --signal=TERM --kill-after=2m 150m \
    "$PYTHON" scripts/contracted_scientific_bundle_eval.py \
    --measure-runtime \
    --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA256" \
    --session "$SESSION" --expected-session-sha256 "$SESSION_SHA256" \
    --checkpoint "$CHECKPOINT" --contract "$CONTRACT" \
    --data-root "$DATA_ROOT" --probe-plan "$PROBE_PLAN" \
    --expected-probe-plan-sha256 "$PROBE_PLAN_SHA256" \
    --raw-output "$RAW_OUTPUT" --decision-output "$DECISION_OUTPUT"
  [[ -s "$RAW_OUTPUT" && -s "$DECISION_OUTPUT" ]] || exit 2
  timeout 8m obsutil sync "$RUN_DIR" "$OBS_PREFIX/qualification-audit"
  exit 0
fi

# Untouched global conditional-create is enforced inside the evaluator core.
# Direct CLI invocation therefore cannot bypass the reviewed helper, immutable
# receipt, or exact readback-byte checks sealed by the session/prerequisite.

# No placeholder PASS is accepted; the evaluator emits raw evidence and the
# separate process reopens identities and recomputes the decision.
timeout --signal=TERM --kill-after=2m 150m \
  "$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --session "$SESSION" --expected-session-sha256 "$SESSION_SHA256" \
  --prerequisite-decision "$PREREQUISITE" \
  --expected-prerequisite-decision-sha256 "$PREREQUISITE_SHA256" \
  --checkpoint "$CHECKPOINT" --contract "$CONTRACT" \
  --panel-file "$PANEL_FILE" --data-root "$DATA_ROOT" \
  --expected-repo-commit "$EXPECTED_REPO_COMMIT"

RAW_SHA256=$("$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --sha256-file "$RAW_OUTPUT")
timeout --signal=TERM --kill-after=30s 10m \
  "$PYTHON" scripts/adjudicate_contracted_scientific_bundle.py \
  --protocol "$PROTOCOL" --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --session "$SESSION" --expected-session-sha256 "$SESSION_SHA256" \
  --prerequisite-decision "$PREREQUISITE" \
  --expected-prerequisite-decision-sha256 "$PREREQUISITE_SHA256" \
  --raw-evidence "$RAW_OUTPUT" --expected-raw-evidence-sha256 "$RAW_SHA256" \
  --checkpoint "$CHECKPOINT" --contract "$CONTRACT" \
  --panel-file "$PANEL_FILE" --data-root "$DATA_ROOT" \
  --expected-repo-commit "$EXPECTED_REPO_COMMIT"

[[ -s "$RAW_OUTPUT" && -s "$DECISION_OUTPUT" && -s "$STATE_ARCHIVE_OUTPUT" ]] || exit 2
grep -Fq '"formal_training_authorized": false' "$RAW_OUTPUT"
grep -Fq '"formal_training_authorized": false' "$DECISION_OUTPUT"
timeout 8m obsutil sync "$RUN_DIR" "$OBS_PREFIX/audit"
timeout 8m obsutil cp "$RAW_OUTPUT" "$OBS_PREFIX/raw.json"
timeout 8m obsutil cp "$DECISION_OUTPUT" "$OBS_PREFIX/decision.json"
timeout 8m obsutil cp "$STATE_ARCHIVE_OUTPUT" "$OBS_PREFIX/state_archive.npz"
mkdir "$READBACK_ONE" "$READBACK_TWO" "$READBACK_THREE"
timeout 8m obsutil cp "$OBS_PREFIX/raw.json" "$READBACK_ONE/raw.json"
timeout 8m obsutil cp "$OBS_PREFIX/decision.json" "$READBACK_TWO/decision.json"
timeout 8m obsutil cp "$OBS_PREFIX/state_archive.npz" "$READBACK_THREE/state_archive.npz"
[[ "$("$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --sha256-file "$READBACK_ONE/raw.json")" == "$RAW_SHA256" ]]
[[ "$("$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --sha256-file "$READBACK_TWO/decision.json")" \
   == "$("$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --sha256-file "$DECISION_OUTPUT")" ]]
[[ "$("$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --sha256-file "$READBACK_THREE/state_archive.npz")" \
   == "$("$PYTHON" scripts/contracted_scientific_bundle_eval.py \
  --sha256-file "$STATE_ARCHIVE_OUTPUT")" ]]
