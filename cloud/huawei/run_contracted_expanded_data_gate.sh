#!/usr/bin/env bash
# Qualify one expanded-data checkpoint through the development-seen panel only.
# This runner never opens an external/untouched panel and never starts formal training.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TORCHRUN=${TORCHRUN:-/data/venvs/deepjump/bin/torchrun}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-480}
HARD_STOP_UNIT="deepjump-contracted-expanded-data-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR=
OBS_DST=

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  if [[ "$code" != 0 ]] && [[ -n "$RUN_DIR" ]] && [[ -n "$OBS_DST" ]] \
    && command -v obsutil >/dev/null; then
    set +e
    failure_status=FAIL
    failure_readback="/tmp/contracted_expanded_data_failure_readback_${RUN_ID:-unknown}_$$"
    if [[ -d "$RUN_DIR/configs" ]] && [[ -d "$RUN_DIR/evidence" ]] \
      && [[ -d "$RUN_DIR/stages" ]]; then
      (
        cd "$RUN_DIR" || exit
        find configs evidence stages -type f -print0 \
          | LC_ALL=C sort -z \
          | xargs -0 sha256sum
      ) > "$RUN_DIR/failure_sha256.txt"
      timeout 300s obsutil sync "$RUN_DIR" "$OBS_DST/failure/audit"
      upload_code=$?
      mkdir -p "$failure_readback"
      timeout 300s obsutil sync "$OBS_DST/failure/audit" "$failure_readback"
      download_code=$?
      if [[ "$upload_code" == 0 ]] && [[ "$download_code" == 0 ]] \
        && cmp "$RUN_DIR/failure_sha256.txt" "$failure_readback/failure_sha256.txt" \
        && (cd "$failure_readback" && sha256sum -c failure_sha256.txt); then
        failure_status=PASS_OBS_FAILURE_READBACK
      fi
    fi
    "$PYTHON" - "$RUN_DIR/failure_readback_status.json" "$failure_status" \
      "${RUN_ID:-unknown}" "$OBS_DST/failure/audit" "$code" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

output, status, run_id, obs, original_exit = sys.argv[1:]
content = (json.dumps({
    "status": status,
    "run_id": run_id,
    "obs": obs,
    "original_exit": int(original_exit),
    "formal_training_authorized": False,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())
PY
    timeout 120s obsutil cp "$RUN_DIR/failure_readback_status.json" \
      "$OBS_DST/failure/failure_readback_status.json"
    timeout 120s obsutil cp "$OBS_DST/failure/failure_readback_status.json" \
      "$failure_readback/failure_readback_status.json"
    cmp "$RUN_DIR/failure_readback_status.json" \
      "$failure_readback/failure_readback_status.json"
  fi
  printf 'Contracted expanded-data gate exit=%s; requesting shutdown at %s\n' \
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
  printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2
  exit 2
}
[[ "$HARD_STOP_MINUTES" == 480 ]] || {
  printf 'HARD_STOP_MINUTES must remain 480\n' >&2
  exit 2
}
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n systemctl show "$HARD_STOP_UNIT.timer" \
  --property=ActiveState,SubState,NextElapseUSecRealtime --no-pager
sudo -n systemctl show "$HARD_STOP_UNIT.service" --property=ExecStart --no-pager \
  | grep -Fq '/usr/bin/systemctl poweroff'
sudo -n shutdown -c 2>/dev/null || true

# Require all run-specific identities only after the independent hard stop is active.
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set the reviewed deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set the authorized GPU hostname}
DATA_ROOT=${DATA_ROOT:?set the read-only qualified full-data root}
CONTRACT=${CONTRACT:?set the full-training contract path}
CONTRACT_SHA256=${CONTRACT_SHA256:?set the exact full-training contract SHA256}
MANIFEST=${MANIFEST:?set the contracted manifest path}
TRAIN_LIST=${TRAIN_LIST:?set the contracted 5,218-domain train-list path}
DEVELOPMENT_PANEL_FILE=${DEVELOPMENT_PANEL_FILE:?set the contracted development panel path}
DEVELOPMENT_PANEL_NAME=${DEVELOPMENT_PANEL_NAME:-legacy_dev20}
CONSUMPTION_LEDGER_ROOT=${CONSUMPTION_LEDGER_ROOT:-/data/deepjump-evaluation-consumption-ledger}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

[[ "$EXPECTED_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'EXPECTED_REPO_COMMIT must be a full lowercase commit SHA\n' >&2
  exit 2
}
[[ "$CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'CONTRACT_SHA256 must be lowercase SHA256\n' >&2
  exit 2
}
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  printf 'RUN_ID must be UTC basic timestamp YYYYMMDDTHHMMSSZ\n' >&2
  exit 2
}
[[ "$BUCKET" == obs://* ]] || { printf 'BUCKET must use obs://\n' >&2; exit 2; }
[[ "$DEVELOPMENT_PANEL_NAME" == legacy_dev20 ]] || {
  printf 'this runner is frozen to the legacy_dev20 development-seen panel\n' >&2
  exit 2
}
for path in "$DATA_ROOT" "$CONTRACT" "$MANIFEST" "$TRAIN_LIST" \
  "$DEVELOPMENT_PANEL_FILE" "$CONSUMPTION_LEDGER_ROOT"; do
  [[ "$path" == /* ]] || { printf 'contracted path must be absolute: %s\n' "$path" >&2; exit 2; }
done

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || {
  printf 'hostname mismatch\n' >&2
  exit 2
}
[[ -x "$PYTHON" && -x "$TORCHRUN" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
command -v nvidia-smi >/dev/null
command -v sha256sum >/dev/null

cd "$REPO"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$REPO/src"
for required_script in \
  scripts/train_ddp.py \
  scripts/validate_training_checkpoint.py \
  scripts/contracted_guarded_endpoint_panel_eval.py \
  scripts/adjudicate_contracted_guarded_endpoint_panel.py \
  scripts/verify_frozen_evaluation_identity.py \
  scripts/summarize_contracted_expanded_data_gate.py \
  scripts/verify_audit_readback.py \
  scripts/verify_obsutil_empty_prefix.py; do
  [[ -f "$required_script" && ! -L "$required_script" ]] || {
    printf 'required contracted script missing or symlinked: %s\n' "$required_script" >&2
    exit 2
  }
done
actual_commit=$(git rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'commit mismatch: actual=%s expected=%s\n' "$actual_commit" "$EXPECTED_REPO_COMMIT" >&2
  exit 2
}
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
  printf 'worktree is dirty or has untracked files\n' >&2
  exit 2
}

mount_options=$(findmnt -T "$DATA_ROOT" -n -o OPTIONS)
case ",$mount_options," in
  *,ro,*) ;;
  *) printf 'qualified DATA_ROOT mount must be read-only: %s\n' "$mount_options" >&2; exit 2 ;;
esac
[[ -f "$CONTRACT" && ! -L "$CONTRACT" ]] || { printf 'contract is not regular\n' >&2; exit 2; }
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || { printf 'manifest is not regular\n' >&2; exit 2; }
[[ -f "$TRAIN_LIST" && ! -L "$TRAIN_LIST" ]] || { printf 'train list is not regular\n' >&2; exit 2; }
[[ -f "$DEVELOPMENT_PANEL_FILE" && ! -L "$DEVELOPMENT_PANEL_FILE" ]] || {
  printf 'development panel is not regular\n' >&2
  exit 2
}

gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
[[ "$gpu_count" == 8 ]] || { printf 'GPU count %s != 8\n' "$gpu_count" >&2; exit 2; }
if pgrep -af '[s]cripts/(train_ddp|contracted_guarded_endpoint_panel_eval).py'; then
  printf 'conflicting training/evaluation process exists\n' >&2
  exit 2
fi
for service in deepjump-mdcath-download.service deepjump-mdcath-hash.service \
  deepjump-mdcath-copy.service; do
  if systemctl is-active --quiet "$service"; then
    printf 'full-data mutation service is still active: %s\n' "$service" >&2
    exit 2
  fi
done

RUN_DIR="$REPO/runs/contracted_expanded_data_gate_$RUN_ID"
READBACK_DIR="/tmp/contracted_expanded_data_gate_readback_$RUN_ID"
OBS_DST="$BUCKET/deepjump-full-data/contracted-expanded-data-gate/$RUN_ID"
for path in "$RUN_DIR" "$READBACK_DIR"; do
  [[ ! -e "$path" ]] || { printf 'refusing to overwrite %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$RUN_DIR/evidence" "$RUN_DIR/configs" "$RUN_DIR/stages"
# This persistent local ledger prevents automation replay but is not root-tamper
# proof or a cross-instance transaction service. Global claims need an external
# conditional-create API; ordinary obsutil copy/sync cannot provide that.
mkdir -p "$CONSUMPTION_LEDGER_ROOT"
[[ -d "$CONSUMPTION_LEDGER_ROOT" && ! -L "$CONSUMPTION_LEDGER_ROOT" ]] || {
  printf 'consumption ledger root must be a real directory\n' >&2
  exit 2
}
chmod 0700 "$CONSUMPTION_LEDGER_ROOT"
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

# Freeze every tracked repository byte and revalidate both Git identity and the
# byte manifest around every long-lived or authority-producing process.
SOURCE_IDENTITY_MANIFEST="$RUN_DIR/evidence/tracked_source_sha256.txt"
git ls-files -z | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "$SOURCE_IDENTITY_MANIFEST"
SOURCE_IDENTITY_MANIFEST_SHA256=$(sha256sum "$SOURCE_IDENTITY_MANIFEST" | awk '{print $1}')
verify_source_identity() {
  [[ "$(git rev-parse HEAD)" == "$actual_commit" ]] || {
    printf 'repository HEAD changed during the gate\n' >&2
    exit 2
  }
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || {
    printf 'repository worktree changed during the gate\n' >&2
    exit 2
  }
  [[ "$(sha256sum "$SOURCE_IDENTITY_MANIFEST" | awk '{print $1}')" == \
    "$SOURCE_IDENTITY_MANIFEST_SHA256" ]] || {
    printf 'tracked source identity manifest changed during the gate\n' >&2
    exit 2
  }
  sha256sum --check --quiet "$SOURCE_IDENTITY_MANIFEST" || {
    printf 'tracked repository bytes changed during the gate\n' >&2
    exit 2
  }
}
verify_source_identity

timeout 30s obsutil ls "$OBS_DST/" -limit=1 \
  | tee "$RUN_DIR/evidence/obs_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py \
  "$RUN_DIR/evidence/obs_prefix_preflight.log"

printf 'gate=contract_verify start=%s\n' "$(date -Is)"
"$PYTHON" - "$CONTRACT" "$CONTRACT_SHA256" "$DATA_ROOT" "$MANIFEST" \
  "$TRAIN_LIST" "$RUN_DIR/evidence/contract_verification.json" <<'PY'
import json
import os
import sys
from pathlib import Path

from deepjump.data_contract import verify_full_training_data_contract

contract, digest, root, manifest, train_list, output = sys.argv[1:]
report = verify_full_training_data_contract(
    contract,
    digest,
    configured_root=root,
    configured_manifest=manifest,
    configured_domains_file=train_list,
)
if report.get("status") != "PASS_FULL_TRAINING_DATA_CONTRACT":
    raise SystemExit("full-training contract did not pass")
content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
descriptor = os.open(Path(output), flags, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(report, sort_keys=True))
PY
CONTRACT_VERIFICATION="$RUN_DIR/evidence/contract_verification.json"
CONTRACT_VERIFICATION_SHA256=$(sha256sum "$CONTRACT_VERIFICATION" | awk '{print $1}')

printf 'gate=sealed_configs start=%s\n' "$(date -Is)"
verify_source_identity
"$PYTHON" - \
  configs/v100_tensorcloud01_full_expanded_d1_smoke100.yaml \
  configs/v100_tensorcloud01_full_expanded_d1_calibration1000.yaml \
  configs/v100_tensorcloud01_full_expanded_d1_development2000.yaml \
  "$RUN_DIR/configs/smoke.yaml" "$RUN_DIR/configs/calibration.yaml" \
  "$RUN_DIR/configs/development.yaml" \
  "$RUN_DIR/stages/smoke" "$RUN_DIR/stages/calibration" \
  "$RUN_DIR/stages/development" \
  "$DATA_ROOT" "$MANIFEST" "$TRAIN_LIST" "$CONTRACT" "$CONTRACT_SHA256" \
  "$RUN_DIR/evidence/config_identity.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

from deepjump.config import load_config, to_dict
from scripts.train_ddp import training_semantics_sha256

(smoke_source, calibration_source, development_source, smoke_output,
 calibration_output, development_output, smoke_run, calibration_run,
 development_run, data_root, manifest, train_list, contract, contract_sha256,
 report_output) = sys.argv[1:]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_exclusive(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())

def seal(source, destination, run_dir, expected_steps):
    cfg = load_config(source)
    if cfg.train.run_class != "full_data_stage" or cfg.train.max_steps != expected_steps:
        raise ValueError("source config is not the frozen full-data stage")
    if cfg.train.resume:
        raise ValueError("full-data stage source config must be fresh-init")
    cfg.data.root = data_root
    cfg.data.manifest = manifest
    cfg.data.domains = []
    cfg.data.domains_file = train_list
    cfg.data.full_training_contract = contract
    cfg.data.full_training_contract_sha256 = contract_sha256
    cfg.train.out_dir = run_dir
    payload = to_dict(cfg)
    content = yaml.safe_dump(payload, sort_keys=False).encode()
    write_exclusive(destination, content)
    reloaded = load_config(destination)
    if to_dict(reloaded) != payload:
        raise RuntimeError("sealed config did not round-trip exactly")
    return reloaded, {
        "source": str(Path(source).resolve()),
        "source_sha256": sha256(source),
        "sealed": str(Path(destination).resolve()),
        "sealed_sha256": sha256(destination),
        "max_steps": expected_steps,
        "lr_horizon_steps": reloaded.train.lr_horizon_steps,
        "training_semantics_sha256": training_semantics_sha256(reloaded),
    }

smoke, smoke_report = seal(smoke_source, smoke_output, smoke_run, 100)
calibration, calibration_report = seal(
    calibration_source, calibration_output, calibration_run, 1_000
)
development, development_report = seal(
    development_source, development_output, development_run, 2_000
)
semantics = {
    training_semantics_sha256(smoke),
    training_semantics_sha256(calibration),
    training_semantics_sha256(development),
}
if len(semantics) != 1:
    raise RuntimeError("smoke, calibration, and development semantics differ")
report = {
    "status": "PASS_SEALED_FULL_DATA_STAGE_CONFIGS",
    "smoke": smoke_report,
    "calibration": calibration_report,
    "development": development_report,
    "formal_training_authorized": False,
}
write_exclusive(
    report_output,
    (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
)
print(json.dumps(report, sort_keys=True))
PY
verify_source_identity

verify_source_identity
timeout --signal=TERM --kill-after=30s 12m \
  "$PYTHON" -m pytest -q \
  tests/test_full_training_data_contract.py \
  tests/test_full_training_entrypoint.py \
  tests/test_evaluation_consumption.py \
  tests/test_expanded_data_candidate_configs.py \
  tests/test_contracted_guarded_endpoint_panel_eval.py \
  tests/test_adjudicate_contracted_guarded_endpoint_panel.py \
  tests/test_contracted_expanded_data_gate_hardening.py \
  tests/test_contracted_expanded_data_gate_runner.py \
  2>&1 | tee "$RUN_DIR/evidence/pytest.log"
verify_source_identity

run_stage() {
  stage=$1
  config=$2
  timeout_minutes=$3
  log="$RUN_DIR/evidence/${stage}.log"
  printf 'gate=%s start=%s\n' "$stage" "$(date -Is)"
  verify_source_identity
  timeout --signal=TERM --kill-after=2m "${timeout_minutes}m" \
    "$TORCHRUN" --standalone --nproc_per_node=8 \
    scripts/train_ddp.py --config "$config" \
    --full-training-contract "$CONTRACT" \
    --expected-full-training-contract-sha256 "$CONTRACT_SHA256" \
    2>&1 | tee "$log"
  grep -q 'world=8 ' "$log"
  grep -q 'done. artifacts in' "$log"
  if grep -Eiq 'FloatingPointError|non-finite|out of memory|NCCL[^[:space:]]* (error|failed)' "$log"; then
    printf '%s contains a fatal numerical/runtime signature\n' "$stage" >&2
    exit 2
  fi
  if grep -Eq 'scaler_skips [1-9][0-9]*' "$log"; then
    printf '%s contains one or more GradScaler skipped updates\n' "$stage" >&2
    exit 2
  fi
  verify_source_identity
}

run_stage eight_gpu_smoke "$RUN_DIR/configs/smoke.yaml" 20
SMOKE_CHECKPOINT="$RUN_DIR/stages/smoke/last.ckpt"
SMOKE_CHECKPOINT_SHA256=$(sha256sum "$SMOKE_CHECKPOINT" | awk '{print $1}')
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$SMOKE_CHECKPOINT" \
  --expected-checkpoint-sha256 "$SMOKE_CHECKPOINT_SHA256" \
  --history "$RUN_DIR/stages/smoke/history.json" \
  --expected-step 100 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --expected-config "$RUN_DIR/configs/smoke.yaml" \
  --expected-contract-verification "$CONTRACT_VERIFICATION" \
  --expected-contract-verification-sha256 "$CONTRACT_VERIFICATION_SHA256" \
  --expected-lr-horizon-steps 500000 \
  --output "$RUN_DIR/evidence/smoke_checkpoint_gate.json"
SMOKE_CHECKPOINT_GATE="$RUN_DIR/evidence/smoke_checkpoint_gate.json"
SMOKE_CHECKPOINT_GATE_SHA256=$(sha256sum "$SMOKE_CHECKPOINT_GATE" | awk '{print $1}')

run_stage short_calibration "$RUN_DIR/configs/calibration.yaml" 60
CALIBRATION_CHECKPOINT="$RUN_DIR/stages/calibration/last.ckpt"
CALIBRATION_CHECKPOINT_SHA256=$(sha256sum "$CALIBRATION_CHECKPOINT" | awk '{print $1}')
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$CALIBRATION_CHECKPOINT" \
  --expected-checkpoint-sha256 "$CALIBRATION_CHECKPOINT_SHA256" \
  --history "$RUN_DIR/stages/calibration/history.json" \
  --expected-step 1000 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --expected-config "$RUN_DIR/configs/calibration.yaml" \
  --expected-contract-verification "$CONTRACT_VERIFICATION" \
  --expected-contract-verification-sha256 "$CONTRACT_VERIFICATION_SHA256" \
  --expected-lr-horizon-steps 500000 \
  --output "$RUN_DIR/evidence/calibration_checkpoint_gate.json"
CALIBRATION_CHECKPOINT_GATE="$RUN_DIR/evidence/calibration_checkpoint_gate.json"
CALIBRATION_CHECKPOINT_GATE_SHA256=$(sha256sum "$CALIBRATION_CHECKPOINT_GATE" | awk '{print $1}')

# The development discriminator is a separate fresh continuous 0->2000 run.
# Resuming calibration would not reproduce worker/crop RNG bitwise and would
# confound the frozen learning trajectory.
run_stage fresh_development2000 "$RUN_DIR/configs/development.yaml" 60
DEVELOPMENT_CHECKPOINT="$RUN_DIR/stages/development/last.ckpt"
DEVELOPMENT_CHECKPOINT_SHA256=$(sha256sum "$DEVELOPMENT_CHECKPOINT" | awk '{print $1}')
"$PYTHON" scripts/validate_training_checkpoint.py \
  --checkpoint "$DEVELOPMENT_CHECKPOINT" \
  --expected-checkpoint-sha256 "$DEVELOPMENT_CHECKPOINT_SHA256" \
  --history "$RUN_DIR/stages/development/history.json" \
  --expected-step 2000 --expected-world-size 8 --expected-delta 1 \
  --require-full-tensor --expected-config "$RUN_DIR/configs/development.yaml" \
  --expected-contract-verification "$CONTRACT_VERIFICATION" \
  --expected-contract-verification-sha256 "$CONTRACT_VERIFICATION_SHA256" \
  --expected-lr-horizon-steps 500000 \
  --output "$RUN_DIR/evidence/development_checkpoint_gate.json"
DEVELOPMENT_CHECKPOINT_GATE="$RUN_DIR/evidence/development_checkpoint_gate.json"
DEVELOPMENT_CHECKPOINT_GATE_SHA256=$(sha256sum "$DEVELOPMENT_CHECKPOINT_GATE" | awk '{print $1}')

verify_checkpoint_identities() {
  [[ "$(sha256sum "$SMOKE_CHECKPOINT" | awk '{print $1}')" == \
    "$SMOKE_CHECKPOINT_SHA256" ]] || { printf 'smoke checkpoint changed\n' >&2; exit 2; }
  [[ "$(sha256sum "$CALIBRATION_CHECKPOINT" | awk '{print $1}')" == \
    "$CALIBRATION_CHECKPOINT_SHA256" ]] || {
    printf 'calibration checkpoint changed\n' >&2
    exit 2
  }
  [[ "$(sha256sum "$DEVELOPMENT_CHECKPOINT" | awk '{print $1}')" == \
    "$DEVELOPMENT_CHECKPOINT_SHA256" ]] || {
    printf 'development checkpoint changed\n' >&2
    exit 2
  }
}

verify_readback_checkpoint_identities() {
  local readback_root=$1
  [[ "$(sha256sum "$readback_root/stages/smoke/last.ckpt" | awk '{print $1}')" == \
    "$SMOKE_CHECKPOINT_SHA256" ]] || return 1
  [[ "$(sha256sum "$readback_root/stages/calibration/last.ckpt" | awk '{print $1}')" == \
    "$CALIBRATION_CHECKPOINT_SHA256" ]] || return 1
  [[ "$(sha256sum "$readback_root/stages/development/last.ckpt" | awk '{print $1}')" == \
    "$DEVELOPMENT_CHECKPOINT_SHA256" ]] || return 1
}

# This capability authorizes only the already-seen development panel. It is
# created after all three bounded training stages pass and remains formal=false.
"$PYTHON" - "$DEVELOPMENT_CHECKPOINT" "$DEVELOPMENT_CHECKPOINT_SHA256" \
  "$CONTRACT" "$CONTRACT_SHA256" "$DEVELOPMENT_PANEL_NAME" \
  "$DEVELOPMENT_PANEL_FILE" "$CONSUMPTION_LEDGER_ROOT" "$RUN_ID" \
  "$actual_commit" "$OBS_DST" "$SOURCE_IDENTITY_MANIFEST_SHA256" \
  "$SMOKE_CHECKPOINT_GATE_SHA256" "$SMOKE_CHECKPOINT_SHA256" \
  "$CALIBRATION_CHECKPOINT_GATE_SHA256" "$CALIBRATION_CHECKPOINT_SHA256" \
  "$DEVELOPMENT_CHECKPOINT_GATE_SHA256" \
  "$RUN_DIR/evidence/development_authorization.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from deepjump.evaluation_contract import verify_frozen_evaluation_identity

(checkpoint, checkpoint_sha, contract, contract_sha, panel_name, panel_file,
 ledger_root, run_id, commit, obs, source_sha, smoke_gate_sha, smoke_checkpoint_sha,
 calibration_gate_sha, calibration_checkpoint_sha, development_gate_sha,
 output) = sys.argv[1:]
identity = verify_frozen_evaluation_identity(
    checkpoint,
    contract,
    contract_sha,
    expected_checkpoint_sha256=checkpoint_sha,
    expected_checkpoint_step=2_000,
    phase="development",
    panel_name=panel_name,
    panel_file=panel_file,
)
payload = {
    "schema": "deepjump.reserved_evaluation_authorization.v2",
    "authorization_id": f"development-{run_id}",
    "consumption_ledger_root": str(Path(ledger_root).resolve()),
    "status": "ADVANCE_EXPANDED_DATA_DEVELOPMENT",
    "phase": "development",
    "checkpoint_sha256": checkpoint_sha,
    "checkpoint_step": 2_000,
    "full_training_contract_sha256": contract_sha,
    "panel_name": panel_name,
    "panel_sha256": identity["panel_sha256"],
    "reserved_panel_authorized": True,
    "formal_training_authorized": False,
    "run_binding": {
        "run_id": run_id,
        "commit": commit,
        "obs": obs,
        "source_identity_manifest_sha256": source_sha,
        "checkpoint_gates": {
            "smoke": {
                "status": "PASS",
                "sha256": smoke_gate_sha,
                "checkpoint_sha256": smoke_checkpoint_sha,
                "checkpoint_step": 100,
            },
            "calibration": {
                "status": "PASS",
                "sha256": calibration_gate_sha,
                "checkpoint_sha256": calibration_checkpoint_sha,
                "checkpoint_step": 1_000,
            },
            "development": {
                "status": "PASS",
                "sha256": development_gate_sha,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": 2_000,
            },
        },
    },
}
content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())
print(hashlib.sha256(content).hexdigest())
PY
DEVELOPMENT_AUTHORIZATION="$RUN_DIR/evidence/development_authorization.json"
DEVELOPMENT_AUTHORIZATION_SHA256=$(sha256sum "$DEVELOPMENT_AUTHORIZATION" | awk '{print $1}')

printf 'gate=development20 start=%s\n' "$(date -Is)"
verify_source_identity
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=2m 260m \
  "$PYTHON" scripts/contracted_guarded_endpoint_panel_eval.py \
  --checkpoint "$DEVELOPMENT_CHECKPOINT" \
  --expected-checkpoint-sha256 "$DEVELOPMENT_CHECKPOINT_SHA256" \
  --expected-checkpoint-step 2000 \
  --contract "$CONTRACT" --expected-contract-sha256 "$CONTRACT_SHA256" \
  --phase development --panel-name "$DEVELOPMENT_PANEL_NAME" \
  --panel-file "$DEVELOPMENT_PANEL_FILE" \
  --prerequisite-decision "$DEVELOPMENT_AUTHORIZATION" \
  --expected-prerequisite-decision-sha256 "$DEVELOPMENT_AUTHORIZATION_SHA256" \
  --runtime-probe-output "$RUN_DIR/evidence/development_runtime_probe.json" \
  --output "$RUN_DIR/evidence/development_result.json" \
  > "$RUN_DIR/evidence/development_eval.log" 2>&1
DEVELOPMENT_RESULT="$RUN_DIR/evidence/development_result.json"
DEVELOPMENT_RESULT_SHA256=$(sha256sum "$DEVELOPMENT_RESULT" | awk '{print $1}')
DEVELOPMENT_RUNTIME_PROBE="$RUN_DIR/evidence/development_runtime_probe.json"
DEVELOPMENT_RUNTIME_PROBE_SHA256=$(sha256sum "$DEVELOPMENT_RUNTIME_PROBE" | awk '{print $1}')
verify_source_identity

printf 'gate=development_adjudication start=%s\n' "$(date -Is)"
verify_source_identity
"$PYTHON" -m scripts.adjudicate_contracted_guarded_endpoint_panel \
  --result "$DEVELOPMENT_RESULT" \
  --expected-result-sha256 "$DEVELOPMENT_RESULT_SHA256" \
  --checkpoint "$DEVELOPMENT_CHECKPOINT" \
  --expected-checkpoint-sha256 "$DEVELOPMENT_CHECKPOINT_SHA256" \
  --expected-checkpoint-step 2000 \
  --contract "$CONTRACT" --expected-contract-sha256 "$CONTRACT_SHA256" \
  --phase development --panel-name "$DEVELOPMENT_PANEL_NAME" \
  --panel-file "$DEVELOPMENT_PANEL_FILE" \
  --prerequisite-decision "$DEVELOPMENT_AUTHORIZATION" \
  --expected-prerequisite-decision-sha256 "$DEVELOPMENT_AUTHORIZATION_SHA256" \
  --runtime-probe "$DEVELOPMENT_RUNTIME_PROBE" \
  --expected-runtime-probe-sha256 "$DEVELOPMENT_RUNTIME_PROBE_SHA256" \
  --output "$RUN_DIR/evidence/development_decision.json"
verify_source_identity
DEVELOPMENT_DECISION="$RUN_DIR/evidence/development_decision.json"
DEVELOPMENT_DECISION_SHA256=$(sha256sum "$DEVELOPMENT_DECISION" | awk '{print $1}')
DEVELOPMENT_PANEL_SHA256=$(sha256sum "$DEVELOPMENT_PANEL_FILE" | awk '{print $1}')
"$PYTHON" scripts/summarize_contracted_expanded_data_gate.py \
  --decision "$DEVELOPMENT_DECISION" \
  --expected-decision-sha256 "$DEVELOPMENT_DECISION_SHA256" \
  --expected-result-sha256 "$DEVELOPMENT_RESULT_SHA256" \
  --expected-checkpoint-sha256 "$DEVELOPMENT_CHECKPOINT_SHA256" \
  --expected-contract-sha256 "$CONTRACT_SHA256" \
  --expected-panel-name "$DEVELOPMENT_PANEL_NAME" \
  --expected-panel-sha256 "$DEVELOPMENT_PANEL_SHA256" \
  --expected-prerequisite-decision-sha256 "$DEVELOPMENT_AUTHORIZATION_SHA256" \
  --runtime-probe "$DEVELOPMENT_RUNTIME_PROBE" \
  --expected-runtime-probe-sha256 "$DEVELOPMENT_RUNTIME_PROBE_SHA256" \
  --run-id "$RUN_ID" --commit "$actual_commit" --obs "$OBS_DST" \
  --source-identity-manifest-sha256 "$SOURCE_IDENTITY_MANIFEST_SHA256" \
  --smoke-checkpoint-gate "$SMOKE_CHECKPOINT_GATE" \
  --expected-smoke-checkpoint-gate-sha256 "$SMOKE_CHECKPOINT_GATE_SHA256" \
  --smoke-checkpoint "$SMOKE_CHECKPOINT" \
  --expected-smoke-checkpoint-sha256 "$SMOKE_CHECKPOINT_SHA256" \
  --calibration-checkpoint-gate "$CALIBRATION_CHECKPOINT_GATE" \
  --expected-calibration-checkpoint-gate-sha256 "$CALIBRATION_CHECKPOINT_GATE_SHA256" \
  --calibration-checkpoint "$CALIBRATION_CHECKPOINT" \
  --expected-calibration-checkpoint-sha256 "$CALIBRATION_CHECKPOINT_SHA256" \
  --development-checkpoint-gate "$DEVELOPMENT_CHECKPOINT_GATE" \
  --expected-development-checkpoint-gate-sha256 "$DEVELOPMENT_CHECKPOINT_GATE_SHA256" \
  --development-checkpoint "$DEVELOPMENT_CHECKPOINT" \
  --output "$RUN_DIR/evidence/summary.json"
SUMMARY_SHA256=$(sha256sum "$RUN_DIR/evidence/summary.json" | awk '{print $1}')

verify_checkpoint_chain() {
  local root=$1
  "$PYTHON" - "$root" "$SUMMARY_SHA256" \
    "$SMOKE_CHECKPOINT_GATE_SHA256" "$SMOKE_CHECKPOINT_SHA256" \
    "$CALIBRATION_CHECKPOINT_GATE_SHA256" "$CALIBRATION_CHECKPOINT_SHA256" \
    "$DEVELOPMENT_CHECKPOINT_GATE_SHA256" "$DEVELOPMENT_CHECKPOINT_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from deepjump.data_contract import _read_regular_bytes

(
    root, summary_sha, smoke_gate_sha, smoke_checkpoint_sha,
    calibration_gate_sha, calibration_checkpoint_sha,
    development_gate_sha, development_checkpoint_sha,
) = sys.argv[1:]
root = Path(root)

def load_json(relative, expected_sha, label):
    raw = _read_regular_bytes(root / relative, label)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"{label} SHA256 mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload

summary = load_json("evidence/summary.json", summary_sha, "expanded summary")
bindings = {}
for stage, step, gate_sha, checkpoint_sha in (
    ("smoke", 100, smoke_gate_sha, smoke_checkpoint_sha),
    ("calibration", 1_000, calibration_gate_sha, calibration_checkpoint_sha),
    ("development", 2_000, development_gate_sha, development_checkpoint_sha),
):
    gate = load_json(
        f"evidence/{stage}_checkpoint_gate.json", gate_sha, f"{stage} checkpoint gate"
    )
    checkpoint = _read_regular_bytes(
        root / f"stages/{stage}/last.ckpt", f"{stage} checkpoint"
    )
    if hashlib.sha256(checkpoint).hexdigest() != checkpoint_sha:
        raise ValueError(f"{stage} checkpoint SHA256 mismatch")
    expected = {
        "status": "PASS",
        "sha256": gate_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": step,
    }
    if any(gate.get(key) != value for key, value in expected.items() if key != "sha256"):
        raise ValueError(f"{stage} checkpoint gate identity mismatch")
    bindings[stage] = expected
if summary.get("checkpoint_gates") != bindings:
    raise ValueError("summary checkpoint-gate binding mismatch")
PY
}

verify_source_identity
verify_checkpoint_identities
verify_checkpoint_chain "$RUN_DIR"

printf 'gate=obs_archive_and_readback start=%s\n' "$(date -Is)"
verify_source_identity
verify_checkpoint_identities
(
  cd "$RUN_DIR"
  find configs evidence stages -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "$RUN_DIR/audit_sha256.txt"
AUDIT_SHA256=$(sha256sum "$RUN_DIR/audit_sha256.txt" | awk '{print $1}')
timeout --signal=TERM --kill-after=30s 8m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/initial"
timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$OBS_DST/audit" "$READBACK_DIR/initial"
cmp "$RUN_DIR/audit_sha256.txt" "$READBACK_DIR/initial/audit_sha256.txt"
(cd "$READBACK_DIR/initial" && sha256sum -c audit_sha256.txt)
"$PYTHON" scripts/verify_audit_readback.py \
  --root "$READBACK_DIR/initial" --allow-relative runner.log
verify_readback_checkpoint_identities "$READBACK_DIR/initial"
verify_checkpoint_identities
verify_checkpoint_chain "$READBACK_DIR/initial"

"$PYTHON" - "$RUN_DIR/readback_completion.json" "$RUN_ID" "$AUDIT_SHA256" \
  "$DEVELOPMENT_DECISION_SHA256" "$SUMMARY_SHA256" \
  "$SOURCE_IDENTITY_MANIFEST_SHA256" "$actual_commit" "$OBS_DST" \
  "$SMOKE_CHECKPOINT_GATE_SHA256" "$SMOKE_CHECKPOINT_SHA256" \
  "$CALIBRATION_CHECKPOINT_GATE_SHA256" "$CALIBRATION_CHECKPOINT_SHA256" \
  "$DEVELOPMENT_CHECKPOINT_GATE_SHA256" "$DEVELOPMENT_CHECKPOINT_SHA256" <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.contracted_guarded_endpoint_panel_eval import _atomic_json_new

(
    output, run_id, audit_sha, decision_sha, summary_sha, source_sha, commit, obs,
    smoke_gate_sha, smoke_checkpoint_sha, calibration_gate_sha,
    calibration_checkpoint_sha, development_gate_sha, development_checkpoint_sha,
) = sys.argv[1:]
_atomic_json_new(Path(output), {
    "status": "PASS_OBS_READBACK",
    "run_id": run_id,
    "audit_sha256": audit_sha,
    "decision_sha256": decision_sha,
    "summary_sha256": summary_sha,
    "source_identity_manifest_sha256": source_sha,
    "commit": commit,
    "obs": obs,
    "checkpoint_gates": {
        "smoke": {
            "status": "PASS",
            "sha256": smoke_gate_sha,
            "checkpoint_sha256": smoke_checkpoint_sha,
            "checkpoint_step": 100,
        },
        "calibration": {
            "status": "PASS",
            "sha256": calibration_gate_sha,
            "checkpoint_sha256": calibration_checkpoint_sha,
            "checkpoint_step": 1_000,
        },
        "development": {
            "status": "PASS",
            "sha256": development_gate_sha,
            "checkpoint_sha256": development_checkpoint_sha,
            "checkpoint_step": 2_000,
        },
    },
    "formal_training_authorized": False,
    "completed_at": datetime.now(timezone.utc).isoformat(),
})
PY
(cd "$RUN_DIR" && sha256sum readback_completion.json > readback_completion.sha256)
verify_checkpoint_identities
verify_checkpoint_chain "$RUN_DIR"
timeout --signal=TERM --kill-after=30s 8m obsutil sync "$RUN_DIR" "$OBS_DST/audit"
mkdir -p "$READBACK_DIR/completion"
timeout --signal=TERM --kill-after=30s 8m obsutil sync \
  "$OBS_DST/audit" "$READBACK_DIR/completion"
cmp "$RUN_DIR/readback_completion.sha256" \
  "$READBACK_DIR/completion/readback_completion.sha256"
(cmp "$RUN_DIR/audit_sha256.txt" "$READBACK_DIR/completion/audit_sha256.txt")
(cd "$READBACK_DIR/completion" && sha256sum -c audit_sha256.txt)
(cd "$READBACK_DIR/completion" && sha256sum -c readback_completion.sha256)
"$PYTHON" scripts/verify_audit_readback.py \
  --root "$READBACK_DIR/completion" \
  --allow-relative runner.log \
  --allow-relative readback_completion.json \
  --allow-relative readback_completion.sha256
verify_readback_checkpoint_identities "$READBACK_DIR/completion"
verify_checkpoint_chain "$READBACK_DIR/completion"
verify_source_identity

printf 'Contracted expanded-data development gate complete; external, untouched, and formal training were not started.\n'
