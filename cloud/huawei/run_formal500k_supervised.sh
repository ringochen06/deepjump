#!/usr/bin/env -S -i DEEPJUMP_SANITIZED_LAUNCH=1 HOME=/root LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin TMPDIR=/tmp /bin/bash --noprofile --norc
# Fail-closed supervisor for an explicitly authorized exact formal500k package.
# This script never creates an authorization. Fresh runs and recovery attempts
# are atomically claimed, and a soft stop always precedes the host hard stop.
set -euo pipefail

MODE=${1:-}
[ "$MODE" = fresh ] || [ "$MODE" = recovery ] || {
  echo "usage: $0 fresh|recovery PACKAGE PACKAGE_SHA AUTH AUTH_SHA AUTH_VERIFICATION AUTH_VERIFICATION_SHA" >&2
  exit 2
}
[ "${DEEPJUMP_SANITIZED_LAUNCH:-}" = 1 ] || {
  echo "!! supervisor must be executed directly through its sanitized shebang" >&2
  exit 2
}
[ "$#" -eq 7 ] || {
  echo "!! exact package and authorization arguments are required" >&2
  exit 2
}
FORMAL_PACKAGE=$2
FORMAL_PACKAGE_SHA256=$3
FORMAL_AUTHORIZATION=$4
FORMAL_AUTHORIZATION_SHA256=$5
FORMAL_AUTHORIZATION_VERIFICATION=$6
FORMAL_AUTHORIZATION_VERIFICATION_SHA256=$7
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0
SUPERVISOR=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

BOOTSTRAP_PYTHON=$(command -v python3) || {
  echo "!! python3 is required to verify the exact package" >&2
  exit 1
}
runtime_assignments=$("$BOOTSTRAP_PYTHON" - \
  "$FORMAL_PACKAGE" "$FORMAL_PACKAGE_SHA256" <<'PY'
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

package_path, expected_sha = sys.argv[1:]
if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
    raise SystemExit("formal package SHA256 is invalid")
raw = Path(package_path).read_bytes()
if hashlib.sha256(raw).hexdigest() != expected_sha:
    raise SystemExit("formal package SHA256 mismatch")
package = json.loads(raw)
candidate = package.get("formal_candidate")
plan = candidate.get("execution_plan") if isinstance(candidate, dict) else None
if not isinstance(plan, dict):
    raise SystemExit("formal package execution_plan is missing")
toolchain = plan.get("toolchain")
if not isinstance(toolchain, dict):
    raise SystemExit("formal package toolchain is missing")
checkpoint_plan = package.get("checkpoint_plan")
if not isinstance(checkpoint_plan, dict):
    raise SystemExit("formal package checkpoint_plan is missing")
runtime_identity = package.get("runtime_identity")
if not isinstance(runtime_identity, dict):
    raise SystemExit("formal package runtime_identity is missing")
recovery_plan = package.get("recovery_plan")
if not isinstance(recovery_plan, dict):
    raise SystemExit("formal package recovery_plan is missing")
values = {
    "REPO_ROOT": plan.get("repo_root"),
    "FORMAL_CONFIG": plan.get("config_path"),
    "CONTRACT_VERIFICATION": plan.get("contract_verification_path"),
    "CONTRACT_VERIFICATION_SHA256": plan.get("contract_verification_sha256"),
    "FULL_TRAINING_CONTRACT": plan.get("full_training_contract_path"),
    "FULL_TRAINING_CONTRACT_SHA256": plan.get("full_training_contract_sha256"),
    "EXPECTED_DATA_UUID": plan.get("data_uuid"),
    "EXPECTED_COMMIT": plan.get("reviewed_commit"),
    "OBS_DST": plan.get("obs_dst"),
    "RUN_ID": plan.get("run_id"),
    "EXPECTED_RUN_DIR": plan.get("run_dir"),
    "PYTHON": (toolchain.get("python") or {}).get("path"),
    "PYTHON_SHA256": (toolchain.get("python") or {}).get("sha256"),
    "TORCHRUN": (toolchain.get("torchrun") or {}).get("path"),
    "TORCHRUN_SHA256": (toolchain.get("torchrun") or {}).get("sha256"),
    "OBSUTIL": (toolchain.get("obsutil") or {}).get("path"),
    "OBSUTIL_SHA256": (toolchain.get("obsutil") or {}).get("sha256"),
    "VALIDATOR": plan.get("validator_path"),
    "VALIDATOR_SHA256": plan.get("validator_sha256"),
    "EMPTY_PREFIX_VALIDATOR": plan.get("empty_prefix_validator_path"),
    "EMPTY_PREFIX_VALIDATOR_SHA256": plan.get("empty_prefix_validator_sha256"),
    "ARCHIVER": plan.get("archiver_path"),
    "ARCHIVER_SHA256": plan.get("archiver_sha256"),
    "TRAINER": plan.get("trainer_path"),
    "TRAINER_SHA256": plan.get("trainer_sha256"),
    "KEEP_LOCAL_VERIFIED": checkpoint_plan.get("archiver_keep_local_verified"),
    "ARCHIVE_POLL_SECONDS": plan.get("archive_poll_seconds"),
    "SOFT_STOP_MINUTES": plan.get("soft_stop_minutes"),
    "HARD_STOP_MINUTES": plan.get("hard_stop_minutes"),
    "ARCHIVE_GRACE_SECONDS": plan.get("archive_kill_grace_seconds"),
    "EXPECTED_HOSTNAME": runtime_identity.get("hostname"),
    "EXPECTED_PRODUCT_UUID": runtime_identity.get("product_uuid"),
    "EXPECTED_PRODUCT_SERIAL": runtime_identity.get("product_serial"),
    "EXPECTED_GPU_MODEL": runtime_identity.get("gpu_model"),
    "MAX_RECOVERY_ATTEMPTS": recovery_plan.get("max_recovery_attempts"),
}
for name, value in values.items():
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise SystemExit(f"formal package execution value is missing: {name}")
    print(f"{name}={shlex.quote(str(value))}")
PY
) || exit 1
eval "$runtime_assignments"

fail() {
  echo "!! $*" >&2
  exit 1
}

"$BOOTSTRAP_PYTHON" - \
  "$FORMAL_PACKAGE" "$FORMAL_PACKAGE_SHA256" \
  "$FORMAL_AUTHORIZATION" "$FORMAL_AUTHORIZATION_SHA256" \
  "$FORMAL_AUTHORIZATION_VERIFICATION" \
  "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

package_path, package_sha, authorization_path, authorization_sha, verified_path, verified_sha = (
    sys.argv[1:]
)
def read_bound(path, expected, label):
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise SystemExit(f"{label} SHA256 is invalid")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(f"{label} SHA256 mismatch")
    return json.loads(raw)

package = read_bound(package_path, package_sha, "formal package")
authorization = read_bound(authorization_path, authorization_sha, "formal authorization")
verified = read_bound(verified_path, verified_sha, "authorization verification")
if package.get("package_ready") is not True or package.get("formal_training_authorized") is not False:
    raise SystemExit("formal package is not ready and unauthorized")
authorization_keys = {
    "schema", "status", "formal_training_authorized", "authorization_id",
    "issued_at", "expires_at", "authorized_package_sha256",
    "authorized_package_id", "authorized_run_id", "authorized_reviewed_commit",
    "scope", "approval_record_sha256",
}
verification_keys = {
    "schema", "status", "formal_training_authorized", "package_sha256",
    "authorization_sha256", "authorization_verifier_sha256", "package_id",
    "run_id", "reviewed_commit", "config_sha256", "target_total_steps",
    "execution_plan", "checkpoint_plan", "stop_plan", "recovery_plan", "scope",
}
if set(authorization) != authorization_keys or set(verified) != verification_keys:
    raise SystemExit("formal authorization or verification schema is not exact")
if (
    authorization.get("schema") != "deepjump.formal500k.user_authorization.v1"
    or authorization.get("status") != "USER_AUTHORIZED_FORMAL_TRAINING"
    or authorization.get("formal_training_authorized") is not True
    or authorization.get("authorized_package_sha256") != package_sha
):
    raise SystemExit("formal authorization does not bind the package")
if (
    verified.get("schema") != "deepjump.formal500k.authorization_verification.v1"
    or verified.get("status") != "PASS_FORMAL_TRAINING_AUTHORIZATION"
    or verified.get("formal_training_authorized") is not True
    or verified.get("package_sha256") != package_sha
    or verified.get("authorization_sha256") != authorization_sha
):
    raise SystemExit("authorization verification does not bind the package and authorization")
python_entry = package["formal_candidate"]["execution_plan"]["toolchain"]["python"]
python_path = Path(python_entry["path"])
if python_path.resolve() != python_path:
    raise SystemExit("package Python path is not canonical")
if hashlib.sha256(python_path.read_bytes()).hexdigest() != python_entry.get("sha256"):
    raise SystemExit("package Python SHA256 mismatch")
PY

[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail "RUN_ID must be an immutable UTC timestamp"
[[ "$OBS_DST" == obs://* ]] || fail "OBS_DST must be an obs:// prefix"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_COMMIT is invalid"
[[ "$SOFT_STOP_MINUTES" =~ ^[1-9][0-9]*$ ]] || fail "SOFT_STOP_MINUTES must be positive"
[[ "$HARD_STOP_MINUTES" =~ ^[1-9][0-9]*$ ]] || fail "HARD_STOP_MINUTES must be positive"
[[ "$ARCHIVE_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "ARCHIVE_GRACE_SECONDS must be positive"
[[ "$KEEP_LOCAL_VERIFIED" =~ ^[1-9][0-9]*$ ]] || fail "KEEP_LOCAL_VERIFIED must be positive"
[[ "$ARCHIVE_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "ARCHIVE_POLL_SECONDS must be positive"
[ "$SOFT_STOP_MINUTES" -lt "$HARD_STOP_MINUTES" ] ||
  fail "soft stop must precede hard stop"
[ $((HARD_STOP_MINUTES - SOFT_STOP_MINUTES)) -ge 30 ] ||
  fail "soft stop must leave at least thirty minutes for final archival"
git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
  fail "REPO_ROOT is not a Git worktree"
[ -f "$FORMAL_CONFIG" ] || fail "formal config not found"
[ -f "$FORMAL_PACKAGE" ] || fail "formal package not found"
[ -f "$FORMAL_AUTHORIZATION" ] || fail "formal authorization not found"
[ -f "$FORMAL_AUTHORIZATION_VERIFICATION" ] || fail "authorization verification not found"
[ -f "$CONTRACT_VERIFICATION" ] || fail "contract verification not found"
[ -f "$FULL_TRAINING_CONTRACT" ] || fail "full training contract not found"
[ -x "$ARCHIVER" ] || fail "checkpoint archiver is not executable"
[ -f "$VALIDATOR" ] || fail "checkpoint validator not found"
[ -f "$EMPTY_PREFIX_VALIDATOR" ] || fail "OBS empty-prefix validator not found"
[ -f "$TRAINER" ] || fail "trainer not found"
command -v "$PYTHON" >/dev/null || fail "python not found: $PYTHON"
command -v "$TORCHRUN" >/dev/null || fail "torchrun not found: $TORCHRUN"
command -v "$OBSUTIL" >/dev/null || fail "obsutil not found: $OBSUTIL"

current_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
[ "$current_commit" = "$EXPECTED_COMMIT" ] ||
  fail "deployed commit differs from EXPECTED_COMMIT"
[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ] ||
  fail "deployed repository is not clean"

preflight=$("$PYTHON" - \
  "$FORMAL_CONFIG" "$FORMAL_PACKAGE" "$FORMAL_PACKAGE_SHA256" \
  "$FORMAL_AUTHORIZATION" "$FORMAL_AUTHORIZATION_SHA256" \
  "$FORMAL_AUTHORIZATION_VERIFICATION" "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" \
  "$CONTRACT_VERIFICATION" "$CONTRACT_VERIFICATION_SHA256" \
  "$FULL_TRAINING_CONTRACT" "$FULL_TRAINING_CONTRACT_SHA256" \
  "$EXPECTED_COMMIT" "$RUN_ID" "$OBS_DST" "$EXPECTED_DATA_UUID" "$REPO_ROOT" \
  "$EXPECTED_RUN_DIR" "$SUPERVISOR" "$ARCHIVER" "$VALIDATOR" \
  "$EMPTY_PREFIX_VALIDATOR" "$TRAINER" \
  "$PYTHON" "$TORCHRUN" "$OBSUTIL" \
  "$SOFT_STOP_MINUTES" "$HARD_STOP_MINUTES" "$ARCHIVE_GRACE_SECONDS" \
  "$KEEP_LOCAL_VERIFIED" "$ARCHIVE_POLL_SECONDS" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

(
    config_path, package_path, package_sha, raw_authorization_path,
    raw_authorization_sha, authorization_path, authorization_sha,
    verification_path, verification_sha, contract_path, contract_sha,
    expected_commit, run_id, obs_dst, data_uuid, repo_root, expected_run_dir,
    supervisor_path, archiver_path, validator_path, empty_prefix_validator_path,
    trainer_path, python_path, torchrun_path, obsutil_path, soft_stop_minutes,
    hard_stop_minutes, archive_grace_seconds,
    keep_local_verified, archive_poll_seconds,
) = sys.argv[1:]
soft_stop_minutes = int(soft_stop_minutes)
hard_stop_minutes = int(hard_stop_minutes)
archive_grace_seconds = int(archive_grace_seconds)
keep_local_verified = int(keep_local_verified)
archive_poll_seconds = int(archive_poll_seconds)

HEX64 = re.compile(r"[0-9a-f]{64}")
def read_bound(path_text, expected, label):
    path = Path(path_text)
    if HEX64.fullmatch(expected) is None:
        raise SystemExit(f"{label} expected SHA256 is invalid")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise SystemExit(f"{label} SHA256 mismatch")
    return raw

config_raw = Path(config_path).read_bytes()
config_sha = hashlib.sha256(config_raw).hexdigest()
cfg = yaml.safe_load(config_raw)
data = cfg.get("data") or {}
model = cfg.get("model") or {}
train = cfg.get("train") or {}
required = {
    "run_class": "formal", "batch_size": 2, "grad_accum": 8,
    "lr": 5.0e-3, "lr_final": 3.0e-3, "warmup_steps": 200,
    "lr_horizon_steps": 500000, "grad_clip": 0.1, "max_steps": 500000,
    "amp": False, "seed": 0,
}
for key, expected in required.items():
    if train.get(key) != expected:
        raise SystemExit(f"formal config train.{key} is not frozen to {expected!r}")
if data.get("crop_length") != 256 or data.get("seed") != 0:
    raise SystemExit("formal config data crop/seed is not frozen")
if model.get("hidden") != 128 or model.get("cond_layers") != 6 or model.get("transport_layers") != 6:
    raise SystemExit("formal config architecture is not the frozen H128 6+6 candidate")
if train["batch_size"] * train["grad_accum"] * 8 != 128:
    raise SystemExit("formal config effective batch is not 128 on world size 8")
for key in ("root", "manifest", "domains_file", "full_training_contract"):
    value = data.get(key)
    if not isinstance(value, str) or not value.startswith("/"):
        raise SystemExit(f"formal config data.{key} must be an absolute rendered path")
if data.get("full_training_contract_sha256") != contract_sha:
    raise SystemExit("formal config does not bind the exact full training contract")
if Path(data["full_training_contract"]).resolve() != Path(contract_path).resolve():
    raise SystemExit("formal config full training contract path mismatch")

package_raw = read_bound(package_path, package_sha, "formal package")
package = json.loads(package_raw)
if package.get("formal_training_authorized") is not False:
    raise SystemExit("formal package must remain unauthorized")
if package.get("package_ready") is not True:
    raise SystemExit("formal package is not ready for a user decision")
candidate = package.get("formal_candidate")
if not isinstance(candidate, dict):
    raise SystemExit("formal package candidate is missing")
execution_plan = candidate.get("execution_plan")
if not isinstance(execution_plan, dict):
    raise SystemExit("formal package execution_plan is missing")
if candidate.get("config_sha256") != config_sha:
    raise SystemExit("formal candidate config SHA256 mismatch")
expected_execution_plan = {
    "reviewed_commit": expected_commit,
    "run_id": run_id,
    "obs_dst": obs_dst,
    "data_uuid": data_uuid,
    "world_size": 8,
    "repo_root": str(Path(repo_root).resolve()),
    "run_dir": expected_run_dir,
    "config_path": str(Path(config_path).resolve()),
    "config_sha256": config_sha,
    "contract_verification_path": str(Path(verification_path).resolve()),
    "contract_verification_sha256": verification_sha,
    "full_training_contract_path": str(Path(contract_path).resolve()),
    "full_training_contract_sha256": contract_sha,
    "supervisor_path": str(Path(supervisor_path).resolve()),
    "supervisor_sha256": hashlib.sha256(Path(supervisor_path).read_bytes()).hexdigest(),
    "archiver_path": str(Path(archiver_path).resolve()),
    "archiver_sha256": hashlib.sha256(Path(archiver_path).read_bytes()).hexdigest(),
    "validator_path": str(Path(validator_path).resolve()),
    "validator_sha256": hashlib.sha256(Path(validator_path).read_bytes()).hexdigest(),
    "empty_prefix_validator_path": str(Path(empty_prefix_validator_path).resolve()),
    "empty_prefix_validator_sha256": hashlib.sha256(
        Path(empty_prefix_validator_path).read_bytes()
    ).hexdigest(),
    "trainer_path": str(Path(trainer_path).resolve()),
    "trainer_sha256": hashlib.sha256(Path(trainer_path).read_bytes()).hexdigest(),
    "soft_stop_minutes": soft_stop_minutes,
    "hard_stop_minutes": hard_stop_minutes,
    "archive_kill_grace_seconds": archive_grace_seconds,
    "archive_poll_seconds": archive_poll_seconds,
}
for field, expected in expected_execution_plan.items():
    if execution_plan.get(field) != expected:
        raise SystemExit(
            f"formal package execution_plan.{field} does not bind runtime: "
            f"{execution_plan.get(field)!r} != {expected!r}"
        )

if execution_plan.get("repo_root") != str(Path(execution_plan["repo_root"]).resolve()):
    raise SystemExit("formal package repo_root is not canonical")
toolchain = execution_plan.get("toolchain")
if not isinstance(toolchain, dict):
    raise SystemExit("formal package toolchain is missing")
for name, configured_path in (
    ("python", python_path),
    ("torchrun", torchrun_path),
    ("obsutil", obsutil_path),
):
    entry = toolchain.get(name)
    if not isinstance(entry, dict):
        raise SystemExit(f"formal package toolchain.{name} is missing")
    realpath = str(Path(configured_path).resolve())
    if configured_path != realpath or entry.get("path") != realpath:
        raise SystemExit(f"formal package toolchain.{name} path is not canonical")
    if entry.get("sha256") != hashlib.sha256(Path(realpath).read_bytes()).hexdigest():
        raise SystemExit(f"formal package toolchain.{name} SHA256 mismatch")
    version_args = entry.get("version_args")
    if (
        not isinstance(version_args, list)
        or not version_args
        or any(not isinstance(item, str) for item in version_args)
    ):
        raise SystemExit(f"formal package toolchain.{name} version_args are invalid")
    completed = subprocess.run(
        [realpath, *version_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != entry.get("version"):
        raise SystemExit(f"formal package toolchain.{name} version mismatch")

checkpoint_plan = package.get("checkpoint_plan")
if not isinstance(checkpoint_plan, dict):
    raise SystemExit("formal package checkpoint_plan is missing")
expected_checkpoint_plan = {
    "ckpt_every": 1000,
    "trainer_keep_last_k": 501,
    "archiver_keep_local_verified": keep_local_verified,
    "immutable_numbered": True,
    "local_strict_validator": True,
    "forced_obs_readback": True,
    "remote_strict_validator": True,
    "verified_remote_required_for_retention": True,
    "latest_verified_required": True,
}
for field, expected in expected_checkpoint_plan.items():
    if checkpoint_plan.get(field) != expected:
        raise SystemExit(f"formal package checkpoint_plan.{field} is not exact")

stop_plan = package.get("stop_plan")
if not isinstance(stop_plan, dict):
    raise SystemExit("formal package stop_plan is missing")
expected_stop_plan = {
    "soft_stop_minutes": soft_stop_minutes,
    "hard_stop_minutes": hard_stop_minutes,
    "archive_kill_grace_seconds": archive_grace_seconds,
    "soft_stop_mechanism": "sealed_attempt_sentinel_at_optimizer_boundary",
    "soft_stop_precedes_hard_stop": True,
    "archive_failure_soft_stop": True,
}
for field, expected in expected_stop_plan.items():
    if stop_plan.get(field) != expected:
        raise SystemExit(f"formal package stop_plan.{field} is not exact")
if hard_stop_minutes - soft_stop_minutes < 30:
    raise SystemExit("formal package leaves less than 30 minutes for final archival")

recovery_plan = package.get("recovery_plan")
if not isinstance(recovery_plan, dict):
    raise SystemExit("formal package recovery_plan is missing")
expected_recovery_plan = {
    "separate_attempt_required": True,
    "resume_history_required": True,
    "strict_checkpoint_preflight": True,
    "latest_verified_only": True,
    "resume_semantics": "state_consistent_non_bitwise_crop_and_noise",
}
for field, expected in expected_recovery_plan.items():
    if recovery_plan.get(field) != expected:
        raise SystemExit(f"formal package recovery_plan.{field} is not exact")

raw_authorization = json.loads(read_bound(
    raw_authorization_path, raw_authorization_sha, "formal user authorization"
))
raw_authorization_keys = {
    "schema", "status", "formal_training_authorized", "authorization_id",
    "issued_at", "expires_at", "authorized_package_sha256",
    "authorized_package_id", "authorized_run_id", "authorized_reviewed_commit",
    "scope", "approval_record_sha256",
}
if set(raw_authorization) != raw_authorization_keys:
    raise SystemExit("formal user authorization schema has missing or extra fields")
if (
    raw_authorization.get("schema") != "deepjump.formal500k.user_authorization.v1"
    or raw_authorization.get("status") != "USER_AUTHORIZED_FORMAL_TRAINING"
):
    raise SystemExit("formal user authorization status mismatch")
if raw_authorization.get("formal_training_authorized") is not True:
    raise SystemExit("formal user authorization is not executable")
if raw_authorization.get("authorized_package_sha256") != package_sha:
    raise SystemExit("formal user authorization does not bind the exact package")
if raw_authorization.get("authorized_package_id") != package.get("package_id"):
    raise SystemExit("formal user authorization package_id mismatch")
if raw_authorization.get("authorized_run_id") != run_id:
    raise SystemExit("formal user authorization run_id mismatch")
if raw_authorization.get("authorized_reviewed_commit") != expected_commit:
    raise SystemExit("formal user authorization reviewed_commit mismatch")
if HEX64.fullmatch(raw_authorization.get("approval_record_sha256") or "") is None:
    raise SystemExit("formal user authorization approval record SHA256 is invalid")
try:
    issued_at = datetime.fromisoformat(raw_authorization["issued_at"].replace("Z", "+00:00"))
    expires_at = datetime.fromisoformat(raw_authorization["expires_at"].replace("Z", "+00:00"))
except (TypeError, ValueError) as exc:
    raise SystemExit("formal user authorization timestamps are invalid") from exc
now = datetime.now(timezone.utc)
if not issued_at <= now < expires_at:
    raise SystemExit("formal user authorization is not currently valid")
authorized_hours = (expires_at - issued_at).total_seconds() / 3600
if (
    authorized_hours <= 0
    or authorized_hours > float(package["estimate_budget"]["hard_cap_hours"])
):
    raise SystemExit("formal authorization lifetime exceeds the package GPU-hour cap")

authorization = json.loads(read_bound(
    authorization_path, authorization_sha, "formal authorization verification"
))
authorization_keys = {
    "schema", "status", "formal_training_authorized", "package_sha256",
    "authorization_sha256", "authorization_verifier_sha256", "package_id",
    "run_id", "reviewed_commit", "config_sha256", "target_total_steps",
    "execution_plan", "checkpoint_plan", "stop_plan", "recovery_plan", "scope",
}
if set(authorization) != authorization_keys:
    raise SystemExit("formal authorization verification schema has missing or extra fields")
if (
    authorization.get("schema")
    != "deepjump.formal500k.authorization_verification.v1"
    or authorization.get("status") != "PASS_FORMAL_TRAINING_AUTHORIZATION"
):
    raise SystemExit("formal authorization verification did not pass")
if authorization.get("formal_training_authorized") is not True:
    raise SystemExit("formal execution lacks explicit user authorization")
if authorization.get("package_sha256") != package_sha:
    raise SystemExit("formal authorization does not bind the exact package")
if authorization.get("config_sha256") != config_sha:
    raise SystemExit("formal authorization does not bind the exact config")
if authorization.get("target_total_steps") != 500000:
    raise SystemExit("formal authorization does not bind the 500k endpoint")
if authorization.get("authorization_sha256") != raw_authorization_sha:
    raise SystemExit("formal authorization verification does not bind the user authorization")
authorization_verifier_path = (
    Path(supervisor_path).resolve().parents[2]
    / "scripts/verify_formal500k_authorization.py"
)
if authorization.get("authorization_verifier_sha256") != hashlib.sha256(
    authorization_verifier_path.read_bytes()
).hexdigest():
    raise SystemExit("formal authorization verifier SHA256 mismatch")
verification_bindings = {
    "package_id": package.get("package_id"),
    "run_id": run_id,
    "reviewed_commit": expected_commit,
    "execution_plan": execution_plan,
    "checkpoint_plan": checkpoint_plan,
    "stop_plan": stop_plan,
    "recovery_plan": recovery_plan,
    "scope": raw_authorization["scope"],
}
for field, expected in verification_bindings.items():
    if authorization.get(field) != expected:
        raise SystemExit(f"formal authorization verification {field} mismatch")

verification = json.loads(read_bound(
    verification_path, verification_sha, "contract verification"
))
if verification.get("status") != "PASS_FULL_TRAINING_DATA_CONTRACT":
    raise SystemExit("full training data contract verification did not pass")
read_bound(contract_path, contract_sha, "full training contract")
print(json.dumps({
    "config_sha256": config_sha,
    "data_root": data["root"],
    "out_dir": train["out_dir"],
    "authorization_expires_epoch": int(expires_at.timestamp()),
}, separators=(",", ":")))
PY
) || fail "formal package/config/authorization preflight failed"
echo ">> exact formal package, authorization, and execution-plan preflight PASS"
command -v timeout >/dev/null || fail "GNU timeout is required"
command -v systemd-run >/dev/null || fail "systemd-run is required"
command -v setsid >/dev/null || fail "setsid is required for archiver process ownership"

data_root=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["data_root"])' "$preflight")
configured_out=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["out_dir"])' "$preflight")
CONFIG_SHA256=$("$PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["config_sha256"])' "$preflight")
AUTHORIZATION_EXPIRES_EPOCH=$("$PYTHON" -c \
  'import json,sys; print(json.loads(sys.argv[1])["authorization_expires_epoch"])' \
  "$preflight")
if [[ "$configured_out" = /* ]]; then
  RUN_DIR=$configured_out
else
  RUN_DIR="$REPO_ROOT/$configured_out"
fi
RUN_DIR=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RUN_DIR")
[ "$RUN_DIR" = "$EXPECTED_RUN_DIR" ] ||
  fail "formal config out_dir differs from package execution_plan.run_dir"
RUN_PARENT=$(dirname "$RUN_DIR")
[ -d "$RUN_PARENT" ] || fail "formal run parent does not exist: $RUN_PARENT"
command -v flock >/dev/null || fail "flock is required for run-level single ownership"
RUN_LOCK="$RUN_PARENT/.formal500k-${RUN_ID}.lock"
exec 9>"$RUN_LOCK"
flock -n 9 || fail "another supervisor already owns this formal RUN_ID"

mount_target=$(findmnt -n -o TARGET --target "$data_root")
mount_uuid=$(findmnt -n -o UUID --target "$data_root")
mount_options=$(findmnt -n -o OPTIONS --target "$data_root")
[ "$mount_uuid" = "$EXPECTED_DATA_UUID" ] || fail "data UUID mismatch"
case ",$mount_options," in
  *,ro,*) ;;
  *) fail "data mount is not read-only: $mount_target" ;;
esac
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')" = 8 ] ||
  fail "exactly eight GPUs are required"
[ "$(hostname)" = "$EXPECTED_HOSTNAME" ] || fail "exact GPU hostname mismatch"
product_uuid=$(tr '[:upper:]' '[:lower:]' </sys/class/dmi/id/product_uuid)
product_serial=$(tr '[:upper:]' '[:lower:]' </sys/class/dmi/id/product_serial)
[ "$product_uuid" = "$EXPECTED_PRODUCT_UUID" ] ||
  fail "exact GPU product UUID mismatch"
[ "$product_serial" = "$EXPECTED_PRODUCT_SERIAL" ] ||
  fail "exact GPU product serial mismatch"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fxc "$EXPECTED_GPU_MODEL")" = 8 ] ||
  fail "exact GPU model inventory mismatch"

emergency_poweroff() {
  local reason=$* backup_unit backup_active
  echo "!! $reason" >&2
  backup_unit="deepjump-formal500k-emergency-stop-${RUN_ID,,}-$$"
  backup_active=no
  if systemd-run --unit "$backup_unit" --on-active=2m \
       --timer-property=AccuracySec=1s \
       /usr/bin/systemctl poweroff >/dev/null 2>&1 &&
     systemctl is-active "${backup_unit}.timer" >/dev/null 2>&1; then
    case "$(systemctl show "${backup_unit}.service" -p ExecStart --value)" in
      *"/usr/bin/systemctl poweroff"*) backup_active=yes ;;
    esac
  fi
  if ! /usr/bin/systemctl poweroff >/dev/null 2>&1; then
    echo "!! immediate poweroff request failed; emergency_timer_active=$backup_active" >&2
  fi
  exit 1
}

record_recovery_exhaustion() {
  "$PYTHON" - "$RUN_DIR/RECOVERY_ATTEMPTS_EXHAUSTED.json" "$RUN_ID" \
    "$FORMAL_PACKAGE_SHA256" "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" \
    "$recovery_attempt_count" "$MAX_RECOVERY_ATTEMPTS" \
    "$AUTHORIZATION_EXPIRES_EPOCH" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path_value, run_id, package_sha, verification_sha, observed_count,
    maximum_count, authorization_expires_epoch,
) = sys.argv[1:]
path = Path(path_value)
expected = {
    "schema": "deepjump.formal500k_recovery_exhaustion.v1",
    "status": "RECOVERY_ATTEMPTS_EXHAUSTED_REQUEST_POWER_OFF",
    "run_id": run_id,
    "formal_package_sha256": package_sha,
    "authorization_verification_sha256": verification_sha,
    "observed_recovery_attempts": int(observed_count),
    "maximum_recovery_attempts": int(maximum_count),
    "authorization_expires_epoch": int(authorization_expires_epoch),
}
if path.exists():
    current = json.loads(path.read_text())
    for field in (
        "schema", "status", "run_id", "formal_package_sha256",
        "authorization_verification_sha256", "maximum_recovery_attempts",
        "authorization_expires_epoch",
    ):
        if current.get(field) != expected[field]:
            raise SystemExit(f"existing recovery exhaustion evidence {field} mismatch")
else:
    expected["recorded_at"] = datetime.now(timezone.utc).isoformat()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
PY
}

# A package-bound authorization-expiry timer is established before any attempt
# claim. It survives cold recovery starts and bounds cost even when an attempt
# cannot be claimed or its narrower hard stop cannot be installed.
authorization_stop_unit="deepjump-formal500k-authorization-expiry-${RUN_ID,,}-${FORMAL_PACKAGE_SHA256:0:12}"
authorization_stop_target_epoch=$((AUTHORIZATION_EXPIRES_EPOCH - 2))
authorization_stop_calendar=$(date -u -d "@$authorization_stop_target_epoch" \
  '+%Y-%m-%d %H:%M:%S UTC') ||
  emergency_poweroff "cannot render authorization-expiry hard-stop deadline"
if ! systemctl is-active "${authorization_stop_unit}.timer" >/dev/null 2>&1; then
  systemd-run --unit "$authorization_stop_unit" \
    --on-calendar="$authorization_stop_calendar" \
    --timer-property=AccuracySec=1s \
    /usr/bin/systemctl poweroff >/dev/null ||
    emergency_poweroff "cannot install authorization-expiry hard stop"
fi
systemctl is-active "${authorization_stop_unit}.timer" >/dev/null ||
  emergency_poweroff "authorization-expiry hard-stop timer is not active"
authorization_stop_accuracy=$(systemctl show "${authorization_stop_unit}.timer" \
  -p AccuracyUSec --value) ||
  emergency_poweroff "cannot inspect authorization-expiry hard-stop accuracy"
[ "$authorization_stop_accuracy" = "1s" ] ||
  emergency_poweroff "authorization-expiry hard-stop accuracy is not one second"
authorization_stop_exec=$(systemctl show "${authorization_stop_unit}.service" \
  -p ExecStart --value) ||
  emergency_poweroff "cannot inspect authorization-expiry hard-stop service"
case "$authorization_stop_exec" in
  *"/usr/bin/systemctl poweroff"*) ;;
  *) emergency_poweroff "authorization-expiry hard-stop ExecStart is not exact poweroff" ;;
esac
authorization_stop_deadline=$(systemctl show "${authorization_stop_unit}.timer" \
  -p NextElapseUSecRealtime --value) ||
  emergency_poweroff "cannot inspect authorization-expiry hard-stop deadline"
[ -n "$authorization_stop_deadline" ] ||
  emergency_poweroff "authorization-expiry hard-stop timer has no deadline"
authorization_stop_deadline_epoch=$(date -d "$authorization_stop_deadline" +%s) ||
  emergency_poweroff "cannot parse authorization-expiry hard-stop deadline"
[ "$authorization_stop_deadline_epoch" -le "$authorization_stop_target_epoch" ] ||
  emergency_poweroff "authorization-expiry hard stop exceeds authorization"

ATTEMPT_DIR=
if [ "$MODE" = fresh ]; then
  [ ! -e "$RUN_DIR" ] || fail "fresh run directory already exists: $RUN_DIR"
  mkdir "$RUN_DIR" || fail "atomic fresh-run claim failed"
  ATTEMPT_DIR="$RUN_DIR/.attempts/fresh-$RUN_ID"
  mkdir -p "$RUN_DIR/.attempts"
  mkdir "$ATTEMPT_DIR" || fail "atomic fresh attempt claim failed"
else
  [ -d "$RUN_DIR" ] || fail "recovery requires the existing run directory"
  RECOVERY_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mkdir -p "$RUN_DIR/.attempts"
  recovery_attempt_count=$(find "$RUN_DIR/.attempts" -mindepth 1 -maxdepth 1 \
    -type d -name 'recovery-*' -print | wc -l | tr -d ' ')
  if [ "$recovery_attempt_count" -ge "$MAX_RECOVERY_ATTEMPTS" ]; then
    record_recovery_exhaustion ||
      emergency_poweroff "cannot publish recovery exhaustion evidence"
    emergency_poweroff "package-bound maximum recovery attempts exhausted"
  fi
  ATTEMPT_DIR="$RUN_DIR/.attempts/recovery-$RECOVERY_ATTEMPT_ID"
  mkdir "$ATTEMPT_DIR" || fail "recovery attempt already exists"
fi

CLAIM="$ATTEMPT_DIR/claim.json"
"$PYTHON" - "$CLAIM" "$MODE" "$RUN_ID" "$EXPECTED_COMMIT" \
  "$FORMAL_PACKAGE_SHA256" "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" \
  "$RUN_LOCK" "$$" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
(
    path, mode, run_id, commit, package_sha, authorization_verification_sha,
    run_lock_path, supervisor_pid,
) = sys.argv[1:]
payload = {
    "schema": "deepjump.formal500k_attempt_claim.v1",
    "mode": mode,
    "run_id": run_id,
    "commit": commit,
    "formal_package_sha256": package_sha,
    "authorization_verification_sha256": authorization_verification_sha,
    "execution_authorization_verified": True,
    "run_lock_path": run_lock_path,
    "run_lock_held": True,
    "supervisor_pid": int(supervisor_pid),
    "claimed_at": datetime.now(timezone.utc).isoformat(),
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

# Install the absolute authorization-bound hard stop before OBS, recovery, or
# trainer work so any fail-closed preflight still has a powered-on cost bound.
attempt_start_epoch=$(date +%s)
package_hard_deadline_epoch=$((attempt_start_epoch + HARD_STOP_MINUTES * 60))
hard_stop_deadline_epoch=$package_hard_deadline_epoch
latest_authorized_hard_stop_epoch=$((AUTHORIZATION_EXPIRES_EPOCH - 2))
[ "$latest_authorized_hard_stop_epoch" -lt "$hard_stop_deadline_epoch" ] &&
  hard_stop_deadline_epoch=$latest_authorized_hard_stop_epoch
soft_stop_epoch=$((attempt_start_epoch + SOFT_STOP_MINUTES * 60))
latest_safe_soft_stop_epoch=$((hard_stop_deadline_epoch - 1800))
[ "$latest_safe_soft_stop_epoch" -lt "$soft_stop_epoch" ] &&
  soft_stop_epoch=$latest_safe_soft_stop_epoch
[ "$soft_stop_epoch" -gt "$attempt_start_epoch" ] ||
  fail "authorization leaves no positive training window"
[ "$hard_stop_deadline_epoch" -le "$AUTHORIZATION_EXPIRES_EPOCH" ] ||
  fail "hard-stop deadline exceeds authorization expiry"
[ $((hard_stop_deadline_epoch - soft_stop_epoch)) -ge 1800 ] ||
  fail "absolute soft stop leaves less than thirty minutes before hard stop"
remaining_authorized_minutes=$(((AUTHORIZATION_EXPIRES_EPOCH - attempt_start_epoch) / 60))
[ "$remaining_authorized_minutes" -gt 30 ] ||
  fail "authorization has less than 30 minutes remaining"
attempt_slug=$(basename "$ATTEMPT_DIR" | tr -cd 'a-zA-Z0-9_.-' | tr 'A-Z' 'a-z')
hard_stop_unit="deepjump-formal500k-hard-stop-${RUN_ID,,}-${attempt_slug}"
hard_stop_calendar=$(date -u -d "@$hard_stop_deadline_epoch" '+%Y-%m-%d %H:%M:%S UTC')
systemd-run --unit "$hard_stop_unit" --on-calendar="$hard_stop_calendar" \
  --timer-property=AccuracySec=1s \
  /usr/bin/systemctl poweroff >/dev/null ||
  emergency_poweroff "cannot install attempt hard stop"
systemctl is-active "${hard_stop_unit}.timer" >/dev/null ||
  emergency_poweroff "hard-stop timer did not become active"
hard_stop_accuracy=$(systemctl show "${hard_stop_unit}.timer" \
  -p AccuracyUSec --value) ||
  emergency_poweroff "cannot inspect hard-stop timer accuracy"
[ "$hard_stop_accuracy" = "1s" ] ||
  emergency_poweroff "hard-stop timer accuracy is not one second"
hard_stop_exec=$(systemctl show "${hard_stop_unit}.service" -p ExecStart --value)
case "$hard_stop_exec" in
  *"/usr/bin/systemctl poweroff"*) ;;
  *) emergency_poweroff "hard-stop service ExecStart is not exact poweroff" ;;
esac
hard_stop_deadline=$(systemctl show "${hard_stop_unit}.timer" \
  -p NextElapseUSecRealtime --value)
[ -n "$hard_stop_deadline" ] ||
  emergency_poweroff "hard-stop timer has no realtime deadline"
scheduled_hard_stop_deadline_epoch=$(date -d "$hard_stop_deadline" +%s) ||
  emergency_poweroff "cannot parse hard-stop timer deadline"
deadline_error=$((scheduled_hard_stop_deadline_epoch - hard_stop_deadline_epoch))
[ "$deadline_error" -lt 0 ] && deadline_error=$((-deadline_error))
[ "$deadline_error" -le 2 ] ||
  emergency_poweroff "hard-stop timer deadline differs from the absolute deadline"
[ "$scheduled_hard_stop_deadline_epoch" -le "$AUTHORIZATION_EXPIRES_EPOCH" ] ||
  emergency_poweroff "scheduled hard-stop exceeds authorization expiry"

# Execute source code from an immutable per-attempt archive of the reviewed
# commit, and copy non-repository inputs with stable descriptor reads.
SOURCE_SNAPSHOT="$ATTEMPT_DIR/source_snapshot"
mkdir "$SOURCE_SNAPSHOT"
git -C "$REPO_ROOT" archive "$EXPECTED_COMMIT" | tar -x -C "$SOURCE_SNAPSHOT"
"$PYTHON" - "$SOURCE_SNAPSHOT" <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
    mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    if path.is_symlink():
        raise SystemExit(f"reviewed source snapshot contains a symlink: {path}")
    path.chmod(mode & ~0o222)
root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
PY
snapshot_repo_member() {
  "$PYTHON" - "$REPO_ROOT" "$SOURCE_SNAPSHOT" "$1" "$2" <<'PY'
import hashlib
import sys
from pathlib import Path
repo, snapshot, source = map(Path, sys.argv[1:4])
expected = sys.argv[4]
repo = repo.resolve()
source = source.resolve()
try:
    relative = source.relative_to(repo)
except ValueError as exc:
    raise SystemExit(f"package-bound source is outside reviewed repo: {source}") from exc
candidate = (snapshot / relative).resolve()
if not candidate.is_relative_to(snapshot.resolve()) or not candidate.is_file():
    raise SystemExit(f"reviewed source snapshot member is missing: {relative}")
if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
    raise SystemExit(f"reviewed source snapshot SHA256 mismatch: {relative}")
print(candidate)
PY
}
snapshot_input() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:3])
expected = sys.argv[3]
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source, flags)
with os.fdopen(descriptor, "rb") as handle:
    before = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"bound input is not regular: {source}")
    raw = handle.read()
    after = os.fstat(handle.fileno())
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
)
if identity(before) != identity(after) or len(raw) != before.st_size:
    raise SystemExit(f"bound input changed while snapshotting: {source}")
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit(f"bound input SHA256 mismatch: {source}")
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(destination, 0o400)
directory = os.open(destination.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
print(destination.resolve())
PY
}
ARCHIVER=$(snapshot_repo_member "$ARCHIVER" "$ARCHIVER_SHA256")
VALIDATOR=$(snapshot_repo_member "$VALIDATOR" "$VALIDATOR_SHA256")
EMPTY_PREFIX_VALIDATOR=$(snapshot_repo_member \
  "$EMPTY_PREFIX_VALIDATOR" "$EMPTY_PREFIX_VALIDATOR_SHA256")
TRAINER=$(snapshot_repo_member "$TRAINER" "$TRAINER_SHA256")
FORMAL_CONFIG=$(snapshot_input "$FORMAL_CONFIG" \
  "$ATTEMPT_DIR/formal_config.yaml" "$CONFIG_SHA256")
CONTRACT_VERIFICATION=$(snapshot_input "$CONTRACT_VERIFICATION" \
  "$ATTEMPT_DIR/contract_verification.json" "$CONTRACT_VERIFICATION_SHA256")
FULL_TRAINING_CONTRACT_SNAPSHOT=$(snapshot_input "$FULL_TRAINING_CONTRACT" \
  "$ATTEMPT_DIR/full_training_contract.json" "$FULL_TRAINING_CONTRACT_SHA256")
export PYTHONPATH="$SOURCE_SNAPSHOT/src"
verify_runtime_file() {
  local actual
  actual=$(sha256sum "$1" | awk '{print $1}')
  [ "$actual" = "$2" ] || fail "runtime tool SHA256 drift: $1"
}
verify_runtime_file "$PYTHON" "$PYTHON_SHA256"
verify_runtime_file "$TORCHRUN" "$TORCHRUN_SHA256"
verify_runtime_file "$OBSUTIL" "$OBSUTIL_SHA256"
verify_runtime_file "$ARCHIVER" "$ARCHIVER_SHA256"
verify_runtime_file "$VALIDATOR" "$VALIDATOR_SHA256"
verify_runtime_file "$EMPTY_PREFIX_VALIDATOR" "$EMPTY_PREFIX_VALIDATOR_SHA256"
verify_runtime_file "$TRAINER" "$TRAINER_SHA256"
verify_runtime_file "$FORMAL_CONFIG" "$CONFIG_SHA256"
verify_runtime_file "$CONTRACT_VERIFICATION" "$CONTRACT_VERIFICATION_SHA256"
verify_runtime_file "$FULL_TRAINING_CONTRACT" "$FULL_TRAINING_CONTRACT_SHA256"
verify_runtime_file "$FULL_TRAINING_CONTRACT_SNAPSHOT" \
  "$FULL_TRAINING_CONTRACT_SHA256"

OWNERSHIP="$ATTEMPT_DIR/OWNERSHIP.expected.json"
"$PYTHON" - "$OWNERSHIP" "$RUN_ID" "$OBS_DST" "$FORMAL_PACKAGE_SHA256" \
  "$FORMAL_AUTHORIZATION_SHA256" \
  "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" "$EXPECTED_PRODUCT_UUID" <<'PY'
import json
import os
import sys
path, run_id, obs_dst, package_sha, authorization_sha, verification_sha, instance_id = (
    sys.argv[1:]
)
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump({
        "schema": "deepjump.formal500k_obs_ownership.v1",
        "status": "CLAIMED_EXACT_HOST_AND_RUN",
        "run_id": run_id,
        "obs_dst": obs_dst,
        "instance_id": instance_id,
        "formal_package_sha256": package_sha,
        "formal_authorization_sha256": authorization_sha,
        "authorization_verification_sha256": verification_sha,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

if [ "$MODE" = fresh ]; then
  OBS_EMPTY_REPORT="$ATTEMPT_DIR/obs_empty_preflight.txt"
  timeout 5m "$OBSUTIL" ls "$OBS_DST/" -limit=0 -bf=raw >"$OBS_EMPTY_REPORT"
  "$PYTHON" "$EMPTY_PREFIX_VALIDATOR" "$OBS_EMPTY_REPORT"
  timeout 5m "$OBSUTIL" cp "$OWNERSHIP" "$OBS_DST/OWNERSHIP.json" -f >/dev/null
  timeout 5m "$OBSUTIL" cp "$OBS_DST/OWNERSHIP.json" \
    "$ATTEMPT_DIR/OWNERSHIP.readback.json" -f >/dev/null
  cmp -s "$OWNERSHIP" "$ATTEMPT_DIR/OWNERSHIP.readback.json" ||
    fail "OBS ownership marker forced readback mismatch"
  timeout 5m "$OBSUTIL" ls "$OBS_DST/" -limit=0 -bf=raw \
    >"$ATTEMPT_DIR/obs_ownership_inventory.txt"
  "$PYTHON" - "$EMPTY_PREFIX_VALIDATOR" \
    "$ATTEMPT_DIR/obs_ownership_inventory.txt" "$OWNERSHIP" <<'PY'
import importlib.util
import sys
from pathlib import Path
module_spec = importlib.util.spec_from_file_location("obs_prefix", sys.argv[1])
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
count, size = module.prefix_file_inventory(Path(sys.argv[2]).read_text())
expected_size = Path(sys.argv[3]).stat().st_size
if (count, size) != (1, expected_size):
    raise SystemExit(
        f"OBS ownership inventory mismatch: {(count, size)} != {(1, expected_size)}"
    )
PY
else
  timeout 5m "$OBSUTIL" cp "$OBS_DST/OWNERSHIP.json" \
    "$ATTEMPT_DIR/OWNERSHIP.readback.json" -f >/dev/null
  cmp -s "$OWNERSHIP" "$ATTEMPT_DIR/OWNERSHIP.readback.json" ||
    fail "recovery OBS ownership does not bind exact host/run/package"
fi

trainer=(
  "$TORCHRUN" --standalone --nproc_per_node=8
  "$TRAINER"
  --config "$FORMAL_CONFIG"
  --full-training-contract "$FULL_TRAINING_CONTRACT"
  --expected-full-training-contract-sha256 "$FULL_TRAINING_CONTRACT_SHA256"
  --graceful-stop-file "$ATTEMPT_DIR/graceful_stop.requested"
)

if [ "$MODE" = recovery ]; then
  RECOVERY_INPUT="$ATTEMPT_DIR/recovery_input"
  mkdir "$RECOVERY_INPUT"
  # LATEST_VERIFIED is itself forced-read-back, monotonic, package-bound state.
  # It is the only selector so arbitrary optimizer-boundary checkpoints remain
  # recoverable. The immutable step marker must independently match it byte for
  # byte before any payload is accepted.
  RESUME_MARKER="$RECOVERY_INPUT/LATEST_VERIFIED.json"
  timeout 5m "$OBSUTIL" cp "$OBS_DST/LATEST_VERIFIED.json" \
    "$RESUME_MARKER" -f >/dev/null
  resume_step=$("$PYTHON" - "$RESUME_MARKER" "$OBS_DST" \
    "$CONFIG_SHA256" "$CONTRACT_VERIFICATION_SHA256" <<'PY'
import json
import re
import sys
from pathlib import Path
marker = json.loads(Path(sys.argv[1]).read_text())
obs_dst, config_sha, verification_sha = sys.argv[2:]
step = marker.get("step")
if not isinstance(step, int) or isinstance(step, bool) or not 0 < step <= 500000:
    raise SystemExit("LATEST_VERIFIED step is invalid")
expected = {
    "status": "PASS_STRICT_CHECKPOINT_OBS_READBACK",
    "checkpoint_object": f"{obs_dst}/checkpoints/ckpt_{step}.pt",
    "history_object": f"{obs_dst}/history/history_{step}.json",
    "local_validator_report_object": f"{obs_dst}/validation/local_{step}.json",
    "remote_validator_report_object": f"{obs_dst}/validation/remote_readback_{step}.json",
    "marker_object": f"{obs_dst}/verified/ckpt_{step}.pt.readback.json",
    "latest_verified_object": f"{obs_dst}/LATEST_VERIFIED.json",
    "config_sha256": config_sha,
    "contract_verification_sha256": verification_sha,
    "resume_semantics": "state_consistent_non_bitwise_crop_and_noise",
    "formal_training_authorized": False,
}
for field, value in expected.items():
    if marker.get(field) != value:
        raise SystemExit(f"LATEST_VERIFIED {field} mismatch")
for field in (
    "checkpoint_sha256", "history_sha256", "local_validator_report_sha256",
    "remote_validator_report_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", marker.get(field) or "") is None:
        raise SystemExit(f"LATEST_VERIFIED {field} is invalid")
print(step)
PY
  ) || fail "LATEST_VERIFIED is not an exact package-bound recovery selector"
  IMMUTABLE_RESUME_MARKER="$RECOVERY_INPUT/ckpt_${resume_step}.pt.readback.json"
  timeout 5m "$OBSUTIL" cp \
    "$OBS_DST/verified/ckpt_${resume_step}.pt.readback.json" \
    "$IMMUTABLE_RESUME_MARKER" -f >/dev/null
  cmp -s "$RESUME_MARKER" "$IMMUTABLE_RESUME_MARKER" ||
    fail "LATEST_VERIFIED does not exactly match its immutable step marker"

  recovery_spec=$("$PYTHON" - "$RESUME_MARKER" "$resume_step" "$OBS_DST" \
    "$CONFIG_SHA256" "$CONTRACT_VERIFICATION_SHA256" <<'PY'
import json
import sys
marker = json.load(open(sys.argv[1]))
step = int(sys.argv[2])
obs_dst, config_sha, verification_sha = sys.argv[3:]
expected = {
    "status": "PASS_STRICT_CHECKPOINT_OBS_READBACK",
    "step": step,
    "checkpoint_object": f"{obs_dst}/checkpoints/ckpt_{step}.pt",
    "history_object": f"{obs_dst}/history/history_{step}.json",
    "local_validator_report_object": f"{obs_dst}/validation/local_{step}.json",
    "remote_validator_report_object": f"{obs_dst}/validation/remote_readback_{step}.json",
    "marker_object": f"{obs_dst}/verified/ckpt_{step}.pt.readback.json",
    "config_sha256": config_sha,
    "contract_verification_sha256": verification_sha,
    "resume_semantics": "state_consistent_non_bitwise_crop_and_noise",
    "formal_training_authorized": False,
}
for field, value in expected.items():
    if marker.get(field) != value:
        raise SystemExit(f"remote recovery marker {field} mismatch")
for field in (
    "checkpoint_sha256", "history_sha256", "local_validator_report_sha256",
    "remote_validator_report_sha256",
):
    value = marker.get(field)
    if not isinstance(value, str) or len(value) != 64:
        raise SystemExit(f"remote recovery marker {field} is invalid")
print(json.dumps(marker, separators=(",", ":")))
PY
  ) || fail "highest remote recovery marker is invalid"
  RESUME_CHECKPOINT="$RECOVERY_INPUT/ckpt_${resume_step}.pt"
  RESUME_HISTORY="$RECOVERY_INPUT/history_${resume_step}.json"
  LOCAL_REPORT="$RECOVERY_INPUT/local_${resume_step}.json"
  REMOTE_REPORT="$RECOVERY_INPUT/remote_readback_${resume_step}.json"
  "$PYTHON" - "$recovery_spec" "$OBSUTIL" \
    "$RESUME_CHECKPOINT" "$RESUME_HISTORY" "$LOCAL_REPORT" "$REMOTE_REPORT" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path
marker = json.loads(sys.argv[1])
obsutil = sys.argv[2]
bindings = (
    ("checkpoint", marker["checkpoint_object"], marker["checkpoint_sha256"], sys.argv[3]),
    ("history", marker["history_object"], marker["history_sha256"], sys.argv[4]),
    ("local report", marker["local_validator_report_object"],
     marker["local_validator_report_sha256"], sys.argv[5]),
    ("remote report", marker["remote_validator_report_object"],
     marker["remote_validator_report_sha256"], sys.argv[6]),
)
for label, source, expected_sha, destination in bindings:
    subprocess.run([obsutil, "cp", source, destination, "-f"], check=True)
    actual = hashlib.sha256(Path(destination).read_bytes()).hexdigest()
    if actual != expected_sha:
        raise SystemExit(f"forced {label} readback SHA mismatch")
for report_path in (sys.argv[5], sys.argv[6]):
    report = json.loads(Path(report_path).read_text())
    if report.get("status") != "PASS":
        raise SystemExit("archived strict validator report is not PASS")
    if report.get("checkpoint_sha256") != marker["checkpoint_sha256"]:
        raise SystemExit("archived strict validator report checkpoint SHA mismatch")
PY
  resume_sha=$("$PYTHON" -c \
    'import json,sys; print(json.loads(sys.argv[1])["checkpoint_sha256"])' \
    "$recovery_spec")
  "$PYTHON" "$VALIDATOR" \
    --checkpoint "$RESUME_CHECKPOINT" \
    --history "$RESUME_HISTORY" \
    --expected-step "$resume_step" \
    --expected-world-size 8 \
    --history-mode through \
    --expected-delta 1 \
    --require-full-tensor \
    --expected-lr-horizon-steps 500000 \
    --expected-config "$FORMAL_CONFIG" \
    --expected-contract-verification "$CONTRACT_VERIFICATION" \
    --expected-contract-verification-sha256 "$CONTRACT_VERIFICATION_SHA256" \
    --expected-checkpoint-sha256 "$resume_sha" \
    --output "$ATTEMPT_DIR/resume_preflight.json" >/dev/null

  # Preserve any locally published but unselected checkpoint beyond the exact
  # remote proof outside the active namespace. This prevents immutable
  # checkpoint publication from colliding when recovery reaches that step.
  "$PYTHON" - "$RUN_DIR" "$ATTEMPT_DIR/stale_local_unverified" \
    "$resume_step" <<'PY'
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
quarantine = Path(sys.argv[2])
resume_step = int(sys.argv[3])
stale = []
for path in run_dir.glob("ckpt_*.pt"):
    match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
    if match and int(match.group(1)) > resume_step:
        stale.append((int(match.group(1)), path))
if stale:
    quarantine.mkdir()
    parent_descriptor = os.open(quarantine.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
records = []
for step, source in sorted(stale):
    destination = quarantine / source.name
    os.rename(source, destination)
    for directory_path in (run_dir, quarantine):
        directory_descriptor = os.open(directory_path, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        raw = handle.read()
    records.append({
        "step": step,
        "source": str(source),
        "preserved_at": str(destination),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    })
if records:
    manifest = quarantine / "MANIFEST.json"
    temporary = manifest.with_name(manifest.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump({
            "schema": "deepjump.formal500k_stale_local_quarantine.v1",
            "status": "PRESERVED_NOT_SELECTED_FOR_RECOVERY",
            "selected_resume_step": resume_step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoints": records,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest)
    for directory_path in (quarantine, run_dir):
        directory = os.open(directory_path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
PY
  trainer+=(--resume "$RESUME_CHECKPOINT" --resume-history "$RESUME_HISTORY")
fi

ARCHIVE_FAILURE_FILE="$ATTEMPT_DIR/archive_failure.json"
ARCHIVER_STOP_SENTINEL="$ATTEMPT_DIR/archiver_stopping"
ARCHIVER_READY_FILE="$ATTEMPT_DIR/archiver.ready"
archiver_pid=
trainer_pid=
stop_archiver_group() {
  local owned_pid
  [ -n "$archiver_pid" ] || return 0
  owned_pid=$archiver_pid
  kill -TERM -- "-$owned_pid" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-$owned_pid" 2>/dev/null || true
  wait "$owned_pid" 2>/dev/null || true
  archiver_pid=
}
cleanup_children() {
  local rc=$?
  trap - EXIT
  if [ -n "$trainer_pid" ] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM -- "-$trainer_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$trainer_pid" 2>/dev/null || true
  fi
  stop_archiver_group
  exit "$rc"
}
trap cleanup_children EXIT

write_status() {
  "$PYTHON" - "$ATTEMPT_DIR/status.json" "$1" "$2" \
    "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump({
        "status": sys.argv[2],
        "exit_code": int(sys.argv[3]),
        "authorization_verification_sha256": sys.argv[4],
        "execution_authorization_verified": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
directory = os.open(path.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

record_archiver_failure() {
  local rc=$1
  [ -e "$ARCHIVE_FAILURE_FILE" ] && return 0
  "$PYTHON" - "$ARCHIVE_FAILURE_FILE" "$rc" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1:]
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump({
        "status": "ARCHIVER_FAILURE_REQUEST_SOFT_STOP",
        "exit_code": int(code),
        "formal_training_authorized": False,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
directory = os.open(os.path.dirname(path), os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

RUN_DIR="$RUN_DIR" OBS_DST="$OBS_DST" FORMAL_CONFIG="$FORMAL_CONFIG" \
CONTRACT_VERIFICATION="$CONTRACT_VERIFICATION" \
CONTRACT_VERIFICATION_SHA256="$CONTRACT_VERIFICATION_SHA256" \
VALIDATOR="$VALIDATOR" OBSUTIL="$OBSUTIL" PYTHON="$PYTHON" \
OBSUTIL_SHA256="$OBSUTIL_SHA256" PYTHON_SHA256="$PYTHON_SHA256" \
KEEP_LOCAL_VERIFIED="$KEEP_LOCAL_VERIFIED" POLL_SECONDS="$ARCHIVE_POLL_SECONDS" \
ARCHIVER_READY_FILE="$ARCHIVER_READY_FILE" \
ARCHIVER_RUN_ID="$RUN_ID" ARCHIVER_ATTEMPT_DIR="$ATTEMPT_DIR" \
ARCHIVE_ONCE=0 setsid "$ARCHIVER" >"$ATTEMPT_DIR/archiver.log" 2>&1 &
archiver_pid=$!
archiver_failed=0
archiver_ready_deadline=$(( $(date +%s) + 30 ))
while [ ! -e "$ARCHIVER_READY_FILE" ]; do
  if ! kill -0 "$archiver_pid" 2>/dev/null; then
    set +e
    wait "$archiver_pid"
    archiver_rc=$?
    set -e
    record_archiver_failure "$archiver_rc"
    stop_archiver_group
    fail "checkpoint archiver exited before readiness"
  fi
  [ "$(date +%s)" -lt "$archiver_ready_deadline" ] ||
    fail "checkpoint archiver readiness timed out"
  sleep 1
done
"$PYTHON" - "$ARCHIVER_READY_FILE" "$archiver_pid" "$RUN_ID" \
  "$ATTEMPT_DIR" <<'PY'
import json
import sys
from pathlib import Path

path, expected_pid, expected_run_id, expected_attempt_dir = sys.argv[1:]
payload = json.loads(Path(path).read_text())
expected = {
    "schema": "deepjump.formal500k_archiver_ready.v1",
    "status": "ARCHIVER_READY_AFTER_INITIAL_ROUND",
    "pid": int(expected_pid),
    "run_id": expected_run_id,
    "attempt_dir": str(Path(expected_attempt_dir).resolve()),
}
for field, value in expected.items():
    if payload.get(field) != value:
        raise SystemExit(f"archiver readiness {field} mismatch")
PY
kill -0 "$archiver_pid" 2>/dev/null ||
  fail "checkpoint archiver exited after readiness"
[ ! -e "$ARCHIVE_FAILURE_FILE" ] ||
  fail "checkpoint archiver failed before trainer launch"
verify_runtime_file "$PYTHON" "$PYTHON_SHA256"
verify_runtime_file "$TORCHRUN" "$TORCHRUN_SHA256"
verify_runtime_file "$OBSUTIL" "$OBSUTIL_SHA256"
verify_runtime_file "$ARCHIVER" "$ARCHIVER_SHA256"
verify_runtime_file "$VALIDATOR" "$VALIDATOR_SHA256"
verify_runtime_file "$TRAINER" "$TRAINER_SHA256"
verify_runtime_file "$FORMAL_CONFIG" "$CONFIG_SHA256"
verify_runtime_file "$CONTRACT_VERIFICATION" "$CONTRACT_VERIFICATION_SHA256"
verify_runtime_file "$FULL_TRAINING_CONTRACT" "$FULL_TRAINING_CONTRACT_SHA256"

set +e
setsid "${trainer[@]}" >"$ATTEMPT_DIR/trainer.log" 2>&1 &
trainer_pid=$!
trainer_rc=
graceful_stop_deadline=0
while kill -0 "$trainer_pid" 2>/dev/null; do
  now=$(date +%s)
  if [ "$archiver_failed" -eq 0 ] &&
     ! kill -0 "$archiver_pid" 2>/dev/null &&
     [ ! -e "$ARCHIVER_STOP_SENTINEL" ]; then
    wait "$archiver_pid"
    archiver_rc=$?
    record_archiver_failure "$archiver_rc"
    stop_archiver_group
    archiver_failed=1
    echo "!! checkpoint archiver exited unexpectedly rc=$archiver_rc" >&2
  fi
  if { [ "$archiver_failed" -eq 1 ] ||
        [ -e "$ARCHIVE_FAILURE_FILE" ] ||
        [ "$now" -ge "$soft_stop_epoch" ]; } &&
     [ "$graceful_stop_deadline" -eq 0 ]; then
    "$PYTHON" - "$ATTEMPT_DIR/graceful_stop.requested" \
      "$FORMAL_AUTHORIZATION_VERIFICATION_SHA256" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
path, authorization_sha = sys.argv[1:]
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump({
        "status": "GRACEFUL_STOP_REQUESTED",
        "authorization_verification_sha256": authorization_sha,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    [ "$?" -eq 0 ] || fail "cannot publish the graceful-stop sentinel"
    graceful_stop_deadline=$((now + ARCHIVE_GRACE_SECONDS))
  fi
  if [ "$graceful_stop_deadline" -gt 0 ] &&
     [ "$now" -ge "$graceful_stop_deadline" ]; then
    kill -TERM -- "-$trainer_pid" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$trainer_pid" 2>/dev/null || true
    break
  fi
  sleep 5
done
wait "$trainer_pid"
trainer_rc=$?
set -e

touch "$ARCHIVER_STOP_SENTINEL"
stop_archiver_group
rm -f "$ARCHIVER_STOP_SENTINEL"
if [ "$archiver_failed" -eq 1 ] || [ -e "$ARCHIVE_FAILURE_FILE" ]; then
  write_status ARCHIVER_FAILURE_SOFT_STOP "$trainer_rc"
  fail "archiver failed; trainer received a coordinated soft stop"
fi

# Drain every numbered checkpoint through the strict sequence before reporting.
set +e
RUN_DIR="$RUN_DIR" OBS_DST="$OBS_DST" FORMAL_CONFIG="$FORMAL_CONFIG" \
CONTRACT_VERIFICATION="$CONTRACT_VERIFICATION" \
CONTRACT_VERIFICATION_SHA256="$CONTRACT_VERIFICATION_SHA256" \
VALIDATOR="$VALIDATOR" OBSUTIL="$OBSUTIL" PYTHON="$PYTHON" \
OBSUTIL_SHA256="$OBSUTIL_SHA256" PYTHON_SHA256="$PYTHON_SHA256" \
KEEP_LOCAL_VERIFIED="$KEEP_LOCAL_VERIFIED" POLL_SECONDS="$ARCHIVE_POLL_SECONDS" \
ARCHIVE_ONCE=1 "$ARCHIVER" >"$ATTEMPT_DIR/final_archive.log" 2>&1
final_archive_rc=$?
set -e
if [ "$final_archive_rc" -ne 0 ]; then
  record_archiver_failure "$final_archive_rc"
  write_status FINAL_ARCHIVE_FAILED "$final_archive_rc"
  fail "final checkpoint archive and readback drain failed"
fi

if [ "$trainer_rc" -eq 0 ]; then
  "$PYTHON" - "$RUN_DIR/.formal500k_archive/verified/LATEST_VERIFIED.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "PASS_STRICT_CHECKPOINT_OBS_READBACK":
    raise SystemExit("final LATEST_VERIFIED status is invalid")
if payload.get("step") != 500000:
    raise SystemExit("formal trainer exited successfully without verified step 500000")
if payload.get("formal_training_authorized") is not False:
    raise SystemExit("final LATEST_VERIFIED authorization field is invalid")
PY
  write_status PASS_FORMAL500K_EXECUTION_AND_READBACK 0
  exit 0
fi
if [ "$trainer_rc" -eq 75 ] || [ "$trainer_rc" -eq 124 ]; then
  write_status SOFT_STOPPED_BEFORE_HARD_STOP "$trainer_rc"
  exit 75
fi
write_status TRAINER_FAILED "$trainer_rc"
exit "$trainer_rc"
