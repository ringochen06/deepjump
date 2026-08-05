#!/usr/bin/env bash
# Incrementally validate and archive immutable formal-training checkpoints.
# LATEST_VERIFIED is published only after an independent download passes the
# same strict checkpoint validator as the local artifact.
set -euo pipefail

RUN_DIR=${RUN_DIR:?set RUN_DIR to the claimed formal run directory}
OBS_DST=${OBS_DST:?set OBS_DST to the immutable formal-run OBS prefix}
FORMAL_CONFIG=${FORMAL_CONFIG:?set FORMAL_CONFIG to the rendered sealed config}
CONTRACT_VERIFICATION=${CONTRACT_VERIFICATION:?set CONTRACT_VERIFICATION}
CONTRACT_VERIFICATION_SHA256=${CONTRACT_VERIFICATION_SHA256:?set CONTRACT_VERIFICATION_SHA256}
VALIDATOR=${VALIDATOR:-$(cd "$(dirname "$0")/../.." && pwd)/scripts/validate_training_checkpoint.py}
OBSUTIL=${OBSUTIL:-obsutil}
PYTHON=${PYTHON:-python3}
OBSUTIL_SHA256=${OBSUTIL_SHA256:?set OBSUTIL_SHA256}
PYTHON_SHA256=${PYTHON_SHA256:?set PYTHON_SHA256}
POLL_SECONDS=${POLL_SECONDS:-60}
ARCHIVE_ONCE=${ARCHIVE_ONCE:-0}
KEEP_LOCAL_VERIFIED=${KEEP_LOCAL_VERIFIED:-3}
EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-8}
EXPECTED_FINAL_STEP=${EXPECTED_FINAL_STEP:-500000}
EXPECTED_LR_HORIZON_STEPS=${EXPECTED_LR_HORIZON_STEPS:-500000}
EXPECTED_DELTA=${EXPECTED_DELTA:-1}
ARCHIVER_READY_FILE=${ARCHIVER_READY_FILE:-}
ARCHIVER_RUN_ID=${ARCHIVER_RUN_ID:-}
ARCHIVER_ATTEMPT_DIR=${ARCHIVER_ATTEMPT_DIR:-}

fail() {
  echo "!! $*" >&2
  exit 1
}

[[ "$OBS_DST" == obs://* ]] || fail "OBS_DST must be an obs:// prefix"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "POLL_SECONDS must be positive"
[[ "$KEEP_LOCAL_VERIFIED" =~ ^[1-9][0-9]*$ ]] || fail "KEEP_LOCAL_VERIFIED must be positive"
[[ "$EXPECTED_WORLD_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "EXPECTED_WORLD_SIZE must be positive"
[[ "$EXPECTED_FINAL_STEP" =~ ^[1-9][0-9]*$ ]] || fail "EXPECTED_FINAL_STEP must be positive"
[[ "$EXPECTED_LR_HORIZON_STEPS" =~ ^[1-9][0-9]*$ ]] || fail "EXPECTED_LR_HORIZON_STEPS must be positive"
[[ "$EXPECTED_DELTA" =~ ^[1-9][0-9]*$ ]] || fail "EXPECTED_DELTA must be positive"
[ -d "$RUN_DIR" ] || fail "run directory not found: $RUN_DIR"
[ -f "$FORMAL_CONFIG" ] || fail "formal config not found: $FORMAL_CONFIG"
[ -f "$CONTRACT_VERIFICATION" ] || fail "contract verification not found: $CONTRACT_VERIFICATION"
[ -f "$VALIDATOR" ] || fail "checkpoint validator not found: $VALIDATOR"
command -v "$OBSUTIL" >/dev/null || fail "obsutil not found: $OBSUTIL"
command -v "$PYTHON" >/dev/null || fail "python not found: $PYTHON"
command -v flock >/dev/null || fail "flock is required for archiver ownership"

STATE_DIR="$RUN_DIR/.formal500k_archive"
VERIFIED_DIR="$STATE_DIR/verified"
HISTORY_DIR="$STATE_DIR/history"
REPORT_DIR="$STATE_DIR/reports"
SNAPSHOT_DIR="$STATE_DIR/snapshots"
mkdir -p "$VERIFIED_DIR" "$HISTORY_DIR" "$REPORT_DIR" "$SNAPSHOT_DIR"
ARCHIVER_LOCK="$STATE_DIR/archiver.lock"
exec 8>"$ARCHIVER_LOCK"
flock -n 8 || fail "another checkpoint archiver owns $ARCHIVER_LOCK"

sha256_file() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_runtime_tools() {
  [ "$(sha256_file "$PYTHON")" = "$PYTHON_SHA256" ] ||
    fail "runtime Python SHA256 drift"
  [ "$(sha256_file "$OBSUTIL")" = "$OBSUTIL_SHA256" ] ||
    fail "runtime obsutil SHA256 drift"
}

CONFIG_SHA256=$(sha256_file "$FORMAL_CONFIG")

obs_cp() {
  "$OBSUTIL" cp "$1" "$2" -f >/dev/null
}

remote_state() {
  local object=$1 output rc
  output=$(mktemp "${TMPDIR:-/tmp}/deepjump-formal500k-stat.XXXXXX")
  set +e
  "$OBSUTIL" stat "$object" >"$output" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    rm -f "$output"
    printf exists
    return 0
  fi
  if grep -Eiq '(^|[^0-9])404([^0-9]|$)|NoSuchKey' "$output"; then
    rm -f "$output"
    printf missing
    return 0
  fi
  echo "!! OBS stat failed without a 404 for $object" >&2
  sed -n '1,20p' "$output" >&2
  rm -f "$output"
  return 2
}

snapshot_regular_file() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source, flags)
with os.fdopen(descriptor, "rb") as handle:
    before = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"source is not a regular file: {source}")
    raw = handle.read()
    after = os.fstat(handle.fileno())
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
)
if identity(before) != identity(after) or len(raw) != before.st_size:
    raise SystemExit(f"source changed while snapshotting: {source}")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
temporary.write_bytes(raw)
os.replace(temporary, destination)
PY
}

snapshot_history_through_step() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_step = int(sys.argv[3])
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source, flags)
with os.fdopen(descriptor, "rb") as handle:
    before = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit(f"trainer history is not a regular file: {source}")
    raw = handle.read()
    after = os.fstat(handle.fileno())
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
)
if identity(before) != identity(after) or len(raw) != before.st_size:
    raise SystemExit(f"trainer history changed while snapshotting: {source}")
history = json.loads(raw.decode("utf-8"))
if not isinstance(history, list):
    raise SystemExit("trainer history must be a JSON list")
for index, record in enumerate(history):
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("step"), int)
        or isinstance(record.get("step"), bool)
    ):
        raise SystemExit(f"trainer history record {index} has an invalid step")
# A lagging archiver may observe validation records newer than an older
# numbered checkpoint. Preserve the exact trainer-owned prefix through that
# checkpoint; never synthesize a missing record.
prefix = [record for record in history if record["step"] <= expected_step]
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(prefix, indent=2) + "\n")
os.replace(temporary, destination)
PY
}

validate_checkpoint() {
  local checkpoint=$1 history=$2 step=$3 digest=$4 output=$5 history_mode
  history_mode=through
  [ "$step" -eq "$EXPECTED_FINAL_STEP" ] && history_mode=final
  "$PYTHON" "$VALIDATOR" \
    --checkpoint "$checkpoint" \
    --history "$history" \
    --expected-step "$step" \
    --expected-world-size "$EXPECTED_WORLD_SIZE" \
    --history-mode "$history_mode" \
    --expected-delta "$EXPECTED_DELTA" \
    --require-full-tensor \
    --expected-lr-horizon-steps "$EXPECTED_LR_HORIZON_STEPS" \
    --expected-config "$FORMAL_CONFIG" \
    --expected-contract-verification "$CONTRACT_VERIFICATION" \
    --expected-contract-verification-sha256 "$CONTRACT_VERIFICATION_SHA256" \
    --expected-checkpoint-sha256 "$digest" \
    --output "$output" >/dev/null
}

upload_immutable() {
  local source=$1 object=$2 expected_sha=$3 state readback
  state=$(remote_state "$object") || return
  readback=$(mktemp "${TMPDIR:-/tmp}/deepjump-formal500k-object.XXXXXX")
  if [ "$state" = missing ]; then
    obs_cp "$source" "$object"
  elif [ "$state" != exists ]; then
    rm -f "$readback"
    fail "unexpected OBS state for $object: $state"
  fi
  obs_cp "$object" "$readback"
  if [ "$(sha256_file "$readback")" != "$expected_sha" ]; then
    rm -f "$readback"
    fail "immutable OBS object conflicts or readback SHA mismatches: $object"
  fi
  rm -f "$readback"
}

write_verified_marker() {
  "$PYTHON" - "$@" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output, step, checkpoint_path, checkpoint_sha, checkpoint_object, history_sha, history_object,
    local_report_sha, local_report_object, remote_report_sha, remote_report_object,
    marker_object, latest_object, config_sha, contract_sha,
) = sys.argv[1:]
payload = {
    "schema": "deepjump.formal500k_verified_checkpoint.v1",
    "status": "PASS_STRICT_CHECKPOINT_OBS_READBACK",
    "step": int(step),
    "checkpoint_sha256": checkpoint_sha,
    "checkpoint_object": checkpoint_object,
    "local_checkpoint_stat": {
        "device": os.stat(checkpoint_path, follow_symlinks=False).st_dev,
        "inode": os.stat(checkpoint_path, follow_symlinks=False).st_ino,
        "size": os.stat(checkpoint_path, follow_symlinks=False).st_size,
        "mtime_ns": os.stat(checkpoint_path, follow_symlinks=False).st_mtime_ns,
        "ctime_ns": os.stat(checkpoint_path, follow_symlinks=False).st_ctime_ns,
    },
    "history_sha256": history_sha,
    "history_object": history_object,
    "local_validator_report_sha256": local_report_sha,
    "local_validator_report_object": local_report_object,
    "remote_validator_report_sha256": remote_report_sha,
    "remote_validator_report_object": remote_report_object,
    "marker_object": marker_object,
    "latest_verified_object": latest_object,
    "config_sha256": config_sha,
    "contract_verification_sha256": contract_sha,
    "resume_semantics": "state_consistent_non_bitwise_crop_and_noise",
    "formal_training_authorized": False,
    "verified_at": datetime.now(timezone.utc).isoformat(),
}
path = Path(output)
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

local_marker_matches() {
  "$PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
valid = (
    payload.get("status") == "PASS_STRICT_CHECKPOINT_OBS_READBACK"
    and payload.get("step") == int(sys.argv[2])
    and payload.get("checkpoint_sha256") == sys.argv[3]
    and payload.get("checkpoint_object") == sys.argv[4]
    and payload.get("config_sha256") == sys.argv[5]
    and payload.get("contract_verification_sha256") == sys.argv[6]
    and payload.get("formal_training_authorized") is False
)
raise SystemExit(0 if valid else 1)
PY
}

local_marker_matches_checkpoint() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import json
import hashlib
import os
import stat
import sys
payload = json.load(open(sys.argv[1]))
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(sys.argv[2], flags)
with os.fdopen(descriptor, "rb") as handle:
    value = os.fstat(handle.fileno())
    if not stat.S_ISREG(value.st_mode):
        raise SystemExit("checkpoint is not a regular file")
    hasher = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
    digest = hasher.hexdigest()
actual = {
    "device": value.st_dev,
    "inode": value.st_ino,
    "size": value.st_size,
    "mtime_ns": value.st_mtime_ns,
    "ctime_ns": value.st_ctime_ns,
}
valid = (
    payload.get("status") == "PASS_STRICT_CHECKPOINT_OBS_READBACK"
    and payload.get("step") == int(sys.argv[3])
    and payload.get("formal_training_authorized") is False
    and payload.get("local_checkpoint_stat") == actual
    and payload.get("checkpoint_sha256") == digest
)
raise SystemExit(0 if valid else 1)
PY
}

write_local_marker_cache() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import json
import os
import sys
from pathlib import Path
source, checkpoint, destination = map(Path, sys.argv[1:])
payload = json.loads(source.read_text())
value = os.stat(checkpoint, follow_symlinks=False)
payload["local_checkpoint_stat"] = {
    "device": value.st_dev,
    "inode": value.st_ino,
    "size": value.st_size,
    "mtime_ns": value.st_mtime_ns,
    "ctime_ns": value.st_ctime_ns,
}
temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY
}

remote_marker_proves_checkpoint() {
  local object=$1 step=$2 digest=$3 readback rc
  readback=$(mktemp "${TMPDIR:-/tmp}/deepjump-formal500k-marker.XXXXXX")
  set +e
  obs_cp "$object" "$readback"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    local_marker_matches "$readback" "$step" "$digest" \
      "$OBS_DST/checkpoints/ckpt_${step}.pt" \
      "$CONFIG_SHA256" "$CONTRACT_VERIFICATION_SHA256"
    rc=$?
  fi
  set -e
  rm -f "$readback"
  return "$rc"
}

publish_latest_monotonically() {
  local candidate=$1 candidate_step=$2 candidate_digest=$3 local_cache=$4
  local latest_object="$OBS_DST/LATEST_VERIFIED.json" readback state action
  readback=$(mktemp "${TMPDIR:-/tmp}/deepjump-formal500k-latest.XXXXXX")
  state=$(remote_state "$latest_object") || {
    rm -f "$readback"
    return 1
  }
  if [ "$state" = exists ]; then
    obs_cp "$latest_object" "$readback"
    action=$("$PYTHON" - "$readback" "$candidate_step" "$candidate_digest" \
      "$OBS_DST" "$CONFIG_SHA256" "$CONTRACT_VERIFICATION_SHA256" <<'PY'
import json
import re
import sys
payload = json.load(open(sys.argv[1]))
candidate_step = int(sys.argv[2])
candidate_digest, obs_dst, config_sha, contract_sha = sys.argv[3:]
step = payload.get("step")
digest = payload.get("checkpoint_sha256")
if (
    payload.get("status") != "PASS_STRICT_CHECKPOINT_OBS_READBACK"
    or payload.get("formal_training_authorized") is not False
    or not isinstance(step, int)
    or isinstance(step, bool)
    or re.fullmatch(r"[0-9a-f]{64}", digest or "") is None
    or payload.get("checkpoint_object") != f"{obs_dst}/checkpoints/ckpt_{step}.pt"
    or payload.get("config_sha256") != config_sha
    or payload.get("contract_verification_sha256") != contract_sha
):
    raise SystemExit("existing LATEST_VERIFIED is malformed or package-inconsistent")
if step == candidate_step and digest != candidate_digest:
    raise SystemExit("existing LATEST_VERIFIED conflicts at the same step")
print("keep" if step >= candidate_step else "replace")
PY
    ) || {
      rm -f "$readback"
      return 1
    }
    if [ "$action" = keep ]; then
      snapshot_regular_file "$readback" "$local_cache"
      rm -f "$readback"
      return 0
    fi
  fi
  obs_cp "$candidate" "$latest_object"
  obs_cp "$latest_object" "$readback"
  cmp -s "$candidate" "$readback" || {
    rm -f "$readback"
    fail "LATEST_VERIFIED forced readback mismatch"
  }
  snapshot_regular_file "$readback" "$local_cache"
  rm -f "$readback"
}

archive_checkpoint() {
  local checkpoint=$1 name step digest marker history_snapshot local_report
  local checkpoint_snapshot
  local checkpoint_object history_object local_report_object remote_report_object
  local marker_object latest_object history_sha local_report_sha remote_report_sha
  local candidate_marker readback_dir remote_checkpoint remote_history remote_report
  local recovered_marker
  local config_sha
  name=$(basename "$checkpoint")
  step=${name#ckpt_}
  step=${step%.pt}
  [[ "$step" =~ ^[0-9]+$ ]] || fail "invalid numbered checkpoint: $name"
  [ "$step" -le "$EXPECTED_FINAL_STEP" ] || fail "checkpoint exceeds frozen endpoint: $name"

  marker="$VERIFIED_DIR/${name}.readback.json"
  if [ -f "$marker" ] &&
     local_marker_matches_checkpoint "$marker" "$checkpoint" "$step"; then
    return 0
  fi
  checkpoint_snapshot="$SNAPSHOT_DIR/$name"
  snapshot_regular_file "$checkpoint" "$checkpoint_snapshot"
  digest=$(sha256_file "$checkpoint_snapshot")
  marker_object="$OBS_DST/verified/${name}.readback.json"
  if [ "$(remote_state "$marker_object")" = exists ] &&
     remote_marker_proves_checkpoint "$marker_object" "$step" "$digest"; then
    recovered_marker=$(mktemp "${TMPDIR:-/tmp}/deepjump-formal500k-recovered-marker.XXXXXX")
    obs_cp "$marker_object" "$recovered_marker"
    write_local_marker_cache "$recovered_marker" "$checkpoint" "$marker"
    publish_latest_monotonically "$recovered_marker" "$step" "$digest" \
      "$VERIFIED_DIR/LATEST_VERIFIED.json"
    rm -f "$recovered_marker"
    rm -f "$checkpoint_snapshot"
    echo ">> recovered prior strict OBS marker step=$step sha256=$digest"
    return 0
  fi
  [ -f "$RUN_DIR/history.json" ] || fail "history.json is missing for $name"
  history_snapshot="$HISTORY_DIR/history_${step}.json"
  snapshot_history_through_step "$RUN_DIR/history.json" "$history_snapshot" "$step"
  local_report="$REPORT_DIR/local_validator_${step}.json"
  validate_checkpoint "$checkpoint_snapshot" "$history_snapshot" "$step" "$digest" "$local_report"

  checkpoint_object="$OBS_DST/checkpoints/$name"
  history_object="$OBS_DST/history/history_${step}.json"
  local_report_object="$OBS_DST/validation/local_${step}.json"
  remote_report_object="$OBS_DST/validation/remote_readback_${step}.json"
  latest_object="$OBS_DST/LATEST_VERIFIED.json"
  history_sha=$(sha256_file "$history_snapshot")
  local_report_sha=$(sha256_file "$local_report")
  config_sha=$CONFIG_SHA256

  # Required order: local validation, immutable upload, forced readback,
  # independent remote validation, marker readback, then LATEST_VERIFIED.
  upload_immutable "$checkpoint_snapshot" "$checkpoint_object" "$digest"
  upload_immutable "$history_snapshot" "$history_object" "$history_sha"
  upload_immutable "$local_report" "$local_report_object" "$local_report_sha"

  readback_dir=$(mktemp -d "${TMPDIR:-/tmp}/deepjump-formal500k-readback.XXXXXX")
  remote_checkpoint="$readback_dir/$name"
  remote_history="$readback_dir/history_${step}.json"
  remote_report="$readback_dir/remote_validator_${step}.json"
  candidate_marker="$readback_dir/${name}.candidate.json"
  obs_cp "$checkpoint_object" "$remote_checkpoint"
  [ "$(sha256_file "$remote_checkpoint")" = "$digest" ] || {
    rm -rf "$readback_dir"
    fail "forced checkpoint readback SHA mismatch: $checkpoint_object"
  }
  obs_cp "$history_object" "$remote_history"
  [ "$(sha256_file "$remote_history")" = "$history_sha" ] || {
    rm -rf "$readback_dir"
    fail "forced history readback SHA mismatch: $history_object"
  }
  validate_checkpoint "$remote_checkpoint" "$remote_history" "$step" "$digest" "$remote_report"
  remote_report_sha=$(sha256_file "$remote_report")
  upload_immutable "$remote_report" "$remote_report_object" "$remote_report_sha"

  write_verified_marker \
    "$candidate_marker" "$step" "$checkpoint" "$digest" "$checkpoint_object" \
    "$history_sha" "$history_object" "$local_report_sha" "$local_report_object" \
    "$remote_report_sha" "$remote_report_object" "$marker_object" "$latest_object" \
    "$config_sha" "$CONTRACT_VERIFICATION_SHA256"
  upload_immutable "$candidate_marker" "$marker_object" "$(sha256_file "$candidate_marker")"
  remote_marker_proves_checkpoint "$marker_object" "$step" "$digest" || {
    rm -rf "$readback_dir"
    fail "remote verified marker readback does not prove $name"
  }
  # LATEST_VERIFIED is monotonic and is independently read back before the
  # durable local verified marker is published.
  publish_latest_monotonically "$candidate_marker" "$step" "$digest" \
    "$VERIFIED_DIR/LATEST_VERIFIED.json"
  write_local_marker_cache "$candidate_marker" "$checkpoint" "$marker"
  rm -f "$checkpoint_snapshot"
  rm -rf "$readback_dir"
  echo ">> strict OBS checkpoint verification PASS step=$step sha256=$digest"
}

retain_only_remote_verified() {
  local keep=$KEEP_LOCAL_VERIFIED index=0 checkpoint name step digest marker marker_object
  while IFS= read -r checkpoint; do
    [ -n "$checkpoint" ] || continue
    index=$((index + 1))
    [ "$index" -le "$keep" ] && continue
    name=$(basename "$checkpoint")
    step=${name#ckpt_}
    step=${step%.pt}
    digest=$(sha256_file "$checkpoint")
    marker="$VERIFIED_DIR/${name}.readback.json"
    marker_object="$OBS_DST/verified/${name}.readback.json"
    [ -f "$marker" ] || fail "retention refused without local readback marker: $name"
    local_marker_matches "$marker" "$step" "$digest" \
      "$OBS_DST/checkpoints/$name" "$CONFIG_SHA256" \
      "$CONTRACT_VERIFICATION_SHA256" ||
      fail "retention refused because local marker mismatches: $name"
    remote_marker_proves_checkpoint "$marker_object" "$step" "$digest" ||
      fail "retention refused without current remote marker proof: $name"
    rm -f "$checkpoint"
    echo ">> retained remotely verified checkpoint; removed local $name"
  done < <("$PYTHON" - "$RUN_DIR" <<'PY'
import re
import sys
from pathlib import Path
items = []
for path in Path(sys.argv[1]).glob("ckpt_*.pt"):
    match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
    if match:
        items.append((int(match.group(1)), path))
for _, path in sorted(items, reverse=True):
    print(path)
PY
  )
}

archive_round() {
  local discovered=0 new_count=0 checkpoint name step marker
  verify_runtime_tools
  while IFS= read -r checkpoint; do
    [ -n "$checkpoint" ] || continue
    discovered=$((discovered + 1))
    name=$(basename "$checkpoint")
    step=${name#ckpt_}
    step=${step%.pt}
    marker="$VERIFIED_DIR/${name}.readback.json"
    if [ -f "$marker" ] &&
       local_marker_matches_checkpoint "$marker" "$checkpoint" "$step"; then
      continue
    fi
    archive_checkpoint "$checkpoint"
    new_count=$((new_count + 1))
  done < <("$PYTHON" - "$RUN_DIR" <<'PY'
import re
import sys
from pathlib import Path
items = []
for path in Path(sys.argv[1]).glob("ckpt_*.pt"):
    match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
    if match:
        items.append((int(match.group(1)), path))
for _, path in sorted(items):
    print(path)
PY
  )
  [ "$discovered" -gt 0 ] && retain_only_remote_verified
  if [ "$ARCHIVE_ONCE" = 1 ] && [ "$discovered" -eq 0 ]; then
    fail "no immutable numbered checkpoints found in $RUN_DIR"
  fi
  echo ">> archive round discovered=$discovered newly_verified=$new_count"
}

publish_ready() {
  [ -n "$ARCHIVER_READY_FILE" ] || return 0
  [[ "$ARCHIVER_READY_FILE" = /* ]] ||
    fail "ARCHIVER_READY_FILE must be absolute"
  [ -n "$ARCHIVER_RUN_ID" ] || fail "ARCHIVER_RUN_ID is required for readiness"
  [ -n "$ARCHIVER_ATTEMPT_DIR" ] ||
    fail "ARCHIVER_ATTEMPT_DIR is required for readiness"
  "$PYTHON" - "$ARCHIVER_READY_FILE" "$$" "$ARCHIVER_RUN_ID" \
    "$ARCHIVER_ATTEMPT_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
shell_pid = int(sys.argv[2])
run_id = sys.argv[3]
attempt_dir = str(Path(sys.argv[4]).resolve())
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w") as handle:
    json.dump({
        "schema": "deepjump.formal500k_archiver_ready.v1",
        "status": "ARCHIVER_READY_AFTER_INITIAL_ROUND",
        "pid": shell_pid,
        "run_id": run_id,
        "attempt_dir": attempt_dir,
        "ready_at": datetime.now(timezone.utc).isoformat(),
    }, handle, indent=2, sort_keys=True)
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

archive_round
publish_ready
[ "$ARCHIVE_ONCE" = 1 ] && exit 0
while true; do
  sleep "$POLL_SECONDS"
  archive_round
done
