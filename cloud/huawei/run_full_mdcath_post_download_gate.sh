#!/usr/bin/env bash
# Qualify the completed 5,398-domain staging tree, persist it to OBS, remount it
# read-only, and only then hand off to the bounded expanded-data runner.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-4320}
HARD_STOP_UNIT="deepjump-full-mdcath-post-download-hard-stop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
DATA_REMOUNTED_RO=0
OBS_BOUND=0

shutdown_on_exit() {
  code=$?
  trap - EXIT
  set +e
  if [[ "$DATA_REMOUNTED_RO" == 1 && -n "${DATA_MOUNT:-}" ]]; then
    sudo -n mount -o remount,rw "$DATA_MOUNT" || true
  fi
  failure_root=${RUN_DIR:-/var/tmp/deepjump-full-mdcath-post-download-failure-$$}
  mkdir -p "$failure_root"
  failure_status_tmp=$(mktemp "$failure_root/.failure_status.XXXXXX")
  printf '{"schema":"deepjump.full_mdcath_post_download_failure.v1","run_id":"%s","exit_code":%s,"obs_safely_bound":%s,"formal_training_authorized":false}\n' \
    "${RUN_ID:-unknown}" "$code" "$([[ "$OBS_BOUND" == 1 ]] && printf true || printf false)" \
    > "$failure_status_tmp"
  mv "$failure_status_tmp" "$failure_root/failure_status.json"
  failure_archive="/var/tmp/deepjump-full-mdcath-post-download-failure-${RUN_ID:-unknown}-$(date -u +%Y%m%dT%H%M%SZ)-$$.tar.gz"
  failure_archive_tmp=$(mktemp "/var/tmp/.deepjump-post-download-failure.XXXXXX.tar.gz")
  if tar -C "$failure_root" -czf "$failure_archive_tmp" .; then
    mv "$failure_archive_tmp" "$failure_archive"
    failure_sha="$failure_archive.sha256"
    (cd "$(dirname "$failure_archive")" && sha256sum "$(basename "$failure_archive")") \
      > "$failure_sha"
    if [[ "$OBS_BOUND" == 1 && -n "${OBS_DST:-}" ]] && command -v obsutil >/dev/null; then
      failure_readback=$(mktemp -d "/var/tmp/deepjump-post-download-failure-readback.XXXXXX")
      archive_name=$(basename "$failure_archive")
      sha_name=$(basename "$failure_sha")
      timeout 10m obsutil cp "$failure_archive" "$OBS_DST/failure/$archive_name"
      upload_archive_code=$?
      timeout 5m obsutil cp "$failure_sha" "$OBS_DST/failure/$sha_name"
      upload_sha_code=$?
      timeout 10m obsutil cp "$OBS_DST/failure/$archive_name" \
        "$failure_readback/$archive_name"
      readback_archive_code=$?
      timeout 5m obsutil cp "$OBS_DST/failure/$sha_name" \
        "$failure_readback/$sha_name"
      readback_sha_code=$?
      if [[ "$upload_archive_code" == 0 && "$upload_sha_code" == 0 && \
        "$readback_archive_code" == 0 && "$readback_sha_code" == 0 ]] \
        && cmp "$failure_sha" "$failure_readback/$sha_name" \
        && (cd "$failure_readback" && sha256sum --check "$sha_name"); then
        printf 'PASS_OBS_FAILURE_ARCHIVE_READBACK\n' \
          > "$failure_root/failure_archive_persistence.txt"
      else
        printf 'FAILURE_ARCHIVE_LOCAL_ONLY_OBS_READBACK_FAILED archive=%s\n' \
          "$failure_archive" > "$failure_root/failure_archive_persistence.txt"
      fi
    else
      persistence_tmp=$(mktemp "$failure_root/.failure_archive_persistence.XXXXXX")
      printf 'FAILURE_ARCHIVE_LOCAL_ONLY_OBS_NOT_SAFELY_BOUND archive=%s\n' \
        "$failure_archive" > "$persistence_tmp"
      mv "$persistence_tmp" "$failure_root/failure_archive_persistence.txt"
    fi
  else
    rm -f "$failure_archive_tmp"
    printf 'FAILURE_ARCHIVE_CREATION_FAILED\n' \
      > "$failure_root/failure_archive_persistence.txt"
  fi
  printf 'Full-mdCATH post-download gate exit=%s; requesting shutdown at %s\n' \
    "$code" "$(date -Is)"
  sudo -n shutdown -h now || true
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || {
  printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2
  exit 2
}
[[ "$HARD_STOP_MINUTES" == 4320 ]] || {
  printf 'HARD_STOP_MINUTES must remain 4320\n' >&2
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

# Run-specific authority is required only after the independent hard stop exists.
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set the reviewed deployed commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set the authorized GPU hostname}
DATA_ROOT=${DATA_ROOT:?set the completed full-mdCATH staging root}
EXPECTED_DATA_MOUNT=${EXPECTED_DATA_MOUNT:-/data-full}
OFFICIAL_LIST=${OFFICIAL_LIST:?set the frozen official 5,398-domain list}
SOURCE_INVENTORY=${SOURCE_INVENTORY:?set the frozen source inventory JSONL}
PANEL_REGISTRY=${PANEL_REGISTRY:-$REPO/configs/full_mdcath_evaluation_exclusion_registry.json}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RECOVERY_PREVIOUS_COMMIT=${RECOVERY_PREVIOUS_COMMIT:-}

EXPECTED_SOURCE_REVISION=5e3ed8aec62b689e01751db16275fdcdbc39e47f
EXPECTED_OFFICIAL_SHA256=295c6da1c9f8846a1ea3993eca12a3232d16a2b3a4b0d8791c7c45392186709b
EXPECTED_SOURCE_INVENTORY_SHA256=2e6e3602a0858aaafc849cfa7cc1ee7e076736cb15335d7914726898f06f6cdf
EXPECTED_PANEL_REGISTRY_SHA256=65f14cb45c1af84ca6a7e97affe6974232fd3ec12da69a875e9a089525943097
EXPECTED_H5_FILES=5398
EXPECTED_H5_BYTES=3613998101757
EXPECTED_TRAJECTORIES=134950

[[ "$EXPECTED_REPO_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'EXPECTED_REPO_COMMIT must be a full lowercase commit SHA\n' >&2
  exit 2
}
if [[ -n "$RECOVERY_PREVIOUS_COMMIT" && \
  ! "$RECOVERY_PREVIOUS_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'RECOVERY_PREVIOUS_COMMIT must be empty or a full lowercase commit SHA\n' >&2
  exit 2
fi
[[ "$RUN_ID" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  printf 'RUN_ID must be UTC basic timestamp YYYYMMDDTHHMMSSZ\n' >&2
  exit 2
}
[[ "$BUCKET" == obs://* ]] || { printf 'BUCKET must use obs://\n' >&2; exit 2; }
for path in "$DATA_ROOT" "$OFFICIAL_LIST" "$SOURCE_INVENTORY" "$PANEL_REGISTRY"; do
  [[ "$path" == /* ]] || { printf 'path must be absolute: %s\n' "$path" >&2; exit 2; }
done
[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
[[ -x "$PYTHON" ]] || { printf 'Python runtime missing\n' >&2; exit 2; }
command -v obsutil >/dev/null
command -v sha256sum >/dev/null
command -v findmnt >/dev/null
command -v tar >/dev/null

cd "$REPO"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$REPO/src"
for required_script in \
  scripts/audit_full_mdcath_staging.py \
  scripts/build_expanded_data_partition.py \
  scripts/build_full_training_data_contract.py \
  scripts/verify_full_mdcath_obs_readback.py \
  scripts/verify_obsutil_empty_prefix.py \
  cloud/huawei/run_contracted_expanded_data_gate.sh; do
  [[ -f "$required_script" && ! -L "$required_script" ]] || {
    printf 'required script missing or symlinked: %s\n' "$required_script" >&2
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

DATA_ROOT=$(realpath -e "$DATA_ROOT")
DATA_MOUNT=$(findmnt -T "$DATA_ROOT" -n -o TARGET)
[[ "$DATA_MOUNT" == "$EXPECTED_DATA_MOUNT" && "$DATA_MOUNT" != / ]] || {
  printf 'staging mount mismatch: actual=%s expected=%s\n' \
    "$DATA_MOUNT" "$EXPECTED_DATA_MOUNT" >&2
  exit 2
}
mount_options=$(findmnt -T "$DATA_ROOT" -n -o OPTIONS)
case ",$mount_options," in
  *,rw,*) ;;
  *) printf 'staging mount must initially be read-write: %s\n' "$mount_options" >&2; exit 2 ;;
esac
[[ -d "$DATA_ROOT/data" && ! -L "$DATA_ROOT/data" ]] || {
  printf 'full-mdCATH data directory is missing or symlinked\n' >&2
  exit 2
}
for path in "$OFFICIAL_LIST" "$SOURCE_INVENTORY" "$PANEL_REGISTRY"; do
  [[ -f "$path" && ! -L "$path" ]] || { printf 'identity input is not regular: %s\n' "$path" >&2; exit 2; }
done
[[ "$(sha256sum "$OFFICIAL_LIST" | awk '{print $1}')" == "$EXPECTED_OFFICIAL_SHA256" ]]
[[ "$(sha256sum "$SOURCE_INVENTORY" | awk '{print $1}')" == "$EXPECTED_SOURCE_INVENTORY_SHA256" ]]
[[ "$(sha256sum "$PANEL_REGISTRY" | awk '{print $1}')" == "$EXPECTED_PANEL_REGISTRY_SHA256" ]]

for service in deepjump-mdcath-download.service deepjump-mdcath-hash.service \
  deepjump-mdcath-copy.service; do
  if systemctl is-active --quiet "$service"; then
    printf 'full-data mutation service is still active: %s\n' "$service" >&2
    exit 2
  fi
done
if pgrep -af '[s]cripts/(download_mdcath|audit_full_mdcath_staging).py'; then
  printf 'conflicting full-data mutation/audit process exists\n' >&2
  exit 2
fi

RUN_DIR="/var/tmp/deepjump-full-mdcath-post-download-$RUN_ID"
QUALIFICATION_ROOT="$DATA_ROOT/control/post_download_qualification/$RUN_ID"
CONTRACT_DIR="$QUALIFICATION_ROOT/full_training_contract"
READBACK_JOURNAL="$QUALIFICATION_ROOT/obs_corpus_readback"
OBS_DST="$BUCKET/deepjump-full-data/full-mdcath-qualified/$RUN_ID"
OBS_CORPUS_DST="$OBS_DST/corpus"
OBS_QUALIFICATION_DST="$OBS_DST/qualification"
mkdir -p "$RUN_DIR" "$QUALIFICATION_ROOT"
[[ -d "$RUN_DIR" && ! -L "$RUN_DIR" ]] || { printf 'invalid run directory\n' >&2; exit 2; }
[[ -d "$QUALIFICATION_ROOT" && ! -L "$QUALIFICATION_ROOT" ]] || {
  printf 'invalid qualification directory\n' >&2
  exit 2
}
[[ ! -L "$DATA_ROOT/control" ]] || { printf 'staging control directory is symlinked\n' >&2; exit 2; }
[[ "$(realpath -e "$QUALIFICATION_ROOT")" == "$QUALIFICATION_ROOT" ]] || {
  printf 'qualification directory escaped or traversed a symlink\n' >&2
  exit 2
}
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

SOURCE_IDENTITY_MANIFEST="$RUN_DIR/tracked_source_sha256.$actual_commit.txt"
SOURCE_IDENTITY_CANDIDATE=$(mktemp "$RUN_DIR/.tracked_source_sha256.XXXXXX")
git ls-files -z | LC_ALL=C sort -z | xargs -0 sha256sum > "$SOURCE_IDENTITY_CANDIDATE"
if [[ -e "$SOURCE_IDENTITY_MANIFEST" ]]; then
  cmp "$SOURCE_IDENTITY_CANDIDATE" "$SOURCE_IDENTITY_MANIFEST"
  rm "$SOURCE_IDENTITY_CANDIDATE"
else
  mv "$SOURCE_IDENTITY_CANDIDATE" "$SOURCE_IDENTITY_MANIFEST"
fi
SOURCE_IDENTITY_MANIFEST_SHA256=$(sha256sum "$SOURCE_IDENTITY_MANIFEST" | awk '{print $1}')
verify_source_identity() {
  [[ "$(git rev-parse HEAD)" == "$actual_commit" ]]
  [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
  [[ "$(sha256sum "$SOURCE_IDENTITY_MANIFEST" | awk '{print $1}')" == \
    "$SOURCE_IDENTITY_MANIFEST_SHA256" ]]
  sha256sum --check --quiet "$SOURCE_IDENTITY_MANIFEST"
}
verify_source_identity

# Own a unique, initially empty OBS hierarchy.  On an interrupted retry the
# exact independently read-back ownership marker is required before reuse.
OWNERSHIP_MARKER="$QUALIFICATION_ROOT/obs_ownership.json"
OWNERSHIP_READBACK="$RUN_DIR/obs_ownership.readback.json"
if [[ ! -e "$OWNERSHIP_MARKER" ]]; then
  timeout 60s obsutil ls "$OBS_DST/" -limit=1 | tee "$RUN_DIR/obs_prefix_preflight.log"
  "$PYTHON" scripts/verify_obsutil_empty_prefix.py "$RUN_DIR/obs_prefix_preflight.log"
  "$PYTHON" - "$OWNERSHIP_MARKER" "$RUN_ID" "$actual_commit" "$OBS_DST" <<'PY'
import json
import os
import sys

output, run_id, commit, obs = sys.argv[1:]
content = (json.dumps({
    "schema": "deepjump.full_mdcath_post_download_obs_ownership.v1",
    "run_id": run_id,
    "commit": commit,
    "obs": obs,
    "formal_training_authorized": False,
}, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(content)
    handle.flush()
    os.fsync(handle.fileno())
PY
  timeout 5m obsutil cp "$OWNERSHIP_MARKER" "$OBS_DST/control/obs_ownership.json"
fi
[[ -f "$OWNERSHIP_MARKER" && ! -L "$OWNERSHIP_MARKER" ]] || {
  printf 'OBS ownership marker is not a regular file\n' >&2
  exit 2
}
rm -f "$OWNERSHIP_READBACK"
timeout 5m obsutil cp "$OBS_DST/control/obs_ownership.json" "$OWNERSHIP_READBACK"
cmp "$OWNERSHIP_MARKER" "$OWNERSHIP_READBACK"
OWNERSHIP_COMMIT=${RECOVERY_PREVIOUS_COMMIT:-$actual_commit}
"$PYTHON" - "$OWNERSHIP_READBACK" "$RUN_ID" "$OWNERSHIP_COMMIT" "$OBS_DST" <<'PY'
import json
import sys

path, run_id, commit, obs = sys.argv[1:]
with open(path, "rb") as handle:
    payload = json.load(handle)
expected = {
    "schema": "deepjump.full_mdcath_post_download_obs_ownership.v1",
    "run_id": run_id,
    "commit": commit,
    "obs": obs,
    "formal_training_authorized": False,
}
if payload != expected:
    raise SystemExit("OBS ownership marker recovery identity mismatch")
PY
OBS_BOUND=1

MANIFEST="$DATA_ROOT/full_mdcath_manifest.json"
STAGING_METADATA="$DATA_ROOT/full_mdcath_staging.metadata.json"
DATA_AUDIT="$DATA_ROOT/full_mdcath_audit.json"
printf 'gate=live_rehash_audit start=%s\n' "$(date -Is)"
verify_source_identity
"$PYTHON" scripts/audit_full_mdcath_staging.py \
  --root "$DATA_ROOT" \
  --official-list "$OFFICIAL_LIST" \
  --source-inventory "$SOURCE_INVENTORY" \
  --source-revision "$EXPECTED_SOURCE_REVISION" \
  --rehash-payloads \
  --manifest-output "$MANIFEST" \
  --metadata-output "$STAGING_METADATA" \
  --audit-output "$DATA_AUDIT" \
  --generating-commit "$actual_commit" \
  | tee "$RUN_DIR/live_rehash_audit.log"
verify_source_identity

printf 'gate=heldout_partition start=%s\n' "$(date -Is)"
PARTITION_TMP=$(mktemp -d "$QUALIFICATION_ROOT/.partition.XXXXXX")
"$PYTHON" scripts/build_expanded_data_partition.py \
  --official-list "$OFFICIAL_LIST" \
  --official-sha256 "$EXPECTED_OFFICIAL_SHA256" \
  --panel-registry "$PANEL_REGISTRY" \
  --expected-panel-registry-sha256 "$EXPECTED_PANEL_REGISTRY_SHA256" \
  --train-output "$PARTITION_TMP/train_eligible_5218.txt" \
  --audit-output "$PARTITION_TMP/expanded_data_partition.json" \
  | tee "$RUN_DIR/partition.log"
if [[ -e "$QUALIFICATION_ROOT/train_eligible_5218.txt" ]]; then
  cmp \
    "$PARTITION_TMP/train_eligible_5218.txt" \
    "$QUALIFICATION_ROOT/train_eligible_5218.txt"
  rm "$PARTITION_TMP/train_eligible_5218.txt"
else
  mv \
    "$PARTITION_TMP/train_eligible_5218.txt" \
    "$QUALIFICATION_ROOT/train_eligible_5218.txt"
fi
if [[ -e "$QUALIFICATION_ROOT/expanded_data_partition.json" ]]; then
  "$PYTHON" scripts/verify_expanded_data_partition_recovery.py \
    --sealed "$QUALIFICATION_ROOT/expanded_data_partition.json" \
    --candidate "$PARTITION_TMP/expanded_data_partition.json"
  rm "$PARTITION_TMP/expanded_data_partition.json"
else
  mv \
    "$PARTITION_TMP/expanded_data_partition.json" \
    "$QUALIFICATION_ROOT/expanded_data_partition.json"
fi
rmdir "$PARTITION_TMP"
TRAIN_LIST="$QUALIFICATION_ROOT/train_eligible_5218.txt"
PARTITION_AUDIT="$QUALIFICATION_ROOT/expanded_data_partition.json"

sha() { sha256sum "$1" | awk '{print $1}'; }
MANIFEST_SHA256=$(sha "$MANIFEST")
STAGING_METADATA_SHA256=$(sha "$STAGING_METADATA")
DATA_AUDIT_SHA256=$(sha "$DATA_AUDIT")
PARTITION_AUDIT_SHA256=$(sha "$PARTITION_AUDIT")
TRAIN_LIST_SHA256=$(sha "$TRAIN_LIST")

printf 'gate=full_training_contract start=%s\n' "$(date -Is)"
if [[ ! -e "$CONTRACT_DIR" ]]; then
  "$PYTHON" scripts/build_full_training_data_contract.py \
    --output-dir "$CONTRACT_DIR" \
    --data-audit "$DATA_AUDIT" --data-audit-sha256 "$DATA_AUDIT_SHA256" \
    --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256" \
    --official-list "$OFFICIAL_LIST" --official-list-sha256 "$EXPECTED_OFFICIAL_SHA256" \
    --panel-registry "$PANEL_REGISTRY" \
    --partition-audit "$PARTITION_AUDIT" --partition-audit-sha256 "$PARTITION_AUDIT_SHA256" \
    --source-inventory "$SOURCE_INVENTORY" \
    --source-inventory-sha256 "$EXPECTED_SOURCE_INVENTORY_SHA256" \
    --staging-metadata "$STAGING_METADATA" \
    --staging-metadata-sha256 "$STAGING_METADATA_SHA256" \
    --train-list "$TRAIN_LIST" --train-list-sha256 "$TRAIN_LIST_SHA256" \
    | tee "$RUN_DIR/contract.log"
fi
CONTRACT="$CONTRACT_DIR/full_training_data_contract.json"
CONTRACT_SHA256=$(sha "$CONTRACT")
[[ "$(awk '{print $1}' "$CONTRACT_DIR/full_training_data_contract.sha256")" == \
  "$CONTRACT_SHA256" ]]
"$PYTHON" - "$CONTRACT" "$CONTRACT_SHA256" "$DATA_ROOT" \
  "$CONTRACT_DIR/full_mdcath_manifest.json" "$CONTRACT_DIR/train_eligible_5218.txt" <<'PY'
import sys
from deepjump.data_contract import verify_full_training_data_contract

report = verify_full_training_data_contract(
    sys.argv[1], sys.argv[2], configured_root=sys.argv[3],
    configured_manifest=sys.argv[4], configured_domains_file=sys.argv[5],
)
if report.get("status") != "PASS_FULL_TRAINING_DATA_CONTRACT":
    raise SystemExit("full-training contract did not pass")
PY

printf 'gate=obs_qualification_archive start=%s\n' "$(date -Is)"
(cd "$CONTRACT_DIR" && find . -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUN_DIR/qualification_sha256.txt")
timeout --signal=TERM --kill-after=2m 2h obsutil sync \
  "$CONTRACT_DIR" "$OBS_QUALIFICATION_DST"
QUALIFICATION_READBACK=$(mktemp -d "$RUN_DIR/qualification_readback.XXXXXX")
timeout --signal=TERM --kill-after=2m 2h obsutil sync \
  "$OBS_QUALIFICATION_DST" "$QUALIFICATION_READBACK"
cmp "$RUN_DIR/qualification_sha256.txt" \
  <(cd "$QUALIFICATION_READBACK" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
(cd "$QUALIFICATION_READBACK" && sha256sum --check "$RUN_DIR/qualification_sha256.txt")

printf 'gate=obs_corpus_upload start=%s\n' "$(date -Is)"
timeout --signal=TERM --kill-after=5m 36h obsutil sync \
  "$DATA_ROOT/data" "$OBS_CORPUS_DST"
timeout --signal=TERM --kill-after=30s 30m obsutil ls "$OBS_CORPUS_DST/" \
  -limit=0 -bf=raw \
  | tee "$RUN_DIR/obs_corpus_listing.log"
"$PYTHON" - "$RUN_DIR/obs_corpus_listing.log" \
  "$EXPECTED_H5_FILES" "$EXPECTED_H5_BYTES" <<'PY'
import sys
from pathlib import Path
from scripts.verify_obsutil_empty_prefix import prefix_file_inventory

count, total_bytes = prefix_file_inventory(Path(sys.argv[1]).read_text())
expected_count, expected_bytes = int(sys.argv[2]), int(sys.argv[3])
if count != expected_count or total_bytes != expected_bytes:
    raise SystemExit(
        "OBS corpus inventory mismatch: "
        f"files={count}/{expected_count} bytes={total_bytes}/{expected_bytes}"
    )
PY

printf 'gate=obs_corpus_content_readback start=%s\n' "$(date -Is)"
timeout --signal=TERM --kill-after=5m 48h \
  "$PYTHON" scripts/verify_full_mdcath_obs_readback.py \
    --manifest "$MANIFEST" \
    --obs-prefix "$OBS_CORPUS_DST" \
    --work-dir "$READBACK_JOURNAL" \
    --expected-count "$EXPECTED_H5_FILES" \
    --expected-bytes "$EXPECTED_H5_BYTES" \
    | tee "$RUN_DIR/obs_corpus_readback.log"

printf 'gate=obs_readback_journal_archive start=%s\n' "$(date -Is)"
JOURNAL_INVENTORY="$RUN_DIR/obs_readback_journal_sha256.txt"
(cd "$READBACK_JOURNAL" && find . -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$JOURNAL_INVENTORY")
JOURNAL_INVENTORY_SHA256=$(sha "$JOURNAL_INVENTORY")
timeout --signal=TERM --kill-after=2m 2h obsutil sync \
  "$READBACK_JOURNAL" "$OBS_DST/corpus_readback_journal"
timeout 5m obsutil cp "$JOURNAL_INVENTORY" \
  "$OBS_DST/control/obs_readback_journal_sha256.txt"
JOURNAL_READBACK=$(mktemp -d "$RUN_DIR/obs_readback_journal_readback.XXXXXX")
timeout --signal=TERM --kill-after=2m 2h obsutil sync \
  "$OBS_DST/corpus_readback_journal" "$JOURNAL_READBACK"
JOURNAL_INVENTORY_READBACK="$RUN_DIR/obs_readback_journal_sha256.readback.txt"
timeout 5m obsutil cp "$OBS_DST/control/obs_readback_journal_sha256.txt" \
  "$JOURNAL_INVENTORY_READBACK"
cmp "$JOURNAL_INVENTORY" "$JOURNAL_INVENTORY_READBACK"
cmp "$JOURNAL_INVENTORY" \
  <(cd "$JOURNAL_READBACK" && find . -type f -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum)
(cd "$JOURNAL_READBACK" && sha256sum --check "$JOURNAL_INVENTORY")

printf 'gate=sync_and_read_only_mount start=%s\n' "$(date -Is)"
sync
sudo -n mount -o remount,ro "$DATA_MOUNT"
DATA_REMOUNTED_RO=1
mount_options=$(findmnt -T "$DATA_ROOT" -n -o OPTIONS)
case ",$mount_options," in
  *,ro,*) ;;
  *) printf 'qualified staging mount did not become read-only: %s\n' "$mount_options" >&2; exit 2 ;;
esac
"$PYTHON" - "$CONTRACT" "$CONTRACT_SHA256" "$DATA_ROOT" \
  "$CONTRACT_DIR/full_mdcath_manifest.json" "$CONTRACT_DIR/train_eligible_5218.txt" <<'PY'
import sys
from deepjump.data_contract import verify_full_training_data_contract

report = verify_full_training_data_contract(
    sys.argv[1], sys.argv[2], configured_root=sys.argv[3],
    configured_manifest=sys.argv[4], configured_domains_file=sys.argv[5],
)
if report.get("status") != "PASS_FULL_TRAINING_DATA_CONTRACT":
    raise SystemExit("read-only full-training contract verification failed")
PY
verify_source_identity

printf 'gate=completion_marker_readback start=%s\n' "$(date -Is)"
COMPLETION_MARKER="$RUN_DIR/post_download_completion.json"
CORPUS_READBACK_COMPLETION_SHA256=$(sha "$READBACK_JOURNAL/completion.json")
"$PYTHON" - "$COMPLETION_MARKER" "$RUN_ID" "$actual_commit" "$OBS_DST" \
  "$OWNERSHIP_COMMIT" \
  "$CONTRACT_SHA256" "$MANIFEST_SHA256" "$CORPUS_READBACK_COMPLETION_SHA256" \
  "$JOURNAL_INVENTORY_SHA256" "$DATA_MOUNT" "$EXPECTED_H5_FILES" \
  "$EXPECTED_H5_BYTES" "$EXPECTED_TRAJECTORIES" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(output, run_id, commit, obs, obs_upload_commit, contract_sha, manifest_sha, readback_sha,
 journal_inventory_sha, mount, objects, size, trajectories) = sys.argv[1:]
payload = {
    "schema": "deepjump.full_mdcath_post_download_completion.v1",
    "status": "PASS_FULL_MDCATH_POST_DOWNLOAD_QUALIFICATION",
    "run_id": run_id,
    "commit": commit,
    "obs": obs,
    "obs_upload_commit": obs_upload_commit,
    "contract_sha256": contract_sha,
    "manifest_sha256": manifest_sha,
    "corpus_readback_completion_sha256": readback_sha,
    "corpus_readback_journal_inventory_sha256": journal_inventory_sha,
    "mount": mount,
    "mount_read_only": True,
    "h5_files": int(objects),
    "h5_bytes": int(size),
    "trajectories": int(trajectories),
    "formal_training_authorized": False,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
stable_keys = set(payload) - {"completed_at"}
if os.path.lexists(output):
    if os.path.islink(output) or not os.path.isfile(output):
        raise SystemExit("completion marker is not a regular file")
    with open(output, "rb") as handle:
        previous = json.load(handle)
    if {key: previous.get(key) for key in stable_keys} != {
        key: payload[key] for key in stable_keys
    } or set(previous) != set(payload):
        raise SystemExit("completion marker resume identity mismatch")
else:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
PY
COMPLETION_READBACK="$RUN_DIR/post_download_completion.readback.json"
timeout 5m obsutil cp "$COMPLETION_MARKER" "$OBS_DST/completion/post_download_completion.json"
timeout 5m obsutil cp "$OBS_DST/completion/post_download_completion.json" "$COMPLETION_READBACK"
cmp "$COMPLETION_MARKER" "$COMPLETION_READBACK"
[[ "$(sha "$COMPLETION_MARKER")" == "$(sha "$COMPLETION_READBACK")" ]]

# The bounded runner creates and verifies its own independent 480-minute stop,
# revalidates the contract on the read-only mount, and powers off on every exit.
printf 'gate=handoff_contracted_expanded_data start=%s\n' "$(date -Is)"
export REPO PYTHON EXPECTED_REPO_COMMIT EXPECTED_HOSTNAME DATA_ROOT CONTRACT \
  CONTRACT_SHA256 BUCKET
export MANIFEST="$CONTRACT_DIR/full_mdcath_manifest.json"
export TRAIN_LIST="$CONTRACT_DIR/train_eligible_5218.txt"
export DEVELOPMENT_PANEL_FILE="$CONTRACT_DIR/dev_20_length_proportional_seed0.txt"
export SHUTDOWN_ON_EXIT=1 HARD_STOP_MINUTES=480
[[ -x "$REPO/cloud/huawei/run_contracted_expanded_data_gate.sh" ]] || {
  printf 'expanded-data handoff runner is not executable\n' >&2
  exit 2
}
# Non-interactive Bash otherwise exits immediately on an exec(2) failure and
# can bypass the commands below.  execfail guarantees the still-installed EXIT
# trap receives the failure whenever process replacement did not occur.
shopt -s execfail
exec "$REPO/cloud/huawei/run_contracted_expanded_data_gate.sh"
handoff_code=$?
# Bash may clear EXIT traps while attempting exec even when execfail returns
# control.  Restore it before reporting/exiting the failed handoff path.
trap shutdown_on_exit EXIT
printf 'exec handoff failed with exit=%s\n' "$handoff_code" >&2
exit "$handoff_code"
