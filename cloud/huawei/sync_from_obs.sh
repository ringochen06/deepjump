#!/usr/bin/env bash
# On the GPU instance: pull the already-staged dataset from OBS to the local EVS SSD.
# This is an INTRA-REGION transfer (OBS -> ECS on the same region's backbone): fast and
# effectively free, and it does NOT touch HuggingFace, so no GPU-hours are burned on a slow
# public download. Run this right after the instance boots, before training.
#
#   BUCKET=obs://my-mdcath bash cloud/huawei/sync_from_obs.sh
#
# Prereq: obsutil configured on the instance (obsutil config -i=AK -k=SK -e=<endpoint>).
# Validate access first:  BUCKET=obs://my-mdcath bash cloud/huawei/obs_roundtrip_test.sh
set -euo pipefail

BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
ROOT=${ROOT:-/data/mdcath}              # local EVS mount (dataloader hot store)
OBS_PREFIX=${OBS_PREFIX:-mdcath}

command -v obsutil >/dev/null || { echo "!! obsutil not found -- install & 'obsutil config' first"; exit 1; }
mkdir -p "$ROOT"

echo ">> sync $BUCKET/$OBS_PREFIX -> $ROOT  (intra-region, no HuggingFace)"
# Exclude any stray huggingface cache metadata that may have been staged into OBS.
obsutil sync "$BUCKET/$OBS_PREFIX" "$ROOT" -exclude="*.cache/*"

# The manifest was built during staging; rebuild only if it didn't come across.
if [ ! -f "$ROOT/manifest.json" ]; then
  echo ">> manifest missing -- building on the instance"
  python scripts/build_manifest.py --root "$ROOT" --out "$ROOT/manifest.json"
fi

nfiles=$(find "$ROOT" -name 'mdcath_dataset_*.h5' | wc -l | tr -d ' ')
echo ">> ready. $nfiles h5 files on disk."
echo "   Point config data.root=$ROOT data.manifest=$ROOT/manifest.json"

# Optional strict gate for a frozen subset. The caller supplies independently recorded
# expectations; staging metadata alone is not trusted as the source of truth.
if [ -n "${EXPECTED_H5:-}" ] || [ -n "${EXPECTED_BYTES:-}" ] || [ -n "${EXPECTED_SUBSET_SHA256:-}" ]; then
  : "${EXPECTED_H5:?set EXPECTED_H5 for strict audit}"
  : "${EXPECTED_BYTES:?set EXPECTED_BYTES for strict audit}"
  : "${EXPECTED_SUBSET_SHA256:?set EXPECTED_SUBSET_SHA256 for strict audit}"
  echo ">> strict local data audit"
  AUDIT_ARGS=(
    --root "$ROOT"
    --expected-h5 "$EXPECTED_H5"
    --expected-bytes "$EXPECTED_BYTES"
    --expected-subset-sha256 "$EXPECTED_SUBSET_SHA256"
    --samples "${AUDIT_SAMPLES:-5}"
  )
  [ -n "${EXPECTED_TRAJECTORIES:-}" ] && AUDIT_ARGS+=(--expected-trajectories "$EXPECTED_TRAJECTORIES")
  [ -n "${EXPECTED_STRATEGY:-}" ] && AUDIT_ARGS+=(--expected-strategy "$EXPECTED_STRATEGY")
  [ -n "${EXPECTED_SEED:-}" ] && AUDIT_ARGS+=(--expected-seed "$EXPECTED_SEED")
  [ -n "${EXPECTED_COMMIT:-}" ] && AUDIT_ARGS+=(--expected-commit "$EXPECTED_COMMIT")
  python scripts/audit_mdcath_staging.py "${AUDIT_ARGS[@]}"
fi
