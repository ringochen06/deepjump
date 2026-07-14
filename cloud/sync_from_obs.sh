#!/usr/bin/env bash
# On the GPU instance: pull the already-staged dataset from OBS to the local EVS SSD.
# This is an INTRA-REGION transfer (OBS -> ECS on the same region's backbone): fast and
# effectively free, and it does NOT touch HuggingFace, so no GPU-hours are burned on a slow
# public download. Run this right after the instance boots, before training.
#
#   BUCKET=obs://my-mdcath bash cloud/sync_from_obs.sh
#
# Prereq: obsutil configured on the instance (obsutil config -i=AK -k=SK -e=<endpoint>).
# Validate access first:  BUCKET=obs://my-mdcath bash cloud/obs_roundtrip_test.sh
set -euo pipefail

BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
ROOT=${ROOT:-/data/mdcath}              # local EVS mount (dataloader hot store)
OBS_PREFIX=${OBS_PREFIX:-mdcath}

command -v obsutil >/dev/null || { echo "!! obsutil not found -- install & 'obsutil config' first"; exit 1; }
mkdir -p "$ROOT"

echo ">> sync $BUCKET/$OBS_PREFIX -> $ROOT  (intra-region, no HuggingFace)"
obsutil sync "$BUCKET/$OBS_PREFIX" "$ROOT"

# The manifest was built during staging; rebuild only if it didn't come across.
if [ ! -f "$ROOT/manifest.json" ]; then
  echo ">> manifest missing -- building on the instance"
  python scripts/build_manifest.py --root "$ROOT" --out "$ROOT/manifest.json"
fi

nfiles=$(find "$ROOT" -name 'mdcath_dataset_*.h5' | wc -l | tr -d ' ')
echo ">> ready. $nfiles h5 files on disk."
echo "   Point config data.root=$ROOT data.manifest=$ROOT/manifest.json"
