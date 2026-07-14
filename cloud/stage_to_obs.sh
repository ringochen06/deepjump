#!/usr/bin/env bash
# STAGE data into OBS from a CHEAP box (a small CPU ECS in the SAME region as the future
# GPU instance, or your laptop). Goal: never spend 8x-V100 GPU-hours pulling data over the
# public internet. Do the slow HuggingFace download once, here, then the GPU instance only
# does an intra-region `obsutil sync` (see cloud/sync_from_obs.sh) which is fast and cheap.
#
#   # one-time: pick the exact subset (writes configs/subset_1000.txt), then stage it
#   python scripts/select_subset.py --n 1000 --max-gb 0.7 --out configs/subset_1000.txt
#   BUCKET=obs://my-mdcath SUBSET=configs/subset_1000.txt bash cloud/stage_to_obs.sh
#
#   # or stage the FULL dataset (~3.6 TB) instead of a subset
#   BUCKET=obs://my-mdcath MODE=full bash cloud/stage_to_obs.sh
#
# Prereqs on this box: obsutil configured with your AK/SK (obsutil config -i=AK -k=SK -e=<endpoint>),
# and a python env with huggingface_hub + h5py (pip install huggingface_hub tqdm h5py pyyaml).
# Behind the GFW, set a mirror:  export HF_ENDPOINT=https://hf-mirror.com
# Validate OBS access first:  BUCKET=obs://my-mdcath bash cloud/obs_roundtrip_test.sh
set -euo pipefail

BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}
ROOT=${ROOT:-/data/mdcath}              # local staging dir on THIS box
MODE=${MODE:-subset}                    # subset | full
SUBSET=${SUBSET:-configs/subset_1000.txt}
OBS_PREFIX=${OBS_PREFIX:-mdcath}        # object key prefix inside the bucket

command -v obsutil >/dev/null || { echo "!! obsutil not found -- install & 'obsutil config' first"; exit 1; }
mkdir -p "$ROOT"

echo ">> [1/3] download to local staging dir: $ROOT  (MODE=$MODE)"
if [ "$MODE" = "full" ]; then
  python - "$ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("compsciencelab/mdCATH", repo_type="dataset", local_dir=sys.argv[1],
                  allow_patterns=["data/*.h5", "mdCATH_domains.txt"], max_workers=16)
PY
else
  [ -f "$SUBSET" ] || { echo "!! $SUBSET missing -- run scripts/select_subset.py first"; exit 1; }
  echo "   staging $(grep -c . "$SUBSET") domains from $SUBSET"
  python scripts/download_mdcath.py --root "$ROOT" --domains-file "$SUBSET"
fi

echo ">> [2/3] build manifest locally (so the GPU box doesn't have to)"
python scripts/build_manifest.py --root "$ROOT" --out "$ROOT/manifest.json"

nfiles=$(find "$ROOT" -name 'mdcath_dataset_*.h5' | wc -l | tr -d ' ')
echo ">> [3/3] upload to OBS: $BUCKET/$OBS_PREFIX  ($nfiles h5 files + manifest)"
# `obsutil sync` = incremental one-way sync of source dir INTO the destination prefix.
# Exclude huggingface_hub's local-dir cache (.cache/huggingface/*): it is download-resume
# metadata, useless in OBS, and would otherwise add thousands of tiny junk objects.
obsutil sync "$ROOT" "$BUCKET/$OBS_PREFIX" -exclude="*.cache/*"

echo ">> done. Data + manifest are in $BUCKET/$OBS_PREFIX."
echo "   On the GPU instance:  BUCKET=$BUCKET bash cloud/sync_from_obs.sh"
