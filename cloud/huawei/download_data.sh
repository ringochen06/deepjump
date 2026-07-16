#!/usr/bin/env bash
# Fetch mdCATH to local NVMe (/data/mdcath) and build the manifest.
#
#   MODE=subset N=1000 ./cloud/huawei/download_data.sh     # 1000 smallest domains (validation run)
#   MODE=full            ./cloud/huawei/download_data.sh     # full ~5398 domains (~2-3 TB)
#
# On Huawei Cloud, prefer: keep the raw dataset in OBS (cheap, durable) and sync the
# working subset to the instance's local NVMe SSD (fast random reads for the dataloader).
# HF download here writes straight to $ROOT; swap in `obsutil sync obs://... $ROOT` if the
# data already lives in your OBS bucket.
set -euo pipefail

ROOT=${ROOT:-/data/mdcath}
MODE=${MODE:-subset}
N=${N:-1000}
MAXGB=${MAXGB:-0.7}
mkdir -p "$ROOT"

if [ "$MODE" = "full" ]; then
  echo ">> full mdCATH snapshot -> $ROOT (this is large; ensure the disk/OBS has 2-3 TB)"
  python - "$ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
root = sys.argv[1]
snapshot_download("compsciencelab/mdCATH", repo_type="dataset", local_dir=root,
                  allow_patterns=["data/*.h5", "mdCATH_domains.txt"], max_workers=16)
print("snapshot done ->", root)
PY
else
  echo ">> subset: $N smallest domains (<= ${MAXGB} GB each) -> $ROOT"
  python scripts/download_mdcath.py --root "$ROOT" --n "$N" --max-gb "$MAXGB"
fi

echo ">> building manifest"
python scripts/build_manifest.py --root "$ROOT" --out "$ROOT/manifest.json"
echo ">> done. Point config data.root=$ROOT and data.manifest=$ROOT/manifest.json"
