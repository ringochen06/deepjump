#!/usr/bin/env python
"""Write staging_metadata.json: a self-describing record of what was staged into OBS.

Staged data outlives the box that produced it, so the prefix must say for itself which
subset it holds, how that subset was chosen, and which commit chose it -- otherwise two
prefixes of 1000 h5 files are indistinguishable six weeks later.

    python scripts/write_staging_metadata.py --root /data/mdcath --subset configs/subset_1000.txt \
        --seed 20260715 --strategy length-proportional --out /data/mdcath/staging_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import subprocess
from pathlib import Path

BANDS = [("<=64", 0, 64), ("65-96", 65, 96), ("97-128", 97, 128), ("129-160", 129, 160),
         ("161-192", 161, 192), ("193-224", 193, 224), ("225-256", 225, 256), (">256", 257, 10**9)]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="staging dir holding data/*.h5 + manifest.json")
    ap.add_argument("--subset", required=True, help="frozen domain id list that was staged")
    ap.add_argument("--seed", type=int, default=None, help="sampling seed (length-proportional)")
    ap.add_argument("--strategy", default="unknown", help="how the subset was selected")
    ap.add_argument("--out", default=None, help="output path (default <root>/staging_metadata.json)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out) if args.out else root / "staging_metadata.json"

    subset_bytes = Path(args.subset).read_bytes()
    ids = [ln.strip() for ln in subset_bytes.decode().splitlines() if ln.strip()]

    h5 = sorted(root.rglob("mdcath_dataset_*.h5"))
    total_bytes = sum(f.stat().st_size for f in h5)

    manifest_path = root / "manifest.json"
    lengths: list[int] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        lengths = sorted(int(e.get("num_residues", 0)) for e in manifest)

    dist = {}
    if lengths:
        dist = {
            "min": lengths[0], "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(len(lengths) * 0.9)], "max": lengths[-1],
            "mean": round(st.mean(lengths), 1),
            "bands": {name: sum(1 for v in lengths if lo <= v <= hi) for name, lo, hi in BANDS},
        }

    meta = {
        "subset_file": Path(args.subset).name,
        "subset_sha256": hashlib.sha256(subset_bytes).hexdigest(),
        "subset_domains": len(ids),
        "selection_strategy": args.strategy,
        "selection_seed": args.seed,
        "generating_commit": _git_commit(),
        "h5_files_staged": len(h5),
        "total_bytes": total_bytes,
        "total_gb_decimal": round(total_bytes / 1e9, 1),
        "manifest_domains": len(json.loads(manifest_path.read_text())) if manifest_path.exists() else None,
        "residue_length_distribution": dist,
    }
    out.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  strategy={args.strategy} seed={args.seed} commit={meta['generating_commit'][:7]}")
    print(f"  h5={meta['h5_files_staged']} manifest={meta['manifest_domains']} "
          f"bytes={total_bytes:,} sha256={meta['subset_sha256'][:16]}...")


if __name__ == "__main__":
    main()
