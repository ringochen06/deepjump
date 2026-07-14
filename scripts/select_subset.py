#!/usr/bin/env python
"""Freeze a reproducible mdCATH domain subset to a text file (one domain id per line).

The training data subset must be *identical* on the staging machine (which uploads to
OBS) and on the GPU instance (which builds the manifest). Sorting the HF listing by size
at download time is deterministic in principle, but freezing the exact ids to a file
removes any ambiguity (tie-ordering, listing changes) and lets you diff/audit the subset.

    # query HuggingFace once, write the 1000 smallest domains (<= 0.7 GB each)
    python scripts/select_subset.py --n 1000 --max-gb 0.7 --out configs/subset_1000.txt

Then stage exactly that list (see cloud/stage_to_obs.sh):
    python scripts/download_mdcath.py --root /data/mdcath --domains-file configs/subset_1000.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "compsciencelab/mdCATH"
REPO_TYPE = "dataset"


def _domain(path: str) -> str:
    return Path(path).stem.replace("mdcath_dataset_", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of smallest domains")
    ap.add_argument("--max-gb", type=float, default=0.7, help="skip files larger than this")
    ap.add_argument("--out", default="configs/subset_1000.txt", help="output id list")
    args = ap.parse_args()

    api = HfApi()
    entries = api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo="data", recursive=False)
    files = [(e.path, int(e.size)) for e in entries if e.path.endswith(".h5") and getattr(e, "size", None)]
    files.sort(key=lambda x: x[1])

    max_bytes = args.max_gb * 1e9
    chosen = [(p, s) for p, s in files if s <= max_bytes][: args.n]
    total_gb = sum(s for _, s in chosen) / 1e9

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_domain(p) for p, _ in chosen) + "\n")
    print(f"wrote {out}: {len(chosen)} domains, {total_gb:.1f} GB total "
          f"(largest {chosen[-1][1]/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
