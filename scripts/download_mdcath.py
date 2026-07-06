#!/usr/bin/env python
"""Download a small mdCATH subset from HuggingFace.

mdCATH is public (compsciencelab/mdCATH). Each protein domain is one HDF5 file
under data/mdcath_dataset_<id>.h5, ~0.3-1.6 GB each. We pick the N smallest
files (below --max-gb) so a laptop can hold a working subset.

Usage:
    python scripts/download_mdcath.py --n 5 --max-gb 0.6
    python scripts/download_mdcath.py --domains 1a02F00 1a0aA00
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "compsciencelab/mdCATH"
REPO_TYPE = "dataset"


def _domain_from_path(path: str) -> str:
    # data/mdcath_dataset_1a02F00.h5 -> 1a02F00
    return Path(path).stem.replace("mdcath_dataset_", "")


def list_data_files() -> list[tuple[str, int]]:
    """Return [(path_in_repo, size_bytes), ...] for every data/*.h5, sorted by size."""
    api = HfApi()
    entries = api.list_repo_tree(
        REPO_ID, repo_type=REPO_TYPE, path_in_repo="data", recursive=False
    )
    files = [
        (e.path, int(e.size))
        for e in entries
        if e.path.endswith(".h5") and getattr(e, "size", None)
    ]
    files.sort(key=lambda x: x[1])
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/hkucds/data/mdcath", help="download destination")
    ap.add_argument("--n", type=int, default=5, help="number of smallest domains to fetch")
    ap.add_argument("--max-gb", type=float, default=0.7, help="skip files larger than this")
    ap.add_argument(
        "--domains", nargs="*", default=None, help="explicit domain ids (overrides --n)"
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    # Always grab the domain list + source index (small, useful for metadata).
    for aux in ("mdCATH_domains.txt",):
        hf_hub_download(REPO_ID, aux, repo_type=REPO_TYPE, local_dir=str(root))

    all_files = list_data_files()
    by_domain = {_domain_from_path(p): (p, s) for p, s in all_files}

    if args.domains:
        chosen = []
        for d in args.domains:
            if d not in by_domain:
                raise SystemExit(f"domain {d!r} not found in mdCATH data/ listing")
            chosen.append(by_domain[d])
    else:
        max_bytes = args.max_gb * 1e9
        chosen = [(p, s) for p, s in all_files if s <= max_bytes][: args.n]

    total_gb = sum(s for _, s in chosen) / 1e9
    print(f"Selected {len(chosen)} domains, {total_gb:.2f} GB total -> {root}")
    for path, size in chosen:
        print(f"  {_domain_from_path(path):>10}  {size/1e6:8.1f} MB  downloading...")
        hf_hub_download(REPO_ID, path, repo_type=REPO_TYPE, local_dir=str(root))
    print("done.")


if __name__ == "__main__":
    main()
