#!/usr/bin/env python
"""Freeze a reproducible mdCATH domain subset to a text file (one domain id per line).

The training data subset must be *identical* on the staging machine (which uploads to
OBS) and on the GPU instance (which builds the manifest). Freezing the exact ids to a file
removes any ambiguity (tie-ordering, listing changes) and lets you diff/audit the subset.

Two strategies:

  * ``smallest`` (legacy) -- the N smallest *files* under --max-gb. Cheap to stage, but
    file size tracks residue count, so this yields a **tiny-protein** subset: the frozen
    configs/subset_1000.txt spans only 50-91 residues (mean 62). ``crop_length: 256`` never
    binds on it, and it is NOT representative of mdCATH. Kept for reproducing that pilot.

  * ``length-proportional`` -- stratified sample over **numResidues read from the official
    mdcath_source.h5**, with per-band quotas proportional to the full 5398-domain population
    (largest-remainder rounded to sum to exactly 1000). This spans the real length range and
    actually exercises crop 256.

    python scripts/select_subset.py --strategy length-proportional \
        --seed 20260715 --out configs/subset_1000_length_proportional.txt

Then stage exactly that list (see cloud/huawei/stage_to_obs.sh):
    python scripts/download_mdcath.py --root /data/mdcath --domains-file <list>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "compsciencelab/mdCATH"
REPO_TYPE = "dataset"
SOURCE_H5 = "mdcath_source.h5"  # official per-domain metadata (numResidues), ~186 MB

# (label, lo, hi, quota). Quotas are proportional to the 5398-domain population in each
# band (11.3/26.1/20.7/14.4/9.0/5.7/3.8/8.9%), largest-remainder rounded to sum to 1000.
LENGTH_BANDS: list[tuple[str, int, int, int]] = [
    ("<=64", 0, 64, 113),
    ("65-96", 65, 96, 261),
    ("97-128", 97, 128, 207),
    ("129-160", 129, 160, 144),
    ("161-192", 161, 192, 90),
    ("193-224", 193, 224, 58),
    ("225-256", 225, 256, 38),
    (">256", 257, 10**9, 89),
]


def _domain(path: str) -> str:
    return Path(path).stem.replace("mdcath_dataset_", "")


def list_data_files() -> list[tuple[str, int]]:
    """[(path_in_repo, size_bytes), ...] for every data/*.h5, sorted by size."""
    api = HfApi()
    entries = api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo="data", recursive=False)
    files = [(e.path, int(e.size)) for e in entries if e.path.endswith(".h5") and getattr(e, "size", None)]
    files.sort(key=lambda x: x[1])
    return files


def residues_from_source() -> dict[str, int]:
    """{domain_id: numResidues} straight from the official mdcath_source.h5."""
    import h5py

    path = hf_hub_download(REPO_ID, SOURCE_H5, repo_type=REPO_TYPE)
    with h5py.File(path, "r") as f:
        return {k: int(f[k].attrs["numResidues"]) for k in f.keys()}


def pick_smallest(n: int, max_gb: float) -> list[str]:
    files = list_data_files()
    chosen = [(p, s) for p, s in files if s <= max_gb * 1e9][:n]
    return [_domain(p) for p, _ in chosen]


def pick_length_proportional(seed: int) -> list[str]:
    """Stratified-by-length sample. One RNG spans all bands, pools are sorted, and the
    final list is sorted -- all three are load-bearing for reproducing the same ids."""
    res = residues_from_source()
    rng = random.Random(seed)
    chosen: list[str] = []
    for label, lo, hi, quota in LENGTH_BANDS:
        pool = sorted(d for d, v in res.items() if lo <= v <= hi)
        if len(pool) < quota:
            raise SystemExit(f"band {label}: need {quota} domains, only {len(pool)} available")
        chosen += rng.sample(pool, quota)
        print(f"  {label:>8}: {quota:>4} / {len(pool):>4} in band")
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=("smallest", "length-proportional"), default="smallest")
    ap.add_argument("--n", type=int, default=1000, help="smallest strategy: number of domains")
    ap.add_argument("--max-gb", type=float, default=0.7, help="smallest strategy: per-file cap")
    ap.add_argument("--seed", type=int, default=20260715, help="length-proportional: sampling seed")
    ap.add_argument("--out", default="configs/subset_1000.txt", help="output id list")
    args = ap.parse_args()

    if args.strategy == "length-proportional":
        domains = pick_length_proportional(args.seed)
    else:
        domains = pick_smallest(args.n, args.max_gb)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(domains) + "\n")

    # Report total staging size (needs the size listing; best-effort).
    try:
        size = {_domain(p): s for p, s in list_data_files()}
        total = sum(size[d] for d in domains)
        print(f"wrote {out}: {len(domains)} domains ({len(set(domains))} unique), "
              f"{total:,} bytes ({total/1e9:.1f} GB)")
    except Exception:
        print(f"wrote {out}: {len(domains)} domains ({len(set(domains))} unique)")


if __name__ == "__main__":
    main()
