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
import hashlib
import json
import math
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


def residues_from_source(source_h5: str | Path | None = None) -> dict[str, int]:
    """{domain_id: numResidues} straight from the official mdcath_source.h5."""
    import h5py

    path = Path(source_h5) if source_h5 else hf_hub_download(
        REPO_ID, SOURCE_H5, repo_type=REPO_TYPE
    )
    with h5py.File(path, "r") as f:
        return {k: int(f[k].attrs["numResidues"]) for k in f.keys()}


def pick_smallest(n: int, max_gb: float) -> list[str]:
    files = list_data_files()
    chosen = [(p, s) for p, s in files if s <= max_gb * 1e9][:n]
    return [_domain(p) for p, _ in chosen]


def length_band_quotas(n: int) -> list[int]:
    """Largest-remainder quotas preserving the frozen 1000-domain proportions."""
    if n < 1:
        raise ValueError("n must be >= 1")
    exact = [quota * n / 1000 for _, _, _, quota in LENGTH_BANDS]
    quotas = [math.floor(value) for value in exact]
    remaining = n - sum(quotas)
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - quotas[i]), i))
    for i in order[:remaining]:
        quotas[i] += 1
    return quotas


def pick_length_proportional(
    seed: int, n: int = 1000, exclude: set[str] | None = None,
    residues: dict[str, int] | None = None,
) -> list[str]:
    """Stratified-by-length sample. One RNG spans all bands, pools are sorted, and the
    final list is sorted -- all three are load-bearing for reproducing the same ids."""
    res = residues if residues is not None else residues_from_source()
    exclude = exclude or set()
    rng = random.Random(seed)
    chosen: list[str] = []
    quotas = length_band_quotas(n)
    for (label, lo, hi, _), quota in zip(LENGTH_BANDS, quotas):
        pool = sorted(d for d, v in res.items() if lo <= v <= hi and d not in exclude)
        if len(pool) < quota:
            raise SystemExit(f"band {label}: need {quota} domains, only {len(pool)} available")
        chosen += rng.sample(pool, quota)
        print(f"  {label:>8}: {quota:>4} / {len(pool):>4} in band")
    return sorted(chosen)


def load_exclusions(paths: list[str] | None) -> tuple[set[str], list[dict[str, object]]]:
    """Load repeated frozen exclusion lists and record their exact identities."""
    excluded: set[str] = set()
    identities: list[dict[str, object]] = []
    for raw_path in paths or []:
        path = Path(raw_path)
        content = path.read_bytes()
        domain_ids = [line.strip() for line in content.decode().splitlines() if line.strip()]
        if len(domain_ids) != len(set(domain_ids)):
            raise ValueError(f"exclusion list contains duplicate domains: {path}")
        identities.append({
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "count": len(domain_ids),
        })
        excluded.update(domain_ids)
    return excluded, identities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=("smallest", "length-proportional"), default="smallest")
    ap.add_argument("--n", type=int, default=1000, help="smallest strategy: number of domains")
    ap.add_argument("--max-gb", type=float, default=0.7, help="smallest strategy: per-file cap")
    ap.add_argument("--seed", type=int, default=20260715, help="length-proportional: sampling seed")
    ap.add_argument("--out", default="configs/subset_1000.txt", help="output id list")
    ap.add_argument(
        "--exclude-domains-file",
        action="append",
        dest="exclude_domains_files",
        help="domain ids to exclude before sampling; repeat for multiple frozen lists",
    )
    ap.add_argument("--metadata-out", help="write frozen selection metadata JSON")
    ap.add_argument("--source-h5", help="local official mdcath_source.h5 (avoids network)")
    ap.add_argument("--skip-size-report", action="store_true")
    args = ap.parse_args()

    if args.strategy == "length-proportional":
        exclude, exclusion_identities = load_exclusions(args.exclude_domains_files)
        residues = residues_from_source(args.source_h5) if args.source_h5 else None
        domains = pick_length_proportional(args.seed, args.n, exclude, residues)
    else:
        domains = pick_smallest(args.n, args.max_gb)
        exclude = set()
        exclusion_identities = []

    expected_h5_bytes: int | None = None
    if not args.skip_size_report:
        try:
            size = {_domain(path): bytes_ for path, bytes_ in list_data_files()}
            expected_h5_bytes = sum(size[domain_id] for domain_id in domains)
        except Exception:
            expected_h5_bytes = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(domains) + "\n")
    if args.metadata_out:
        content = out.read_bytes()
        metadata = {
            "strategy": args.strategy,
            "seed": args.seed,
            "n": len(domains),
            "subset_sha256": hashlib.sha256(content).hexdigest(),
            "exclude_domains_file": (
                args.exclude_domains_files[0]
                if args.exclude_domains_files and len(args.exclude_domains_files) == 1
                else None
            ),
            "exclude_domains_files": exclusion_identities,
            "exclude_count": len(exclude),
            "length_band_quotas": length_band_quotas(args.n)
            if args.strategy == "length-proportional" else None,
            "source_h5_sha256": (
                hashlib.sha256(Path(args.source_h5).read_bytes()).hexdigest()
                if args.source_h5 else None
            ),
            "expected_h5_bytes": expected_h5_bytes,
        }
        metadata_out = Path(args.metadata_out)
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")

    # Report total staging size (needs the size listing; best-effort).
    try:
        if args.skip_size_report:
            raise RuntimeError("size report disabled")
        if expected_h5_bytes is None:
            raise RuntimeError("size report unavailable")
        total = expected_h5_bytes
        print(f"wrote {out}: {len(domains)} domains ({len(set(domains))} unique), "
              f"{total:,} bytes ({total/1e9:.1f} GB)")
    except Exception:
        print(f"wrote {out}: {len(domains)} domains ({len(set(domains))} unique)")


if __name__ == "__main__":
    main()
