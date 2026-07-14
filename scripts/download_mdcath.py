#!/usr/bin/env python
"""Download an mdCATH subset from HuggingFace.

mdCATH is public (compsciencelab/mdCATH). Each protein domain is one HDF5 file
under data/mdcath_dataset_<id>.h5, ~0.2-1.6 GB each.

Two modes:
  * --n / --max-gb : query the HF listing once to pick the N smallest domains.
  * --domains / --domains-file : download EXACTLY these ids. This path builds the
    repo path directly from each id (data/mdcath_dataset_<id>.h5) and does NOT call
    the HF tree API -- so staging a frozen 1000-id list needs no listing round-trip,
    supports a mirror (HF_ENDPOINT), retries with resume, and writes a failure list.

Usage:
    python scripts/download_mdcath.py --n 5 --max-gb 0.6
    python scripts/download_mdcath.py --domains 1a02F00 1a0aA00
    python scripts/download_mdcath.py --domains-file configs/subset_1000.txt
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_mdcath.py --domains-file configs/subset_1000.txt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "compsciencelab/mdCATH"
REPO_TYPE = "dataset"


def _domain_from_path(path: str) -> str:
    # data/mdcath_dataset_1a02F00.h5 -> 1a02F00
    return Path(path).stem.replace("mdcath_dataset_", "")


def _repo_path(domain: str) -> str:
    # 1a02F00 -> data/mdcath_dataset_1a02F00.h5 (no listing needed)
    return f"data/mdcath_dataset_{domain}.h5"


def list_data_files() -> list[tuple[str, int]]:
    """Return [(path_in_repo, size_bytes), ...] for every data/*.h5, sorted by size.

    Only used by the --n discovery mode (needs sizes to pick the smallest N).
    """
    from huggingface_hub import HfApi

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


def _download_one(repo_path: str, root: Path, retries: int) -> bool:
    """Download one file with resume + bounded retries. Returns True on success."""
    # huggingface_hub >=1.0 resumes partial downloads by default (and dropped the
    # resume_download kwarg), so a retry after a network drop continues, not restarts.
    for attempt in range(1, retries + 1):
        try:
            hf_hub_download(REPO_ID, repo_path, repo_type=REPO_TYPE, local_dir=str(root))
            return True
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            wait = min(30, 2 ** attempt)
            print(f"    attempt {attempt}/{retries} failed: {type(e).__name__}: {e}"
                  f"{'' if attempt == retries else f' -- retrying in {wait}s'}")
            if attempt < retries:
                time.sleep(wait)
    return False


def _download_domains(domains: list[str], root: Path, retries: int) -> None:
    """Direct-path download of an explicit id list (no HF tree API)."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co (default)")
    print(f"Downloading {len(domains)} domains -> {root}  (endpoint: {endpoint})")
    failures: list[str] = []
    for i, dom in enumerate(domains, 1):
        rp = _repo_path(dom)
        print(f"  [{i}/{len(domains)}] {dom}  {rp}")
        if not _download_one(rp, root, retries):
            failures.append(dom)
            print(f"    !! giving up on {dom}")
    if failures:
        fpath = root / "download_failures.txt"
        fpath.write_text("\n".join(failures) + "\n")
        print(f"done with {len(failures)} FAILURE(S) -> {fpath}")
        print("   re-run to retry:  python scripts/download_mdcath.py "
              f"--root {root} --domains-file {fpath}")
        raise SystemExit(1)
    print("done. all domains fetched.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/hkucds/data/mdcath", help="download destination")
    ap.add_argument("--n", type=int, default=5, help="number of smallest domains to fetch")
    ap.add_argument("--max-gb", type=float, default=0.7, help="skip files larger than this")
    ap.add_argument(
        "--domains", nargs="*", default=None, help="explicit domain ids (overrides --n)"
    )
    ap.add_argument(
        "--domains-file", default=None,
        help="file with one domain id per line (e.g. configs/subset_1000.txt); overrides --n",
    )
    ap.add_argument("--retries", type=int, default=4, help="per-file download attempts (resume)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    domains = list(args.domains) if args.domains else []
    if args.domains_file:
        ids = [ln.strip() for ln in Path(args.domains_file).read_text().splitlines() if ln.strip()]
        domains += ids

    # Always grab the small domain-list metadata (best effort; mirror may not host it).
    try:
        hf_hub_download(REPO_ID, "mdCATH_domains.txt", repo_type=REPO_TYPE, local_dir=str(root))
    except Exception as e:  # noqa: BLE001 - metadata is optional
        print(f"  [warn] could not fetch mdCATH_domains.txt: {e}")

    if domains:
        # Explicit ids: build paths directly, NO HF tree API.
        _download_domains(domains, root, args.retries)
        return

    # Discovery mode: query the listing, pick the N smallest under --max-gb.
    all_files = list_data_files()
    max_bytes = args.max_gb * 1e9
    chosen = [(p, s) for p, s in all_files if s <= max_bytes][: args.n]
    total_gb = sum(s for _, s in chosen) / 1e9
    print(f"Selected {len(chosen)} domains, {total_gb:.2f} GB total -> {root}")
    failures: list[str] = []
    for path, size in chosen:
        dom = _domain_from_path(path)
        print(f"  {dom:>10}  {size/1e6:8.1f} MB  downloading...")
        if not _download_one(path, root, args.retries):
            failures.append(dom)
    if failures:
        (root / "download_failures.txt").write_text("\n".join(failures) + "\n")
        raise SystemExit(f"done with {len(failures)} failure(s) -> {root}/download_failures.txt")
    print("done.")


if __name__ == "__main__":
    main()
