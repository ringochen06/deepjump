#!/usr/bin/env python
"""Audit a staged mdCATH subset before expensive GPU work.

The audit is intentionally read-only. It reconciles the frozen subset, HDF5 files,
manifest, and staging metadata, then reads a few coordinates from deterministic HDF5
samples so an object-count-only sync cannot pass the data gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(count: int, samples: int) -> list[int]:
    if count == 0 or samples <= 0:
        return []
    if samples == 1:
        return [0]
    if samples >= count:
        return list(range(count))
    return sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})


def audit(args: argparse.Namespace) -> list[str]:
    root = Path(args.root).expanduser().resolve()
    errors: list[str] = []

    files = sorted(root.rglob("mdcath_dataset_*.h5"))
    total_bytes = sum(path.stat().st_size for path in files)
    if len(files) != args.expected_h5:
        errors.append(f"H5 count {len(files)} != {args.expected_h5}")
    if total_bytes != args.expected_bytes:
        errors.append(f"H5 bytes {total_bytes} != {args.expected_bytes}")

    failure_files = sorted(root.rglob("download_failures.txt"))
    unresolved = sum(
        len([line for line in path.read_text().splitlines() if line.strip()])
        for path in failure_files
    )
    if unresolved:
        errors.append(f"unresolved download failures: {unresolved}")

    subset_path = root / "subset.txt"
    manifest_path = root / "manifest.json"
    metadata_path = root / "staging_metadata.json"
    for path in (subset_path, manifest_path, metadata_path):
        if not path.is_file():
            errors.append(f"missing required artifact: {path.name}")
    if errors:
        return errors

    subset_ids = [line.strip() for line in subset_path.read_text().splitlines() if line.strip()]
    subset_sha256 = _sha256(subset_path)
    if len(subset_ids) != args.expected_h5:
        errors.append(f"subset domains {len(subset_ids)} != {args.expected_h5}")
    if subset_sha256 != args.expected_subset_sha256:
        errors.append(
            f"subset SHA256 {subset_sha256} != {args.expected_subset_sha256}"
        )

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list):
        errors.append("manifest root is not a list")
        return errors
    if len(manifest) != args.expected_h5:
        errors.append(f"manifest domains {len(manifest)} != {args.expected_h5}")
    trajectories = sum(len(entry.get("trajectories", [])) for entry in manifest)
    if args.expected_trajectories is not None and trajectories != args.expected_trajectories:
        errors.append(
            f"manifest trajectories {trajectories} != {args.expected_trajectories}"
        )
    actual_names = {path.name for path in files}
    manifest_names = {Path(entry["file"]).name for entry in manifest}
    if actual_names != manifest_names:
        missing = len(manifest_names - actual_names)
        extra = len(actual_names - manifest_names)
        errors.append(f"manifest/H5 filename mismatch: missing={missing} extra={extra}")

    metadata = json.loads(metadata_path.read_text())
    expected_metadata = {
        "subset_sha256": args.expected_subset_sha256,
        "subset_domains": args.expected_h5,
        "h5_files_staged": args.expected_h5,
        "total_bytes": args.expected_bytes,
        "manifest_domains": args.expected_h5,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            errors.append(f"metadata {key}={metadata.get(key)!r} != {expected!r}")
    if args.expected_strategy and metadata.get("selection_strategy") != args.expected_strategy:
        errors.append(
            f"metadata selection_strategy={metadata.get('selection_strategy')!r} "
            f"!= {args.expected_strategy!r}"
        )
    if args.expected_seed is not None and metadata.get("selection_seed") != args.expected_seed:
        errors.append(
            f"metadata selection_seed={metadata.get('selection_seed')!r} "
            f"!= {args.expected_seed!r}"
        )
    if args.expected_commit:
        commit = str(metadata.get("generating_commit", ""))
        if not commit.startswith(args.expected_commit):
            errors.append(
                f"metadata generating_commit={commit!r} does not start with "
                f"{args.expected_commit!r}"
            )

    for index in _sample_indices(len(files), args.samples):
        path = files[index]
        expected_domain = path.stem.replace("mdcath_dataset_", "")
        try:
            with h5py.File(path, "r") as handle:
                domain = next(iter(handle.keys()))
                if domain != expected_domain:
                    raise ValueError(f"domain key {domain!r} != {expected_domain!r}")
                group = handle[domain]
                if int(group.attrs.get("numResidues", 0)) <= 0:
                    raise ValueError("numResidues is missing or non-positive")
                coords = None
                for temperature in ("320", "348", "379", "413", "450"):
                    if temperature not in group:
                        continue
                    for replica in group[temperature].values():
                        if "coords" in replica:
                            coords = replica["coords"]
                            break
                    if coords is not None:
                        break
                if coords is None or coords.shape[0] == 0:
                    raise ValueError("no non-empty coords dataset found")
                if not np.isfinite(np.asarray(coords[0, 0])).all():
                    raise ValueError("first coordinate is non-finite")
        except Exception as exc:  # noqa: BLE001 - report every audit failure uniformly
            errors.append(f"HDF5 readback failed for {path.name}: {exc}")

    print(
        json.dumps(
            {
                "root": str(root),
                "h5_files": len(files),
                "h5_bytes": total_bytes,
                "manifest_domains": len(manifest),
                "manifest_trajectories": trajectories,
                "subset_sha256": subset_sha256,
                "unresolved_failures": unresolved,
                "hdf5_samples": len(_sample_indices(len(files), args.samples)),
                "status": "PASS" if not errors else "FAIL",
            },
            indent=2,
        )
    )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-h5", required=True, type=int)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--expected-subset-sha256", required=True)
    parser.add_argument("--expected-trajectories", type=int)
    parser.add_argument("--expected-strategy")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    errors = audit(args)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
