#!/usr/bin/env python
"""Strict audit for a frozen external mdCATH panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deepjump.data import discover_domains
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import (
    MDCATH_REPLICAS,
    MDCATH_TEMPERATURES,
    load_frozen_domain_ids,
    resolve_frozen_domains,
)


def _available_replica_frames(
    available: list[tuple[int, int, int]], expected_temperature: int
) -> list[tuple[int, int]]:
    """Extract ``(replica, frames)`` from validated availability rows."""
    rows = []
    for temperature, replica, frames in available:
        if int(temperature) != int(expected_temperature):
            raise ValueError("availability temperature does not match the requested grid")
        rows.append((int(replica), int(frames)))
    return rows


def audit(root: str, domain_list: str, domain_list_sha256: str, expected_bytes: int) -> dict:
    domain_ids, actual_sha256 = load_frozen_domain_ids(domain_list, domain_list_sha256)
    if len(domain_ids) != 20 or len(set(domain_ids)) != 20:
        raise ValueError("external data audit requires exactly 20 unique domains")
    paths = resolve_frozen_domains(discover_domains(root), domain_ids)
    actual_names = {path.stem.replace("mdcath_dataset_", "") for path in paths}
    if actual_names != set(domain_ids):
        raise ValueError("external data filenames do not match the frozen panel")
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes != expected_bytes:
        raise ValueError(f"external data bytes {total_bytes} != {expected_bytes}")
    failure_files = sorted(Path(root).rglob("download_failures.txt"))
    unresolved = sum(
        len([line for line in path.read_text().splitlines() if line.strip()])
        for path in failure_files
    )
    if unresolved:
        raise ValueError(f"external data has {unresolved} unresolved download failures")

    domains = []
    trajectories = 0
    for domain_id, path in zip(domain_ids, paths):
        handle = _DomainHandle(path)
        try:
            if handle.name != domain_id:
                raise ValueError(f"HDF5 domain key {handle.name} != {domain_id}")
            domain_trajectories = 0
            min_frames = None
            for temperature in MDCATH_TEMPERATURES:
                available = handle.replicas(temperature, MDCATH_REPLICAS)
                replica_frames = _available_replica_frames(available, temperature)
                if [replica for replica, _ in replica_frames] != list(MDCATH_REPLICAS):
                    raise ValueError(f"{domain_id}/{temperature} replica grid is incomplete")
                for replica, frames in replica_frames:
                    if int(frames) < 3:
                        raise ValueError(f"{domain_id}/{temperature}/{replica} has fewer than 3 frames")
                    first = np.asarray(handle.coords(temperature, replica, 0))
                    last = np.asarray(handle.coords(temperature, replica, int(frames) - 1))
                    if not np.isfinite(first).all() or not np.isfinite(last).all():
                        raise ValueError(f"{domain_id}/{temperature}/{replica} has non-finite coordinates")
                    min_frames = int(frames) if min_frames is None else min(min_frames, int(frames))
                    domain_trajectories += 1
            if domain_trajectories != 25:
                raise ValueError(f"{domain_id} trajectory count {domain_trajectories} != 25")
            trajectories += domain_trajectories
            domains.append({
                "domain": domain_id,
                "file": path.name,
                "bytes": path.stat().st_size,
                "residues": handle.layout.num_residues,
                "trajectories": domain_trajectories,
                "min_frames": min_frames,
            })
        finally:
            handle.close()
    if trajectories != 500:
        raise ValueError(f"external data trajectories {trajectories} != 500")
    return {
        "status": "PASS",
        "root": str(Path(root).resolve()),
        "domain_list_sha256": actual_sha256,
        "h5_files": len(paths),
        "total_bytes": total_bytes,
        "trajectories": trajectories,
        "unresolved_failures": unresolved,
        "domains": domains,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(
        args.root,
        args.domain_list,
        args.domain_list_sha256,
        args.expected_bytes,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
