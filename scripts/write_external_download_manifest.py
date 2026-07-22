#!/usr/bin/env python
"""Write an exact, hash-bound manifest for a frozen external HDF5 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepjump.evaluation import load_frozen_domain_ids


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    *,
    root: Path,
    domain_list: Path,
    domain_list_sha256: str,
    expected_bytes: int,
    audit_path: Path,
    claim_path: Path,
    claim_sha256: str,
    run_id: str,
    commit: str,
) -> dict:
    domain_ids, panel_sha = load_frozen_domain_ids(
        domain_list, domain_list_sha256
    )
    if len(domain_ids) != 20 or len(set(domain_ids)) != 20:
        raise ValueError("external manifest requires exactly 20 unique domains")
    if _sha(claim_path) != claim_sha256:
        raise ValueError("external claim SHA256 mismatch")
    claim = json.loads(claim_path.read_text())
    if claim.get("status") != "CLAIMED_FOR_SINGLE_USE" or claim.get(
        "panel_sha256"
    ) != panel_sha:
        raise ValueError("external claim identity mismatch")
    audit = json.loads(audit_path.read_text())
    expected_audit = {
        "status": "PASS",
        "domain_list_sha256": panel_sha,
        "h5_files": 20,
        "total_bytes": expected_bytes,
        "trajectories": 500,
        "unresolved_failures": 0,
    }
    for key, value in expected_audit.items():
        if audit.get(key) != value:
            raise ValueError(f"external audit mismatch: {key}")
    rows = audit.get("domains")
    if not isinstance(rows, list) or [row.get("domain") for row in rows] != domain_ids:
        raise ValueError("external audit domain order mismatch")
    root = root.resolve()
    expected_paths = [root / "data" / f"mdcath_dataset_{domain}.h5" for domain in domain_ids]
    discovered = sorted(path.resolve() for path in root.rglob("*.h5"))
    if discovered != sorted(expected_paths):
        raise ValueError("external HDF5 exact inventory mismatch")
    files = []
    for domain, path, audit_row in zip(domain_ids, expected_paths, rows, strict=True):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"external HDF5 is missing, non-file, or symlink: {domain}")
        relative = path.relative_to(root).as_posix()
        if relative != f"data/mdcath_dataset_{domain}.h5":
            raise ValueError(f"external HDF5 relative path mismatch: {domain}")
        size = path.stat().st_size
        if audit_row.get("file") != path.name or audit_row.get("bytes") != size:
            raise ValueError(f"external audit file metadata mismatch: {domain}")
        files.append({
            "domain": domain,
            "relative_path": relative,
            "bytes": size,
            "sha256": _sha(path),
            "residues": audit_row.get("residues"),
            "trajectories": audit_row.get("trajectories"),
            "min_frames": audit_row.get("min_frames"),
        })
    total_bytes = sum(row["bytes"] for row in files)
    if total_bytes != expected_bytes:
        raise ValueError("external manifest total bytes mismatch")
    inventory_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "deepjump.external_download_inventory.v1",
        "status": "PASS",
        "panel_sha256": panel_sha,
        "claim_sha256": claim_sha256,
        "run_id": run_id,
        "commit": commit,
        "root": str(root),
        "files_count": len(files),
        "total_bytes": total_bytes,
        "trajectories": sum(int(row["trajectories"]) for row in files),
        "unresolved_failures": 0,
        "inventory_sha256": inventory_sha,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--domain-list", required=True, type=Path)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_manifest(
        root=args.root,
        domain_list=args.domain_list,
        domain_list_sha256=args.domain_list_sha256,
        expected_bytes=args.expected_bytes,
        audit_path=args.audit,
        claim_path=args.claim,
        claim_sha256=args.claim_sha256,
        run_id=args.run_id,
        commit=args.commit,
    )
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
