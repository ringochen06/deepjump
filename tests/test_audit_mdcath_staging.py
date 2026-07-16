from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _make_h5(path: Path, domain: str) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group(domain)
        group.attrs["numResidues"] = 2
        replica = group.create_group("320").create_group("0")
        replica.create_dataset("coords", data=np.zeros((2, 3, 3), dtype=np.float32))


def test_audit_accepts_consistent_staging(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    domains = ["1abcA00", "2defB00"]
    for domain in domains:
        _make_h5(data / f"mdcath_dataset_{domain}.h5", domain)

    subset = tmp_path / "subset.txt"
    subset.write_text("".join(f"{domain}\n" for domain in domains))
    subset_sha = hashlib.sha256(subset.read_bytes()).hexdigest()
    h5_files = sorted(data.glob("*.h5"))
    total_bytes = sum(path.stat().st_size for path in h5_files)
    manifest = [
        {
            "file": path.name,
            "domain": domain,
            "trajectories": [{"temp": 320, "replica": 0, "num_frames": 2}],
        }
        for path, domain in zip(h5_files, domains, strict=True)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "staging_metadata.json").write_text(
        json.dumps(
            {
                "subset_sha256": subset_sha,
                "subset_domains": 2,
                "selection_strategy": "test",
                "selection_seed": 7,
                "generating_commit": "abc1234",
                "h5_files_staged": 2,
                "total_bytes": total_bytes,
                "manifest_domains": 2,
            }
        )
    )

    command = [
        sys.executable,
        "scripts/audit_mdcath_staging.py",
        "--root",
        str(tmp_path),
        "--expected-h5",
        "2",
        "--expected-bytes",
        str(total_bytes),
        "--expected-subset-sha256",
        subset_sha,
        "--expected-trajectories",
        "2",
        "--expected-strategy",
        "test",
        "--expected-seed",
        "7",
        "--expected-commit",
        "abc1234",
        "--samples",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "PASS"' in result.stdout

    (tmp_path / "download_failures.txt").write_text("2defB00\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "unresolved download failures: 1" in result.stderr
