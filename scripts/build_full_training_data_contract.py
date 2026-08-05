#!/usr/bin/env python
"""Build a self-contained, atomically published full-training data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from deepjump.data_contract import (
    CONTRACT_SCHEMA,
    ContractExpectations,
    _read_regular_bytes,
    verify_full_training_data_contract,
)


ARTIFACT_NAMES = {
    "data_audit": "full_mdcath_audit.json",
    "manifest": "full_mdcath_manifest.json",
    "official_list": "mdCATH_domains.txt",
    "panel_registry": "evaluation_exclusion_registry.json",
    "partition_audit": "expanded_data_partition.json",
    "source_inventory": "mdcath_source_inventory.jsonl",
    "staging_metadata": "full_mdcath_staging.metadata.json",
    "train_list": "train_eligible_5218.txt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_exact(source: Path, destination: Path, expected_sha256: str, label: str) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    actual = _sha256(source)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected_sha256}")
    shutil.copyfile(source, destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError(f"{label} copy SHA256 mismatch")
    return actual


def build_bundle(
    output_dir: str | Path,
    *,
    data_audit: str | Path,
    data_audit_sha256: str,
    manifest: str | Path,
    manifest_sha256: str,
    official_list: str | Path,
    official_list_sha256: str,
    panel_registry: str | Path,
    partition_audit: str | Path,
    partition_audit_sha256: str,
    source_inventory: str | Path,
    source_inventory_sha256: str,
    staging_metadata: str | Path,
    staging_metadata_sha256: str,
    train_list: str | Path,
    train_list_sha256: str,
    expectations: ContractExpectations | None = None,
) -> dict:
    expectations = expectations or ContractExpectations()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        sources = {
            "data_audit": (Path(data_audit).expanduser(), data_audit_sha256),
            "manifest": (Path(manifest).expanduser(), manifest_sha256),
            "official_list": (
                Path(official_list).expanduser(), official_list_sha256
            ),
            "panel_registry": (
                Path(panel_registry).expanduser(), expectations.panel_registry_sha256
            ),
            "partition_audit": (
                Path(partition_audit).expanduser(), partition_audit_sha256
            ),
            "source_inventory": (
                Path(source_inventory).expanduser(), source_inventory_sha256
            ),
            "staging_metadata": (
                Path(staging_metadata).expanduser(), staging_metadata_sha256
            ),
            "train_list": (Path(train_list).expanduser(), train_list_sha256),
        }
        artifact_rows = {}
        for name, (source, expected_sha256) in sources.items():
            destination = temporary / ARTIFACT_NAMES[name]
            actual_sha256 = _copy_exact(source, destination, expected_sha256, name)
            artifact_rows[name] = {
                "path": destination.name,
                "sha256": actual_sha256,
            }

        registry_source = sources["panel_registry"][0]
        copied_registry = temporary / ARTIFACT_NAMES["panel_registry"]
        registry_raw = _read_regular_bytes(copied_registry, "copied panel registry")
        if hashlib.sha256(registry_raw).hexdigest() != expectations.panel_registry_sha256:
            raise ValueError("copied panel registry changed before semantic parsing")
        registry = json.loads(registry_raw.decode("utf-8"))
        for row in registry.get("panels", []):
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("panel registry path must be relative without parent traversal")
            _copy_exact(
                registry_source.parent / relative,
                temporary / relative,
                row["sha256"],
                f"panel {row['name']}",
            )

        contract_path = temporary / "full_training_data_contract.json"
        contract = {
            "schema": CONTRACT_SCHEMA,
            "source_revision": expectations.source_revision,
            "official_domain_list_sha256": expectations.official_sha256,
            "artifacts": artifact_rows,
        }
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        contract_sha256 = _sha256(contract_path)
        audit_payload = json.loads((temporary / ARTIFACT_NAMES["data_audit"]).read_text())
        verification = verify_full_training_data_contract(
            contract_path,
            contract_sha256,
            configured_root=audit_payload["root"],
            configured_manifest=temporary / ARTIFACT_NAMES["manifest"],
            configured_domains_file=temporary / ARTIFACT_NAMES["train_list"],
            expectations=expectations,
        )
        (temporary / "full_training_data_contract.sha256").write_text(
            f"{contract_sha256}  full_training_data_contract.json\n"
        )
        for path in temporary.iterdir():
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        final_contract = output / contract_path.name
        final_verification = verify_full_training_data_contract(
            final_contract,
            contract_sha256,
            configured_root=audit_payload["root"],
            configured_manifest=output / ARTIFACT_NAMES["manifest"],
            configured_domains_file=output / ARTIFACT_NAMES["train_list"],
            expectations=expectations,
        )
        if final_verification != verification:
            raise RuntimeError("post-publish contract verification changed")
        return {"bundle": str(output), **final_verification}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    for name in (
        "data-audit",
        "manifest",
        "official-list",
        "partition-audit",
        "source-inventory",
        "staging-metadata",
        "train-list",
    ):
        parser.add_argument(f"--{name}", required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--panel-registry", required=True)
    args = parser.parse_args()
    report = build_bundle(
        args.output_dir,
        data_audit=args.data_audit,
        data_audit_sha256=args.data_audit_sha256,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        official_list=args.official_list,
        official_list_sha256=args.official_list_sha256,
        panel_registry=args.panel_registry,
        partition_audit=args.partition_audit,
        partition_audit_sha256=args.partition_audit_sha256,
        source_inventory=args.source_inventory,
        source_inventory_sha256=args.source_inventory_sha256,
        staging_metadata=args.staging_metadata,
        staging_metadata_sha256=args.staging_metadata_sha256,
        train_list=args.train_list,
        train_list_sha256=args.train_list_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
