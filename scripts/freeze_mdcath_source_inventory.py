#!/usr/bin/env python
"""Freeze and verify the canonical Hugging Face mdCATH source inventory.

The inventory contains source metadata only. It never downloads the 3.6 TB HDF5
payload. ``fetch`` talks only to the official Hugging Face endpoint at one frozen
commit; ``verify`` is completely offline and fails closed against the known
canonical inventory digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_REPO = "compsciencelab/mdCATH"
SOURCE_REVISION = "5e3ed8aec62b689e01751db16275fdcdbc39e47f"
SOURCE_ENDPOINT = "https://huggingface.co"
SOURCE_DOMAIN_LIST = "mdCATH_domains.txt"
EXPECTED_H5_COUNT = 5_398
EXPECTED_H5_BYTES = 3_613_998_101_757
EXPECTED_DOMAIN_LIST_SHA256 = (
    "295c6da1c9f8846a1ea3993eca12a3232d16a2b3a4b0d8791c7c45392186709b"
)
EXPECTED_CANONICAL_INVENTORY_SHA256 = (
    "2e6e3602a0858aaafc849cfa7cc1ee7e076736cb15335d7914726898f06f6cdf"
)
INVENTORY_NAME = "mdcath_source_inventory.jsonl"
METADATA_NAME = "mdcath_source_inventory.metadata.json"
INVENTORY_SCHEMA = 1

ROW_FIELDS = {
    "git_blob_sha1",
    "lfs_pointer_size",
    "lfs_sha256",
    "lfs_size",
    "path",
    "size",
    "xet_hash",
}
_DOMAIN_PATH = re.compile(r"data/mdcath_dataset_([A-Za-z0-9]+)\.h5")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class InventoryError(ValueError):
    """Raised when source or cached inventory evidence is inconsistent."""


@dataclass(frozen=True)
class Expectations:
    revision: str = SOURCE_REVISION
    h5_count: int = EXPECTED_H5_COUNT
    h5_bytes: int = EXPECTED_H5_BYTES
    domain_list_sha256: str = EXPECTED_DOMAIN_LIST_SHA256
    canonical_inventory_sha256: str = EXPECTED_CANONICAL_INVENTORY_SHA256


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_line(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_line(row) for row in rows)


def _domain_from_path(path: str) -> str:
    match = _DOMAIN_PATH.fullmatch(path)
    if match is None:
        raise InventoryError(f"unexpected mdCATH source path: {path!r}")
    return match.group(1)


def rows_from_repo_tree(entries: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert official ``RepoFile`` entries to the stable inventory schema."""

    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str):
            raise InventoryError(f"tree entry has no file path: {entry!r}")
        _domain_from_path(path)
        lfs = getattr(entry, "lfs", None)
        if lfs is None:
            raise InventoryError(f"source HDF5 is not LFS-backed: {path}")
        rows.append(
            {
                "git_blob_sha1": getattr(entry, "blob_id", None),
                "lfs_pointer_size": getattr(lfs, "pointer_size", None),
                "lfs_sha256": getattr(lfs, "sha256", None),
                "lfs_size": getattr(lfs, "size", None),
                "path": path,
                "size": getattr(entry, "size", None),
                "xet_hash": getattr(entry, "xet_hash", None),
            }
        )
    rows.sort(key=lambda row: row["path"])
    return rows


def _validate_row(row: dict[str, Any], index: int) -> None:
    if set(row) != ROW_FIELDS:
        missing = sorted(ROW_FIELDS - set(row))
        extra = sorted(set(row) - ROW_FIELDS)
        raise InventoryError(f"row {index} fields mismatch: missing={missing} extra={extra}")
    _domain_from_path(row["path"])
    if not isinstance(row["size"], int) or row["size"] <= 0:
        raise InventoryError(f"row {index} has invalid size")
    if row["lfs_size"] != row["size"]:
        raise InventoryError(f"row {index} LFS size differs from file size")
    if not isinstance(row["lfs_pointer_size"], int) or row["lfs_pointer_size"] <= 0:
        raise InventoryError(f"row {index} has invalid LFS pointer size")
    if not isinstance(row["git_blob_sha1"], str) or _SHA1.fullmatch(row["git_blob_sha1"]) is None:
        raise InventoryError(f"row {index} has invalid Git blob SHA1")
    for key in ("lfs_sha256", "xet_hash"):
        if not isinstance(row[key], str) or _SHA256.fullmatch(row[key]) is None:
            raise InventoryError(f"row {index} has invalid {key}")


def validate_inventory(
    rows: list[dict[str, Any]],
    *,
    domain_list_bytes: bytes | None,
    expectations: Expectations | None = None,
) -> dict[str, Any]:
    """Validate exact corpus identity and return its deterministic summary."""

    if expectations is None:
        expectations = Expectations()
    for index, row in enumerate(rows):
        _validate_row(row, index)
    paths = [row["path"] for row in rows]
    domains = [_domain_from_path(path) for path in paths]
    lfs_oids = [row["lfs_sha256"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise InventoryError("inventory paths are not strictly sorted and unique")
    if len(domains) != len(set(domains)):
        raise InventoryError("inventory domain identifiers are not unique")
    if len(lfs_oids) != len(set(lfs_oids)):
        raise InventoryError("inventory LFS SHA256 identifiers are not unique")
    if len(rows) != expectations.h5_count:
        raise InventoryError(f"HDF5 count {len(rows)} != {expectations.h5_count}")
    total_bytes = sum(row["size"] for row in rows)
    if total_bytes != expectations.h5_bytes:
        raise InventoryError(f"HDF5 bytes {total_bytes} != {expectations.h5_bytes}")

    inventory_bytes = canonical_jsonl(rows)
    inventory_sha256 = _sha256(inventory_bytes)
    if inventory_sha256 != expectations.canonical_inventory_sha256:
        raise InventoryError(
            "canonical inventory SHA256 "
            f"{inventory_sha256} != {expectations.canonical_inventory_sha256}"
        )

    domain_list_sha256 = expectations.domain_list_sha256
    if domain_list_bytes is not None:
        domain_list_sha256 = _sha256(domain_list_bytes)
        if domain_list_sha256 != expectations.domain_list_sha256:
            raise InventoryError(
                f"domain-list SHA256 {domain_list_sha256} != "
                f"{expectations.domain_list_sha256}"
            )
        try:
            source_domains = [
                line.strip()
                for line in domain_list_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
        except UnicodeDecodeError as exc:
            raise InventoryError("domain list is not valid UTF-8") from exc
        if source_domains != domains:
            raise InventoryError("official domain list does not exactly match inventory order")

    return {
        "canonical_inventory_bytes": len(inventory_bytes),
        "canonical_inventory_sha256": inventory_sha256,
        "domain_list_sha256": domain_list_sha256,
        "h5_count": len(rows),
        "total_bytes": total_bytes,
        "unique_domains": len(set(domains)),
        "unique_lfs_sha256": len(set(lfs_oids)),
    }


def _fetch_remote() -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    try:
        import huggingface_hub
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise InventoryError("fetch requires the huggingface_hub package") from exc

    api = HfApi(endpoint=SOURCE_ENDPOINT, token=False)
    info = api.dataset_info(SOURCE_REPO, revision=SOURCE_REVISION)
    if info.sha != SOURCE_REVISION:
        raise InventoryError(f"resolved revision {info.sha} != {SOURCE_REVISION}")
    if info.private or info.gated:
        raise InventoryError("frozen public source unexpectedly became private or gated")
    entries = list(
        api.list_repo_tree(
            SOURCE_REPO,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            path_in_repo="data",
            recursive=True,
            expand=False,
        )
    )
    rows = rows_from_repo_tree(entries)
    domain_list_path = hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        filename=SOURCE_DOMAIN_LIST,
        endpoint=SOURCE_ENDPOINT,
        token=False,
    )
    domain_list_bytes = Path(domain_list_path).read_bytes()
    source = {
        "huggingface_hub_version": huggingface_hub.__version__,
        "repository_last_modified": info.last_modified.isoformat()
        if info.last_modified is not None
        else None,
    }
    return rows, domain_list_bytes, source


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def freeze(output_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    inventory_path = output_dir / INVENTORY_NAME
    metadata_path = output_dir / METADATA_NAME
    if not overwrite and (inventory_path.exists() or metadata_path.exists()):
        raise InventoryError("refusing to overwrite an existing frozen inventory")

    rows, domain_list_bytes, source = _fetch_remote()
    summary = validate_inventory(rows, domain_list_bytes=domain_list_bytes)
    metadata = {
        **summary,
        **source,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_serialization": "UTF-8 JSON Lines; sorted paths; sorted keys; compact separators; LF",
        "inventory_schema": INVENTORY_SCHEMA,
        "source_domain_list": SOURCE_DOMAIN_LIST,
        "source_domain_list_matches_inventory": True,
        "source_endpoint": SOURCE_ENDPOINT,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    }
    inventory_bytes = canonical_jsonl(rows)
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(inventory_path, inventory_bytes)
    _atomic_write(metadata_path, metadata_bytes)
    return {**summary, "inventory": str(inventory_path), "metadata": str(metadata_path)}


def _read_canonical_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise InventoryError("inventory must be non-empty canonical LF-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(keepends=True)):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InventoryError(f"invalid JSONL row {index}") from exc
        if not isinstance(row, dict) or _canonical_line(row) != line:
            raise InventoryError(f"row {index} is not canonically serialized")
        rows.append(row)
    return rows, payload


def verify_offline(output_dir: Path) -> dict[str, Any]:
    inventory_path = output_dir / INVENTORY_NAME
    metadata_path = output_dir / METADATA_NAME
    rows, payload = _read_canonical_rows(inventory_path)
    summary = validate_inventory(rows, domain_list_bytes=None)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        **summary,
        "inventory_schema": INVENTORY_SCHEMA,
        "source_domain_list": SOURCE_DOMAIN_LIST,
        "source_domain_list_matches_inventory": True,
        "source_endpoint": SOURCE_ENDPOINT,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise InventoryError(f"metadata {key}={metadata.get(key)!r} != {expected!r}")
    if _sha256(payload) != metadata["canonical_inventory_sha256"]:
        raise InventoryError("inventory payload does not match metadata SHA256")
    return {**summary, "inventory": str(inventory_path), "metadata": str(metadata_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--output-dir", required=True, type=Path)
    fetch_parser.add_argument("--overwrite", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        if args.command == "fetch":
            result = freeze(args.output_dir, overwrite=args.overwrite)
        else:
            result = verify_offline(args.output_dir)
    except (InventoryError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({**result, "status": "PASS"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
