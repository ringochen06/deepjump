#!/usr/bin/env python
"""Build and atomically publish a strict full-mdCATH staging manifest.

The command is intentionally fail-closed: no output artifact is replaced until the
official 5,398-domain identity, exact local inventory, every HDF5 file, and the
complete 5x5 trajectory grid have all passed validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import h5py
import numpy as np


EXPECTED_DOMAINS = 5_398
EXPECTED_H5_BYTES = 3_613_998_101_757
EXPECTED_TRAJECTORIES = 134_950
EXPECTED_OFFICIAL_LIST_SHA256 = (
    "295c6da1c9f8846a1ea3993eca12a3232d16a2b3a4b0d8791c7c45392186709b"
)
EXPECTED_SOURCE_REVISION = "5e3ed8aec62b689e01751db16275fdcdbc39e47f"
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "2e6e3602a0858aaafc849cfa7cc1ee7e076736cb15335d7914726898f06f6cdf"
)
TEMPERATURES = (320, 348, 379, 413, 450)
REPLICAS = (0, 1, 2, 3, 4)
DOMAIN_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SOURCE_PATH_PATTERN = re.compile(
    r"^data/mdcath_dataset_([A-Za-z0-9]+)\.h5$"
)
SOURCE_INVENTORY_FIELDS = {
    "path",
    "size",
    "git_blob_sha1",
    "lfs_sha256",
    "lfs_size",
    "lfs_pointer_size",
    "xet_hash",
}
PAYLOAD_HASH_SIDECAR_SCHEMA = "deepjump.payload_sha256.v1"
PAYLOAD_HASH_SIDECAR_FIELDS = {
    "schema",
    "source_revision",
    "file",
    "sha256",
    "fingerprint",
    "hash_script_sha256",
    "completed_at",
}
PAYLOAD_FINGERPRINT_FIELDS = {
    "device",
    "inode",
    "size",
    "mtime_ns",
    "ctime_ns",
}
LIVE_REHASH_PASS_STATUS = "PASS_FULL_LIVE_PAYLOAD_REHASH"
SIDECAR_PREAUDIT_STATUS = "PREAUDIT_SIDECAR_ATTESTATION_ONLY"
REHASH_JOURNAL_SCHEMA = "deepjump.full_payload_rehash_journal.v1"
REHASH_RECORD_SCHEMA = "deepjump.full_payload_rehash_record.v1"
REHASH_COMPLETION_SCHEMA = "deepjump.full_payload_rehash_completion.v1"
REHASH_SESSION_FILE = "session.json"
REHASH_COMPLETION_FILE = "completion.json"
REHASH_JOURNAL_FIELDS = {
    "schema",
    "nonce",
    "audit_script_sha256",
    "source_revision",
    "source_inventory_sha256",
    "official_list_sha256",
    "root_identity",
    "expected_records",
    "created_at",
}
REHASH_RECORD_FIELDS = {
    "schema",
    "session_sha256",
    "audit_script_sha256",
    "source_revision",
    "source_inventory_sha256",
    "official_list_sha256",
    "root_identity",
    "domain",
    "file",
    "expected_lfs_sha256",
    "payload_sha256",
    "fingerprint",
    "completed_at",
}
REHASH_COMPLETION_FIELDS = {
    "schema",
    "session_sha256",
    "audit_script_sha256",
    "source_revision",
    "source_inventory_sha256",
    "official_list_sha256",
    "root_identity",
    "expected_records",
    "record_set_sha256",
    "completed_at",
}
ROOT_IDENTITY_FIELDS = {"path", "device", "inode"}


@dataclass(frozen=True)
class AuditExpectations:
    domains: int = EXPECTED_DOMAINS
    h5_bytes: int = EXPECTED_H5_BYTES
    trajectories: int = EXPECTED_TRAJECTORIES
    official_list_sha256: str = EXPECTED_OFFICIAL_LIST_SHA256
    source_revision: str = EXPECTED_SOURCE_REVISION
    source_inventory_sha256: str = EXPECTED_SOURCE_INVENTORY_SHA256


@dataclass(frozen=True)
class SourceFileIdentity:
    path: str
    domain: str
    size: int
    lfs_sha256: str


FileFingerprint = tuple[int, int, int, int, int]


def _fingerprint(stat_result: os.stat_result) -> FileFingerprint:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _fingerprint_dict(fingerprint: FileFingerprint) -> dict[str, int]:
    return dict(
        zip(
            ("device", "inode", "size", "mtime_ns", "ctime_ns"),
            fingerprint,
            strict=True,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, label: str) -> bytes:
    """Hash and parse the same stable regular-file descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} must be a regular non-symlink file") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        content = handle.read()
        after = os.fstat(handle.fileno())
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(content) != before.st_size:
        raise ValueError(f"{label} changed while it was being read")
    return content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write(
    path: Path,
    content: bytes,
    validate_temporary: Callable[[Path], None] | None = None,
) -> None:
    """Write bytes durably beside the destination, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink output: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if validate_temporary is not None:
            validate_temporary(temporary_path)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _exclusive_publish(path: Path, content: bytes) -> None:
    """Atomically publish immutable bytes without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise ValueError(f"refusing to replace existing journal entry: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError(
                f"journal entry appeared concurrently: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _report_progress(phase: str, completed: int, total: int, path: Path) -> None:
    if completed == 1 or completed == total or completed % 100 == 0:
        print(
            json.dumps(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "file": path.name,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


def _write_with_sha(
    path: Path,
    content: bytes,
    validate_temporary: Callable[[Path], None] | None = None,
) -> str:
    digest = _sha256_bytes(content)
    _atomic_write(path, content, validate_temporary)
    sidecar = path.with_name(path.name + ".sha256")
    _atomic_write(sidecar, f"{digest}  {path.name}\n".encode())
    return digest


def _load_official_domains(
    official_list: Path, expectations: AuditExpectations
) -> tuple[list[str], str]:
    content = _read_regular_bytes(official_list, "official domain list")
    digest = _sha256_bytes(content)
    if digest != expectations.official_list_sha256:
        raise ValueError(
            f"official domain list SHA256 mismatch: {digest}"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("official domain list is not valid UTF-8") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise ValueError("official domain list must be canonical LF-terminated text")
    domains = text.splitlines()
    if len(domains) != expectations.domains:
        raise ValueError(
            f"official domain count {len(domains)} != {expectations.domains}"
        )
    if len(set(domains)) != len(domains):
        raise ValueError("official domain list contains duplicates")
    invalid = [domain for domain in domains if not DOMAIN_PATTERN.fullmatch(domain)]
    if invalid:
        raise ValueError(f"official domain list contains invalid ids: {invalid[:5]}")
    return domains, digest


def _load_source_inventory(
    source_inventory: Path,
    source_revision: str,
    domains: list[str],
    expectations: AuditExpectations,
) -> tuple[dict[str, SourceFileIdentity], str]:
    if source_revision != expectations.source_revision:
        raise ValueError(
            f"source revision {source_revision!r} != "
            f"{expectations.source_revision!r}"
        )
    if not COMMIT_PATTERN.fullmatch(source_revision):
        raise ValueError("source revision must be 40 lowercase hex characters")
    content = _read_regular_bytes(source_inventory, "source inventory")
    digest = _sha256_bytes(content)
    if digest != expectations.source_inventory_sha256:
        raise ValueError(f"source inventory SHA256 mismatch: {digest}")
    if content and not content.endswith(b"\n"):
        raise ValueError("source inventory must end with LF")

    identities: list[SourceFileIdentity] = []
    paths: list[str] = []
    lfs_hashes: list[str] = []
    for line_number, raw_line in enumerate(content.splitlines(keepends=True), 1):
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            raise ValueError(
                f"source inventory line {line_number} is blank or lacks LF"
            )
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"source inventory line {line_number} is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(row, dict) or set(row) != SOURCE_INVENTORY_FIELDS:
            raise ValueError(
                f"source inventory line {line_number} has non-canonical fields"
            )
        canonical = (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if raw_line != canonical:
            raise ValueError(
                f"source inventory line {line_number} is not canonically serialized"
            )

        path = row["path"]
        match = SOURCE_PATH_PATTERN.fullmatch(path) if isinstance(path, str) else None
        if match is None:
            raise ValueError(
                f"source inventory line {line_number} has invalid HDF5 path"
            )
        size = row["size"]
        lfs_size = row["lfs_size"]
        pointer_size = row["lfs_pointer_size"]
        if (
            type(size) is not int
            or size <= 0
            or type(lfs_size) is not int
            or lfs_size != size
            or type(pointer_size) is not int
            or pointer_size <= 0
        ):
            raise ValueError(
                f"source inventory line {line_number} has invalid size fields"
            )
        lfs_sha256 = row["lfs_sha256"]
        if not isinstance(lfs_sha256, str) or not SHA256_PATTERN.fullmatch(
            lfs_sha256
        ):
            raise ValueError(
                f"source inventory line {line_number} has invalid LFS SHA256"
            )
        git_blob_sha1 = row["git_blob_sha1"]
        if not isinstance(git_blob_sha1, str) or not SHA1_PATTERN.fullmatch(
            git_blob_sha1
        ):
            raise ValueError(
                f"source inventory line {line_number} has invalid git blob SHA1"
            )
        xet_hash = row["xet_hash"]
        if not isinstance(xet_hash, str) or not SHA256_PATTERN.fullmatch(xet_hash):
            raise ValueError(
                f"source inventory line {line_number} has invalid Xet hash"
            )
        domain = match.group(1)
        identities.append(SourceFileIdentity(path, domain, size, lfs_sha256))
        paths.append(path)
        lfs_hashes.append(lfs_sha256)

    if len(identities) != expectations.domains:
        raise ValueError(
            f"source inventory files {len(identities)} != {expectations.domains}"
        )
    if paths != sorted(paths):
        raise ValueError("source inventory paths are not UTF-8 sorted")
    if len(set(paths)) != len(paths):
        raise ValueError("source inventory contains duplicate paths")
    if len(set(lfs_hashes)) != len(lfs_hashes):
        raise ValueError("source inventory contains duplicate LFS SHA256 values")
    source_domains = [identity.domain for identity in identities]
    if set(source_domains) != set(domains) or len(set(source_domains)) != len(domains):
        raise ValueError("source inventory domain union differs from official list")
    total_bytes = sum(identity.size for identity in identities)
    if total_bytes != expectations.h5_bytes:
        raise ValueError(
            f"source inventory bytes {total_bytes} != {expectations.h5_bytes}"
        )
    return {identity.domain: identity for identity in identities}, digest


def _walk_symlinks(data_root: Path) -> list[Path]:
    symlinks: list[Path] = []
    for directory, dirnames, filenames in os.walk(data_root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            path = base / name
            if path.is_symlink():
                symlinks.append(path)
    return sorted(symlinks)


def _is_retained_huggingface_cache_incomplete(root: Path, path: Path) -> bool:
    """Identify resumable cache fragments that are outside the payload union."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return (
        relative.parts[:4] == (".cache", "huggingface", "download", "data")
        and path.name.endswith(".incomplete")
    )


def _validate_resolved_download_failure_ledger(
    root: Path,
    domains: list[str],
) -> None:
    """Allow one preserved failure ledger only when every entry is now resolved."""

    failure_ledgers = sorted(root.rglob("download_failures.txt"))
    if not failure_ledgers:
        return
    expected = root / "download_failures.txt"
    if failure_ledgers != [expected]:
        raise ValueError(
            "ambiguous shared download failure file remains: "
            f"{failure_ledgers[:5]}"
        )
    content = _read_regular_bytes(expected, "download failure ledger")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("download failure ledger is not valid UTF-8") from exc
    entries = text.splitlines()
    if (
        not entries
        or not text.endswith("\n")
        or any(not DOMAIN_PATTERN.fullmatch(entry) for entry in entries)
        or len(entries) != len(set(entries))
    ):
        raise ValueError("download failure ledger is not a canonical domain list")
    unknown = sorted(set(entries) - set(domains))
    if unknown:
        raise ValueError(
            f"download failure ledger contains unknown domains: {unknown[:5]}"
        )
    unresolved = [
        domain
        for domain in entries
        if not (root / "data" / f"mdcath_dataset_{domain}.h5").is_file()
    ]
    if unresolved:
        raise ValueError(
            "download failure ledger contains unresolved domains: "
            f"{unresolved[:5]}"
        )
    print(
        f"retained_resolved_download_failure_ledger={len(entries)}",
        flush=True,
    )


def _validate_inventory(
    root: Path,
    domains: list[str],
    source_by_domain: dict[str, SourceFileIdentity],
    expectations: AuditExpectations,
) -> tuple[list[Path], int]:
    data_root = root / "data"
    if root.is_symlink() or data_root.is_symlink():
        raise ValueError("staging root and data directory must not be symlinks")
    if not data_root.is_dir():
        raise ValueError(f"missing data directory: {data_root}")
    symlinks = _walk_symlinks(root)
    if symlinks:
        raise ValueError(f"staging tree contains symlinks: {symlinks[:5]}")

    incomplete = sorted(
        path for path in root.rglob("*") if ".incomplete" in path.name
    )
    retained_cache_incomplete = [
        path
        for path in incomplete
        if _is_retained_huggingface_cache_incomplete(root, path)
    ]
    blocking_incomplete = [
        path
        for path in incomplete
        if not _is_retained_huggingface_cache_incomplete(root, path)
    ]
    if blocking_incomplete:
        raise ValueError(
            f"incomplete downloads remain: {blocking_incomplete[:5]}"
        )
    if retained_cache_incomplete:
        print(
            "retained_huggingface_cache_incomplete="
            f"{len(retained_cache_incomplete)}",
            flush=True,
        )
    _validate_resolved_download_failure_ledger(root, domains)

    expected_by_name = {
        f"mdcath_dataset_{domain}.h5": data_root / f"mdcath_dataset_{domain}.h5"
        for domain in domains
    }
    actual = sorted(root.rglob("mdcath_dataset_*.h5"))
    actual_names = [path.name for path in actual]
    if len(actual_names) != len(set(actual_names)):
        raise ValueError("duplicate HDF5 filenames exist under staging root")
    expected_names = set(expected_by_name)
    actual_name_set = set(actual_names)
    missing = sorted(expected_names - actual_name_set)
    extra = sorted(actual_name_set - expected_names)
    misplaced = sorted(
        str(path)
        for path in actual
        if path.resolve() != expected_by_name.get(path.name, Path("/__missing__")).resolve()
    )
    if missing or extra or misplaced:
        raise ValueError(
            "HDF5 exact inventory mismatch: "
            f"missing={missing[:5]} extra={extra[:5]} misplaced={misplaced[:5]}"
        )
    if len(actual) != expectations.domains:
        raise ValueError(f"HDF5 count {len(actual)} != {expectations.domains}")
    if any(path.is_symlink() or not path.is_file() for path in actual):
        raise ValueError("every HDF5 path must be a regular non-symlink file")
    size_mismatches = []
    for domain in domains:
        path = data_root / f"mdcath_dataset_{domain}.h5"
        expected_size = source_by_domain[domain].size
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            size_mismatches.append((path.name, actual_size, expected_size))
    if size_mismatches:
        raise ValueError(
            f"per-file source size mismatch: {size_mismatches[:5]}"
        )
    total_bytes = sum(path.stat().st_size for path in actual)
    if total_bytes != expectations.h5_bytes:
        raise ValueError(
            f"HDF5 bytes {total_bytes} != {expectations.h5_bytes}"
        )
    return [
        data_root / f"mdcath_dataset_{domain}.h5" for domain in domains
    ], total_bytes


def _verify_payload_sha256(
    path: Path, expected_sha256: str
) -> tuple[str, FileFingerprint]:
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
        after = os.fstat(handle.fileno())
    path_after = path.stat()
    if _fingerprint(before) != _fingerprint(after) or _fingerprint(
        after
    ) != _fingerprint(path_after):
        raise ValueError(f"HDF5 file changed during payload hashing: {path.name}")
    digest = digest_builder.hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"payload SHA256 mismatch for {path.name}: {digest} != {expected_sha256}"
        )
    return digest, _fingerprint(after)


def _root_identity(root: Path) -> dict[str, str | int]:
    stat_result = root.stat()
    return {
        "path": str(root),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
    }


def _load_canonical_json(path: Path, label: str) -> tuple[dict, bytes, str]:
    raw = _read_regular_bytes(path, label)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or raw != _json_bytes(payload):
        raise ValueError(f"{label} is not canonically serialized")
    return payload, raw, _sha256_bytes(raw)


def _validate_timestamp(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} lacks timezone")


def _journal_identity(
    *,
    audit_script_sha256: str,
    source_revision: str,
    source_inventory_sha256: str,
    official_list_sha256: str,
    root_identity: dict[str, str | int],
) -> dict[str, object]:
    return {
        "audit_script_sha256": audit_script_sha256,
        "source_revision": source_revision,
        "source_inventory_sha256": source_inventory_sha256,
        "official_list_sha256": official_list_sha256,
        "root_identity": root_identity,
    }


def _prepare_rehash_journal(
    journal_dir: Path,
    *,
    identity: dict[str, object],
    expected_records: int,
) -> tuple[dict, str]:
    if journal_dir.is_symlink():
        raise ValueError("rehash journal directory must not be a symlink")
    journal_dir.mkdir(parents=True, exist_ok=True)
    if journal_dir.is_symlink() or not journal_dir.is_dir():
        raise ValueError("rehash journal directory must be a real directory")
    children = list(journal_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("rehash journal contains non-regular entries")
    session_path = journal_dir / REHASH_SESSION_FILE
    if not session_path.exists():
        if children:
            raise ValueError("rehash journal entries exist before its session")
        session = {
            "schema": REHASH_JOURNAL_SCHEMA,
            "nonce": secrets.token_hex(32),
            **identity,
            "expected_records": expected_records,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _exclusive_publish(session_path, _json_bytes(session))
    session, _, session_sha256 = _load_canonical_json(
        session_path, "rehash journal session"
    )
    if set(session) != REHASH_JOURNAL_FIELDS:
        raise ValueError("rehash journal session fields mismatch")
    if (
        session["schema"] != REHASH_JOURNAL_SCHEMA
        or not isinstance(session["nonce"], str)
        or not SHA256_PATTERN.fullmatch(session["nonce"])
        or session["expected_records"] != expected_records
        or any(session.get(key) != value for key, value in identity.items())
    ):
        raise ValueError("rehash journal session identity mismatch")
    root_value = session["root_identity"]
    if not isinstance(root_value, dict) or set(root_value) != ROOT_IDENTITY_FIELDS:
        raise ValueError("rehash journal root identity is invalid")
    _validate_timestamp(session["created_at"], "rehash journal creation time")
    return session, session_sha256


def _record_path(journal_dir: Path, domain: str) -> Path:
    return journal_dir / f"payload_{domain}.json"


def _validate_rehash_record(
    record_path: Path,
    *,
    session_sha256: str,
    identity: dict[str, object],
    domain: str,
    source: SourceFileIdentity,
    payload_path: Path,
) -> tuple[str, FileFingerprint, bytes]:
    record, raw, _ = _load_canonical_json(record_path, "rehash journal record")
    if set(record) != REHASH_RECORD_FIELDS:
        raise ValueError(f"rehash journal record fields mismatch: {record_path}")
    if (
        record["schema"] != REHASH_RECORD_SCHEMA
        or record["session_sha256"] != session_sha256
        or any(record.get(key) != value for key, value in identity.items())
        or record["domain"] != domain
        or record["file"] != source.path
        or record["expected_lfs_sha256"] != source.lfs_sha256
        or record["payload_sha256"] != source.lfs_sha256
    ):
        raise ValueError(f"rehash journal record identity mismatch: {record_path}")
    fingerprint = _fingerprint_from_sidecar(record["fingerprint"], record_path)
    if _fingerprint(payload_path.stat()) != fingerprint:
        raise ValueError(f"rehash journal payload fingerprint drift: {record_path}")
    _validate_timestamp(record["completed_at"], "rehash record completion time")
    return record["payload_sha256"], fingerprint, raw


def _journal_record_set_sha256(
    domains: list[str], record_bytes: dict[str, bytes]
) -> str:
    digest = hashlib.sha256()
    for domain in domains:
        raw = record_bytes[domain]
        digest.update(f"payload_{domain}.json\0".encode())
        digest.update(hashlib.sha256(raw).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_journal_inventory(
    journal_dir: Path,
    domains: list[str],
    *,
    require_completion: bool,
) -> None:
    expected = {REHASH_SESSION_FILE, *[f"payload_{domain}.json" for domain in domains]}
    if require_completion:
        expected.add(REHASH_COMPLETION_FILE)
    elif (journal_dir / REHASH_COMPLETION_FILE).exists():
        expected.add(REHASH_COMPLETION_FILE)
    children = list(journal_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("rehash journal contains non-regular entries")
    names = [path.name for path in children]
    if len(names) != len(set(names)) or set(names) != expected:
        missing = sorted(expected - set(names))
        extra = sorted(set(names) - expected)
        raise ValueError(
            "rehash journal exact inventory mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def _validate_or_publish_completion(
    journal_dir: Path,
    *,
    session_sha256: str,
    identity: dict[str, object],
    expected_records: int,
    record_set_sha256: str,
) -> str:
    completion_path = journal_dir / REHASH_COMPLETION_FILE
    expected = {
        "schema": REHASH_COMPLETION_SCHEMA,
        "session_sha256": session_sha256,
        **identity,
        "expected_records": expected_records,
        "record_set_sha256": record_set_sha256,
    }
    if not completion_path.exists():
        _exclusive_publish(
            completion_path,
            _json_bytes(
                {
                    **expected,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        )
    completion, _, completion_sha256 = _load_canonical_json(
        completion_path, "rehash journal completion"
    )
    if set(completion) != REHASH_COMPLETION_FIELDS or any(
        completion.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("rehash journal completion identity mismatch")
    _validate_timestamp(completion["completed_at"], "rehash completion time")
    return completion_sha256


def _fingerprint_from_sidecar(
    value: object, sidecar: Path
) -> FileFingerprint:
    if not isinstance(value, dict) or set(value) != PAYLOAD_FINGERPRINT_FIELDS:
        raise ValueError(f"payload hash sidecar has invalid fingerprint: {sidecar}")
    ordered = (
        value["device"],
        value["inode"],
        value["size"],
        value["mtime_ns"],
        value["ctime_ns"],
    )
    if any(type(item) is not int or item < 0 for item in ordered):
        raise ValueError(
            f"payload hash sidecar has non-integer fingerprint: {sidecar}"
        )
    return ordered


def _load_payload_hash_sidecars(
    sidecar_dir: Path,
    files: list[Path],
    domains: list[str],
    source_by_domain: dict[str, SourceFileIdentity],
    source_revision: str,
    expected_hash_script_sha256: str,
) -> tuple[dict[str, str], dict[str, FileFingerprint]]:
    if not SHA256_PATTERN.fullmatch(expected_hash_script_sha256):
        raise ValueError("expected hash script SHA256 must be 64 lowercase hex")
    if sidecar_dir.is_symlink() or not sidecar_dir.is_dir():
        raise ValueError(
            "payload hash sidecar directory must be a regular non-symlink directory"
        )
    children = sorted(sidecar_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("payload hash sidecar directory contains non-regular entries")
    expected_names = {f"{path.name}.json" for path in files}
    actual_names = {path.name for path in children}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(
            "payload hash sidecar exact inventory mismatch: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    hashes: dict[str, str] = {}
    fingerprints: dict[str, FileFingerprint] = {}
    for index, (domain, payload_path) in enumerate(
        zip(domains, files, strict=True), 1
    ):
        sidecar = sidecar_dir / f"{payload_path.name}.json"
        try:
            row = json.loads(sidecar.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid payload hash sidecar JSON: {sidecar}") from exc
        if not isinstance(row, dict) or set(row) != PAYLOAD_HASH_SIDECAR_FIELDS:
            raise ValueError(f"payload hash sidecar fields mismatch: {sidecar}")
        if row["schema"] != PAYLOAD_HASH_SIDECAR_SCHEMA:
            raise ValueError(f"payload hash sidecar schema mismatch: {sidecar}")
        if row["source_revision"] != source_revision:
            raise ValueError(f"payload hash sidecar revision mismatch: {sidecar}")
        if row["file"] != source_by_domain[domain].path:
            raise ValueError(f"payload hash sidecar file mismatch: {sidecar}")
        if row["hash_script_sha256"] != expected_hash_script_sha256:
            raise ValueError(f"payload hash sidecar script SHA256 mismatch: {sidecar}")
        if row["sha256"] != source_by_domain[domain].lfs_sha256:
            raise ValueError(f"payload hash sidecar/LFS SHA256 mismatch: {sidecar}")
        completed_at = row["completed_at"]
        if not isinstance(completed_at, str):
            raise ValueError(f"payload hash sidecar completion time invalid: {sidecar}")
        try:
            completed = datetime.fromisoformat(
                completed_at[:-1] + "+00:00"
                if completed_at.endswith("Z")
                else completed_at
            )
        except ValueError as exc:
            raise ValueError(
                f"payload hash sidecar completion time invalid: {sidecar}"
            ) from exc
        if completed.tzinfo is None:
            raise ValueError(
                f"payload hash sidecar completion time lacks timezone: {sidecar}"
            )
        fingerprint = _fingerprint_from_sidecar(row["fingerprint"], sidecar)
        if _fingerprint(payload_path.stat()) != fingerprint:
            raise ValueError(f"payload hash sidecar fingerprint is stale: {sidecar}")
        hashes[domain] = row["sha256"]
        fingerprints[domain] = fingerprint
        _report_progress("payload_sha256_sidecars", index, len(files), sidecar)
    return hashes, fingerprints


def _finite_frame(dataset: h5py.Dataset, frame: int, label: str) -> None:
    values = np.asarray(dataset[frame])
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{label} is not entirely finite")


def _scan_hdf5(
    path: Path,
    expected_domain: str,
    hashed_fingerprint: FileFingerprint,
) -> tuple[dict, str]:
    before = path.stat()
    if _fingerprint(before) != hashed_fingerprint:
        raise ValueError(f"HDF5 file changed after payload hashing: {path.name}")
    with h5py.File(path, "r") as handle:
        if list(handle.keys()) != [expected_domain]:
            raise ValueError(
                f"{path.name} top-level domain identity mismatch: {list(handle.keys())}"
            )
        group = handle[expected_domain]
        residues = int(group.attrs.get("numResidues", 0))
        atoms = int(group.attrs.get("numProteinAtoms", 0))
        if residues <= 0 or atoms <= 0:
            raise ValueError(f"{path.name} has invalid residue/atom counts")
        for required in ("psf", "resid", "resname"):
            if required not in group:
                raise ValueError(f"{path.name} missing topology field: {required}")
        if len(group["resid"]) != atoms or len(group["resname"]) != atoms:
            raise ValueError(f"{path.name} topology array length mismatch")

        trajectories: list[dict[str, int]] = []
        for temperature in TEMPERATURES:
            key = str(temperature)
            if key not in group or not isinstance(group[key], h5py.Group):
                raise ValueError(f"{path.name} missing temperature {temperature}")
            temperature_group = group[key]
            expected_replicas = {str(replica) for replica in REPLICAS}
            replica_keys = set(temperature_group.keys())
            if replica_keys != expected_replicas or any(
                not isinstance(temperature_group[name], h5py.Group)
                for name in replica_keys
            ):
                raise ValueError(
                    f"{path.name} temperature {temperature} replica grid mismatch"
                )
            for replica in REPLICAS:
                replica_group = temperature_group[str(replica)]
                if "coords" not in replica_group:
                    raise ValueError(
                        f"{path.name} temperature {temperature} replica {replica} missing coords"
                    )
                coords = replica_group["coords"]
                if not isinstance(coords, h5py.Dataset) or len(coords.shape) != 3:
                    raise ValueError(f"{path.name} coords must have rank 3")
                frames = int(replica_group.attrs.get("numFrames", 0))
                if frames <= 0 or coords.shape != (frames, atoms, 3):
                    raise ValueError(
                        f"{path.name} temperature {temperature} replica {replica} shape/numFrames mismatch"
                    )
                _finite_frame(
                    coords, 0,
                    f"{path.name} temperature {temperature} replica {replica} first frame",
                )
                _finite_frame(
                    coords, frames - 1,
                    f"{path.name} temperature {temperature} replica {replica} last frame",
                )
                trajectories.append({
                    "temp": temperature,
                    "replica": replica,
                    "num_frames": frames,
                })
    after = path.stat()
    if _fingerprint(before) != _fingerprint(after):
        raise ValueError(f"HDF5 file changed during audit: {path.name}")
    entry = {
        "file": path.name,
        "domain": expected_domain,
        "local_fingerprint": _fingerprint_dict(_fingerprint(after)),
        "num_residues": residues,
        "num_atoms": atoms,
        "trajectories": trajectories,
    }
    inventory_line = (
        f"{path.name}\t{before.st_size}\t{expected_domain}\t{residues}\t{atoms}\t"
        f"{len(trajectories)}\n"
    )
    return entry, inventory_line


def _validate_manifest(
    manifest: list[dict], domains: list[str], expectations: AuditExpectations
) -> int:
    if len(manifest) != expectations.domains:
        raise ValueError("manifest domain count mismatch")
    if [entry.get("domain") for entry in manifest] != domains:
        raise ValueError("manifest domain order or identity mismatch")
    files = [entry.get("file") for entry in manifest]
    if len(files) != len(set(files)):
        raise ValueError("manifest contains duplicate files")
    trajectories = 0
    expected_grid = [(temperature, replica) for temperature in TEMPERATURES for replica in REPLICAS]
    for entry in manifest:
        rows = entry.get("trajectories")
        if not isinstance(rows, list) or len(rows) != len(expected_grid):
            raise ValueError(f"manifest domain {entry.get('domain')} does not have 25 trajectories")
        grid = [(row.get("temp"), row.get("replica")) for row in rows]
        if grid != expected_grid:
            raise ValueError(f"manifest domain {entry.get('domain')} trajectory grid mismatch")
        if any(type(row.get("num_frames")) is not int or row["num_frames"] <= 0 for row in rows):
            raise ValueError(f"manifest domain {entry.get('domain')} has invalid frame counts")
        trajectories += len(rows)
    if trajectories != expectations.trajectories:
        raise ValueError(
            f"manifest trajectories {trajectories} != {expectations.trajectories}"
        )
    return trajectories


def _resolve_commit(value: str | None) -> str:
    commit = value
    if commit is None:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    if not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("generating commit must be 40 lowercase hex characters")
    return commit


def build_and_publish(
    *,
    root: str | Path,
    official_list: str | Path,
    source_inventory: str | Path,
    source_revision: str,
    payload_hash_sidecar_dir: str | Path | None,
    expected_hash_script_sha256: str | None,
    rehash_payloads: bool,
    manifest_output: str | Path,
    metadata_output: str | Path,
    audit_output: str | Path,
    generating_commit: str | None,
    rehash_journal_dir: str | Path | None = None,
    expectations: AuditExpectations = AuditExpectations(),
) -> dict:
    root_input = Path(root).expanduser().absolute()
    if root_input.is_symlink():
        raise ValueError("staging root must not be a symlink")
    root_path = root_input.resolve()
    if not root_path.is_dir():
        raise ValueError(f"staging root is not a directory: {root_path}")
    official_path = Path(official_list).expanduser().absolute()
    source_inventory_path = Path(source_inventory).expanduser().absolute()
    sidecar_dir = (
        Path(payload_hash_sidecar_dir).expanduser().absolute()
        if payload_hash_sidecar_dir is not None
        else None
    )
    if (sidecar_dir is None and not rehash_payloads) or (
        sidecar_dir is not None and rehash_payloads
    ):
        raise ValueError(
            "choose exactly one payload verification mode: "
            "--payload-hash-sidecar-dir or --rehash-payloads"
        )
    if sidecar_dir is not None and expected_hash_script_sha256 is None:
        raise ValueError(
            "expected hash script SHA256 is required with payload hash sidecars"
        )
    if rehash_payloads and expected_hash_script_sha256 is not None:
        raise ValueError(
            "expected hash script SHA256 is only valid with payload hash sidecars"
        )
    if sidecar_dir is not None and rehash_journal_dir is not None:
        raise ValueError("rehash journal is only valid with --rehash-payloads")
    manifest_path = Path(manifest_output).expanduser().absolute()
    metadata_path = Path(metadata_output).expanduser().absolute()
    audit_path = Path(audit_output).expanduser().absolute()
    outputs = (manifest_path, metadata_path, audit_path)
    sidecars = tuple(output.with_name(output.name + ".sha256") for output in outputs)
    if len(set((*outputs, *sidecars))) != 6:
        raise ValueError("artifact and SHA256 sidecar paths must all be distinct")
    if {official_path, source_inventory_path} & set((*outputs, *sidecars)):
        raise ValueError("source identity inputs must not be output artifacts")
    for output in (*outputs, *sidecars):
        if output.is_symlink():
            raise ValueError(f"refusing to replace symlink output: {output}")
        if output.exists() and not output.is_file():
            raise ValueError(f"output path is not a regular file: {output}")
        if output.parent.resolve() != root_path:
            raise ValueError("manifest, metadata, and audit outputs must be direct children of staging root")
    commit = _resolve_commit(generating_commit)
    domains, official_sha256 = _load_official_domains(official_path, expectations)
    source_by_domain, source_inventory_sha256 = _load_source_inventory(
        source_inventory_path,
        source_revision,
        domains,
        expectations,
    )
    files, total_bytes = _validate_inventory(
        root_path,
        domains,
        source_by_domain,
        expectations,
    )

    audit_script_sha256 = _sha256_bytes(
        _read_regular_bytes(Path(__file__).resolve(), "audit script")
    )
    journal_path: Path | None = None
    journal_session_sha256: str | None = None
    journal_completion_sha256: str | None = None
    journal_records_resumed = 0
    journal_records_created = 0

    if sidecar_dir is not None:
        assert expected_hash_script_sha256 is not None
        payload_hashes, payload_fingerprints = _load_payload_hash_sidecars(
            sidecar_dir,
            files,
            domains,
            source_by_domain,
            source_revision,
            expected_hash_script_sha256,
        )
        payload_verification_mode = "sidecar"
        payload_hash_sidecars_verified = len(payload_hashes)
    else:
        journal_path = (
            Path(rehash_journal_dir).expanduser().absolute()
            if rehash_journal_dir is not None
            else root_path / "control" / "full_payload_rehash_journal_v1"
        )
        try:
            journal_path.resolve().relative_to(root_path)
        except ValueError as exc:
            raise ValueError("rehash journal must remain inside staging root") from exc
        root_identity = _root_identity(root_path)
        identity = _journal_identity(
            audit_script_sha256=audit_script_sha256,
            source_revision=source_revision,
            source_inventory_sha256=source_inventory_sha256,
            official_list_sha256=official_sha256,
            root_identity=root_identity,
        )
        _, journal_session_sha256 = _prepare_rehash_journal(
            journal_path,
            identity=identity,
            expected_records=len(files),
        )
        completion_exists = (journal_path / REHASH_COMPLETION_FILE).exists()
        if completion_exists:
            _validate_journal_inventory(
                journal_path, domains, require_completion=True
            )
        payload_hashes = {}
        payload_fingerprints = {}
        record_bytes: dict[str, bytes] = {}
        for index, (domain, path) in enumerate(
            zip(domains, files, strict=True), 1
        ):
            record_path = _record_path(journal_path, domain)
            if record_path.exists():
                payload_hash, fingerprint, raw = _validate_rehash_record(
                    record_path,
                    session_sha256=journal_session_sha256,
                    identity=identity,
                    domain=domain,
                    source=source_by_domain[domain],
                    payload_path=path,
                )
                journal_records_resumed += 1
            else:
                if completion_exists:
                    raise ValueError(
                        "rehash journal completion exists with a missing record"
                    )
                payload_hash, fingerprint = _verify_payload_sha256(
                    path, source_by_domain[domain].lfs_sha256
                )
                record = {
                    "schema": REHASH_RECORD_SCHEMA,
                    "session_sha256": journal_session_sha256,
                    **identity,
                    "domain": domain,
                    "file": source_by_domain[domain].path,
                    "expected_lfs_sha256": source_by_domain[domain].lfs_sha256,
                    "payload_sha256": payload_hash,
                    "fingerprint": _fingerprint_dict(fingerprint),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                _exclusive_publish(record_path, _json_bytes(record))
                payload_hash, fingerprint, raw = _validate_rehash_record(
                    record_path,
                    session_sha256=journal_session_sha256,
                    identity=identity,
                    domain=domain,
                    source=source_by_domain[domain],
                    payload_path=path,
                )
                journal_records_created += 1
            payload_hashes[domain] = payload_hash
            payload_fingerprints[domain] = fingerprint
            record_bytes[domain] = raw
            _report_progress("payload_sha256", index, len(files), path)
        _validate_journal_inventory(
            journal_path,
            domains,
            require_completion=completion_exists,
        )
        payload_verification_mode = "full_rehash"
        payload_hash_sidecars_verified = 0

    manifest: list[dict] = []
    inventory_digest = hashlib.sha256()
    for index, (domain, path) in enumerate(
        zip(domains, files, strict=True), 1
    ):
        entry, inventory_line = _scan_hdf5(
            path,
            domain,
            payload_fingerprints[domain],
        )
        entry["size"] = source_by_domain[domain].size
        entry["sha256"] = payload_hashes[domain]
        manifest.append(entry)
        inventory_digest.update(
            (
                inventory_line.rstrip("\n")
                + f"\t{payload_hashes[domain]}\n"
            ).encode()
        )
        _report_progress("hdf5_structure", index, len(files), path)
    trajectories = _validate_manifest(manifest, domains, expectations)

    if payload_verification_mode == "full_rehash":
        assert journal_path is not None
        assert journal_session_sha256 is not None
        final_record_bytes: dict[str, bytes] = {}
        for domain, path in zip(domains, files, strict=True):
            payload_hash, fingerprint, raw = _validate_rehash_record(
                _record_path(journal_path, domain),
                session_sha256=journal_session_sha256,
                identity=identity,
                domain=domain,
                source=source_by_domain[domain],
                payload_path=path,
            )
            if (
                payload_hash != payload_hashes[domain]
                or fingerprint != payload_fingerprints[domain]
                or _fingerprint(path.stat()) != fingerprint
            ):
                raise ValueError(
                    f"rehash journal final fingerprint sweep failed: {path.name}"
                )
            final_record_bytes[domain] = raw
        record_set_sha256 = _journal_record_set_sha256(
            domains, final_record_bytes
        )
        journal_completion_sha256 = _validate_or_publish_completion(
            journal_path,
            session_sha256=journal_session_sha256,
            identity=identity,
            expected_records=len(files),
            record_set_sha256=record_set_sha256,
        )
        _validate_journal_inventory(
            journal_path, domains, require_completion=True
        )
        for domain, path in zip(domains, files, strict=True):
            if _fingerprint(path.stat()) != payload_fingerprints[domain]:
                raise ValueError(
                    f"payload changed after rehash completion: {path.name}"
                )

    manifest_content = _json_bytes(manifest)
    manifest_sha256 = _sha256_bytes(manifest_content)
    created_at = datetime.now(timezone.utc).isoformat()
    data_gate_passed = payload_verification_mode == "full_rehash"
    status = (
        LIVE_REHASH_PASS_STATUS
        if data_gate_passed
        else SIDECAR_PREAUDIT_STATUS
    )
    metadata = {
        "schema": "deepjump.full_mdcath_staging.v1",
        "selection_strategy": "official-full-5398",
        "generating_commit": commit,
        "source_repo": "compsciencelab/mdCATH",
        "source_revision": source_revision,
        "source_inventory_file": source_inventory_path.name,
        "source_inventory_sha256": source_inventory_sha256,
        "official_domain_list_file": official_path.name,
        "official_domain_list_sha256": official_sha256,
        "domains": len(domains),
        "h5_files": len(files),
        "h5_bytes": total_bytes,
        "trajectories": trajectories,
        "temperature_replica_grid": "5x5",
        "hdf5_files_structurally_verified": len(files),
        "payload_hash_verification_mode": payload_verification_mode,
        "payload_hashes_verified": len(payload_hashes),
        "payload_hash_sidecars_verified": payload_hash_sidecars_verified,
        "expected_hash_script_sha256": expected_hash_script_sha256,
        "audit_script_sha256": audit_script_sha256,
        "payload_rehash_journal_dir": (
            str(journal_path) if journal_path is not None else None
        ),
        "payload_rehash_journal_session_sha256": journal_session_sha256,
        "payload_rehash_journal_completion_sha256": journal_completion_sha256,
        "payload_rehash_records_resumed": journal_records_resumed,
        "payload_rehash_records_created": journal_records_created,
        "data_gate_passed": data_gate_passed,
        "live_payload_bytes_rehashed": data_gate_passed,
        "all_coordinate_frames_finite_verified": False,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "finite_endpoint_frames_verified": trajectories * 2,
        "verified_local_inventory_sha256": inventory_digest.hexdigest(),
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "created_at": created_at,
    }
    metadata_content = _json_bytes(metadata)
    metadata_sha256 = _sha256_bytes(metadata_content)
    audit = {
        "schema": "deepjump.full_mdcath_audit.v1",
        "status": status,
        "root": str(root_path),
        "source_repo": "compsciencelab/mdCATH",
        "source_revision": source_revision,
        "source_inventory_sha256": source_inventory_sha256,
        "official_domain_list_sha256": official_sha256,
        "domains": len(domains),
        "h5_files": len(files),
        "h5_bytes": total_bytes,
        "trajectories": trajectories,
        "hdf5_files_structurally_verified": len(files),
        "payload_hash_verification_mode": payload_verification_mode,
        "payload_hashes_verified": len(payload_hashes),
        "payload_hash_sidecars_verified": payload_hash_sidecars_verified,
        "expected_hash_script_sha256": expected_hash_script_sha256,
        "audit_script_sha256": audit_script_sha256,
        "payload_rehash_journal_dir": (
            str(journal_path) if journal_path is not None else None
        ),
        "payload_rehash_journal_session_sha256": journal_session_sha256,
        "payload_rehash_journal_completion_sha256": journal_completion_sha256,
        "payload_rehash_records_resumed": journal_records_resumed,
        "payload_rehash_records_created": journal_records_created,
        "data_gate_passed": data_gate_passed,
        "live_payload_bytes_rehashed": data_gate_passed,
        "all_coordinate_frames_finite_verified": False,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "finite_endpoint_frames_verified": trajectories * 2,
        "verified_local_inventory_sha256": inventory_digest.hexdigest(),
        "manifest_sha256": manifest_sha256,
        "metadata_sha256": metadata_sha256,
        "generating_commit": commit,
        "completed_at": created_at,
    }
    audit_content = _json_bytes(audit)

    # Publish only after every validation above succeeded. Each artifact and sidecar
    # is individually durable and atomic; the audit is written last as the commit marker.
    def validate_manifest_temporary(temporary_path: Path) -> None:
        temporary_manifest = json.loads(temporary_path.read_text())
        if not isinstance(temporary_manifest, list):
            raise ValueError("temporary manifest root is not a list")
        _validate_manifest(temporary_manifest, domains, expectations)
        if _sha256(temporary_path) != manifest_sha256:
            raise ValueError("temporary manifest SHA256 mismatch")

    published_manifest_sha = _write_with_sha(
        manifest_path,
        manifest_content,
        validate_manifest_temporary,
    )
    published_metadata_sha = _write_with_sha(metadata_path, metadata_content)
    published_audit_sha = _write_with_sha(audit_path, audit_content)
    if published_manifest_sha != manifest_sha256 or published_metadata_sha != metadata_sha256:
        raise AssertionError("published artifact SHA256 mismatch")
    return {
        **audit,
        "audit_sha256": published_audit_sha,
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "audit_path": str(audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--official-list", required=True, type=Path)
    parser.add_argument("--source-inventory", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    payload_group = parser.add_mutually_exclusive_group(required=True)
    payload_group.add_argument("--payload-hash-sidecar-dir", type=Path)
    payload_group.add_argument("--rehash-payloads", action="store_true")
    parser.add_argument("--expected-hash-script-sha256")
    parser.add_argument("--rehash-journal-dir", type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--generating-commit")
    args = parser.parse_args()
    report = build_and_publish(
        root=args.root,
        official_list=args.official_list,
        source_inventory=args.source_inventory,
        source_revision=args.source_revision,
        payload_hash_sidecar_dir=args.payload_hash_sidecar_dir,
        expected_hash_script_sha256=args.expected_hash_script_sha256,
        rehash_payloads=args.rehash_payloads,
        manifest_output=args.manifest_output,
        metadata_output=args.metadata_output,
        audit_output=args.audit_output,
        generating_commit=args.generating_commit,
        rehash_journal_dir=args.rehash_journal_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
