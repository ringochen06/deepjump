#!/usr/bin/env python
"""Verify an OBS audit readback against an exact, fail-closed file tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from deepjump.data_contract import _read_regular_bytes


_HEX = frozenset("0123456789abcdef")
_AUDITED_ROOTS = frozenset({"configs", "evidence", "stages"})


def _parse_manifest(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("audit manifest is not valid UTF-8") from exc
    entries: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("audit manifest contains a malformed line")
        digest, name = line[:64], line[66:]
        if set(digest) - _HEX:
            raise ValueError("audit manifest contains an invalid SHA256")
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in _AUDITED_ROOTS
        ):
            raise ValueError("audit manifest contains an out-of-scope path")
        normalized = path.as_posix()
        if normalized != name or normalized in entries:
            raise ValueError("audit manifest path is non-canonical or duplicated")
        entries[normalized] = digest
    if not entries:
        raise ValueError("audit manifest must contain at least one audited file")
    return entries


def _actual_files(root: Path) -> set[str]:
    actual: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"audit readback contains a symlink directory: {path}")
        for name in files:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"audit readback contains a symlink file: {path}")
            actual.add(path.relative_to(root).as_posix())
    return actual


def verify_audit_readback(
    root: Path,
    *,
    manifest_relative: str = "audit_sha256.txt",
    allowed_relative: tuple[str, ...] = (),
) -> dict:
    configured_root = root.expanduser()
    if configured_root.is_symlink():
        raise ValueError("audit readback root must be a real directory")
    root = configured_root.resolve()
    if not root.is_dir():
        raise ValueError("audit readback root must be a real directory")
    manifest_parts = PurePosixPath(manifest_relative)
    if (
        not manifest_relative
        or manifest_parts.is_absolute()
        or ".." in manifest_parts.parts
        or manifest_parts.as_posix() != manifest_relative
    ):
        raise ValueError("audit manifest relative path is invalid")
    manifest_path = root / manifest_relative
    manifest_raw = _read_regular_bytes(manifest_path, "audit manifest")
    entries = _parse_manifest(manifest_raw)

    allowed: set[str] = set()
    for value in allowed_relative:
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError("allowed readback path is invalid")
        allowed.add(value)
    expected = set(entries) | {manifest_relative} | allowed
    actual = _actual_files(root)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"audit readback exact-set mismatch: missing={missing}, extra={extra}")

    for relative, expected_sha256 in entries.items():
        raw = _read_regular_bytes(root / relative, f"audited file {relative}")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"audited file SHA256 mismatch: {relative}")

    return {
        "status": "PASS_EXACT_AUDIT_READBACK",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_entries": len(entries),
        "allowed_operational_files": sorted(allowed),
        "exact_files": len(actual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest-relative", default="audit_sha256.txt")
    parser.add_argument("--allow-relative", action="append", default=[])
    args = parser.parse_args()
    report = verify_audit_readback(
        args.root,
        manifest_relative=args.manifest_relative,
        allowed_relative=tuple(args.allow_relative),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
