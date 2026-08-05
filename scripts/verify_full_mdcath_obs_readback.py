#!/usr/bin/env python
"""Stream and verify every full-mdCATH object from an OBS corpus prefix.

The verifier keeps an immutable per-object journal so an interrupted 3.6 TB
readback resumes without trusting partially downloaded bytes.  Credentials are
left entirely to ``obsutil``; this process neither reads nor reports them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


RECORD_SCHEMA = "deepjump.full_mdcath_obs_readback_record.v1"
SESSION_SCHEMA = "deepjump.full_mdcath_obs_readback_session.v1"
COMPLETION_SCHEMA = "deepjump.full_mdcath_obs_readback_completion.v1"
_FILE_RE = re.compile(r"^mdcath_dataset_([A-Za-z0-9]+)\.h5$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open {label} as a regular non-symlink file") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        raw = handle.read()
        after = os.fstat(handle.fileno())
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while it was read")
    return raw


def _hash_regular_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("OBS readback is not a regular file")
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("OBS readback changed while it was hashed")
    return before.st_size, digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _exclusive_publish(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace journal entry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_manifest(path: Path, expected_count: int, expected_bytes: int) -> tuple[list[dict], str]:
    raw = _read_regular_bytes(path, "full-mdCATH manifest")
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        rows = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("manifest object count mismatch")
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest row is not an object")
        filename = row.get("file")
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(filename, str) or _FILE_RE.fullmatch(filename) is None:
            raise ValueError("manifest contains an invalid corpus filename")
        if filename in seen:
            raise ValueError("manifest contains duplicate corpus filenames")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("manifest contains an invalid object size")
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            raise ValueError("manifest contains an invalid object SHA256")
        seen.add(filename)
        total_bytes += size
    if total_bytes != expected_bytes:
        raise ValueError("manifest corpus byte count mismatch")
    return rows, manifest_sha256


def verify_obs_readback(
    *,
    manifest: str | Path,
    obs_prefix: str,
    work_dir: str | Path,
    expected_count: int,
    expected_bytes: int,
    obsutil: str = "obsutil",
) -> dict:
    if not obs_prefix.startswith("obs://") or obs_prefix.endswith("/"):
        raise ValueError("OBS corpus prefix must be an obs:// URI without a trailing slash")
    if expected_count <= 0 or expected_bytes <= 0:
        raise ValueError("expected corpus count and bytes must be positive")
    manifest_rows, manifest_sha256 = _load_manifest(
        Path(manifest), expected_count, expected_bytes
    )
    work = Path(work_dir).expanduser().absolute()
    if work.is_symlink():
        raise ValueError("readback work directory must not be a symlink")
    work.mkdir(parents=True, exist_ok=True)
    if not work.is_dir():
        raise ValueError("readback work directory is not a directory")
    records = work / "records"
    temporary = work / "temporary"
    for directory in (records, temporary):
        if directory.is_symlink():
            raise ValueError("readback journal directory must not be a symlink")
        directory.mkdir(mode=0o700, exist_ok=True)
    unexpected_root_entries = {
        path.name for path in work.iterdir()
    } - {"session.json", "records", "temporary", "completion.json"}
    if unexpected_root_entries:
        raise ValueError("readback journal root exact inventory mismatch")

    session = {
        "schema": SESSION_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "obs_prefix": obs_prefix,
        "objects": expected_count,
        "bytes": expected_bytes,
        "formal_training_authorized": False,
    }
    session_path = work / "session.json"
    if session_path.exists() or session_path.is_symlink():
        existing_session = json.loads(
            _read_regular_bytes(session_path, "readback session")
        )
        if existing_session != session:
            raise ValueError("readback session identity mismatch")
    else:
        _exclusive_publish(session_path, _json_bytes(session))

    expected_record_names: set[str] = set()
    resumed = 0
    created = 0
    for index, row in enumerate(manifest_rows, start=1):
        filename = row["file"]
        record_path = records / f"{filename}.json"
        expected_record_names.add(record_path.name)
        identity = {
            "schema": RECORD_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "obs_object": f"{obs_prefix}/{filename}",
            "file": filename,
            "size": row["size"],
            "sha256": row["sha256"],
        }
        record_exists = record_path.exists() or record_path.is_symlink()
        if record_exists:
            existing = json.loads(_read_regular_bytes(record_path, "readback record"))
            if existing != identity:
                raise ValueError(f"readback journal identity mismatch: {filename}")

        # A local record is only a progress record.  It is never accepted as
        # evidence for the current remote object: every invocation obtains and
        # hashes the object from OBS again, including completed resumptions.
        descriptor, temporary_name = tempfile.mkstemp(prefix="obs-readback-", dir=temporary)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        try:
            subprocess.run(
                [obsutil, "cp", identity["obs_object"], str(temporary_path)],
                check=True,
            )
            size, digest = _hash_regular_file(temporary_path)
            if size != row["size"] or digest != row["sha256"]:
                raise ValueError(f"OBS content readback mismatch: {filename}")
            if record_exists:
                resumed += 1
            else:
                _exclusive_publish(record_path, _json_bytes(identity))
                created += 1
        finally:
            temporary_path.unlink(missing_ok=True)
        if index == 1 or index == expected_count or index % 100 == 0:
            print(
                json.dumps({"verified": index, "total": expected_count, "file": filename}),
                flush=True,
            )

    record_entries = list(records.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in record_entries):
        raise ValueError("readback journal contains a non-regular entry")
    actual_record_names = {path.name for path in record_entries}
    if actual_record_names != expected_record_names:
        raise ValueError("readback journal exact inventory mismatch")
    if list(temporary.iterdir()):
        raise ValueError("readback temporary directory is not empty")
    record_inventory = []
    for name in sorted(expected_record_names):
        raw = _read_regular_bytes(records / name, "readback record inventory")
        record_inventory.append(
            {"path": f"records/{name}", "sha256": hashlib.sha256(raw).hexdigest()}
        )
    record_inventory_sha256 = hashlib.sha256(
        json.dumps(record_inventory, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "PASS_FULL_MDCATH_OBS_CONTENT_READBACK",
        "manifest_sha256": manifest_sha256,
        "obs_prefix": obs_prefix,
        "objects": expected_count,
        "bytes": expected_bytes,
        "record_inventory_sha256": record_inventory_sha256,
        "records_resumed": resumed,
        "records_created": created,
        "formal_training_authorized": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    completion_path = work / "completion.json"
    if completion_path.exists() or completion_path.is_symlink():
        previous = json.loads(_read_regular_bytes(completion_path, "readback completion"))
        stable = {key: value for key, value in completion.items() if key not in {
            "records_resumed", "records_created", "completed_at"
        }}
        previous_stable = {key: value for key, value in previous.items() if key not in {
            "records_resumed", "records_created", "completed_at"
        }}
        if previous_stable != stable:
            raise ValueError("readback completion identity mismatch")
        root_entries = {path.name for path in work.iterdir()}
        if root_entries != {"session.json", "records", "temporary", "completion.json"}:
            raise ValueError("readback journal root exact inventory mismatch")
        return previous
    _exclusive_publish(completion_path, _json_bytes(completion))
    root_entries = {path.name for path in work.iterdir()}
    if root_entries != {"session.json", "records", "temporary", "completion.json"}:
        raise ValueError("readback journal root exact inventory mismatch")
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--obs-prefix", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--obsutil", default="obsutil")
    args = parser.parse_args()
    report = verify_obs_readback(
        manifest=args.manifest,
        obs_prefix=args.obs_prefix,
        work_dir=args.work_dir,
        expected_count=args.expected_count,
        expected_bytes=args.expected_bytes,
        obsutil=args.obsutil,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
