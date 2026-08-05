"""Fail-closed local ledger for one-time reserved-panel authorizations.

The ledger prevents accidental or automated replay on one persistent filesystem.
It is not tamper-proof against root and is not a cross-instance transactional
service; global one-time semantics require an external conditional-create API.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


CLAIM_SCHEMA = "deepjump.reserved_evaluation_consumption_claim.v1"
_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def _ledger_root(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("authorization must bind a consumption ledger root")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("consumption ledger root must be absolute")
    if path.resolve() != path:
        raise ValueError("consumption ledger root must be canonical and non-symlinked")
    return path


def _open_ledger_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("cannot open bound consumption ledger root") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("consumption ledger root is not a directory")
    return descriptor


def _claim_payload(
    authorization: dict,
    authorization_sha256: str,
    *,
    runtime_probe_output: str | Path,
    output: str | Path,
) -> dict:
    authorization_sha256 = _require_sha256(
        authorization_sha256, "authorization SHA256"
    )
    if authorization.get("sha256") != authorization_sha256:
        raise ValueError("verified authorization SHA256 binding mismatch")
    authorization_id = authorization.get("authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("authorization_id must be a non-empty string")
    return {
        "schema": CLAIM_SCHEMA,
        "authorization_id": authorization_id,
        "authorization_sha256": authorization_sha256,
        "phase": authorization["phase"],
        "checkpoint_sha256": authorization["checkpoint_sha256"],
        "checkpoint_step": authorization["checkpoint_step"],
        "full_training_contract_sha256": authorization[
            "full_training_contract_sha256"
        ],
        "panel_name": authorization["panel_name"],
        "panel_sha256": authorization["panel_sha256"],
        "runtime_probe_output": str(Path(runtime_probe_output).expanduser().resolve()),
        "output": str(Path(output).expanduser().resolve()),
    }


def claim_reserved_evaluation(
    authorization: dict,
    authorization_sha256: str,
    *,
    runtime_probe_output: str | Path,
    output: str | Path,
) -> dict:
    """Atomically consume one authorization before any reserved data is opened.

    The final claim path is created directly with ``O_EXCL``. Any later failure
    leaves the claim in place and therefore burns the authorization fail-closed.
    """

    root = _ledger_root(authorization.get("consumption_ledger_root"))
    payload = _claim_payload(
        authorization,
        authorization_sha256,
        runtime_probe_output=runtime_probe_output,
        output=output,
    )
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    filename = f"{authorization_sha256}.claim.json"
    directory_fd = _open_ledger_directory(root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError as exc:
            raise FileExistsError("reserved evaluation authorization was already consumed") from exc
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    digest = hashlib.sha256(content).hexdigest()
    return {
        **payload,
        "path": str(root / filename),
        "sha256": digest,
    }


def verify_reserved_evaluation_claim(
    authorization: dict,
    authorization_sha256: str,
    expected_claim: object,
    *,
    runtime_probe_output: str | Path,
    output: str | Path,
) -> dict:
    """Read back the immutable claim from the authorization-bound ledger."""

    authorization_sha256 = _require_sha256(
        authorization_sha256, "authorization SHA256"
    )
    if not isinstance(expected_claim, dict):
        raise ValueError("evaluator result has no consumption claim")
    expected_sha256 = _require_sha256(
        expected_claim.get("sha256"), "consumption claim SHA256"
    )
    root = _ledger_root(authorization.get("consumption_ledger_root"))
    filename = f"{authorization_sha256}.claim.json"
    directory_fd = _open_ledger_directory(root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(filename, flags, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError("cannot open reserved evaluation consumption claim") from exc
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after = os.fstat(handle.fileno())
    finally:
        os.close(directory_fd)
    if not stat.S_ISREG(before.st_mode) or (
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
        raise ValueError("consumption claim changed while it was being read")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("consumption claim SHA256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("consumption claim is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("consumption claim must be a JSON object")
    enriched = {
        **payload,
        "path": str(root / filename),
        "sha256": expected_sha256,
    }
    if enriched != expected_claim:
        raise ValueError("evaluator result consumption claim binding mismatch")
    static = _claim_payload(
        authorization,
        authorization_sha256,
        runtime_probe_output=runtime_probe_output,
        output=output,
    )
    if payload != static:
        raise ValueError("consumption claim does not bind the exact authorization")
    return enriched
