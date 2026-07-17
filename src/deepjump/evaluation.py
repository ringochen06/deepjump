"""Fail-closed helpers for reproducible DeepJump evaluation panels."""

from __future__ import annotations

import hashlib
import hmac
from numbers import Integral
from pathlib import Path
from typing import Iterable, Sequence


def domain_id(path: str | Path) -> str:
    """Return the mdCATH domain identifier encoded in a dataset filename."""
    name = Path(path).name
    prefix = "mdcath_dataset_"
    suffix = ".h5"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"not an mdCATH dataset filename: {name}")
    value = name[len(prefix):-len(suffix)]
    if not value:
        raise ValueError(f"empty mdCATH domain identifier in {name}")
    return value


def require_single_delta(value: object) -> int:
    """Return one positive integer delta, rejecting silent multi-delta mixing."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "evaluation requires exactly one delta_frames value; "
                f"got {value!r}. Evaluate each delta in a separate report."
            )
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"delta_frames must be a positive integer, got {value!r}")
    delta = int(value)
    if delta <= 0:
        raise ValueError(f"delta_frames must be positive, got {delta}")
    return delta


def load_frozen_domain_ids(path: str | Path, expected_sha256: str) -> tuple[list[str], str]:
    """Load an exact domain panel and verify its byte-level SHA256 identity."""
    domain_path = Path(path).expanduser()
    raw = domain_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected domain-list SHA256 must be 64 lowercase hex characters")
    if not hmac.compare_digest(digest, expected):
        raise ValueError(
            f"domain-list SHA256 mismatch for {domain_path}: {digest} != {expected}"
        )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"domain list is not UTF-8: {domain_path}") from exc
    ids = [line.strip() for line in lines if line.strip()]
    if not ids:
        raise ValueError(f"domain list is empty: {domain_path}")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"domain list contains duplicate identifiers: {duplicates}")
    return ids, digest


def resolve_frozen_domains(
    discovered: Iterable[str | Path], domain_ids: Sequence[str]
) -> list[Path]:
    """Resolve a frozen ordered ID list against discovered HDF5 files."""
    by_id: dict[str, Path] = {}
    duplicate_files: list[str] = []
    for candidate in discovered:
        path = Path(candidate)
        identifier = domain_id(path)
        if identifier in by_id:
            duplicate_files.append(identifier)
        by_id[identifier] = path
    if duplicate_files:
        raise ValueError(
            "multiple HDF5 files resolve to the same domain identifiers: "
            f"{sorted(set(duplicate_files))}"
        )
    missing = [identifier for identifier in domain_ids if identifier not in by_id]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frozen evaluation domains are missing: {missing[:10]}"
        )
    return [by_id[identifier] for identifier in domain_ids]


def reference_transition_deltas(values, delta: int):
    """Return observed increments separated by the checkpoint's physical delta."""
    delta = require_single_delta(delta)
    if len(values) <= delta:
        raise ValueError(
            f"need more than {delta} reference frames for delta={delta}; got {len(values)}"
        )
    return values[delta:] - values[:-delta]
