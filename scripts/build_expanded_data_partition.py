#!/usr/bin/env python
"""Build a fail-closed train/held-out partition from sealed domain lists.

This tool operates only on domain identifiers.  It does not open mdCATH HDF5
files or inspect evaluation results.  Every input list must be accompanied by
an expected SHA256, so a future sealed panel can be supplied by path without
embedding its contents in code or configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCHEMA = "deepjump.expanded_data_partition.v1"
STATUS = "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION"
PANEL_REGISTRY_SCHEMA = "deepjump.full_mdcath_evaluation_exclusion_registry.v1"
EXPECTED_SOURCE_REVISION = "5e3ed8aec62b689e01751db16275fdcdbc39e47f"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9]+$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_REGISTRY_FIELDS = {
    "schema",
    "source_revision",
    "official_domain_list_sha256",
    "panel_contract_sha256",
    "panels",
}
_REGISTRY_PANEL_FIELDS = {"name", "path", "sha256", "domains", "role"}


@dataclass(frozen=True)
class PanelSpec:
    name: str
    path: Path
    expected_sha256: str


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
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed while it was being read")
    return raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path, str(path))).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")


def _resolve_regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _load_list(path: Path, expected_sha256: str, label: str) -> tuple[list[str], str]:
    _validate_sha256(expected_sha256, f"{label} expected SHA256")
    raw = _read_regular_bytes(path, label)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise ValueError(f"{label} must be non-empty and end with a newline")
    domains: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line != line.strip():
            raise ValueError(f"{label} has a blank or padded line at {line_number}")
        if not _DOMAIN_RE.fullmatch(line):
            raise ValueError(f"{label} has an invalid domain at line {line_number}: {line!r}")
        domains.append(line)
    duplicates = sorted(domain for domain, count in Counter(domains).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate domains: {duplicates[:10]}")
    return domains, actual_sha256


def _canonical_sha256(domains: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{domain}\n" for domain in domains).encode()).hexdigest()


def _panel_contract_sha256(panel_audits: Sequence[dict]) -> str:
    identity = {
        "schema": SCHEMA,
        "held_out_panels": sorted(
            (
                {key: panel[key] for key in ("name", "sha256", "domains")}
                for panel in panel_audits
            ),
            key=lambda panel: panel["name"],
        ),
    }
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _load_panel_registry(
    registry: str | Path,
    expected_registry_sha256: str,
    official_sha256: str,
) -> tuple[list[PanelSpec], str, dict]:
    registry_path = _resolve_regular_file(registry, "panel registry")
    _validate_sha256(expected_registry_sha256, "expected panel registry SHA256")
    registry_raw = _read_regular_bytes(registry_path, "panel registry")
    actual_registry_sha256 = hashlib.sha256(registry_raw).hexdigest()
    if actual_registry_sha256 != expected_registry_sha256:
        raise ValueError(
            "panel registry SHA256 mismatch: "
            f"{actual_registry_sha256} != {expected_registry_sha256}"
        )
    try:
        payload = json.loads(registry_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("panel registry is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _REGISTRY_FIELDS:
        raise ValueError("panel registry fields mismatch")
    if payload["schema"] != PANEL_REGISTRY_SCHEMA:
        raise ValueError("panel registry schema mismatch")
    if payload["source_revision"] != EXPECTED_SOURCE_REVISION:
        raise ValueError("panel registry source revision mismatch")
    if payload["official_domain_list_sha256"] != official_sha256:
        raise ValueError("panel registry official domain list SHA256 mismatch")
    panel_contract_sha256 = payload["panel_contract_sha256"]
    _validate_sha256(panel_contract_sha256, "panel registry contract SHA256")
    rows = payload["panels"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("panel registry must contain at least one panel")

    panels: list[PanelSpec] = []
    expected_audits: list[dict] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _REGISTRY_PANEL_FIELDS:
            raise ValueError("panel registry entry fields mismatch")
        if not _NAME_RE.fullmatch(row["name"]):
            raise ValueError(f"invalid registry panel name: {row['name']!r}")
        if not _ROLE_RE.fullmatch(row["role"]):
            raise ValueError(f"invalid registry panel role: {row['role']!r}")
        if type(row["domains"]) is not int or row["domains"] <= 0:
            raise ValueError("registry panel domains must be a positive integer")
        _validate_sha256(row["sha256"], f"registry panel {row['name']} SHA256")
        relative_path = Path(row["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("registry panel path must be relative without parent traversal")
        panel_path = registry_path.parent / relative_path
        panels.append(PanelSpec(row["name"], panel_path, row["sha256"]))
        expected_audits.append(
            {
                "name": row["name"],
                "sha256": row["sha256"],
                "domains": row["domains"],
            }
        )
    if _panel_contract_sha256(expected_audits) != panel_contract_sha256:
        raise ValueError("panel registry canonical contract SHA256 mismatch")
    identity = {
        "path": str(registry_path),
        "sha256": actual_registry_sha256,
        "schema": PANEL_REGISTRY_SCHEMA,
        "source_revision": EXPECTED_SOURCE_REVISION,
        "panel_contract_sha256": panel_contract_sha256,
    }
    return panels, panel_contract_sha256, identity


def build_partition(
    official_list: str | Path,
    official_sha256: str,
    panels: Sequence[PanelSpec],
    expected_panel_contract_sha256: str,
) -> tuple[list[str], dict]:
    """Return official-order train domains and a domain-level partition audit."""

    official_path = _resolve_regular_file(official_list, "official list")
    official_domains, official_actual_sha256 = _load_list(
        official_path, official_sha256, "official list"
    )
    official_set = set(official_domains)

    if not panels:
        raise ValueError("at least one held-out panel is required")
    _validate_sha256(
        expected_panel_contract_sha256, "expected panel contract SHA256"
    )
    panel_names: set[str] = set()
    panel_paths: set[Path] = set()
    owner_by_domain: dict[str, str] = {}
    panel_audits: list[dict] = []

    for panel in panels:
        if not _NAME_RE.fullmatch(panel.name):
            raise ValueError(
                f"invalid panel name {panel.name!r}; use lowercase letters, digits, '_' or '-'"
            )
        if panel.name in panel_names:
            raise ValueError(f"duplicate panel name: {panel.name}")
        panel_names.add(panel.name)

        panel_path = _resolve_regular_file(panel.path, f"panel {panel.name}")
        if panel_path == official_path:
            raise ValueError(f"panel {panel.name} aliases the official list")
        if panel_path in panel_paths:
            raise ValueError(f"panel {panel.name} aliases another panel path")
        panel_paths.add(panel_path)

        panel_domains, panel_sha256 = _load_list(
            panel_path, panel.expected_sha256, f"panel {panel.name}"
        )
        unknown = sorted(set(panel_domains) - official_set)
        if unknown:
            raise ValueError(
                f"panel {panel.name} contains domains absent from the official list: {unknown[:10]}"
            )
        overlaps = sorted(domain for domain in panel_domains if domain in owner_by_domain)
        if overlaps:
            details = [f"{domain}:{owner_by_domain[domain]}" for domain in overlaps[:10]]
            raise ValueError(
                f"panel {panel.name} overlaps another held-out panel: {details}"
            )
        owner_by_domain.update({domain: panel.name for domain in panel_domains})
        panel_audits.append(
            {
                "name": panel.name,
                "path": str(panel_path),
                "sha256": panel_sha256,
                "domains": len(panel_domains),
            }
        )

    actual_panel_contract_sha256 = _panel_contract_sha256(panel_audits)
    if actual_panel_contract_sha256 != expected_panel_contract_sha256:
        raise ValueError(
            "held-out panel contract SHA256 mismatch: "
            f"{actual_panel_contract_sha256} != {expected_panel_contract_sha256}"
        )

    train_domains = [domain for domain in official_domains if domain not in owner_by_domain]
    excluded_domains = [domain for domain in official_domains if domain in owner_by_domain]
    if not train_domains:
        raise ValueError("held-out panels exclude every official domain")

    assignments = [
        {
            "domain": domain,
            "partition": "excluded" if domain in owner_by_domain else "train_eligible",
            "panel": owner_by_domain.get(domain),
        }
        for domain in official_domains
    ]
    report = {
        "schema": SCHEMA,
        "status": STATUS,
        "official": {
            "path": str(official_path),
            "sha256": official_actual_sha256,
            "domains": len(official_domains),
        },
        "held_out_panels": panel_audits,
        "held_out_panel_contract_sha256": actual_panel_contract_sha256,
        "partition": {
            "official_domains": len(official_domains),
            "excluded_domains": len(excluded_domains),
            "train_eligible_domains": len(train_domains),
            "excluded_union_sha256": _canonical_sha256(excluded_domains),
            "train_eligible_sha256": _canonical_sha256(train_domains),
            "panels_are_pairwise_disjoint": True,
            "all_excluded_domains_are_official": True,
        },
        "domain_assignments": assignments,
    }
    contract_identity = {
        "schema": SCHEMA,
        "official": {
            "sha256": official_actual_sha256,
            "domains": len(official_domains),
        },
        "held_out_panels": sorted(
            (
                {key: panel[key] for key in ("name", "sha256", "domains")}
                for panel in panel_audits
            ),
            key=lambda panel: panel["name"],
        ),
        "held_out_panel_contract_sha256": actual_panel_contract_sha256,
        "partition": report["partition"],
        "domain_assignments": assignments,
    }
    report["partition_contract_sha256"] = hashlib.sha256(
        json.dumps(contract_identity, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return train_domains, report


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_partition(
    official_list: str | Path,
    official_sha256: str,
    panels: Sequence[PanelSpec],
    expected_panel_contract_sha256: str,
    train_output: str | Path,
    audit_output: str | Path,
    panel_registry_identity: dict | None = None,
) -> dict:
    train_path = Path(train_output).expanduser().resolve()
    audit_path = Path(audit_output).expanduser().resolve()
    if train_path == audit_path:
        raise ValueError("train and audit outputs must be different paths")
    if train_path.exists() or train_path.is_symlink():
        raise ValueError(f"refusing to overwrite output: {train_path}")
    if audit_path.exists() or audit_path.is_symlink():
        raise ValueError(f"refusing to overwrite output: {audit_path}")

    train_domains, report = build_partition(
        official_list,
        official_sha256,
        panels,
        expected_panel_contract_sha256,
    )
    train_content = "".join(f"{domain}\n" for domain in train_domains)
    report["outputs"] = {
        "train_eligible_list": {
            "path": str(train_path),
            "sha256": hashlib.sha256(train_content.encode()).hexdigest(),
            "domains": len(train_domains),
        },
        "audit": {"path": str(audit_path)},
    }
    if panel_registry_identity is not None:
        report["panel_registry"] = panel_registry_identity
    audit_content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _atomic_write(train_path, train_content)
    _atomic_write(audit_path, audit_content)
    return report


def _parse_panel(value: str) -> PanelSpec:
    name, first_separator, remainder = value.partition("=")
    path, last_separator, expected_sha256 = remainder.rpartition("=")
    if not first_separator or not last_separator or not name or not path or not expected_sha256:
        raise argparse.ArgumentTypeError("panel must be NAME=PATH=EXPECTED_SHA256")
    return PanelSpec(name, Path(path), expected_sha256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-list", required=True)
    parser.add_argument("--official-sha256", required=True)
    parser.add_argument(
        "--exclude-panel",
        action="append",
        type=_parse_panel,
        default=[],
        metavar="NAME=PATH=EXPECTED_SHA256",
        help="sealed held-out panel; repeat for every historical or future panel",
    )
    parser.add_argument(
        "--expected-panel-contract-sha256",
        help="frozen SHA256 of the exact canonical held-out panel name/SHA/count set",
    )
    parser.add_argument("--panel-registry")
    parser.add_argument("--expected-panel-registry-sha256")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    registry_identity = None
    if args.panel_registry is not None:
        if args.exclude_panel or args.expected_panel_contract_sha256 is not None:
            parser.error(
                "--panel-registry cannot be combined with --exclude-panel or "
                "--expected-panel-contract-sha256"
            )
        if args.expected_panel_registry_sha256 is None:
            parser.error(
                "--expected-panel-registry-sha256 is required with --panel-registry"
            )
        panels, panel_contract_sha256, registry_identity = _load_panel_registry(
            args.panel_registry,
            args.expected_panel_registry_sha256,
            args.official_sha256,
        )
    else:
        if args.expected_panel_registry_sha256 is not None:
            parser.error(
                "--expected-panel-registry-sha256 requires --panel-registry"
            )
        if args.expected_panel_contract_sha256 is None:
            parser.error(
                "--expected-panel-contract-sha256 is required with --exclude-panel"
            )
        panels = args.exclude_panel
        panel_contract_sha256 = args.expected_panel_contract_sha256
    report = write_partition(
        args.official_list,
        args.official_sha256,
        panels,
        panel_contract_sha256,
        args.train_output,
        args.audit_output,
        registry_identity,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
