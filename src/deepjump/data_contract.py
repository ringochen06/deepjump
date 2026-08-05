"""Fail-closed verification for full-mdCATH training data contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "deepjump.full_training_data_contract.v1"
AUDIT_SCHEMA = "deepjump.full_mdcath_audit.v1"
AUDIT_STATUS = "PASS_FULL_LIVE_PAYLOAD_REHASH"
PARTITION_SCHEMA = "deepjump.expanded_data_partition.v1"
PARTITION_STATUS = "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION"
REGISTRY_SCHEMA = "deepjump.full_mdcath_evaluation_exclusion_registry.v1"
EXPECTED_SOURCE_REVISION = "5e3ed8aec62b689e01751db16275fdcdbc39e47f"
EXPECTED_OFFICIAL_SHA256 = (
    "295c6da1c9f8846a1ea3993eca12a3232d16a2b3a4b0d8791c7c45392186709b"
)
EXPECTED_PANEL_REGISTRY_SHA256 = (
    "65f14cb45c1af84ca6a7e97affe6974232fd3ec12da69a875e9a089525943097"
)
EXPECTED_PANEL_CONTRACT_SHA256 = (
    "f5a772daa77a1f3118cc2e9151363d6a5a9f9737634b4a45561f09c302db2865"
)
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "2e6e3602a0858aaafc849cfa7cc1ee7e076736cb15335d7914726898f06f6cdf"
)
EXPECTED_DOMAINS = 5_398
EXPECTED_EXCLUDED_DOMAINS = 180
EXPECTED_TRAIN_DOMAINS = 5_218
EXPECTED_H5_BYTES = 3_613_998_101_757
EXPECTED_TRAJECTORIES = 134_950
_HEX = frozenset("0123456789abcdef")
_CONTRACT_FIELDS = {"schema", "source_revision", "official_domain_list_sha256", "artifacts"}
TEMPERATURES = (320, 348, 379, 413, 450)
REPLICAS = (0, 1, 2, 3, 4)
_ARTIFACT_FIELDS = {
    "data_audit",
    "manifest",
    "official_list",
    "panel_registry",
    "partition_audit",
    "source_inventory",
    "staging_metadata",
    "train_list",
}
_ARTIFACT_ID_FIELDS = {"path", "sha256"}
_SOURCE_FIELDS = {
    "git_blob_sha1", "lfs_pointer_size", "lfs_sha256", "lfs_size",
    "path", "size", "xet_hash",
}
_MANIFEST_FIELDS = {
    "domain", "file", "local_fingerprint", "num_atoms", "num_residues",
    "sha256", "size", "trajectories",
}
_FINGERPRINT_FIELDS = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
_TRAJECTORY_FIELDS = {"temp", "replica", "num_frames"}
_AUDIT_FIELDS = {
    "all_coordinate_frames_finite_verified", "completed_at",
    "audit_script_sha256",
    "coordinate_finiteness_scope", "data_gate_passed", "domains",
    "expected_hash_script_sha256", "external_development_authorized",
    "finite_endpoint_frames_verified", "formal_training_authorized",
    "generating_commit", "h5_bytes", "h5_files",
    "hdf5_files_structurally_verified", "live_payload_bytes_rehashed",
    "manifest_sha256", "metadata_sha256", "official_domain_list_sha256",
    "payload_hash_sidecars_verified", "payload_hash_verification_mode",
    "payload_hashes_verified", "root", "schema", "second_seed_authorized",
    "payload_rehash_journal_completion_sha256", "payload_rehash_journal_dir",
    "payload_rehash_journal_session_sha256", "payload_rehash_records_created",
    "payload_rehash_records_resumed",
    "source_inventory_sha256", "source_repo", "source_revision", "status",
    "trajectories", "untouched_confirmation_authorized",
    "verified_local_inventory_sha256",
}
_METADATA_FIELDS = {
    "all_coordinate_frames_finite_verified", "coordinate_finiteness_scope",
    "audit_script_sha256",
    "created_at", "data_gate_passed", "domains", "expected_hash_script_sha256",
    "finite_endpoint_frames_verified", "generating_commit", "h5_bytes",
    "h5_files", "hdf5_files_structurally_verified",
    "live_payload_bytes_rehashed", "manifest_file", "manifest_sha256",
    "official_domain_list_file", "official_domain_list_sha256",
    "payload_hash_sidecars_verified", "payload_hash_verification_mode",
    "payload_hashes_verified", "schema", "selection_strategy", "source_inventory_file",
    "payload_rehash_journal_completion_sha256", "payload_rehash_journal_dir",
    "payload_rehash_journal_session_sha256", "payload_rehash_records_created",
    "payload_rehash_records_resumed",
    "source_inventory_sha256", "source_repo", "source_revision",
    "temperature_replica_grid", "trajectories", "verified_local_inventory_sha256",
}
_REHASH_IDENTITY_FIELDS = {
    "audit_script_sha256",
    "payload_rehash_journal_dir",
    "payload_rehash_journal_session_sha256",
    "payload_rehash_journal_completion_sha256",
    "payload_rehash_records_resumed",
    "payload_rehash_records_created",
}
_SOURCE_PATH = re.compile(r"^data/mdcath_dataset_([A-Za-z0-9]+)\.h5$")
_DOMAIN = re.compile(r"^[A-Za-z0-9]+$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ContractExpectations:
    source_revision: str = EXPECTED_SOURCE_REVISION
    official_sha256: str = EXPECTED_OFFICIAL_SHA256
    source_inventory_sha256: str = EXPECTED_SOURCE_INVENTORY_SHA256
    panel_registry_sha256: str = EXPECTED_PANEL_REGISTRY_SHA256
    panel_contract_sha256: str = EXPECTED_PANEL_CONTRACT_SHA256
    domains: int = EXPECTED_DOMAINS
    excluded_domains: int = EXPECTED_EXCLUDED_DOMAINS
    train_domains: int = EXPECTED_TRAIN_DOMAINS
    h5_bytes: int = EXPECTED_H5_BYTES
    trajectories: int = EXPECTED_TRAJECTORIES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, label: str) -> bytes:
    """Read one regular non-symlink inode for both hashing and semantic parsing."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open {label} as a regular non-symlink file: {path}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
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
        raise ValueError(f"{label} changed while it was being read: {path}")
    return raw


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _verify_rehash_journal_identity(
    payload: dict[str, Any], root: Path, expected_records: int
) -> None:
    for key in (
        "audit_script_sha256",
        "payload_rehash_journal_session_sha256",
        "payload_rehash_journal_completion_sha256",
    ):
        _validate_sha256(payload.get(key), key.replace("_", " "))
    resumed = payload.get("payload_rehash_records_resumed")
    created = payload.get("payload_rehash_records_created")
    if (
        isinstance(resumed, bool)
        or isinstance(created, bool)
        or not isinstance(resumed, int)
        or not isinstance(created, int)
        or resumed < 0
        or created < 0
        or resumed + created != expected_records
    ):
        raise ValueError("payload rehash journal record counts mismatch")
    journal_value = payload.get("payload_rehash_journal_dir")
    if not isinstance(journal_value, str) or not journal_value:
        raise ValueError("payload rehash journal directory is invalid")
    journal = Path(journal_value).expanduser()
    if not journal.is_absolute() or journal.is_symlink():
        raise ValueError("payload rehash journal must be an absolute non-symlink directory")
    try:
        journal.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("payload rehash journal must remain inside the data root") from exc
    if not journal.is_dir():
        raise ValueError("payload rehash journal directory is missing")
    for filename, key in (
        ("session.json", "payload_rehash_journal_session_sha256"),
        ("completion.json", "payload_rehash_journal_completion_sha256"),
    ):
        raw = _read_regular_bytes(journal / filename, f"payload rehash {filename}")
        if hashlib.sha256(raw).hexdigest() != payload[key]:
            raise ValueError(f"payload rehash {filename} SHA256 mismatch")


def _load_json_bytes(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _load_json(path: Path, label: str) -> Any:
    return _load_json_bytes(_read_regular_bytes(path, label), label)


def _resolve_artifact(
    contract_path: Path, row: object, label: str
) -> tuple[Path, str, bytes]:
    if not isinstance(row, dict) or set(row) != _ARTIFACT_ID_FIELDS:
        raise ValueError(f"{label} artifact identity fields mismatch")
    expected_sha256 = _validate_sha256(row["sha256"], f"{label} SHA256")
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path must be relative without parent traversal")
    path = _regular_file(contract_path.parent / relative, label)
    raw = _read_regular_bytes(path, label)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: {actual_sha256} != {expected_sha256}")
    return path, actual_sha256, raw


def _domain_from_manifest(entry: object) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("domain"), str):
        raise ValueError("manifest entry has no domain identity")
    return entry["domain"]


def _load_domain_list(path: Path, expected_sha256: str, expected_count: int, label: str) -> list[str]:
    raw = _read_regular_bytes(path, label)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise ValueError(f"{label} must be non-empty and newline terminated")
    domains = text.splitlines()
    if len(domains) != expected_count or len(set(domains)) != expected_count:
        raise ValueError(f"{label} count or uniqueness mismatch")
    return domains


def _canonical_domain_sha256(domains: list[str]) -> str:
    return hashlib.sha256("".join(f"{domain}\n" for domain in domains).encode()).hexdigest()


def _strict_domain_list(raw: bytes, expected_count: int, label: str) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    if not text or not text.endswith("\n") or "\r" in text:
        raise ValueError(f"{label} must be canonical LF-terminated text")
    domains = text.splitlines()
    if (
        len(domains) != expected_count
        or len(set(domains)) != expected_count
        or any(_DOMAIN.fullmatch(domain) is None for domain in domains)
    ):
        raise ValueError(f"{label} count, uniqueness, or domain syntax mismatch")
    return domains


def _load_source_inventory(raw: bytes, expectations: ContractExpectations) -> list[dict]:
    if hashlib.sha256(raw).hexdigest() != expectations.source_inventory_sha256:
        raise ValueError("source inventory is not the canonical frozen inventory")
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("source inventory must be canonical LF-terminated JSONL")
    rows: list[dict] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"source inventory row {line_number} is invalid") from exc
        if (
            not isinstance(row, dict)
            or set(row) != _SOURCE_FIELDS
            or (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            != line
        ):
            raise ValueError(f"source inventory row {line_number} is not canonical")
        match = _SOURCE_PATH.fullmatch(row.get("path", ""))
        if match is None:
            raise ValueError(f"source inventory row {line_number} has invalid path")
        if (
            type(row.get("size")) is not int
            or row["size"] <= 0
            or row.get("lfs_size") != row["size"]
            or type(row.get("lfs_pointer_size")) is not int
            or row["lfs_pointer_size"] <= 0
            or _SHA1.fullmatch(row.get("git_blob_sha1", "")) is None
        ):
            raise ValueError(f"source inventory row {line_number} has invalid size or SHA1")
        _validate_sha256(row.get("lfs_sha256"), f"source inventory row {line_number} LFS SHA256")
        _validate_sha256(row.get("xet_hash"), f"source inventory row {line_number} Xet SHA256")
        rows.append({**row, "domain": match.group(1)})
    paths = [row["path"] for row in rows]
    domains = [row["domain"] for row in rows]
    hashes = [row["lfs_sha256"] for row in rows]
    if (
        len(rows) != expectations.domains
        or paths != sorted(paths)
        or len(set(paths)) != expectations.domains
        or len(set(domains)) != expectations.domains
        or len(set(hashes)) != expectations.domains
        or sum(row["size"] for row in rows) != expectations.h5_bytes
    ):
        raise ValueError("source inventory count, order, uniqueness, or byte total mismatch")
    return rows


def _validate_manifest(
    manifest: object, source_rows: list[dict], expectations: ContractExpectations
) -> tuple[list[str], str]:
    if not isinstance(manifest, list) or len(manifest) != expectations.domains:
        raise ValueError("manifest domain count mismatch")
    expected_grid = [
        (temperature, replica)
        for temperature in TEMPERATURES
        for replica in REPLICAS
    ]
    inventory_digest = hashlib.sha256()
    domains: list[str] = []
    trajectories = 0
    for index, (entry, source) in enumerate(zip(manifest, source_rows, strict=True), 1):
        if not isinstance(entry, dict) or set(entry) != _MANIFEST_FIELDS:
            raise ValueError(f"manifest row {index} fields mismatch")
        domain = source["domain"]
        filename = f"mdcath_dataset_{domain}.h5"
        if (
            entry["domain"] != domain
            or entry["file"] != filename
            or entry["size"] != source["size"]
            or entry["sha256"] != source["lfs_sha256"]
            or type(entry["num_atoms"]) is not int
            or entry["num_atoms"] <= 0
            or type(entry["num_residues"]) is not int
            or entry["num_residues"] <= 0
        ):
            raise ValueError(f"manifest row {index} differs from source identity")
        fingerprint = entry["local_fingerprint"]
        if (
            not isinstance(fingerprint, dict)
            or set(fingerprint) != _FINGERPRINT_FIELDS
            or any(type(value) is not int or value < 0 for value in fingerprint.values())
            or fingerprint["size"] != entry["size"]
        ):
            raise ValueError(f"manifest row {index} fingerprint mismatch")
        rows = entry["trajectories"]
        if not isinstance(rows, list) or len(rows) != len(expected_grid):
            raise ValueError(f"manifest row {index} trajectory count mismatch")
        grid = []
        for trajectory in rows:
            if (
                not isinstance(trajectory, dict)
                or set(trajectory) != _TRAJECTORY_FIELDS
                or type(trajectory["temp"]) is not int
                or type(trajectory["replica"]) is not int
                or type(trajectory["num_frames"]) is not int
                or trajectory["num_frames"] <= 0
            ):
                raise ValueError(f"manifest row {index} has invalid trajectory")
            grid.append((trajectory["temp"], trajectory["replica"]))
        if grid != expected_grid:
            raise ValueError(f"manifest row {index} trajectory grid mismatch")
        trajectories += len(rows)
        domains.append(domain)
        inventory_digest.update(
            (
                f"{filename}\t{entry['size']}\t{domain}\t{entry['num_residues']}\t"
                f"{entry['num_atoms']}\t{len(rows)}\t{entry['sha256']}\n"
            ).encode()
        )
    if trajectories != expectations.trajectories:
        raise ValueError("manifest trajectory total mismatch")
    return domains, inventory_digest.hexdigest()


def _verify_live_payload_fingerprints(root: Path, manifest: list[dict]) -> None:
    """Bind the sealed manifest to the currently mounted payload inodes.

    The qualifying audit already performed the expensive full-payload rehash.  This
    verifier deliberately performs the cheaper final inode/fingerprint sweep on
    stable directory descriptors so a stale audit cannot qualify an empty,
    replaced, or symlinked live data root.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        root_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise ValueError(f"cannot open configured data root as a non-symlink directory: {root}") from exc
    try:
        try:
            data_fd = os.open("data", directory_flags, dir_fd=root_fd)
        except OSError as exc:
            raise ValueError("configured data root has no regular non-symlink data directory") from exc
        try:
            expected_names = {entry["file"] for entry in manifest}
            actual_names = {
                name for name in os.listdir(data_fd) if name.endswith(".h5")
            }
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)[:10]
                extra = sorted(actual_names - expected_names)[:10]
                raise ValueError(
                    f"live HDF5 file set differs from manifest: missing={missing}, extra={extra}"
                )
            for entry in manifest:
                try:
                    payload_fd = os.open(entry["file"], flags, dir_fd=data_fd)
                except OSError as exc:
                    raise ValueError(
                        f"cannot open live payload as a regular non-symlink file: {entry['file']}"
                    ) from exc
                try:
                    current = os.fstat(payload_fd)
                finally:
                    os.close(payload_fd)
                if not stat.S_ISREG(current.st_mode):
                    raise ValueError(f"live payload is not a regular file: {entry['file']}")
                fingerprint = {
                    "device": current.st_dev,
                    "inode": current.st_ino,
                    "size": current.st_size,
                    "mtime_ns": current.st_mtime_ns,
                    "ctime_ns": current.st_ctime_ns,
                }
                if fingerprint != entry["local_fingerprint"]:
                    raise ValueError(f"live payload fingerprint differs from manifest: {entry['file']}")
        finally:
            os.close(data_fd)
    finally:
        os.close(root_fd)


def _partition_contract_sha256(partition: dict) -> str:
    identity = {
        "schema": PARTITION_SCHEMA,
        "official": {
            "sha256": partition["official"]["sha256"],
            "domains": partition["official"]["domains"],
        },
        "held_out_panels": sorted(
            (
                {key: panel[key] for key in ("name", "sha256", "domains")}
                for panel in partition["held_out_panels"]
            ),
            key=lambda panel: panel["name"],
        ),
        "held_out_panel_contract_sha256": partition["held_out_panel_contract_sha256"],
        "partition": partition["partition"],
        "domain_assignments": partition["domain_assignments"],
    }
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def verify_full_training_data_contract(
    contract: str | Path,
    expected_contract_sha256: str,
    *,
    configured_root: str | Path,
    configured_manifest: str | Path,
    configured_domains_file: str | Path,
    expectations: ContractExpectations | None = None,
) -> dict:
    """Verify every data identity required before a full training process starts."""

    expectations = expectations or ContractExpectations()
    contract_path = _regular_file(Path(contract).expanduser(), "training data contract")
    expected_contract_sha256 = _validate_sha256(
        expected_contract_sha256, "training data contract SHA256"
    )
    contract_raw = _read_regular_bytes(contract_path, "training data contract")
    actual_contract_sha256 = hashlib.sha256(contract_raw).hexdigest()
    if actual_contract_sha256 != expected_contract_sha256:
        raise ValueError(
            "training data contract SHA256 mismatch: "
            f"{actual_contract_sha256} != {expected_contract_sha256}"
        )
    payload = _load_json_bytes(contract_raw, "training data contract")
    if not isinstance(payload, dict) or set(payload) != _CONTRACT_FIELDS:
        raise ValueError("training data contract fields mismatch")
    if payload["schema"] != CONTRACT_SCHEMA:
        raise ValueError("training data contract schema mismatch")
    if payload["source_revision"] != expectations.source_revision:
        raise ValueError("training data contract source revision mismatch")
    if payload["official_domain_list_sha256"] != expectations.official_sha256:
        raise ValueError("training data contract official domain SHA256 mismatch")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACT_FIELDS:
        raise ValueError("training data contract artifact set mismatch")
    resolved = {
        name: _resolve_artifact(contract_path, artifacts[name], name)
        for name in sorted(_ARTIFACT_FIELDS)
    }

    source_rows = _load_source_inventory(
        resolved["source_inventory"][2], expectations
    )
    official_domains = _strict_domain_list(
        resolved["official_list"][2], expectations.domains, "official domain list"
    )
    if resolved["official_list"][1] != expectations.official_sha256:
        raise ValueError("official domain list SHA256 mismatch")
    source_domains = [row["domain"] for row in source_rows]
    if official_domains != source_domains:
        raise ValueError("official domain list differs from canonical source inventory")

    audit_path = resolved["data_audit"][0]
    audit = _load_json_bytes(resolved["data_audit"][2], "data audit")
    required_audit = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "source_repo": "compsciencelab/mdCATH",
        "source_revision": expectations.source_revision,
        "source_inventory_sha256": expectations.source_inventory_sha256,
        "official_domain_list_sha256": expectations.official_sha256,
        "domains": expectations.domains,
        "h5_files": expectations.domains,
        "h5_bytes": expectations.h5_bytes,
        "trajectories": expectations.trajectories,
        "hdf5_files_structurally_verified": expectations.domains,
        "payload_hash_verification_mode": "full_rehash",
        "payload_hashes_verified": expectations.domains,
        "payload_hash_sidecars_verified": 0,
        "expected_hash_script_sha256": None,
        "data_gate_passed": True,
        "live_payload_bytes_rehashed": True,
        "all_coordinate_frames_finite_verified": False,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "finite_endpoint_frames_verified": expectations.trajectories * 2,
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    if (
        not isinstance(audit, dict)
        or set(audit) != _AUDIT_FIELDS
        or any(audit.get(key) != value for key, value in required_audit.items())
    ):
        raise ValueError("data audit is not a qualifying exact live-rehash audit")
    for key in ("manifest_sha256", "metadata_sha256", "verified_local_inventory_sha256"):
        _validate_sha256(audit.get(key), f"data audit {key}")
    for key in (
        "audit_script_sha256",
        "payload_rehash_journal_session_sha256",
        "payload_rehash_journal_completion_sha256",
    ):
        _validate_sha256(audit.get(key), f"data audit {key}")
    if _SHA1.fullmatch(audit.get("generating_commit", "")) is None:
        raise ValueError("data audit generating commit is invalid")
    if not isinstance(audit.get("completed_at"), str) or not audit["completed_at"]:
        raise ValueError("data audit completion time is invalid")

    manifest_path = resolved["manifest"][0]
    manifest = _load_json_bytes(resolved["manifest"][2], "manifest")
    manifest_domains, inventory_sha256 = _validate_manifest(
        manifest, source_rows, expectations
    )
    if manifest_domains != official_domains:
        raise ValueError("manifest does not exactly follow the official domain list")
    if audit.get("manifest_sha256") != resolved["manifest"][1]:
        raise ValueError("data audit does not bind the exact manifest")
    if audit["verified_local_inventory_sha256"] != inventory_sha256:
        raise ValueError("data audit verified inventory SHA256 mismatch")

    metadata = _load_json_bytes(resolved["staging_metadata"][2], "staging metadata")
    required_metadata = {
        "schema": "deepjump.full_mdcath_staging.v1",
        "selection_strategy": "official-full-5398",
        "source_repo": "compsciencelab/mdCATH",
        "source_revision": expectations.source_revision,
        "source_inventory_file": resolved["source_inventory"][0].name,
        "source_inventory_sha256": expectations.source_inventory_sha256,
        "official_domain_list_file": resolved["official_list"][0].name,
        "official_domain_list_sha256": expectations.official_sha256,
        "domains": expectations.domains,
        "h5_files": expectations.domains,
        "h5_bytes": expectations.h5_bytes,
        "trajectories": expectations.trajectories,
        "temperature_replica_grid": "5x5",
        "hdf5_files_structurally_verified": expectations.domains,
        "payload_hash_verification_mode": "full_rehash",
        "payload_hashes_verified": expectations.domains,
        "payload_hash_sidecars_verified": 0,
        "expected_hash_script_sha256": None,
        "data_gate_passed": True,
        "live_payload_bytes_rehashed": True,
        "all_coordinate_frames_finite_verified": False,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "finite_endpoint_frames_verified": expectations.trajectories * 2,
        "verified_local_inventory_sha256": inventory_sha256,
        "manifest_file": manifest_path.name,
        "manifest_sha256": resolved["manifest"][1],
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != _METADATA_FIELDS
        or any(metadata.get(key) != value for key, value in required_metadata.items())
        or resolved["staging_metadata"][1] != audit["metadata_sha256"]
        or metadata.get("generating_commit") != audit["generating_commit"]
        or metadata.get("created_at") != audit["completed_at"]
    ):
        raise ValueError("staging metadata does not match the qualifying audit")
    if any(metadata.get(key) != audit.get(key) for key in _REHASH_IDENTITY_FIELDS):
        raise ValueError("staging metadata and audit rehash journal identities differ")

    registry_path = resolved["panel_registry"][0]
    if resolved["panel_registry"][1] != expectations.panel_registry_sha256:
        raise ValueError("panel registry is not the preregistered five-panel registry")
    registry = _load_json_bytes(resolved["panel_registry"][2], "panel registry")
    if (
        not isinstance(registry, dict)
        or registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("source_revision") != expectations.source_revision
        or registry.get("official_domain_list_sha256") != expectations.official_sha256
        or registry.get("panel_contract_sha256") != expectations.panel_contract_sha256
        or sum(row.get("domains", 0) for row in registry.get("panels", []))
        != expectations.excluded_domains
    ):
        raise ValueError("panel registry identity or domain count mismatch")
    owner_by_domain: dict[str, str] = {}
    registry_panels: list[dict] = []
    for row in registry["panels"]:
        if not isinstance(row, dict) or set(row) != {"name", "path", "sha256", "domains", "role"}:
            raise ValueError("panel registry entry fields mismatch")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("panel registry path must be relative without parent traversal")
        panel_path = _regular_file(registry_path.parent / relative, f"panel {row['name']}")
        panel_domains = _load_domain_list(
            panel_path, row["sha256"], row["domains"], f"panel {row['name']}"
        )
        overlap = set(panel_domains) & set(owner_by_domain)
        if overlap:
            raise ValueError(f"panel registry contains overlapping domains: {sorted(overlap)[:10]}")
        owner_by_domain.update({domain: row["name"] for domain in panel_domains})
        registry_panels.append(
            {"name": row["name"], "sha256": row["sha256"], "domains": row["domains"]}
        )
    if len(owner_by_domain) != expectations.excluded_domains:
        raise ValueError("panel registry union is not exactly 180 domains")

    partition_path = resolved["partition_audit"][0]
    partition = _load_json_bytes(resolved["partition_audit"][2], "partition audit")
    partition_counts = partition.get("partition", {}) if isinstance(partition, dict) else {}
    if (
        not isinstance(partition, dict)
        or partition.get("schema") != PARTITION_SCHEMA
        or partition.get("status") != PARTITION_STATUS
        or partition.get("official", {}).get("sha256") != expectations.official_sha256
        or partition.get("official", {}).get("domains") != expectations.domains
        or partition.get("held_out_panel_contract_sha256") != expectations.panel_contract_sha256
        or partition.get("panel_registry", {}).get("sha256") != expectations.panel_registry_sha256
        or partition_counts.get("official_domains") != expectations.domains
        or partition_counts.get("excluded_domains") != expectations.excluded_domains
        or partition_counts.get("train_eligible_domains") != expectations.train_domains
        or partition_counts.get("panels_are_pairwise_disjoint") is not True
        or partition_counts.get("all_excluded_domains_are_official") is not True
    ):
        raise ValueError("partition audit does not bind the qualifying 5,218/180 split")
    if sorted(
        ({key: row[key] for key in ("name", "sha256", "domains")} for row in partition["held_out_panels"]),
        key=lambda row: row["name"],
    ) != sorted(registry_panels, key=lambda row: row["name"]):
        raise ValueError("partition held-out panels differ from the frozen registry")

    train_path = resolved["train_list"][0]
    try:
        train_domains = resolved["train_list"][2].decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("train list is not valid UTF-8") from exc
    if len(train_domains) != expectations.train_domains or len(set(train_domains)) != expectations.train_domains:
        raise ValueError("train list count or uniqueness mismatch")
    output_identity = partition.get("outputs", {}).get("train_eligible_list", {})
    if (
        output_identity.get("sha256") != resolved["train_list"][1]
        or output_identity.get("domains") != expectations.train_domains
        or partition_counts.get("train_eligible_sha256") != resolved["train_list"][1]
    ):
        raise ValueError("partition audit does not bind the exact train list")
    if not set(owner_by_domain).issubset(manifest_domains):
        raise ValueError("frozen panel contains a domain absent from the manifest")
    expected_train = [domain for domain in manifest_domains if domain not in owner_by_domain]
    expected_excluded = [domain for domain in manifest_domains if domain in owner_by_domain]
    if train_domains != expected_train:
        raise ValueError("train list is not the exact manifest-minus-panel union")
    expected_assignments = [
        {
            "domain": domain,
            "partition": "excluded" if domain in owner_by_domain else "train_eligible",
            "panel": owner_by_domain.get(domain),
        }
        for domain in manifest_domains
    ]
    if partition.get("domain_assignments") != expected_assignments:
        raise ValueError("partition assignments differ from the exact panel ownership map")
    if partition_counts.get("excluded_union_sha256") != _canonical_domain_sha256(expected_excluded):
        raise ValueError("partition excluded union SHA256 mismatch")
    if partition_counts.get("train_eligible_sha256") != _canonical_domain_sha256(expected_train):
        raise ValueError("partition train-eligible SHA256 mismatch")
    if partition.get("partition_contract_sha256") != _partition_contract_sha256(partition):
        raise ValueError("partition contract SHA256 mismatch")

    configured_root_path = Path(configured_root).expanduser()
    if configured_root_path.is_symlink():
        raise ValueError("configured data root must not be a symlink")
    if configured_root_path.resolve() != Path(audit.get("root", "")).resolve():
        raise ValueError("configured data root differs from the qualified audit root")
    _verify_rehash_journal_identity(audit, configured_root_path, expectations.domains)
    if Path(configured_manifest).expanduser().resolve() != manifest_path:
        raise ValueError("configured manifest differs from the contracted manifest")
    if Path(configured_domains_file).expanduser().resolve() != train_path:
        raise ValueError("configured domains file differs from the contracted train list")

    _verify_live_payload_fingerprints(configured_root_path, manifest)

    return {
        "status": "PASS_FULL_TRAINING_DATA_CONTRACT",
        "contract_sha256": actual_contract_sha256,
        "data_audit_sha256": resolved["data_audit"][1],
        "manifest_sha256": resolved["manifest"][1],
        "official_list_sha256": resolved["official_list"][1],
        "panel_registry_sha256": resolved["panel_registry"][1],
        "partition_audit_sha256": resolved["partition_audit"][1],
        "source_inventory_sha256": resolved["source_inventory"][1],
        "staging_metadata_sha256": resolved["staging_metadata"][1],
        "train_list_sha256": resolved["train_list"][1],
        "source_revision": expectations.source_revision,
        "official_domain_list_sha256": expectations.official_sha256,
        "train_domains": expectations.train_domains,
        "excluded_domains": expectations.excluded_domains,
    }
