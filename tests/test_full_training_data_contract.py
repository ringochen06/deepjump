import hashlib
import json
from pathlib import Path

import pytest
import torch

from deepjump import data_contract as contract_module
from deepjump import evaluation_contract as evaluation_module
from deepjump.data_contract import (
    ContractExpectations,
    EXPECTED_SOURCE_INVENTORY_SHA256,
    EXPECTED_SOURCE_REVISION,
    verify_full_training_data_contract,
)
from deepjump.evaluation_contract import verify_frozen_evaluation_identity
from scripts.build_full_training_data_contract import build_bundle


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _fixture(tmp_path: Path):
    data_root = tmp_path / "data-root"
    data_root.mkdir()
    data_dir = data_root / "data"
    data_dir.mkdir()
    domains = [f"d{i:03d}" for i in range(8)]
    sizes = [101 + index for index in range(len(domains))]

    payloads: dict[str, bytes] = {}
    for domain, size in zip(domains, sizes, strict=True):
        seed = f"payload-{domain}".encode()
        raw = (seed * ((size // len(seed)) + 1))[:size]
        payloads[domain] = raw
        (data_dir / f"mdcath_dataset_{domain}.h5").write_bytes(raw)

    official = tmp_path / "mdCATH_domains.txt"
    official.write_text("".join(f"{domain}\n" for domain in domains))
    source_inventory = tmp_path / "mdcath_source_inventory.jsonl"
    source_rows = [
        {
            "git_blob_sha1": hashlib.sha1(f"git-{domain}".encode()).hexdigest(),
            "lfs_pointer_size": 130,
            "lfs_sha256": hashlib.sha256(payloads[domain]).hexdigest(),
            "lfs_size": size,
            "path": f"data/mdcath_dataset_{domain}.h5",
            "size": size,
            "xet_hash": hashlib.sha256(f"xet-{domain}".encode()).hexdigest(),
        }
        for domain, size in zip(domains, sizes, strict=True)
    ]
    source_inventory.write_bytes(_canonical_jsonl(source_rows))

    panel_definitions = [
        ("fresh_external20", "external_dev_20_length_proportional_seed20260721.txt", "development_seen"),
        ("guarded_external20", "guarded_external_dev_20_length_proportional_seed20260722.txt", "development_seen"),
        ("legacy_dev20", "dev_20_length_proportional_seed0.txt", "development_seen"),
        ("paper_horizon_external20", "paper_horizon_external_dev_20_length_proportional_seed20260723.txt", "external_reserved"),
        ("untouched_confirmation100", "confirmation_100_length_proportional_seed20260717.txt", "untouched_confirmation_reserved"),
    ]
    registry_rows = []
    owner_by_domain: dict[str, str] = {}
    held_out_panels = []
    for domain, (name, filename, role) in zip(domains[3:], panel_definitions, strict=True):
        target = tmp_path / filename
        target.write_text(f"{domain}\n")
        row = {
            "domains": 1,
            "name": name,
            "path": filename,
            "role": role,
            "sha256": _sha(target),
        }
        registry_rows.append(row)
        owner_by_domain[domain] = name
        held_out_panels.append(
            {
                "name": name,
                "path": str(target),
                "sha256": row["sha256"],
                "domains": 1,
            }
        )
    train = [domain for domain in domains if domain not in owner_by_domain]

    panel_contract_payload = {
        "schema": "deepjump.expanded_data_partition.v1",
        "held_out_panels": sorted(
            ({key: row[key] for key in ("name", "sha256", "domains")} for row in registry_rows),
            key=lambda row: row["name"],
        ),
    }
    panel_contract_sha = hashlib.sha256(
        json.dumps(panel_contract_payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    registry = tmp_path / "evaluation_exclusion_registry.json"
    _write_json(
        registry,
        {
            "official_domain_list_sha256": _sha(official),
            "panel_contract_sha256": panel_contract_sha,
            "panels": registry_rows,
            "schema": "deepjump.full_mdcath_evaluation_exclusion_registry.v1",
            "source_revision": EXPECTED_SOURCE_REVISION,
        },
    )
    expectations = ContractExpectations(
        official_sha256=_sha(official),
        source_inventory_sha256=_sha(source_inventory),
        panel_registry_sha256=_sha(registry),
        panel_contract_sha256=panel_contract_sha,
        domains=len(domains),
        excluded_domains=len(owner_by_domain),
        train_domains=len(train),
        h5_bytes=sum(sizes),
        trajectories=len(domains) * 25,
    )

    manifest = tmp_path / "full_mdcath_manifest.json"
    manifest_rows = []
    inventory_digest = hashlib.sha256()
    for index, (domain, size, source) in enumerate(
        zip(domains, sizes, source_rows, strict=True), 1
    ):
        filename = f"mdcath_dataset_{domain}.h5"
        trajectories = [
            {"temp": temp, "replica": replica, "num_frames": index + 1}
            for temp in contract_module.TEMPERATURES
            for replica in contract_module.REPLICAS
        ]
        manifest_rows.append(
            {
                "domain": domain,
                "file": filename,
                "local_fingerprint": {
                    "device": (data_dir / filename).stat().st_dev,
                    "inode": (data_dir / filename).stat().st_ino,
                    "size": (data_dir / filename).stat().st_size,
                    "mtime_ns": (data_dir / filename).stat().st_mtime_ns,
                    "ctime_ns": (data_dir / filename).stat().st_ctime_ns,
                },
                "num_atoms": 10 + index,
                "num_residues": 5 + index,
                "sha256": source["lfs_sha256"],
                "size": size,
                "trajectories": trajectories,
            }
        )
        inventory_digest.update(
            f"{filename}\t{size}\t{domain}\t{5 + index}\t{10 + index}\t25\t{source['lfs_sha256']}\n".encode()
        )
    _write_json(manifest, manifest_rows)
    train_list = tmp_path / "train_eligible_5218.txt"
    train_list.write_text("".join(f"{domain}\n" for domain in train))

    assignments = [
        {
            "domain": domain,
            "partition": "excluded" if domain in owner_by_domain else "train_eligible",
            "panel": owner_by_domain.get(domain),
        }
        for domain in domains
    ]
    partition = tmp_path / "partition.json"
    partition_payload = {
            "schema": "deepjump.expanded_data_partition.v1",
            "status": "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION",
            "official": {"sha256": expectations.official_sha256, "domains": len(domains)},
            "held_out_panels": held_out_panels,
            "held_out_panel_contract_sha256": panel_contract_sha,
            "panel_registry": {"sha256": expectations.panel_registry_sha256},
            "partition": {
                "official_domains": len(domains),
                "excluded_domains": len(owner_by_domain),
                "train_eligible_domains": len(train),
                "excluded_union_sha256": contract_module._canonical_domain_sha256(
                    [domain for domain in domains if domain in owner_by_domain]
                ),
                "train_eligible_sha256": _sha(train_list),
                "panels_are_pairwise_disjoint": True,
                "all_excluded_domains_are_official": True,
            },
            "outputs": {
                "train_eligible_list": {
                    "sha256": _sha(train_list),
                    "domains": len(train),
                }
            },
            "domain_assignments": assignments,
    }
    partition_payload["partition_contract_sha256"] = (
        contract_module._partition_contract_sha256(partition_payload)
    )
    _write_json(partition, partition_payload)

    completed_at = "2026-07-23T00:00:00+00:00"
    generating_commit = "a" * 40
    journal = data_root / "control" / "full_payload_rehash_journal_v1"
    journal.mkdir(parents=True)
    journal_session = journal / "session.json"
    journal_completion = journal / "completion.json"
    _write_json(journal_session, {"schema": "deepjump.full_payload_rehash_journal.v1"})
    _write_json(journal_completion, {"schema": "deepjump.full_payload_rehash_completion.v1"})
    rehash_identity = {
        "audit_script_sha256": "b" * 64,
        "payload_rehash_journal_dir": str(journal.resolve()),
        "payload_rehash_journal_session_sha256": _sha(journal_session),
        "payload_rehash_journal_completion_sha256": _sha(journal_completion),
        "payload_rehash_records_resumed": expectations.domains,
        "payload_rehash_records_created": 0,
    }
    metadata = tmp_path / "full_mdcath_staging.metadata.json"
    metadata_payload = {
        "all_coordinate_frames_finite_verified": False,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "created_at": completed_at,
        "data_gate_passed": True,
        "domains": expectations.domains,
        "expected_hash_script_sha256": None,
        "finite_endpoint_frames_verified": expectations.trajectories * 2,
        "generating_commit": generating_commit,
        "h5_bytes": expectations.h5_bytes,
        "h5_files": expectations.domains,
        "hdf5_files_structurally_verified": expectations.domains,
        "live_payload_bytes_rehashed": True,
        "manifest_file": manifest.name,
        "manifest_sha256": _sha(manifest),
        "official_domain_list_file": official.name,
        "official_domain_list_sha256": expectations.official_sha256,
        "payload_hash_sidecars_verified": 0,
        "payload_hash_verification_mode": "full_rehash",
        "payload_hashes_verified": expectations.domains,
        "schema": "deepjump.full_mdcath_staging.v1",
        "selection_strategy": "official-full-5398",
        "source_inventory_file": source_inventory.name,
        "source_inventory_sha256": expectations.source_inventory_sha256,
        "source_repo": "compsciencelab/mdCATH",
        "source_revision": EXPECTED_SOURCE_REVISION,
        "temperature_replica_grid": "5x5",
        "trajectories": expectations.trajectories,
        "verified_local_inventory_sha256": inventory_digest.hexdigest(),
        **rehash_identity,
    }
    _write_json(metadata, metadata_payload)
    data_audit = tmp_path / "full_mdcath_audit.json"
    audit_payload = {
        "all_coordinate_frames_finite_verified": False,
        "completed_at": completed_at,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "data_gate_passed": True,
        "domains": expectations.domains,
        "expected_hash_script_sha256": None,
        "external_development_authorized": False,
        "finite_endpoint_frames_verified": expectations.trajectories * 2,
        "formal_training_authorized": False,
        "generating_commit": generating_commit,
        "h5_bytes": expectations.h5_bytes,
        "h5_files": expectations.domains,
        "hdf5_files_structurally_verified": expectations.domains,
        "live_payload_bytes_rehashed": True,
        "manifest_sha256": _sha(manifest),
        "metadata_sha256": _sha(metadata),
        "official_domain_list_sha256": expectations.official_sha256,
        "payload_hash_sidecars_verified": 0,
        "payload_hash_verification_mode": "full_rehash",
        "payload_hashes_verified": expectations.domains,
        "root": str(data_root.resolve()),
        "schema": "deepjump.full_mdcath_audit.v1",
        "second_seed_authorized": False,
        "source_inventory_sha256": expectations.source_inventory_sha256,
        "source_repo": "compsciencelab/mdCATH",
        "source_revision": EXPECTED_SOURCE_REVISION,
        "status": "PASS_FULL_LIVE_PAYLOAD_REHASH",
        "trajectories": expectations.trajectories,
        "untouched_confirmation_authorized": False,
        "verified_local_inventory_sha256": inventory_digest.hexdigest(),
        **rehash_identity,
    }
    _write_json(data_audit, audit_payload)
    artifacts = {
        "data_audit": data_audit,
        "manifest": manifest,
        "official_list": official,
        "panel_registry": registry,
        "partition_audit": partition,
        "source_inventory": source_inventory,
        "staging_metadata": metadata,
        "train_list": train_list,
    }
    contract = tmp_path / "contract.json"
    _write_json(
        contract,
        {
            "schema": "deepjump.full_training_data_contract.v1",
            "source_revision": EXPECTED_SOURCE_REVISION,
            "official_domain_list_sha256": expectations.official_sha256,
            "artifacts": {
                name: {"path": path.name, "sha256": _sha(path)}
                for name, path in artifacts.items()
            },
        },
    )
    return contract, _sha(contract), data_root, manifest, train_list, expectations


def _reseal_metadata_audit_contract(contract: Path) -> str:
    contract_payload = json.loads(contract.read_text())
    metadata = contract.parent / contract_payload["artifacts"]["staging_metadata"]["path"]
    audit = contract.parent / contract_payload["artifacts"]["data_audit"]["path"]
    audit_payload = json.loads(audit.read_text())
    audit_payload["metadata_sha256"] = _sha(metadata)
    _write_json(audit, audit_payload)
    contract_payload["artifacts"]["staging_metadata"]["sha256"] = _sha(metadata)
    contract_payload["artifacts"]["data_audit"]["sha256"] = _sha(audit)
    _write_json(contract, contract_payload)
    return _sha(contract)


def test_full_training_data_contract_passes_exact_frozen_artifacts(tmp_path):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    report = verify_full_training_data_contract(
        contract,
        contract_sha,
        configured_root=root,
        configured_manifest=manifest,
        configured_domains_file=train_list,
        expectations=expectations,
    )
    assert report["status"] == "PASS_FULL_TRAINING_DATA_CONTRACT"
    assert report["train_domains"] == expectations.train_domains
    assert report["excluded_domains"] == expectations.excluded_domains


def test_full_training_data_contract_rejects_stale_audit_with_empty_live_root(tmp_path):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    for payload in (root / "data").glob("*.h5"):
        payload.unlink()
    with pytest.raises(ValueError, match="live HDF5 file set differs"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_full_training_data_contract_rejects_replaced_live_payload(tmp_path):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    payload = next((root / "data").glob("*.h5"))
    payload.write_bytes(b"x" * payload.stat().st_size)
    with pytest.raises(ValueError, match="live payload fingerprint differs"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_full_training_data_contract_rejects_missing_rehash_completion(tmp_path):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    (root / "control" / "full_payload_rehash_journal_v1" / "completion.json").unlink()
    with pytest.raises(ValueError, match="cannot open payload rehash completion.json"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_full_training_data_contract_rejects_rehash_identity_drift(tmp_path):
    contract, _, root, manifest, train_list, expectations = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    metadata = tmp_path / payload["artifacts"]["staging_metadata"]["path"]
    metadata_payload = json.loads(metadata.read_text())
    metadata_payload["audit_script_sha256"] = "c" * 64
    _write_json(metadata, metadata_payload)
    contract_sha = _reseal_metadata_audit_contract(contract)
    with pytest.raises(ValueError, match="rehash journal identities differ"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


@pytest.mark.parametrize(
    ("resumed", "created"),
    [(0, 0), (-1, 9), (True, 7)],
)
def test_full_training_data_contract_rejects_bad_rehash_record_counts(
    tmp_path, resumed, created
):
    contract, _, root, manifest, train_list, expectations = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    metadata = tmp_path / payload["artifacts"]["staging_metadata"]["path"]
    audit = tmp_path / payload["artifacts"]["data_audit"]["path"]
    for path in (metadata, audit):
        document = json.loads(path.read_text())
        document["payload_rehash_records_resumed"] = resumed
        document["payload_rehash_records_created"] = created
        _write_json(path, document)
    contract_sha = _reseal_metadata_audit_contract(contract)
    with pytest.raises(ValueError, match="record counts mismatch"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


@pytest.mark.parametrize("journal_value", ["relative/journal", "/tmp/outside-journal"])
def test_full_training_data_contract_rejects_unbound_rehash_journal_path(
    tmp_path, journal_value
):
    contract, _, root, manifest, train_list, expectations = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    metadata = tmp_path / payload["artifacts"]["staging_metadata"]["path"]
    audit = tmp_path / payload["artifacts"]["data_audit"]["path"]
    for path in (metadata, audit):
        document = json.loads(path.read_text())
        document["payload_rehash_journal_dir"] = journal_value
        _write_json(path, document)
    contract_sha = _reseal_metadata_audit_contract(contract)
    with pytest.raises(ValueError, match="payload rehash journal"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


@pytest.mark.parametrize("filename", ["session.json", "completion.json"])
def test_full_training_data_contract_rejects_rehash_journal_content_drift(
    tmp_path, filename
):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    journal_file = root / "control" / "full_payload_rehash_journal_v1" / filename
    journal_file.write_text("changed\n")
    with pytest.raises(ValueError, match=rf"payload rehash {filename} SHA256 mismatch"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_production_contract_pins_canonical_source_inventory():
    assert ContractExpectations().source_inventory_sha256 == EXPECTED_SOURCE_INVENTORY_SHA256
    assert EXPECTED_SOURCE_INVENTORY_SHA256 == (
        "2e6e3602a0858aaafc849cfa7cc1ee7e076736cb15335d7914726898f06f6cdf"
    )


def test_full_training_data_contract_rejects_domain_only_manifest(tmp_path):
    contract, _, root, manifest, train_list, expectations = _fixture(tmp_path)
    contract_payload = json.loads(contract.read_text())
    _write_json(manifest, [{"domain": f"d{index:03d}"} for index in range(8)])

    metadata = tmp_path / contract_payload["artifacts"]["staging_metadata"]["path"]
    metadata_payload = json.loads(metadata.read_text())
    metadata_payload["manifest_sha256"] = _sha(manifest)
    _write_json(metadata, metadata_payload)
    audit = tmp_path / contract_payload["artifacts"]["data_audit"]["path"]
    audit_payload = json.loads(audit.read_text())
    audit_payload["manifest_sha256"] = _sha(manifest)
    audit_payload["metadata_sha256"] = _sha(metadata)
    _write_json(audit, audit_payload)
    for name, path in {
        "manifest": manifest,
        "staging_metadata": metadata,
        "data_audit": audit,
    }.items():
        contract_payload["artifacts"][name]["sha256"] = _sha(path)
    _write_json(contract, contract_payload)

    with pytest.raises(ValueError, match="manifest row 1 fields mismatch"):
        verify_full_training_data_contract(
            contract,
            _sha(contract),
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_full_training_data_contract_rejects_sidecar_only_audit(tmp_path):
    contract, _, root, manifest, train_list, expectations = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    audit = tmp_path / payload["artifacts"]["data_audit"]["path"]
    audit_payload = json.loads(audit.read_text())
    audit_payload.update(
        {
            "status": "PREAUDIT_SIDECAR_ATTESTATION_ONLY",
            "payload_hash_verification_mode": "sidecar",
            "data_gate_passed": False,
            "live_payload_bytes_rehashed": False,
        }
    )
    _write_json(audit, audit_payload)
    payload["artifacts"]["data_audit"]["sha256"] = _sha(audit)
    _write_json(contract, payload)
    with pytest.raises(ValueError, match="live-rehash"):
        verify_full_training_data_contract(
            contract,
            _sha(contract),
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_full_training_data_contract_rejects_configured_domain_drift(tmp_path):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    alternate = tmp_path / "alternate.txt"
    alternate.write_text(train_list.read_text())
    with pytest.raises(ValueError, match="configured domains file"):
        verify_full_training_data_contract(
            contract,
            contract_sha,
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=alternate,
            expectations=expectations,
        )


def test_full_training_data_contract_rejects_swapped_heldout_domains(tmp_path):
    contract, _, root, manifest, train_list, expectations = _fixture(tmp_path)
    contract_payload = json.loads(contract.read_text())
    partition = tmp_path / contract_payload["artifacts"]["partition_audit"]["path"]
    partition_payload = json.loads(partition.read_text())
    excluded_index = next(
        index
        for index, row in enumerate(partition_payload["domain_assignments"])
        if row["partition"] == "excluded"
    )
    train_index = next(
        index
        for index, row in enumerate(partition_payload["domain_assignments"])
        if row["partition"] == "train_eligible"
    )
    partition_payload["domain_assignments"][excluded_index] = {
        "domain": partition_payload["domain_assignments"][excluded_index]["domain"],
        "partition": "train_eligible",
        "panel": None,
    }
    partition_payload["domain_assignments"][train_index] = {
        "domain": partition_payload["domain_assignments"][train_index]["domain"],
        "partition": "excluded",
        "panel": "legacy_dev20",
    }
    partition_payload["partition_contract_sha256"] = (
        contract_module._partition_contract_sha256(partition_payload)
    )
    _write_json(partition, partition_payload)
    contract_payload["artifacts"]["partition_audit"]["sha256"] = _sha(partition)
    _write_json(contract, contract_payload)
    with pytest.raises(ValueError, match="ownership map"):
        verify_full_training_data_contract(
            contract,
            _sha(contract),
            configured_root=root,
            configured_manifest=manifest,
            configured_domains_file=train_list,
            expectations=expectations,
        )


def test_contract_bundle_builder_copies_and_reverifies_every_artifact(tmp_path):
    contract, _, root, _, _, expectations = _fixture(tmp_path)
    payload = json.loads(contract.read_text())
    artifacts = {
        name: tmp_path / row["path"]
        for name, row in payload["artifacts"].items()
    }
    output = tmp_path / "published"
    report = build_bundle(
        output,
        data_audit=artifacts["data_audit"],
        data_audit_sha256=_sha(artifacts["data_audit"]),
        manifest=artifacts["manifest"],
        manifest_sha256=_sha(artifacts["manifest"]),
        official_list=artifacts["official_list"],
        official_list_sha256=_sha(artifacts["official_list"]),
        panel_registry=artifacts["panel_registry"],
        partition_audit=artifacts["partition_audit"],
        partition_audit_sha256=_sha(artifacts["partition_audit"]),
        source_inventory=artifacts["source_inventory"],
        source_inventory_sha256=_sha(artifacts["source_inventory"]),
        staging_metadata=artifacts["staging_metadata"],
        staging_metadata_sha256=_sha(artifacts["staging_metadata"]),
        train_list=artifacts["train_list"],
        train_list_sha256=_sha(artifacts["train_list"]),
        expectations=expectations,
    )
    assert report["status"] == "PASS_FULL_TRAINING_DATA_CONTRACT"
    assert Path(report["bundle"]) == output
    assert (output / "full_training_data_contract.json").is_file()
    assert (output / "full_training_data_contract.sha256").is_file()
    assert json.loads((output / "full_mdcath_audit.json").read_text())["root"] == str(root)


def _contracted_checkpoint(tmp_path: Path):
    contract, contract_sha, root, manifest, train_list, expectations = _fixture(tmp_path)
    verification = verify_full_training_data_contract(
        contract,
        contract_sha,
        configured_root=root,
        configured_manifest=manifest,
        configured_domains_file=train_list,
        expectations=expectations,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {"weight": torch.tensor([1.0])},
            "step": 2_000,
            "cfg": {
                "data": {
                    "root": str(root),
                    "manifest": str(manifest),
                    "domains_file": str(train_list),
                    "full_training_contract": str(contract),
                    "full_training_contract_sha256": contract_sha,
                },
                "train": {"run_class": "full_data_stage"},
            },
            "train_state": {"full_training_data_contract": verification},
        },
        checkpoint,
    )
    return checkpoint, contract, contract_sha, expectations


def test_external_identity_accepts_only_reserved_external_panel(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    report = verify_frozen_evaluation_identity(
        checkpoint,
        contract,
        contract_sha,
        expected_checkpoint_sha256=_sha(checkpoint),
        expected_checkpoint_step=2_000,
        phase="external",
        panel_name="paper_horizon_external20",
        panel_file=panel,
        contract_expectations=expectations,
    )
    assert report["status"] == "PASS_FROZEN_EVALUATION_IDENTITY"
    assert report["panel_domains"] == 1
    assert report["formal_training_authorized"] is False


def test_development_identity_accepts_only_development_seen_panel(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    panel = tmp_path / "dev_20_length_proportional_seed0.txt"
    report = verify_frozen_evaluation_identity(
        checkpoint,
        contract,
        contract_sha,
        expected_checkpoint_sha256=_sha(checkpoint),
        expected_checkpoint_step=2_000,
        phase="development",
        panel_name="legacy_dev20",
        panel_file=panel,
        contract_expectations=expectations,
    )
    assert report["panel_role"] == "development_seen"


def test_external_identity_rejects_development_seen_panel(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    panel = tmp_path / "guarded_external_dev_20_length_proportional_seed20260722.txt"
    with pytest.raises(ValueError, match="cannot serve as external"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="guarded_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )


def test_untouched_identity_accepts_only_confirmation_reserved_panel(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    panel = tmp_path / "confirmation_100_length_proportional_seed20260717.txt"
    report = verify_frozen_evaluation_identity(
        checkpoint,
        contract,
        contract_sha,
        expected_checkpoint_sha256=_sha(checkpoint),
        expected_checkpoint_step=2_000,
        phase="untouched",
        panel_name="untouched_confirmation100",
        panel_file=panel,
        contract_expectations=expectations,
    )
    assert report["panel_domains"] == 1
    assert report["panel_role"] == "untouched_confirmation_reserved"


def test_evaluation_identity_rejects_checkpoint_contract_drift(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["train_state"]["full_training_data_contract"] = None
    torch.save(payload, checkpoint)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    with pytest.raises(ValueError, match="train state"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )


def test_evaluation_identity_rejects_modified_checkpoint_weight(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    expected_checkpoint_sha = _sha(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model"]["weight"].add_(1.0)
    torch.save(payload, checkpoint)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=expected_checkpoint_sha,
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )


def test_evaluation_identity_loads_the_exact_hashed_checkpoint_snapshot(tmp_path, monkeypatch):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    original_sha = _sha(checkpoint)
    original_reader = evaluation_module._read_regular_bytes

    def replace_after_read(path, label):
        raw = original_reader(path, label)
        if label == "checkpoint":
            replacement = torch.load(checkpoint, map_location="cpu", weights_only=True)
            replacement["model"]["weight"] = torch.tensor([9.0])
            torch.save(replacement, checkpoint)
        return raw

    monkeypatch.setattr(evaluation_module, "_read_regular_bytes", replace_after_read)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    report = verify_frozen_evaluation_identity(
        checkpoint,
        contract,
        contract_sha,
        expected_checkpoint_sha256=original_sha,
        expected_checkpoint_step=2_000,
        phase="external",
        panel_name="paper_horizon_external20",
        panel_file=panel,
        contract_expectations=expectations,
    )
    assert report["checkpoint_sha256"] == original_sha


def test_evaluation_identity_rejects_symlinked_checkpoint(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    checkpoint_link = tmp_path / "checkpoint-link.pt"
    checkpoint_link.symlink_to(checkpoint)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    with pytest.raises(ValueError, match="checkpoint must be a regular non-symlink"):
        verify_frozen_evaluation_identity(
            checkpoint_link,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )


def test_evaluation_identity_rejects_symlinked_contract_and_panel(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    contract_link = tmp_path / "contract-link.json"
    contract_link.symlink_to(contract)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    panel_link = tmp_path / "panel-link.txt"
    panel_link.symlink_to(panel)
    with pytest.raises(ValueError, match="contract must be a regular non-symlink"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract_link,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )
    with pytest.raises(ValueError, match="panel file must be a regular non-symlink"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel_link,
            contract_expectations=expectations,
        )


def test_evaluation_identity_rejects_empty_model_state(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model"] = {}
    torch.save(payload, checkpoint)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    with pytest.raises(ValueError, match="model state must be a non-empty"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )


def test_evaluation_identity_rejects_nonfinite_model_state(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model"]["weight"] = torch.tensor([float("nan")])
    torch.save(payload, checkpoint)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    with pytest.raises(ValueError, match="non-finite tensors"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_000,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )


def test_evaluation_identity_rejects_wrong_checkpoint_step(tmp_path):
    checkpoint, contract, contract_sha, expectations = _contracted_checkpoint(tmp_path)
    panel = tmp_path / "paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    with pytest.raises(ValueError, match="checkpoint step mismatch"):
        verify_frozen_evaluation_identity(
            checkpoint,
            contract,
            contract_sha,
            expected_checkpoint_sha256=_sha(checkpoint),
            expected_checkpoint_step=2_001,
            phase="external",
            panel_name="paper_horizon_external20",
            panel_file=panel,
            contract_expectations=expectations,
        )
