import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import scripts.audit_full_mdcath_staging as auditor
from deepjump import data_contract as contract_module


TEST_COMMIT = "a" * 40
TEST_DOMAINS = ["1abcA00", "2defB00"]
TEST_HASH_SCRIPT_SHA256 = (
    "7278690eb54eb84e0fb823dce79821ee8118aa7e43907de93453eb0f00a4268b"
)


def _sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_row(path: Path, domain: str) -> dict:
    payload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": f"data/{path.name}",
        "size": path.stat().st_size,
        "git_blob_sha1": hashlib.sha1(domain.encode()).hexdigest(),
        "lfs_sha256": payload_sha256,
        "lfs_size": path.stat().st_size,
        "lfs_pointer_size": 130,
        "xet_hash": payload_sha256,
    }


def _write_source_inventory(path: Path, data: Path) -> bytes:
    rows = [
        _source_row(data / f"mdcath_dataset_{domain}.h5", domain)
        for domain in TEST_DOMAINS
    ]
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: row["path"])
    ).encode()
    path.write_bytes(content)
    return content


def _write_payload_hash_sidecars(case: dict) -> Path:
    sidecar_dir = case["root"] / "control" / "payload_hashes_v1"
    sidecar_dir.mkdir(parents=True)
    for domain in TEST_DOMAINS:
        payload = (
            case["root"] / "data" / f"mdcath_dataset_{domain}.h5"
        )
        stat_result = payload.stat()
        sidecar = {
            "schema": auditor.PAYLOAD_HASH_SIDECAR_SCHEMA,
            "source_revision": case["source_revision"],
            "file": f"data/{payload.name}",
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "fingerprint": {
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "ctime_ns": stat_result.st_ctime_ns,
            },
            "hash_script_sha256": TEST_HASH_SCRIPT_SHA256,
            "completed_at": "2026-07-23T00:00:00+00:00",
        }
        (sidecar_dir / f"{payload.name}.json").write_text(
            json.dumps(sidecar, sort_keys=True) + "\n"
        )
    case["payload_hash_sidecar_dir"] = sidecar_dir
    case["expected_hash_script_sha256"] = TEST_HASH_SCRIPT_SHA256
    case["rehash_payloads"] = False
    return sidecar_dir


def _write_domain(path: Path, domain: str, *, atoms: int = 3, frames: int = 2) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group(domain)
        group.attrs["numResidues"] = 1
        group.attrs["numProteinAtoms"] = atoms
        group.create_dataset("psf", data=np.bytes_("PSF fixture"))
        group.create_dataset("resid", data=np.arange(atoms, dtype=np.int32))
        group.create_dataset(
            "resname", data=np.asarray([b"ALA"] * atoms, dtype="S3")
        )
        for temperature in auditor.TEMPERATURES:
            temperature_group = group.create_group(str(temperature))
            for replica in auditor.REPLICAS:
                replica_group = temperature_group.create_group(str(replica))
                replica_group.attrs["numFrames"] = frames
                coords = np.full(
                    (frames, atoms, 3),
                    temperature + replica / 10,
                    dtype=np.float32,
                )
                replica_group.create_dataset("coords", data=coords)


def _make_case(tmp_path: Path) -> dict:
    root = tmp_path / "staging"
    data = root / "data"
    data.mkdir(parents=True)
    for domain in TEST_DOMAINS:
        _write_domain(data / f"mdcath_dataset_{domain}.h5", domain)
    official_content = ("\n".join(TEST_DOMAINS) + "\n").encode()
    official = tmp_path / "mdCATH_domains.txt"
    official.write_bytes(official_content)
    source_inventory = tmp_path / "source_inventory.jsonl"
    source_inventory_content = _write_source_inventory(source_inventory, data)
    return {
        "root": root,
        "official_list": official,
        "source_inventory": source_inventory,
        "source_revision": auditor.EXPECTED_SOURCE_REVISION,
        "payload_hash_sidecar_dir": None,
        "expected_hash_script_sha256": None,
        "rehash_payloads": True,
        "manifest_output": root / "full_manifest.json",
        "metadata_output": root / "full_metadata.json",
        "audit_output": root / "full_audit.json",
        "generating_commit": TEST_COMMIT,
        "expectations": auditor.AuditExpectations(
            domains=len(TEST_DOMAINS),
            h5_bytes=sum(path.stat().st_size for path in data.glob("*.h5")),
            trajectories=len(TEST_DOMAINS) * 25,
            official_list_sha256=_sha_bytes(official_content),
            source_revision=auditor.EXPECTED_SOURCE_REVISION,
            source_inventory_sha256=_sha_bytes(source_inventory_content),
        ),
    }


def _refresh_source_identity(case: dict) -> None:
    old = case["expectations"]
    source_inventory_content = _write_source_inventory(
        case["source_inventory"], case["root"] / "data"
    )
    case["expectations"] = auditor.AuditExpectations(
        domains=old.domains,
        h5_bytes=sum(
            path.stat().st_size
            for path in (case["root"] / "data").glob("*.h5")
        ),
        trajectories=old.trajectories,
        official_list_sha256=old.official_list_sha256,
        source_revision=old.source_revision,
        source_inventory_sha256=_sha_bytes(source_inventory_content),
    )


def _publish(case: dict) -> dict:
    return auditor.build_and_publish(**case)


def _interrupt_rehash_after_first_record(case: dict, monkeypatch) -> Path:
    original = auditor._verify_payload_sha256
    calls = 0

    def interrupt_second(path, expected_sha256):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated rehash interruption")
        return original(path, expected_sha256)

    monkeypatch.setattr(auditor, "_verify_payload_sha256", interrupt_second)
    with pytest.raises(RuntimeError, match="simulated rehash interruption"):
        _publish(case)
    monkeypatch.setattr(auditor, "_verify_payload_sha256", original)
    journal = case["root"] / "control" / "full_payload_rehash_journal_v1"
    assert (journal / auditor.REHASH_SESSION_FILE).is_file()
    assert (journal / f"payload_{TEST_DOMAINS[0]}.json").is_file()
    assert not (journal / f"payload_{TEST_DOMAINS[1]}.json").exists()
    assert not (journal / auditor.REHASH_COMPLETION_FILE).exists()
    return journal


def test_full_audit_publishes_verified_artifacts_atomically(tmp_path):
    case = _make_case(tmp_path)
    report = _publish(case)

    manifest = json.loads(case["manifest_output"].read_text())
    metadata = json.loads(case["metadata_output"].read_text())
    audit = json.loads(case["audit_output"].read_text())
    assert set(metadata) == contract_module._METADATA_FIELDS
    assert set(audit) == contract_module._AUDIT_FIELDS
    assert [entry["domain"] for entry in manifest] == TEST_DOMAINS
    assert set(manifest[0]["local_fingerprint"]) == {
        "device", "inode", "size", "mtime_ns", "ctime_ns"
    }
    assert all(len(entry["trajectories"]) == 25 for entry in manifest)
    assert all(
        entry["sha256"]
        == hashlib.sha256(
            (case["root"] / "data" / entry["file"]).read_bytes()
        ).hexdigest()
        for entry in manifest
    )
    assert metadata["domains"] == 2
    assert metadata["h5_files"] == 2
    assert metadata["trajectories"] == 50
    assert metadata["source_revision"] == auditor.EXPECTED_SOURCE_REVISION
    assert metadata["source_inventory_sha256"] == case[
        "expectations"
    ].source_inventory_sha256
    assert metadata["payload_hashes_verified"] == 2
    assert metadata["finite_endpoint_frames_verified"] == 100
    assert audit["status"] == auditor.LIVE_REHASH_PASS_STATUS
    assert audit["data_gate_passed"] is True
    assert audit["live_payload_bytes_rehashed"] is True
    assert audit["formal_training_authorized"] is False
    assert audit["all_coordinate_frames_finite_verified"] is False
    assert audit["coordinate_finiteness_scope"] == (
        "first_and_last_frame_per_trajectory"
    )
    assert audit["manifest_sha256"] == hashlib.sha256(
        case["manifest_output"].read_bytes()
    ).hexdigest()
    assert audit["metadata_sha256"] == hashlib.sha256(
        case["metadata_output"].read_bytes()
    ).hexdigest()
    assert report["audit_sha256"] == hashlib.sha256(
        case["audit_output"].read_bytes()
    ).hexdigest()

    for output_key in ("manifest_output", "metadata_output", "audit_output"):
        output = case[output_key]
        digest, filename = output.with_name(output.name + ".sha256").read_text().split()
        assert filename == output.name
        assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not list(case["root"].glob(".*.tmp.*"))


def test_live_rehash_journal_resumes_only_missing_payloads(tmp_path, monkeypatch):
    case = _make_case(tmp_path)
    journal = _interrupt_rehash_after_first_record(case, monkeypatch)
    original = auditor._verify_payload_sha256
    rehashed = []

    def count_rehash(path, expected_sha256):
        rehashed.append(path.name)
        return original(path, expected_sha256)

    monkeypatch.setattr(auditor, "_verify_payload_sha256", count_rehash)
    report = _publish(case)
    assert rehashed == [f"mdcath_dataset_{TEST_DOMAINS[1]}.h5"]
    assert report["payload_rehash_records_resumed"] == 1
    assert report["payload_rehash_records_created"] == 1
    assert report["payload_rehash_journal_session_sha256"]
    assert report["payload_rehash_journal_completion_sha256"]
    assert {path.name for path in journal.iterdir()} == {
        auditor.REHASH_SESSION_FILE,
        auditor.REHASH_COMPLETION_FILE,
        *[f"payload_{domain}.json" for domain in TEST_DOMAINS],
    }

    rehashed.clear()
    repeated = _publish(case)
    assert rehashed == []
    assert repeated["payload_rehash_records_resumed"] == 2
    assert repeated["payload_rehash_records_created"] == 0
    assert repeated["payload_rehash_journal_completion_sha256"] == report[
        "payload_rehash_journal_completion_sha256"
    ]


def test_live_rehash_journal_rejects_forged_record(tmp_path, monkeypatch):
    case = _make_case(tmp_path)
    journal = _interrupt_rehash_after_first_record(case, monkeypatch)
    forged = json.loads(
        (journal / f"payload_{TEST_DOMAINS[0]}.json").read_text()
    )
    forged["domain"] = TEST_DOMAINS[1]
    forged["file"] = f"data/mdcath_dataset_{TEST_DOMAINS[1]}.h5"
    (journal / f"payload_{TEST_DOMAINS[1]}.json").write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="record identity mismatch"):
        _publish(case)
    assert not case["audit_output"].exists()


def test_live_rehash_journal_rejects_payload_fingerprint_drift(
    tmp_path, monkeypatch
):
    case = _make_case(tmp_path)
    _interrupt_rehash_after_first_record(case, monkeypatch)
    payload = case["root"] / "data" / f"mdcath_dataset_{TEST_DOMAINS[0]}.h5"
    before = payload.stat()
    os.utime(payload, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))

    with pytest.raises(ValueError, match="payload fingerprint drift"):
        _publish(case)
    assert not case["audit_output"].exists()


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "incomplete"])
def test_live_rehash_journal_rejects_non_exact_union(
    tmp_path, monkeypatch, mutation
):
    case = _make_case(tmp_path)
    journal = _interrupt_rehash_after_first_record(case, monkeypatch)
    first = journal / f"payload_{TEST_DOMAINS[0]}.json"
    if mutation == "extra":
        (journal / "unexpected.json").write_text("{}\n")
    elif mutation == "duplicate":
        (journal / f"payload_{TEST_DOMAINS[1]}.json").write_bytes(
            first.read_bytes()
        )
    else:
        (journal / auditor.REHASH_COMPLETION_FILE).write_text("{}\n")

    message = (
        "record identity mismatch"
        if mutation == "duplicate"
        else "exact inventory mismatch"
    )
    with pytest.raises(ValueError, match=message):
        _publish(case)
    assert not case["manifest_output"].exists()
    assert not case["metadata_output"].exists()
    assert not case["audit_output"].exists()


def test_valid_payload_hash_sidecars_avoid_rehash_without_weakening_identity(
    tmp_path, monkeypatch
):
    case = _make_case(tmp_path)
    _write_payload_hash_sidecars(case)

    def forbidden_rehash(*_args, **_kwargs):
        raise AssertionError("sidecar mode must not stream payloads again")

    monkeypatch.setattr(auditor, "_verify_payload_sha256", forbidden_rehash)
    report = _publish(case)
    metadata = json.loads(case["metadata_output"].read_text())
    assert report["payload_hash_verification_mode"] == "sidecar"
    assert report["payload_hashes_verified"] == 2
    assert report["payload_hash_sidecars_verified"] == 2
    assert report["expected_hash_script_sha256"] == TEST_HASH_SCRIPT_SHA256
    assert report["status"] == auditor.SIDECAR_PREAUDIT_STATUS
    assert report["data_gate_passed"] is False
    assert report["live_payload_bytes_rehashed"] is False
    assert report["formal_training_authorized"] is False
    assert metadata["payload_hash_verification_mode"] == "sidecar"
    assert metadata["data_gate_passed"] is False


def test_self_consistent_sidecar_claim_never_qualifies_live_payload(tmp_path):
    case = _make_case(tmp_path)
    sidecar_dir = _write_payload_hash_sidecars(case)
    payload = case["root"] / "data" / "mdcath_dataset_1abcA00.h5"
    sidecar = sidecar_dir / f"{payload.name}.json"
    original_claim = json.loads(sidecar.read_text())["sha256"]

    with h5py.File(payload, "r+") as handle:
        handle["1abcA00"]["320"]["0"]["coords"][0, 0, 0] += 1.0
    assert hashlib.sha256(payload.read_bytes()).hexdigest() != original_claim

    row = json.loads(sidecar.read_text())
    stat_result = payload.stat()
    row["fingerprint"] = {
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }
    sidecar.write_text(json.dumps(row, sort_keys=True) + "\n")

    report = _publish(case)
    assert report["status"] == auditor.SIDECAR_PREAUDIT_STATUS
    assert report["data_gate_passed"] is False
    assert report["formal_training_authorized"] is False


def test_payload_verification_mode_must_be_explicit(tmp_path):
    case = _make_case(tmp_path / "neither")
    case["rehash_payloads"] = False
    with pytest.raises(ValueError, match="choose exactly one"):
        _publish(case)

    case = _make_case(tmp_path / "both")
    _write_payload_hash_sidecars(case)
    case["rehash_payloads"] = True
    with pytest.raises(ValueError, match="choose exactly one"):
        _publish(case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "exact inventory mismatch"),
        ("extra", "exact inventory mismatch"),
        ("wrong_script", "script SHA256 mismatch"),
        ("wrong_sha", "sidecar/LFS SHA256 mismatch"),
        ("stale", "fingerprint is stale"),
        ("tampered", "fingerprint is stale"),
    ],
)
def test_payload_hash_sidecars_fail_closed(tmp_path, mutation, message):
    case = _make_case(tmp_path)
    sidecar_dir = _write_payload_hash_sidecars(case)
    payload = case["root"] / "data" / "mdcath_dataset_1abcA00.h5"
    sidecar = sidecar_dir / f"{payload.name}.json"
    if mutation == "missing":
        sidecar.unlink()
    elif mutation == "extra":
        (sidecar_dir / "unexpected.json").write_text("{}\n")
    elif mutation in {"wrong_script", "wrong_sha"}:
        row = json.loads(sidecar.read_text())
        if mutation == "wrong_script":
            row["hash_script_sha256"] = "d" * 64
        else:
            row["sha256"] = "d" * 64
        sidecar.write_text(json.dumps(row, sort_keys=True) + "\n")
    elif mutation == "stale":
        stat_result = payload.stat()
        os.utime(
            payload,
            ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000),
        )
    else:
        with payload.open("r+b") as handle:
            handle.seek(100)
            original = handle.read(1)
            handle.seek(100)
            handle.write(bytes([original[0] ^ 1]))

    with pytest.raises(ValueError, match=message):
        _publish(case)
    assert not case["manifest_output"].exists()
    assert not case["metadata_output"].exists()
    assert not case["audit_output"].exists()


def test_temporary_manifest_is_revalidated_before_atomic_replace(
    tmp_path, monkeypatch
):
    case = _make_case(tmp_path)
    sentinel = b"existing verified manifest\n"
    case["manifest_output"].write_bytes(sentinel)
    original_validate = auditor._validate_manifest
    calls = 0

    def fail_on_temporary(manifest, domains, expectations):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("temporary manifest validation failed")
        return original_validate(manifest, domains, expectations)

    monkeypatch.setattr(auditor, "_validate_manifest", fail_on_temporary)
    with pytest.raises(ValueError, match="temporary manifest validation failed"):
        _publish(case)

    assert calls == 2
    assert case["manifest_output"].read_bytes() == sentinel
    assert not case["metadata_output"].exists()
    assert not case["audit_output"].exists()
    assert not list(case["root"].glob(".*.tmp.*"))


@pytest.mark.parametrize("mutation", ["missing_replica", "nonfinite_last", "bad_frames"])
def test_structural_failure_never_replaces_existing_manifest(tmp_path, mutation):
    case = _make_case(tmp_path)
    target = case["root"] / "data" / "mdcath_dataset_1abcA00.h5"
    with h5py.File(target, "r+") as handle:
        domain = handle["1abcA00"]
        if mutation == "missing_replica":
            del domain["320"]["4"]
        elif mutation == "nonfinite_last":
            domain["320"]["4"]["coords"][-1, 0, 0] = np.nan
        else:
            domain["320"]["4"].attrs["numFrames"] = 3
    _refresh_source_identity(case)
    sentinel = b"existing verified manifest\n"
    case["manifest_output"].write_bytes(sentinel)

    with pytest.raises(ValueError):
        _publish(case)

    assert case["manifest_output"].read_bytes() == sentinel
    assert not case["metadata_output"].exists()
    assert not case["audit_output"].exists()
    assert not case["manifest_output"].with_name(
        case["manifest_output"].name + ".sha256"
    ).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "exact inventory mismatch"),
        ("extra", "exact inventory mismatch"),
        ("incomplete", "incomplete downloads remain"),
        ("failure_file", "not a canonical domain list"),
        ("symlink", "staging tree contains symlinks"),
    ],
)
def test_inventory_is_an_exact_regular_file_union(tmp_path, mutation, message):
    case = _make_case(tmp_path)
    data = case["root"] / "data"
    target = data / "mdcath_dataset_1abcA00.h5"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        _write_domain(data / "mdcath_dataset_extra.h5", "extra")
    elif mutation == "incomplete":
        (case["root"] / "shard_00.incomplete").write_text("partial")
    elif mutation == "failure_file":
        (case["root"] / "download_failures.txt").write_text("")
    else:
        outside = tmp_path / "outside.h5"
        target.rename(outside)
        target.symlink_to(outside)

    with pytest.raises(ValueError, match=message):
        _publish(case)


def test_huggingface_resumable_cache_fragments_are_retained_and_reported(
    tmp_path, capsys
):
    case = _make_case(tmp_path)
    cache = (
        case["root"]
        / ".cache"
        / "huggingface"
        / "download"
        / "data"
    )
    cache.mkdir(parents=True)
    (cache / "payload.abc123.incomplete").write_text("resumable cache")

    report = _publish(case)

    assert report["status"] == auditor.LIVE_REHASH_PASS_STATUS
    assert "retained_huggingface_cache_incomplete=1" in capsys.readouterr().out
    assert (cache / "payload.abc123.incomplete").is_file()


def test_resolved_download_failure_ledger_is_retained_and_reported(
    tmp_path, capsys
):
    case = _make_case(tmp_path)
    ledger = case["root"] / "download_failures.txt"
    ledger.write_text("1abcA00\n")

    report = _publish(case)

    assert report["status"] == auditor.LIVE_REHASH_PASS_STATUS
    assert (
        "retained_resolved_download_failure_ledger=1"
        in capsys.readouterr().out
    )
    assert ledger.read_text() == "1abcA00\n"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("unknown00\n", "contains unknown domains"),
        ("1abcA00\n1abcA00\n", "not a canonical domain list"),
        ("1abcA00", "not a canonical domain list"),
        ("\n", "not a canonical domain list"),
    ],
)
def test_invalid_download_failure_ledger_still_fails(
    tmp_path, content, message
):
    case = _make_case(tmp_path)
    (case["root"] / "download_failures.txt").write_text(content)

    with pytest.raises(ValueError, match=message):
        _publish(case)


def test_download_failure_ledger_with_missing_payload_still_fails(tmp_path):
    case = _make_case(tmp_path)
    (case["root"] / "download_failures.txt").write_text("1abcA00\n")
    (case["root"] / "data" / "mdcath_dataset_1abcA00.h5").unlink()

    with pytest.raises(ValueError, match="contains unresolved domains"):
        _publish(case)


def test_misplaced_download_failure_ledger_still_fails(tmp_path):
    case = _make_case(tmp_path)
    misplaced = case["root"] / "control" / "download_failures.txt"
    misplaced.parent.mkdir()
    misplaced.write_text("1abcA00\n")

    with pytest.raises(
        ValueError, match="ambiguous shared download failure file"
    ):
        _publish(case)


def test_incomplete_outside_huggingface_payload_cache_still_fails(tmp_path):
    case = _make_case(tmp_path)
    cache = (
        case["root"]
        / ".cache"
        / "huggingface"
        / "download"
        / "metadata"
    )
    cache.mkdir(parents=True)
    (cache / "payload.incomplete").write_text("not an allowed cache fragment")

    with pytest.raises(ValueError, match="incomplete downloads remain"):
        _publish(case)


def test_same_size_payload_tampering_is_detected_before_hdf5_scan(tmp_path):
    case = _make_case(tmp_path)
    target = case["root"] / "data" / "mdcath_dataset_1abcA00.h5"
    size_before = target.stat().st_size
    with target.open("r+b") as handle:
        handle.seek(100)
        original = handle.read(1)
        handle.seek(100)
        handle.write(bytes([original[0] ^ 1]))
    assert target.stat().st_size == size_before

    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        _publish(case)
    assert not case["manifest_output"].exists()
    assert not case["metadata_output"].exists()
    assert not case["audit_output"].exists()


def test_wrong_source_revision_inventory_digest_and_lfs_sha_are_rejected(tmp_path):
    case = _make_case(tmp_path / "revision")
    case["source_revision"] = "b" * 40
    with pytest.raises(ValueError, match="source revision"):
        _publish(case)

    case = _make_case(tmp_path / "inventory")
    old = case["expectations"]
    case["expectations"] = auditor.AuditExpectations(
        domains=old.domains,
        h5_bytes=old.h5_bytes,
        trajectories=old.trajectories,
        official_list_sha256=old.official_list_sha256,
        source_revision=old.source_revision,
        source_inventory_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="source inventory SHA256 mismatch"):
        _publish(case)

    case = _make_case(tmp_path / "lfs")
    rows = [json.loads(line) for line in case["source_inventory"].read_text().splitlines()]
    rows[0]["lfs_sha256"] = "f" * 64
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    case["source_inventory"].write_bytes(content)
    old = case["expectations"]
    case["expectations"] = auditor.AuditExpectations(
        domains=old.domains,
        h5_bytes=old.h5_bytes,
        trajectories=old.trajectories,
        official_list_sha256=old.official_list_sha256,
        source_revision=old.source_revision,
        source_inventory_sha256=_sha_bytes(content),
    )
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        _publish(case)


def test_per_file_official_size_is_checked_not_only_aggregate_bytes(tmp_path):
    case = _make_case(tmp_path)
    rows = [json.loads(line) for line in case["source_inventory"].read_text().splitlines()]
    rows[0]["size"] += 1
    rows[0]["lfs_size"] += 1
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    case["source_inventory"].write_bytes(content)
    old = case["expectations"]
    case["expectations"] = auditor.AuditExpectations(
        domains=old.domains,
        h5_bytes=old.h5_bytes + 1,
        trajectories=old.trajectories,
        official_list_sha256=old.official_list_sha256,
        source_revision=old.source_revision,
        source_inventory_sha256=_sha_bytes(content),
    )
    with pytest.raises(ValueError, match="per-file source size mismatch"):
        _publish(case)


def test_official_identity_requires_fixed_sha_count_and_uniqueness(tmp_path):
    case = _make_case(tmp_path)
    case["official_list"].write_text("1abcA00\n1abcA00\n")
    case["expectations"] = auditor.AuditExpectations(
        domains=2,
        h5_bytes=case["expectations"].h5_bytes,
        trajectories=50,
        official_list_sha256=hashlib.sha256(
            case["official_list"].read_bytes()
        ).hexdigest(),
        source_revision=case["expectations"].source_revision,
        source_inventory_sha256=case[
            "expectations"
        ].source_inventory_sha256,
    )
    with pytest.raises(ValueError, match="duplicates"):
        _publish(case)

    case = _make_case(tmp_path / "sha")
    case["expectations"] = auditor.AuditExpectations(
        domains=2,
        h5_bytes=case["expectations"].h5_bytes,
        trajectories=50,
        official_list_sha256="0" * 64,
        source_revision=case["expectations"].source_revision,
        source_inventory_sha256=case[
            "expectations"
        ].source_inventory_sha256,
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _publish(case)


def test_symlink_inputs_and_outputs_are_rejected(tmp_path):
    case = _make_case(tmp_path)
    official_link = tmp_path / "official-link.txt"
    official_link.symlink_to(case["official_list"])
    case["official_list"] = official_link
    with pytest.raises(ValueError, match="official domain list"):
        _publish(case)

    case = _make_case(tmp_path / "output")
    outside = tmp_path / "outside-manifest.json"
    outside.write_text("do not replace")
    case["manifest_output"].symlink_to(outside)
    with pytest.raises(ValueError, match="symlink output"):
        _publish(case)
    assert outside.read_text() == "do not replace"


def test_artifact_paths_and_sidecars_must_be_distinct(tmp_path):
    case = _make_case(tmp_path)
    case["metadata_output"] = case["manifest_output"].with_name(
        case["manifest_output"].name + ".sha256"
    )
    with pytest.raises(ValueError, match="must all be distinct"):
        _publish(case)


def test_production_full_mdcath_identity_is_frozen():
    assert auditor.EXPECTED_DOMAINS == 5_398
    assert auditor.EXPECTED_H5_BYTES == 3_613_998_101_757
    assert auditor.EXPECTED_TRAJECTORIES == 134_950
    assert auditor.EXPECTED_TRAJECTORIES == auditor.EXPECTED_DOMAINS * 25
    assert auditor.EXPECTED_OFFICIAL_LIST_SHA256 == (
        "295c6da1c9f8846a1ea3993eca12a3232d16a2b3a4b0d8791c7c45392186709b"
    )
    assert auditor.EXPECTED_SOURCE_REVISION == (
        "5e3ed8aec62b689e01751db16275fdcdbc39e47f"
    )
    assert auditor.EXPECTED_SOURCE_INVENTORY_SHA256 == (
        "2e6e3602a0858aaafc849cfa7cc1ee7e076736cb15335d7914726898f06f6cdf"
    )
