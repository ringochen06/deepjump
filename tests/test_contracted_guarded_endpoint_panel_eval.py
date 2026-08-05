import hashlib
import json
import os
from pathlib import Path

import h5py
import pytest

import scripts.contracted_guarded_endpoint_panel_eval as contracted_eval
import scripts.guarded_endpoint_panel_eval as guarded_eval
from scripts.contracted_guarded_endpoint_panel_eval import (
    PREREQUISITE_SCHEMA,
    _pinned_domain_handle,
    _pinned_mechanism_probe,
    _preflight_output_paths,
    rehash_contracted_panel_payloads,
    validate_evaluation_completeness,
    verify_reserved_evaluation_prerequisite,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reserved_evaluation_prerequisite_binds_exact_run(tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "schema": PREREQUISITE_SCHEMA,
        "authorization_id": "external-test-authorization",
        "consumption_ledger_root": str((tmp_path / "ledger").resolve()),
        "status": "ADVANCE_EXPANDED_DATA_EXTERNAL",
        "phase": "external",
        "checkpoint_sha256": "1" * 64,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": "2" * 64,
        "panel_name": "paper_horizon_external20",
        "panel_sha256": "3" * 64,
        "reserved_panel_authorized": True,
        "formal_training_authorized": False,
    }))
    report = verify_reserved_evaluation_prerequisite(
        decision,
        _sha(decision),
        phase="external",
        checkpoint_sha256="1" * 64,
        checkpoint_step=2_000,
        contract_sha256="2" * 64,
        panel_name="paper_horizon_external20",
        panel_sha256="3" * 64,
    )
    assert report["status"] == "ADVANCE_EXPANDED_DATA_EXTERNAL"
    assert report["formal_training_authorized"] is False


def test_reserved_evaluation_prerequisite_rejects_early_panel_access(tmp_path):
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "schema": PREREQUISITE_SCHEMA,
        "authorization_id": "external-stop-authorization",
        "consumption_ledger_root": str((tmp_path / "ledger").resolve()),
        "status": "STOP_EXPANDED_DATA_CANDIDATE",
        "phase": "external",
        "checkpoint_sha256": "1" * 64,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": "2" * 64,
        "panel_name": "paper_horizon_external20",
        "panel_sha256": "3" * 64,
        "reserved_panel_authorized": False,
        "formal_training_authorized": False,
    }))
    with pytest.raises(ValueError, match="does not authorize"):
        verify_reserved_evaluation_prerequisite(
            decision,
            _sha(decision),
            phase="external",
            checkpoint_sha256="1" * 64,
            checkpoint_step=2_000,
            contract_sha256="2" * 64,
            panel_name="paper_horizon_external20",
            panel_sha256="3" * 64,
        )


def test_consumption_claim_precedes_any_panel_identity_or_payload_read():
    source = Path("scripts/contracted_guarded_endpoint_panel_eval.py").read_text()
    main_source = source[source.index("def main()") :]
    assert main_source.index("claim_reserved_evaluation(") < main_source.index(
        "verify_frozen_evaluation_identity("
    )
    assert main_source.index("claim_reserved_evaluation(") < main_source.index(
        "rehash_contracted_panel_payloads("
    )
    assert "--consumption-ledger" not in main_source


def test_panel_payloads_are_live_rehashed_from_contracted_manifest(tmp_path):
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    payload = data / "mdcath_dataset_d1.h5"
    payload.write_bytes(b"exact-payload")
    stat_result = payload.stat()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "domain": "d1",
        "file": payload.name,
        "sha256": _sha(payload),
        "local_fingerprint": {
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "ctime_ns": stat_result.st_ctime_ns,
        },
    }]))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "artifacts": {
            "manifest": {"path": manifest.name, "sha256": _sha(manifest)},
        },
    }))
    panel = tmp_path / "panel.txt"
    panel.write_text("d1\n")
    report = rehash_contracted_panel_payloads(contract, panel, root)
    assert report["status"] == "PASS_CONTRACTED_PANEL_LIVE_PAYLOAD_REHASH"
    assert report["panel_domains"] == 1
    assert report["payloads"][0]["sha256"] == _sha(payload)

    payload.write_bytes(b"forged-payload")
    with pytest.raises(ValueError, match="fingerprint mismatch|SHA256 mismatch"):
        rehash_contracted_panel_payloads(contract, panel, root)


def _complete_cell():
    row = {
        "target_position_finite": True,
        "selected_position_exact": True,
        "selected_vector_exact": True,
        "noop_rmsd": 1.0,
        "source": {"position_finite": True, "vector_finite": True},
        "guarded": {
            "position_finite": True,
            "vector_finite": True,
            "rmsd": 0.9,
            "minus_noop": -0.1,
        },
    }
    return {
        "by_start": [dict(row) for _ in range(3)],
        "source_cell_physical": True,
        "guarded_cell_physical": True,
    }


def test_evidence_completeness_is_not_scientific_adjudication():
    report = validate_evaluation_completeness(
        [{"cells": [_complete_cell() for _ in range(25)]}], 1
    )
    assert report == {
        "status": "PASS_RESERVED_EVALUATION_EVIDENCE_COMPLETENESS",
        "domains": 1,
        "cells": 25,
        "starts": 75,
        "scientific_adjudication_performed": False,
    }


def test_evidence_completeness_rejects_nonfinite_or_missing_metric():
    cells = [_complete_cell() for _ in range(25)]
    cells[0]["by_start"][0]["guarded"]["rmsd"] = None
    with pytest.raises(ValueError, match="non-finite or incomplete"):
        validate_evaluation_completeness([{"cells": cells}], 1)


def test_pinned_payload_descriptor_has_explicit_lifetime(tmp_path):
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    payload = data / "mdcath_dataset_d1.h5"
    payload.write_bytes(b"exact-payload")
    stat_result = payload.stat()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "domain": "d1",
        "file": payload.name,
        "sha256": _sha(payload),
        "local_fingerprint": {
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "ctime_ns": stat_result.st_ctime_ns,
        },
    }]))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "artifacts": {
            "manifest": {"path": manifest.name, "sha256": _sha(manifest)},
        },
    }))
    panel = tmp_path / "panel.txt"
    panel.write_text("d1\n")
    report = rehash_contracted_panel_payloads(
        contract, panel, root, keep_open=True
    )
    pins = report["pins"]
    descriptor = pins[0]["descriptor"]
    assert os.fstat(descriptor).st_ino == stat_result.st_ino
    pins.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_pinned_domain_handle_duplicates_verified_descriptor(tmp_path):
    payload = tmp_path / "mdcath_dataset_d1.h5"
    with h5py.File(payload, "w") as handle:
        handle.create_group("d1")
    descriptor = os.open(payload, os.O_RDONLY)
    pin = {
        "domain": "d1",
        "path": payload,
        "descriptor": descriptor,
        "fd_path": Path(f"/proc/self/fd/{descriptor}"),
    }
    domain = _pinned_domain_handle(pin)
    try:
        assert domain.f["d1"].name == "/d1"
        domain.close()
        assert os.fstat(descriptor).st_ino == payload.stat().st_ino
    finally:
        domain.close()
        os.close(descriptor)


def test_pinned_mechanism_probe_forwards_verified_descriptor(monkeypatch):
    captured = {}

    def fake_probe(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(contracted_eval, "_mechanism_probe", fake_probe)
    pin = {
        "domain": "d1",
        "path": Path("/data/mdcath_dataset_d1.h5"),
        "descriptor": 17,
    }
    result = _pinned_mechanism_probe(
        {"checkpoint": True},
        object(),
        pin,
        320,
        0,
        1,
        False,
        object(),
    )

    assert result == {"status": "ok"}
    assert captured["args"][2] == pin["path"]
    assert captured["kwargs"] == {
        "domain_name": "d1",
        "descriptor": 17,
    }


def test_mechanism_probe_opens_from_verified_descriptor(monkeypatch):
    captured = {}

    class ProbeReached(Exception):
        pass

    def fake_handle(path, *, descriptor=None):
        captured["path"] = path
        captured["descriptor"] = descriptor
        raise ProbeReached

    monkeypatch.setattr(guarded_eval, "_DomainHandle", fake_handle)
    with pytest.raises(ProbeReached):
        guarded_eval._mechanism_probe(
            {},
            object(),
            Path("/data/mdcath_dataset_d1.h5"),
            320,
            0,
            1,
            False,
            object(),
            domain_name="d1",
            descriptor=17,
        )

    assert captured == {
        "path": Path("/data/mdcath_dataset_d1.h5"),
        "descriptor": 17,
    }


def test_output_preflight_rejects_collision_before_panel_use(tmp_path):
    output = tmp_path / "same.json"
    with pytest.raises(ValueError, match="must be distinct"):
        _preflight_output_paths(output, output)
    runtime = tmp_path / "runtime.json"
    output.write_text("already exists")
    with pytest.raises(FileExistsError, match="existing output"):
        _preflight_output_paths(runtime, output)
