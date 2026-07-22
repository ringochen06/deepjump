import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.verify_paper_vector_source_stop as source_stop


SOURCE_RUNNER = Path("cloud/huawei/run_paper_horizon_ab2000.sh")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_decision(root: Path) -> Path:
    root.mkdir()
    decision = {
        "status": "STOP_PAPER_HORIZON_OBJECTIVE_GAIN",
        "candidate_checkpoint_sha256": "c" * 64,
        "external_development_authorized": False,
        "second_seed_scientifically_eligible": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    path = root / "decision.json"
    path.write_text(json.dumps(decision, sort_keys=True) + "\n")
    return path


def _patch_fixture_identity(monkeypatch, decision: Path) -> None:
    monkeypatch.setattr(source_stop, "SOURCE_DECISION_SHA256", _sha(decision))
    monkeypatch.setattr(source_stop, "SOURCE_CANDIDATE_CHECKPOINT_SHA256", "c" * 64)


def test_source_stop_binds_fixed_decision_and_runner_control_flow(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    decision = _write_decision(first)
    _write_decision(second)
    _patch_fixture_identity(monkeypatch, decision)
    report = source_stop.verify(first, second, SOURCE_RUNNER)
    assert report == {
        "schema": "deepjump.prior_source_control_flow_proof.v1",
        "status": "PASS_PRIOR_AUTHORITATIVE_RUN_EXTERNAL_UNCONSUMED",
        "source_run_id": source_stop.SOURCE_RUN_ID,
        "source_commit": source_stop.SOURCE_COMMIT,
        "source_audit_obs_uri": source_stop.SOURCE_AUDIT_OBS_URI,
        "source_decision_sha256": _sha(decision),
        "source_runner_sha256": source_stop.SOURCE_RUNNER_SHA256,
        "source_status": source_stop.SOURCE_STOP_STATUS,
        "required_advance_status": source_stop.SOURCE_ADVANCE_STATUS,
        "proof_basis": "fixed_decision_and_fixed_runner_control_flow",
        "prior_authoritative_run_consumed": False,
    }


def test_source_stop_does_not_trust_forgeable_dynamic_metadata(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    decision = _write_decision(first)
    _write_decision(second)
    _patch_fixture_identity(monkeypatch, decision)
    for root, marker in ((first, "FORGED-A"), (second, "FORGED-B")):
        (root / "summary.json").write_text(marker)
        (root / "external_status.json").write_text(marker)
        (root / "audit_sha256.txt").write_text(marker)
    report = source_stop.verify(first, second, SOURCE_RUNNER)
    assert "source_summary_sha256" not in report
    assert "source_external_status_sha256" not in report


def test_source_stop_rejects_shared_decision_inode(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    decision = _write_decision(first)
    second.mkdir()
    os.link(decision, second / "decision.json")
    _patch_fixture_identity(monkeypatch, decision)
    with pytest.raises(ValueError, match="independent"):
        source_stop.verify(first, second, SOURCE_RUNNER)


def test_source_stop_rejects_modified_runner(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    decision = _write_decision(first)
    _write_decision(second)
    _patch_fixture_identity(monkeypatch, decision)
    runner = tmp_path / "runner.sh"
    runner.write_bytes(SOURCE_RUNNER.read_bytes() + b"\n# modified\n")
    with pytest.raises(ValueError, match="runner SHA256"):
        source_stop.verify(first, second, runner)
