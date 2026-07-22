import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.certify_paper_horizon_postrun import (
    EXPECTED_PASS_AUDIT_MANIFEST,
    certify,
)


RUN_ID = "20260722T012922Z"
SOURCE_COMMIT = "d" * 40
VERIFIER_COMMIT = "e" * 40
OBS = "obs://example/paper-horizon/20260722T012922Z"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def _write_manifest(path: Path, root: Path, names: list[str]) -> None:
    path.write_text("".join(f"{_sha(root / name)}  {name}\n" for name in names))


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "readback"
    audit = root / "audit"
    baseline = root / "baseline"
    candidate = root / "candidate"
    for directory in (audit, baseline, candidate):
        directory.mkdir(parents=True)
    checkpoint_names = [
        "config.json",
        "history.json",
        "last.ckpt",
        "ckpt_1000.pt",
        "ckpt_2000.pt",
    ]
    for directory, label in ((baseline, "baseline"), (candidate, "candidate")):
        for name in checkpoint_names:
            content_name = "ckpt_2000.pt" if name == "last.ckpt" else name
            (directory / name).write_bytes(f"{label}:{content_name}".encode())
    _write_manifest(audit / "baseline_sha256.txt", baseline, checkpoint_names)
    _write_manifest(audit / "candidate_sha256.txt", candidate, checkpoint_names)

    baseline_sha = _sha(baseline / "ckpt_2000.pt")
    candidate_sha = _sha(candidate / "ckpt_2000.pt")
    (audit / "obs_prefix_preflight.log").write_text(
        "Listing objects .\nFolder number: 0\nFile number: 0\n"
    )
    (audit / "pytest.log").write_text("89 passed\n")
    (audit / "runner.log").write_text("runner complete\n")
    _write_json(
        audit / "data_audit.json",
        {
            "status": "PASS",
            "h5_files": 1000,
            "h5_bytes": 668131379559,
            "manifest_domains": 1000,
            "manifest_trajectories": 25000,
            "subset_sha256": "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734",
            "unresolved_failures": 0,
            "hdf5_samples": 5,
        },
    )
    _write_json(
        audit / "baseline_checkpoint_2000_gate.json",
        {
            "status": "PASS",
            "checkpoint_step": 2000,
            "world_size": 8,
            "lr_horizon_steps": 1000,
            "nonfinite_model_tensors": [],
        },
    )
    _write_json(
        audit / "candidate_checkpoint_2000_gate.json",
        {
            "status": "PASS",
            "checkpoint_step": 2000,
            "world_size": 8,
            "lr_horizon_steps": 500000,
            "nonfinite_model_tensors": [],
        },
    )
    for name in ("baseline_panel.json", "candidate_panel.json"):
        _write_json(audit / name, {"panel": name})
    _write_json(audit / "baseline_decision.json", {"status": "BASELINE_PASS"})
    _write_json(audit / "candidate_decision.json", {"status": "CANDIDATE_PASS"})
    _write_json(
        audit / "training_ab_decision.json",
        {"status": "ADVANCE_PAPER_HORIZON_EXTERNAL20"},
    )
    for name in ("external_baseline_panel.json", "external_candidate_panel.json"):
        _write_json(audit / name, {"panel": name})
    _write_json(
        audit / "external_baseline_decision.json", {"status": "EXTERNAL_BASELINE_PASS"}
    )
    _write_json(
        audit / "external_candidate_decision.json", {"status": "EXTERNAL_CANDIDATE_PASS"}
    )
    _write_json(
        audit / "external_data_audit.json",
        {
            "status": "PASS",
            "domain_list_sha256": "9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb",
            "h5_files": 20,
            "total_bytes": 14236836972,
            "trajectories": 500,
            "unresolved_failures": 0,
        },
    )
    decision = {
        "status": "PASS_PAPER_HORIZON_EXTERNAL20",
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "external_development_authorized": False,
        "second_seed_scientifically_eligible": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    _write_json(audit / "decision.json", decision)
    summary = {
        **decision,
        "run_id": RUN_ID,
        "commit": SOURCE_COMMIT,
        "obs": OBS,
        "external_development_authorized": False,
        "completed_at": "2026-07-22T02:00:00+00:00",
    }
    summary = {
        key: summary[key]
        for key in (
            "status",
            "external_development_authorized",
            "second_seed_scientifically_eligible",
            "second_seed_authorized",
            "untouched_confirmation_authorized",
            "formal_training_authorized",
            "run_id",
            "commit",
            "baseline_checkpoint_sha256",
            "candidate_checkpoint_sha256",
            "obs",
            "completed_at",
        )
    }
    _write_json(audit / "summary.json", summary)
    audit_names = sorted(EXPECTED_PASS_AUDIT_MANIFEST)
    for name in audit_names:
        path = audit / name
        if not path.exists():
            path.write_text("{}\n" if name.endswith(".json") else f"{name}\n")
    _write_manifest(audit / "audit_sha256.txt", audit, audit_names)
    common = {
        "run_id": RUN_ID,
        "commit": SOURCE_COMMIT,
        "scientific_status": decision["status"],
        "decision_sha256": _sha(audit / "decision.json"),
        "audit_manifest_sha256": _sha(audit / "audit_sha256.txt"),
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "second_seed_scientifically_eligible": True,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "completed_at": "2026-07-22T02:00:01+00:00",
    }
    _write_json(
        audit / "authorization.json",
        {
            **common,
            "status": "SECOND_SEED_AUTHORIZED",
            "second_seed_authorized": True,
            "authorization_requires_independent_readback": True,
        },
    )
    _write_json(
        audit / "readback_completion.json",
        {
            **common,
            "status": "OBS_DOUBLE_READBACK_PASS",
            "second_seed_authorized": False,
        },
    )
    _write_manifest(
        audit / "final_markers.sha256",
        audit,
        ["authorization.json", "readback_completion.json"],
    )
    return root


def _patch_recomputation(monkeypatch) -> None:
    import scripts.certify_paper_horizon_postrun as module

    monkeypatch.setattr(module, "validate_checkpoint", lambda *args, **kwargs: ({}, []))

    def guarded(result_path, *args, **kwargs):
        name = Path(result_path).name.replace("_panel.json", "_decision.json")
        return json.loads((Path(result_path).parent / name).read_text())

    def horizon(baseline_decision_path, *args, panel_kind="training", **kwargs):
        root = Path(baseline_decision_path).parent
        name = "training_ab_decision.json" if panel_kind == "training" else "decision.json"
        return json.loads((root / name).read_text())

    monkeypatch.setattr(module, "adjudicate_guarded", guarded)
    monkeypatch.setattr(module, "adjudicate_horizon", horizon)


def _certify(root: Path, monkeypatch) -> dict:
    _patch_recomputation(monkeypatch)
    second = root.parent / "readback_two"
    shutil.copytree(root, second)
    return certify(
        root,
        second,
        Path(__file__).parents[1],
        expected_run_id=RUN_ID,
        expected_source_commit=SOURCE_COMMIT,
        expected_verifier_commit=VERIFIER_COMMIT,
        expected_obs=OBS,
    )


def test_postrun_certification_accepts_fully_bound_second_readback(tmp_path, monkeypatch):
    report = _certify(_fixture(tmp_path), monkeypatch)
    assert report["status"] == "PASS_PAPER_HORIZON_POSTRUN_CERTIFICATION"
    assert report["second_seed_authorized"] is True
    assert report["untouched_confirmation_authorized"] is False
    assert report["formal_training_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_prefix_count",
        "checkpoint_tamper",
        "decision_tamper",
        "authorization_flag",
        "completion_flag",
        "summary_obs",
    ],
)
def test_postrun_certification_rejects_tampering(tmp_path, mutation, monkeypatch):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    audit = root / "audit"
    if mutation == "duplicate_prefix_count":
        with (audit / "obs_prefix_preflight.log").open("a") as handle:
            handle.write("File number: 1\n")
    elif mutation == "checkpoint_tamper":
        with (root / "candidate" / "ckpt_2000.pt").open("ab") as handle:
            handle.write(b"tamper")
    elif mutation == "decision_tamper":
        decision = json.loads((audit / "decision.json").read_text())
        decision["second_seed_scientifically_eligible"] = False
        _write_json(audit / "decision.json", decision)
    elif mutation == "authorization_flag":
        authorization = json.loads((audit / "authorization.json").read_text())
        authorization["formal_training_authorized"] = True
        _write_json(audit / "authorization.json", authorization)
    elif mutation == "completion_flag":
        completion = json.loads((audit / "readback_completion.json").read_text())
        completion["second_seed_authorized"] = True
        _write_json(audit / "readback_completion.json", completion)
    elif mutation == "summary_obs":
        summary = json.loads((audit / "summary.json").read_text())
        summary["obs"] = "obs://wrong"
        _write_json(audit / "summary.json", summary)
    with pytest.raises(ValueError):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_manifest_path_escape(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    audit = root / "audit"
    outside = tmp_path / "outside"
    outside.write_text("outside")
    for manifest in (
        audit / "audit_sha256.txt",
        second / "audit" / "audit_sha256.txt",
    ):
        with manifest.open("a") as handle:
            handle.write(f"{_sha(outside)}  ../outside\n")
    with pytest.raises(ValueError, match="expected exact entries"):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_readback_inventory_difference(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    with (second / "audit" / "runner.log").open("a") as handle:
        handle.write("changed\n")
    with pytest.raises(ValueError, match="inventories differ"):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_same_readback_root(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    with pytest.raises(ValueError, match="roots must be different"):
        certify(
            root,
            root,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_shared_file_inode(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    shared = second / "audit" / "runner.log"
    shared.unlink()
    shared.hardlink_to(root / "audit" / "runner.log")
    with pytest.raises(ValueError, match="share file inodes"):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_unmanifested_checkpoint_file(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    for readback in (root, second):
        (readback / "candidate" / "extra.ckpt").write_bytes(b"extra")
    with pytest.raises(ValueError, match="missing or extra files"):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_self_consistent_manifested_audit_extra(
    tmp_path, monkeypatch
):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    for readback in (root, second):
        audit = readback / "audit"
        (audit / "unexpected.txt").write_text("stable unexpected evidence\n")
        with (audit / "audit_sha256.txt").open("a") as handle:
            handle.write(f"{_sha(audit / 'unexpected.txt')}  unexpected.txt\n")
        audit_sha = _sha(audit / "audit_sha256.txt")
        for marker_name in ("authorization.json", "readback_completion.json"):
            marker = json.loads((audit / marker_name).read_text())
            marker["audit_manifest_sha256"] = audit_sha
            _write_json(audit / marker_name, marker)
        _write_manifest(
            audit / "final_markers.sha256",
            audit,
            ["authorization.json", "readback_completion.json"],
        )
    with pytest.raises(ValueError, match="expected exact entries"):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


def test_postrun_certification_rejects_stable_extra_permission_alias(
    tmp_path, monkeypatch
):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    for readback in (root, second):
        audit = readback / "audit"
        authorization = json.loads((audit / "authorization.json").read_text())
        authorization["formal_training_authorized_override"] = True
        _write_json(audit / "authorization.json", authorization)
        _write_manifest(
            audit / "final_markers.sha256",
            audit,
            ["authorization.json", "readback_completion.json"],
        )
    with pytest.raises(ValueError, match="exact schema mismatch"):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )


@pytest.mark.parametrize(
    ("audit_name", "key", "value", "message"),
    [
        ("data_audit.json", "h5_bytes", 1, "training data audit h5_bytes mismatch"),
        (
            "external_data_audit.json",
            "status",
            "FAIL",
            "external data audit status mismatch",
        ),
    ],
)
def test_postrun_certification_rejects_self_consistent_wrong_data_audit(
    tmp_path, monkeypatch, audit_name, key, value, message
):
    root = _fixture(tmp_path)
    _patch_recomputation(monkeypatch)
    second = tmp_path / "readback_two"
    shutil.copytree(root, second)
    for readback in (root, second):
        audit = readback / "audit"
        payload = json.loads((audit / audit_name).read_text())
        payload[key] = value
        _write_json(audit / audit_name, payload)
        _write_manifest(
            audit / "audit_sha256.txt", audit, sorted(EXPECTED_PASS_AUDIT_MANIFEST)
        )
        audit_sha = _sha(audit / "audit_sha256.txt")
        for marker_name in ("authorization.json", "readback_completion.json"):
            marker = json.loads((audit / marker_name).read_text())
            marker["audit_manifest_sha256"] = audit_sha
            _write_json(audit / marker_name, marker)
        _write_manifest(
            audit / "final_markers.sha256",
            audit,
            ["authorization.json", "readback_completion.json"],
        )
    with pytest.raises(ValueError, match=message):
        certify(
            root,
            second,
            Path(__file__).parents[1],
            expected_run_id=RUN_ID,
            expected_source_commit=SOURCE_COMMIT,
            expected_verifier_commit=VERIFIER_COMMIT,
            expected_obs=OBS,
        )
