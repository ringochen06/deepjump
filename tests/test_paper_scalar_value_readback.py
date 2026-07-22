import hashlib
import json
from pathlib import Path

import pytest

import scripts.adjudicate_paper_scalar_value_ab as scalar_value_ab
from scripts.adjudicate_paper_scalar_value_ab import adjudicate
from scripts.adjudicate_paper_vector_ab import EXPECTED_STEPS
from scripts.verify_paper_scalar_value_readback import (
    ARM_FILES,
    AUDIT_FILES,
    COMPLETION_ADDITIONS,
    INITIAL_UNMANIFESTED,
    verify,
    verify_pair,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _restore_scalar_value_trust_root():
    names = (
        "EXPECTED_BASELINE_CHECKPOINT_SHA256",
        "EXPECTED_BASELINE_HISTORY_SHA256",
        "EXPECTED_BASELINE_DECISION_SHA256",
        "EXPECTED_BASELINE_FINAL_VAL_LOSS",
        "EXPECTED_BASELINE_ABSOLUTE_STATUS",
    )
    original = {name: getattr(scalar_value_ab, name) for name in names}
    yield
    for name, value in original.items():
        setattr(scalar_value_ab, name, value)


def _write_files(root: Path, names: set[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text(f"fixture:{name}\n")


def _write_manifest(path: Path, root: Path, names: set[str]) -> None:
    path.write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in sorted(names)
    ))


def _fixture(root: Path, *, completion: bool) -> Path:
    baseline = root / "baseline"
    candidate = root / "candidate"
    audit = root / "audit"
    _write_files(baseline, set(ARM_FILES))
    _write_files(candidate, set(ARM_FILES))
    (candidate / "ckpt_2000.pt").write_text("candidate checkpoint fixture\n")
    _write_files(audit, set(AUDIT_FILES))
    run_id = "20260722T120000Z"
    commit = "f" * 40
    baseline_sha = _sha(baseline / "ckpt_2000.pt")
    candidate_sha = _sha(candidate / "ckpt_2000.pt")
    baseline_history = [
        {
            "step": step,
            "val_loss": 4.2 + (2000 - step) * 1e-5,
            "val_rmsd": 2.0,
            "noop_rmsd": 3.0,
        }
        for step in EXPECTED_STEPS
    ]
    candidate_history = [
        {
            "step": step,
            "val_loss": 4.0 + (2000 - step) * 1e-5,
            "val_rmsd": 2.0,
            "noop_rmsd": 3.0,
        }
        for step in EXPECTED_STEPS
    ]
    (baseline / "history.json").write_text(json.dumps(baseline_history))
    (candidate / "history.json").write_text(json.dumps(candidate_history))
    training_sha = "c" * 64
    domain_sha = "d" * 64
    baseline_absolute = {
        "status": "STOP_CONDITIONAL_SAFEGUARD_FALLBACK_CAP",
        "checkpoint_profile": "paper-horizon-vector-only-500k",
        "checkpoint_step": 2000,
        "checkpoint_sha256": baseline_sha,
        "training_domain_list_sha256": training_sha,
        "domain_list_sha256": domain_sha,
        "domains": 20,
        "starts": 1500,
        "domain_mean_guarded_minus_noop": [0.1 + 0.01 * (index % 4) for index in range(20)],
    }
    candidate_absolute = {
        "status": "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20",
        "checkpoint_profile": "paper-horizon-vector-scalar-value-500k",
        "checkpoint_step": 2000,
        "checkpoint_sha256": candidate_sha,
        "training_domain_list_sha256": training_sha,
        "domain_list_sha256": domain_sha,
        "domains": 20,
        "starts": 1500,
        "domain_mean_guarded_minus_noop": [
            -0.1 + 0.01 * (index % 5) for index in range(20)
        ],
    }
    (audit / "sealed_baseline_decision.json").write_text(
        json.dumps(baseline_absolute)
    )
    (audit / "baseline_decision.json").write_text(json.dumps(baseline_absolute))
    (audit / "candidate_decision.json").write_text(json.dumps(candidate_absolute))
    (audit / "candidate_h20.json").write_text(json.dumps({
        "checkpoint_sha256": candidate_sha,
        "checkpoint_step": 2000,
        "delta_frames": 1,
        "domain_panel": {
            "sha256": "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af",
            "evaluated_count": 3,
        },
        "settings": {
            "domains": 3,
            "starts": 2,
            "steps": 20,
            "methods": "mean,ode_1",
            "seed": 20260718,
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
        },
        "summary": {
            "noop": {"mean_final_rmsd": 2.0, "finite": True},
            "mean": {
                "mean_final_rmsd": 1.8,
                "mean_final_bond_mean": 3.8,
                "mean_final_bond_max": 5.0,
                "finite": True,
            },
            "ode_1": {
                "mean_final_rmsd": 2.5,
                "mean_final_bond_mean": 6.0,
                "mean_final_bond_max": 8.0,
                "finite": True,
            },
        },
    }))
    authorizations = {
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    evidence = {
        "schema": "deepjump.scalar_value_training_evidence.v1",
        "run_id": run_id,
        "commit": commit,
        "candidate_config_sha256": _sha(audit / "candidate_config.yaml"),
        "baseline_decision_sha256": _sha(
            audit / "sealed_baseline_decision.json"
        ),
        "baseline_replay_decision_sha256": _sha(
            audit / "baseline_decision.json"
        ),
        "candidate_decision_sha256": _sha(audit / "candidate_decision.json"),
        "baseline_history_sha256": _sha(baseline / "history.json"),
        "candidate_history_sha256": _sha(candidate / "history.json"),
        "candidate_h20_sha256": _sha(audit / "candidate_h20.json"),
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "training_domain_list_sha256": training_sha,
        "domain_list_sha256": domain_sha,
    }
    (audit / "training_evidence.json").write_text(json.dumps(evidence))
    scalar_value_ab.EXPECTED_BASELINE_CHECKPOINT_SHA256 = baseline_sha
    scalar_value_ab.EXPECTED_BASELINE_HISTORY_SHA256 = _sha(
        baseline / "history.json"
    )
    scalar_value_ab.EXPECTED_BASELINE_DECISION_SHA256 = _sha(
        audit / "sealed_baseline_decision.json"
    )
    scalar_value_ab.EXPECTED_BASELINE_FINAL_VAL_LOSS = baseline_history[-1][
        "val_loss"
    ]
    scalar_value_ab.EXPECTED_BASELINE_ABSOLUTE_STATUS = baseline_absolute[
        "status"
    ]
    decision = adjudicate(
        audit / "sealed_baseline_decision.json",
        audit / "baseline_decision.json",
        audit / "candidate_decision.json",
        baseline / "history.json",
        candidate / "history.json",
        audit / "candidate_h20.json",
        audit / "training_evidence.json",
    )
    (audit / "decision.json").write_text(json.dumps(decision))
    summary = {
        "status": decision["status"],
        "run_id": run_id,
        "commit": commit,
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "external_development_scientifically_eligible": decision[
            "external_development_scientifically_eligible"
        ],
        **authorizations,
    }
    (audit / "summary.json").write_text(json.dumps(summary))
    _write_manifest(audit / "baseline_sha256.txt", baseline, set(ARM_FILES))
    _write_manifest(audit / "candidate_sha256.txt", candidate, set(ARM_FILES))
    _write_manifest(audit / "audit_sha256.txt", audit, set(AUDIT_FILES))
    _write_manifest(
        audit / "readback_manifests.sha256",
        audit,
        {"baseline_sha256.txt", "candidate_sha256.txt", "audit_sha256.txt"},
    )
    if completion:
        (audit / "readback_completion.json").write_text(json.dumps({
            "status": "OBS_PRECOMPLETION_DOUBLE_READBACK_PASS",
            "run_id": run_id,
            "commit": commit,
            "scientific_status": decision["status"],
            "decision_sha256": _sha(audit / "decision.json"),
            "audit_manifest_sha256": _sha(audit / "audit_sha256.txt"),
            "baseline_checkpoint_sha256": baseline_sha,
            "candidate_checkpoint_sha256": candidate_sha,
            **authorizations,
        }))
        _write_manifest(
            audit / "final_marker.sha256",
            audit,
            {"readback_completion.json", "readback_manifests.sha256"},
        )
    expected = set(AUDIT_FILES) | set(INITIAL_UNMANIFESTED)
    if completion:
        expected |= set(COMPLETION_ADDITIONS)
    assert {path.name for path in audit.iterdir()} == expected
    return root


@pytest.mark.parametrize("completion", [False, True])
def test_scalar_value_readback_requires_exact_inventory(tmp_path, completion):
    root = _fixture(tmp_path, completion=completion)
    report = verify(root, "completion" if completion else "initial")
    assert report["status"] == "PASS"


def test_scalar_value_readback_rejects_extra_object(tmp_path):
    root = _fixture(tmp_path, completion=True)
    (root / "audit" / "unexpected.txt").write_text("must fail\n")
    with pytest.raises(ValueError, match="exact inventory"):
        verify(root, "completion")


@pytest.mark.parametrize("key", [
    "external_development_authorized",
    "second_seed_authorized",
    "untouched_confirmation_authorized",
    "formal_training_authorized",
])
def test_scalar_value_readback_rejects_any_authorization(tmp_path, key):
    root = _fixture(tmp_path, completion=True)
    completion = root / "audit" / "readback_completion.json"
    payload = json.loads(completion.read_text())
    payload[key] = True
    completion.write_text(json.dumps(payload))
    _write_manifest(
        root / "audit" / "final_marker.sha256",
        root / "audit",
        {"readback_completion.json", "readback_manifests.sha256"},
    )
    with pytest.raises(ValueError, match=f"completion must keep {key}=false"):
        verify(root, "completion")


@pytest.mark.parametrize("mutation", [
    "status", "decision_sha256", "audit_manifest_sha256",
    "baseline_checkpoint_sha256", "candidate_checkpoint_sha256", "run_id", "commit",
])
def test_scalar_value_readback_rejects_reanchored_semantic_forgery(
    tmp_path, mutation
):
    root = _fixture(tmp_path, completion=True)
    completion = root / "audit" / "readback_completion.json"
    payload = json.loads(completion.read_text())
    payload[mutation] = "wrong"
    completion.write_text(json.dumps(payload))
    _write_manifest(
        root / "audit" / "final_marker.sha256",
        root / "audit",
        {"readback_completion.json", "readback_manifests.sha256"},
    )
    with pytest.raises(ValueError):
        verify(root, "completion")


@pytest.mark.parametrize("key", [
    "candidate_config_sha256",
    "baseline_decision_sha256",
    "baseline_replay_decision_sha256",
    "candidate_decision_sha256",
    "baseline_history_sha256",
    "candidate_history_sha256",
    "candidate_h20_sha256",
    "training_domain_list_sha256",
    "domain_list_sha256",
])
def test_scalar_value_readback_rejects_reanchored_incomplete_evidence(
    tmp_path, key
):
    root = _fixture(tmp_path, completion=True)
    audit = root / "audit"
    evidence_path = audit / "training_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence.pop(key)
    evidence_path.write_text(json.dumps(evidence))
    decision_path = audit / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["training_evidence"] = evidence
    decision["training_evidence_manifest_sha256"] = _sha(evidence_path)
    decision_path.write_text(json.dumps(decision))
    _write_manifest(audit / "audit_sha256.txt", audit, set(AUDIT_FILES))
    _write_manifest(
        audit / "readback_manifests.sha256",
        audit,
        {"baseline_sha256.txt", "candidate_sha256.txt", "audit_sha256.txt"},
    )
    completion_path = audit / "readback_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["decision_sha256"] = _sha(decision_path)
    completion["audit_manifest_sha256"] = _sha(audit / "audit_sha256.txt")
    completion_path.write_text(json.dumps(completion))
    _write_manifest(
        audit / "final_marker.sha256",
        audit,
        {"readback_completion.json", "readback_manifests.sha256"},
    )
    with pytest.raises(ValueError, match=f"training evidence {key} mismatch"):
        verify(root, "completion")


def test_scalar_value_readback_rejects_reanchored_sealed_identity_forgery(
    tmp_path,
):
    root = _fixture(tmp_path, completion=True)
    audit = root / "audit"
    sealed_path = audit / "sealed_baseline_decision.json"
    sealed = json.loads(sealed_path.read_text())
    sealed["checkpoint_sha256"] = "9" * 64
    sealed["training_domain_list_sha256"] = "8" * 64
    sealed["domain_list_sha256"] = "7" * 64
    sealed_path.write_text(json.dumps(sealed))
    evidence_path = audit / "training_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["baseline_decision_sha256"] = _sha(sealed_path)
    evidence_path.write_text(json.dumps(evidence))
    decision_path = audit / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["training_evidence"] = evidence
    decision["training_evidence_manifest_sha256"] = _sha(evidence_path)
    decision_path.write_text(json.dumps(decision))
    _write_manifest(audit / "audit_sha256.txt", audit, set(AUDIT_FILES))
    _write_manifest(
        audit / "readback_manifests.sha256",
        audit,
        {"baseline_sha256.txt", "candidate_sha256.txt", "audit_sha256.txt"},
    )
    completion_path = audit / "readback_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["decision_sha256"] = _sha(decision_path)
    completion["audit_manifest_sha256"] = _sha(audit / "audit_sha256.txt")
    completion_path.write_text(json.dumps(completion))
    _write_manifest(
        audit / "final_marker.sha256",
        audit,
        {"readback_completion.json", "readback_manifests.sha256"},
    )
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        verify(root, "completion")


def test_scalar_value_readback_recomputes_and_rejects_false_advance(tmp_path):
    root = _fixture(tmp_path, completion=True)
    audit = root / "audit"
    candidate_absolute_path = audit / "candidate_decision.json"
    candidate_absolute = json.loads(candidate_absolute_path.read_text())
    candidate_absolute["status"] = "STOP_CONDITIONAL_SAFEGUARD_NO_ADVANTAGE"
    candidate_absolute_path.write_text(json.dumps(candidate_absolute))
    evidence_path = audit / "training_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["candidate_decision_sha256"] = _sha(candidate_absolute_path)
    evidence_path.write_text(json.dumps(evidence))
    decision_path = audit / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["training_evidence"] = evidence
    decision["training_evidence_manifest_sha256"] = _sha(evidence_path)
    decision["candidate_absolute_status"] = candidate_absolute["status"]
    decision["candidate_absolute_pass"] = False
    decision["status"] = "ADVANCE_SCALAR_VALUE_EXTERNAL20"
    decision["external_development_scientifically_eligible"] = True
    decision_path.write_text(json.dumps(decision))
    summary_path = audit / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = decision["status"]
    summary["external_development_scientifically_eligible"] = True
    summary_path.write_text(json.dumps(summary))
    _write_manifest(audit / "audit_sha256.txt", audit, set(AUDIT_FILES))
    _write_manifest(
        audit / "readback_manifests.sha256",
        audit,
        {"baseline_sha256.txt", "candidate_sha256.txt", "audit_sha256.txt"},
    )
    completion_path = audit / "readback_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["scientific_status"] = decision["status"]
    completion["decision_sha256"] = _sha(decision_path)
    completion["audit_manifest_sha256"] = _sha(audit / "audit_sha256.txt")
    completion_path.write_text(json.dumps(completion))
    _write_manifest(
        audit / "final_marker.sha256",
        audit,
        {"readback_completion.json", "readback_manifests.sha256"},
    )
    with pytest.raises(ValueError, match="does not match recomputation"):
        verify(root, "completion")


def test_scalar_value_readback_rejects_fully_reanchored_baseline_replacement(
    tmp_path,
):
    root = _fixture(tmp_path, completion=True)
    audit = root / "audit"
    baseline = root / "baseline"
    candidate = root / "candidate"
    (baseline / "ckpt_2000.pt").write_text("forged baseline checkpoint\n")
    forged_baseline_sha = _sha(baseline / "ckpt_2000.pt")
    sealed_path = audit / "sealed_baseline_decision.json"
    replay_path = audit / "baseline_decision.json"
    for path in (sealed_path, replay_path):
        payload = json.loads(path.read_text())
        payload["checkpoint_sha256"] = forged_baseline_sha
        path.write_text(json.dumps(payload))
    evidence_path = audit / "training_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["baseline_checkpoint_sha256"] = forged_baseline_sha
    evidence["baseline_decision_sha256"] = _sha(sealed_path)
    evidence["baseline_replay_decision_sha256"] = _sha(replay_path)
    evidence_path.write_text(json.dumps(evidence))
    forged_decision = adjudicate(
        sealed_path,
        replay_path,
        audit / "candidate_decision.json",
        baseline / "history.json",
        candidate / "history.json",
        audit / "candidate_h20.json",
        evidence_path,
        expected_baseline_checkpoint_sha256=forged_baseline_sha,
        expected_baseline_history_sha256=_sha(baseline / "history.json"),
        expected_baseline_decision_sha256=_sha(sealed_path),
        expected_baseline_final_val_loss=json.loads(
            (baseline / "history.json").read_text()
        )[-1]["val_loss"],
        expected_baseline_absolute_status=json.loads(sealed_path.read_text())[
            "status"
        ],
    )
    decision_path = audit / "decision.json"
    decision_path.write_text(json.dumps(forged_decision))
    summary_path = audit / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = forged_decision["status"]
    summary["baseline_checkpoint_sha256"] = forged_baseline_sha
    summary["external_development_scientifically_eligible"] = forged_decision[
        "external_development_scientifically_eligible"
    ]
    summary_path.write_text(json.dumps(summary))
    _write_manifest(audit / "baseline_sha256.txt", baseline, set(ARM_FILES))
    _write_manifest(audit / "audit_sha256.txt", audit, set(AUDIT_FILES))
    _write_manifest(
        audit / "readback_manifests.sha256",
        audit,
        {"baseline_sha256.txt", "candidate_sha256.txt", "audit_sha256.txt"},
    )
    completion_path = audit / "readback_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["scientific_status"] = forged_decision["status"]
    completion["decision_sha256"] = _sha(decision_path)
    completion["audit_manifest_sha256"] = _sha(audit / "audit_sha256.txt")
    completion["baseline_checkpoint_sha256"] = forged_baseline_sha
    completion_path.write_text(json.dumps(completion))
    _write_manifest(
        audit / "final_marker.sha256",
        audit,
        {"readback_completion.json", "readback_manifests.sha256"},
    )
    with pytest.raises(ValueError, match="sealed vector-only artifact"):
        verify(root, "completion")


def test_scalar_value_readback_pair_requires_independent_files(tmp_path):
    first = _fixture(tmp_path / "one", completion=False)
    second = _fixture(tmp_path / "two", completion=False)
    assert verify_pair(first, second, "initial")["status"] == (
        "PASS_INDEPENDENT_DOUBLE_READBACK"
    )
    target = second / "audit" / "summary.json"
    target.unlink()
    target.hardlink_to(first / "audit" / "summary.json")
    with pytest.raises(ValueError, match="share file inodes"):
        verify_pair(first, second, "initial")
