import hashlib
import json
from pathlib import Path

import pytest
import torch

import scripts.adjudicate_paper_vector_ab as vector_ab
from deepjump.config import load_config, to_dict
from scripts.external_endpoint_identity import (
    verify_multidomain_checkpoint,
    verify_paper_vector_ab_prerequisite,
)
from scripts.guarded_endpoint_panel_eval import (
    PAPER_HORIZON_PROFILE,
    PAPER_SCALAR_VALUE_PROFILE,
    PAPER_VECTOR_PROFILE,
    checkpoint_profile_requirements,
)


def _write_history(path: Path, val_loss: float) -> None:
    path.write_text(json.dumps([
        {
            "step": step,
            "val_loss": val_loss + (2000 - step) * 1e-5,
            "val_rmsd": 2.0,
            "noop_rmsd": 3.0,
        }
        for step in vector_ab.EXPECTED_STEPS
    ]))


def _write_decision(path: Path, profile: str, values: list[float], status: str) -> None:
    path.write_text(json.dumps({
        "status": status,
        "checkpoint_profile": profile,
        "checkpoint_step": 2000,
        "checkpoint_sha256": (
            vector_ab.EXPECTED_BASELINE_CHECKPOINT_SHA256
            if profile == PAPER_HORIZON_PROFILE else "b" * 64
        ),
        "training_domain_list_sha256": "c" * 64,
        "domain_list_sha256": "d" * 64,
        "domains": 20,
        "starts": 1500,
        "domain_mean_guarded_minus_noop": values,
    }))


def _write_h20(path: Path, *, rmsd: float = 1.8, bond_mean: float = 3.8) -> None:
    path.write_text(json.dumps({
        "checkpoint_sha256": "b" * 64,
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
                "mean_final_rmsd": rmsd,
                "mean_final_bond_mean": bond_mean,
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


def _case(root: Path, monkeypatch) -> tuple[Path, Path, Path, Path, Path]:
    baseline_decision = root / "baseline_decision.json"
    candidate_decision = root / "candidate_decision.json"
    baseline_history = root / "baseline_history.json"
    candidate_history = root / "candidate_history.json"
    h20 = root / "h20.json"
    baseline = [0.10 + 0.01 * (index % 4) for index in range(20)]
    candidate = [value - 0.20 + 0.01 * (index % 5) for index, value in enumerate(baseline)]
    _write_decision(
        baseline_decision, PAPER_HORIZON_PROFILE, baseline,
        vector_ab.ABSOLUTE_PASS,
    )
    _write_decision(
        candidate_decision, PAPER_VECTOR_PROFILE, candidate,
        vector_ab.ABSOLUTE_PASS,
    )
    _write_history(baseline_history, 4.2)
    _write_history(candidate_history, 4.0)
    _write_h20(h20)
    monkeypatch.setattr(
        vector_ab, "EXPECTED_BASELINE_HISTORY_SHA256",
        hashlib.sha256(baseline_history.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(vector_ab, "EXPECTED_BASELINE_FINAL_VAL_LOSS", 4.2)
    return baseline_decision, candidate_decision, baseline_history, candidate_history, h20


def test_paper_vector_ab_requires_all_training_gates(tmp_path, monkeypatch):
    paths = _case(tmp_path, monkeypatch)
    report = vector_ab.adjudicate(*paths[:4], candidate_h20_path=paths[4])
    assert report["status"] == "ADVANCE_PAPER_VECTOR_EXTERNAL20"
    assert report["baseline_reproduced"] is True
    assert report["candidate_absolute_pass"] is True
    assert report["objective"]["passes"] is True
    assert report["paired_pass"] is True
    assert report["candidate_h20_gate"]["passes"] is True
    assert "candidate_minus_baseline" in next(iter(report["paired_primary"]))
    assert report["external_development_authorized"] is True
    assert report["second_seed_authorized"] is False
    assert report["formal_training_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("baseline", "STOP_PAPER_VECTOR_BASELINE_REPRODUCIBILITY"),
        ("absolute", "STOP_PAPER_VECTOR_ABSOLUTE_GATE"),
        ("objective", "STOP_PAPER_VECTOR_OBJECTIVE_GAIN"),
        ("paired", "STOP_PAPER_VECTOR_PAIRED_ADVANTAGE"),
        ("h20", "STOP_PAPER_VECTOR_H20_GATE"),
    ],
)
def test_paper_vector_ab_stops_on_each_failed_conjunct(
    tmp_path, monkeypatch, mutation, expected
):
    paths = _case(tmp_path, monkeypatch)
    if mutation == "baseline":
        payload = json.loads(paths[0].read_text())
        payload["status"] = "STOP_CONDITIONAL_SAFEGUARD_NO_ADVANTAGE"
        paths[0].write_text(json.dumps(payload))
    elif mutation == "absolute":
        payload = json.loads(paths[1].read_text())
        payload["status"] = "STOP_CONDITIONAL_SAFEGUARD_NO_ADVANTAGE"
        paths[1].write_text(json.dumps(payload))
    elif mutation == "objective":
        _write_history(paths[3], 4.19)
    elif mutation == "paired":
        payload = json.loads(paths[1].read_text())
        payload["domain_mean_guarded_minus_noop"] = [
            value + (-0.01 if index < 10 else 0.01)
            for index, value in enumerate(
                json.loads(paths[0].read_text())["domain_mean_guarded_minus_noop"]
            )
        ]
        paths[1].write_text(json.dumps(payload))
    else:
        _write_h20(paths[4], rmsd=8.0, bond_mean=2.0)
    report = vector_ab.adjudicate(*paths[:4], candidate_h20_path=paths[4])
    assert report["status"] == expected


def test_paper_vector_profile_binds_exact_recipe_and_architecture(tmp_path):
    config = to_dict(load_config(
        "configs/v100_tensorcloud01_vector_only_d1_fp32_paper_horizon500k_2000.yaml"
    ))
    checkpoint = tmp_path / "candidate.pt"
    torch.save({
        "step": 2000,
        "checkpoint_schema": 2,
        "cfg": config,
        "model": {"weight": torch.ones(1)},
        "train_state": {"world_size": 8, "train_fingerprint": "e" * 64},
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_data, expected_model, expected_train = checkpoint_profile_requirements(
        PAPER_VECTOR_PROFILE, digest
    )
    loaded, _ = verify_multidomain_checkpoint(
        checkpoint,
        digest,
        expected_step=2000,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )
    assert loaded["cfg"]["model"]["tensor_cloud01_vector_only_attention"] is True
    wrong_model = dict(expected_model)
    wrong_model["tensor_cloud01_vector_only_attention"] = False
    with pytest.raises(ValueError, match="architecture flag mismatch"):
        verify_multidomain_checkpoint(
            checkpoint,
            digest,
            expected_step=2000,
            expected_model_config=wrong_model,
        )


def test_scalar_value_profile_is_disjoint_from_pure_vector_profile(tmp_path):
    config = to_dict(load_config(
        "configs/v100_tensorcloud01_vector_scalar_value_d1_fp32_"
        "paper_horizon500k_2000.yaml"
    ))
    checkpoint = tmp_path / "scalar_value.pt"
    torch.save({
        "step": 2000,
        "checkpoint_schema": 2,
        "cfg": config,
        "model": {"weight": torch.ones(1)},
        "train_state": {"world_size": 8, "train_fingerprint": "e" * 64},
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_data, expected_model, expected_train = checkpoint_profile_requirements(
        PAPER_SCALAR_VALUE_PROFILE, digest
    )
    loaded, _ = verify_multidomain_checkpoint(
        checkpoint,
        digest,
        expected_step=2000,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )
    assert loaded["cfg"]["model"]["tensor_cloud01_vector_only_attention"] is True
    assert loaded["cfg"]["model"]["tensor_cloud01_vector_only_scalar_value"] is True

    _, vector_model, _ = checkpoint_profile_requirements(
        PAPER_VECTOR_PROFILE, digest
    )
    with pytest.raises(ValueError, match="scalar_value"):
        verify_multidomain_checkpoint(
            checkpoint,
            digest,
            expected_step=2000,
            expected_model_config=vector_model,
        )


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_scalar_value_profile_rejects_non_boolean_flag(tmp_path, invalid):
    config = to_dict(load_config(
        "configs/v100_tensorcloud01_vector_scalar_value_d1_fp32_"
        "paper_horizon500k_2000.yaml"
    ))
    config["model"]["tensor_cloud01_vector_only_scalar_value"] = invalid
    checkpoint = tmp_path / "scalar_value.pt"
    torch.save({
        "step": 2000,
        "checkpoint_schema": 2,
        "cfg": config,
        "model": {"weight": torch.ones(1)},
        "train_state": {"world_size": 8, "train_fingerprint": "e" * 64},
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _, expected_model, _ = checkpoint_profile_requirements(
        PAPER_SCALAR_VALUE_PROFILE, digest
    )
    with pytest.raises(ValueError, match="not boolean"):
        verify_multidomain_checkpoint(
            checkpoint,
            digest,
            expected_step=2000,
            expected_model_config=expected_model,
        )


def test_paper_vector_external_prerequisite_is_exact_and_fail_closed(tmp_path):
    decision = tmp_path / "decision.json"
    payload = {
        "status": "ADVANCE_PAPER_VECTOR_EXTERNAL20",
        "scope": "matched_fresh_continuous_0_to_2000_paper_vector_attention_training_dev_ab",
        "panel_kind": "training",
        "baseline_checkpoint_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "training_domain_list_sha256": "c" * 64,
        "domain_list_sha256": "d" * 64,
        "baseline_reproduced": True,
        "candidate_absolute_pass": True,
        "objective": {"required_candidate_factor": 0.995, "passes": True},
        "paired_pass": True,
        "candidate_h20_gate": {"passes": True},
        "external_development_authorized": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    decision.write_text(json.dumps(payload))
    digest = hashlib.sha256(decision.read_bytes()).hexdigest()
    report = verify_paper_vector_ab_prerequisite(
        decision,
        digest,
        expected_baseline_checkpoint_sha256="a" * 64,
        expected_candidate_checkpoint_sha256="b" * 64,
        expected_training_sha256="c" * 64,
        expected_training_panel_sha256="d" * 64,
    )
    assert report["status"] == "ADVANCE_PAPER_VECTOR_EXTERNAL20"
    payload["status"] = "ADVANCE_PAPER_HORIZON_EXTERNAL20"
    decision.write_text(json.dumps(payload))
    changed = hashlib.sha256(decision.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="status"):
        verify_paper_vector_ab_prerequisite(
            decision,
            changed,
            expected_baseline_checkpoint_sha256="a" * 64,
            expected_candidate_checkpoint_sha256="b" * 64,
            expected_training_sha256="c" * 64,
            expected_training_panel_sha256="d" * 64,
        )


def test_paper_vector_external_requires_both_arms_to_pass_absolute_gate(
    tmp_path, monkeypatch
):
    paths = _case(tmp_path, monkeypatch)
    baseline_payload = json.loads(paths[0].read_text())
    candidate_payload = json.loads(paths[1].read_text())
    baseline_payload["status"] = "STOP_CONDITIONAL_SAFEGUARD_RAW_NONFINITE"
    candidate_payload["status"] = vector_ab.EXTERNAL_ABSOLUTE_PASS
    evidence = {"claim_sha256": "e" * 64, "inventory_sha256": "f" * 64}
    baseline_payload["external_evidence"] = evidence
    candidate_payload["external_evidence"] = evidence
    paths[0].write_text(json.dumps(baseline_payload))
    paths[1].write_text(json.dumps(candidate_payload))
    report = vector_ab.adjudicate(
        *paths[:4], panel_kind="paper-vector-external"
    )
    assert report["status"] == "STOP_PAPER_VECTOR_BASELINE_REPRODUCIBILITY"
    assert report["baseline_reproduced"] is False
    assert report["second_seed_scientifically_eligible"] is False


def test_paper_vector_external_rejects_mismatched_evidence_bindings(
    tmp_path, monkeypatch
):
    paths = _case(tmp_path, monkeypatch)
    baseline_payload = json.loads(paths[0].read_text())
    candidate_payload = json.loads(paths[1].read_text())
    baseline_payload["status"] = vector_ab.EXTERNAL_ABSOLUTE_PASS
    candidate_payload["status"] = vector_ab.EXTERNAL_ABSOLUTE_PASS
    baseline_payload["external_evidence"] = {"claim_sha256": "e" * 64}
    candidate_payload["external_evidence"] = {"claim_sha256": "f" * 64}
    paths[0].write_text(json.dumps(baseline_payload))
    paths[1].write_text(json.dumps(candidate_payload))
    with pytest.raises(ValueError, match="evidence bindings differ"):
        vector_ab.adjudicate(*paths[:4], panel_kind="paper-vector-external")
