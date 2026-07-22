import json
import hashlib
from pathlib import Path

import pytest
import torch

from deepjump.config import load_config, to_dict
from scripts.external_endpoint_identity import (
    load_paper_horizon_external_panels,
    verify_multidomain_checkpoint,
    verify_paper_horizon_ab_prerequisite,
)
from scripts.adjudicate_paper_horizon_ab import EXPECTED_STEPS, adjudicate
from scripts.guarded_endpoint_panel_eval import (
    HORIZON_AB_BASELINE_PROFILE,
    PAPER_HORIZON_PROFILE,
    checkpoint_profile_requirements,
)


def _write_history(path: Path, val_loss: float, *, noop_shift: float = 0.0) -> None:
    path.write_text(json.dumps([
        {
            "step": step,
            "val_loss": val_loss + (2000 - step) * 1e-5,
            "val_rmsd": 2.0,
            "noop_rmsd": 3.0 + noop_shift,
        }
        for step in EXPECTED_STEPS
    ]))


def _write_decision(
    path: Path,
    profile: str,
    values: list[float],
    *,
    status: str = "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20",
) -> None:
    path.write_text(json.dumps({
        "status": status,
        "checkpoint_profile": profile,
        "checkpoint_step": 2000,
        "checkpoint_sha256": ("a" if profile == HORIZON_AB_BASELINE_PROFILE else "b") * 64,
        "training_domain_list_sha256": "c" * 64,
        "domain_list_sha256": "d" * 64,
        "domains": 20,
        "starts": 1500,
        "domain_mean_guarded_minus_noop": values,
    }))


def _case(root: Path, *, candidate_status: str = "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"):
    baseline_decision = root / "baseline_decision.json"
    candidate_decision = root / "candidate_decision.json"
    baseline_history = root / "baseline_history.json"
    candidate_history = root / "candidate_history.json"
    baseline = [0.10 + 0.01 * (index % 4) for index in range(20)]
    paired = [-0.20 + 0.01 * (index % 5) for index in range(20)]
    candidate = [left + right for left, right in zip(baseline, paired)]
    _write_decision(baseline_decision, HORIZON_AB_BASELINE_PROFILE, baseline)
    _write_decision(
        candidate_decision, PAPER_HORIZON_PROFILE, candidate, status=candidate_status
    )
    _write_history(baseline_history, 4.0)
    _write_history(candidate_history, 3.9)
    return baseline_decision, candidate_decision, baseline_history, candidate_history


def test_paper_horizon_ab_advances_only_on_absolute_objective_and_paired_pass(tmp_path):
    report = adjudicate(*_case(tmp_path))
    assert report["status"] == "ADVANCE_PAPER_HORIZON_EXTERNAL20"
    assert report["candidate_absolute_pass"] is True
    assert report["objective"]["passes"] is True
    assert report["paired_pass"] is True
    assert report["paired_domains_better"] == 20
    assert report["external_development_authorized"] is True
    assert report["formal_training_authorized"] is False


def test_paper_horizon_external_ab_authorizes_only_second_seed(tmp_path):
    paths = _case(tmp_path)
    payload = json.loads(paths[1].read_text())
    payload["status"] = "PASS_CONDITIONAL_SAFEGUARD_PAPER_HORIZON_EXTERNAL20"
    paths[1].write_text(json.dumps(payload))
    report = adjudicate(*paths, panel_kind="paper-horizon-external")
    assert report["status"] == "PASS_PAPER_HORIZON_EXTERNAL20"
    assert report["external_development_authorized"] is False
    assert report["second_seed_scientifically_eligible"] is True
    assert report["second_seed_authorized"] is False
    assert report["untouched_confirmation_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_paper_horizon_ab_stops_on_absolute_gate(tmp_path):
    report = adjudicate(*_case(tmp_path, candidate_status="STOP_CONDITIONAL_SAFEGUARD_NO_ADVANTAGE"))
    assert report["status"] == "STOP_PAPER_HORIZON_ABSOLUTE_GATE"


def test_paper_horizon_ab_stops_without_material_objective_gain(tmp_path):
    paths = _case(tmp_path)
    _write_history(paths[3], 3.99)
    report = adjudicate(*paths)
    assert report["status"] == "STOP_PAPER_HORIZON_OBJECTIVE_GAIN"


def test_paper_horizon_ab_stops_without_paired_advantage(tmp_path):
    paths = _case(tmp_path)
    baseline = [0.1 + 0.01 * (index % 4) for index in range(20)]
    candidate = [value + (-0.01 if index < 10 else 0.01) for index, value in enumerate(baseline)]
    _write_decision(paths[1], PAPER_HORIZON_PROFILE, candidate)
    report = adjudicate(*paths)
    assert report["status"] == "STOP_PAPER_HORIZON_PAIRED_ADVANTAGE"


def test_paper_horizon_ab_fails_closed_on_validation_panel_mismatch(tmp_path):
    paths = _case(tmp_path)
    _write_history(paths[3], 3.9, noop_shift=0.01)
    with pytest.raises(ValueError, match="no-op histories differ"):
        adjudicate(*paths)


def test_paper_horizon_checkpoint_profiles_bind_exact_training_recipe():
    baseline_data, baseline_model, baseline_train = checkpoint_profile_requirements(
        HORIZON_AB_BASELINE_PROFILE, "a" * 64
    )
    candidate_data, candidate_model, candidate_train = checkpoint_profile_requirements(
        PAPER_HORIZON_PROFILE, "b" * 64
    )
    assert baseline_model == candidate_model
    assert baseline_data == candidate_data
    assert baseline_data["root"] == "/data/mdcath"
    assert baseline_data["manifest"] == "/data/mdcath/manifest.json"
    left, right = dict(baseline_train), dict(candidate_train)
    assert left.pop("lr_horizon_steps") == 1000
    assert right.pop("lr_horizon_steps") == 500000
    left.pop("out_dir"); right.pop("out_dir")
    assert left == right
    assert left["resume"] == ""
    assert left["geom_huber_delta"] == 0.05
    with pytest.raises(ValueError, match="lowercase hex"):
        checkpoint_profile_requirements(PAPER_HORIZON_PROFILE, "not-a-digest")


def test_paper_horizon_profile_verifies_checkpoint_recipe(tmp_path):
    config = to_dict(load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000.yaml"
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
        PAPER_HORIZON_PROFILE, digest
    )
    loaded, fingerprint = verify_multidomain_checkpoint(
        checkpoint,
        digest,
        expected_step=2000,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )
    assert loaded["step"] == 2000
    assert fingerprint == "e" * 64
    wrong_train = dict(expected_train)
    wrong_train["lr_horizon_steps"] = 1000
    with pytest.raises(ValueError, match="lr_horizon_steps mismatch"):
        verify_multidomain_checkpoint(
            checkpoint,
            digest,
            expected_step=2000,
            expected_train_config=wrong_train,
        )


def test_paper_horizon_external_panel_is_frozen_before_ab_results():
    panel = Path(
        "configs/paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    )
    metadata = json.loads(Path(
        "configs/paper_horizon_external_dev_20_length_proportional_seed20260723.metadata.json"
    ).read_text())
    ids = panel.read_text().splitlines()
    excluded_paths = [
        "configs/subset_1000_length_proportional.txt",
        "configs/external_dev_20_length_proportional_seed20260721.txt",
        "configs/guarded_external_dev_20_length_proportional_seed20260722.txt",
        "configs/confirmation_100_length_proportional_seed20260717.txt",
    ]
    excluded = set()
    for path in excluded_paths:
        excluded.update(Path(path).read_text().splitlines())
    assert len(ids) == len(set(ids)) == 20
    assert len(excluded) == 1140
    assert not (set(ids) & excluded)
    assert hashlib.sha256(panel.read_bytes()).hexdigest() == (
        "9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb"
    )
    assert metadata["seed"] == 20260723
    assert metadata["exclude_count"] == 1140
    assert metadata["expected_h5_bytes"] == 14_236_836_972
    assert metadata["source_h5_sha256"] == (
        "7f35d1c93a3c215859fbb401854d9e113ac670bccd45e1438d4be4abbbe3aa77"
    )


def test_paper_horizon_external_identity_and_prerequisite_are_fail_closed(tmp_path):
    contract = load_paper_horizon_external_panels(
        "configs/subset_1000_length_proportional.txt", "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734",
        "configs/external_dev_20_length_proportional_seed20260721.txt", "9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245",
        "configs/guarded_external_dev_20_length_proportional_seed20260722.txt", "9bae11fa0e6336e7451c372efa25ca55af77aa9cb27f91e1fd241612531a920f",
        "configs/confirmation_100_length_proportional_seed20260717.txt", "e56ed7de735db542f4e20fb73f2654a6c1bcf67f3082849f63f0ab74f4208c38",
        "configs/paper_horizon_external_dev_20_length_proportional_seed20260723.txt", "9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb",
    )
    assert contract["exclusion_union_count"] == 1140

    candidate_sha = "b" * 64
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({
        "status": "ADVANCE_PAPER_HORIZON_EXTERNAL20",
        "candidate_checkpoint_sha256": candidate_sha,
        "training_domain_list_sha256": "c" * 64,
        "domain_list_sha256": "d" * 64,
        "external_development_authorized": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }))
    digest = hashlib.sha256(decision.read_bytes()).hexdigest()
    report = verify_paper_horizon_ab_prerequisite(
        decision,
        digest,
        expected_candidate_checkpoint_sha256=candidate_sha,
        expected_training_sha256="c" * 64,
        expected_training_panel_sha256="d" * 64,
    )
    assert report["status"] == "ADVANCE_PAPER_HORIZON_EXTERNAL20"
    with pytest.raises(ValueError, match="candidate_checkpoint_sha256"):
        verify_paper_horizon_ab_prerequisite(
            decision,
            digest,
            expected_candidate_checkpoint_sha256="a" * 64,
            expected_training_sha256="c" * 64,
            expected_training_panel_sha256="d" * 64,
        )
