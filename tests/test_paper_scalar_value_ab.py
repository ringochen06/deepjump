import hashlib
import json
from pathlib import Path

import pytest

import scripts.adjudicate_paper_scalar_value_ab as scalar_value_ab
import scripts.adjudicate_paper_vector_ab as vector_ab
from scripts.guarded_endpoint_panel_eval import (
    PAPER_SCALAR_VALUE_PROFILE,
    PAPER_VECTOR_PROFILE,
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


def _write_decision(
    path: Path,
    *,
    profile: str,
    checkpoint_sha256: str,
    values: list[float],
    status: str,
) -> None:
    path.write_text(json.dumps({
        "status": status,
        "checkpoint_profile": profile,
        "checkpoint_step": 2000,
        "checkpoint_sha256": checkpoint_sha256,
        "training_domain_list_sha256": "c" * 64,
        "domain_list_sha256": "d" * 64,
        "domains": 20,
        "starts": 1500,
        "domain_mean_guarded_minus_noop": values,
    }))


def _write_h20(
    path: Path,
    *,
    checkpoint_sha256: str = "b" * 64,
    rmsd: float = 1.8,
    bond_mean: float = 3.8,
) -> None:
    path.write_text(json.dumps({
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": 2000,
        "delta_frames": 1,
        "domain_panel": {
            "sha256": (
                "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
            ),
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


def _refresh_evidence(paths: tuple[Path, ...]) -> None:
    (
        baseline_decision, baseline_replay_decision, candidate_decision,
        baseline_history, candidate_history, h20, evidence,
    ) = paths
    baseline = json.loads(baseline_decision.read_text())
    candidate = json.loads(candidate_decision.read_text())
    evidence.write_text(json.dumps({
        "schema": "deepjump.scalar_value_training_evidence.v1",
        "run_id": "20260722T120000Z",
        "commit": "f" * 40,
        "candidate_config_sha256": "a" * 64,
        "baseline_decision_sha256": hashlib.sha256(
            baseline_decision.read_bytes()
        ).hexdigest(),
        "baseline_replay_decision_sha256": hashlib.sha256(
            baseline_replay_decision.read_bytes()
        ).hexdigest(),
        "candidate_decision_sha256": hashlib.sha256(
            candidate_decision.read_bytes()
        ).hexdigest(),
        "baseline_history_sha256": hashlib.sha256(
            baseline_history.read_bytes()
        ).hexdigest(),
        "candidate_history_sha256": hashlib.sha256(
            candidate_history.read_bytes()
        ).hexdigest(),
        "candidate_h20_sha256": hashlib.sha256(h20.read_bytes()).hexdigest(),
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "training_domain_list_sha256": candidate["training_domain_list_sha256"],
        "domain_list_sha256": candidate["domain_list_sha256"],
    }))


def _case(root: Path, monkeypatch) -> tuple[
    Path, Path, Path, Path, Path, Path, Path
]:
    baseline_decision = root / "baseline_decision.json"
    baseline_replay_decision = root / "baseline_replay_decision.json"
    candidate_decision = root / "candidate_decision.json"
    baseline_history = root / "baseline_history.json"
    candidate_history = root / "candidate_history.json"
    h20 = root / "candidate_h20.json"
    evidence = root / "training_evidence.json"
    baseline = [0.10 + 0.01 * (index % 4) for index in range(20)]
    candidate = [
        value - 0.20 + 0.01 * (index % 5)
        for index, value in enumerate(baseline)
    ]
    _write_decision(
        baseline_decision,
        profile=PAPER_VECTOR_PROFILE,
        checkpoint_sha256=scalar_value_ab.EXPECTED_BASELINE_CHECKPOINT_SHA256,
        values=baseline,
        status=scalar_value_ab.EXPECTED_BASELINE_ABSOLUTE_STATUS,
    )
    baseline_replay_decision.write_bytes(baseline_decision.read_bytes())
    _write_decision(
        candidate_decision,
        profile=PAPER_SCALAR_VALUE_PROFILE,
        checkpoint_sha256="b" * 64,
        values=candidate,
        status=vector_ab.ABSOLUTE_PASS,
    )
    _write_history(baseline_history, 4.2)
    _write_history(candidate_history, 4.0)
    _write_h20(h20)
    monkeypatch.setattr(
        scalar_value_ab,
        "EXPECTED_BASELINE_HISTORY_SHA256",
        hashlib.sha256(baseline_history.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        scalar_value_ab,
        "EXPECTED_BASELINE_DECISION_SHA256",
        hashlib.sha256(baseline_decision.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(scalar_value_ab, "EXPECTED_BASELINE_FINAL_VAL_LOSS", 4.2)
    paths = (
        baseline_decision,
        baseline_replay_decision,
        candidate_decision,
        baseline_history,
        candidate_history,
        h20,
        evidence,
    )
    _refresh_evidence(paths)
    return paths


def test_scalar_value_ab_requires_every_training_gate(tmp_path, monkeypatch):
    report = scalar_value_ab.adjudicate(*_case(tmp_path, monkeypatch))
    assert report["status"] == "ADVANCE_SCALAR_VALUE_EXTERNAL20"
    assert report["paper_equivalence"] == (
        "architecture_hypothesis_not_paper_verified"
    )
    assert report["baseline_reproduced"] is True
    assert report["candidate_absolute_pass"] is True
    assert report["objective"]["passes"] is True
    assert report["paired_pass"] is True
    assert report["candidate_h20_gate"]["passes"] is True
    assert report["external_development_scientifically_eligible"] is True
    assert report["external_development_authorized"] is False
    assert report["second_seed_scientifically_eligible"] is False
    assert report["second_seed_authorized"] is False
    assert report["untouched_confirmation_authorized"] is False
    assert report["formal_training_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("baseline", "STOP_SCALAR_VALUE_BASELINE_REPRODUCIBILITY"),
        ("baseline_replay", "STOP_SCALAR_VALUE_BASELINE_REPRODUCIBILITY"),
        ("absolute", "STOP_SCALAR_VALUE_ABSOLUTE_GATE"),
        ("objective", "STOP_SCALAR_VALUE_OBJECTIVE_GAIN"),
        ("paired", "STOP_SCALAR_VALUE_PAIRED_ADVANTAGE"),
        ("h20", "STOP_SCALAR_VALUE_H20_GATE"),
    ],
)
def test_scalar_value_ab_stops_on_each_failed_conjunct(
    tmp_path, monkeypatch, mutation, expected
):
    paths = _case(tmp_path, monkeypatch)
    if mutation == "baseline":
        payload = json.loads(paths[0].read_text())
        payload["status"] = vector_ab.ABSOLUTE_PASS
        paths[0].write_text(json.dumps(payload))
        monkeypatch.setattr(
            scalar_value_ab,
            "EXPECTED_BASELINE_DECISION_SHA256",
            hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        )
    elif mutation == "baseline_replay":
        payload = json.loads(paths[1].read_text())
        payload["domain_mean_guarded_minus_noop"][0] += 1e-12
        paths[1].write_text(json.dumps(payload))
    elif mutation == "absolute":
        payload = json.loads(paths[2].read_text())
        payload["status"] = "STOP_CONDITIONAL_SAFEGUARD_NO_ADVANTAGE"
        paths[2].write_text(json.dumps(payload))
    elif mutation == "objective":
        _write_history(paths[4], 4.19)
    elif mutation == "paired":
        baseline = json.loads(paths[0].read_text())["domain_mean_guarded_minus_noop"]
        payload = json.loads(paths[2].read_text())
        payload["domain_mean_guarded_minus_noop"] = [
            value + (-0.01 if index < 10 else 0.01)
            for index, value in enumerate(baseline)
        ]
        paths[2].write_text(json.dumps(payload))
    else:
        _write_h20(paths[5], rmsd=8.0, bond_mean=2.0)
    _refresh_evidence(paths)
    report = scalar_value_ab.adjudicate(*paths)
    assert report["status"] == expected
    assert report["external_development_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("baseline_checkpoint", "sealed vector-only artifact"),
        ("candidate_profile", "checkpoint profile mismatch"),
        ("same_checkpoint", "checkpoints must be distinct"),
        ("domain_identity", "domain_list_sha256 mismatch"),
        ("history_sha", "history SHA256 mismatch"),
        ("h20_checkpoint", "H20 checkpoint SHA256 mismatch"),
        ("candidate_history_evidence", "candidate_history_sha256 mismatch"),
        ("baseline_decision_sha", "baseline absolute decision SHA256 mismatch"),
    ],
)
def test_scalar_value_ab_rejects_identity_mutations(
    tmp_path, monkeypatch, mutation, message
):
    paths = _case(tmp_path, monkeypatch)
    if mutation == "baseline_checkpoint":
        payload = json.loads(paths[0].read_text())
        payload["checkpoint_sha256"] = "e" * 64
        paths[0].write_text(json.dumps(payload))
    elif mutation == "candidate_profile":
        payload = json.loads(paths[2].read_text())
        payload["checkpoint_profile"] = PAPER_VECTOR_PROFILE
        paths[2].write_text(json.dumps(payload))
    elif mutation == "same_checkpoint":
        payload = json.loads(paths[2].read_text())
        payload["checkpoint_sha256"] = (
            scalar_value_ab.EXPECTED_BASELINE_CHECKPOINT_SHA256
        )
        paths[2].write_text(json.dumps(payload))
    elif mutation == "domain_identity":
        payload = json.loads(paths[2].read_text())
        payload["domain_list_sha256"] = "e" * 64
        paths[2].write_text(json.dumps(payload))
    elif mutation == "history_sha":
        paths[3].write_text(paths[3].read_text() + "\n")
    elif mutation == "candidate_history_evidence":
        _write_history(paths[4], 3.0)
    elif mutation == "baseline_decision_sha":
        payload = json.loads(paths[0].read_text())
        payload["domain_mean_guarded_minus_noop"][0] -= 1.0
        paths[0].write_text(json.dumps(payload))
    else:
        _write_h20(paths[5], checkpoint_sha256="e" * 64)
    with pytest.raises(ValueError, match=message):
        scalar_value_ab.adjudicate(*paths)
