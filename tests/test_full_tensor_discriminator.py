import json
from pathlib import Path

import pytest

from scripts.adjudicate_full_tensor_discriminator import adjudicate
from scripts.adjudicate_paper_loss_continuation import EXPECTED_STEPS


DOMAIN_SHA = "a" * 64
METHODS = ("noop", "mean", "ode_1")


def _domain(name: str, *, k1: dict[str, float], k20: dict[str, float]) -> dict:
    methods = {}
    for method in METHODS:
        rmsd = [0.0] + [k1[method]] * 19 + [k20[method]]
        methods[method] = {
            "rmsd": rmsd,
            "bond_mean": [3.8] * 21,
            "bond_max": [4.1 if method == "noop" else 5.0] * 21,
        }
    return {"domain": name, "starts": [0, 100], "methods": methods}


def _result(step: int, *, model_rmsd: float, k1: float, vector: bool = False) -> dict:
    model_h20 = 8.0 if vector else model_rmsd
    model_k1 = 1.4 if vector else k1
    domains = [
        _domain(
            name,
            k1={"noop": 1.35, "mean": model_k1, "ode_1": 2.5},
            k20={"noop": 2.0, "mean": model_h20, "ode_1": 3.0},
        )
        for name in ("a", "b", "c")
    ]
    return {
        "checkpoint_step": step,
        "delta_frames": 1,
        "domain_panel": {"sha256": DOMAIN_SHA, "evaluated_count": 3},
        "settings": {
            "ckpt": "/runs/vector/ckpt_2000.pt" if vector else "/runs/full/ckpt_2000.pt",
            "domain_list": "configs/dev_20_length_proportional_seed0.txt",
            "domain_list_sha256": DOMAIN_SHA,
            "domains": 3,
            "starts": 2,
            "steps": 20,
            "methods": "mean,ode_1",
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
            "output": "/runs/vector/rollout_2000.json" if vector else "/runs/full/rollout_2000.json",
        },
        "summary": {
            "noop": {"mean_final_rmsd": 2.0, "finite": True},
            "mean": {
                "mean_final_rmsd": model_h20,
                "mean_final_bond_mean": 3.8,
                "mean_final_bond_max": 5.0,
                "finite": True,
            },
            "ode_1": {
                "mean_final_rmsd": 3.0,
                "mean_final_bond_mean": 6.0,
                "mean_final_bond_max": 8.0,
                "finite": True,
            },
        },
        "domains": domains,
    }


def _write_case(root: Path, *, full_k1: float) -> tuple[Path, Path, Path]:
    history = root / "history.json"
    history.write_text(
        json.dumps(
            [
                {"step": step, "val_loss": 4.0, "val_rmsd": 1.0, "noop_rmsd": 2.0}
                for step in EXPECTED_STEPS
            ]
        )
    )
    rollouts = root / "rollouts"
    rollouts.mkdir()
    for index, step in enumerate(EXPECTED_STEPS):
        model_rmsd = 2.5
        if index >= len(EXPECTED_STEPS) - 3:
            model_rmsd = (1.9, 1.8, 1.7)[index - (len(EXPECTED_STEPS) - 3)]
        (rollouts / f"rollout_{step}.json").write_text(
            json.dumps(_result(step, model_rmsd=model_rmsd, k1=full_k1))
        )
    vector = root / "vector_2000.json"
    vector.write_text(json.dumps(_result(2000, model_rmsd=8.0, k1=1.4, vector=True)))
    return history, rollouts, vector


def test_full_tensor_gate_requires_absolute_and_relative_improvement(tmp_path):
    history, rollouts, vector = _write_case(tmp_path, full_k1=1.35)
    report = adjudicate(history, rollouts, vector, DOMAIN_SHA)
    assert report["status"] == "GO_STRICT_INTEGRATION"
    assert report["formal_training_authorized"] is False
    comparison = report["relative_vector_only_comparison"]
    assert comparison["passes"] is True
    assert comparison["k1_rmsd_improvement_angstrom"] == pytest.approx(0.05)


def test_full_tensor_gate_stops_if_k1_improvement_is_too_small(tmp_path):
    history, rollouts, vector = _write_case(tmp_path, full_k1=1.395)
    report = adjudicate(history, rollouts, vector, DOMAIN_SHA)
    assert report["absolute_gate"]["status"] == "GO_BOUNDED_EXTENSION"
    assert report["status"] == "STOP_ARCHITECTURE_PAIR"
    assert report["relative_vector_only_comparison"]["passes"] is False


def test_full_tensor_gate_fails_closed_on_different_starts(tmp_path):
    history, rollouts, vector = _write_case(tmp_path, full_k1=1.35)
    baseline = json.loads(vector.read_text())
    baseline["domains"][0]["starts"] = [1, 101]
    vector.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="starts mismatch"):
        adjudicate(history, rollouts, vector, DOMAIN_SHA)
