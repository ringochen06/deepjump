import json
from pathlib import Path

import pytest

from scripts.adjudicate_paper_loss_continuation import EXPECTED_STEPS, adjudicate


DOMAIN_SHA = "a" * 64


def _write_case(
    root: Path,
    *,
    val_losses: list[float],
    rmsd_by_method: dict[str, list[float]],
    bond_mean_by_method: dict[str, list[float]],
    bond_max_by_method: dict[str, list[float]],
) -> tuple[Path, Path]:
    history = root / "history.json"
    history.write_text(
        json.dumps(
            [
                {"step": step, "val_loss": loss, "val_rmsd": 1.0, "noop_rmsd": 2.0}
                for step, loss in zip(EXPECTED_STEPS, val_losses)
            ]
        )
    )
    rollouts = root / "rollouts"
    rollouts.mkdir()
    for index, step in enumerate(EXPECTED_STEPS):
        summary = {"noop": {"mean_final_rmsd": 2.0, "finite": True}}
        for method in ("mean", "ode_1"):
            summary[method] = {
                "mean_final_rmsd": rmsd_by_method[method][index],
                "mean_final_bond_mean": bond_mean_by_method[method][index],
                "mean_final_bond_max": bond_max_by_method[method][index],
                "finite": True,
            }
        result = {
            "checkpoint_step": step,
            "delta_frames": 1,
            "domain_panel": {"sha256": DOMAIN_SHA, "evaluated_count": 3},
            "settings": {
                "domains": 3,
                "starts": 2,
                "steps": 20,
                "methods": "mean,ode_1",
                "integrator": "euler",
                "tau_max": 1.0,
                "terminal_denoise": False,
                "drift_anchor": "state",
            },
            "summary": summary,
        }
        (rollouts / f"rollout_{step}.json").write_text(json.dumps(result))
    return history, rollouts


def _constant(value: float) -> list[float]:
    return [value] * len(EXPECTED_STEPS)


def test_continuation_gate_releases_only_bounded_extension(tmp_path):
    mean_rmsd = _constant(2.5)
    mean_rmsd[-3:] = [1.9, 1.8, 1.7]
    history, rollouts = _write_case(
        tmp_path,
        val_losses=_constant(4.0),
        rmsd_by_method={"mean": mean_rmsd, "ode_1": _constant(3.0)},
        bond_mean_by_method={"mean": _constant(3.8), "ode_1": _constant(6.0)},
        bond_max_by_method={"mean": _constant(5.0), "ode_1": _constant(8.0)},
    )
    report = adjudicate(history, rollouts, DOMAIN_SHA)
    assert report["status"] == "GO_BOUNDED_EXTENSION"
    assert report["selected_method"] == "mean"
    assert report["formal_training_authorized"] is False


def test_continuation_gate_calls_objective_mismatch_when_loss_falls(tmp_path):
    history, rollouts = _write_case(
        tmp_path,
        val_losses=[4.0 - 0.03 * i for i in range(len(EXPECTED_STEPS))],
        rmsd_by_method={"mean": _constant(3.0), "ode_1": _constant(4.0)},
        bond_mean_by_method={"mean": _constant(6.0), "ode_1": _constant(7.0)},
        bond_max_by_method={"mean": _constant(8.0), "ode_1": _constant(9.0)},
    )
    report = adjudicate(history, rollouts, DOMAIN_SHA)
    assert report["status"] == "STOP_OBJECTIVE_MISMATCH"


def test_continuation_gate_calls_optimization_inconclusive_without_loss_trend(tmp_path):
    history, rollouts = _write_case(
        tmp_path,
        val_losses=_constant(4.0),
        rmsd_by_method={"mean": _constant(3.0), "ode_1": _constant(4.0)},
        bond_mean_by_method={"mean": _constant(6.0), "ode_1": _constant(7.0)},
        bond_max_by_method={"mean": _constant(8.0), "ode_1": _constant(9.0)},
    )
    report = adjudicate(history, rollouts, DOMAIN_SHA)
    assert report["status"] == "STOP_OPTIMIZATION_INCONCLUSIVE"


def test_continuation_gate_fails_closed_on_wrong_panel(tmp_path):
    history, rollouts = _write_case(
        tmp_path,
        val_losses=_constant(4.0),
        rmsd_by_method={"mean": _constant(1.0), "ode_1": _constant(1.0)},
        bond_mean_by_method={"mean": _constant(3.8), "ode_1": _constant(3.8)},
        bond_max_by_method={"mean": _constant(5.0), "ode_1": _constant(5.0)},
    )
    with pytest.raises(ValueError, match="panel mismatch"):
        adjudicate(history, rollouts, "b" * 64)
