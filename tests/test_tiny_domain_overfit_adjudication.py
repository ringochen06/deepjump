import json
from pathlib import Path

import pytest

from scripts.adjudicate_tiny_domain_overfit import EXPECTED_STEPS, adjudicate, select_checkpoint


DOMAIN_SHA = "c" * 64


def _write_history(root: Path, losses: list[float], *, val_rmsd: float = 1.0) -> Path:
    path = root / "history.json"
    path.write_text(
        json.dumps(
            [
                {
                    "step": step,
                    "val_loss": loss,
                    "val_rmsd": val_rmsd,
                    "noop_rmsd": 2.0,
                }
                for step, loss in zip(EXPECTED_STEPS, losses)
            ]
        )
    )
    return path


def _write_rollout(
    root: Path,
    *,
    step: int,
    mean_rmsd: float = 1.5,
    mean_bond_mean: float = 3.8,
    mean_bond_max: float = 5.0,
) -> Path:
    path = root / "rollout.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint_step": step,
                "delta_frames": 1,
                "domain_panel": {"sha256": DOMAIN_SHA, "evaluated_count": 1},
                "settings": {
                    "domains": 1,
                    "starts": 5,
                    "steps": 20,
                    "methods": "mean,ode_1",
                    "integrator": "euler",
                    "tau_max": 1.0,
                    "terminal_denoise": False,
                    "drift_anchor": "state",
                },
                "summary": {
                    "noop": {"mean_final_rmsd": 2.0, "finite": True},
                    "mean": {
                        "mean_final_rmsd": mean_rmsd,
                        "mean_final_bond_mean": mean_bond_mean,
                        "mean_final_bond_max": mean_bond_max,
                        "finite": True,
                    },
                    "ode_1": {
                        "mean_final_rmsd": 3.0,
                        "mean_final_bond_mean": 6.0,
                        "mean_final_bond_max": 8.0,
                        "finite": True,
                    },
                },
            }
        )
    )
    return path


def _converged_losses() -> list[float]:
    return [4.0, 2.5, 1.8, 1.2, 0.9, 0.7, 0.55, 0.49, 0.485, 0.486]


def test_selects_by_validation_loss_only_within_fixed_final_window(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    selection = select_checkpoint(history)
    assert selection["selected_step"] == 4500
    assert selection["converged"] is True


def test_adjudication_calls_in_domain_operator_learnable(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    rollout = _write_rollout(tmp_path, step=4500)
    report = adjudicate(history, rollout, DOMAIN_SHA)
    assert report["status"] == "IN_DOMAIN_OPERATOR_LEARNABLE"
    assert report["methods"]["mean"]["passes"] is True
    assert report["formal_training_authorized"] is False


def test_adjudication_calls_recurrence_failure_after_convergence(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    rollout = _write_rollout(tmp_path, step=4500, mean_bond_max=8.0)
    report = adjudicate(history, rollout, DOMAIN_SHA)
    assert report["status"] == "IN_DOMAIN_RECURRENCE_FAILURE"


def test_adjudication_is_inconclusive_without_convergence(tmp_path):
    history = _write_history(tmp_path, [4.0 - 0.25 * i for i in range(10)])
    rollout = _write_rollout(tmp_path, step=5000)
    report = adjudicate(history, rollout, DOMAIN_SHA)
    assert report["status"] == "INCONCLUSIVE_NOT_CONVERGED"
    assert report["methods"]["mean"]["passes"] is True


def test_adjudication_fails_closed_on_panel_or_checkpoint_mismatch(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    rollout = _write_rollout(tmp_path, step=5000)
    with pytest.raises(ValueError, match="checkpoint"):
        adjudicate(history, rollout, DOMAIN_SHA)
    rollout = _write_rollout(tmp_path, step=4500)
    with pytest.raises(ValueError, match="panel"):
        adjudicate(history, rollout, "d" * 64)


def test_selection_fails_closed_on_history_step_set_mismatch(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    rows = json.loads(history.read_text())
    history.write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="history steps mismatch"):
        select_checkpoint(history)


def test_selection_fails_closed_when_validation_noop_changes(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    rows = json.loads(history.read_text())
    rows[-1]["noop_rmsd"] = 2.001
    history.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="no-op RMSD changed"):
        select_checkpoint(history)


@pytest.mark.parametrize("invalid_loss", [-1.0, float("nan")])
def test_selection_rejects_invalid_validation_loss(tmp_path, invalid_loss):
    losses = _converged_losses()
    losses[0] = invalid_loss
    history = _write_history(tmp_path, losses)
    with pytest.raises(ValueError, match="validation losses|must be finite"):
        select_checkpoint(history)


def test_adjudication_fails_closed_on_rollout_settings_mismatch(tmp_path):
    history = _write_history(tmp_path, _converged_losses())
    rollout = _write_rollout(tmp_path, step=4500)
    result = json.loads(rollout.read_text())
    result["settings"]["starts"] = 4
    rollout.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="settings mismatch"):
        adjudicate(history, rollout, DOMAIN_SHA)
