import copy

import pytest

from scripts.select_calibration_checkpoints import select_checkpoints


def _history():
    return [
        {"step": 250, "val_loss": 1.0, "val_rmsd": 2.0, "noop_rmsd": 1.9},
        {"step": 500, "val_loss": 0.8, "val_rmsd": 2.1, "noop_rmsd": 1.9},
        {"step": 750, "val_loss": 0.7, "val_rmsd": 2.2, "noop_rmsd": 1.9},
        {"step": 1000, "val_loss": 0.7, "val_rmsd": 2.3, "noop_rmsd": 1.9},
    ]


def _config():
    return {
        "data": {"delta_frames": 1},
        "model": {"tensor_cloud01": True},
        "train": {"max_steps": 1000},
    }


def test_selects_two_lowest_losses_with_earlier_tie_break():
    result = select_checkpoints(_history(), _config(), expected_delta=1)
    assert result["ranked_steps"] == [750, 1000, 500, 250]
    assert [row["step"] for row in result["selected"]] == [750, 1000]
    assert result["scientific_metrics_used_for_selection"] is False


def test_rejects_missing_duplicate_or_nonfinite_history():
    with pytest.raises(ValueError, match="exactly steps"):
        select_checkpoints(_history()[:-1], _config(), expected_delta=1)
    duplicate = _history()
    duplicate[-1]["step"] = 750
    with pytest.raises(ValueError, match="duplicate"):
        select_checkpoints(duplicate, _config(), expected_delta=1)
    nonfinite = _history()
    nonfinite[0]["val_loss"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        select_checkpoints(nonfinite, _config(), expected_delta=1)


def test_rejects_wrong_calibration_identity():
    wrong = copy.deepcopy(_config())
    wrong["data"]["delta_frames"] = 10
    with pytest.raises(ValueError, match="delta_frames"):
        select_checkpoints(_history(), wrong, expected_delta=1)
    wrong = copy.deepcopy(_config())
    wrong["model"]["tensor_cloud01"] = False
    with pytest.raises(ValueError, match="TensorCloud01"):
        select_checkpoints(_history(), wrong, expected_delta=1)
