import copy

import pytest

from scripts.validate_full_grid_evaluation import (
    REPLICAS,
    TEMPERATURES,
    validate_geometry,
    validate_transition,
)


CHECKPOINT = "/data/checkpoint.pt"


def _cells(*, transition: bool):
    cells = []
    for temperature in TEMPERATURES:
        for replica in REPLICAS:
            cell = {"temperature": temperature, "replica": replica, "metric": 0.1}
            if transition:
                cell["reference_replica"] = REPLICAS[(replica + 1) % len(REPLICAS)]
            cells.append(cell)
    return cells


def _result(*, transition: bool, steps: int | None = None):
    trajectories = _cells(transition=transition)
    return {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": 30,
        "domain_panel": {"evaluated_count": 1},
        "trajectory_grid": {
            "formal_full_grid": True,
            "temperatures": TEMPERATURES,
            "replicas": REPLICAS,
            "required_cells_per_domain": 25,
        },
        "settings": {} if steps is None else {"steps": steps},
        "summary": {"noop": {"metric": 0.2}, "mean": {"metric": 0.1}},
        "domains": [
            {
                "domain": "domain0",
                "grid": {"cells": 25},
                "trajectories": trajectories,
            }
        ],
    }


def test_strict_full_grid_evaluation_artifacts_pass():
    validate_transition(
        _result(transition=True),
        expected_checkpoint=CHECKPOINT,
        expected_step=30,
        expected_domains=1,
    )
    for steps in (20, 100):
        validate_geometry(
            _result(transition=False, steps=steps),
            expected_checkpoint=CHECKPOINT,
            expected_step=30,
            expected_domains=1,
            expected_rollout_steps=steps,
        )


def test_incomplete_grid_fails_closed():
    result = _result(transition=True)
    result["domains"][0]["trajectories"].pop()
    with pytest.raises(ValueError, match="25 trajectories"):
        validate_transition(
            result,
            expected_checkpoint=CHECKPOINT,
            expected_step=30,
            expected_domains=1,
        )


def test_transition_crossfit_reuse_fails_closed():
    result = _result(transition=True)
    result["domains"][0]["trajectories"][0]["reference_replica"] = 0
    with pytest.raises(ValueError, match="reuses"):
        validate_transition(
            result,
            expected_checkpoint=CHECKPOINT,
            expected_step=30,
            expected_domains=1,
        )


def test_nonfinite_metric_fails_closed():
    result = _result(transition=False, steps=20)
    result["summary"]["mean"]["metric"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_geometry(
            result,
            expected_checkpoint=CHECKPOINT,
            expected_step=30,
            expected_domains=1,
            expected_rollout_steps=20,
        )


def test_wrong_geometry_horizon_fails_closed():
    result = copy.deepcopy(_result(transition=False, steps=19))
    with pytest.raises(ValueError, match="rollout length"):
        validate_geometry(
            result,
            expected_checkpoint=CHECKPOINT,
            expected_step=30,
            expected_domains=1,
            expected_rollout_steps=20,
        )
