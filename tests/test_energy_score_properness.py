"""The transition gate's energy score must rank a calibrated ensemble first.

`transition_robustness_eval.py` scores `mode="mean"` alongside the ODE samplers.
That is only informative if the score is a proper rule: `mean` has exactly zero
ensemble spread, so a score that ignored spread would rank it as well as, or
better than, a correctly calibrated forecast and the whole comparison would be
vacuous.

Properness holds in expectation over observations drawn from the true law, which
is what the script approximates by averaging over many start frames -- not for a
single observation, where a narrow forecast can win by luck. These tests check it
the way the script uses it.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
N_OBSERVATIONS = 3000
N_DRAWS = 16


@pytest.fixture(scope="module")
def energy_score():
    spec = importlib.util.spec_from_file_location(
        "transition_robustness_under_test",
        REPO_ROOT / "scripts" / "transition_robustness_eval.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.energy_score


def _expected_score(energy_score, make_forecast, observations):
    return float(np.mean([energy_score(make_forecast(), y) for y in observations]))


@pytest.fixture(scope="module")
def scores(energy_score):
    """Expected score of each forecast against the same standard-normal truth."""
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(N_OBSERVATIONS, 2))
    forecasts = {
        "deterministic": lambda: np.zeros((N_DRAWS, 2)),
        "calibrated": lambda: rng.normal(size=(N_DRAWS, 2)),
        "under_dispersed": lambda: rng.normal(size=(N_DRAWS, 2)) * 0.5,
        "over_dispersed": lambda: rng.normal(size=(N_DRAWS, 2)) * 2.0,
    }
    return {
        name: _expected_score(energy_score, fn, observations)
        for name, fn in forecasts.items()
    }


def test_calibrated_ensemble_is_ranked_best(scores):
    assert scores["calibrated"] == min(scores.values())


@pytest.mark.parametrize("competitor", ["deterministic", "under_dispersed", "over_dispersed"])
def test_miscalibration_is_penalised(scores, competitor):
    assert scores[competitor] > scores["calibrated"]


def test_zero_spread_forecast_is_ranked_worst(scores):
    """mode="mean" produces N identical draws; the gate must not reward that."""
    assert scores["deterministic"] == max(scores.values())


def test_identical_draws_get_no_spread_credit(energy_score):
    """The pairwise term is what credits spread; it must vanish for a point mass."""
    observation = np.array([0.4, -0.3])
    point_mass = np.zeros((8, 2))
    assert energy_score(point_mass, observation) == pytest.approx(
        float(np.linalg.norm(observation)), rel=1e-12
    )


def test_single_draw_matches_its_distance(energy_score):
    observation = np.array([1.0, 0.0])
    assert energy_score(np.zeros((1, 2)), observation) == pytest.approx(1.0, rel=1e-12)
