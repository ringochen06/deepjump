import numpy as np

import pytest

from scripts.transition_robustness_eval import (
    crossfit_reference_replica,
    energy_distance,
    energy_score,
)


def test_energy_score_is_zero_for_exact_ensemble():
    observation = np.array([1.0, -2.0])
    samples = np.repeat(observation[None], 4, axis=0)
    assert energy_score(samples, observation) == 0.0


def test_energy_score_rewards_a_closer_forecast():
    observation = np.array([1.0, 0.0])
    close = np.array([[0.9, 0.0], [1.1, 0.0]])
    far = np.array([[-2.0, 0.0], [-1.0, 0.0]])
    assert energy_score(close, observation) < energy_score(far, observation)


def test_energy_score_uses_off_diagonal_fair_ensemble_term():
    observation = np.array([0.0])
    samples = np.array([[-1.0], [1.0]])
    # First term is 1; the only two ordered off-diagonal distances are both 2.
    assert energy_score(samples, observation) == 0.0


def test_energy_distance_detects_shift_and_has_zero_self_distance():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert energy_distance(x, x) == 0.0
    assert energy_distance(x, x + np.array([3.0, 0.0])) > 0.0


def test_crossfit_reference_replica_is_different_and_deterministic():
    replicas = [0, 1, 2, 3, 4]
    assert [crossfit_reference_replica(replica, replicas) for replica in replicas] == [
        1, 2, 3, 4, 0
    ]
    with pytest.raises(ValueError, match="at least two"):
        crossfit_reference_replica(0, [0])
    with pytest.raises(ValueError, match="not in"):
        crossfit_reference_replica(5, replicas)
