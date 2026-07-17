import numpy as np
import torch
from types import SimpleNamespace

import pytest

from scripts.transition_robustness_eval import (
    _evaluate_cell,
    crossfit_reference_replica,
    energy_distance,
    energy_score,
    summarize_transition_methods,
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


class _SyntheticHandle:
    name = "synthetic"

    def __init__(self):
        self.layout = SimpleNamespace(
            num_residues=4,
            res_index=np.arange(4),
            bond_mask=np.ones(3, dtype=bool),
            atom_mask=np.ones((4, 13), dtype=bool),
        )

    def replicas(self, temperature, replicas):
        return [(temperature, replicas[0], 48)]

    def coords(self, temperature, replica, frame):
        rng = np.random.default_rng(1000 * replica + frame)
        return rng.normal(size=(4, 3)).astype(np.float32)


class _NoopModel:
    def sample(self, batch, mode):
        return batch["P_t"], batch["V_t"]


def test_evaluate_cell_pairs_model_equal_to_noop_on_identical_evidence(monkeypatch):
    monkeypatch.setattr(
        "scripts.transition_robustness_eval.apply_layout",
        lambda coordinates, layout: (
            coordinates,
            torch.zeros(layout.num_residues, 13, 3),
        ),
    )
    args = SimpleNamespace(
        max_features=6,
        real_frames=40,
        lag=1,
        tica_components=2,
        clusters=4,
        seed=17,
        msm_lag=1,
        msm_pseudocount=1e-8,
        starts=6,
        draws=4,
    )
    result = _evaluate_cell(
        _SyntheticHandle(),
        _SyntheticHandle().layout,
        _NoopModel(),
        torch.device("cpu"),
        {"crop_length": 4},
        1,
        ["mean"],
        [0, 1],
        args,
        temperature=320,
        replica=0,
        seed_offset=0,
    )
    for metric in (
        "mean_energy_score",
        "transition_energy_distance",
        "msm_row_jsd_bits",
    ):
        assert result["methods"]["mean"][metric] == pytest.approx(
            result["methods"]["noop"][metric]
        )


def test_transition_summary_gain_is_noop_minus_model_and_equal_model_fails():
    rows = []
    for index in range(20):
        noop = {
            "mean_energy_score": 1.0 + index / 100,
            "transition_energy_distance": 1.0,
            "msm_row_jsd_bits": 0.8 + index / 100,
        }
        model = {
            "mean_energy_score": noop["mean_energy_score"] - 0.2,
            "transition_energy_distance": 0.9,
            "msm_row_jsd_bits": noop["msm_row_jsd_bits"] - 0.1,
        }
        rows.append({"methods": {"noop": noop, "mean": model}})
    summary = summarize_transition_methods(rows, ["mean"], seed=5)
    assert summary["mean"]["paired_energy_score_gain"]["passes"] is True
    assert summary["mean"]["paired_energy_score_gain"][
        "mean_baseline_minus_model"
    ] == pytest.approx(0.2)
    assert summary["mean"]["paired_msm_row_jsd_gain"]["passes"] is True

    equal_rows = [
        {"methods": {"noop": row["methods"]["noop"], "mean": row["methods"]["noop"]}}
        for row in rows
    ]
    equal_summary = summarize_transition_methods(equal_rows, ["mean"], seed=5)
    assert equal_summary["mean"]["paired_energy_score_gain"]["passes"] is False
    assert equal_summary["mean"]["paired_msm_row_jsd_gain"]["passes"] is False
