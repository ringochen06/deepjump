import numpy as np
import torch

from scripts.rollout_robustness_eval import (
    _local_geometry,
    one_step_persistence_trajectory,
    select_validation_domains,
    summarize_domains,
    teacher_forced_mean_trajectory,
)
from deepjump.sampling import _static_fields


def test_select_validation_domains_is_deterministic_and_spread():
    paths = [f"domain_{i:02d}" for i in range(10)]
    assert select_validation_domains(paths, 3) == ["domain_00", "domain_04", "domain_09"]


def test_summarize_domains_compares_final_step_to_noop():
    rows = [
        {"methods": {"noop": {"rmsd": [0.0, 2.0, 4.0]}, "model": {"rmsd": [0.0, 1.0, 3.0]}}},
        {"methods": {"noop": {"rmsd": [0.0, 2.0, 3.0]}, "model": {"rmsd": [0.0, 3.0, 4.0]}}},
    ]
    summary = summarize_domains(rows, "model", 2)
    assert np.isclose(summary["mean_final_rmsd"], 3.5)
    assert np.isclose(summary["mean_rollout_rmsd"], 2.75)
    assert summary["domains_better_than_noop_final"] == 1
    assert summary["finite"]


def test_local_geometry_reports_tails_and_respects_gaps():
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [10.0, 0.0, 0.0], [11.0, 0.0, 0.0],
                            [11.0, 1.0, 0.0]]])
    pred = target.clone()
    pred[:, 2] = torch.tensor([100.0, 100.0, 0.0])  # ignored across topology gap
    pred[:, 3] = torch.tensor([101.0, 100.0, 0.0])
    pred[:, 4] = torch.tensor([102.0, 100.0, 0.0])  # valid bond/angle error
    bond_mask = torch.tensor([[True, False, True, True]])

    stats = _local_geometry(pred, target, bond_mask)
    assert np.isclose(stats["bond_mean"], 1.0)
    assert np.isclose(stats["bond_max"], 1.0)
    assert stats["bond_mae_real"] == 0.0
    assert stats["angle_cos_mae_real"] > 0.0


def test_rollout_static_fields_preserve_atom_mask_for_joint_source_noise():
    batch = {
        "res_index": torch.zeros(1, 3, dtype=torch.long),
        "delta_ns": torch.ones(1),
        "residue_mask": torch.ones(1, 3, dtype=torch.bool),
        "atom_mask": torch.ones(1, 3, 13, dtype=torch.bool),
    }
    assert _static_fields(batch)["atom_mask"] is batch["atom_mask"]


class _OffsetModel:
    def __init__(self):
        self.inputs = []

    def sample(self, batch, *, steps, mode):
        assert steps == 1 and mode == "mean"
        self.inputs.append(batch["P_t"].clone())
        return batch["P_t"] + 10.0, batch["V_t"]


def test_teacher_forced_trajectory_uses_each_real_preceding_state():
    model = _OffsetModel()
    positions = [torch.full((1, 2, 3), float(step)) for step in range(4)]
    vectors = [torch.full((1, 2, 13, 3), float(step)) for step in range(4)]
    static = {"residue_mask": torch.ones(1, 2, dtype=torch.bool)}

    trajectory = teacher_forced_mean_trajectory(model, positions, vectors, static)

    assert len(trajectory) == 4
    assert [value[0, 0, 0].item() for value in model.inputs] == [0.0, 1.0, 2.0]
    assert [frame[0][0, 0, 0].item() for frame in trajectory] == [0.0, 10.0, 11.0, 12.0]


def test_one_step_persistence_uses_preceding_real_frame():
    positions = [torch.full((1, 2, 3), float(step)) for step in range(4)]
    vectors = [torch.full((1, 2, 13, 3), float(step)) for step in range(4)]

    trajectory = one_step_persistence_trajectory(positions, vectors)

    assert len(trajectory) == 4
    assert [frame[0][0, 0, 0].item() for frame in trajectory] == [0.0, 0.0, 1.0, 2.0]
