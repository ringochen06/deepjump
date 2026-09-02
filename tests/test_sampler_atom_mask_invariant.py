"""V's zero-padded atom slots must stay exactly zero through sampling.

`V` is [N, 13, 3] but residues carry different heavy-atom counts, so a large
fraction of its slots is structural zero padding that training never populates.
The endpoint-prediction ODE has no reason to respect that: it writes into every
slot, the garbage re-enters the conditioner on the next iteration as an input
outside the training distribution, and it compounds until the structure
disintegrates -- measured on the formal 500k checkpoint, padded-slot magnitude
reached 3.2e3 and the CA-CA bond went 3.8 -> 40.8 A by 150 Euler steps.

Per-step re-masking (`project_v_atom_mask`, the first-party release's behaviour)
is therefore a correctness property and is on by default. These tests pin the
invariant, the default, and the failure mode it guards against.
"""

import pytest
import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite


def _model(seed=0, predict_heavy=True):
    torch.manual_seed(seed)
    cfg = ModelConfig(hidden=16, vector_channels=8, num_heads=2, cond_layers=1,
                      transport_layers=1, predict_heavy=predict_heavy)
    return DeepJumpLite(cfg, noise_sigma=0.1, predict_heavy=predict_heavy).eval()


def _batch(B=2, N=6, seed=0):
    """Toy batch whose atom_mask leaves a known set of V slots as padding."""
    g = torch.Generator().manual_seed(seed)
    atom_mask = torch.zeros(B, N, 13, dtype=torch.bool)
    atom_mask[..., :4] = True  # every residue keeps 4 real slots, 9 are padding
    atom_mask[:, 0, 4:7] = True  # one residue carries a longer side chain
    V_t = torch.randn(B, N, 13, 3, generator=g) * 0.5 * atom_mask.unsqueeze(-1)
    return {
        "P_t": torch.randn(B, N, 3, generator=g) * 5,
        "V_t": V_t,
        "res_index": torch.randint(0, 20, (B, N), generator=g),
        "residue_mask": torch.ones(B, N, dtype=torch.bool),
        "atom_mask": atom_mask,
        "delta_ns": torch.ones(B),
    }


def test_project_v_atom_mask_defaults_on():
    """The zero-padding invariant is a correctness property, not an opt-in."""
    import inspect

    default = inspect.signature(DeepJumpLite.sample).parameters["project_v_atom_mask"].default
    assert default is True


@pytest.mark.parametrize("steps", [1, 20, 150])
@pytest.mark.parametrize("mode", ["ode", "mean"])
def test_padded_v_slots_stay_zero(steps, mode):
    """Sampling must not write into slots the atom_mask marks as absent."""
    model, batch = _model(), _batch()
    padding = ~batch["atom_mask"]
    assert batch["V_t"][padding].abs().max() == 0, "fixture must start clean"

    with torch.no_grad():
        _, V = model.sample(batch, steps=steps, mode=mode,
                            generator=torch.Generator().manual_seed(0))

    assert torch.isfinite(V).all()
    assert V[padding].abs().max().item() == 0.0


@pytest.mark.parametrize("steps", [20, 150])
def test_disabling_the_mask_lets_padding_drift(steps):
    """Guard the guard: without re-masking the padded slots are provably polluted.

    If this ever stops holding, the invariant test above has gone vacuous -- it
    would be passing because the sampler no longer touches V at all.
    """
    model, batch = _model(), _batch()
    padding = ~batch["atom_mask"]

    with torch.no_grad():
        _, V = model.sample(batch, steps=steps, mode="ode",
                            generator=torch.Generator().manual_seed(0),
                            project_v_atom_mask=False)

    assert V[padding].abs().max().item() > 0.0


def test_missing_atom_mask_is_a_loud_error():
    """A caller that cannot supply atom_mask cannot maintain the invariant."""
    model, batch = _model(), _batch()
    del batch["atom_mask"]
    with pytest.raises(ValueError, match="atom_mask"):
        model.sample(batch, steps=2, mode="ode")


def test_rollout_keeps_the_invariant_across_chained_jumps():
    """Chaining is where the pollution compounded, so pin it end to end."""
    from deepjump.sampling import rollout

    model, batch = _model(), _batch()
    padding = ~batch["atom_mask"]
    traj, _ = rollout(model, batch, n_steps=4, ode_steps=10, mode="ode",
                      generator=torch.Generator().manual_seed(0))

    assert len(traj) == 5
    for index, (P, V) in enumerate(traj):
        assert torch.isfinite(P).all(), f"positions diverged at step {index}"
        assert V[padding].abs().max().item() == 0.0, f"padding polluted at step {index}"
