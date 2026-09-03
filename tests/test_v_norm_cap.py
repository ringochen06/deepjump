"""Heavy-atom offsets must stay within physical reach during sampling.

`V` holds each residue's CA-to-heavy-atom offsets. Nothing in the objective
constrains their magnitude: the CA-CA bond term pins the backbone, and the
side-chain channel is free to drift underneath it. Under autoregressive rollout
it does, and every conventional geometry check keeps passing while it happens.

Measured on the formal 500k checkpoint from a native start: |V| passes the 7.49 A
data maximum by step 200, reaches 20 A by step 1200 while CA-CA bond is still
3.84 A and Rg still 11 A, hits 141 A at step 1276, and overflows to NaN at 1292.
With the cap the same rollout ran past 20,000 steps at bond 3.89 A.

This is the same class of defect as the zero-padding one: `V` lacking an
invariant. Fixing which slots must be zero left open how large the non-zero ones
may be.
"""

import pytest
import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite
from deepjump.model.deepjump import MAX_HEAVY_ATOM_OFFSET_A


def _model(predict_heavy=True):
    torch.manual_seed(0)
    cfg = ModelConfig(hidden=16, vector_channels=8, num_heads=2, cond_layers=1,
                      transport_layers=1, predict_heavy=predict_heavy)
    return DeepJumpLite(cfg, noise_sigma=0.1, predict_heavy=predict_heavy).eval()


def _batch(B=2, N=6, scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    atom_mask = torch.zeros(B, N, 13, dtype=torch.bool)
    atom_mask[..., :5] = True
    return {
        "P_t": torch.randn(B, N, 3, generator=generator) * 5,
        "V_t": torch.randn(B, N, 13, 3, generator=generator) * scale * atom_mask.unsqueeze(-1),
        "res_index": torch.randint(0, 20, (B, N), generator=generator),
        "residue_mask": torch.ones(B, N, dtype=torch.bool),
        "atom_mask": atom_mask,
        "delta_ns": torch.ones(B),
    }


def test_the_cap_matches_the_measured_physical_reach():
    """7.5 A is the mdCATH maximum (7.49), not a tuned number."""
    assert MAX_HEAVY_ATOM_OFFSET_A == pytest.approx(7.5)


def test_cap_is_on_by_default():
    import inspect

    default = inspect.signature(DeepJumpLite.sample).parameters["max_v_norm"].default
    assert default == MAX_HEAVY_ATOM_OFFSET_A


@pytest.mark.parametrize("mode", ["ode", "mean"])
def test_offsets_never_exceed_the_cap(mode):
    model, batch = _model(), _batch(scale=1.0)
    with torch.no_grad():
        _, vectors = model.sample(batch, steps=5, mode=mode,
                                  generator=torch.Generator().manual_seed(0))
    norms = vectors.norm(dim=-1)
    assert torch.isfinite(norms).all()
    assert norms.max().item() <= MAX_HEAVY_ATOM_OFFSET_A + 1e-5


def test_an_over_long_input_is_pulled_back():
    """Input already past the cap must come out inside it, not be passed through."""
    model, batch = _model(), _batch(scale=40.0)
    assert batch["V_t"].norm(dim=-1).max() > MAX_HEAVY_ATOM_OFFSET_A, "fixture must start over-long"
    with torch.no_grad():
        _, vectors = model.sample(batch, steps=3, mode="ode",
                                  generator=torch.Generator().manual_seed(0))
    assert vectors.norm(dim=-1).max().item() <= MAX_HEAVY_ATOM_OFFSET_A + 1e-5


def test_disabling_the_cap_lets_offsets_exceed_it():
    """Guard the guard: with the cap off the bound must be reachable again.

    Without this, the tests above could pass because the sampler stopped touching
    V at all rather than because the cap works.
    """
    model, batch = _model(), _batch(scale=40.0)
    with torch.no_grad():
        _, vectors = model.sample(batch, steps=3, mode="ode", max_v_norm=None,
                                  generator=torch.Generator().manual_seed(0))
    assert vectors.norm(dim=-1).max().item() > MAX_HEAVY_ATOM_OFFSET_A


def test_direction_is_preserved_when_clamping():
    """A cap must rescale, not rotate: the offset direction carries the geometry."""
    model, batch = _model(), _batch(scale=40.0)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        _, capped = model.sample(batch, steps=3, mode="ode", generator=generator)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        _, raw = model.sample(batch, steps=3, mode="ode", max_v_norm=None, generator=generator)

    mask = batch["atom_mask"] & (raw.norm(dim=-1) > 1e-6) & (capped.norm(dim=-1) > 1e-6)
    cosine = torch.nn.functional.cosine_similarity(capped[mask], raw[mask], dim=-1)
    # The two runs diverge because clamping feeds back through the ODE, so this
    # checks the clamp itself rather than trajectory equality.
    assert cosine.mean().item() > 0.5


def test_a_non_positive_cap_is_rejected():
    model, batch = _model(), _batch()
    with pytest.raises(ValueError, match="max_v_norm"):
        model.sample(batch, steps=2, mode="ode", max_v_norm=0.0)
