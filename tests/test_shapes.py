"""Smoke tests: forward + sample produce correct shapes."""

import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite


def _toy_batch(B=2, N=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "P_t": torch.randn(B, N, 3, generator=g),
        "V_t": torch.randn(B, N, 13, 3, generator=g) * 0.3,
        "P_1": torch.randn(B, N, 3, generator=g),
        "V_1": torch.randn(B, N, 13, 3, generator=g) * 0.3,
        "res_index": torch.randint(0, 20, (B, N), generator=g),
        "residue_mask": torch.ones(B, N, dtype=torch.bool),
        "delta_ns": torch.ones(B),
    }


def test_forward_shape():
    model = DeepJumpLite(ModelConfig(cond_layers=2, transport_layers=2))
    batch = _toy_batch()
    out = model(batch)
    assert out["P_hat_1"].shape == batch["P_t"].shape
    assert out["V_hat_1"] is None  # heavy output off by default


def test_forward_shape_with_heavy():
    model = DeepJumpLite(ModelConfig(cond_layers=2, transport_layers=2), predict_heavy=True)
    batch = _toy_batch()
    out = model(batch)
    assert out["P_hat_1"].shape == batch["P_t"].shape
    assert out["V_hat_1"].shape == batch["V_t"].shape


def test_sample_shape():
    model = DeepJumpLite(ModelConfig(cond_layers=2, transport_layers=2))
    batch = _toy_batch()
    P, V = model.sample(batch, steps=5)
    assert P.shape == batch["P_t"].shape
    assert V.shape == batch["V_t"].shape


def test_zero_source_validation_endpoint_matches_mean_sampling_bitwise():
    batch = _toy_batch(B=2, N=6, seed=11)
    batch["atom_mask"] = torch.ones(2, 6, 13, dtype=torch.bool)
    model = DeepJumpLite(
        ModelConfig(
            cond_layers=1,
            transport_layers=1,
            source_noise_v=True,
            source_noise_sigma_v=0.0,
        ),
        noise_sigma=0.0,
        predict_heavy=True,
    ).eval()

    endpoint = model(batch, tau=torch.zeros(2))
    mean_P, mean_V = model.sample(batch, steps=1, mode="mean")

    assert torch.equal(endpoint["P_hat_1"], mean_P)
    assert torch.equal(endpoint["V_hat_1"], mean_V)


def test_rollout_shape():
    from deepjump.sampling import rollout

    model = DeepJumpLite(ModelConfig(cond_layers=2, transport_layers=2), predict_heavy=True)
    batch = _toy_batch()
    traj, accepts = rollout(model, batch, n_steps=4, ode_steps=3)
    assert len(traj) == 5 and accepts == []
    assert traj[-1][0].shape == batch["P_t"].shape
    assert traj[-1][1].shape == batch["V_t"].shape

    traj_g, accepts_g = rollout(model, batch, n_steps=4, ode_steps=3, mode="mean", gate=True)
    assert len(traj_g) == 5 and len(accepts_g) == 4  # one acceptance flag per step


def test_backward_runs():
    from deepjump.losses import pairwise_vector_huber_loss

    model = DeepJumpLite(ModelConfig(cond_layers=2, transport_layers=2))
    batch = _toy_batch()
    out = model(batch)
    loss = pairwise_vector_huber_loss(out["P_hat_1"], batch["P_1"], batch["residue_mask"])
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert all(grads) and loss.item() >= 0
