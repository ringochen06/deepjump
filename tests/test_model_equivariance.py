"""End-to-end SE(3)-equivariance of the model (the silent-bug gate for the net).

If the input structure is rotated by R, the predicted CA structure X_hat_1 must
rotate by the same R. We test the deterministic x1 prediction (fixed tau, fixed
interpolation noise) so the only variable is the rotation.
"""

import torch

from deepjump.config import ModelConfig
from deepjump.model import DeepJumpLite
from deepjump.model.deepjump import count_parameters


def _random_rotation(seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _toy_batch(B=2, N=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    P_t = torch.randn(B, N, 3, generator=g) * 5
    V_t = torch.randn(B, N, 13, 3, generator=g) * 0.5
    P_1 = P_t + torch.randn(B, N, 3, generator=g) * 0.3
    V_1 = V_t + torch.randn(B, N, 13, 3, generator=g) * 0.1
    res_index = torch.randint(0, 20, (B, N), generator=g)
    residue_mask = torch.ones(B, N, dtype=torch.bool)
    delta_ns = torch.ones(B)
    return {
        "P_t": P_t, "V_t": V_t, "P_1": P_1, "V_1": V_1,
        "res_index": res_index, "residue_mask": residue_mask, "delta_ns": delta_ns,
    }


def _rotate_batch(batch, R):
    out = dict(batch)
    out["P_t"] = batch["P_t"] @ R.T
    out["V_t"] = batch["V_t"] @ R.T
    out["P_1"] = batch["P_1"] @ R.T
    out["V_1"] = batch["V_1"] @ R.T
    return out


def test_model_is_rotation_equivariant():
    torch.manual_seed(0)
    model = DeepJumpLite(ModelConfig(hidden=32, vector_channels=16, num_heads=4,
                                     cond_layers=2, transport_layers=2)).eval()
    model.noise_sigma = 0.0  # isolate the network from lab-frame interpolation noise
    batch = _toy_batch()
    R = _random_rotation(1)

    tau = torch.tensor([0.3, 0.7])
    out = model(batch, tau=tau)["P_hat_1"]
    out_rot = model(_rotate_batch(batch, R), tau=tau)["P_hat_1"]

    assert torch.allclose(out_rot, out @ R.T, atol=1e-4), (
        (out_rot - out @ R.T).abs().max().item()
    )


def test_heavy_output_is_rotation_equivariant():
    """With predict_heavy, both P_hat_1 and V_hat_1 must rotate with the input."""
    torch.manual_seed(0)
    model = DeepJumpLite(ModelConfig(hidden=32, vector_channels=16, num_heads=4,
                                     cond_layers=2, transport_layers=2),
                         predict_heavy=True).eval()
    model.noise_sigma = 0.0
    batch = _toy_batch()
    R = _random_rotation(3)
    tau = torch.tensor([0.4, 0.6])

    out = model(batch, tau=tau)
    out_rot = model(_rotate_batch(batch, R), tau=tau)
    assert torch.allclose(out_rot["P_hat_1"], out["P_hat_1"] @ R.T, atol=1e-4)
    assert out["V_hat_1"] is not None
    assert torch.allclose(out_rot["V_hat_1"], out["V_hat_1"] @ R.T, atol=1e-4), (
        (out_rot["V_hat_1"] - out["V_hat_1"] @ R.T).abs().max().item()
    )


def test_param_count_reasonable():
    model = DeepJumpLite(ModelConfig())
    n = count_parameters(model)
    # H=32 lite model should be in the low 100k-1M range
    assert 50_000 < n < 3_000_000, n
