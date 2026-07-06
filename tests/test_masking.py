"""Padding must not affect the loss or masked outputs."""

import torch

from deepjump.config import ModelConfig
from deepjump.losses import pairwise_vector_huber_loss
from deepjump.model import DeepJumpLite


def test_loss_ignores_padding():
    torch.manual_seed(0)
    B, N = 2, 8
    P_hat = torch.randn(B, N, 3)
    P_gt = torch.randn(B, N, 3)
    mask = torch.ones(B, N, dtype=torch.bool)

    base = pairwise_vector_huber_loss(P_hat, P_gt, mask)

    # Append padded residues with garbage values -> loss must be unchanged.
    pad = 3
    P_hat2 = torch.cat([P_hat, torch.randn(B, pad, 3) * 100], dim=1)
    P_gt2 = torch.cat([P_gt, torch.randn(B, pad, 3) * 100], dim=1)
    mask2 = torch.cat([mask, torch.zeros(B, pad, dtype=torch.bool)], dim=1)
    padded = pairwise_vector_huber_loss(P_hat2, P_gt2, mask2)

    assert torch.allclose(base, padded, atol=1e-5), (base.item(), padded.item())


def test_padding_does_not_change_real_predictions():
    """A model prediction on real residues is invariant to added padding."""
    torch.manual_seed(0)
    model = DeepJumpLite(ModelConfig(hidden=32, vector_channels=16, num_heads=4,
                                     cond_layers=2, transport_layers=2)).eval()
    model.noise_sigma = 0.0  # deterministic: isolate masking from interpolation noise
    B, N = 2, 6
    g = torch.Generator().manual_seed(1)

    def make(n):
        return {
            "P_t": torch.randn(B, n, 3, generator=g),
            "V_t": torch.randn(B, n, 13, 3, generator=g) * 0.3,
            "P_1": torch.randn(B, n, 3, generator=g),
            "V_1": torch.randn(B, n, 13, 3, generator=g) * 0.3,
            "res_index": torch.randint(0, 20, (B, n), generator=g),
            "residue_mask": torch.ones(B, n, dtype=torch.bool),
            "delta_ns": torch.ones(B),
        }

    base = make(N)
    padded = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
    pad = 4
    for key, filler in [("P_t", 50.0), ("P_1", 50.0)]:
        padded[key] = torch.cat([base[key], torch.randn(B, pad, 3) * filler], dim=1)
    padded["V_t"] = torch.cat([base["V_t"], torch.randn(B, pad, 13, 3)], dim=1)
    padded["V_1"] = torch.cat([base["V_1"], torch.randn(B, pad, 13, 3)], dim=1)
    padded["res_index"] = torch.cat([base["res_index"], torch.zeros(B, pad, dtype=torch.long)], dim=1)
    padded["residue_mask"] = torch.cat(
        [base["residue_mask"], torch.zeros(B, pad, dtype=torch.bool)], dim=1
    )

    tau = torch.tensor([0.4, 0.6])
    g1 = torch.Generator().manual_seed(7)
    out_base = model(base, tau=tau, generator=g1)["P_hat_1"]
    g2 = torch.Generator().manual_seed(7)
    out_pad = model(padded, tau=tau, generator=g2)["P_hat_1"][:, :N]

    assert torch.allclose(out_base, out_pad, atol=1e-4), (out_base - out_pad).abs().max().item()
