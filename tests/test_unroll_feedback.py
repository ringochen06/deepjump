import torch
from torch import nn

from deepjump.config import Config
from deepjump.training import total_loss


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, batch, tau=None):
        self.calls.append({
            "tau": None if tau is None else tau.detach().clone(),
            "P_t": batch["P_t"].detach().clone(),
        })
        return {
            "P_hat_1": batch["P_t"] + self.bias,
            "V_hat_1": batch["V_t"] + self.bias,
            "tau": tau,
        }


def _batch():
    B, N = 2, 4
    zeros_p = torch.zeros(B, N, 3)
    zeros_v = torch.zeros(B, N, 13, 3)
    return {
        "P_t": zeros_p,
        "V_t": zeros_v,
        "P_1": torch.ones_like(zeros_p),
        "V_1": torch.ones_like(zeros_v),
        "P_2": torch.full_like(zeros_p, 2.0),
        "V_2": torch.full_like(zeros_v, 2.0),
        "P_3": torch.full_like(zeros_p, 3.0),
        "V_3": torch.full_like(zeros_v, 3.0),
        "residue_mask": torch.ones(B, N, dtype=torch.bool),
        "atom_mask": torch.ones(B, N, 13, dtype=torch.bool),
    }


def test_unroll_feedback_uses_honest_tau_zero_chain():
    batch = _batch()
    cfg = Config()
    cfg.train.w_unroll = 0.5
    cfg.train.w_ca = 1.0
    model = RecordingModel()
    # Primary training output represents a random-tau call and deliberately
    # contains an implausible value that must never seed the feedback chain.
    primary = {
        "P_hat_1": torch.full_like(batch["P_t"], 99.0, requires_grad=True),
        "V_hat_1": torch.full_like(batch["V_t"], 99.0, requires_grad=True),
    }

    loss, comps = total_loss(primary, batch, cfg, model)

    assert len(model.calls) == 3  # tau=0 seed, supervised step 2, supervised step 3
    assert all(torch.count_nonzero(call["tau"]) == 0 for call in model.calls)
    assert torch.equal(model.calls[0]["P_t"], batch["P_t"])
    assert torch.allclose(model.calls[1]["P_t"], batch["P_t"] + 0.25)
    assert torch.allclose(model.calls[2]["P_t"], batch["P_t"] + 0.50)
    assert "ca2" in comps and "ca3" in comps
    loss.backward()
    assert model.bias.grad is not None
