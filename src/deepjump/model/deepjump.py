"""DeepJump-lite model: p(X_{t+delta} | X_t, sequence, delta) via x1 prediction.

Generative framework (AlphaFlow-style, simplified from EquiJump's two-sided
stochastic interpolant to a plain ODE that learns only x1_hat):

    X^0     = X_t + sigma * noise           (noised current frame)
    X^tau   = (1 - tau) X^0 + tau X_{t+delta}   (linear interpolant, tau ~ U(0,1))
    network predicts  X_hat_1 ~= X_{t+delta}
    sampling ODE drift  b = (X_hat_1 - X^tau) / (1 - tau)   integrated tau: 0 -> 1

This pass predicts CA only (P); heavy-atom offsets V are used as INPUT features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .conditioner import Conditioner
from .transport import Transport


class DeepJumpLite(nn.Module):
    def __init__(self, cfg: ModelConfig, noise_sigma: float = 0.1, predict_heavy: bool = False):
        super().__init__()
        self.cfg = cfg
        self.noise_sigma = noise_sigma
        self.predict_heavy = predict_heavy
        self.input_aug_sigma = getattr(cfg, "input_aug_sigma", 0.0)
        common = dict(
            hidden=cfg.hidden,
            vec_channels=cfg.vector_channels,
            num_heads=cfg.num_heads,
            seq_ks=cfg.seq_embed_ks,
            num_dist_basis=cfg.num_dist_basis,
            dist_cutoff=cfg.dist_cutoff,
        )
        self.conditioner = Conditioner(num_layers=cfg.cond_layers, **common)
        self.transport = Transport(
            num_layers=cfg.transport_layers, predict_heavy=predict_heavy, **common
        )

    # ---- pieces -------------------------------------------------------------
    def encode(self, batch):
        return self.conditioner(
            batch["P_t"], batch["V_t"], batch["res_index"],
            batch["delta_ns"], batch["residue_mask"],
        )

    def interpolate(self, P_t, V_t, P_1, V_1, tau, generator=None):
        """Build the intermediate state X^tau. tau: [B]."""
        noise = torch.randn(P_t.shape, generator=generator, device=P_t.device)
        P0 = P_t + self.noise_sigma * noise
        aP = tau.view(-1, 1, 1)
        aV = tau.view(-1, 1, 1, 1)
        P_tau = (1 - aP) * P0 + aP * P_1
        V_tau = (1 - aV) * V_t + aV * V_1
        return P_tau, V_tau

    def predict_x1(self, P_tau, V_tau, tau, ctx, P_t, V_t, residue_mask):
        s_ctx, vec_ctx = ctx
        return self.transport(P_tau, V_tau, tau, s_ctx, vec_ctx, P_t, V_t, residue_mask)

    # ---- training forward ---------------------------------------------------
    def forward(self, batch, tau=None, generator=None):
        B = batch["P_t"].shape[0]
        device = batch["P_t"].device
        if tau is None:
            tau = torch.rand(B, device=device, generator=generator)

        # Rollout-robustness augmentation: corrupt the CONDITIONER input X_t (and the
        # residual anchor / interpolant base) with per-sample noise, but keep the clean
        # target X_1. This teaches the model to recover from imperfect inputs -- the
        # exact situation it faces when fed its own predictions during a rollout.
        P_t, V_t = batch["P_t"], batch["V_t"]
        if self.training and self.input_aug_sigma > 0:
            a = (self.input_aug_sigma * torch.rand(B, device=device)).view(-1, 1, 1)
            P_t = P_t + a * torch.randn_like(P_t)
            V_t = V_t + a.unsqueeze(-1) * torch.randn_like(V_t)
        aug_batch = {**batch, "P_t": P_t, "V_t": V_t}

        ctx = self.encode(aug_batch)
        P_tau, V_tau = self.interpolate(
            P_t, V_t, batch["P_1"], batch["V_1"], tau, generator
        )
        P_hat_1, V_hat_1 = self.predict_x1(
            P_tau, V_tau, tau, ctx, P_t, V_t, batch["residue_mask"]
        )
        return {"P_hat_1": P_hat_1, "V_hat_1": V_hat_1, "tau": tau}

    # ---- sampling (interface for eval/rollout) ------------------------------
    @torch.no_grad()
    def sample(self, batch, steps: int = 20, mode: str = "ode", generator=None):
        """Predict X_{t+delta} from X_t. Returns (P, V).

        mode="ode"  : integrate the Euler ODE from tau=0 (X_t + noise) to tau=1.
        mode="mean" : one-shot x1 prediction at tau=0 (the deterministic conditional
                      mean); no integration, no noise. Stable but conservative.
        When predict_heavy, V is produced the same way; else V stays V_t.
        """
        device = batch["P_t"].device
        P_t, V_t = batch["P_t"], batch["V_t"]
        mask = batch["residue_mask"]
        ctx = self.encode(batch)

        if mode == "mean":
            tau0 = torch.zeros(P_t.shape[0], device=device)
            P_hat_1, V_hat_1 = self.predict_x1(P_t, V_t, tau0, ctx, P_t, V_t, mask)
            V_out = V_hat_1 if (self.predict_heavy and V_hat_1 is not None) else V_t
            return P_hat_1, V_out

        noise = torch.randn(P_t.shape, generator=generator).to(device)
        P = P_t + self.noise_sigma * noise
        V = V_t.clone()
        taus = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps):
            tau = taus[i].expand(P_t.shape[0])
            dt = (taus[i + 1] - taus[i]).item()
            inv = dt / max(1 - taus[i].item(), 1e-3)
            P_hat_1, V_hat_1 = self.predict_x1(P, V, tau, ctx, P_t, V_t, mask)
            P = P + (P_hat_1 - P) * inv
            if self.predict_heavy and V_hat_1 is not None:
                V = V + (V_hat_1 - V) * inv
        return P, V


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
