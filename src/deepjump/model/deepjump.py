"""DeepJump-lite model: p(X_{t+delta} | X_t, sequence, delta) via x1 prediction.

Generative framework (AlphaFlow-style, simplified from EquiJump's two-sided
stochastic interpolant to a plain ODE that learns only x1_hat):

    X^0     = X_t + sigma * noise           (noised current frame; P and optionally V)
    X^tau   = (1 - tau) X^0 + tau X_{t+delta}   (linear interpolant, tau ~ U(0,1))
    network predicts  X_hat_1 ~= X_{t+delta}
    sampling ODE drift  b = (X_hat_1 - X^tau) / (1 - tau)   integrated tau: 0 -> 1

The full configuration predicts both CA positions P and heavy-atom offsets V.
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
        self.noise_sigma_v = getattr(cfg, "source_noise_sigma_v", None)
        self.predict_heavy = predict_heavy
        self.input_aug_sigma = getattr(cfg, "input_aug_sigma", 0.0)
        self.source_noise_v = getattr(cfg, "source_noise_v", False)
        tensor_cloud01 = getattr(cfg, "tensor_cloud01", False)
        tensor_cloud01_vector_only_attention = getattr(
            cfg, "tensor_cloud01_vector_only_attention", False
        )
        if tensor_cloud01_vector_only_attention and not tensor_cloud01:
            raise ValueError(
                "tensor_cloud01_vector_only_attention requires tensor_cloud01"
            )
        if tensor_cloud01 and any(
            getattr(cfg, name, False) for name in ("vector_qk", "tensor_qkv", "paper_ff")
        ):
            raise ValueError(
                "tensor_cloud01 is a dedicated path and cannot be combined with "
                "vector_qk, tensor_qkv, or paper_ff"
            )
        common = dict(
            hidden=cfg.hidden,
            vec_channels=cfg.vector_channels,
            num_heads=cfg.num_heads,
            seq_ks=cfg.seq_embed_ks,
            num_dist_basis=cfg.num_dist_basis,
            dist_cutoff=cfg.dist_cutoff,
            vector_qk=getattr(cfg, "vector_qk", False),
            tensor_qkv=getattr(cfg, "tensor_qkv", False),
            paper_ff=getattr(cfg, "paper_ff", False),
            tensor_cloud01=tensor_cloud01,
            tensor_cloud01_vector_only_attention=tensor_cloud01_vector_only_attention,
        )
        self.conditioner = Conditioner(num_layers=cfg.cond_layers, **common)
        self.transport = Transport(
            num_layers=cfg.transport_layers, predict_heavy=predict_heavy, **common
        )

    def _vector_source_noise_sigma(self) -> float:
        """Return the V-source sigma while preserving legacy shared-sigma behavior."""
        if self.noise_sigma_v is None:
            return self.noise_sigma
        return self.noise_sigma_v

    # ---- pieces -------------------------------------------------------------
    def encode(self, batch):
        return self.conditioner(
            batch["P_t"], batch["V_t"], batch["res_index"],
            batch["delta_ns"], batch["residue_mask"],
        )

    def interpolate(
        self, P_t, V_t, P_1, V_1, tau, generator=None,
        residue_mask=None, atom_mask=None,
    ):
        """Build the intermediate state X^tau. tau: [B]."""
        noise = torch.randn(P_t.shape, generator=generator, device=P_t.device)
        if residue_mask is not None:
            noise = noise * residue_mask.unsqueeze(-1)
        P0 = P_t + self.noise_sigma * noise
        if self.source_noise_v:
            if atom_mask is None:
                raise ValueError("source_noise_v requires atom_mask")
            v_noise = torch.randn(V_t.shape, generator=generator, device=V_t.device)
            v_noise = v_noise * atom_mask.unsqueeze(-1)
            V0 = V_t + self._vector_source_noise_sigma() * v_noise
        else:
            V0 = V_t
        aP = tau.view(-1, 1, 1)
        aV = tau.view(-1, 1, 1, 1)
        P_tau = (1 - aP) * P0 + aP * P_1
        V_tau = (1 - aV) * V0 + aV * V_1
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
            P_t = P_t + a * torch.randn_like(P_t) * batch["residue_mask"].unsqueeze(-1)
            V_t = V_t + a.unsqueeze(-1) * torch.randn_like(V_t) * batch["atom_mask"].unsqueeze(-1)
        aug_batch = {**batch, "P_t": P_t, "V_t": V_t}

        ctx = self.encode(aug_batch)
        P_tau, V_tau = self.interpolate(
            P_t, V_t, batch["P_1"], batch["V_1"], tau, generator,
            batch["residue_mask"], batch.get("atom_mask"),
        )
        P_hat_1, V_hat_1 = self.predict_x1(
            P_tau, V_tau, tau, ctx, P_t, V_t, batch["residue_mask"]
        )
        return {"P_hat_1": P_hat_1, "V_hat_1": V_hat_1, "tau": tau}

    # ---- sampling (interface for eval/rollout) ------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch,
        steps: int = 20,
        mode: str = "ode",
        generator=None,
        *,
        integrator: str = "euler",
        tau_max: float = 1.0,
        terminal_denoise: bool = False,
        drift_anchor: str = "state",
        project_v_atom_mask: bool = False,
    ):
        """Predict X_{t+delta} from X_t. Returns (P, V).

        mode="ode"  : integrate the endpoint-prediction ODE from tau=0.
        mode="mean" : one-shot x1 prediction at tau=0 (the deterministic conditional
                      mean); no integration, no noise. Stable but conservative.
        integrator="heun" uses a second-order predictor-corrector.  Setting
        tau_max<1 avoids the singular endpoint; terminal_denoise then returns one
        final endpoint prediction at tau_max.  Defaults preserve legacy sampling.
        When predict_heavy, V is produced the same way; else V stays V_t.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if integrator not in {"euler", "heun"}:
            raise ValueError("integrator must be 'euler' or 'heun'")
        if not 0 < tau_max <= 1:
            raise ValueError("tau_max must be in (0, 1]")
        if integrator == "heun" and tau_max == 1:
            raise ValueError("Heun integration requires tau_max < 1")
        if drift_anchor not in {"state", "conditioner"}:
            raise ValueError("drift_anchor must be 'state' or 'conditioner'")
        device = batch["P_t"].device
        P_t, V_t = batch["P_t"], batch["V_t"]
        mask = batch["residue_mask"]
        atom_mask = batch.get("atom_mask")
        if project_v_atom_mask:
            if atom_mask is None:
                raise ValueError("project_v_atom_mask requires atom_mask")
            V_t = V_t * atom_mask.unsqueeze(-1)
            model_batch = {**batch, "V_t": V_t}
        else:
            model_batch = batch
        ctx = self.encode(model_batch)

        def project_v(value):
            if project_v_atom_mask:
                return value * atom_mask.unsqueeze(-1)
            return value

        if mode == "mean":
            tau0 = torch.zeros(P_t.shape[0], device=device)
            P_hat_1, V_hat_1 = self.predict_x1(P_t, V_t, tau0, ctx, P_t, V_t, mask)
            V_out = V_hat_1 if (self.predict_heavy and V_hat_1 is not None) else V_t
            return P_hat_1, project_v(V_out)

        noise = torch.randn(P_t.shape, generator=generator).to(device)
        noise = noise * mask.unsqueeze(-1)
        P = P_t + self.noise_sigma * noise
        if self.source_noise_v:
            if "atom_mask" not in batch:
                raise ValueError("source_noise_v requires atom_mask")
            v_noise = torch.randn(V_t.shape, generator=generator).to(device)
            v_noise = v_noise * batch["atom_mask"].unsqueeze(-1)
            V = project_v(V_t + self._vector_source_noise_sigma() * v_noise)
        else:
            V = project_v(V_t.clone())
        taus = torch.linspace(0, tau_max, steps + 1, device=device)
        for i in range(steps):
            tau = taus[i].expand(P_t.shape[0])
            dt = (taus[i + 1] - taus[i]).item()
            P_hat_1, V_hat_1 = self.predict_x1(P, V, tau, ctx, P_t, V_t, mask)
            denom = max(1 - taus[i].item(), 1e-6)
            anchor_P = P if drift_anchor == "state" else P_t
            drift_P = (P_hat_1 - anchor_P) / denom
            drift_V = None
            if self.predict_heavy and V_hat_1 is not None:
                anchor_V = V if drift_anchor == "state" else V_t
                drift_V = (V_hat_1 - anchor_V) / denom

            P_euler = P + dt * drift_P
            V_euler = V + dt * drift_V if drift_V is not None else V
            V_euler = project_v(V_euler)
            if integrator == "heun":
                tau_next = taus[i + 1].expand(P_t.shape[0])
                P_hat_next, V_hat_next = self.predict_x1(
                    P_euler, V_euler, tau_next, ctx, P_t, V_t, mask
                )
                denom_next = max(1 - taus[i + 1].item(), 1e-6)
                next_anchor_P = P_euler if drift_anchor == "state" else P_t
                drift_P_next = (P_hat_next - next_anchor_P) / denom_next
                P = P + 0.5 * dt * (drift_P + drift_P_next)
                if drift_V is not None and V_hat_next is not None:
                    next_anchor_V = V_euler if drift_anchor == "state" else V_t
                    drift_V_next = (V_hat_next - next_anchor_V) / denom_next
                    V = project_v(V + 0.5 * dt * (drift_V + drift_V_next))
            else:
                P, V = P_euler, V_euler

        if terminal_denoise:
            tau = torch.full((P_t.shape[0],), tau_max, device=device)
            P, V_hat = self.predict_x1(P, V, tau, ctx, P_t, V_t, mask)
            if self.predict_heavy and V_hat is not None:
                V = project_v(V_hat)
        return P, project_v(V)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
