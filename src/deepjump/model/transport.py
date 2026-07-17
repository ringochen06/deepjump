"""Stage 2: transport network. Predicts x1 = X_{t+delta} from the interpolant.

Consumes the intermediate state (P^tau, V^tau), the latent time tau, and the
conditioner context H_t. Outputs, as residuals from the current frame:
    P_hat_1 = P_t + dP      per-residue CA displacement
    V_hat_1 = V_t + dV      per-residue heavy-atom offset update  (if predict_heavy)
Both dP and dV are l=1 vector outputs, so with P_t/V_t also l=1 the predictions
are SE(3)-equivariant. Residual parameterisation suits the small aligned jump.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..atom_constants import MAX_HEAVY
from .embeddings import ScalarTimeEmbedding
from .layers import EquivBlock, EquivLinear
from .tensor_cloud01 import TensorCloud01Block


class Transport(nn.Module):
    def __init__(
        self,
        hidden: int,
        vec_channels: int,
        num_heads: int,
        num_layers: int,
        seq_ks: int,
        num_dist_basis: int,
        dist_cutoff: float,
        predict_heavy: bool = False,
        vector_qk: bool = False,
        tensor_qkv: bool = False,
        paper_ff: bool = False,
        tensor_cloud01: bool = False,
    ):
        super().__init__()
        self.predict_heavy = predict_heavy
        self.tau_embed = ScalarTimeEmbedding(hidden)
        self.ctx_proj = nn.Linear(hidden, hidden)
        self.vec_in = EquivLinear(MAX_HEAVY, vec_channels)
        self.vec_ctx_proj = EquivLinear(vec_channels, vec_channels)
        if tensor_cloud01:
            if vec_channels != hidden:
                raise ValueError("tensor_cloud01 requires vector_channels == hidden")
            self.blocks = nn.ModuleList(
                TensorCloud01Block(
                    hidden, num_heads, seq_ks, num_dist_basis, dist_cutoff
                )
                for _ in range(num_layers)
            )
        else:
            self.blocks = nn.ModuleList(
                EquivBlock(
                    hidden, vec_channels, num_heads, seq_ks, num_dist_basis, dist_cutoff,
                    vector_qk=vector_qk,
                    tensor_qkv=tensor_qkv,
                    paper_ff=paper_ff,
                )
                for _ in range(num_layers)
            )
        self.head = EquivLinear(vec_channels, 1)  # -> per-residue CA displacement
        if predict_heavy:
            self.head_v = EquivLinear(vec_channels, MAX_HEAVY)  # -> offset update dV

    def forward(self, P_tau, V_tau, tau, s_ctx, vec_ctx, P_t, V_t, residue_mask):
        s = self.tau_embed(tau)[:, None, :] + self.ctx_proj(s_ctx)
        vec = self.vec_in(V_tau) + self.vec_ctx_proj(vec_ctx)
        for blk in self.blocks:
            s, vec = blk(s, vec, P_tau, residue_mask)
        P_hat_1 = P_t + self.head(vec).squeeze(-2)  # [B,N,3]
        V_hat_1 = (V_t + self.head_v(vec)) if self.predict_heavy else None
        return P_hat_1, V_hat_1
