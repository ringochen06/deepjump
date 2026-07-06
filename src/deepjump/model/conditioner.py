"""Stage 1: conditioning encoder. Encodes (X_t, sequence, delta) -> H_t.

H_t (scalar + vector context) does not depend on the latent time tau, so it is
computed once and reused across all ODE steps in Stage 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..atom_constants import MAX_HEAVY
from .embeddings import DeltaEmbedding, ResidueEmbedding
from .layers import EquivBlock, EquivLinear


class Conditioner(nn.Module):
    def __init__(
        self,
        hidden: int,
        vec_channels: int,
        num_heads: int,
        num_layers: int,
        seq_ks: int,
        num_dist_basis: int,
        dist_cutoff: float,
    ):
        super().__init__()
        self.res_embed = ResidueEmbedding(hidden)
        self.delta_embed = DeltaEmbedding(hidden)
        self.vec_in = EquivLinear(MAX_HEAVY, vec_channels)
        self.blocks = nn.ModuleList(
            EquivBlock(hidden, vec_channels, num_heads, seq_ks, num_dist_basis, dist_cutoff)
            for _ in range(num_layers)
        )

    def forward(self, P_t, V_t, res_index, delta_ns, residue_mask):
        s = self.res_embed(res_index) + self.delta_embed(delta_ns)[:, None, :]
        vec = self.vec_in(V_t)  # [B,N,Cv,3]
        for blk in self.blocks:
            s, vec = blk(s, vec, P_t, residue_mask)
        return s, vec  # H_t: (scalar_ctx, vector_ctx)
