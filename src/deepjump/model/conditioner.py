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
from .tensor_cloud01 import TensorCloud01Block


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
        vector_qk: bool = False,
        tensor_qkv: bool = False,
        paper_ff: bool = False,
        tensor_cloud01: bool = False,
    ):
        super().__init__()
        self.res_embed = ResidueEmbedding(hidden)
        self.delta_embed = DeltaEmbedding(hidden)
        self.vec_in = EquivLinear(MAX_HEAVY, vec_channels)
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

    def forward(self, P_t, V_t, res_index, delta_ns, residue_mask):
        s = self.res_embed(res_index) + self.delta_embed(delta_ns)[:, None, :]
        vec = self.vec_in(V_t)  # [B,N,Cv,3]
        for blk in self.blocks:
            s, vec = blk(s, vec, P_t, residue_mask)
        return s, vec  # H_t: (scalar_ctx, vector_ctx)
