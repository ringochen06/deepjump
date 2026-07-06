"""SE(3)-equivariant building blocks (GVP / EGNN style, no e3nn tensor products).

State = (s, vec):
    s   : [B, N, H]        scalar (l=0) features   -- SO(3)-INVARIANT
    vec : [B, N, C, 3]     vector (l=1) features   -- SO(3)-EQUIVARIANT

Equivariance recipe used throughout:
  * scalars derived from vectors only via norms (invariants);
  * vectors only ever scaled/mixed by invariant scalars or channel-linear maps
    (no bias on the xyz axis), and combined with other equivariant vectors.
Under a global rotation R (vec -> vec @ R^T, positions P -> P @ R^T), every op
below commutes with R, and every scalar is unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .embeddings import GaussianDistanceBias, SequenceDistanceBias

EPS = 1e-8


class EquivLinear(nn.Module):
    """Channel-mixing linear over vector features (no bias): [B,N,Cin,3]->[B,N,Cout,3]."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...cd,oc->...od", vec, self.weight)


class ScalarVectorLayerNorm(nn.Module):
    """LayerNorm on scalars; RMS(norm)-normalization on vectors (both equivariant)."""

    def __init__(self, hidden: int, vec_channels: int):
        super().__init__()
        self.scalar_norm = nn.LayerNorm(hidden)
        self.vec_gamma = nn.Parameter(torch.ones(vec_channels))

    def forward(self, s, vec):
        s = self.scalar_norm(s)
        vnorm = vec.norm(dim=-1)  # [B,N,C]
        rms = vnorm.pow(2).mean(dim=-1, keepdim=True).clamp_min(EPS).sqrt()  # [B,N,1]
        vec = vec / rms.unsqueeze(-1) * self.vec_gamma[None, None, :, None]
        return s, vec


class GVP(nn.Module):
    """Geometric Vector Perceptron: mixes scalars and vectors equivariantly.

    Vector norms feed the scalar update; a scalar-derived gate modulates the
    vector output. This is the feed-forward (Algorithm 2 / GVP) block.
    """

    def __init__(self, hidden: int, vec_in: int, vec_out: int, mid: int | None = None):
        super().__init__()
        mid = mid or max(vec_in, vec_out)
        self.wh = EquivLinear(vec_in, mid)  # hidden vector projection (for norms)
        self.wu = EquivLinear(mid, vec_out)  # output vectors (to be gated)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden + mid, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.gate = nn.Linear(hidden, vec_out)

    def forward(self, s, vec):
        vh = self.wh(vec)  # [B,N,mid,3]
        vh_norm = vh.norm(dim=-1)  # [B,N,mid] invariant
        s_new = self.scalar_mlp(torch.cat([s, vh_norm], dim=-1))  # [B,N,H]
        gate = torch.sigmoid(self.gate(s_new))  # [B,N,vec_out] invariant gate
        vec_new = self.wu(vh) * gate.unsqueeze(-1)  # equivariant
        return s_new, vec_new


class EquivAttention(nn.Module):
    """Equivariant multi-head self-attention (Algorithm 1).

    Attention logits are invariant (q.k + sequence bias + gaussian distance bias),
    so softmax weights are invariant; they aggregate invariant scalar values and
    equivariant vector values (including relative-direction unit vectors).
    """

    def __init__(
        self,
        hidden: int,
        vec_channels: int,
        num_heads: int,
        seq_ks: int = 32,
        num_dist_basis: int = 16,
        dist_cutoff: float = 25.0,
    ):
        super().__init__()
        assert hidden % num_heads == 0 and vec_channels % num_heads == 0
        self.h = num_heads
        self.dh = hidden // num_heads  # scalar head dim
        self.cvh = vec_channels // num_heads  # vector head dim

        self.to_q = nn.Linear(hidden, hidden)
        self.to_k = nn.Linear(hidden, hidden)
        self.to_sv = nn.Linear(hidden, hidden)  # scalar values
        self.to_vv = EquivLinear(vec_channels, vec_channels)  # vector values
        self.seq_bias = SequenceDistanceBias(num_heads, ks=seq_ks)
        self.dist_bias = GaussianDistanceBias(num_heads, num_dist_basis, dist_cutoff)

        self.out_s = nn.Linear(hidden, hidden)
        # vector output: per head cvh value channels + 1 relative-direction channel
        self.out_vec = EquivLinear(num_heads * (self.cvh + 1), vec_channels)

    def forward(self, s, vec, P, residue_mask):
        B, N, H = s.shape
        q = self.to_q(s).view(B, N, self.h, self.dh)
        k = self.to_k(s).view(B, N, self.h, self.dh)
        sv = self.to_sv(s).view(B, N, self.h, self.dh)
        vv = self.to_vv(vec).view(B, N, self.h, self.cvh, 3)

        logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dh)
        logits = logits + self.seq_bias(N, s.device)[None]

        dvec = P[:, None, :, :] - P[:, :, None, :]  # [B,N,N,3] = P_j - P_i
        dist = dvec.norm(dim=-1)  # [B,N,N]
        logits = logits + self.dist_bias(dist)

        if residue_mask is not None:
            key_mask = ~residue_mask[:, None, None, :]  # [B,1,1,N]
            logits = logits.masked_fill(key_mask, float("-inf"))
        attn = torch.softmax(logits, dim=-1)  # [B,H,N,N]
        attn = torch.nan_to_num(attn)  # fully-masked rows -> 0

        # scalar output
        out_s = torch.einsum("bhij,bjhd->bihd", attn, sv).reshape(B, N, H)
        out_s = self.out_s(out_s)

        # vector value aggregation
        out_vv = torch.einsum("bhij,bjhcx->bihcx", attn, vv)  # [B,N,H,cvh,3]
        direction = dvec / dist.clamp_min(EPS).unsqueeze(-1)  # [B,N,N,3] equivariant
        out_dir = torch.einsum("bhij,bijx->bihx", attn, direction)  # [B,N,H,3]
        out_vec = torch.cat(
            [out_vv.reshape(B, N, self.h * self.cvh, 3), out_dir], dim=2
        )  # [B,N,H*cvh + H,3]
        out_vec = self.out_vec(out_vec)
        return out_s, out_vec


class EquivBlock(nn.Module):
    """Pre-norm attention + GVP feed-forward, each with residual connections."""

    def __init__(self, hidden, vec_channels, num_heads, seq_ks, num_dist_basis, dist_cutoff):
        super().__init__()
        self.norm1 = ScalarVectorLayerNorm(hidden, vec_channels)
        self.attn = EquivAttention(
            hidden, vec_channels, num_heads, seq_ks, num_dist_basis, dist_cutoff
        )
        self.norm2 = ScalarVectorLayerNorm(hidden, vec_channels)
        self.ff = GVP(hidden, vec_channels, vec_channels)

    def forward(self, s, vec, P, residue_mask):
        sn, vn = self.norm1(s, vec)
        das, dav = self.attn(sn, vn, P, residue_mask)
        s = s + das
        vec = vec + dav
        sn, vn = self.norm2(s, vec)
        ds, dv = self.ff(sn, vn)
        s = s + ds
        vec = vec + dv
        return s, vec
