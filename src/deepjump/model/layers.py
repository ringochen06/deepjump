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
        # matmul(weight[c_out,c_in], vec[...,c_in,3]) -> [...,c_out,3]. Equivalent to
        # einsum("...cd,oc->...od") but yields CONTIGUOUS weight grads, which DDP's bucket
        # reducer requires to all-reduce correctly (einsum grads are non-contiguous and get
        # silently dropped from sync -- see tests/test_ddp_sync.py).
        return torch.matmul(self.weight, vec)


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


class PaperFeedForward(nn.Module):
    """Algorithm-2-style scalar/vector feed-forward with expansion F=2.

    It keeps the repository's separate scalar ``hidden`` and vector-channel
    widths while following the paper's two projected branches, multiplicity-wise
    ``tanh(norm^2)`` activation, and scalar/vector cross-gating.
    """

    def __init__(self, hidden: int, vec_channels: int, factor: int = 2):
        super().__init__()
        expanded = factor * hidden
        self.scalar_in = nn.Linear(hidden, 2 * expanded)
        self.vector_in = EquivLinear(vec_channels, 2 * expanded)
        self.scalar_out = nn.Linear(3 * expanded, hidden)
        self.vector_out = EquivLinear(expanded, vec_channels)

    def forward(self, s, vec):
        scalar_q, scalar_cross = self.scalar_in(s).chunk(2, dim=-1)
        vector_q, vector_cross = self.vector_in(vec).chunk(2, dim=-2)
        # Norm-squared nonlinearities are evaluated in fp32 under AMP, then
        # cast back before the output projections.
        scalar_q_act = torch.tanh(scalar_q.float().square()).to(s.dtype)
        vector_q_act = torch.tanh(vector_q.float().square().sum(-1)).to(s.dtype)
        vector_gate = torch.tanh(vector_cross.float().square().sum(-1)).to(s.dtype)
        scalar_gate = torch.tanh(scalar_cross.float().square()).to(s.dtype)
        cross_scalar = vector_gate * scalar_cross
        cross_vector = scalar_gate.unsqueeze(-1) * vector_cross
        scalar_out = self.scalar_out(
            torch.cat([scalar_q_act, vector_q_act, cross_scalar], dim=-1)
        )
        vector_out = self.vector_out(cross_vector)
        return scalar_out, vector_out


class EquivAttention(nn.Module):
    """Equivariant multi-head self-attention (Algorithm 1).

    Attention logits are invariant (q.k + sequence bias + gaussian distance bias),
    so softmax weights are invariant; they aggregate invariant scalar values and
    equivariant vector values (including relative-direction unit vectors).  The
    ``tensor_qkv`` path follows Algorithm 1 by taking one joint inner product over
    the scalar and vector parts of the Tensor Cloud and normalizing by their total
    real dimension.  ``vector_qk`` preserves the earlier gated approximation.
    """

    def __init__(
        self,
        hidden: int,
        vec_channels: int,
        num_heads: int,
        seq_ks: int = 32,
        num_dist_basis: int = 16,
        dist_cutoff: float = 25.0,
        vector_qk: bool = False,
        tensor_qkv: bool = False,
    ):
        super().__init__()
        assert hidden % num_heads == 0 and vec_channels % num_heads == 0
        if vector_qk and tensor_qkv:
            raise ValueError("vector_qk and tensor_qkv are mutually exclusive")
        self.h = num_heads
        self.dh = hidden // num_heads  # scalar head dim
        self.cvh = vec_channels // num_heads  # vector head dim

        self.to_q = nn.Linear(hidden, hidden)
        self.to_k = nn.Linear(hidden, hidden)
        self.vector_qk = vector_qk
        self.tensor_qkv = tensor_qkv
        if vector_qk or tensor_qkv:
            self.to_qv = EquivLinear(vec_channels, vec_channels)
            self.to_kv = EquivLinear(vec_channels, vec_channels)
        if vector_qk:
            # Zero gate makes a warm-started model exactly reproduce the legacy
            # scalar-q/k logits before learning to use tensor-cloud q/k.
            self.vector_qk_gate = nn.Parameter(torch.zeros(num_heads))
        self.to_sv = nn.Linear(hidden, hidden)  # scalar values
        self.to_vv = EquivLinear(vec_channels, vec_channels)  # vector values
        self.seq_bias = SequenceDistanceBias(num_heads, ks=seq_ks)
        self.dist_bias = GaussianDistanceBias(num_heads, num_dist_basis, dist_cutoff)

        self.out_s = nn.Linear(hidden, hidden)
        # vector output: per head cvh value channels + 1 relative-direction channel
        self.out_vec = EquivLinear(num_heads * (self.cvh + 1), vec_channels)

    def _content_logits(self, s, vec):
        """Invariant q/k content logits before sequence and distance biases."""
        B, N, _ = s.shape
        q = self.to_q(s).view(B, N, self.h, self.dh)
        k = self.to_k(s).view(B, N, self.h, self.dh)
        scalar_logits = torch.einsum("bihd,bjhd->bhij", q, k)
        if self.tensor_qkv:
            qv = self.to_qv(vec).view(B, N, self.h, self.cvh, 3)
            kv = self.to_kv(vec).view(B, N, self.h, self.cvh, 3)
            vector_logits = torch.einsum("bihcx,bjhcx->bhij", qv.float(), kv.float())
            # Each vector channel contributes three real coordinates to the
            # Tensor-Cloud inner product; compute the reduction in fp32 under AMP.
            logits = (scalar_logits.float() + vector_logits) / math.sqrt(
                self.dh + 3 * self.cvh
            )
            logits = logits.to(s.dtype)
        else:
            logits = scalar_logits / math.sqrt(self.dh)
        if self.vector_qk:
            qv = self.to_qv(vec).view(B, N, self.h, self.cvh, 3)
            kv = self.to_kv(vec).view(B, N, self.h, self.cvh, 3)
            vector_logits = torch.einsum("bihcx,bjhcx->bhij", qv.float(), kv.float())
            vector_logits = vector_logits / math.sqrt(3 * self.cvh)
            logits = logits + torch.tanh(self.vector_qk_gate).view(1, self.h, 1, 1) * vector_logits.to(logits.dtype)
        return logits

    def forward(self, s, vec, P, residue_mask):
        B, N, H = s.shape
        sv = self.to_sv(s).view(B, N, self.h, self.dh)
        vv = self.to_vv(vec).view(B, N, self.h, self.cvh, 3)

        logits = self._content_logits(s, vec)
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

    def __init__(
        self, hidden, vec_channels, num_heads, seq_ks, num_dist_basis, dist_cutoff,
        vector_qk=False,
        tensor_qkv=False,
        paper_ff=False,
    ):
        super().__init__()
        self.norm1 = ScalarVectorLayerNorm(hidden, vec_channels)
        self.attn = EquivAttention(
            hidden, vec_channels, num_heads, seq_ks, num_dist_basis, dist_cutoff,
            vector_qk=vector_qk,
            tensor_qkv=tensor_qkv,
        )
        self.norm2 = ScalarVectorLayerNorm(hidden, vec_channels)
        self.ff = (PaperFeedForward(hidden, vec_channels)
                   if paper_ff else GVP(hidden, vec_channels, vec_channels))

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
