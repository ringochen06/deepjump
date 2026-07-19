"""Evidence-bounded l=0/l=1 Tensor Cloud implementation for DeepJump."""

from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import GaussianDistanceBias, SequenceDistanceBias
from .layers import EquivLinear, PaperFeedForward, ScalarVectorLayerNorm

EPS = 1e-8


class TensorCloud01Attention(nn.Module):
    """Literal Algorithm-1 attention over equal-multiplicity scalar/vector irreps.

    The published pseudocode does not specify dot-product scaling, so this path
    intentionally applies none. Pair values explicitly append Y0=1 and the unit
    l=1 direction Y1(P_i-P_j). The exact spherical-harmonic normalization is not
    disclosed; unit normalization is therefore an explicit implementation choice.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        seq_ks: int = 32,
        num_dist_basis: int = 16,
        dist_cutoff: float = 25.0,
        vector_only: bool = False,
    ):
        super().__init__()
        if hidden % num_heads:
            raise ValueError("hidden must be divisible by num_heads")
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.vector_only = vector_only
        if not vector_only:
            self.to_qkv_scalar = nn.Linear(hidden, 3 * hidden)
        self.to_qkv_vector = EquivLinear(hidden, 3 * hidden)
        self.sequence_bias = SequenceDistanceBias(num_heads, ks=seq_ks)
        self.distance_bias = GaussianDistanceBias(
            num_heads, num_basis=num_dist_basis, cutoff=dist_cutoff
        )
        pair_channels = num_heads * (self.head_dim + 1)
        if not vector_only:
            self.out_scalar = nn.Linear(pair_channels, hidden)
        self.out_vector = EquivLinear(pair_channels, hidden)

    def _project_qkv(self, scalar: torch.Tensor, vector: torch.Tensor):
        batch, residues, _ = scalar.shape
        vector_qkv = self.to_qkv_vector(vector).view(
            batch, residues, 3, self.num_heads, self.head_dim, 3
        )
        vector_parts = vector_qkv.unbind(dim=2)
        if self.vector_only:
            return (None, None, None, *vector_parts)
        scalar_qkv = self.to_qkv_scalar(scalar).view(
            batch, residues, 3, self.num_heads, self.head_dim
        )
        return (*scalar_qkv.unbind(dim=2), *vector_parts)

    def _content_logits(self, scalar: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        scalar_q, scalar_k, _, vector_q, vector_k, _ = self._project_qkv(scalar, vector)
        # Published indexing is k_i dot q_j (not the legacy q_i dot k_j path).
        vector_logits = torch.einsum(
            "bihcx,bjhcx->bhij", vector_k.float(), vector_q.float()
        )
        if self.vector_only:
            return vector_logits.to(scalar.dtype)
        scalar_logits = torch.einsum(
            "bihd,bjhd->bhij", scalar_k.float(), scalar_q.float()
        )
        return (scalar_logits + vector_logits).to(scalar.dtype)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        positions: torch.Tensor,
        residue_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, residues, _ = scalar.shape
        _, _, scalar_value, _, _, vector_value = self._project_qkv(scalar, vector)
        logits = self._content_logits(scalar, vector)
        logits = logits + self.sequence_bias(residues, scalar.device)[None]

        # [B,i,j,3] follows the literal published Y(P_i-P_j) sign convention.
        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distance = displacement.norm(dim=-1)
        logits = logits + self.distance_bias(distance)
        if residue_mask is not None:
            logits = logits.masked_fill(
                ~residue_mask[:, None, None, :], float("-inf")
            )
        attention = torch.nan_to_num(torch.softmax(logits, dim=-1))

        y1 = displacement / distance.clamp_min(EPS).unsqueeze(-1)
        y1 = y1[:, :, :, None, None, :].expand(
            -1, -1, -1, self.num_heads, -1, -1
        )
        vector_pair = torch.cat(
            [
                vector_value[:, None].expand(-1, residues, -1, -1, -1, -1),
                y1,
            ],
            dim=-2,
        )

        vector_aggregate = torch.einsum(
            "bhij,bijhcx->bihcx", attention, vector_pair
        ).reshape(batch, residues, -1, 3)
        if self.vector_only:
            scalar_out = torch.zeros_like(scalar)
        else:
            y0 = torch.ones(
                batch, residues, residues, self.num_heads, 1,
                dtype=scalar.dtype, device=scalar.device,
            )
            scalar_pair = torch.cat(
                [
                    scalar_value[:, None].expand(-1, residues, -1, -1, -1),
                    y0,
                ],
                dim=-1,
            )
            scalar_aggregate = torch.einsum(
                "bhij,bijhc->bihc", attention, scalar_pair
            ).reshape(batch, residues, -1)
            scalar_out = self.out_scalar(scalar_aggregate)
        vector_out = self.out_vector(vector_aggregate)
        if residue_mask is not None:
            scalar_out = scalar_out * residue_mask.unsqueeze(-1)
            vector_out = vector_out * residue_mask.unsqueeze(-1).unsqueeze(-1)
        return scalar_out, vector_out


class TensorCloud01FeedForward(PaperFeedForward):
    """Literal l=0/l=1 Algorithm-2 branches with multiplicity H and F=2."""

    def __init__(self, hidden: int, factor: int = 2):
        super().__init__(hidden=hidden, vec_channels=hidden, factor=factor)


class TensorCloud01VectorAttentionNorm(nn.Module):
    """Normalize only the vector stream consumed by vector-only attention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.vec_gamma = nn.Parameter(torch.ones(hidden))

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor):
        vector_norm = vector.norm(dim=-1)
        rms = vector_norm.square().mean(dim=-1, keepdim=True).clamp_min(EPS).sqrt()
        normalized = vector / rms.unsqueeze(-1)
        normalized = normalized * self.vec_gamma[None, None, :, None]
        return scalar, normalized


class TensorCloud01Block(nn.Module):
    """Pre-norm residual TensorCloud01 block.

    Residual and equivariant LayerNorm are stated in the paper; their ordering is
    not disclosed. Pre-norm is retained as an explicit stability choice.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        seq_ks: int,
        num_dist_basis: int,
        dist_cutoff: float,
        vector_only_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = (
            TensorCloud01VectorAttentionNorm(hidden)
            if vector_only_attention
            else ScalarVectorLayerNorm(hidden, hidden)
        )
        self.attention = TensorCloud01Attention(
            hidden, num_heads, seq_ks, num_dist_basis, dist_cutoff,
            vector_only=vector_only_attention,
        )
        self.norm2 = ScalarVectorLayerNorm(hidden, hidden)
        self.feedforward = TensorCloud01FeedForward(hidden, factor=2)

    def forward(self, scalar, vector, positions, residue_mask):
        scalar_norm, vector_norm = self.norm1(scalar, vector)
        delta_scalar, delta_vector = self.attention(
            scalar_norm, vector_norm, positions, residue_mask
        )
        scalar = scalar + delta_scalar
        vector = vector + delta_vector
        scalar_norm, vector_norm = self.norm2(scalar, vector)
        delta_scalar, delta_vector = self.feedforward(scalar_norm, vector_norm)
        scalar = scalar + delta_scalar
        vector = vector + delta_vector
        if residue_mask is not None:
            scalar = scalar * residue_mask.unsqueeze(-1)
            vector = vector * residue_mask.unsqueeze(-1).unsqueeze(-1)
        return scalar, vector
