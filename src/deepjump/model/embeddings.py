"""Scalar embeddings and distance/sequence bias features (all SE(3)-invariant)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..atom_constants import NUM_RESIDUE_TYPES


class ResidueEmbedding(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(NUM_RESIDUE_TYPES, hidden)

    def forward(self, res_index: torch.Tensor) -> torch.Tensor:  # [B,N] -> [B,N,H]
        return self.embed(res_index)


class ScalarTimeEmbedding(nn.Module):
    """Sinusoidal features of a scalar (tau in [0,1] or log-delta) -> hidden."""

    def __init__(self, hidden: int, num_freq: int = 16, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), num_freq))
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(2 * num_freq, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B] or [B,1] -> [B,H]
        x = x.reshape(-1, 1) * self.scale
        ang = x * self.freqs[None, :]
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(feats)


class DeltaEmbedding(nn.Module):
    """Embed the jump size delta (ns) via log-scale sinusoidal features."""

    def __init__(self, hidden: int):
        super().__init__()
        self.embed = ScalarTimeEmbedding(hidden, num_freq=16, scale=1.0)

    def forward(self, delta_ns: torch.Tensor) -> torch.Tensor:  # [B] -> [B,H]
        return self.embed(torch.log(delta_ns.clamp_min(1e-3)))


class SequenceDistanceBias(nn.Module):
    """Learned bias per head from residue index offset (i - j), clamped to +-ks."""

    def __init__(self, num_heads: int, ks: int = 32):
        super().__init__()
        self.ks = ks
        self.embed = nn.Embedding(2 * ks + 1, num_heads)

    def forward(self, n: int, device) -> torch.Tensor:  # -> [num_heads, N, N]
        idx = torch.arange(n, device=device)
        offset = (idx[:, None] - idx[None, :]).clamp(-self.ks, self.ks) + self.ks
        return self.embed(offset).permute(2, 0, 1)  # [H, N, N]


class GaussianDistanceBias(nn.Module):
    """Gaussian radial basis of pairwise CA distance -> per-head bias."""

    def __init__(self, num_heads: int, num_basis: int = 16, cutoff: float = 25.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers, persistent=False)
        self.width = cutoff / num_basis
        self.proj = nn.Linear(num_basis, num_heads)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:  # [B,N,N] -> [B,H,N,N]
        rbf = torch.exp(-((dist[..., None] - self.centers) ** 2) / (2 * self.width**2))
        return self.proj(rbf).permute(0, 3, 1, 2)
