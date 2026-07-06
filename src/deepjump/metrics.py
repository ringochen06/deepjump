"""Single-step evaluation metrics (masked, batched)."""

from __future__ import annotations

import torch

from .representation import kabsch_rotation


def _masked_center(P, residue_mask):
    m = residue_mask.to(P.dtype).unsqueeze(-1)  # [B,N,1]
    centroid = (P * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
    return (P - centroid) * m


def masked_ca_rmsd(P_hat, P_gt, residue_mask):
    """Per-sample translation-free CA RMSD.

    Targets are already rotationally aligned to X_t, so orientation is fixed;
    the pairwise-vector loss is translation-invariant, so we remove the global
    translation (center both) before measuring RMSD.
    """
    P_hat = _masked_center(P_hat, residue_mask)
    P_gt = _masked_center(P_gt, residue_mask)
    sq = ((P_hat - P_gt) ** 2).sum(-1)  # [B,N]
    m = residue_mask.to(sq.dtype)
    per = (sq * m).sum(1) / m.sum(1).clamp_min(1.0)
    return per.sqrt()  # [B]


def masked_pair_distance_mae(P_hat, P_gt, residue_mask):
    """Mean absolute error of CA-CA pairwise distances."""
    d_hat = torch.cdist(P_hat, P_hat)
    d_gt = torch.cdist(P_gt, P_gt)
    pair_mask = (residue_mask[:, :, None] & residue_mask[:, None, :]).to(d_hat.dtype)
    err = (d_hat - d_gt).abs() * pair_mask
    return err.sum((1, 2)) / pair_mask.sum((1, 2)).clamp_min(1.0)  # [B]


def aligned_ca_rmsd(P_hat, P_gt):
    """Kabsch-superposed CA RMSD for a single structure (no padding). [N,3] each."""
    R = kabsch_rotation(P_hat, P_gt)
    Pc = P_hat - P_hat.mean(0, keepdim=True)
    Qc = P_gt - P_gt.mean(0, keepdim=True)
    Pa = (R @ Pc.T).T
    return ((Pa - Qc) ** 2).sum(-1).mean().sqrt()


def ca_bond_stats(P):
    """Mean/std of consecutive CA-CA distances (~3.8 A for real backbones). [N,3]."""
    d = (P[1:] - P[:-1]).norm(dim=-1)
    return d.mean(), d.std()


def contact_fraction_native(P_hat, P_gt, residue_mask, cutoff=8.0, seq_sep=3):
    """Fraction of native CA contacts (< cutoff, |i-j|>=seq_sep) recovered."""
    B, N, _ = P_hat.shape
    d_gt = torch.cdist(P_gt, P_gt)
    d_hat = torch.cdist(P_hat, P_hat)
    idx = torch.arange(N, device=P_hat.device)
    sep = (idx[:, None] - idx[None, :]).abs() >= seq_sep
    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :] & sep[None]
    native = (d_gt < cutoff) & pair_mask
    recovered = native & (d_hat < cutoff)
    out = []
    for b in range(B):
        n = native[b].sum()
        out.append((recovered[b].sum() / n) if n > 0 else torch.tensor(0.0, device=P_hat.device))
    return torch.stack(out)  # [B]
