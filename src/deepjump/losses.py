"""Pairwise vector Huber loss (DeepJump / Ophiuchus Vector-Map loss).

Loss on 3D DIFFERENCE VECTORS between residue pairs (not scalar distances):
    V(P)^{ij} = P^i - P^j  in R^3,   L = Huber( V(P_hat), V(P_1) )
This is equivariant (rotates with the structure) and translation-invariant.
Lite stage: all CA pairs (no 25 A cutoff). Padding excluded via residue_mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_vectors(P: torch.Tensor) -> torch.Tensor:
    """P [B,N,3] -> difference vectors [B,N,N,3] where out[i,j] = P_i - P_j."""
    return P[:, :, None, :] - P[:, None, :, :]


def pairwise_vector_huber_loss(
    P_hat: torch.Tensor,
    P_gt: torch.Tensor,
    residue_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    diff_pred = pairwise_vectors(P_hat)
    diff_gt = pairwise_vectors(P_gt)
    per = F.huber_loss(diff_pred, diff_gt, delta=delta, reduction="none").sum(-1)  # [B,N,N]

    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]  # [B,N,N]
    pair_mask = pair_mask.to(per.dtype)
    denom = pair_mask.sum().clamp_min(1.0)
    return (per * pair_mask).sum() / denom


def ca_bond_length_huber_loss(
    P_hat: torch.Tensor,
    P_gt: torch.Tensor,
    residue_mask: torch.Tensor,
    bond_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss on CA distances for true consecutive residues only."""
    pred = (P_hat[:, 1:] - P_hat[:, :-1]).norm(dim=-1)
    target = (P_gt[:, 1:] - P_gt[:, :-1]).norm(dim=-1)
    per = F.huber_loss(pred, target, delta=delta, reduction="none")
    if bond_mask.shape != pred.shape:
        raise ValueError(f"bond_mask shape {tuple(bond_mask.shape)} != {tuple(pred.shape)}")
    valid = (residue_mask[:, 1:] & residue_mask[:, :-1] & bond_mask).to(per.dtype)
    return (per * valid).sum() / valid.sum().clamp_min(1.0)


def ca_local_geometry_huber_losses(
    P_hat: torch.Tensor,
    P_gt: torch.Tensor,
    residue_mask: torch.Tensor,
    bond_mask: torch.Tensor,
    delta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Topology-aware local CA geometry losses.

    The length term acts on relative bond-length error so its scale is not tied
    to Angstrom units.  The angle term compares adjacent-bond cosines rather
    than angles themselves, avoiding unstable ``acos`` gradients near 0/pi.
    """
    pred_bond = P_hat[:, 1:] - P_hat[:, :-1]
    target_bond = P_gt[:, 1:] - P_gt[:, :-1]
    pred_len = pred_bond.norm(dim=-1)
    target_len = target_bond.norm(dim=-1)
    if bond_mask.shape != pred_len.shape:
        raise ValueError(f"bond_mask shape {tuple(bond_mask.shape)} != {tuple(pred_len.shape)}")

    valid_bond = residue_mask[:, 1:] & residue_mask[:, :-1] & bond_mask
    relative_error = (pred_len - target_len) / target_len.clamp_min(1e-6)
    length_per = F.huber_loss(
        relative_error, torch.zeros_like(relative_error), delta=delta, reduction="none"
    )
    length_valid = valid_bond.to(length_per.dtype)
    length_loss = (length_per * length_valid).sum() / length_valid.sum().clamp_min(1.0)

    pred_unit = pred_bond / pred_len.clamp_min(1e-6).unsqueeze(-1)
    target_unit = target_bond / target_len.clamp_min(1e-6).unsqueeze(-1)
    pred_cos = (pred_unit[:, :-1] * pred_unit[:, 1:]).sum(dim=-1)
    target_cos = (target_unit[:, :-1] * target_unit[:, 1:]).sum(dim=-1)
    angle_per = F.huber_loss(pred_cos, target_cos, delta=delta, reduction="none")
    valid_angle = (valid_bond[:, :-1] & valid_bond[:, 1:]).to(angle_per.dtype)
    angle_loss = (angle_per * valid_angle).sum() / valid_angle.sum().clamp_min(1.0)
    return length_loss, angle_loss


def allatom_coords(P, V, atom_mask):
    """Reconstruct absolute all-atom coords from (P, V). -> coords [B,N,14,3], valid [B,N,14].

    Slot 0 is the CA (always valid); slots 1..13 are heavy atoms (P + offset), valid per mask.
    """
    B, N = P.shape[:2]
    heavy = P[:, :, None, :] + V  # [B,N,13,3]
    coords = torch.cat([P[:, :, None, :], heavy], dim=2)  # [B,N,14,3]
    ca_valid = torch.ones(B, N, 1, dtype=torch.bool, device=P.device)
    valid = torch.cat([ca_valid, atom_mask], dim=2)  # [B,N,14]
    return coords, valid


def allatom_pairwise_huber_loss(
    P_hat, V_hat, P_gt, V_gt, atom_mask, residue_mask, cutoff=25.0, delta=1.0
):
    """DeepJump all-atom Vector-Map loss: Huber on 3D difference vectors between every
    atom pair within `cutoff` (defined by the ground-truth structure), padding/atom-masked.
    """
    A_hat, valid = allatom_coords(P_hat, V_hat, atom_mask)
    A_gt, _ = allatom_coords(P_gt, V_gt, atom_mask)
    B, N = P_hat.shape[:2]
    M = N * 14
    A_hat = A_hat.reshape(B, M, 3)
    A_gt = A_gt.reshape(B, M, 3)
    atom_valid = (valid & residue_mask[:, :, None]).reshape(B, M)  # [B,M]

    diff_hat = A_hat[:, :, None, :] - A_hat[:, None, :, :]  # [B,M,M,3]
    diff_gt = A_gt[:, :, None, :] - A_gt[:, None, :, :]
    per = F.huber_loss(diff_hat, diff_gt, delta=delta, reduction="none").sum(-1)  # [B,M,M]

    dist_gt = diff_gt.norm(dim=-1)  # [B,M,M]
    pair_mask = (atom_valid[:, :, None] & atom_valid[:, None, :]) & (dist_gt < cutoff)
    pair_mask = pair_mask.to(per.dtype)
    return (per * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def heavy_atom_offset_loss(
    V_hat: torch.Tensor,
    V_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss on heavy-atom offset vectors V (B,N,13,3), masked to real atoms.

    Offsets are already CA-relative (translation-invariant) and rotate with the
    structure, so this is a directly-supervised equivariant target.
    """
    per = F.huber_loss(V_hat, V_gt, delta=delta, reduction="none").sum(-1)  # [B,N,13]
    m = atom_mask.to(per.dtype)
    return (per * m).sum() / m.sum().clamp_min(1.0)
