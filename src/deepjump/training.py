"""Loss composition and LR schedule shared by the single-GPU and DDP trainers."""

from __future__ import annotations

import math

import torch

from .losses import (
    allatom_pairwise_huber_loss,
    ca_bond_length_huber_loss,
    ca_local_geometry_huber_losses,
    heavy_atom_offset_loss,
    pairwise_vector_huber_loss,
)


def _step_loss(P_hat, V_hat, P_gt, V_gt, batch, cfg, bond_weight=None):
    # Ca pairwise term, weighted by w_ca (default 1.0 = legacy behaviour). Set w_ca=0 to train
    # on the all-atom Vector-Map loss alone (CA positions are already inside the all-atom set).
    ca = pairwise_vector_huber_loss(P_hat, P_gt, batch["residue_mask"], cfg.train.huber_delta)
    loss = getattr(cfg.train, "w_ca", 1.0) * ca
    bond_val = None
    if bond_weight is None:
        bond_weight = getattr(cfg.train, "w_bond", 0.0)
    if bond_weight > 0:
        bond = ca_bond_length_huber_loss(
            P_hat, P_gt, batch["residue_mask"], batch["bond_mask"], cfg.train.huber_delta
        )
        loss = loss + bond_weight * bond
        bond_val = bond.item()
    off_val = None
    if cfg.train.w_offset > 0 and V_hat is not None:
        off = heavy_atom_offset_loss(V_hat, V_gt, batch["atom_mask"], cfg.train.huber_delta)
        loss = loss + cfg.train.w_offset * off  # accumulate; do NOT reset (would drop w_ca weight)
        off_val = off.item()
    aa_val = None
    if cfg.train.w_allatom > 0 and V_hat is not None:
        aa = allatom_pairwise_huber_loss(
            P_hat, V_hat, P_gt, V_gt, batch["atom_mask"], batch["residue_mask"],
            cutoff=cfg.model.dist_cutoff, delta=cfg.train.huber_delta,
        )
        loss = loss + cfg.train.w_allatom * aa
        aa_val = aa.item()
    return loss, ca.item(), off_val, bond_val, aa_val


def total_loss(out, batch, cfg, model=None):
    """Step-1 loss + optional self-conditioned step-2..K losses (k-step unroll).

    `model` may be a DDP-wrapped module; its forward signature is unchanged.
    """
    loss1, ca1, off1, bond1, aa1 = _step_loss(
        out["P_hat_1"], out.get("V_hat_1"), batch["P_1"], batch["V_1"], batch, cfg
    )
    loss, comps = loss1, {"ca": ca1}
    if off1 is not None:
        comps["offset"] = off1
    if bond1 is not None:
        comps["bond"] = bond1
    if aa1 is not None:
        comps["allatom"] = aa1
    if cfg.train.w_unroll > 0 and model is not None and out.get("V_hat_1") is not None:
        # Build the feedback chain from honest tau=0 predictions.  Reusing the
        # primary random-tau output here leaks part of the true endpoint through
        # X^tau and does not represent inference-time self-conditioning.
        tau0 = torch.zeros(batch["P_t"].shape[0], device=batch["P_t"].device)
        with torch.no_grad():
            feedback = model(batch, tau=tau0)
        P_prev = feedback["P_hat_1"].detach()
        V_prev = feedback["V_hat_1"].detach()
        k = 2
        while f"P_{k}" in batch:
            batch_k = {**batch, "P_t": P_prev, "V_t": V_prev,
                       "P_1": batch[f"P_{k}"], "V_1": batch[f"V_{k}"]}
            out_k = model(batch_k, tau=tau0)
            unroll_bond_weight = getattr(cfg.train, "w_bond_unroll", 0.0) or None
            loss_k, ca_k, _, bond_k, aa_k = _step_loss(
                out_k["P_hat_1"], out_k.get("V_hat_1"),
                batch[f"P_{k}"], batch[f"V_{k}"], batch, cfg,
                bond_weight=unroll_bond_weight,
            )
            loss = loss + cfg.train.w_unroll * loss_k
            comps[f"ca{k}"] = ca_k
            if bond_k is not None:
                comps[f"bond{k}"] = bond_k
            if aa_k is not None:
                comps[f"allatom{k}"] = aa_k
            length_weight = getattr(cfg.train, "w_geom_length_unroll", 0.0)
            angle_weight = getattr(cfg.train, "w_geom_angle_unroll", 0.0)
            if length_weight > 0 or angle_weight > 0:
                length_geom, angle_geom = ca_local_geometry_huber_losses(
                    out_k["P_hat_1"], batch[f"P_{k}"], batch["residue_mask"],
                    batch["bond_mask"], getattr(cfg.train, "geom_huber_delta", 0.05),
                )
                loss = loss + cfg.train.w_unroll * (
                    length_weight * length_geom + angle_weight * angle_geom
                )
                comps[f"geom_length{k}"] = length_geom.item()
                comps[f"geom_angle{k}"] = angle_geom.item()
            P_prev, V_prev = out_k["P_hat_1"].detach(), out_k["V_hat_1"].detach()
            k += 1
    return loss, comps


def lr_at(step: int, cfg) -> float:
    """Warmup (linear 0->lr over warmup_steps) then linear decay lr -> lr_final.

    Matches the paper recipe when lr=5e-3, lr_final=3e-3 with a short warmup.
    """
    lr, lr_final = cfg.train.lr, cfg.train.lr_final
    warmup = getattr(cfg.train, "warmup_steps", 0)
    horizon = getattr(cfg.train, "lr_horizon_steps", 0)
    if horizon < 0:
        raise ValueError("lr_horizon_steps must be non-negative")
    if horizon and horizon < warmup:
        raise ValueError("lr_horizon_steps must be at least warmup_steps")
    total = max(1, horizon or cfg.train.max_steps)
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if lr_final <= 0:
        return lr
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr + (lr_final - lr) * frac
