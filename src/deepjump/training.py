"""Loss composition and LR schedule shared by the single-GPU and DDP trainers."""

from __future__ import annotations

import math

from .losses import (
    allatom_pairwise_huber_loss,
    heavy_atom_offset_loss,
    pairwise_vector_huber_loss,
)


def _step_loss(P_hat, V_hat, P_gt, V_gt, batch, cfg):
    ca = pairwise_vector_huber_loss(P_hat, P_gt, batch["residue_mask"], cfg.train.huber_delta)
    loss = ca
    off_val = None
    if cfg.train.w_offset > 0 and V_hat is not None:
        off = heavy_atom_offset_loss(V_hat, V_gt, batch["atom_mask"], cfg.train.huber_delta)
        loss = ca + cfg.train.w_offset * off
        off_val = off.item()
    if cfg.train.w_allatom > 0 and V_hat is not None:
        aa = allatom_pairwise_huber_loss(
            P_hat, V_hat, P_gt, V_gt, batch["atom_mask"], batch["residue_mask"],
            cutoff=cfg.model.dist_cutoff, delta=cfg.train.huber_delta,
        )
        loss = loss + cfg.train.w_allatom * aa
    return loss, ca.item(), off_val


def total_loss(out, batch, cfg, model=None):
    """Step-1 loss + optional self-conditioned step-2..K losses (k-step unroll).

    `model` may be a DDP-wrapped module; its forward signature is unchanged.
    """
    loss1, ca1, off1 = _step_loss(
        out["P_hat_1"], out.get("V_hat_1"), batch["P_1"], batch["V_1"], batch, cfg
    )
    loss, comps = loss1, {"ca": ca1}
    if off1 is not None:
        comps["offset"] = off1
    if cfg.train.w_unroll > 0 and model is not None and out.get("V_hat_1") is not None:
        P_prev, V_prev = out["P_hat_1"].detach(), out["V_hat_1"].detach()
        k = 2
        while f"P_{k}" in batch:
            batch_k = {**batch, "P_t": P_prev, "V_t": V_prev,
                       "P_1": batch[f"P_{k}"], "V_1": batch[f"V_{k}"]}
            out_k = model(batch_k)
            loss_k, ca_k, _ = _step_loss(
                out_k["P_hat_1"], out_k.get("V_hat_1"),
                batch[f"P_{k}"], batch[f"V_{k}"], batch, cfg
            )
            loss = loss + cfg.train.w_unroll * loss_k
            comps[f"ca{k}"] = ca_k
            P_prev, V_prev = out_k["P_hat_1"].detach(), out_k["V_hat_1"].detach()
            k += 1
    return loss, comps


def lr_at(step: int, cfg) -> float:
    """Warmup (linear 0->lr over warmup_steps) then linear decay lr -> lr_final.

    Matches the paper recipe when lr=5e-3, lr_final=3e-3 with a short warmup.
    """
    lr, lr_final = cfg.train.lr, cfg.train.lr_final
    warmup = getattr(cfg.train, "warmup_steps", 0)
    total = max(1, cfg.train.max_steps)
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if lr_final <= 0:
        return lr
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr + (lr_final - lr) * frac
