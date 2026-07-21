"""Multi-step rollout: chain the learned jump operator to generate a trajectory.

Each step samples X_hat_{t+delta} ~ p(.|X_t) via the ODE, then feeds that
prediction back as the next X_t. This is the time-coarsened trajectory the model
is meant to accelerate; rollout drift/stability is the key thing it stresses.
"""

from __future__ import annotations

import math

import torch


def _static_fields(batch: dict) -> dict:
    """Per-structure fields that do not change across a rollout."""
    out = {
        "res_index": batch["res_index"],
        "delta_ns": batch["delta_ns"],
        "residue_mask": batch["residue_mask"],
    }
    if "bond_mask" in batch:
        out["bond_mask"] = batch["bond_mask"]
    if "atom_mask" in batch:
        out["atom_mask"] = batch["atom_mask"]
    return out


def bond_geometry_ok(P, bond_mask=None, lo=3.2, hi=4.5, max_bond=5.5):
    """Per-sample CA-CA bond-geometry validity. P [B,N,3] -> bool [B].

    Real backbones have consecutive CA-CA ~3.8 A; a blown-up structure has a large
    mean and/or a stretched max bond. This is a cheap, energy-free plausibility gate
    (the spirit of Timewarp's accept/reject, without an explicit force field).
    """
    d = (P[:, 1:] - P[:, :-1]).norm(dim=-1)  # [B,N-1]
    if bond_mask is None:
        bond_mask = torch.ones_like(d, dtype=torch.bool)
    valid = bond_mask.to(d.dtype)
    mean = (d * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    mx = d.masked_fill(~bond_mask, float("-inf")).max(1).values
    mx = torch.where(bond_mask.any(1), mx, torch.full_like(mx, float("inf")))
    return (mean > lo) & (mean < hi) & (mx < max_bond)


def reject_to_source(
    proposed_P: torch.Tensor,
    proposed_V: torch.Tensor,
    source_P: torch.Tensor,
    source_V: torch.Tensor,
    bond_mask: torch.Tensor,
    *,
    lo: float = 3.2,
    hi: float = 4.5,
    max_bond: float = 5.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reject non-finite or nonphysical samples and return their exact source state.

    The decision is per sample and depends only on that sample's proposed state and
    fixed topology. Callers that use this as a scientific safeguard must still
    report raw finite/geometry metrics; a fallback must not hide model failures.
    """
    if proposed_P.shape != source_P.shape or proposed_V.shape != source_V.shape:
        raise ValueError("proposed and source tensors must have matching shapes")
    tensors = (proposed_P, proposed_V, source_P, source_V, bond_mask)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("proposed, source, and topology tensors must share a device")
    if proposed_P.dtype != source_P.dtype or proposed_V.dtype != source_V.dtype:
        raise ValueError("proposed and source tensors must have matching dtypes")
    if not all(math.isfinite(value) for value in (lo, hi, max_bond)):
        raise ValueError("bond thresholds must be finite")
    if not 0 <= lo < hi or max_bond <= 0:
        raise ValueError("bond thresholds are invalid")
    if proposed_P.ndim != 3 or proposed_P.shape[-1] != 3:
        raise ValueError("position tensors must have shape [batch, residues, 3]")
    if proposed_P.shape[1] < 2:
        raise ValueError("at least two residues are required for a bond safeguard")
    if proposed_V.ndim != 4 or proposed_V.shape[:2] != proposed_P.shape[:2]:
        raise ValueError(
            "vector tensors must have shape [batch, residues, channels, 3]"
        )
    if proposed_V.shape[-1] != 3:
        raise ValueError("vector tensors must have a final Cartesian dimension of 3")
    if bond_mask.dtype != torch.bool:
        raise ValueError("bond_mask must be boolean")
    if bond_mask.shape != (proposed_P.shape[0], proposed_P.shape[1] - 1):
        raise ValueError("bond_mask must have shape [batch, residues - 1]")
    if not bond_mask.any(1).all():
        raise ValueError("every sample must have at least one topology-valid bond")

    source_finite = torch.isfinite(source_P).flatten(1).all(1)
    source_finite &= torch.isfinite(source_V).flatten(1).all(1)
    source_physical = bond_geometry_ok(
        source_P, bond_mask, lo=lo, hi=hi, max_bond=max_bond
    )
    if not (source_finite & source_physical).all():
        raise ValueError("source state must be finite and physically valid")

    geometry_ok = bond_geometry_ok(
        proposed_P, bond_mask, lo=lo, hi=hi, max_bond=max_bond
    )
    finite = torch.isfinite(proposed_P).flatten(1).all(1)
    finite &= torch.isfinite(proposed_V).flatten(1).all(1)
    accepted = geometry_ok & finite
    guarded_P = torch.where(accepted[:, None, None], proposed_P, source_P)
    guarded_V = torch.where(accepted[:, None, None, None], proposed_V, source_V)
    return guarded_P, guarded_V, accepted


@torch.no_grad()
def rollout(model, init_batch: dict, n_steps: int, ode_steps: int = 20,
            mode: str = "ode", gate: bool = False, generator=None, sample_kwargs=None):
    """Roll the jump operator n_steps times from init_batch's (P_t, V_t).

    With gate=True, a proposed jump is accepted only if its CA-CA geometry stays
    physical; otherwise the previous frame is kept (a rejected step). This bounds
    the rollout when the single-step model would otherwise diverge off-distribution.

    Returns (traj, accepts): traj is a list of (P, V) of length n_steps + 1;
    accepts is a list of per-step acceptance bool tensors [B] (empty if gate=False).
    """
    static = _static_fields(init_batch)
    P, V = init_batch["P_t"], init_batch["V_t"]
    traj = [(P, V)]
    accepts = []
    sample_kwargs = sample_kwargs or {}
    for _ in range(n_steps):
        cur = {"P_t": P, "V_t": V, **static}
        P_new, V_new = model.sample(
            cur, steps=ode_steps, mode=mode, generator=generator, **sample_kwargs
        )
        if gate:
            ok = bond_geometry_ok(P_new, cur.get("bond_mask"))  # [B]
            P = torch.where(ok.view(-1, 1, 1), P_new, P)
            V = torch.where(ok.view(-1, 1, 1, 1), V_new, V)
            accepts.append(ok)
        else:
            P, V = P_new, V_new
        traj.append((P, V))
    return traj, accepts
