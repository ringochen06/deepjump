"""Multi-step rollout: chain the learned jump operator to generate a trajectory.

Each step samples X_hat_{t+delta} ~ p(.|X_t) via the ODE, then feeds that
prediction back as the next X_t. This is the time-coarsened trajectory the model
is meant to accelerate; rollout drift/stability is the key thing it stresses.
"""

from __future__ import annotations

import torch


def _static_fields(batch: dict) -> dict:
    """Per-structure fields that do not change across a rollout."""
    return {
        "res_index": batch["res_index"],
        "delta_ns": batch["delta_ns"],
        "residue_mask": batch["residue_mask"],
    }


def bond_geometry_ok(P, lo=3.2, hi=4.5, max_bond=5.5):
    """Per-sample CA-CA bond-geometry validity. P [B,N,3] -> bool [B].

    Real backbones have consecutive CA-CA ~3.8 A; a blown-up structure has a large
    mean and/or a stretched max bond. This is a cheap, energy-free plausibility gate
    (the spirit of Timewarp's accept/reject, without an explicit force field).
    """
    d = (P[:, 1:] - P[:, :-1]).norm(dim=-1)  # [B,N-1]
    mean = d.mean(1)
    mx = d.max(1).values
    return (mean > lo) & (mean < hi) & (mx < max_bond)


@torch.no_grad()
def rollout(model, init_batch: dict, n_steps: int, ode_steps: int = 20,
            mode: str = "ode", gate: bool = False, generator=None):
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
    for _ in range(n_steps):
        cur = {"P_t": P, "V_t": V, **static}
        P_new, V_new = model.sample(cur, steps=ode_steps, mode=mode, generator=generator)
        if gate:
            ok = bond_geometry_ok(P_new)  # [B]
            P = torch.where(ok.view(-1, 1, 1), P_new, P)
            V = torch.where(ok.view(-1, 1, 1, 1), V_new, V)
            accepts.append(ok)
        else:
            P, V = P_new, V_new
        traj.append((P, V))
    return traj, accepts
