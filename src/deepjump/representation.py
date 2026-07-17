"""Build the DeepJump (P, V) representation from all-atom coordinates.

    P in R^{N x 3}       : CA global coordinate per residue
    V in R^{N x 13 x 3}  : offsets (heavy_atom - CA) in canonical order, zero-padded
    atom_mask in {0,1}^{N x 13} : 1 where a real heavy atom exists

Topology (which atom index maps to which (residue, slot)) is fixed across all
frames of a trajectory, so we precompute an integer layout ONCE per domain and
then gather cheaply for any batch of frames. Keeping P and V as l=1 vectors makes
the representation SE(3)-equivariant by construction: under R, P -> P R^T and
V -> V R^T leave the mask and slot assignment unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .atom_constants import (
    HEAVY_ATOM_INDEX,
    MAX_HEAVY,
    canonical_resname,
    residue_index,
)


@dataclass
class AtomLayout:
    """Static per-domain mapping from a flat atom list to (residue, slot)."""

    ca_index: np.ndarray  # [N] int  -> atom index of each residue's CA
    heavy_index: np.ndarray  # [N, 13] int -> atom index per slot (0 where missing)
    atom_mask: np.ndarray  # [N, 13] bool -> True where a real heavy atom exists
    res_index: np.ndarray  # [N] int -> residue type id (0..20)
    bond_mask: np.ndarray  # [N-1] bool -> true consecutive residues in the topology
    num_residues: int


def _to_str_array(a) -> np.ndarray:
    """Normalise atom/residue name arrays to python str (handles bytes)."""
    out = []
    for x in np.asarray(a).ravel():
        if isinstance(x, (bytes, np.bytes_)):
            x = x.decode(errors="replace")
        out.append(str(x).strip())
    return np.array(out, dtype=object)


def build_layout(atom_names, resids, resnames) -> AtomLayout:
    """Precompute the (residue, slot) layout for one domain topology.

    Args:
        atom_names: [A] atom names, e.g. "CA", "CB", "N" (bytes or str).
        resids:     [A] integer residue id per atom (contiguous per residue).
        resnames:   [A] residue 3-letter name per atom.
    """
    atom_names = _to_str_array(atom_names)
    resnames = _to_str_array(resnames)
    resids = np.asarray(resids).ravel()
    A = len(atom_names)
    assert len(resids) == A and len(resnames) == A, "per-atom arrays must align"

    # Preserve first-appearance order of residues.
    _, first_idx = np.unique(resids, return_index=True)
    ordered_resid = resids[np.sort(first_idx)]
    N = len(ordered_resid)

    ca_index = np.full(N, -1, dtype=np.int64)
    heavy_index = np.zeros((N, MAX_HEAVY), dtype=np.int64)
    atom_mask = np.zeros((N, MAX_HEAVY), dtype=bool)
    res_index = np.zeros(N, dtype=np.int64)

    for ri, rid in enumerate(ordered_resid):
        sel = np.where(resids == rid)[0]
        rname_raw = resnames[sel[0]]
        canon = canonical_resname(rname_raw)
        res_index[ri] = residue_index(rname_raw)
        slot_map = HEAVY_ATOM_INDEX.get(canon, {})
        for ai in sel:
            name = atom_names[ai]
            if name == "CA":
                ca_index[ri] = ai
            elif name in slot_map:
                slot = slot_map[name]
                heavy_index[ri, slot] = ai
                atom_mask[ri, slot] = True
    if (ca_index < 0).any():
        missing = np.where(ca_index < 0)[0]
        raise ValueError(f"{len(missing)} residues have no CA atom (idx {missing[:5]}...)")

    # `res_index` above is an amino-acid TYPE id and must never be used to infer
    # topology.  Preserve the raw residue numbering separately as an adjacency
    # mask so sequence gaps are excluded from local backbone losses/metrics.
    bond_mask = np.asarray(ordered_resid[1:] == ordered_resid[:-1] + 1, dtype=bool)
    return AtomLayout(ca_index, heavy_index, atom_mask, res_index, bond_mask, N)


def apply_layout(coords, layout: AtomLayout):
    """Gather (P, V) for a batch of frames using a precomputed layout.

    Args:
        coords: array/tensor [..., A, 3] all-atom coordinates.
    Returns:
        P: [..., N, 3], V: [..., N, 13, 3]  (torch tensors, float32)
    """
    is_torch = isinstance(coords, torch.Tensor)
    if not is_torch:
        coords = torch.as_tensor(np.asarray(coords))
    coords = coords.to(torch.float32)

    ca_index = torch.as_tensor(layout.ca_index, device=coords.device)
    heavy_index = torch.as_tensor(layout.heavy_index, device=coords.device)  # [N,13]
    mask = torch.as_tensor(layout.atom_mask, device=coords.device)  # [N,13]

    P = coords.index_select(-2, ca_index)  # [..., N, 3]

    flat_heavy = heavy_index.reshape(-1)  # [N*13]
    heavy_pos = coords.index_select(-2, flat_heavy)  # [..., N*13, 3]
    lead = coords.shape[:-2]
    heavy_pos = heavy_pos.reshape(*lead, layout.num_residues, MAX_HEAVY, 3)

    V = heavy_pos - P.unsqueeze(-2)  # offsets from CA
    V = V * mask.to(V.dtype).unsqueeze(-1)  # zero out padded slots
    return P, V


def kabsch_rotation(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Optimal rotation R (3x3) that best superimposes centered P onto centered Q.

    P, Q are [N, 3]. Returns R such that (R @ P_centered^T)^T ~= Q_centered.
    """
    Pc = P - P.mean(0, keepdim=True)
    Qc = Q - Q.mean(0, keepdim=True)
    H = Pc.T @ Qc
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=P.dtype, device=P.device))
    return Vt.T @ D @ U.T


def kabsch_align_target(P_t, V_t, P_1, V_1):
    """Remove rigid-body motion: align X_1 onto X_t and center both at X_t centroid.

    Rotating out the global tumbling between frames turns the jump into a purely
    internal conformational change (the only part a conditioned model can predict).
    Both P are centered at the origin (X_t's centroid); V_1 offsets are rotated.

    Returns centered/aligned (P_t, V_t, P_1, V_1).
    """
    centroid_t = P_t.mean(0, keepdim=True)
    R = kabsch_rotation(P_1, P_t)  # rotate X_1 into X_t's frame
    P1_c = P_1 - P_1.mean(0, keepdim=True)
    P_1_aln = (R @ P1_c.T).T  # centered at origin, aligned to X_t
    V_1_aln = V_1 @ R.T  # rotate heavy-atom offsets consistently
    P_t_c = P_t - centroid_t
    return P_t_c, V_t, P_1_aln, V_1_aln


def canonicalize_symmetric(V: torch.Tensor, res_index) -> torch.Tensor:
    """Fix the arbitrary labelling of symmetric sidechain atom pairs (ASP/GLU/PHE/TYR/ARG).

    For each symmetric pair (a, b), order them by a rotation/translation-invariant key --
    the distance from each atom to the residue's backbone N (an asymmetric reference) --
    swapping slots so the smaller-key atom is always first. V is [N, 13, 3] offsets from CA.
    """
    from .atom_constants import N_SLOT, SYMMETRIC_SLOTS

    V = V.clone()
    ri = res_index.tolist() if hasattr(res_index, "tolist") else list(res_index)
    for r, rtype in enumerate(ri):
        for (a, b) in SYMMETRIC_SLOTS.get(int(rtype), ()):
            n = V[r, N_SLOT]
            ka = (V[r, a] - n).norm()
            kb = (V[r, b] - n).norm()
            if ka > kb:
                tmp = V[r, a].clone()
                V[r, a] = V[r, b]
                V[r, b] = tmp
    return V


def kabsch_align_futures(P_t, V_t, futures):
    """Center X_t at origin and Kabsch-align each future (P, V) into X_t's frame.

    futures: list of (P, V). Returns (P_t_centered, V_t, [(P_aligned, V_aligned), ...]).
    All frames end up in a common rigid frame (X_t's), so a multi-step target is a
    consistent internal-motion sequence — the setup for unrolled/self-conditioning training.
    """
    P_t_c = P_t - P_t.mean(0, keepdim=True)
    out = []
    for P, V in futures:
        R = kabsch_rotation(P, P_t)
        P_al = (R @ (P - P.mean(0, keepdim=True)).T).T
        out.append((P_al, V @ R.T))
    return P_t_c, V_t, out


def res_index_tensor(layout: AtomLayout, device=None) -> torch.Tensor:
    return torch.as_tensor(layout.res_index, dtype=torch.long, device=device)


def atom_mask_tensor(layout: AtomLayout, device=None) -> torch.Tensor:
    return torch.as_tensor(layout.atom_mask, dtype=torch.bool, device=device)
