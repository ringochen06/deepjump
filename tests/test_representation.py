"""Rotation-equivariance tests for the (P, V) representation.

These are the most important correctness gate in the whole project: an
equivariance bug is silent (loss still drops) but wrong. Nothing downstream
should be trusted until these pass.
"""

import numpy as np
import pytest
import torch

from deepjump.atom_constants import HEAVY_ATOM_INDEX, HEAVY_ATOM_ORDER, MAX_HEAVY
from deepjump.representation import apply_layout, apply_model_layout, build_layout


def _toy_topology():
    """Three residues (GLY, ALA, TRP) -> exercises 0, 1, and 10 sidechain atoms."""
    atom_names, resids, resnames = [], [], []
    for rid, res in enumerate(["GLY", "ALA", "TRP"]):
        # backbone incl. CA, then canonical heavy atoms
        names = ["N", "CA", "C", "O"] + [a for a in HEAVY_ATOM_ORDER[res] if a not in ("N", "C", "O")]
        for nm in names:
            atom_names.append(nm)
            resids.append(rid)
            resnames.append(res)
    return atom_names, np.array(resids), resnames


def _random_rotation(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))  # make it a proper rotation-ish orthonormal
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_layout_shapes_and_mask():
    names, resids, resnames = _toy_topology()
    layout = build_layout(names, resids, resnames)
    assert layout.num_residues == 3
    # GLY has 0 sidechain -> 3 heavy (N,C,O); ALA -> 4; TRP -> 13
    assert layout.atom_mask.sum(axis=1).tolist() == [3, 4, 13]
    assert layout.atom_mask.shape == (3, MAX_HEAVY)
    assert layout.bond_mask.tolist() == [True, True]


def test_layout_bond_mask_tracks_raw_residue_gaps_not_residue_types():
    names, resids, resnames = _toy_topology()
    resids = resids.copy()
    resids[resids == 2] = 5
    layout = build_layout(names, resids, resnames)
    assert layout.res_index.tolist() != [0, 1, 2]
    assert layout.bond_mask.tolist() == [True, False]


def test_pair_distance_invariant_and_offsets_covariant():
    names, resids, resnames = _toy_topology()
    layout = build_layout(names, resids, resnames)

    A = len(names)
    coords = torch.randn(A, 3, generator=torch.Generator().manual_seed(1)) * 10.0
    P, V = apply_layout(coords, layout)

    R = _random_rotation(2)
    P_rot, V_rot = apply_layout(coords @ R.T, layout)

    # CA pairwise distances are rotation-invariant.
    d = torch.cdist(P, P)
    d_rot = torch.cdist(P_rot, P_rot)
    assert torch.allclose(d, d_rot, atol=1e-4)

    # P and V are l=1: they rotate covariantly.
    assert torch.allclose(P_rot, P @ R.T, atol=1e-4)
    assert torch.allclose(V_rot, V @ R.T, atol=1e-4)


def test_padded_slots_are_zero():
    names, resids, resnames = _toy_topology()
    layout = build_layout(names, resids, resnames)
    coords = torch.randn(len(names), 3, generator=torch.Generator().manual_seed(3)) * 10.0
    _, V = apply_layout(coords, layout)
    mask = torch.as_tensor(layout.atom_mask)
    # every masked-out slot must be exactly zero
    assert torch.count_nonzero(V[~mask]) == 0


def test_batched_frames():
    names, resids, resnames = _toy_topology()
    layout = build_layout(names, resids, resnames)
    coords = torch.randn(7, len(names), 3, generator=torch.Generator().manual_seed(4))
    P, V = apply_layout(coords, layout)
    assert P.shape == (7, 3, 3)
    assert V.shape == (7, 3, MAX_HEAVY, 3)


def test_apply_model_layout_matches_training_symmetric_canonicalization():
    names = ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]
    layout = build_layout(names, np.zeros(len(names), dtype=int), ["ASP"] * len(names))
    coords = torch.zeros(len(names), 3)
    coords[names.index("OD1"), 0] = 3.0
    coords[names.index("OD2"), 0] = 1.0

    _, raw = apply_model_layout(coords, layout, canon_symmetric=False)
    _, canonical = apply_model_layout(coords, layout, canon_symmetric=True)
    od1 = HEAVY_ATOM_INDEX["ASP"]["OD1"]
    od2 = HEAVY_ATOM_INDEX["ASP"]["OD2"]

    assert raw[0, od1, 0] == 3.0 and raw[0, od2, 0] == 1.0
    assert canonical[0, od1, 0] == 1.0 and canonical[0, od2, 0] == 3.0


def test_apply_model_layout_rejects_batched_frames():
    names, resids, resnames = _toy_topology()
    layout = build_layout(names, resids, resnames)
    coords = torch.zeros(2, len(names), 3)
    with pytest.raises(ValueError, match="one coordinate frame"):
        apply_model_layout(coords, layout, canon_symmetric=True)
