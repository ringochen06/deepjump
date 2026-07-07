"""Canonical heavy-atom ordering per residue for the V representation.

V in R^{N x 13 x 3} holds, for each residue, the offset (atom_pos - CA_pos) of up
to 13 non-CA heavy atoms in a fixed canonical order. The order is:
    backbone [N, C, O]  +  sidechain heavy atoms (standard PDB order).
Tryptophan has the most (3 + 10 = 13), which sets MAX_HEAVY = 13.

Symmetric sidechains (ASP/GLU/PHE/TYR/ARG) are, in the lite stage, encoded with
their raw atom offsets (l=1). Ophiuchus' l=2 symmetric encoding is deferred.
"""

from __future__ import annotations

MAX_HEAVY = 13  # non-CA heavy atoms per residue (TRP is the max)

# 20 standard amino acids; index 20 reserved for UNK.
RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
UNK_INDEX = len(RESIDUES)  # 20
NUM_RESIDUE_TYPES = len(RESIDUES) + 1  # 21 (incl. UNK)

RESIDUE_TO_INDEX = {r: i for i, r in enumerate(RESIDUES)}

# Common 3-letter aliases seen in MD/PDB (protonation / histidine variants).
RESIDUE_ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS",
    "MSE": "MET",
}

_BACKBONE = ["N", "C", "O"]  # non-CA backbone heavy atoms

_SIDECHAIN: dict[str, list[str]] = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}

# Full canonical order (backbone + sidechain), padded conceptually to MAX_HEAVY.
HEAVY_ATOM_ORDER: dict[str, list[str]] = {
    res: _BACKBONE + _SIDECHAIN[res] for res in RESIDUES
}

# atom name -> slot index, per residue (for fast building of V).
HEAVY_ATOM_INDEX: dict[str, dict[str, int]] = {
    res: {name: i for i, name in enumerate(order)}
    for res, order in HEAVY_ATOM_ORDER.items()
}

for _res, _order in HEAVY_ATOM_ORDER.items():
    assert len(_order) <= MAX_HEAVY, f"{_res} has {len(_order)} > {MAX_HEAVY} heavy atoms"


# Symmetric sidechain atom pairs whose labelling is arbitrary (mirror-equivalent).
# DeepJump/Ophiuchus handle these with an l=2 encoding; we instead canonicalise the pair
# order by a rotation-invariant key (see representation.canonicalize_symmetric).
_SYMMETRIC_PAIRS: dict[str, list[tuple[str, str]]] = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ARG": [("NH1", "NH2")],
}

# residue index -> list of (slotA, slotB) into the [13] heavy-atom axis
SYMMETRIC_SLOTS: dict[int, list[tuple[int, int]]] = {}
for _res, _pairs in _SYMMETRIC_PAIRS.items():
    _idx = RESIDUE_TO_INDEX[_res]
    _slotmap = HEAVY_ATOM_INDEX[_res]
    SYMMETRIC_SLOTS[_idx] = [(_slotmap[a], _slotmap[b]) for a, b in _pairs]

# N atom lives in slot 0 (backbone order [N, C, O, ...]) -- used as the asymmetric reference.
N_SLOT = 0


def canonical_resname(resname: str) -> str:
    """Map a raw residue name to a standard 3-letter code (or 'UNK')."""
    r = resname.strip().upper()
    r = RESIDUE_ALIASES.get(r, r)
    return r if r in RESIDUE_TO_INDEX else "UNK"


def residue_index(resname: str) -> int:
    r = canonical_resname(resname)
    return RESIDUE_TO_INDEX.get(r, UNK_INDEX)
