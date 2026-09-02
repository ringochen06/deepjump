from pathlib import Path

import numpy as np
import torch

from scripts.export_rollout_pdb import (
    ResidueRecord,
    _write_pymol_scripts,
    read_pdb_input,
    write_multistate_pdb,
)


def test_write_multistate_pdb_preserves_states_and_atom_order(tmp_path: Path):
    residues = [ResidueRecord(resid=7, resname="ALA", chain="A")]
    atom_mask = np.zeros((1, 13), dtype=bool)
    atom_mask[0, :4] = True  # N, C, O, CB
    p0 = torch.tensor([[1.0, 2.0, 3.0]])
    v0 = torch.zeros(1, 13, 3)
    v0[0, 0] = torch.tensor([-1.0, 0.0, 0.0])
    p1 = p0 + 1.0

    output = tmp_path / "rollout.pdb"
    write_multistate_pdb(output, [(p0, v0), (p1, v0)], residues, atom_mask)

    text = output.read_text()
    assert text.count("MODEL") == 2
    assert text.count("ENDMDL") == 2
    assert text.count(" CA ") == 2
    assert text.count(" CB ") == 2
    assert "   1.000   2.000   3.000" in text
    assert "   2.000   3.000   4.000" in text
    assert text.endswith("END\n")

    coords, layout, parsed_residues = read_pdb_input(output)
    assert coords.shape == (5, 3)
    assert layout.num_residues == 1
    assert layout.atom_mask[0, :4].all()
    assert parsed_residues == residues


def test_pymol_script_aligns_states_for_a_stable_camera(tmp_path: Path):
    pdb_path = tmp_path / "rollout.pdb"
    pdb_path.write_text("END\n")

    interactive, render = _write_pymol_scripts(pdb_path, n_states=81, fps=5)

    for script in (interactive, render):
        text = script.read_text()
        assert "intra_fit rollout and name CA, 1" in text
        assert "set all_states, off" in text
        assert "mset 1 -81" in text
