import torch

from deepjump.data.mdcath import collate_pairs
from deepjump.losses import ca_bond_length_huber_loss


def test_bond_loss_uses_topology_mask_not_residue_type_ids():
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [20.0, 0.0, 0.0]]]
    )
    pred = target.clone()
    pred[:, 1] = torch.tensor([2.8, 0.0, 0.0])
    pred[:, 2] = torch.tensor([6.6, 0.0, 0.0])
    pred[:, 3] = torch.tensor([200.0, 0.0, 0.0])
    mask = torch.tensor([[True, True, True, False]])
    # Residue type IDs are deliberately arbitrary; topology is a separate field.
    bond_mask = torch.tensor([[True, False, True]])

    loss = ca_bond_length_huber_loss(pred, target, mask, bond_mask)

    # Only residues 10--11 form a valid, unpadded consecutive bond.
    assert torch.allclose(loss, torch.tensor(0.5))


def test_bond_loss_has_finite_gradients_and_zero_at_target():
    target = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]]]
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    bond_mask = torch.ones(1, 2, dtype=torch.bool)
    exact = target.clone().requires_grad_(True)
    assert ca_bond_length_huber_loss(exact, target, mask, bond_mask).item() == 0.0

    pred = (target * 0.8).requires_grad_(True)
    loss = ca_bond_length_huber_loss(pred, target, mask, bond_mask)
    loss.backward()
    assert loss.item() > 0
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_bond_loss_rejects_wrong_mask_shape():
    P = torch.zeros(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    try:
        ca_bond_length_huber_loss(P, P, mask, torch.ones(1, 3, dtype=torch.bool))
    except ValueError as exc:
        assert "bond_mask shape" in str(exc)
    else:
        raise AssertionError("expected a shape error")


def test_collate_preserves_bond_mask_and_pads_it_false():
    def item(n, bonds, domain, temperature, replica, start_frame):
        return {
            "P_t": torch.zeros(n, 3),
            "V_t": torch.zeros(n, 13, 3),
            "P_1": torch.zeros(n, 3),
            "V_1": torch.zeros(n, 13, 3),
            "res_index": torch.arange(n) % 20,
            "atom_mask": torch.ones(n, 13, dtype=torch.bool),
            "bond_mask": torch.tensor(bonds, dtype=torch.bool),
            "delta_ns": torch.tensor(1.0),
            "temperature": torch.tensor(temperature),
            "replica": torch.tensor(replica),
            "start_frame": torch.tensor(start_frame),
            "residue_start": torch.tensor(3),
            "n_res": n,
            "domain": domain,
        }

    batch = collate_pairs([
        item(4, [True, False, True], "a", 320, 0, 7),
        item(2, [True], "b", 450, 4, 19),
    ])
    assert batch["bond_mask"].tolist() == [[True, False, True], [True, False, False]]
    assert batch["temperature"].dtype == torch.long
    assert batch["temperature"].tolist() == [320, 450]
    assert batch["replica"].tolist() == [0, 4]
    assert batch["start_frame"].tolist() == [7, 19]
    assert batch["residue_start"].tolist() == [3, 3]
