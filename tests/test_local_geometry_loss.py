import torch

from deepjump.losses import ca_local_geometry_huber_losses


def test_local_geometry_ignores_topology_gaps():
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [20.0, 0.0, 0.0], [21.0, 0.0, 0.0]]])
    pred = target.clone()
    pred[:, 2] = torch.tensor([200.0, 80.0, 0.0])
    pred[:, 3] = pred[:, 2] + torch.tensor([1.0, 0.0, 0.0])
    residue_mask = torch.ones(1, 4, dtype=torch.bool)
    bond_mask = torch.tensor([[True, False, True]])

    length, angle = ca_local_geometry_huber_losses(pred, target, residue_mask, bond_mask)
    assert length.item() == 0.0
    assert angle.item() == 0.0  # no valid three-residue run crosses the gap


def test_local_geometry_penalizes_length_and_angle_with_finite_gradients():
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0]]])
    pred = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0],
                          [1.8, 0.8, 0.0]]], requires_grad=True)
    residue_mask = torch.ones(1, 3, dtype=torch.bool)
    bond_mask = torch.ones(1, 2, dtype=torch.bool)

    length, angle = ca_local_geometry_huber_losses(pred, target, residue_mask, bond_mask)
    assert length.item() > 0.0
    assert angle.item() > 0.0
    (length + angle).backward()
    assert torch.isfinite(pred.grad).all()


def test_local_geometry_rejects_wrong_bond_mask_shape():
    points = torch.zeros(1, 3, 3)
    residue_mask = torch.ones(1, 3, dtype=torch.bool)
    try:
        ca_local_geometry_huber_losses(
            points, points, residue_mask, torch.ones(1, 3, dtype=torch.bool)
        )
    except ValueError as exc:
        assert "bond_mask shape" in str(exc)
    else:
        raise AssertionError("expected a shape validation error")
