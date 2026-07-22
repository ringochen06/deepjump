import pytest
import torch

from deepjump.sampling import reject_to_source


def _states():
    source_P = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]],
        ]
    )
    source_V = torch.arange(2 * 3 * 2 * 3, dtype=torch.float32).reshape(2, 3, 2, 3)
    proposed_P = source_P.clone()
    proposed_P[0, :, 1] = torch.tensor([0.1, 0.2, 0.3])
    proposed_P[1, 2, 0] = 20.0
    proposed_V = source_V + 100.0
    bond_mask = torch.ones(2, 2, dtype=torch.bool)
    return proposed_P, proposed_V, source_P, source_V, bond_mask


def test_reject_to_source_is_per_sample_and_exact():
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()

    guarded_P, guarded_V, accepted = reject_to_source(
        proposed_P, proposed_V, source_P, source_V, bond_mask
    )

    assert accepted.tolist() == [True, False]
    assert torch.equal(guarded_P[0], proposed_P[0])
    assert torch.equal(guarded_V[0], proposed_V[0])
    assert torch.equal(guarded_P[1], source_P[1])
    assert torch.equal(guarded_V[1], source_V[1])


@pytest.mark.parametrize("field", ["P", "V"])
def test_reject_to_source_rejects_nonfinite_proposals(field):
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()
    proposed_P[1] = source_P[1]
    if field == "P":
        proposed_P[1, 0, 0] = float("nan")
    else:
        proposed_V[1, 0, 0, 0] = float("inf")

    guarded_P, guarded_V, accepted = reject_to_source(
        proposed_P, proposed_V, source_P, source_V, bond_mask
    )

    assert accepted.tolist() == [True, False]
    assert torch.equal(guarded_P[1], source_P[1])
    assert torch.equal(guarded_V[1], source_V[1])


def test_reject_to_source_uses_only_topology_valid_bonds():
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()
    proposed_P[1] = source_P[1]
    proposed_P[1, 2, 0] = 100.0
    bond_mask[1, 1] = False

    guarded_P, guarded_V, accepted = reject_to_source(
        proposed_P, proposed_V, source_P, source_V, bond_mask
    )

    assert accepted.tolist() == [True, True]
    assert torch.equal(guarded_P, proposed_P)
    assert torch.equal(guarded_V, proposed_V)


@pytest.mark.parametrize("field", ["P", "V", "geometry"])
def test_reject_to_source_rejects_invalid_source(field):
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()
    proposed_P[1] = source_P[1]
    if field == "P":
        source_P[1, 0, 0] = float("nan")
    elif field == "V":
        source_V[1, 0, 0, 0] = float("inf")
    else:
        source_P[1, 2, 0] = 20.0

    with pytest.raises(ValueError, match="source state"):
        reject_to_source(proposed_P, proposed_V, source_P, source_V, bond_mask)


def test_reject_to_source_uses_strict_thresholds():
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()
    proposed_P[:] = source_P
    proposed_P[0, 1, 0] = 3.2
    proposed_P[0, 2, 0] = 6.4

    guarded_P, _, accepted = reject_to_source(
        proposed_P, proposed_V, source_P, source_V, bond_mask
    )

    assert accepted.tolist() == [False, True]
    assert torch.equal(guarded_P[0], source_P[0])


def test_reject_to_source_requires_nonempty_topology_for_every_sample():
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()
    bond_mask[1] = False
    with pytest.raises(ValueError, match="topology-valid bond"):
        reject_to_source(proposed_P, proposed_V, source_P, source_V, bond_mask)


@pytest.mark.parametrize(
    "lo, hi, maximum",
    [(float("nan"), 4.5, 5.5), (4.5, 3.2, 5.5), (3.2, 4.5, 0.0)],
)
def test_reject_to_source_rejects_invalid_thresholds(lo, hi, maximum):
    values = _states()
    with pytest.raises(ValueError, match="thresholds"):
        reject_to_source(*values, lo=lo, hi=hi, max_bond=maximum)


def test_reject_to_source_rejects_mixed_dtypes():
    proposed_P, proposed_V, source_P, source_V, bond_mask = _states()
    with pytest.raises(ValueError, match="matching dtypes"):
        reject_to_source(
            proposed_P.double(), proposed_V, source_P, source_V, bond_mask
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda values: values.__setitem__(2, values[2][:, :-1]), "matching shapes"),
        (lambda values: (values.__setitem__(0, values[0][..., :2]), values.__setitem__(2, values[2][..., :2])), "position tensors"),
        (lambda values: values.__setitem__(4, values[4][:, :-1]), "bond_mask"),
        (lambda values: values.__setitem__(4, values[4].float()), "boolean"),
    ],
)
def test_reject_to_source_rejects_shape_mismatches(mutation, message):
    values = list(_states())
    mutation(values)
    with pytest.raises(ValueError, match=message):
        reject_to_source(*values)
