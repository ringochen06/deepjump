import numpy as np
import pytest

from deepjump.data.mdcath import MdcathPairDataset


def _index_only_dataset(counts):
    ds = object.__new__(MdcathPairDataset)
    ds._cum = np.cumsum(counts)
    ds._total = int(ds._cum[-1])
    return ds


def test_stratified_indices_cover_every_trajectory_deterministically():
    ds = _index_only_dataset([499, 120, 3])

    first = ds.stratified_indices(samples_per_trajectory=2, seed=17)
    second = ds.stratified_indices(samples_per_trajectory=2, seed=17)

    assert first == second
    assert len(first) == 6
    assert sum(0 <= i < 499 for i in first) == 2
    assert sum(499 <= i < 619 for i in first) == 2
    assert sum(619 <= i < 622 for i in first) == 2


def test_stratified_indices_reject_non_positive_sample_count():
    ds = _index_only_dataset([10])

    with pytest.raises(ValueError, match="samples_per_trajectory"):
        ds.stratified_indices(samples_per_trajectory=0)
