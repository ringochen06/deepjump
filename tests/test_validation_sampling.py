from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


class _FrozenValidationDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def stratified_indices(self, seed):
        assert seed == 9
        return [0, 2]

    def __getitem__(self, index):
        return {"value": torch.tensor([float(index)])}


def test_frozen_validation_loader_avoids_worker_ipc_descriptors():
    from scripts.train_ddp import build_frozen_validation_loader

    cfg = SimpleNamespace(
        data=SimpleNamespace(seed=7),
        train=SimpleNamespace(batch_size=2, num_workers=8),
    )
    loader = build_frozen_validation_loader(_FrozenValidationDataset(), cfg)

    assert loader.num_workers == 0
    assert loader.dataset.indices == [0, 2]
