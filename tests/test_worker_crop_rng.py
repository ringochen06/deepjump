import numpy as np
import torch
from torch.utils.data import DataLoader

from deepjump.data.mdcath import MdcathPairDataset


class _CropOnlyDataset(MdcathPairDataset):
    def __init__(self):
        self.crop_length = 5
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)
        self._rng_worker_id = None

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return self._crop(100).start


def _crop_sequence():
    torch.manual_seed(17)
    return [int(value) for value in DataLoader(
        _CropOnlyDataset(), batch_size=1, num_workers=2,
    )]


def test_crop_rng_is_distinct_per_worker_and_reproducible():
    first = _crop_sequence()
    second = _crop_sequence()

    assert first == second
    assert any(a != b for a, b in zip(first[::2], first[1::2]))
