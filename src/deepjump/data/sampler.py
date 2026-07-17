"""Distributed sampler with checkpointable per-rank index offsets."""

from __future__ import annotations

import itertools

from torch.utils.data import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """Slice the deterministic per-rank index stream without reading samples.

    ``start_offset`` counts indices already consumed by this rank in the current
    sampler epoch. The trainer, not the sampler, advances it because DataLoader
    prefetch may request indices that never reach an optimizer step.
    """

    def __init__(self, *args, start_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_start_offset(start_offset)

    def set_start_offset(self, start_offset: int) -> None:
        start_offset = int(start_offset)
        if not 0 <= start_offset <= self.num_samples:
            raise ValueError(
                f"start_offset must be in [0, {self.num_samples}], got {start_offset}"
            )
        self.start_offset = start_offset

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_offset, None)

    def __len__(self) -> int:
        return self.num_samples - self.start_offset
