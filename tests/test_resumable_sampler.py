import pytest

from deepjump.data.sampler import ResumableDistributedSampler


def _sampler(rank=0, offset=0):
    return ResumableDistributedSampler(
        list(range(23)), num_replicas=2, rank=rank, shuffle=True,
        seed=17, drop_last=True, start_offset=offset,
    )


def test_resumed_sampler_matches_uninterrupted_rank_stream():
    full = list(_sampler(rank=0))
    consumed = 4
    resumed = list(_sampler(rank=0, offset=consumed))

    assert resumed == full[consumed:]
    assert len(resumed) == len(full) - consumed


def test_sampler_epoch_and_rank_remain_deterministic_after_offset():
    original = _sampler(rank=1)
    original.set_epoch(3)
    full = list(original)
    resumed = _sampler(rank=1, offset=6)
    resumed.set_epoch(3)

    assert list(resumed) == full[6:]


def test_sampler_rejects_invalid_offsets():
    sampler = _sampler()
    with pytest.raises(ValueError, match="start_offset"):
        sampler.set_start_offset(-1)
    with pytest.raises(ValueError, match="start_offset"):
        sampler.set_start_offset(sampler.num_samples + 1)
