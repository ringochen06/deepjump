from pathlib import Path

import pytest

from scripts.probe_loss_scales import _filter_checkpoint_domains


def test_filter_checkpoint_domains_honors_frozen_domain_list():
    files = [
        Path("/data/mdcath_dataset_1a0hA01.h5"),
        Path("/data/mdcath_dataset_2kl5A00.h5"),
    ]

    assert _filter_checkpoint_domains(files, ["1a0hA01"]) == [files[0]]
    assert _filter_checkpoint_domains(files, []) == files


def test_filter_checkpoint_domains_fails_closed_when_missing():
    with pytest.raises(ValueError, match="checkpoint domains not found"):
        _filter_checkpoint_domains(
            [Path("/data/mdcath_dataset_2kl5A00.h5")],
            ["1a0hA01"],
        )
