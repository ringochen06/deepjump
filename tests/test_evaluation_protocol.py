import hashlib
from pathlib import Path

import numpy as np
import pytest

from deepjump.evaluation import (
    load_frozen_domain_ids,
    reference_transition_deltas,
    require_single_delta,
    resolve_frozen_domains,
)


def test_require_single_delta_rejects_mixed_or_invalid_values():
    assert require_single_delta(10) == 10
    assert require_single_delta([100]) == 100
    for value in ([1, 10], [], 0, -1, 1.5, True):
        with pytest.raises(ValueError):
            require_single_delta(value)


def test_frozen_domain_list_requires_matching_sha_and_exact_files(tmp_path: Path):
    panel = tmp_path / "dev.txt"
    panel.write_text("1abcA00\n2defB01\n")
    digest = hashlib.sha256(panel.read_bytes()).hexdigest()
    ids, actual = load_frozen_domain_ids(panel, digest)
    assert ids == ["1abcA00", "2defB01"]
    assert actual == digest
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_frozen_domain_ids(panel, "0" * 64)

    files = [
        tmp_path / "mdcath_dataset_2defB01.h5",
        tmp_path / "mdcath_dataset_1abcA00.h5",
    ]
    assert resolve_frozen_domains(files, ids) == [files[1], files[0]]
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_frozen_domains(files[:1], ids)


def test_reference_transition_deltas_use_checkpoint_delta():
    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    assert np.array_equal(
        reference_transition_deltas(values, 2),
        np.full((4, 2), 4.0),
    )
    with pytest.raises(ValueError, match="more than 6"):
        reference_transition_deltas(values, 6)
