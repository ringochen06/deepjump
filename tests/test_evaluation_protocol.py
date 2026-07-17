import hashlib
from pathlib import Path

import numpy as np
import pytest

from deepjump.evaluation import (
    aggregate_geometry_panel,
    assign_clusters,
    calibrate_geometry_envelope,
    fit_kmeans,
    geometry_frame_statistics,
    geometry_panel_passes,
    jsd_bits,
    load_frozen_domain_ids,
    paired_domain_bootstrap_gain,
    reference_transition_deltas,
    require_single_delta,
    resolve_frozen_domains,
    transition_matrix,
    weighted_row_jsd_bits,
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


def test_kmeans_and_cluster_assignment_are_deterministic():
    values = np.array([[-2.1], [-2.0], [-1.9], [1.9], [2.0], [2.1]])
    centers_a, labels_a = fit_kmeans(values, 2, seed=7)
    centers_b, labels_b = fit_kmeans(values, 2, seed=7)
    assert np.array_equal(centers_a, centers_b)
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(assign_clusters(values, centers_a), labels_a)
    assert sorted(np.bincount(labels_a).tolist()) == [3, 3]


def test_transition_power_and_row_jsd_bits_distinguish_noop():
    labels = np.array([0, 1, 0, 1, 0, 1])
    one_step, _ = transition_matrix(labels, n_states=2, pseudocount=0.0)
    assert np.array_equal(one_step, np.array([[0.0, 1.0], [1.0, 0.0]]))
    two_step = np.linalg.matrix_power(one_step, 2)
    assert np.array_equal(two_step, np.eye(2))

    sample_good, weights = transition_matrix(
        np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]),
        n_states=2, pseudocount=0.0,
    )
    good, rows = weighted_row_jsd_bits(two_step, sample_good, weights)
    assert good == 0.0
    assert np.array_equal(rows, np.zeros(2))

    sample_bad, _ = transition_matrix(
        np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]),
        n_states=2, pseudocount=0.0,
    )
    bad, _ = weighted_row_jsd_bits(two_step, sample_bad, weights)
    assert bad == 1.0
    assert jsd_bits(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_geometry_envelope_is_topology_aware_and_detects_collapse():
    base = np.array([
        [0.0, 0.0, 0.0],
        [3.8, 0.0, 0.0],
        [7.6, 0.2, 0.0],
        [11.4, 0.2, 0.0],
    ])
    real = np.stack([base + np.array([0.0, 0.01 * i, 0.0]) for i in range(20)])
    bond_mask = np.array([True, True, True])
    statistics = geometry_frame_statistics(real, bond_mask)
    panel = aggregate_geometry_panel({name: values[:4] for name, values in statistics.items()})
    envelope = calibrate_geometry_envelope(statistics, 4, draws=200, seed=7)
    passed, checks = geometry_panel_passes(panel, envelope)
    assert passed and all(checks.values())

    collapsed = real[:4].copy()
    collapsed[:, 1:] *= 0.2
    collapsed_panel = aggregate_geometry_panel(
        geometry_frame_statistics(collapsed, bond_mask)
    )
    passed, checks = geometry_panel_passes(collapsed_panel, envelope)
    assert not passed
    assert not checks["bond_mean"]


def test_paired_domain_bootstrap_gain_requires_positive_lower_bound():
    baseline = np.ones(20)
    clear_gain = paired_domain_bootstrap_gain(
        np.full(20, 0.8), baseline, draws=1000, seed=3
    )
    assert clear_gain["passes"]
    assert clear_gain["ci95"][0] > 0

    mixed = paired_domain_bootstrap_gain(
        np.array([0.0, 2.0] * 10), baseline, draws=1000, seed=3
    )
    assert not mixed["passes"]
    assert mixed["ci95"][0] <= 0
