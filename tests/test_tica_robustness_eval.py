import numpy as np

from scripts.tica_robustness_eval import (
    contiguous_frame_ids,
    fit_tica,
    reference_histogram_jsd,
    selected_pair_indices,
)


def test_reference_frames_form_centered_contiguous_window():
    assert np.array_equal(contiguous_frame_ids(10, 4), np.array([3, 4, 5, 6]))
    assert np.array_equal(contiguous_frame_ids(3, 10), np.array([0, 1, 2]))


def test_selected_pairs_are_deterministic_unique_and_bounded():
    first = selected_pair_indices(64, 128)
    second = selected_pair_indices(64, 128)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(first[0]) == 128
    assert len(set(zip(first[0], first[1]))) == 128
    assert np.all(first[0] < first[1])


def test_tica_projection_is_finite_and_has_requested_shape():
    rng = np.random.default_rng(7)
    latent = np.cumsum(rng.normal(size=(200, 3)), axis=0)
    features = latent @ rng.normal(size=(3, 16)) + 0.01 * rng.normal(size=(200, 16))
    mean, projection = fit_tica(features, lag=2)
    assert mean.shape == (16,)
    assert projection.shape == (16, 2)
    assert np.isfinite(projection).all()


def test_reference_fixed_jsd_is_zero_for_identity_and_positive_for_shift():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(500, 2))
    identity = reference.copy()
    shifted = reference + 3.0
    assert reference_histogram_jsd(reference, identity) < 1e-10
    assert reference_histogram_jsd(reference, shifted) > 0.1
