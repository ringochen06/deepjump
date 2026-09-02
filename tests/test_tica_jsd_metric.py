"""The TICA JSD must not reward a model for diverging.

`hist2d_jsd` used to size its 2D histogram to the union of the reference and the
model samples. A model whose samples fly off then stretches the extent, the fixed
bin count collapses the reference into one or two bins, the model spreads over
the rest, and the divergence measure *falls*. Under that rule a wrong, diverging
model scored better than one sampling the reference law exactly (0.004 vs 0.174),
which silently inverted every distributional result in the project.

The grid is now fixed by the reference alone and out-of-range model samples are
clipped into the edge bins, so leaving the reference's support costs probability
mass in the wrong place. These tests pin that property.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tica_eval():
    """Import scripts/tica_eval.py without executing its __main__ path."""
    spec = importlib.util.spec_from_file_location(
        "tica_eval_under_test", REPO_ROOT / "scripts" / "tica_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tica_eval():
    return _load_tica_eval()


def _samples(seed=0, n=400):
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(n, 2))
    faithful = rng.normal(size=(n, 2))
    wrong = rng.normal(size=(n, 2)) + 1.5
    return reference, faithful, wrong


def test_faithful_sample_beats_a_shifted_one(tica_eval):
    reference, faithful, wrong = _samples()
    assert tica_eval.hist2d_jsd(reference, faithful) < tica_eval.hist2d_jsd(reference, wrong)


@pytest.mark.parametrize("magnitude", [20, 100, 1000])
@pytest.mark.parametrize("count", [5, 20])
def test_divergence_never_improves_the_score(tica_eval, magnitude, count):
    """The regression that inverted the metric: strays used to lower the JSD."""
    reference, faithful, wrong = _samples()
    rng = np.random.default_rng(99)

    diverged = wrong.copy()
    diverged[:count] = rng.normal(size=(count, 2)) * magnitude

    faithful_score = tica_eval.hist2d_jsd(reference, faithful)
    wrong_score = tica_eval.hist2d_jsd(reference, wrong)
    diverged_score = tica_eval.hist2d_jsd(reference, diverged)

    assert diverged_score > faithful_score, "a diverging model outscored a faithful one"
    assert diverged_score >= wrong_score, "diverging further improved the score"


def test_reference_range_ignores_the_model(tica_eval):
    """A model excursion must not move the grid; that is what inverted the metric."""
    reference, _, _ = _samples()
    tame = tica_eval.reference_range(reference)
    exploded = tica_eval.reference_range(reference)  # same input, model plays no part

    assert tame == exploded
    lo, hi = reference.min(0), reference.max(0)
    assert tame[0][0] < lo[0] and tame[0][1] > hi[0]
    assert tame[1][0] < lo[1] and tame[1][1] > hi[1]


def test_identical_samples_score_near_zero(tica_eval):
    reference, _, _ = _samples()
    assert tica_eval.hist2d_jsd(reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_score_is_symmetric_when_the_grid_is_shared(tica_eval):
    reference, faithful, _ = _samples()
    rng = tica_eval.reference_range(reference)
    forward = tica_eval.hist2d_jsd(reference, faithful, rng=rng)
    backward = tica_eval.hist2d_jsd(faithful, reference, rng=rng)
    assert forward == pytest.approx(backward, rel=1e-12)
