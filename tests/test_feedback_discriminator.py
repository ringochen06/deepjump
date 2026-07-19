import json

import pytest

from scripts.adjudicate_feedback_discriminator import adjudicate


DOMAIN_SHA = "a" * 64


def _series(rmsd, bond_mean=None, bond_max=None):
    return {
        "rmsd": [0.0, *rmsd],
        "bond_mean": [3.8, *(bond_mean or [3.8] * 6)],
        "bond_max": [4.0, *(bond_max or [4.2] * 6)],
    }


def _write_case(tmp_path, teacher_rmsd, *, teacher_bond_max=None, auto_bond_max=None):
    result = {
        "checkpoint_step": 5000,
        "delta_frames": 1,
        "settings": {
            "domains": 1,
            "starts": 5,
            "steps": 6,
            "methods": "mean",
            "teacher_forced_mean": True,
        },
        "preprocessing": {"canon_symmetric": True},
        "domain_panel": {"sha256": DOMAIN_SHA, "evaluated_count": 1},
        "domains": [{
            "methods": {
                "noop": _series([2.0] * 6),
                "mean": _series(
                    [1.5, 1.8, 2.2, 4.0, 7.0, 10.0],
                    bond_max=auto_bond_max or [4.2, 4.3, 4.4, 7.0, 20.0, 40.0],
                ),
                "teacher_forced_mean": _series(
                    teacher_rmsd,
                    bond_max=teacher_bond_max,
                ),
            }
        }],
    }
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(result))
    return path


def test_feedback_distribution_shift_requires_physical_teacher_and_four_wins(tmp_path):
    path = _write_case(tmp_path, [1.0, 1.1, 1.2, 1.3, 2.1, 2.2])
    report = adjudicate(path, DOMAIN_SHA)
    assert report["status"] == "FEEDBACK_DISTRIBUTION_SHIFT"
    assert report["autoregressive_first_nonphysical_step"] == 4
    assert report["teacher_forced_first_nonphysical_step"] is None
    assert report["teacher_forced_steps_better_than_noop"] == 4


def test_endpoint_failure_when_teacher_forced_geometry_breaks_by_h4(tmp_path):
    path = _write_case(
        tmp_path,
        [1.0] * 6,
        teacher_bond_max=[4.2, 4.3, 6.0, 7.0, 8.0, 9.0],
    )
    report = adjudicate(path, DOMAIN_SHA)
    assert report["status"] == "ENDPOINT_OPERATOR_FAILURE"
    assert report["teacher_forced_first_nonphysical_step"] == 3


def test_endpoint_failure_when_teacher_forced_rarely_beats_noop(tmp_path):
    path = _write_case(tmp_path, [1.9, 2.1, 2.2, 2.3, 2.4, 2.5])
    report = adjudicate(path, DOMAIN_SHA)
    assert report["status"] == "ENDPOINT_OPERATOR_FAILURE"
    assert report["teacher_forced_steps_better_than_noop"] == 1


def test_intermediate_outcome_remains_inconclusive(tmp_path):
    path = _write_case(tmp_path, [1.0, 1.1, 1.2, 2.1, 2.2, 2.3])
    report = adjudicate(path, DOMAIN_SHA)
    assert report["status"] == "INCONCLUSIVE_FEEDBACK_DISCRIMINATOR"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_step", 4500, "checkpoint step"),
        ("delta_frames", 10, "delta=1"),
    ],
)
def test_identity_mismatches_fail_closed(tmp_path, field, value, message):
    path = _write_case(tmp_path, [1.0] * 6)
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match=message):
        adjudicate(path, DOMAIN_SHA)


def test_preprocessing_and_panel_identity_fail_closed(tmp_path):
    path = _write_case(tmp_path, [1.0] * 6)
    result = json.loads(path.read_text())
    result["preprocessing"]["canon_symmetric"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="canonical symmetric"):
        adjudicate(path, DOMAIN_SHA)
    with pytest.raises(ValueError, match="SHA256"):
        adjudicate(_write_case(tmp_path, [1.0] * 6), "b" * 64)
