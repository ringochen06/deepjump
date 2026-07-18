import hashlib
import json

import pytest

from scripts.adjudicate_feedback_adaptation import adjudicate


DOMAIN_SHA = "a" * 64


def _series(rmsd, bond_max=None):
    return {
        "rmsd": [0.0, *rmsd],
        "bond_mean": [3.8] * 7,
        "bond_max": [4.0, *(bond_max or [4.2] * 6)],
    }


def _case(tmp_path, *, auto_rmsd=None, auto_bond_max=None, teacher_rmsd=None):
    checkpoint = tmp_path / "ckpt_250.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": 250,
        "delta_frames": 1,
        "settings": {
            "domains": 1,
            "starts": 5,
            "steps": 6,
            "methods": "mean",
            "teacher_forced_mean": True,
            "integrator": "euler",
            "tau_max": 1.0,
            "drift_anchor": "state",
        },
        "preprocessing": {"canon_symmetric": True},
        "domain_panel": {"sha256": DOMAIN_SHA, "evaluated_count": 1},
        "domains": [{"methods": {
            "noop": _series([3.0] * 6),
            "one_step_persistence": _series([2.0] * 6),
            "mean": _series(auto_rmsd or [1.5] * 6, auto_bond_max),
            "teacher_forced_mean": _series(teacher_rmsd or [1.5] * 6),
        }}],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    return path, checkpoint, checkpoint_sha


def test_feedback_adaptation_passes_only_full_rule(tmp_path):
    path, checkpoint, checkpoint_sha = _case(tmp_path)
    report = adjudicate(path, DOMAIN_SHA, checkpoint, checkpoint_sha)
    assert report["status"] == "PASS_FEEDBACK_ADAPTATION"
    assert report["teacher_forced_steps_better_than_one_step_persistence"] == 6


def test_feedback_adaptation_fails_early_nonphysical_rollout(tmp_path):
    path, checkpoint, checkpoint_sha = _case(
        tmp_path, auto_bond_max=[4.2, 4.3, 4.4, 8.0, 20.0, 40.0]
    )
    report = adjudicate(path, DOMAIN_SHA, checkpoint, checkpoint_sha)
    assert report["status"] == "FAIL_FEEDBACK_ADAPTATION"
    assert report["autoregressive_first_nonphysical_step"] == 4


def test_feedback_adaptation_fails_closed_on_checkpoint_identity(tmp_path):
    path, checkpoint, checkpoint_sha = _case(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        adjudicate(path, DOMAIN_SHA, checkpoint, "b" * 64)
    with pytest.raises(ValueError, match="path"):
        adjudicate(path, DOMAIN_SHA, tmp_path / "other.pt", checkpoint_sha)
