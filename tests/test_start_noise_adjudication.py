import hashlib
import json

import pytest

from scripts.adjudicate_start_noise import adjudicate


DOMAIN_SHA = "a" * 64


def _case(tmp_path, deltas):
    checkpoint = tmp_path / "ckpt_250.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    noop_final = [2.0] * 5
    model_final = [noop + delta for noop, delta in zip(noop_final, deltas)]
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
            "seed": 20260718,
            "integrator": "euler",
            "tau_max": 1.0,
            "drift_anchor": "state",
        },
        "preprocessing": {"canon_symmetric": True},
        "domain_panel": {"sha256": DOMAIN_SHA, "evaluated_count": 1},
        "domains": [{"methods": {
            "noop": {"rmsd_by_start": [[0.0] * 5] * 6 + [noop_final]},
            "mean": {"rmsd_by_start": [[0.0] * 5] * 6 + [model_final]},
        }}],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    return path, checkpoint, checkpoint_sha


@pytest.mark.parametrize(
    ("deltas", "status", "authorized"),
    [
        ([-0.2, -0.2, -0.2, -0.2, -0.2], "ROBUST_ADVANTAGE", True),
        ([0.2, 0.2, 0.2, 0.2, 0.2], "ROBUST_DISADVANTAGE", False),
        ([0.0, 0.0, 0.0, 0.0, 0.0], "NOISE_DOMINATED", False),
        ([-0.3, -0.1, 0.0, 0.1, 0.2], "NOISE_DOMINATED", False),
    ],
)
def test_start_noise_statuses(tmp_path, deltas, status, authorized):
    path, checkpoint, checkpoint_sha = _case(tmp_path, deltas)
    report = adjudicate(path, DOMAIN_SHA, checkpoint, checkpoint_sha)
    assert report["status"] == status
    assert report["second_seed_authorized"] is authorized


def test_start_noise_fails_closed_on_missing_per_start_values(tmp_path):
    path, checkpoint, checkpoint_sha = _case(tmp_path, [-0.1] * 5)
    result = json.loads(path.read_text())
    del result["domains"][0]["methods"]["mean"]["rmsd_by_start"]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="missing mean"):
        adjudicate(path, DOMAIN_SHA, checkpoint, checkpoint_sha)
