import json
from pathlib import Path

import pytest
import torch

from scripts.adjudicate_source_law_candidate import _sha256, adjudicate


PANEL_SHA = "a" * 64


def _checkpoint(path: Path) -> str:
    torch.save(
        {
            "step": 1000,
            "cfg": {
                "data": {
                    "noise_sigma": 1.5,
                    "unroll": 1,
                    "canon_symmetric": True,
                },
                "model": {
                    "source_noise_v": True,
                    "source_noise_sigma_v": 1.0,
                    "tensor_cloud01": True,
                    "tensor_cloud01_vector_only_attention": False,
                },
            },
        },
        path,
    )
    return _sha256(path)


def _result(path: Path, checkpoint: Path, steps: int, deltas, physical=True) -> None:
    noop_final = [2.0] * 5
    model_final = [base + delta for base, delta in zip(noop_final, deltas)]

    def method(final, bond_mean=3.8, bond_max=4.1):
        rmsd_by_start = [[0.0] * 5 for _ in range(steps)] + [final]
        return {
            "rmsd_by_start": rmsd_by_start,
            "bond_mean": [3.8] * steps + [bond_mean],
            "bond_max": [4.1] * steps + [bond_max],
        }

    row = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": 1000,
        "delta_frames": 1,
        "settings": {
            "domains": 1,
            "starts": 5,
            "steps": steps,
            "methods": "ode_150",
            "seed": 20260718,
            "noise_sigma": None,
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
            "teacher_forced_mean": False,
        },
        "preprocessing": {"canon_symmetric": True},
        "domain_panel": {"sha256": PANEL_SHA, "evaluated_count": 1},
        "domains": [{
            "methods": {
                "noop": method(noop_final),
                "one_step_persistence": method(noop_final),
                "ode_150": method(
                    model_final,
                    bond_mean=3.8 if physical else 3.0,
                    bond_max=4.1 if physical else 6.0,
                ),
            }
        }],
    }
    path.write_text(json.dumps(row))


def test_source_law_h6_then_h20_pass(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    h6 = tmp_path / "h6.json"
    h20 = tmp_path / "h20.json"
    _result(h6, checkpoint, 6, [-0.4, -0.3, -0.2, -0.1, -0.2])
    _result(h20, checkpoint, 20, [-0.5, -0.4, -0.3, -0.2, -0.3])

    interim = adjudicate(h6, None, checkpoint, digest, PANEL_SHA)
    assert interim["status"] == "ADVANCE_SOURCE_LAW_H20"
    assert interim["formal_training_authorized"] is False
    final = adjudicate(h6, h20, checkpoint, digest, PANEL_SHA)
    assert final["status"] == "PASS_SOURCE_LAW_H20"
    assert final["twenty_domain_authorized"] is True
    assert final["formal_training_authorized"] is False


def test_source_law_h20_stops_after_h6_pass(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    h6 = tmp_path / "h6.json"
    h20 = tmp_path / "h20.json"
    _result(h6, checkpoint, 6, [-0.4, -0.3, -0.2, -0.1, -0.2])
    _result(h20, checkpoint, 20, [0.1, 0.2, 0.1, 0.2, 0.1])

    report = adjudicate(h6, h20, checkpoint, digest, PANEL_SHA)
    assert report["status"] == "STOP_SOURCE_LAW_H20"
    assert report["twenty_domain_authorized"] is False
    assert report["formal_training_authorized"] is False


@pytest.mark.parametrize(
    ("deltas", "physical"),
    [
        ([0.1, 0.1, 0.1, 0.1, 0.1], True),
        ([-0.1, -0.1, -0.1, -0.1, -0.1], True),
        ([-0.01, 0.01, -0.01, 0.01, 0.0], True),
        ([-0.4, -0.3, -0.2, -0.1, -0.2], False),
    ],
)
def test_source_law_h6_stops_fail_closed(tmp_path, deltas, physical):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    h6 = tmp_path / "h6.json"
    _result(h6, checkpoint, 6, deltas, physical=physical)

    report = adjudicate(h6, None, checkpoint, digest, PANEL_SHA)
    assert report["status"] == "STOP_SOURCE_LAW_H6"
    assert report["twenty_domain_authorized"] is False


def test_source_law_identity_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["cfg"]["model"]["source_noise_sigma_v"] = 0.1
    torch.save(payload, checkpoint)
    digest = _sha256(checkpoint)
    h6 = tmp_path / "h6.json"
    _result(h6, checkpoint, 6, [-0.4, -0.3, -0.2, -0.1, -0.2])

    with pytest.raises(ValueError, match="source-law identity mismatch"):
        adjudicate(h6, None, checkpoint, digest, PANEL_SHA)


def test_source_law_checkpoint_step_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    _checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload["step"] = 999
    torch.save(payload, checkpoint)
    digest = _sha256(checkpoint)
    h6 = tmp_path / "h6.json"
    _result(h6, checkpoint, 6, [-0.4, -0.3, -0.2, -0.1, -0.2])

    with pytest.raises(ValueError, match="checkpoint must be at step 1000"):
        adjudicate(h6, None, checkpoint, digest, PANEL_SHA)


def test_source_law_panel_count_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    h6 = tmp_path / "h6.json"
    _result(h6, checkpoint, 6, [-0.4, -0.3, -0.2, -0.1, -0.2])
    payload = json.loads(h6.read_text())
    payload["domain_panel"]["evaluated_count"] = 2
    h6.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="domain panel identity mismatch"):
        adjudicate(h6, None, checkpoint, digest, PANEL_SHA)


@pytest.mark.parametrize("field", ["rmsd_by_start", "bond_mean", "bond_max"])
def test_source_law_nonfinite_intermediate_metric_is_rejected(tmp_path, field):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    h6 = tmp_path / "h6.json"
    _result(h6, checkpoint, 6, [-0.4, -0.3, -0.2, -0.1, -0.2])
    payload = json.loads(h6.read_text())
    values = payload["domains"][0]["methods"]["ode_150"][field]
    if field == "rmsd_by_start":
        values[1][0] = float("nan")
    else:
        values[1] = float("nan")
    h6.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="must be finite"):
        adjudicate(h6, None, checkpoint, digest, PANEL_SHA)
