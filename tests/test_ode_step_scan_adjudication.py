import json
from pathlib import Path

import pytest
import torch

from scripts.adjudicate_ode_step_scan import EXPECTED_METHODS, adjudicate
from scripts.adjudicate_source_law_candidate import _sha256


PANEL_SHA = "c" * 64
NOOP = [3.0, 3.2, 2.9, 3.1, 2.8]
PASSING = [2.4, 2.7, 2.3, 2.5, 2.2]
FAILING = [3.2, 3.4, 3.1, 3.3, 3.0]


def _checkpoint(path: Path) -> str:
    torch.save(
        {
            "step": 1000,
            "cfg": {
                "data": {"noise_sigma": 1.5, "unroll": 1, "canon_symmetric": True},
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


def _method(final, *, bond_mean=3.8, bond_max=4.1):
    return {
        "rmsd_by_start": [[0.0] * 5, final],
        "bond_mean": [3.8, bond_mean],
        "bond_max": [4.1, bond_max],
    }


def _result(path: Path, checkpoint: Path, method: str, final) -> None:
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": 1000,
        "delta_frames": 1,
        "settings": {
            "ckpt": str(checkpoint),
            "domain_list": "panel.txt",
            "domain_list_sha256": PANEL_SHA,
            "domains": 1,
            "starts": 5,
            "steps": 1,
            "methods": method,
            "seed": 20260718,
            "noise_sigma": None,
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
            "project_v_atom_mask": False,
            "teacher_forced_mean": False,
            "output": str(path),
        },
        "preprocessing": {"canon_symmetric": True},
        "domain_panel": {
            "path": "panel.txt",
            "sha256": PANEL_SHA,
            "count": 1,
            "evaluated_count": 1,
        },
        "domains": [
            {
                "domain": "1a0hA01",
                "residues_total": 89,
                "residues_evaluated": 89,
                "frames": 440,
                "starts": [0, 108, 216, 324, 433],
                "methods": {
                    "noop": _method(NOOP),
                    "one_step_persistence": _method([2.8, 2.9, 3.0, 3.1, 3.2]),
                    method: _method(final),
                },
            }
        ],
    }
    path.write_text(json.dumps(payload))


def _suite(tmp_path: Path, checkpoint: Path):
    paths = {}
    for method in EXPECTED_METHODS:
        path = tmp_path / f"{method}.json"
        _result(path, checkpoint, method, PASSING)
        paths[method] = path
    reference = tmp_path / "reference_ode150.json"
    _result(reference, checkpoint, "ode_150", PASSING)
    return paths, reference


def test_ode1_failure_stops_before_internal_feedback_claim(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    paths, reference = _suite(tmp_path, checkpoint)
    _result(paths["ode_1"], checkpoint, "ode_1", FAILING)

    report = adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)

    assert report["status"] == "STOP_SOURCE_ENDPOINT_H1"
    assert report["first_failed_ode_method"] == "ode_1"
    assert report["metrics"]["ode_1"]["passes"] is False
    assert report["metrics"]["mean"]["passes"] is True
    assert report["formal_training_authorized"] is False


def test_first_later_failure_is_reported_as_internal_feedback(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    paths, reference = _suite(tmp_path, checkpoint)
    _result(paths["ode_5"], checkpoint, "ode_5", FAILING)
    _result(paths["ode_20"], checkpoint, "ode_20", FAILING)

    report = adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)

    assert report["status"] == "STOP_INTERNAL_ODE_FEEDBACK"
    assert report["first_failed_ode_method"] == "ode_5"
    assert report["metrics"]["ode_1"]["passes"] is True


def test_all_ode_step_counts_pass_without_authorizing_scaleup(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    paths, reference = _suite(tmp_path, checkpoint)

    report = adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)

    assert report["status"] == "PASS_ODE_STEP_SCAN"
    assert report["first_failed_ode_method"] is None
    assert report["ode150_reproduced"] is True
    assert report["twenty_domain_authorized"] is False
    assert report["second_seed_authorized"] is False
    assert report["confirmation_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_ode150_must_reproduce_frozen_method_payload(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    paths, reference = _suite(tmp_path, checkpoint)
    payload = json.loads(reference.read_text())
    payload["domains"][0]["methods"]["ode_150"]["rmsd_by_start"][-1][0] += 0.01
    reference.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="did not reproduce"):
        adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)


def test_baseline_or_setting_drift_is_rejected(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    paths, reference = _suite(tmp_path, checkpoint)
    payload = json.loads(paths["ode_10"].read_text())
    payload["domains"][0]["methods"]["noop"]["rmsd_by_start"][-1][0] += 0.01
    paths["ode_10"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="baseline mismatch"):
        adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)

    _result(paths["ode_10"], checkpoint, "ode_10", PASSING)
    payload = json.loads(paths["ode_10"].read_text())
    payload["settings"]["seed"] += 1
    paths["ode_10"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="settings mismatch"):
        adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)


def test_missing_method_and_nonfinite_result_fail_closed(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    paths, reference = _suite(tmp_path, checkpoint)
    missing = dict(paths)
    missing.pop("ode_75")
    with pytest.raises(ValueError, match="exactly the preregistered methods"):
        adjudicate(missing, reference, checkpoint, digest, PANEL_SHA)

    payload = json.loads(paths["ode_75"].read_text())
    payload["domains"][0]["methods"]["ode_75"]["bond_max"][-1] = float("nan")
    paths["ode_75"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be finite"):
        adjudicate(paths, reference, checkpoint, digest, PANEL_SHA)
