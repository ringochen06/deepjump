import json
from pathlib import Path

import pytest
import torch

from scripts.adjudicate_source_law_candidate import _sha256
from scripts.adjudicate_v_mask_projection import adjudicate


PANEL_SHA = "b" * 64


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


def _method(final, steps, *, bond_mean=3.8, bond_max=4.1):
    return {
        "rmsd_by_start": [[0.0] * 5 for _ in range(steps)] + [final],
        "bond_mean": [3.8] * steps + [bond_mean],
        "bond_max": [4.1] * steps + [bond_max],
    }


def _result(
    path: Path,
    checkpoint: Path,
    *,
    steps: int,
    projected: bool,
    model_final,
    physical: bool = True,
) -> None:
    noop = [3.0] * 5
    persistence = [2.8, 2.9, 3.0, 3.1, 3.2]
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
            "steps": steps,
            "methods": "ode_150",
            "seed": 20260718,
            "noise_sigma": None,
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
            "project_v_atom_mask": projected,
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
        "domains": [{
            "domain": "1a0hA01",
            "residues_total": 89,
            "residues_evaluated": 89,
            "frames": 440,
            "starts": [0, 108, 216, 324, 433],
            "methods": {
                "noop": _method(noop, steps),
                "one_step_persistence": _method(persistence, steps),
                "ode_150": _method(
                    model_final,
                    steps,
                    bond_mean=3.8 if physical else 3.0,
                    bond_max=4.1 if physical else 6.0,
                ),
            },
        }],
    }
    path.write_text(json.dumps(payload))


def _pair(tmp_path: Path, checkpoint: Path, *, steps=1, physical=True):
    current = tmp_path / f"current_h{steps}.json"
    masked = tmp_path / f"masked_h{steps}.json"
    _result(
        current,
        checkpoint,
        steps=steps,
        projected=False,
        model_final=[3.4, 3.6, 3.5, 3.8, 3.7],
    )
    _result(
        masked,
        checkpoint,
        steps=steps,
        projected=True,
        model_final=[2.4, 2.5, 2.6, 2.7, 2.8],
        physical=physical,
    )
    return current, masked


def test_v_mask_h1_then_h6_passes_without_authorizing_scaleup(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current_h1, masked_h1 = _pair(tmp_path, checkpoint, steps=1)
    h1 = adjudicate(
        current_h1, masked_h1, checkpoint, digest, PANEL_SHA, steps=1
    )
    assert h1["status"] == "ADVANCE_MASKED_H6"
    assert h1["formal_training_authorized"] is False
    h1_path = tmp_path / "h1_decision.json"
    h1_path.write_text(json.dumps(h1))

    current_h6, masked_h6 = _pair(tmp_path, checkpoint, steps=6)
    h6 = adjudicate(
        current_h6,
        masked_h6,
        checkpoint,
        digest,
        PANEL_SHA,
        steps=6,
        h1_decision_path=h1_path,
    )
    assert h6["status"] == "PASS_MASKED_H6"
    assert h6["twenty_domain_authorized"] is False
    assert h6["second_seed_authorized"] is False
    assert h6["confirmation_authorized"] is False
    assert h6["formal_training_authorized"] is False


def test_v_mask_nonphysical_h1_stops(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current, masked = _pair(tmp_path, checkpoint, physical=False)
    report = adjudicate(current, masked, checkpoint, digest, PANEL_SHA, steps=1)
    assert report["status"] == "STOP_MASKED_H1"
    assert report["physical_through_horizon"] is False


def test_v_mask_zero_width_advantage_stops(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current, masked = _pair(tmp_path, checkpoint)
    payload = json.loads(masked.read_text())
    payload["domains"][0]["methods"]["ode_150"]["rmsd_by_start"][-1] = [2.0] * 5
    masked.write_text(json.dumps(payload))
    report = adjudicate(current, masked, checkpoint, digest, PANEL_SHA, steps=1)
    assert report["status"] == "STOP_MASKED_H1"
    assert report["masked_vs_noop"]["standard_error"] == 0


@pytest.mark.parametrize("baseline", ["noop", "one_step_persistence"])
def test_v_mask_baseline_drift_is_rejected(tmp_path, baseline):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current, masked = _pair(tmp_path, checkpoint)
    payload = json.loads(masked.read_text())
    payload["domains"][0]["methods"][baseline]["rmsd_by_start"][-1][0] += 0.01
    masked.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="baseline mismatch"):
        adjudicate(current, masked, checkpoint, digest, PANEL_SHA, steps=1)


def test_v_mask_nonprojection_setting_drift_is_rejected(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current, masked = _pair(tmp_path, checkpoint)
    payload = json.loads(masked.read_text())
    payload["settings"]["seed"] += 1
    masked.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="settings mismatch"):
        adjudicate(current, masked, checkpoint, digest, PANEL_SHA, steps=1)


def test_v_mask_h6_requires_matching_advance_decision(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current, masked = _pair(tmp_path, checkpoint, steps=6)
    with pytest.raises(ValueError, match="requires the H1 decision"):
        adjudicate(current, masked, checkpoint, digest, PANEL_SHA, steps=6)

    h1_path = tmp_path / "h1.json"
    h1_path.write_text(json.dumps({"status": "STOP_MASKED_H1", "checkpoint_sha256": digest}))
    with pytest.raises(ValueError, match="requires ADVANCE_MASKED_H6"):
        adjudicate(
            current,
            masked,
            checkpoint,
            digest,
            PANEL_SHA,
            steps=6,
            h1_decision_path=h1_path,
        )


def test_v_mask_h6_rejects_h1_domain_identity_drift(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    digest = _checkpoint(checkpoint)
    current, masked = _pair(tmp_path, checkpoint, steps=6)
    h1_path = tmp_path / "h1.json"
    h1_path.write_text(
        json.dumps(
            {
                "status": "ADVANCE_MASKED_H6",
                "steps": 1,
                "checkpoint_sha256": digest,
                "domain_list_sha256": "wrong",
                "twenty_domain_authorized": False,
                "second_seed_authorized": False,
                "confirmation_authorized": False,
                "formal_training_authorized": False,
            }
        )
    )
    with pytest.raises(ValueError, match="H1 domain list identity mismatch"):
        adjudicate(
            current,
            masked,
            checkpoint,
            digest,
            PANEL_SHA,
            steps=6,
            h1_decision_path=h1_path,
        )
