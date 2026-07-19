import json
from pathlib import Path

import pytest
import torch

from scripts.adjudicate_endpoint_grid import adjudicate
from scripts.adjudicate_heldout_endpoint_grid import adjudicate as adjudicate_heldout
from scripts.adjudicate_source_law_candidate import _sha256


DOMAIN_SHA = "3" * 64
TEMPERATURES = [320, 348, 379, 413, 450]
REPLICAS = [0, 1, 2, 3, 4]


def _checkpoint(path: Path) -> str:
    torch.save({
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
    }, path)
    return _sha256(path)


def _result(
    path: Path,
    checkpoint: Path,
    *,
    passing: bool,
    nonphysical: bool = False,
    domain: str = "1a0hA01",
    residues_total: int = 89,
    residues_evaluated: int | None = None,
):
    if residues_evaluated is None:
        residues_evaluated = residues_total
    cells = []
    for index, (temperature, replica) in enumerate(
        (temperature, replica)
        for temperature in TEMPERATURES
        for replica in REPLICAS
    ):
        if passing:
            delta = -0.30 + 0.002 * index
        else:
            delta = -0.02 + 0.04 * (index % 2)
        noop = [2.0 + 0.01 * start for start in range(5)]
        model = [value + delta for value in noop]
        cells.append({
            "domain": domain,
            "temperature": temperature,
            "replica": replica,
            "frames": 102,
            "starts": [0, 25, 50, 75, 100],
            "model_rmsd_by_start": model,
            "noop_rmsd_by_start": noop,
            "model_minus_noop_by_start": [delta] * 5,
            "mean_model_minus_noop": delta,
            "bond_mean": 3.8,
            "bond_max": 6.0 if nonphysical and index == 0 else 4.1,
        })
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": 1000,
        "delta_frames": 1,
        "settings": {"starts": 5, "method": "mean", "source_noise": False},
        "domain_panel": {"sha256": DOMAIN_SHA, "ids": [domain]},
        "grid": {"temperatures": TEMPERATURES, "replicas": REPLICAS},
        "preprocessing": {
            "canon_symmetric": True,
            "residues_total": residues_total,
            "residues_evaluated": residues_evaluated,
        },
        "cells": cells,
    }
    path.write_text(json.dumps(payload))


def test_endpoint_grid_passes_only_with_cell_balanced_robust_advantage(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True)

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "PASS_CLEAN_ENDPOINT_GRID"
    assert report["cells"] == 25
    assert report["starts"] == 125
    assert report["robust_endpoint_advantage"] is True
    assert report["formal_training_authorized"] is False


def test_endpoint_grid_stops_a_null_endpoint(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=False)

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "STOP_NULL_ENDPOINT_GRID"
    assert report["robust_endpoint_advantage"] is False


def test_endpoint_grid_rejects_zero_width_advantage_as_non_robust(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True)
    payload = json.loads(result.read_text())
    for cell in payload["cells"]:
        cell["model_rmsd_by_start"] = [value - 0.1 for value in cell["noop_rmsd_by_start"]]
        cell["model_minus_noop_by_start"] = [-0.1] * 5
        cell["mean_model_minus_noop"] = -0.1
    result.write_text(json.dumps(payload))

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "STOP_NULL_ENDPOINT_GRID"
    assert report["standard_error"] == 0.0
    assert report["robust_endpoint_advantage"] is False


def test_endpoint_grid_stops_nonphysical_output_before_skill_claim(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True, nonphysical=True)

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "STOP_NONPHYSICAL_ENDPOINT_GRID"
    assert report["physical_cells"] == 24


def test_endpoint_grid_rejects_missing_or_duplicate_cells(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True)
    payload = json.loads(result.read_text())
    payload["cells"][-1] = payload["cells"][0]
    result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="missing, duplicate, or extra"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)


def test_endpoint_grid_rejects_cropped_residues_or_repeated_starts(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True)
    payload = json.loads(result.read_text())
    payload["preprocessing"]["residues_evaluated"] = 88
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="all 89 residues"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    _result(result, checkpoint, passing=True)
    payload = json.loads(result.read_text())
    payload["cells"][0]["frames"] = 5
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="five distinct starts"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)


def test_endpoint_grid_rejects_corrupted_paired_difference(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True)
    payload = json.loads(result.read_text())
    payload["cells"][0]["model_minus_noop_by_start"][0] += 0.1
    result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="recorded paired RMSD"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)


def test_heldout_endpoint_grid_accepts_full_non89_domain(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(
        result,
        checkpoint,
        passing=True,
        domain="1gxlA02",
        residues_total=86,
    )

    report = adjudicate_heldout(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "PASS_HELDOUT_ENDPOINT_GRID"
    assert report["cells"] == 25
    assert report["starts"] == 125
    assert report["twenty_domain_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_heldout_endpoint_grid_rejects_training_domain_or_crop(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(result, checkpoint, passing=True)
    with pytest.raises(ValueError, match="domain identity mismatch"):
        adjudicate_heldout(result, checkpoint, digest, DOMAIN_SHA)

    _result(
        result,
        checkpoint,
        passing=True,
        domain="1gxlA02",
        residues_total=86,
        residues_evaluated=85,
    )
    with pytest.raises(ValueError, match="requires all 86 residues"):
        adjudicate_heldout(result, checkpoint, digest, DOMAIN_SHA)

    _result(
        result,
        checkpoint,
        passing=True,
        domain="1gxlA02",
        residues_total=87,
    )
    with pytest.raises(ValueError, match="requires all 86 residues"):
        adjudicate_heldout(result, checkpoint, digest, DOMAIN_SHA)


@pytest.mark.parametrize(
    ("passing", "nonphysical", "expected_status"),
    [
        (False, False, "STOP_NULL_HELDOUT_ENDPOINT_GRID"),
        (True, True, "STOP_NONPHYSICAL_HELDOUT_ENDPOINT_GRID"),
    ],
)
def test_heldout_endpoint_grid_stop_statuses(
    tmp_path, passing, nonphysical, expected_status
):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _result(
        result,
        checkpoint,
        passing=passing,
        nonphysical=nonphysical,
        domain="1gxlA02",
        residues_total=86,
    )

    report = adjudicate_heldout(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == expected_status


def test_heldout_endpoint_panel_is_first_frozen_dev_domain():
    heldout = Path("configs/heldout_endpoint_domain_seed0.txt").read_text().splitlines()
    dev = Path("configs/dev_20_length_proportional_seed0.txt").read_text().splitlines()

    assert heldout == [dev[0]]
    assert heldout != ["1a0hA01"]
