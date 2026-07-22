import json
import statistics
from pathlib import Path

import pytest
import torch

from scripts.adjudicate_endpoint_panel import EXPECTED_DOMAIN_IDS, adjudicate
from scripts.adjudicate_source_law_candidate import _sha256
from scripts.endpoint_panel_eval import _panel_starts, _runtime_probe_status


DOMAIN_SHA = "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
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
                "domains": ["1a0hA01"],
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


def _write_result(
    path: Path,
    checkpoint: Path,
    domain_deltas: list[float],
    *,
    nonphysical: bool = False,
) -> None:
    domains = []
    for domain_index, (domain_id, domain_delta) in enumerate(
        zip(EXPECTED_DOMAIN_IDS, domain_deltas)
    ):
        cells = []
        for cell_index, (temperature, replica) in enumerate(
            (temperature, replica)
            for temperature in TEMPERATURES
            for replica in REPLICAS
        ):
            delta = domain_delta + 0.0001 * (cell_index - 12)
            noop = [2.0 + 0.01 * cell_index + 0.001 * start for start in range(3)]
            model = [value + delta for value in noop]
            cells.append({
                "domain": domain_id,
                "temperature": temperature,
                "replica": replica,
                "frames": 102,
                "starts": [0, 50, 100],
                "model_rmsd_by_start": model,
                "noop_rmsd_by_start": noop,
                "model_minus_noop_by_start": [delta] * 3,
                "mean_model_minus_noop": delta,
                "bond_mean": 3.8,
                "bond_max": 6.0 if nonphysical and domain_index == cell_index == 0 else 4.1,
            })
        cell_deltas = [cell["mean_model_minus_noop"] for cell in cells]
        domains.append({
            "domain": domain_id,
            "preprocessing": {
                "canon_symmetric": True,
                "residues_total": 80 + domain_index,
                "residues_evaluated": 80 + domain_index,
            },
            "summary": {
                "cells": 25,
                "mean_model_minus_noop": statistics.fmean(cell_deltas),
                "cells_better_than_noop": sum(value < 0 for value in cell_deltas),
            },
            "cells": cells,
        })
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": 1000,
        "delta_frames": 1,
        "settings": {
            "starts": 3,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
        },
        "domain_panel": {"sha256": DOMAIN_SHA, "ids": list(EXPECTED_DOMAIN_IDS)},
        "grid": {"temperatures": TEMPERATURES, "replicas": REPLICAS},
        "runtime_probe": {
            "status": "PASS_RUNTIME_PROBE",
            "domain": EXPECTED_DOMAIN_IDS[-1],
            "residues": 300,
            "batch_size": 3,
            "cell_seconds": 1.0,
            "projected_500_cell_minutes": 8.333333333333334,
            "peak_memory_bytes": 8_000_000_000,
            "total_memory_bytes": 16_000_000_000,
            "peak_memory_fraction": 0.5,
            "limits": {
                "max_peak_memory_fraction": 0.8,
                "max_projected_minutes": 50.0,
            },
        },
        "domains": domains,
    }
    path.write_text(json.dumps(payload))


def test_panel_starts_span_the_valid_source_frames():
    assert _panel_starts(102, 1) == [0, 50, 100]
    assert _panel_starts(101, 1) == [0, 49, 99]
    with pytest.raises(ValueError, match="no valid H1 pair"):
        _panel_starts(1, 1)
    with pytest.raises(ValueError, match="three distinct"):
        _panel_starts(2, 1)


def test_runtime_probe_limits_are_fail_closed():
    assert _runtime_probe_status(0.8, 50.0) == "PASS_RUNTIME_PROBE"
    assert _runtime_probe_status(0.8001, 1.0) == "STOP_MEMORY_HEADROOM"
    assert _runtime_probe_status(0.5, 50.1) == "STOP_PROJECTED_RUNTIME"
    with pytest.raises(ValueError, match="finite and non-negative"):
        _runtime_probe_status(float("nan"), 1.0)


def test_endpoint_panel_passes_only_at_domain_level(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [10.0] + [-0.20 + 0.005 * index for index in range(19)])

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "PASS_DEV20_ENDPOINT"
    assert report["domains"] == 20
    assert report["cells"] == 500
    assert report["starts"] == 1500
    assert report["observed_anchor"]["included_in_primary_inference"] is False
    assert report["primary_domains_better_than_noop"] == 19
    assert report["primary"]["ci95_model_minus_noop"][1] < 0
    assert all(item["passes_negative"] for item in report["leave_one_domain_out"])
    assert report["second_seed_authorized"] is False
    assert report["untouched_confirmation_authorized"] is False
    assert report["recursive_evaluation_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_endpoint_panel_zero_variance_is_inconclusive(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [10.0] + [-0.1] * 19)

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "INCONCLUSIVE_ZERO_VARIANCE_DEV20_ENDPOINT"
    assert report["primary"]["standard_error"] == pytest.approx(0.0, abs=1e-15)


def test_endpoint_panel_stops_clear_disadvantage(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-10.0] + [0.10 + 0.005 * index for index in range(19)])

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "STOP_DOMAIN_DISADVANTAGE_DEV20_ENDPOINT"
    assert report["primary_domains_better_than_noop"] == 0


def test_endpoint_panel_keeps_mixed_effect_inconclusive(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(
        result,
        checkpoint,
        [10.0] + [-0.01] * 14 + [0.05] * 5,
    )

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "INCONCLUSIVE_DEV20_ENDPOINT"


def test_endpoint_panel_jackknife_catches_a_fragile_full_panel_pass(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [10.0] + [-0.01] * 14 + [0.01] * 5)

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["primary"]["ci95_model_minus_noop"][1] < 0
    assert report["primary_domains_better_than_noop"] == 14
    assert any(not item["passes_negative"] for item in report["leave_one_domain_out"])
    assert report["status"] == "INCONCLUSIVE_DEV20_ENDPOINT"


def test_endpoint_panel_stops_an_unresolved_null(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [10.0] + [-0.1] * 9 + [0.1] * 10)

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["primary"]["ci95_model_minus_noop"][0] < 0
    assert report["primary"]["ci95_model_minus_noop"][1] > 0
    assert report["status"] == "STOP_NULL_DEV20_ENDPOINT"


def test_endpoint_panel_stops_any_nonphysical_cell(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(
        result,
        checkpoint,
        [-0.20 + 0.005 * index for index in range(20)],
        nonphysical=True,
    )

    report = adjudicate(result, checkpoint, digest, DOMAIN_SHA)

    assert report["status"] == "STOP_NONPHYSICAL_DEV20_ENDPOINT"
    assert report["physical_cells"] == 499


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["domain_panel"]["ids"].reverse(), "identity or order"),
        (
            lambda payload: payload["domains"][0]["preprocessing"].update(
                residues_evaluated=79
            ),
            "all residues",
        ),
        (
            lambda payload: payload["domains"][0]["cells"][0].update(starts=[0, 1, 2]),
            "start panel",
        ),
        (
            lambda payload: payload["domains"][0]["cells"][0][
                "model_minus_noop_by_start"
            ].__setitem__(0, 1.0),
            "recorded paired RMSD",
        ),
    ],
)
def test_endpoint_panel_rejects_identity_crop_start_or_pair_corruption(
    tmp_path, mutation, match
):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-0.20 + 0.005 * index for index in range(20)])
    payload = json.loads(result.read_text())
    mutation(payload)
    result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)


def test_endpoint_panel_matches_the_frozen_dev_list():
    expected = Path("configs/dev_20_length_proportional_seed0.txt").read_text().splitlines()
    assert expected == list(EXPECTED_DOMAIN_IDS)
    assert "1a0hA01" not in expected


def test_endpoint_panel_rejects_checkpoint_with_nonmatching_training_domain(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    _checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["cfg"]["data"]["domains"] = ["1gxlA02"]
    torch.save(payload, checkpoint)
    digest = _sha256(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-0.20 + 0.005 * index for index in range(20)])

    with pytest.raises(ValueError, match="training domain mismatch"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda probe: probe.update(status="STOP_MEMORY_HEADROOM"),
        lambda probe: probe.update(peak_memory_fraction=0.81),
        lambda probe: probe.update(projected_500_cell_minutes=50.1),
        lambda probe: probe.update(batch_size=1),
        lambda probe: probe.update(peak_memory_bytes=7_000_000_000),
        lambda probe: probe.update(cell_seconds=2.0),
    ],
)
def test_endpoint_panel_rejects_failed_or_inconsistent_runtime_probe(
    tmp_path, mutation
):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-0.20 + 0.005 * index for index in range(20)])
    payload = json.loads(result.read_text())
    mutation(payload["runtime_probe"])
    result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="runtime probe"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)


def test_endpoint_panel_adjudicator_rejects_repeated_short_trajectory_starts(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-0.20 + 0.005 * index for index in range(20)])
    payload = json.loads(result.read_text())
    cell = payload["domains"][0]["cells"][0]
    cell["frames"] = 3
    cell["starts"] = [0, 0, 1]
    result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="three distinct starts"):
        adjudicate(result, checkpoint, digest, DOMAIN_SHA)
