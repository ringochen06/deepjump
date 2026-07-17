import copy
import json
import sys

import pytest

from scripts.validate_full_grid_evaluation import REPLICAS, TEMPERATURES
from scripts.validate_scientific_evaluation import (
    main,
    validate_scientific_geometry,
    validate_scientific_transition,
)


CHECKPOINT = "/data/checkpoint.pt"
PANEL_SHA256 = "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
DOMAINS = 20


def _trajectories(*, transition: bool):
    cells = []
    for temperature in TEMPERATURES:
        for replica in REPLICAS:
            cell = {"temperature": temperature, "replica": replica, "metric": 0.1}
            if transition:
                cell["reference_replica"] = REPLICAS[(replica + 1) % len(REPLICAS)]
            cells.append(cell)
    return cells


def _base_result(*, transition: bool, steps: int | None = None):
    trajectories = _trajectories(transition=transition)
    return {
        "checkpoint": CHECKPOINT,
        "checkpoint_step": 1000,
        "delta_frames": 1,
        "domain_panel": {
            "sha256": PANEL_SHA256,
            "evaluated_count": DOMAINS,
        },
        "trajectory_grid": {
            "formal_full_grid": True,
            "temperatures": TEMPERATURES,
            "replicas": REPLICAS,
            "required_cells_per_domain": 25,
        },
        "settings": {} if steps is None else {"steps": steps},
        "domains": [
            {
                "domain": f"domain{index}",
                "grid": {"cells": 25},
                "trajectories": copy.deepcopy(trajectories),
            }
            for index in range(DOMAINS)
        ],
    }


def _gain():
    return {
        "mean_baseline_minus_model": 0.2,
        "ci95": [0.1, 0.3],
        "domains": DOMAINS,
        "passes": True,
    }


def _transition_result():
    result = _base_result(transition=True)
    result["summary"] = {
        "noop": {"mean_energy_score": 1.0},
        "mean": {
            "mean_energy_score": 0.8,
            "paired_energy_score_gain": _gain(),
            "paired_msm_row_jsd_gain": _gain(),
        },
    }
    return result


def _geometry_result(steps: int):
    result = _base_result(transition=False, steps=steps)
    statistic = {
        "mean": -0.1,
        "one_sided_upper": -0.01,
        "alpha": 0.05,
        "domains": DOMAINS,
        "passes": True,
    }
    result["summary"] = {
        "noop": {"passes": True},
        "mean": {
            "domain_count": DOMAINS,
            "domains_all_cells_all_steps_pass": DOMAINS,
            "domain_mean_worst_excess": {"bond_mean": statistic},
            "hard_envelope_pass": True,
            "passes": True,
        },
    }
    return result


def _common():
    return {
        "expected_checkpoint": CHECKPOINT,
        "expected_step": 1000,
        "expected_domains": DOMAINS,
        "expected_delta": 1,
        "expected_domain_list_sha256": PANEL_SHA256,
    }


def test_scientific_gate_passes_complete_positive_results():
    validate_scientific_transition(_transition_result(), **_common())
    for steps in (20, 100):
        validate_scientific_geometry(
            _geometry_result(steps), **_common(), expected_rollout_steps=steps
        )


def test_transition_nonpositive_ci_fails_closed():
    result = _transition_result()
    result["summary"]["mean"]["paired_energy_score_gain"]["ci95"][0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_scientific_transition(result, **_common())


def test_transition_wrong_paired_domain_count_fails_closed():
    result = _transition_result()
    result["summary"]["mean"]["paired_energy_score_gain"]["domains"] = 1
    with pytest.raises(ValueError, match="expected number of domains"):
        validate_scientific_transition(result, **_common())


def test_transition_wrong_panel_or_delta_fails_closed():
    result = _transition_result()
    result["domain_panel"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="domain-list SHA256"):
        validate_scientific_transition(result, **_common())
    result = _transition_result()
    result["delta_frames"] = 10
    with pytest.raises(ValueError, match="delta_frames"):
        validate_scientific_transition(result, **_common())


def test_geometry_any_hard_failure_fails_closed():
    result = _geometry_result(100)
    result["summary"]["mean"]["domains_all_cells_all_steps_pass"] = DOMAINS - 1
    result["summary"]["mean"]["hard_envelope_pass"] = False
    result["summary"]["mean"]["passes"] = False
    with pytest.raises(ValueError, match="domain/cell/step"):
        validate_scientific_geometry(
            result, **_common(), expected_rollout_steps=100
        )


def test_geometry_positive_domain_upper_bound_fails_closed():
    result = _geometry_result(20)
    statistic = result["summary"]["mean"]["domain_mean_worst_excess"]["bond_mean"]
    statistic["one_sided_upper"] = 0.01
    statistic["passes"] = False
    with pytest.raises(ValueError, match="exceeds"):
        validate_scientific_geometry(
            result, **_common(), expected_rollout_steps=20
        )


def test_cli_persists_scientific_failure_report(tmp_path, monkeypatch):
    transition = _transition_result()
    transition["summary"]["mean"]["paired_energy_score_gain"]["ci95"][0] = -0.01
    paths = {
        "transition": tmp_path / "transition.json",
        "geometry_20": tmp_path / "geometry20.json",
        "geometry_100": tmp_path / "geometry100.json",
    }
    paths["transition"].write_text(json.dumps(transition))
    paths["geometry_20"].write_text(json.dumps(_geometry_result(20)))
    paths["geometry_100"].write_text(json.dumps(_geometry_result(100)))
    output = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", [
        "validate_scientific_evaluation.py",
        "--transition", str(paths["transition"]),
        "--geometry-20", str(paths["geometry_20"]),
        "--geometry-100", str(paths["geometry_100"]),
        "--expected-checkpoint", CHECKPOINT,
        "--expected-step", "1000",
        "--expected-domains", str(DOMAINS),
        "--expected-delta", "1",
        "--expected-domain-list-sha256", PANEL_SHA256,
        "--output", str(output),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL"
    assert "strictly positive" in report["errors"][0]
