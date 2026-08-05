import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import adjudicate_contracted_scientific_bundle as adjudicator
from scripts import contracted_scientific_bundle_eval as evaluator


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (ROOT / "configs" / "contracted_scientific_bundle_protocol_v1.json").read_text()
    )


def _session(*, delta=1, oracle=evaluator.ORACLE_UNRESOLVED):
    return {
        "phase": "development",
        "checkpoint_delta": delta,
        "msm_oracle_status": oracle,
        "protocol_sha256": "1" * 64,
        "sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "checkpoint_step": 1000,
        "full_training_contract_sha256": "4" * 64,
        "panel_name": "legacy_dev20",
        "panel_sha256": "5" * 64,
        "panel_domains": 20,
        "state_archive_output": "/tmp/scientific-state-archive.npz",
    }


def _raw(session):
    return {
        "schema": evaluator.RAW_EVIDENCE_SCHEMA,
        "status": evaluator.EVALUATOR_STATUS,
        "numerical_kernel_implemented": True,
        "protocol_sha256": session["protocol_sha256"],
        "session_sha256": session["sha256"],
        "prerequisite_sha256": "6" * 64,
        "identity": {
            "checkpoint_sha256": session["checkpoint_sha256"],
            "checkpoint_step": session["checkpoint_step"],
            "full_training_contract_sha256": session["full_training_contract_sha256"],
            "panel_name": session["panel_name"],
            "panel_sha256": session["panel_sha256"],
            "panel_domains": session["panel_domains"],
        },
        "payload_verification": {
            "status": "PASS_CONTRACTED_PANEL_LIVE_PAYLOAD_REHASH",
            "panel_domains": session["panel_domains"],
        },
        "data_prerequisite_status": adjudicator.DATA_PASS,
        "oracle_artifact": None,
        "runtime_feasibility": {"status": evaluator.RUNTIME_FEASIBILITY_PASS},
        "global_obs_claim": {"required": False, "completed": False},
        "state_archive": {
            "schema": evaluator.STATE_ARCHIVE_SCHEMA,
            "path": session["state_archive_output"],
            "sha256": "8" * 64,
            "arrays": 1,
        },
        "consumption_claim": {"sha256": "7" * 64},
        "conditional_raw": {"domains": []},
        "geometry_raw": {"domains": []},
        "scientific_adjudication_performed": False,
        "formal_training_authorized": False,
    }


def test_delta1_missing_oracle_is_inconclusive_not_pass_or_fail():
    decision = adjudicator.delta1_oracle_disposition(_session())
    assert decision["status"] == adjudicator.INCONCLUSIVE_ORACLE
    assert "PASS" not in decision["status"]
    assert "FAIL" not in decision["status"]
    assert decision["formal_training_authorized"] is False


def test_resolved_delta1_and_other_delta_are_only_ready_for_recomputation():
    assert adjudicator.delta1_oracle_disposition(
        _session(oracle=evaluator.ORACLE_PASS)
    )["status"] == adjudicator.READY_FOR_RECOMPUTATION
    assert adjudicator.delta1_oracle_disposition(
        _session(delta=10)
    )["status"] == adjudicator.READY_FOR_RECOMPUTATION


def test_missing_or_incomplete_data_evidence_cannot_become_model_failure():
    session = _session()
    raw = _raw(session)
    raw["data_prerequisite_status"] = evaluator.DATA_PREREQUISITE_FAILURE
    with pytest.raises(ValueError, match=evaluator.DATA_PREREQUISITE_FAILURE):
        adjudicator.validate_raw_evidence(raw, session)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.pop("geometry_raw"),
        lambda raw: raw.__setitem__("numerical_kernel_implemented", False),
        lambda raw: raw.__setitem__("scientific_adjudication_performed", True),
        lambda raw: raw.__setitem__("formal_training_authorized", True),
        lambda raw: raw["payload_verification"].__setitem__("panel_domains", 19),
    ],
)
def test_incomplete_or_self_adjudicated_raw_evidence_is_rejected(mutation):
    session = _session()
    raw = _raw(session)
    mutation(raw)
    with pytest.raises(ValueError):
        adjudicator.validate_raw_evidence(raw, session)


def test_complete_envelope_is_only_schema_valid_not_a_scientific_pass():
    session = _session()
    raw = _raw(session)
    with pytest.raises(ValueError, match="domain count"):
        adjudicator.validate_raw_evidence(copy.deepcopy(raw), session)
    report = adjudicator.implementation_report()
    assert report["independent_numerical_recomputation_implemented"] is True
    assert report["formal_training_authorized"] is False


def _conditional_cell():
    protocol = _protocol()
    starts = protocol["conditional_transition"]["starts_per_cell"]
    draws = protocol["conditional_transition"]["draws_per_start"]
    x = np.linspace(-2.0, 2.0, 80)
    real = np.stack([x, x**2, np.sin(x), np.cos(x)], axis=1)
    source = real[[10, 35, 60]]
    target = real[[11, 36, 61]]
    raw = np.repeat(source[:, None, :], draws, axis=1) + 0.01
    observation = {
        "position_finite": True,
        "vector_finite": True,
        "bond_mean": 3.8,
        "bond_max": 4.1,
    }
    source_hashes = [hashlib.sha256(f"source-{i}".encode()).hexdigest() for i in range(starts)]
    raw_hashes = [
        [hashlib.sha256(f"raw-{i}-{j}".encode()).hexdigest() for j in range(draws)]
        for i in range(starts)
    ]
    return {
        "delta_frames": 1,
        "draws": draws,
        "tica_components": 4,
        "start_frames": [0, 10, 20],
        "real_tic": real.tolist(),
        "source_tic": source.tolist(),
        "target_tic": target.tolist(),
        "raw_predicted_tic": raw.tolist(),
        "guarded_predicted_tic": raw.tolist(),
        "accepted": [[True] * draws for _ in range(starts)],
        "source_state_sha256": source_hashes,
        "raw_state_sha256": raw_hashes,
        "selected_state_sha256": copy.deepcopy(raw_hashes),
        "source_guard_observations": [copy.deepcopy(observation) for _ in range(starts)],
        "raw_guard_observations": [
            [copy.deepcopy(observation) for _ in range(draws)] for _ in range(starts)
        ],
    }


def test_conditional_metrics_are_recomputed_from_draw_level_tic_evidence():
    result = adjudicator.recompute_conditional_cell(_conditional_cell(), _protocol(), seed=7)
    assert set(result) == {"noop", "raw", "guarded"}
    assert result["raw"] == result["guarded"]
    assert all(np.isfinite(value) for row in result.values() for value in row.values())


def test_conditional_guard_flag_and_exact_source_are_recomputed():
    cell = _conditional_cell()
    cell["raw_guard_observations"][0][0]["bond_max"] = 9.0
    with pytest.raises(ValueError, match="accepted flag"):
        adjudicator.recompute_conditional_cell(cell, _protocol(), seed=7)


def test_conditional_reject_branch_detects_selected_vector_state_tamper():
    cell = _conditional_cell()
    cell["raw_guard_observations"][0][0]["vector_finite"] = False
    cell["accepted"][0][0] = False
    cell["guarded_predicted_tic"][0][0] = cell["source_tic"][0]
    cell["selected_state_sha256"][0][0] = cell["raw_state_sha256"][0][0]
    with pytest.raises(ValueError, match="selected P/V"):
        adjudicator.recompute_conditional_cell(cell, _protocol(), seed=7)

    cell = _conditional_cell()
    cell["accepted"][0][0] = False
    cell["guarded_predicted_tic"][0][0] = cell["source_tic"][0]
    with pytest.raises(ValueError, match="accepted flag"):
        adjudicator.recompute_conditional_cell(cell, _protocol(), seed=7)


def test_canonical_conditional_archive_recomputes_hashes_and_rejects_forged_fallback():
    starts, draws = 3, 16
    source_p = np.arange(starts * 6, dtype=np.float32).reshape(starts, 2, 3)
    source_v = np.zeros((starts, 2, 1, 3), dtype=np.float32)
    archive = {
        "c.source.P": source_p,
        "c.source.V": source_v,
    }
    cell = {
        "state_archive_prefix": "c",
        "accepted": [[False] + [True] * (draws - 1) for _ in range(starts)],
        "source_state_sha256": [
            adjudicator._numpy_state_sha256(source_p[i], source_v[i])
            for i in range(starts)
        ],
        "raw_state_sha256": [],
        "selected_state_sha256": [],
    }
    for start in range(starts):
        raw_p = np.repeat(source_p[start][None], draws, axis=0) + 1.0
        raw_v = np.repeat(source_v[start][None], draws, axis=0) + 1.0
        selected_p, selected_v = raw_p.copy(), raw_v.copy()
        selected_p[0], selected_v[0] = source_p[start], source_v[start]
        archive[f"c.raw{start}.P"] = raw_p
        archive[f"c.raw{start}.V"] = raw_v
        archive[f"c.selected{start}.P"] = selected_p
        archive[f"c.selected{start}.V"] = selected_v
        cell["raw_state_sha256"].append([
            adjudicator._numpy_state_sha256(raw_p[i], raw_v[i]) for i in range(draws)
        ])
        cell["selected_state_sha256"].append([
            adjudicator._numpy_state_sha256(selected_p[i], selected_v[i])
            for i in range(draws)
        ])
    adjudicator._verify_conditional_archive(cell, archive)

    archive["c.selected0.P"][0] = archive["c.raw0.P"][0]
    archive["c.selected0.V"][0] = archive["c.raw0.V"][0]
    cell["selected_state_sha256"][0][0] = adjudicator._numpy_state_sha256(
        archive["c.selected0.P"][0], archive["c.selected0.V"][0]
    )
    with pytest.raises(ValueError, match="exact raw-or-source fallback"):
        adjudicator._verify_conditional_archive(cell, archive)


def _geometry_cell():
    protocol = _protocol()
    starts = protocol["geometry"]["starts_per_cell"]
    frames = 60
    reference = {
        "bond_mean": (3.8 + np.linspace(-0.02, 0.02, frames)).tolist(),
        "bond_p99": (4.0 + np.linspace(-0.02, 0.02, frames)).tolist(),
        "bond_max": (4.15 + np.linspace(-0.02, 0.02, frames)).tolist(),
        "angle_cos_mean": np.linspace(-0.4, -0.3, frames).tolist(),
        "angle_cos_p01": np.linspace(-0.9, -0.8, frames).tolist(),
        "angle_cos_p99": np.linspace(0.1, 0.2, frames).tolist(),
        "collision_fraction": np.linspace(0.0, 0.002, frames).tolist(),
    }
    initial = {key: [values[10], values[25], values[40]] for key, values in reference.items()}
    previous = copy.deepcopy(initial)
    previous_hashes = [
        hashlib.sha256(f"geometry-initial-{index}".encode()).hexdigest()
        for index in range(starts)
    ]
    initial_hashes = copy.deepcopy(previous_hashes)
    steps = []
    for step in range(1, 101):
        proposed = copy.deepcopy(previous)
        proposed["collision_fraction"] = [value + 0.00001 for value in previous["collision_fraction"]]
        source_obs = [
            {
                "position_finite": True,
                "vector_finite": True,
                "bond_mean": previous["bond_mean"][index],
                "bond_max": previous["bond_max"][index],
            }
            for index in range(starts)
        ]
        raw_obs = [
            {
                "position_finite": True,
                "vector_finite": True,
                "bond_mean": proposed["bond_mean"][index],
                "bond_max": proposed["bond_max"][index],
            }
            for index in range(starts)
        ]
        raw_hashes = [
            hashlib.sha256(f"geometry-raw-{step}-{index}".encode()).hexdigest()
            for index in range(starts)
        ]
        steps.append({
            "step": step,
            "accepted": [True] * starts,
            "selected_position_exact": [True] * starts,
            "selected_vector_exact": [True] * starts,
            "source_state_sha256": copy.deepcopy(previous_hashes),
            "raw_state_sha256": raw_hashes,
            "selected_state_sha256": copy.deepcopy(raw_hashes),
            "source_guard_observations": source_obs,
            "raw_guard_observations": raw_obs,
            "raw_statistics": copy.deepcopy(proposed),
            "guarded_statistics": copy.deepcopy(proposed),
        })
        previous = proposed
        previous_hashes = raw_hashes
    return {
        "start_frames": [0, 10, 20],
        "reference_statistics": reference,
        "initial_statistics": initial,
        "initial_state_sha256": initial_hashes,
        "steps_h100": steps,
    }


def test_geometry_h20_is_derived_from_exact_h100_prefix():
    protocol = _protocol()
    original = adjudicator.recompute_geometry_cell(_geometry_cell(), protocol, seed=17)
    modified_cell = _geometry_cell()
    for index in range(3):
        modified_cell["steps_h100"][20]["raw_statistics"]["collision_fraction"][index] += 0.2
        modified_cell["steps_h100"][20]["guarded_statistics"]["collision_fraction"][index] += 0.2
    modified = adjudicator.recompute_geometry_cell(modified_cell, protocol, seed=17)
    assert modified["h20"] == original["h20"]
    assert modified["h100"] != original["h100"]


def test_geometry_rejected_step_must_equal_exact_current_source():
    cell = _geometry_cell()
    cell["steps_h100"][0]["accepted"] = [False, True, True]
    cell["steps_h100"][0]["raw_guard_observations"][0]["bond_max"] = 9.0
    cell["steps_h100"][0]["raw_statistics"]["bond_max"][0] = 9.0
    cell["steps_h100"][0]["selected_state_sha256"][0] = \
        cell["steps_h100"][0]["source_state_sha256"][0]
    cell["steps_h100"][0]["guarded_statistics"]["collision_fraction"][0] += 0.1
    with pytest.raises(ValueError, match="exact current source"):
        adjudicator.recompute_geometry_cell(cell, _protocol(), seed=17)


def test_geometry_reject_branch_detects_selected_vector_state_tamper():
    cell = _geometry_cell()
    first = cell["steps_h100"][0]
    first["accepted"][0] = False
    first["raw_guard_observations"][0]["vector_finite"] = False
    for metric in first["guarded_statistics"]:
        first["guarded_statistics"][metric][0] = cell["initial_statistics"][metric][0]
    first["selected_state_sha256"][0] = first["raw_state_sha256"][0]
    with pytest.raises(ValueError, match="selected P/V"):
        adjudicator.recompute_geometry_cell(cell, _protocol(), seed=17)


def test_h20_archive_tamper_is_rejected_from_canonical_pv_bytes():
    initial_p = np.zeros((3, 2, 3), dtype=np.float32)
    initial_v = np.zeros((3, 2, 1, 3), dtype=np.float32)
    archive = {"g.initial.P": initial_p, "g.initial.V": initial_v}
    initial_hashes = [
        adjudicator._numpy_state_sha256(initial_p[i], initial_v[i])
        for i in range(3)
    ]
    cell = {
        "state_archive_prefix": "g",
        "initial_state_sha256": initial_hashes,
        "steps_h100": [],
    }
    previous_p, previous_v = initial_p, initial_v
    for step_index in range(1, 21):
        source_p, source_v = previous_p.copy(), previous_v.copy()
        raw_p, raw_v = source_p + 1.0, source_v + 1.0
        selected_p, selected_v = raw_p.copy(), raw_v.copy()
        for kind, p_value, v_value in (
            ("source", source_p, source_v),
            ("raw", raw_p, raw_v),
            ("selected", selected_p, selected_v),
        ):
            archive[f"g.{kind}{step_index}.P"] = p_value
            archive[f"g.{kind}{step_index}.V"] = v_value
        cell["steps_h100"].append({
            "accepted": [True, True, True],
            "source_state_sha256": [
                adjudicator._numpy_state_sha256(source_p[i], source_v[i])
                for i in range(3)
            ],
            "raw_state_sha256": [
                adjudicator._numpy_state_sha256(raw_p[i], raw_v[i])
                for i in range(3)
            ],
            "selected_state_sha256": [
                adjudicator._numpy_state_sha256(selected_p[i], selected_v[i])
                for i in range(3)
            ],
        })
        previous_p, previous_v = selected_p, selected_v
    adjudicator._verify_geometry_archive(cell, archive)

    archive["g.selected20.V"][0, 0, 0, 0] += 7.0
    with pytest.raises(ValueError, match="canonical archive bytes"):
        adjudicator._verify_geometry_archive(cell, archive)


def test_paired_gain_uses_domain_as_outer_unit_and_strict_lower_bound():
    equal = adjudicator._paired_gain([1.0, 2.0], [1.0, 2.0], draws=1000, seed=3)
    assert equal["passes"] is False
    positive = adjudicator._paired_gain(
        [0.1, 0.25, 0.35, 0.5],
        [1.0, 1.1, 1.2, 1.3],
        draws=1000,
        seed=3,
    )
    assert positive["passes"] is True


def _raw_domains():
    cells = [
        {
            "temperature": temperature,
            "replica": replica,
            "reference_replica": [0, 1, 2, 3, 4][
                ([0, 1, 2, 3, 4].index(replica) + 1) % 5
            ],
            "delta_frames": 1,
        }
        for temperature in [320, 348, 379, 413, 450]
        for replica in [0, 1, 2, 3, 4]
    ]
    return [
        {"domain": f"domain-{index:02d}", "cells": copy.deepcopy(cells)}
        for index in range(20)
    ]


def test_independent_aggregation_cannot_override_missing_delta1_oracle(monkeypatch):
    protocol = _protocol()
    raw = {
        "conditional_raw": {"domains": _raw_domains()},
        "geometry_raw": {"domains": _raw_domains()},
    }
    monkeypatch.setattr(
        adjudicator,
        "recompute_conditional_cell",
        lambda *args, **kwargs: {
            "noop": {"mean_energy_score": 2.0, "msm_row_jsd_bits": 2.0},
            "guarded": {"mean_energy_score": 1.0, "msm_row_jsd_bits": 1.0},
        },
    )
    monkeypatch.setattr(adjudicator, "_load_state_archive", lambda raw: {})
    monkeypatch.setattr(adjudicator, "_verify_conditional_archive", lambda *args: None)
    monkeypatch.setattr(adjudicator, "_verify_geometry_archive", lambda *args: None)
    geometry_metrics = {
        name: -0.1
        for name in (
            "bond_mean", "bond_p99", "bond_max", "angle_cos_mean",
            "angle_cos_p01", "angle_cos_p99", "collision_fraction",
        )
    }
    monkeypatch.setattr(
        adjudicator,
        "recompute_geometry_cell",
        lambda *args, **kwargs: {
            "h20": {
                "guarded_worst_excess": geometry_metrics,
                "guarded_step_excess": [geometry_metrics] * 20,
            },
            "h100": {
                "guarded_worst_excess": geometry_metrics,
                "guarded_step_excess": [geometry_metrics] * 100,
            },
        },
    )
    decision = adjudicator.independently_adjudicate(raw, protocol, _session())
    assert decision["conditional_pass"] is True
    assert decision["geometry_pass"] is True
    assert decision["status"] == adjudicator.INCONCLUSIVE_ORACLE
    assert "PASS" not in decision["status"] and "FAIL" not in decision["status"]
    assert decision["formal_training_authorized"] is False


def test_single_positive_geometry_cell_fails_hard_envelope_even_if_domain_ci_passes(monkeypatch):
    protocol = _protocol()
    raw = {
        "conditional_raw": {"domains": _raw_domains()},
        "geometry_raw": {"domains": _raw_domains()},
    }
    monkeypatch.setattr(
        adjudicator,
        "recompute_conditional_cell",
        lambda *args, **kwargs: {
            "noop": {"mean_energy_score": 2.0, "msm_row_jsd_bits": 2.0},
            "guarded": {"mean_energy_score": 1.0, "msm_row_jsd_bits": 1.0},
        },
    )
    monkeypatch.setattr(adjudicator, "_load_state_archive", lambda raw: {})
    monkeypatch.setattr(adjudicator, "_verify_conditional_archive", lambda *args: None)
    monkeypatch.setattr(adjudicator, "_verify_geometry_archive", lambda *args: None)
    calls = {"count": 0}

    def geometry_result(*args, **kwargs):
        calls["count"] += 1
        value = 0.1 if calls["count"] == 1 else -1.0
        metrics = {
            name: value
            for name in (
                "bond_mean", "bond_p99", "bond_max", "angle_cos_mean",
                "angle_cos_p01", "angle_cos_p99", "collision_fraction",
            )
        }
        return {
            "h20": {
                "guarded_worst_excess": metrics,
                "guarded_step_excess": [metrics] * 20,
            },
            "h100": {
                "guarded_worst_excess": metrics,
                "guarded_step_excess": [metrics] * 100,
            },
        }

    monkeypatch.setattr(adjudicator, "recompute_geometry_cell", geometry_result)
    decision = adjudicator.independently_adjudicate(raw, protocol, _session())
    assert decision["geometry_domain_mean_pass"] is True
    assert decision["geometry"]["hard_envelope"]["passes"] is False
    assert decision["geometry_pass"] is False
    assert decision["geometry"]["hard_envelope"]["violations"]


def test_adjudicator_reopens_checkpoint_contract_panel_and_payload():
    source = inspect.getsource(adjudicator.independently_verify_bound_inputs)
    assert "verify_frozen_evaluation_identity(" in source
    assert "verify_scientific_data_prerequisite(" in source
    assert "rehash_contracted_panel_payloads(" in source
    assert "_load_verified_checkpoint(" in source
