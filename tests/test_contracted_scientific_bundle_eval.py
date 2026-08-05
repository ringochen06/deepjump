import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import contracted_scientific_bundle_eval as bundle


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "contracted_scientific_bundle_protocol_v1.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return _sha256(path)


def _session(tmp_path, protocol_sha, *, delta=1, oracle=bundle.ORACLE_UNRESOLVED):
    return {
        "schema": bundle.SESSION_SCHEMA,
        "session_id": "scientific-dev20-seed0",
        "repo_commit": "1" * 40,
        "protocol_sha256": protocol_sha,
        "phase": "development",
        "authorization_id": "scientific-dev20-once",
        "checkpoint_sha256": "2" * 64,
        "checkpoint_step": 1000,
        "checkpoint_delta": delta,
        "data_root": str((tmp_path / "data-root").resolve()),
        "full_training_contract_sha256": "3" * 64,
        "data_manifest_sha256": "8" * 64,
        "evaluation_exclusion_registry_sha256": "9" * 64,
        "panel_name": "legacy_dev20",
        "panel_sha256": "4" * 64,
        "panel_domains": 20,
        "msm_oracle_status": oracle,
        "msm_oracle_prerequisite_path": (
            str((tmp_path / "oracle.json").resolve())
            if oracle == bundle.ORACLE_PASS else None
        ),
        "msm_oracle_prerequisite_sha256": (
            "a" * 64 if oracle == bundle.ORACLE_PASS else None
        ),
        "msm_oracle_raw_path": (
            str((tmp_path / "oracle-raw.json").resolve())
            if oracle == bundle.ORACLE_PASS else None
        ),
        "msm_oracle_raw_sha256": (
            "d" * 64 if oracle == bundle.ORACLE_PASS else None
        ),
        "runtime_feasibility_path": str((tmp_path / "runtime-feasibility.json").resolve()),
        "runtime_feasibility_sha256": "b" * 64,
        "runtime_probe_output": str((tmp_path / "runtime.json").resolve()),
        "raw_output": str((tmp_path / "raw.json").resolve()),
        "decision_output": str((tmp_path / "decision.json").resolve()),
        "state_archive_output": str((tmp_path / "state-archive.npz").resolve()),
        "obs_prefix": "obs://bucket/scientific/dev20",
        "global_claim_descriptor_path": None,
        "global_claim_receipt_path": None,
        "global_claim_readback_path": None,
        "formal_training_authorized": False,
    }


def _prerequisite(tmp_path, session, session_sha):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    return {
        "schema": bundle.PREREQUISITE_SCHEMA,
        "authorization_id": session["authorization_id"],
        "consumption_ledger_root": str(ledger.resolve()),
        "status": bundle.PREREQUISITE_STATUS[session["phase"]],
        "phase": session["phase"],
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "data_manifest_sha256": session["data_manifest_sha256"],
        "evaluation_exclusion_registry_sha256": session[
            "evaluation_exclusion_registry_sha256"
        ],
        "panel_name": session["panel_name"],
        "panel_sha256": session["panel_sha256"],
        "reserved_panel_authorized": True,
        "scientific_bundle_authorized": True,
        "scientific_protocol_sha256": session["protocol_sha256"],
        "scientific_session_sha256": session_sha,
        "msm_oracle_status": session["msm_oracle_status"],
        "msm_oracle_prerequisite_path": session["msm_oracle_prerequisite_path"],
        "msm_oracle_prerequisite_sha256": session["msm_oracle_prerequisite_sha256"],
        "msm_oracle_raw_path": session["msm_oracle_raw_path"],
        "msm_oracle_raw_sha256": session["msm_oracle_raw_sha256"],
        "runtime_feasibility_path": session["runtime_feasibility_path"],
        "runtime_feasibility_sha256": session["runtime_feasibility_sha256"],
        "state_archive_output": session["state_archive_output"],
        "global_claim_descriptor_path": session["global_claim_descriptor_path"],
        "global_claim_receipt_path": session["global_claim_receipt_path"],
        "global_claim_readback_path": session["global_claim_readback_path"],
        "formal_training_authorized": False,
    }


def _runtime_artifact(session, protocol_sha, *, elapsed=1.0):
    factor = 1.25
    per_cell = elapsed / 20.0
    coordinates = [
        (temperature, replica)
        for temperature in [320, 348, 379, 413, 450]
        for replica in [0, 1, 2, 3, 4]
    ][:20]
    probe_domains = []
    for index in range(5):
        probe_domains.append({
            "domain": f"non-reserved-runtime-domain-{index}",
            "payload_sha256": f"{index + 1:x}" * 64,
            "payload_size_bytes": 1000 + index,
            "is_largest_non_reserved_payload": index == 4,
            "cells": [
                {
                    "temperature": temperature,
                    "replica": replica,
                    "elapsed_seconds": per_cell,
                }
                for temperature, replica in coordinates
            ],
        })
    raw_path = Path(session["runtime_feasibility_path"]).with_name("runtime-raw.json")
    raw = {
        "schema": bundle.RUNTIME_RAW_SCHEMA,
        "phase": session["phase"],
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "data_manifest_sha256": session["data_manifest_sha256"],
        "evaluation_exclusion_registry_sha256": session[
            "evaluation_exclusion_registry_sha256"
        ],
        "panel_name": session["panel_name"],
        "panel_sha256": session["panel_sha256"],
        "protocol_sha256": protocol_sha,
        "projected_domains": session["panel_domains"],
        "probe_source_role": "non_reserved_representative_domains_including_largest_payload",
        "workload": "five_non_reserved_domains_20_cells_each_conditional16_and_guarded_h100",
        "probe_domains": probe_domains,
        "gpu_environment": {
            "device_name": "Tesla V100",
            "device_uuid": "GPU-test",
            "driver_version": "535.1",
            "cuda_runtime_version": "12.1",
            "compute_capability": "7.0",
        },
        "peak_memory_bytes": 1024,
        "formal_training_authorized": False,
    }
    raw_sha = _write_json(raw_path, raw)
    return {
        "schema": bundle.RUNTIME_FEASIBILITY_SCHEMA,
        "status": bundle.RUNTIME_FEASIBILITY_PASS,
        "execution_mode": "single_gpu_sequential_domains",
        "phase": session["phase"],
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "data_manifest_sha256": session["data_manifest_sha256"],
        "evaluation_exclusion_registry_sha256": session[
            "evaluation_exclusion_registry_sha256"
        ],
        "panel_name": session["panel_name"],
        "panel_sha256": session["panel_sha256"],
        "protocol_sha256": protocol_sha,
        "measured_domains": 5,
        "projected_domains": session["panel_domains"],
        "projection_safety_factor": factor,
        "probe_source_role": "non_reserved_representative_domains_including_largest_payload",
        "workload": "five_non_reserved_domains_20_cells_each_conditional16_and_guarded_h100",
        "max_projected_seconds": 8400.0,
        "raw_measurement_path": str(raw_path.resolve()),
        "raw_measurement_sha256": raw_sha,
        "formal_training_authorized": False,
    }


def _oracle_artifact(session, protocol_sha):
    return {
        "schema": bundle.ORACLE_SCHEMA,
        "status": bundle.ORACLE_PASS,
        "decision": "PASS",
        "evidence_type": "measured_delta1_msm_oracle",
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "protocol_sha256": protocol_sha,
        "seed": 20260723,
        "raw_draws_path": session["msm_oracle_raw_path"],
        "raw_draws_sha256": session["msm_oracle_raw_sha256"],
        "decision_rule": "domain_bootstrap_ci95_lower_gt_zero",
        "formal_training_authorized": False,
    }


def _oracle_source_provenance(session, protocol_sha):
    source_root = Path(session["msm_oracle_raw_path"]).parent / "oracle-source"
    source_root.mkdir(exist_ok=True)
    source = _session(source_root, protocol_sha, oracle=bundle.ORACLE_UNRESOLVED)
    source.update({
        "session_id": "delta1-oracle-development-source",
        "authorization_id": "delta1-oracle-development-source-once",
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "data_manifest_sha256": session["data_manifest_sha256"],
        "evaluation_exclusion_registry_sha256": session[
            "evaluation_exclusion_registry_sha256"
        ],
        "panel_name": "oracle-development-20",
        "panel_sha256": "e" * 64,
        "raw_output": session["msm_oracle_raw_path"],
    })
    source_path = source_root / "source-session.json"
    source_sha = _write_json(source_path, source)
    loaded = bundle.load_session(source_path, source_sha)
    prerequisite_path = source_root / "source-prerequisite.json"
    prerequisite_sha = _write_json(
        prerequisite_path, _prerequisite(source_root, loaded, source_sha)
    )
    prerequisite = bundle.load_scientific_prerequisite(
        prerequisite_path, prerequisite_sha, session=loaded
    )
    claim = bundle.claim_reserved_evaluation(
        prerequisite,
        prerequisite_sha,
        runtime_probe_output=loaded["runtime_probe_output"],
        output=loaded["raw_output"],
    )
    return {
        "source_session_path": str(source_path.resolve()),
        "source_session_sha256": source_sha,
        "source_prerequisite_path": str(prerequisite_path.resolve()),
        "source_prerequisite_sha256": prerequisite_sha,
        "consumption_claim": claim,
    }


def _oracle_raw(session, protocol_sha, *, gain=0.02, provenance=None):
    states = 32
    identity = np.eye(states, dtype=int)
    uniform_counts = np.ones((states, states), dtype=int)
    diagonal_counts = identity * 32
    model_counts = diagonal_counts if gain > 0 else uniform_counts
    noop_counts = uniform_counts if gain > 0 else diagonal_counts
    cells = [
        {
            "temperature": temperature,
            "replica": replica,
            "reference_transition_rows": identity.tolist(),
            "noop_transition_counts": noop_counts.tolist(),
            "model_transition_counts": model_counts.tolist(),
            "shared_origin_counts": [1] * states,
        }
        for temperature in [320, 348, 379, 413, 450]
        for replica in [0, 1, 2, 3, 4]
    ]
    return {
        "schema": bundle.ORACLE_RAW_SCHEMA,
        "phase": "development",
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "protocol_sha256": protocol_sha,
        "seed": 20260723,
        "oracle_panel_name": "oracle-development-20",
        "oracle_panel_sha256": "e" * 64,
        **(provenance or _oracle_source_provenance(session, protocol_sha)),
        "domains": [
            {"domain": f"oracle-domain-{index:02d}", "cells": copy.deepcopy(cells)}
            for index in range(20)
        ],
        "formal_training_authorized": False,
    }


def _materialize_oracle(session, protocol_sha, *, gain=0.02):
    provenance = _oracle_source_provenance(session, protocol_sha)
    session["msm_oracle_raw_sha256"] = _write_json(
        Path(session["msm_oracle_raw_path"]),
        _oracle_raw(session, protocol_sha, gain=gain, provenance=provenance),
    )
    session["msm_oracle_prerequisite_sha256"] = _write_json(
        Path(session["msm_oracle_prerequisite_path"]),
        _oracle_artifact(session, protocol_sha),
    )


def _runtime_contract(tmp_path, session):
    manifest = [
        {
            "domain": f"non-reserved-runtime-domain-{index}",
            "sha256": f"{index + 1:x}" * 64,
            "size": 1000 + index,
        }
        for index in range(5)
    ]
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    registry_path = tmp_path / "runtime-registry.json"
    registry_sha = _write_json(registry_path, {})
    train_path = tmp_path / "runtime-train.txt"
    train_path.write_text(
        "".join(f"non-reserved-runtime-domain-{index}\n" for index in range(5)),
        encoding="utf-8",
    )
    train_sha = _sha256(train_path)
    contract_path = tmp_path / "runtime-contract.json"
    contract_sha = _write_json(
        contract_path,
        {
            "artifacts": {
                "manifest": {"path": manifest_path.name, "sha256": manifest_sha},
                "panel_registry": {"path": registry_path.name, "sha256": registry_sha},
                "train_list": {"path": train_path.name, "sha256": train_sha},
            }
        },
    )
    session["data_manifest_sha256"] = manifest_sha
    session["evaluation_exclusion_registry_sha256"] = registry_sha
    session["full_training_contract_sha256"] = contract_sha
    return contract_path


def test_frozen_protocol_encodes_exact_scientific_and_data_semantics():
    protocol = bundle.load_protocol(PROTOCOL, _sha256(PROTOCOL))
    assert protocol["conditional_transition"]["methods"] == ["ode_1"]
    assert protocol["conditional_transition"]["draws_per_start"] >= 16
    assert protocol["geometry"]["horizons"] == [20, 100]
    assert protocol["geometry"]["derive_h20_from_h100_prefix"] is True
    assert protocol["data_prerequisite"]["fail_status"] == bundle.DATA_PREREQUISITE_FAILURE
    assert protocol["formal_training_authorized"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("conditional_transition", "methods"), ["mean"]),
        (("conditional_transition", "draws_per_start"), 15),
        (("conditional_transition", "guard", "fallback"), "keep_invalid"),
        (("conditional_transition", "real_frames_per_cell"), 499),
        (("geometry", "derive_h20_from_h100_prefix"), False),
        (("geometry", "reference_frames_per_cell"), 499),
        (("geometry", "real_envelope_alpha"), 0.02),
        (("geometry", "collision_distance_angstrom"), 2.6),
        (("adjudication", "bootstrap_draws"), 9999),
        (("adjudication", "energy_gain_ci95_lower_gt"), -1.0),
        (("adjudication", "geometry_hard_envelope_all_domain_cell_step_lte"), 1.0),
        (("runtime_feasibility", "projection_safety_factor"), 1.0),
        (("runtime_feasibility", "workload"), "partial_probe"),
        (("untouched_global_claim", "helper_sha256"), "f" * 64),
        (("untouched_global_claim", "helper_id"), "replacement-helper"),
        (("oracle_prerequisite", "source_evaluator"), "handwritten_json"),
        (("qualification_trust_boundary",), "arbitrary_local_python_is_trusted"),
        (("data_prerequisite", "zero_unresolved_failures"), False),
    ],
)
def test_protocol_rejects_semantic_drift(path, value):
    payload = json.loads(PROTOCOL.read_text())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        bundle.validate_protocol(payload)


def test_session_and_prerequisite_bind_exact_identity(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, _session(tmp_path, protocol_sha))
    session = bundle.load_session(session_path, session_sha)
    prereq_path = tmp_path / "prerequisite.json"
    prereq_sha = _write_json(prereq_path, _prerequisite(tmp_path, session, session_sha))
    prereq = bundle.load_scientific_prerequisite(
        prereq_path, prereq_sha, session=session
    )
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    bundle.validate_bundle_bindings(
        protocol, session, prereq, expected_repo_commit="1" * 40
    )


def test_session_rejects_relative_data_root(tmp_path):
    session_path = tmp_path / "session.json"
    payload = _session(tmp_path, _sha256(PROTOCOL))
    payload["data_root"] = "relative/data"
    digest = _write_json(session_path, payload)
    with pytest.raises(ValueError, match="data_root"):
        bundle.load_session(session_path, digest)


def test_session_cannot_select_or_replace_obs_helper_identity(tmp_path):
    session_path = tmp_path / "session-with-helper.json"
    payload = _session(tmp_path, _sha256(PROTOCOL))
    payload["obs_conditional_create_helper_path"] = "/tmp/attacker-helper"
    payload["obs_conditional_create_helper_sha256"] = "f" * 64
    digest = _write_json(session_path, payload)
    with pytest.raises(ValueError, match="exact schema"):
        bundle.load_session(session_path, digest)


def test_prerequisite_rejects_wrong_scientific_session(tmp_path):
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, _session(tmp_path, _sha256(PROTOCOL)))
    session = bundle.load_session(session_path, session_sha)
    payload = _prerequisite(tmp_path, session, session_sha)
    payload["scientific_session_sha256"] = "f" * 64
    prerequisite_path = tmp_path / "prerequisite.json"
    digest = _write_json(prerequisite_path, payload)
    with pytest.raises(ValueError, match="exact session"):
        bundle.load_scientific_prerequisite(prerequisite_path, digest, session=session)


def test_post_claim_identity_failure_burns_authorization_and_produces_no_result(tmp_path, monkeypatch):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, delta=10)
    contract_path = _runtime_contract(tmp_path, session_payload)
    session_payload["runtime_feasibility_sha256"] = _write_json(
        Path(session_payload["runtime_feasibility_path"]),
        _runtime_artifact(session_payload, protocol_sha),
    )
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, session_payload)
    loaded_session = bundle.load_session(session_path, session_sha)
    prereq_path = tmp_path / "prerequisite.json"
    prereq_sha = _write_json(
        prereq_path, _prerequisite(tmp_path, loaded_session, session_sha)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "contracted_scientific_bundle_eval.py",
            "--protocol", str(PROTOCOL),
            "--expected-protocol-sha256", protocol_sha,
            "--session", str(session_path),
            "--expected-session-sha256", session_sha,
            "--prerequisite-decision", str(prereq_path),
            "--expected-prerequisite-decision-sha256", prereq_sha,
            "--checkpoint", str(tmp_path / "unused.ckpt"),
            "--contract", str(contract_path),
            "--panel-file", str(tmp_path / "unused-panel.txt"),
            "--data-root", session_payload["data_root"],
            "--expected-repo-commit", "1" * 40,
        ],
    )
    with pytest.raises(ValueError):
        bundle.main()
    assert len(list((tmp_path / "ledger").iterdir())) == 1
    assert not Path(session_payload["raw_output"]).exists()


def test_unresolved_delta1_is_non_consumptive_before_model_or_panel(tmp_path, monkeypatch):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha)
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    prereq_path = tmp_path / "prerequisite.json"
    prereq_sha = _write_json(prereq_path, _prerequisite(tmp_path, session, session_sha))
    monkeypatch.setattr(
        bundle,
        "establish_claimed_inputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("claim/panel opened")),
    )
    monkeypatch.setattr(sys, "argv", [
        "contracted_scientific_bundle_eval.py",
        "--protocol", str(PROTOCOL), "--expected-protocol-sha256", protocol_sha,
        "--session", str(session_path), "--expected-session-sha256", session_sha,
        "--prerequisite-decision", str(prereq_path),
        "--expected-prerequisite-decision-sha256", prereq_sha,
        "--checkpoint", str(tmp_path / "absent.ckpt"),
        "--contract", str(tmp_path / "absent-contract.json"),
        "--panel-file", str(tmp_path / "absent-panel.txt"),
        "--data-root", session_payload["data_root"],
        "--expected-repo-commit", "1" * 40,
    ])
    bundle.main()
    assert not list((tmp_path / "ledger").iterdir())
    decision = json.loads(Path(session_payload["decision_output"]).read_text())
    assert decision["status"] == bundle.INCONCLUSIVE_ORACLE
    assert decision["authorization_consumed"] is False
    assert decision["panel_opened"] is False
    assert decision["model_opened"] is False


def test_bound_oracle_requires_real_exact_artifact_not_status_and_random_sha(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    _write_json(
        Path(session_payload["msm_oracle_raw_path"]),
        _oracle_raw(session_payload, protocol_sha),
    )
    fake_path = Path(session_payload["msm_oracle_prerequisite_path"])
    session_payload["msm_oracle_prerequisite_sha256"] = _write_json(
        fake_path, {"status": bundle.ORACLE_PASS}
    )
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    with pytest.raises(ValueError, match="raw draws SHA256 mismatch"):
        bundle.load_delta1_oracle_artifact(protocol, session)


def test_bound_oracle_accepts_only_measured_positive_ci_artifact(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    _materialize_oracle(session_payload, protocol_sha)
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    loaded = bundle.load_delta1_oracle_artifact(protocol, session)
    assert loaded["decision"] == "PASS"
    assert loaded["recomputed"]["paired_msm_gain_ci95_lower"] > 0

    prior_raw = json.loads(Path(session_payload["msm_oracle_raw_path"]).read_text())
    provenance = {
        key: prior_raw[key]
        for key in (
            "source_session_path",
            "source_session_sha256",
            "source_prerequisite_path",
            "source_prerequisite_sha256",
            "consumption_claim",
        )
    }
    session_payload["msm_oracle_raw_sha256"] = _write_json(
        Path(session_payload["msm_oracle_raw_path"]),
        _oracle_raw(session_payload, protocol_sha, gain=-0.01, provenance=provenance),
    )
    session_payload["msm_oracle_prerequisite_sha256"] = _write_json(
        Path(session_payload["msm_oracle_prerequisite_path"]),
        _oracle_artifact(session_payload, protocol_sha),
    )
    bad_session_path = tmp_path / "bad-session.json"
    bad_session_sha = _write_json(bad_session_path, session_payload)
    bad_session = bundle.load_session(bad_session_path, bad_session_sha)
    with pytest.raises(ValueError, match="recomputed CI"):
        bundle.load_delta1_oracle_artifact(protocol, bad_session)


def test_oracle_rejects_self_reported_ci_and_requires_companion_raw(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    _materialize_oracle(session_payload, protocol_sha)
    forged = _oracle_artifact(session_payload, protocol_sha)
    forged["paired_msm_gain_ci95_lower"] = 999.0
    session_payload["msm_oracle_prerequisite_sha256"] = _write_json(
        Path(session_payload["msm_oracle_prerequisite_path"]), forged
    )
    session_path = tmp_path / "forged-ci-session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    with pytest.raises(ValueError, match="exact bound PASS"):
        bundle.load_delta1_oracle_artifact(protocol, session)


def test_oracle_rejects_unclaimed_handwritten_count_evidence(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    raw = _oracle_raw(session_payload, protocol_sha)
    raw["consumption_claim"] = {"sha256": "0" * 64}
    session_payload["msm_oracle_raw_sha256"] = _write_json(
        Path(session_payload["msm_oracle_raw_path"]), raw
    )
    session_payload["msm_oracle_prerequisite_sha256"] = _write_json(
        Path(session_payload["msm_oracle_prerequisite_path"]),
        _oracle_artifact(session_payload, protocol_sha),
    )
    session_path = tmp_path / "handwritten-oracle-session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    with pytest.raises(ValueError, match="consumption claim"):
        bundle.load_delta1_oracle_artifact(protocol, session)


def test_oracle_rejects_handwritten_gain_instead_of_raw_transition_counts(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    provenance = _oracle_source_provenance(session_payload, protocol_sha)
    raw = _oracle_raw(session_payload, protocol_sha, provenance=provenance)
    forged_cell = raw["domains"][0]["cells"][0]
    forged_cell.clear()
    forged_cell.update({
        "temperature": 320,
        "replica": 0,
        "noop_minus_model_msm_gain": 1.0,
    })
    session_payload["msm_oracle_raw_sha256"] = _write_json(
        Path(session_payload["msm_oracle_raw_path"]), raw
    )
    session_payload["msm_oracle_prerequisite_sha256"] = _write_json(
        Path(session_payload["msm_oracle_prerequisite_path"]),
        _oracle_artifact(session_payload, protocol_sha),
    )
    session_path = tmp_path / "handwritten-gain-session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    with pytest.raises(ValueError, match="raw cell schema mismatch"):
        bundle.load_delta1_oracle_artifact(protocol, session)


def test_forged_oracle_is_rejected_before_claim(tmp_path, monkeypatch):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    session_payload["msm_oracle_raw_sha256"] = _write_json(
        Path(session_payload["msm_oracle_raw_path"]),
        _oracle_raw(session_payload, protocol_sha),
    )
    session_payload["msm_oracle_prerequisite_sha256"] = _write_json(
        Path(session_payload["msm_oracle_prerequisite_path"]),
        {"status": bundle.ORACLE_PASS, "sha256": "f" * 64},
    )
    session_payload["runtime_feasibility_sha256"] = _write_json(
        Path(session_payload["runtime_feasibility_path"]),
        _runtime_artifact(session_payload, protocol_sha),
    )
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    prereq_path = tmp_path / "prerequisite.json"
    prereq_sha = _write_json(prereq_path, _prerequisite(tmp_path, session, session_sha))
    monkeypatch.setattr(
        bundle,
        "establish_claimed_inputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("claim called")),
    )
    monkeypatch.setattr(sys, "argv", [
        "contracted_scientific_bundle_eval.py",
        "--protocol", str(PROTOCOL), "--expected-protocol-sha256", protocol_sha,
        "--session", str(session_path), "--expected-session-sha256", session_sha,
        "--prerequisite-decision", str(prereq_path),
        "--expected-prerequisite-decision-sha256", prereq_sha,
        "--checkpoint", str(tmp_path / "absent.ckpt"),
        "--contract", str(tmp_path / "absent-contract.json"),
        "--panel-file", str(tmp_path / "absent-panel.txt"),
        "--data-root", session_payload["data_root"],
        "--expected-repo-commit", "1" * 40,
    ])
    with pytest.raises(ValueError, match="exact bound PASS"):
        bundle.main()
    assert not list((tmp_path / "ledger").iterdir())


def test_runtime_projection_rejects_before_claim(tmp_path, monkeypatch):
    protocol_sha = _sha256(PROTOCOL)
    session_payload = _session(tmp_path, protocol_sha, delta=10)
    session_payload["runtime_feasibility_sha256"] = _write_json(
        Path(session_payload["runtime_feasibility_path"]),
        _runtime_artifact(session_payload, protocol_sha, elapsed=400.0),
    )
    session_path = tmp_path / "session.json"
    session_sha = _write_json(session_path, session_payload)
    session = bundle.load_session(session_path, session_sha)
    prereq_path = tmp_path / "prerequisite.json"
    prereq_sha = _write_json(prereq_path, _prerequisite(tmp_path, session, session_sha))
    monkeypatch.setattr(
        bundle,
        "establish_claimed_inputs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("claim called")),
    )
    monkeypatch.setattr(sys, "argv", [
        "contracted_scientific_bundle_eval.py",
        "--protocol", str(PROTOCOL), "--expected-protocol-sha256", protocol_sha,
        "--session", str(session_path), "--expected-session-sha256", session_sha,
        "--prerequisite-decision", str(prereq_path),
        "--expected-prerequisite-decision-sha256", prereq_sha,
        "--checkpoint", str(tmp_path / "absent.ckpt"),
        "--contract", str(tmp_path / "absent-contract.json"),
        "--panel-file", str(tmp_path / "absent-panel.txt"),
        "--data-root", session_payload["data_root"],
        "--expected-repo-commit", "1" * 40,
    ])
    with pytest.raises(ValueError, match="STOP_PROJECTED"):
        bundle.main()
    assert not list((tmp_path / "ledger").iterdir())


def test_handwritten_oracle_raw_has_no_decision_producer_capability(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    oracle_session = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_PASS)
    oracle_raw_path = Path(oracle_session["msm_oracle_raw_path"])
    oracle_raw_sha = _write_json(
        oracle_raw_path, _oracle_raw(oracle_session, protocol_sha)
    )
    main_source = inspect.getsource(bundle.main)
    assert "--measure-delta1-oracle" in main_source
    assert "--produce-oracle-decision" not in main_source
    assert not hasattr(bundle, "_finalize_delta1_oracle_decision")
    assert not hasattr(bundle, "_OracleMeasurementToken")
    assert not hasattr(bundle, "_MEASUREMENT_CAPABILITY_SEAL")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_runtime_loader_independently_recomputes_executor_raw(tmp_path):
    protocol_sha = _sha256(PROTOCOL)

    runtime_session = _session(tmp_path, protocol_sha, delta=10)
    runtime_decision_template = _runtime_artifact(runtime_session, protocol_sha)
    runtime_raw_path = Path(runtime_decision_template["raw_measurement_path"])
    runtime_raw_sha = runtime_decision_template["raw_measurement_sha256"]
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    raw, projected = bundle._load_runtime_raw(
        protocol, runtime_session, runtime_raw_path, runtime_raw_sha
    )
    assert projected > 0
    assert raw["sha256"] == runtime_raw_sha


def test_runtime_producer_rejects_single_domain_and_self_reported_projection(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session = _session(tmp_path, protocol_sha, delta=10)
    template = _runtime_artifact(session, protocol_sha)
    raw_path = Path(template["raw_measurement_path"])
    raw = json.loads(raw_path.read_text())
    raw["probe_domains"] = raw["probe_domains"][:1]
    raw_sha = _write_json(raw_path, raw)
    with pytest.raises(ValueError, match="multi-domain"):
        bundle._load_runtime_raw(
            bundle.load_protocol(PROTOCOL, protocol_sha), session, raw_path, raw_sha
        )

    raw = _runtime_artifact(session, protocol_sha)
    raw_path = Path(raw["raw_measurement_path"])
    raw_payload = json.loads(raw_path.read_text())
    raw_payload["projected_total_seconds"] = 1.0
    raw_sha = _write_json(raw_path, raw_payload)
    with pytest.raises(ValueError, match="exact schema"):
        bundle._load_runtime_raw(
            bundle.load_protocol(PROTOCOL, protocol_sha), session, raw_path, raw_sha
        )


def test_qualification_modes_do_not_require_the_artifacts_they_create(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("qualification mode must not load an existing artifact")

    monkeypatch.setattr(bundle, "load_delta1_oracle_artifact", unexpected)
    monkeypatch.setattr(bundle, "load_runtime_feasibility_artifact", unexpected)
    unresolved = {"checkpoint_delta": 1, "msm_oracle_status": bundle.ORACLE_UNRESOLVED}
    assert bundle.load_bundle_prerequisites_for_mode({}, unresolved, "runtime") == (
        None,
        None,
    )
    assert bundle.load_bundle_prerequisites_for_mode(
        {}, unresolved, "delta1_oracle"
    ) == (None, None)
    with pytest.raises(ValueError, match="oracle unresolved"):
        bundle.load_bundle_prerequisites_for_mode({}, unresolved, "bundle")


def test_final_bundle_requires_both_existing_qualification_artifacts(monkeypatch):
    monkeypatch.setattr(
        bundle, "load_delta1_oracle_artifact", lambda *_args: {"sha256": "a" * 64}
    )
    monkeypatch.setattr(
        bundle,
        "load_runtime_feasibility_artifact",
        lambda *_args: {"sha256": "b" * 64},
    )
    resolved = {"checkpoint_delta": 1, "msm_oracle_status": bundle.ORACLE_PASS}
    assert bundle.load_bundle_prerequisites_for_mode({}, resolved, "bundle") == (
        {"sha256": "a" * 64},
        {"sha256": "b" * 64},
    )


def test_runtime_qualification_has_only_real_workload_cli_and_no_json_timing_producer():
    source = inspect.getsource(bundle.execute_runtime_feasibility_measurement)
    main_source = inspect.getsource(bundle.main)
    assert "time.monotonic_ns()" in source
    assert source.count("torch.cuda.synchronize(device)") >= 2
    assert "torch.cuda.max_memory_allocated(device)" in source
    assert "evaluate_conditional_cell(" in source
    assert "evaluate_geometry_cell(" in source
    assert "--measure-runtime" in main_source
    assert "--produce-runtime-decision" not in main_source
    assert not hasattr(bundle, "_finalize_runtime_feasibility_decision")
    assert not hasattr(bundle, "_RuntimeMeasurementToken")
    assert not hasattr(bundle, "_MEASUREMENT_CAPABILITY_SEAL")


@pytest.mark.parametrize(
    "forbidden_flag",
    ["--produce-oracle-decision", "--produce-runtime-decision"],
)
def test_public_cli_cannot_qualify_handwritten_raw(
    tmp_path, monkeypatch, forbidden_flag
):
    handwritten = tmp_path / "handwritten-raw.json"
    handwritten.write_text('{"claimed_pass": true}\n', encoding="utf-8")
    decision = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(sys, "argv", [
        "contracted_scientific_bundle_eval.py",
        forbidden_flag,
        str(handwritten),
        str(decision),
    ])
    with pytest.raises(SystemExit):
        bundle.main()
    assert not decision.exists()


def test_runtime_executor_measures_exact_5x20_workload_from_payloads(tmp_path, monkeypatch):
    protocol_sha = _sha256(PROTOCOL)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    session = _session(tmp_path, protocol_sha, delta=10)
    data_root = tmp_path / "runtime-data"
    data_root.mkdir()
    manifest = []
    plan = []
    for index in range(5):
        path = data_root / f"domain-{index}.h5"
        path.write_bytes((f"payload-{index}" * (index + 1)).encode())
        digest = _sha256(path)
        size = path.stat().st_size
        manifest.append({"domain": f"domain-{index}", "sha256": digest, "size": size})
        plan.append({
            "domain": f"domain-{index}",
            "path": str(path.resolve()),
            "payload_sha256": digest,
            "payload_size_bytes": size,
            "is_largest_non_reserved_payload": index == 4,
        })
    manifest_path = tmp_path / "measured-manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    registry_path = tmp_path / "measured-registry.json"
    registry_sha = _write_json(registry_path, {})
    train_path = tmp_path / "measured-train.txt"
    train_path.write_text("".join(f"domain-{index}\n" for index in range(5)))
    contract_path = tmp_path / "measured-contract.json"
    contract_sha = _write_json(contract_path, {"artifacts": {
        "manifest": {"path": manifest_path.name, "sha256": manifest_sha},
        "panel_registry": {"path": registry_path.name, "sha256": registry_sha},
        "train_list": {"path": train_path.name, "sha256": _sha256(train_path)},
    }})
    session.update({
        "full_training_contract_sha256": contract_sha,
        "data_manifest_sha256": manifest_sha,
        "evaluation_exclusion_registry_sha256": registry_sha,
    })

    class FakeHandle:
        layout = object()
        def __init__(self, path): self.path = path
        def close(self): pass

    class FakeModel:
        def __init__(self, *args, **kwargs): pass
        def to(self, device): return self
        def load_state_dict(self, state, strict=True): pass
        def eval(self): return self

    calls = {"conditional": 0, "geometry": 0, "sync": 0}
    monkeypatch.setattr(bundle, "_DomainHandle", FakeHandle)
    monkeypatch.setattr(bundle, "DeepJumpLite", FakeModel)
    monkeypatch.setattr(bundle, "resolve_device", lambda value: torch.device("cuda:0"))
    monkeypatch.setattr(bundle, "_load_verified_checkpoint", lambda *args: ({
        "cfg": {
            "data": {"noise_sigma": 0.1},
            "model": {"predict_heavy": True},
            "train": {"device": "cuda"},
        },
        "model": {},
    }, session["checkpoint_sha256"]))
    monkeypatch.setattr(bundle, "evaluate_conditional_cell", lambda **kwargs: calls.__setitem__("conditional", calls["conditional"] + 1))
    monkeypatch.setattr(bundle, "evaluate_geometry_cell", lambda **kwargs: calls.__setitem__("geometry", calls["geometry"] + 1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.__setitem__("sync", calls["sync"] + 1))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123456)
    monkeypatch.setattr(bundle, "_gpu_environment", lambda device: {
        "device_name": "V100", "device_uuid": "GPU-real-path",
        "driver_version": "535", "cuda_runtime_version": "12.1",
        "compute_capability": "7.0",
    })
    ticks = iter(range(1, 1000))
    monkeypatch.setattr(bundle.time, "monotonic_ns", lambda: next(ticks) * 1_000_000)
    output = tmp_path / "measured-runtime-raw.json"
    decision_output = tmp_path / "measured-runtime-decision.json"
    result = bundle.execute_runtime_feasibility_measurement(
        protocol=protocol,
        session=session,
        checkpoint=tmp_path / "checkpoint.pt",
        contract=contract_path,
        data_root=data_root,
        probe_plan=plan,
        raw_output=output,
        decision_output=decision_output,
    )
    measured = json.loads(output.read_text())
    decision = json.loads(decision_output.read_text())
    assert result["status"] == bundle.RUNTIME_FEASIBILITY_PASS
    assert decision["raw_measurement_sha256"] == _sha256(output)
    assert calls == {"conditional": 100, "geometry": 100, "sync": 200}
    assert len(measured["probe_domains"]) == 5
    assert all(len(row["cells"]) == 20 for row in measured["probe_domains"])
    assert measured["peak_memory_bytes"] == 123456
    with pytest.raises(FileExistsError, match="both be fresh"):
        bundle.execute_runtime_feasibility_measurement(
            protocol=protocol,
            session=session,
            checkpoint=tmp_path / "checkpoint.pt",
            contract=contract_path,
            data_root=data_root,
            probe_plan=plan,
            raw_output=output,
            decision_output=decision_output,
        )
    assert calls == {"conditional": 100, "geometry": 100, "sync": 200}


def test_oracle_executor_runs_exact_consumed_20x25_conditional_workload(
    tmp_path, monkeypatch
):
    protocol_sha = _sha256(PROTOCOL)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    session = _session(tmp_path, protocol_sha, oracle=bundle.ORACLE_UNRESOLVED)
    session.update({
        "panel_name": "oracle-development-20",
        "panel_sha256": "e" * 64,
        "raw_output": str((tmp_path / "measured-oracle-raw.json").resolve()),
        "path": str((tmp_path / "source-session.json").resolve()),
        "sha256": "a" * 64,
    })

    class Pins(list):
        closed = False

        def close(self):
            self.closed = True

    pins = Pins({"domain": f"oracle-domain-{index:02d}"} for index in range(20))
    inputs = {
        "protocol": protocol,
        "session": session,
        "prerequisite": {
            "path": str((tmp_path / "source-prerequisite.json").resolve()),
            "sha256": "b" * 64,
        },
        "consumption_claim": {"schema": "measured-consumption-claim"},
        "payload_verification": {"pins": pins},
    }

    class FakeModel:
        def to(self, device): return self
        def load_state_dict(self, state, strict): assert strict is True
        def eval(self): return self

    class FakeHandle:
        layout = object()
        def close(self): pass

    checkpoint = {
        "cfg": {
            "data": {
                "noise_sigma": 0.1,
                "temperatures": [320, 348, 379, 413, 450],
                "replicas": [0, 1, 2, 3, 4],
                "delta_frames": [1],
            },
            "model": {"predict_heavy": True},
            "train": {"device": "cuda"},
        },
        "model": {},
    }
    calls = []
    real = np.repeat(np.arange(32, dtype=float), 2)[:, None]
    real = np.concatenate([real + offset * 0.01 for offset in range(4)], axis=1)
    source = np.asarray([[0.0] * 4, [15.0] * 4, [31.0] * 4])
    guarded = np.repeat(source[:, None, :], 16, axis=1)

    def measured_cell(**kwargs):
        calls.append((kwargs["temperature"], kwargs["replica"]))
        return {
            "temperature": kwargs["temperature"],
            "replica": kwargs["replica"],
            "delta_frames": 1,
            "draws": 16,
            "real_tic": real.tolist(),
            "source_tic": source.tolist(),
            "guarded_predicted_tic": guarded.tolist(),
        }

    monkeypatch.setattr(
        bundle, "_load_verified_checkpoint", lambda *args: (checkpoint, "2" * 64)
    )
    monkeypatch.setattr(bundle, "resolve_device", lambda value: torch.device("cuda"))
    monkeypatch.setattr(bundle, "ModelConfig", lambda **kwargs: object())
    monkeypatch.setattr(bundle, "DeepJumpLite", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(bundle, "_pinned_domain_handle", lambda pin: FakeHandle())
    monkeypatch.setattr(bundle, "evaluate_conditional_cell", measured_cell)
    def reopen_oracle(protocol_value, synthetic_session):
        path = Path(synthetic_session["msm_oracle_raw_path"])
        return (
            {**json.loads(path.read_text()), "path": str(path), "sha256": _sha256(path)},
            {"paired_msm_gain_ci95_lower": 0.01},
        )
    monkeypatch.setattr(bundle, "_load_oracle_raw", reopen_oracle)

    decision_output = tmp_path / "measured-oracle-decision.json"
    result = bundle.execute_delta1_oracle_measurement(
        inputs=inputs,
        checkpoint_path=tmp_path / "checkpoint.pt",
        raw_output=session["raw_output"],
        decision_output=decision_output,
    )
    assert result["status"] == bundle.ORACLE_PASS
    assert len(calls) == 20 * 25
    assert pins.closed is True
    raw = json.loads(Path(session["raw_output"]).read_text())
    decision = json.loads(decision_output.read_text())
    assert decision["raw_draws_sha256"] == _sha256(session["raw_output"])
    assert len(raw["domains"]) == 20
    assert all(len(row["cells"]) == 25 for row in raw["domains"])
    first = raw["domains"][0]["cells"][0]
    assert sum(first["shared_origin_counts"]) == 3 * 16
    assert sum(map(sum, first["noop_transition_counts"])) == 3 * 16
    assert sum(map(sum, first["model_transition_counts"])) == 3 * 16
    with pytest.raises(FileExistsError, match="both be fresh"):
        bundle.execute_delta1_oracle_measurement(
            inputs=inputs,
            checkpoint_path=tmp_path / "checkpoint.pt",
            raw_output=session["raw_output"],
            decision_output=decision_output,
        )
    assert len(calls) == 20 * 25


def test_runtime_loader_binds_nonreserved_payload_contract_and_largest_domain(tmp_path):
    protocol_sha = _sha256(PROTOCOL)
    session = _session(tmp_path, protocol_sha, delta=10)
    contract = _runtime_contract(tmp_path, session)
    session["runtime_feasibility_sha256"] = _write_json(
        Path(session["runtime_feasibility_path"]),
        _runtime_artifact(session, protocol_sha),
    )
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    runtime = bundle.load_runtime_feasibility_artifact(protocol, session)
    bundle.verify_runtime_probe_bindings(runtime, contract, session)
    runtime["probe_domains"][0]["payload_size_bytes"] += 1
    with pytest.raises(ValueError, match="payload identity mismatch"):
        bundle.verify_runtime_probe_bindings(runtime, contract, session)


def _fake_obs_executor(protocol, descriptor_path, receipt_path, readback_path, *, created=True):
    descriptor = json.loads(Path(descriptor_path).read_text())
    payload = (json.dumps(descriptor["payload"], indent=2, sort_keys=True) + "\n").encode()
    Path(readback_path).write_bytes(payload)
    receipt = {
        "schema": bundle.GLOBAL_CLAIM_RECEIPT_SCHEMA,
        "created": created,
        "condition": "If-None-Match:*",
        "uri": descriptor["uri"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "helper_sha256": protocol["untouched_global_claim"]["helper_sha256"],
    }
    Path(receipt_path).write_text(json.dumps(receipt, sort_keys=True) + "\n")


def test_untouched_direct_core_requires_reviewed_conditional_create_receipt(tmp_path, monkeypatch):
    protocol_sha = _sha256(PROTOCOL)
    protocol = bundle.load_protocol(PROTOCOL, protocol_sha)
    monkeypatch.setattr(bundle, "_run_reviewed_obs_conditional_create", _fake_obs_executor)
    session = _session(tmp_path, protocol_sha, delta=10)
    session.update({
        "phase": "untouched",
        "panel_name": "untouched-100",
        "panel_domains": 100,
        "sha256": "f" * 64,
        "obs_prefix": "obs://bucket/deepjump-scientific/untouched",
        "global_claim_descriptor_path": str((tmp_path / "claim-descriptor.json").resolve()),
        "global_claim_receipt_path": str((tmp_path / "claim-receipt.json").resolve()),
        "global_claim_readback_path": str((tmp_path / "claim-readback.json").resolve()),
    })
    result = bundle.establish_global_obs_claim(protocol, session)
    assert result["completed"] is True
    assert result["helper_sha256"] == protocol["untouched_global_claim"]["helper_sha256"]
    with pytest.raises(FileExistsError):
        bundle.establish_global_obs_claim(protocol, session)

    monkeypatch.setattr(
        bundle,
        "_run_reviewed_obs_conditional_create",
        lambda protocol, descriptor_path, receipt_path, readback_path: _fake_obs_executor(
            protocol, descriptor_path, receipt_path, readback_path, created=False
        ),
    )
    bad = dict(session)
    bad.update({
        "authorization_id": "bad-global-claim",
        "global_claim_descriptor_path": str((tmp_path / "bad-descriptor.json").resolve()),
        "global_claim_receipt_path": str((tmp_path / "bad-receipt.json").resolve()),
        "global_claim_readback_path": str((tmp_path / "bad-readback.json").resolve()),
    })
    with pytest.raises(ValueError, match="receipt mismatch"):
        bundle.establish_global_obs_claim(protocol, bad)


def test_claim_precedes_identity_and_live_payload_open():
    source = inspect.getsource(bundle.establish_claimed_inputs)
    assert source.index("claim_reserved_evaluation(") < source.index(
        "verify_frozen_evaluation_identity("
    ) < source.index("rehash_contracted_panel_payloads(") < source.index(
        "establish_global_obs_claim("
    )


def test_output_collision_is_rejected(tmp_path):
    session = _session(tmp_path, _sha256(PROTOCOL))
    Path(session["raw_output"]).write_text("existing")
    with pytest.raises(FileExistsError, match="refusing existing"):
        bundle.preflight_output_paths(session)


def _data_audit():
    return {
        "schema": bundle.AUDIT_SCHEMA,
        "status": bundle.AUDIT_STATUS,
        "domains": bundle.EXPECTED_DOMAINS,
        "h5_files": bundle.EXPECTED_DOMAINS,
        "h5_bytes": bundle.EXPECTED_H5_BYTES,
        "trajectories": bundle.EXPECTED_TRAJECTORIES,
        "hdf5_files_structurally_verified": bundle.EXPECTED_DOMAINS,
        "payload_hash_verification_mode": "full_rehash",
        "payload_hashes_verified": bundle.EXPECTED_DOMAINS,
        "data_gate_passed": True,
        "live_payload_bytes_rehashed": True,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "finite_endpoint_frames_verified": bundle.EXPECTED_TRAJECTORIES * 2,
        "formal_training_authorized": False,
    }


def test_scientific_data_prerequisite_requires_complete_structural_audit(tmp_path):
    audit_path = tmp_path / "audit.json"
    audit_sha = _write_json(audit_path, _data_audit())
    contract_path = tmp_path / "contract.json"
    contract_sha = _write_json(
        contract_path,
        {"artifacts": {"data_audit": {"path": "audit.json", "sha256": audit_sha}}},
    )
    report = bundle.verify_scientific_data_prerequisite(contract_path, contract_sha)
    assert report["status"] == "PASS_CONTRACTED_DATA_PREREQUISITE"
    assert report["hdf5_files_structurally_verified"] == 5398

    incomplete = _data_audit()
    incomplete["hdf5_files_structurally_verified"] = 5397
    incomplete_path = tmp_path / "incomplete-audit.json"
    incomplete_sha = _write_json(incomplete_path, incomplete)
    incomplete_contract = tmp_path / "incomplete-contract.json"
    incomplete_contract_sha = _write_json(
        incomplete_contract,
        {
            "artifacts": {
                "data_audit": {
                    "path": "incomplete-audit.json",
                    "sha256": incomplete_sha,
                }
            }
        },
    )
    with pytest.raises(ValueError, match=bundle.DATA_PREREQUISITE_FAILURE):
        bundle.verify_scientific_data_prerequisite(
            incomplete_contract, incomplete_contract_sha
        )


def test_implementation_status_is_numerically_ready_but_never_formal():
    report = bundle.implementation_report()
    assert report["numerical_kernel_implemented"] is True
    assert report["formal_training_authorized"] is False


class _FakeHandle:
    name = "fake-domain"

    def replicas(self, temperature, replicas):
        return [(temperature, replicas[0], 30)]

    def coords(self, temperature, replica, frame):
        positions = np.zeros((4, 3), dtype=np.float32)
        positions[:, 0] = np.arange(4, dtype=np.float32) * 3.8
        return positions


def _fake_layout():
    return SimpleNamespace(
        num_residues=4,
        res_index=np.arange(4),
        bond_mask=np.ones(3, dtype=bool),
        atom_mask=np.ones((4, 1), dtype=bool),
    )


def _patch_layout_and_tica(monkeypatch):
    monkeypatch.setattr(
        bundle,
        "apply_model_layout",
        lambda coordinates, layout, canon_symmetric: (
            coordinates.clone(),
            torch.zeros(4, 1, 3),
        ),
    )
    monkeypatch.setattr(
        bundle, "apply_layout", lambda coordinates, layout: (coordinates.clone(), None)
    )
    monkeypatch.setattr(
        bundle,
        "_trajectory_features",
        lambda *args, **kwargs: np.stack(
            [np.linspace(0, 1, 30), np.linspace(1, 2, 30),
             np.linspace(2, 3, 30), np.linspace(3, 4, 30)], axis=1
        ),
    )
    monkeypatch.setattr(bundle, "fit_tica", lambda values, lag, n_components: (np.zeros(4), np.eye(4)))
    monkeypatch.setattr(bundle, "selected_pair_indices", lambda n, maximum: (np.array([0]), np.array([1])))
    monkeypatch.setattr(
        bundle,
        "pairdist_features",
        lambda positions, pairs: np.stack(
            [
                positions[..., 0, 0].cpu().numpy(),
                positions[..., 1, 0].cpu().numpy(),
                positions[..., 2, 0].cpu().numpy(),
                positions[..., 3, 0].cpu().numpy(),
            ],
            axis=-1,
        ),
    )


def test_conditional_kernel_guards_each_ode_draw_to_exact_source(monkeypatch):
    _patch_layout_and_tica(monkeypatch)

    class Model:
        def sample(self, batch, **kwargs):
            proposed = batch["P_t"].clone()
            proposed[::2, 1, 0] += 10.0
            return proposed, batch["V_t"].clone()

    protocol = json.loads(PROTOCOL.read_text())
    result = bundle.evaluate_conditional_cell(
        handle=_FakeHandle(),
        layout=_fake_layout(),
        model=Model(),
        device=torch.device("cpu"),
        data_cfg={"crop_length": 4, "canon_symmetric": False},
        delta=1,
        temperature=320,
        replica=0,
        replicas=[0, 1, 2, 3, 4],
        protocol=protocol,
        seed_offset=0,
    )
    assert result["draws"] == 16
    assert all(row == [False, True] * 8 for row in result["accepted"])
    for start in range(3):
        for draw in range(0, 16, 2):
            assert result["guarded_predicted_tic"][start][draw] == result["source_tic"][start]


def test_geometry_kernel_is_one_guarded_h100_chain(monkeypatch):
    _patch_layout_and_tica(monkeypatch)

    class Model:
        def __init__(self):
            self.sources = []

        def sample(self, batch, **kwargs):
            self.sources.append(batch["P_t"].clone())
            proposed = batch["P_t"].clone()
            proposed[0, 1, 0] += 10.0
            proposed[1, :, 0] *= 1.001
            return proposed, batch["V_t"].clone()

    model = Model()
    protocol = json.loads(PROTOCOL.read_text())
    result = bundle.evaluate_geometry_cell(
        handle=_FakeHandle(),
        layout=_fake_layout(),
        model=model,
        device=torch.device("cpu"),
        data_cfg={"crop_length": 4, "canon_symmetric": False},
        delta=1,
        temperature=320,
        replica=0,
        protocol=protocol,
    )
    assert len(result["steps_h100"]) == 100
    assert all(step["accepted"][0] is False for step in result["steps_h100"])
    assert all(step["selected_position_exact"] == [True, True, True] for step in result["steps_h100"])
    assert torch.equal(model.sources[0][0], model.sources[1][0])
    assert not torch.equal(model.sources[0][1], model.sources[1][1])
