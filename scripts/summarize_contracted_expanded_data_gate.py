#!/usr/bin/env python
"""Safely bind a contracted development decision into an immutable summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from deepjump.data_contract import _read_regular_bytes
from scripts.contracted_guarded_endpoint_panel_eval import SCOPE, _atomic_json_new


_HEX = frozenset("0123456789abcdef")
_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def _load_exact_json(path: Path, expected_sha256: str, label: str) -> tuple[dict, str]:
    expected_sha256 = _sha256(expected_sha256, f"{label} SHA256")
    raw = _read_regular_bytes(path.expanduser(), label)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, actual_sha256


def _require_fields(payload: dict, expected: dict, label: str) -> None:
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"{label} identity mismatch: {mismatches}")


def _checkpoint_gate_binding(
    path: Path,
    expected_gate_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    expected_step: int,
    label: str,
) -> dict:
    payload, gate_sha256 = _load_exact_json(path, expected_gate_sha256, label)
    expected_checkpoint_sha256 = _sha256(
        expected_checkpoint_sha256, f"{label} checkpoint SHA256"
    )
    checkpoint_raw = _read_regular_bytes(checkpoint_path.expanduser(), f"{label} checkpoint")
    if hashlib.sha256(checkpoint_raw).hexdigest() != expected_checkpoint_sha256:
        raise ValueError(f"{label} checkpoint SHA256 mismatch")
    _require_fields(
        payload,
        {
            "status": "PASS",
            "checkpoint_sha256": expected_checkpoint_sha256,
            "checkpoint_step": expected_step,
        },
        label,
    )
    return {
        "status": "PASS",
        "sha256": gate_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_step": expected_step,
    }


def summarize(
    decision_path: Path,
    *,
    expected_decision_sha256: str,
    expected_result_sha256: str,
    expected_checkpoint_sha256: str,
    expected_contract_sha256: str,
    expected_panel_name: str,
    expected_panel_sha256: str,
    expected_prerequisite_decision_sha256: str,
    runtime_probe_path: Path,
    expected_runtime_probe_sha256: str,
    run_id: str,
    commit: str,
    obs: str,
    source_identity_manifest_sha256: str,
    smoke_checkpoint_gate_path: Path,
    expected_smoke_checkpoint_gate_sha256: str,
    smoke_checkpoint_path: Path,
    expected_smoke_checkpoint_sha256: str,
    calibration_checkpoint_gate_path: Path,
    expected_calibration_checkpoint_gate_sha256: str,
    calibration_checkpoint_path: Path,
    expected_calibration_checkpoint_sha256: str,
    development_checkpoint_gate_path: Path,
    expected_development_checkpoint_gate_sha256: str,
    development_checkpoint_path: Path,
) -> dict:
    """Re-read exact evidence bytes and return a fully identity-bound summary."""

    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("run ID must be a UTC basic timestamp")
    if len(commit) != 40 or set(commit) - _HEX:
        raise ValueError("commit must be a lowercase 40-hex SHA")
    if not obs.startswith("obs://"):
        raise ValueError("OBS destination must use obs://")

    expected_result_sha256 = _sha256(expected_result_sha256, "result SHA256")
    expected_checkpoint_sha256 = _sha256(
        expected_checkpoint_sha256, "checkpoint SHA256"
    )
    expected_contract_sha256 = _sha256(expected_contract_sha256, "contract SHA256")
    expected_panel_sha256 = _sha256(expected_panel_sha256, "panel SHA256")
    expected_prerequisite_decision_sha256 = _sha256(
        expected_prerequisite_decision_sha256, "prerequisite decision SHA256"
    )
    source_identity_manifest_sha256 = _sha256(
        source_identity_manifest_sha256, "source identity manifest SHA256"
    )
    checkpoint_gates = {
        "smoke": _checkpoint_gate_binding(
            smoke_checkpoint_gate_path,
            expected_smoke_checkpoint_gate_sha256,
            smoke_checkpoint_path,
            expected_smoke_checkpoint_sha256,
            100,
            "smoke checkpoint gate",
        ),
        "calibration": _checkpoint_gate_binding(
            calibration_checkpoint_gate_path,
            expected_calibration_checkpoint_gate_sha256,
            calibration_checkpoint_path,
            expected_calibration_checkpoint_sha256,
            1_000,
            "calibration checkpoint gate",
        ),
        "development": _checkpoint_gate_binding(
            development_checkpoint_gate_path,
            expected_development_checkpoint_gate_sha256,
            development_checkpoint_path,
            expected_checkpoint_sha256,
            2_000,
            "development checkpoint gate",
        ),
    }
    runtime_probe, runtime_probe_sha256 = _load_exact_json(
        runtime_probe_path,
        expected_runtime_probe_sha256,
        "development runtime probe",
    )
    decision, decision_sha256 = _load_exact_json(
        decision_path,
        expected_decision_sha256,
        "development decision",
    )

    expected = {
        "scope": SCOPE,
        "phase": "development",
        "result_sha256": expected_result_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": expected_contract_sha256,
        "panel_name": expected_panel_name,
        "panel_sha256": expected_panel_sha256,
        "prerequisite_decision_sha256": expected_prerequisite_decision_sha256,
        "formal_training_authorized": False,
    }
    _require_fields(decision, expected, "development decision")
    if decision.get("runtime_probe") != {**runtime_probe, "sha256": runtime_probe_sha256}:
        raise ValueError("development decision runtime-probe binding mismatch")

    identity = decision.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("development decision identity is missing")
    _require_fields(
        identity,
        {
            "status": "PASS_FROZEN_EVALUATION_IDENTITY",
            "phase": "development",
            "panel_name": expected_panel_name,
            "panel_sha256": expected_panel_sha256,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": expected_contract_sha256,
            "formal_training_authorized": False,
        },
        "development decision nested identity",
    )
    prerequisite = decision.get("prerequisite")
    if not isinstance(prerequisite, dict):
        raise ValueError("development decision prerequisite is missing")
    _require_fields(
        prerequisite,
        {
            "status": "ADVANCE_EXPANDED_DATA_DEVELOPMENT",
            "phase": "development",
            "panel_name": expected_panel_name,
            "panel_sha256": expected_panel_sha256,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": expected_contract_sha256,
            "reserved_panel_authorized": True,
            "formal_training_authorized": False,
            "sha256": expected_prerequisite_decision_sha256,
        },
        "development decision nested prerequisite",
    )
    run_binding = {
        "run_id": run_id,
        "commit": commit,
        "obs": obs,
        "source_identity_manifest_sha256": source_identity_manifest_sha256,
        "checkpoint_gates": checkpoint_gates,
    }
    if prerequisite.get("run_binding") != run_binding:
        raise ValueError("development decision run binding mismatch")
    consumption_claim = decision.get("consumption_claim")
    if not isinstance(consumption_claim, dict):
        raise ValueError("development decision consumption claim is missing")
    _require_fields(
        consumption_claim,
        {
            "authorization_sha256": expected_prerequisite_decision_sha256,
            "phase": "development",
            "panel_name": expected_panel_name,
            "panel_sha256": expected_panel_sha256,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": expected_contract_sha256,
        },
        "development decision nested consumption claim",
    )
    _sha256(consumption_claim.get("sha256"), "consumption claim SHA256")

    status = decision.get("status")
    gate_status = decision.get("gate_status")
    if status == "ADVANCE_EXPANDED_DATA_EXTERNAL":
        if gate_status != "PASS_CONTRACTED_GUARD_DEVELOPMENT20":
            raise ValueError("advance decision does not bind the qualifying gate status")
    elif not (
        isinstance(status, str)
        and status.startswith("STOP_")
        and status == gate_status
    ):
        raise ValueError("development decision status/gate-status relation is invalid")

    return {
        "status": status,
        "gate_status": gate_status,
        "external_evaluation_authorized": status == "ADVANCE_EXPANDED_DATA_EXTERNAL",
        "external_evaluation_started": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "run_id": run_id,
        "commit": commit,
        "source_identity_manifest_sha256": source_identity_manifest_sha256,
        "decision_sha256": decision_sha256,
        "result_sha256": expected_result_sha256,
        "runtime_probe_sha256": runtime_probe_sha256,
        "prerequisite_decision_sha256": expected_prerequisite_decision_sha256,
        "full_training_contract_sha256": expected_contract_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_step": 2_000,
        "checkpoint_gates": checkpoint_gates,
        "panel_name": expected_panel_name,
        "panel_sha256": expected_panel_sha256,
        "obs": obs,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--expected-panel-name", required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--expected-prerequisite-decision-sha256", required=True)
    parser.add_argument("--runtime-probe", required=True, type=Path)
    parser.add_argument("--expected-runtime-probe-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--source-identity-manifest-sha256", required=True)
    parser.add_argument("--smoke-checkpoint-gate", required=True, type=Path)
    parser.add_argument("--expected-smoke-checkpoint-gate-sha256", required=True)
    parser.add_argument("--smoke-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-smoke-checkpoint-sha256", required=True)
    parser.add_argument("--calibration-checkpoint-gate", required=True, type=Path)
    parser.add_argument("--expected-calibration-checkpoint-gate-sha256", required=True)
    parser.add_argument("--calibration-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-calibration-checkpoint-sha256", required=True)
    parser.add_argument("--development-checkpoint-gate", required=True, type=Path)
    parser.add_argument("--expected-development-checkpoint-gate-sha256", required=True)
    parser.add_argument("--development-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(
        args.decision,
        expected_decision_sha256=args.expected_decision_sha256,
        expected_result_sha256=args.expected_result_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_contract_sha256=args.expected_contract_sha256,
        expected_panel_name=args.expected_panel_name,
        expected_panel_sha256=args.expected_panel_sha256,
        expected_prerequisite_decision_sha256=(
            args.expected_prerequisite_decision_sha256
        ),
        runtime_probe_path=args.runtime_probe,
        expected_runtime_probe_sha256=args.expected_runtime_probe_sha256,
        run_id=args.run_id,
        commit=args.commit,
        obs=args.obs,
        source_identity_manifest_sha256=args.source_identity_manifest_sha256,
        smoke_checkpoint_gate_path=args.smoke_checkpoint_gate,
        expected_smoke_checkpoint_gate_sha256=(
            args.expected_smoke_checkpoint_gate_sha256
        ),
        smoke_checkpoint_path=args.smoke_checkpoint,
        expected_smoke_checkpoint_sha256=args.expected_smoke_checkpoint_sha256,
        calibration_checkpoint_gate_path=args.calibration_checkpoint_gate,
        expected_calibration_checkpoint_gate_sha256=(
            args.expected_calibration_checkpoint_gate_sha256
        ),
        calibration_checkpoint_path=args.calibration_checkpoint,
        expected_calibration_checkpoint_sha256=(
            args.expected_calibration_checkpoint_sha256
        ),
        development_checkpoint_gate_path=args.development_checkpoint_gate,
        expected_development_checkpoint_gate_sha256=(
            args.expected_development_checkpoint_gate_sha256
        ),
        development_checkpoint_path=args.development_checkpoint,
    )
    digest = _atomic_json_new(args.output, summary)
    print(json.dumps({**summary, "summary_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
