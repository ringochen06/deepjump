#!/usr/bin/env python
"""Raw-only evaluator for the one-shot contracted scientific bundle.

The evaluator emits draw/step-level evidence and never adjudicates it. Identity,
data completeness, per-draw/per-step safeguards, and H100-prefix semantics are
sealed here; scientific PASS/FAIL is computed only by the independent reader.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepjump.config import ModelConfig
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation_consumption import (
    claim_reserved_evaluation,
    verify_reserved_evaluation_claim,
)
from deepjump.data_contract import (
    AUDIT_SCHEMA,
    AUDIT_STATUS,
    EXPECTED_DOMAINS,
    EXPECTED_H5_BYTES,
    EXPECTED_TRAJECTORIES,
)
from deepjump.evaluation import (
    assign_clusters,
    fit_kmeans,
    require_mdcath_full_grid,
    require_single_delta,
    transition_matrix,
    weighted_row_jsd_bits,
)
from deepjump.evaluation_contract import (
    _load_verified_checkpoint,
    verify_frozen_evaluation_identity,
)
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_layout, apply_model_layout
from deepjump.sampling import reject_to_source
from deepjump.utils import resolve_device
from scripts.contracted_guarded_endpoint_panel_eval import (
    PREREQUISITE_SCHEMA,
    PREREQUISITE_STATUS,
    _pinned_domain_handle,
    rehash_contracted_panel_payloads,
)
from scripts.tica_robustness_eval import (
    contiguous_frame_ids,
    fit_tica,
    pairdist_features,
    selected_pair_indices,
)


PROTOCOL_SCHEMA = "deepjump.contracted_scientific_protocol.v1"
SESSION_SCHEMA = "deepjump.contracted_scientific_session.v1"
RAW_EVIDENCE_SCHEMA = "deepjump.contracted_scientific_raw.v1"
EVALUATOR_STATUS = "SCIENTIFIC_BUNDLE_COMPLETE_NOT_ADJUDICATED"
ORACLE_PASS = "PASS_DELTA1_MSM_ORACLE_REACHABILITY"
ORACLE_UNRESOLVED = "UNRESOLVED"
ORACLE_SCHEMA = "deepjump.delta1_msm_oracle.v2"
ORACLE_RAW_SCHEMA = "deepjump.delta1_msm_oracle_raw.v1"
RUNTIME_FEASIBILITY_SCHEMA = "deepjump.contracted_scientific_runtime_feasibility.v1"
RUNTIME_RAW_SCHEMA = "deepjump.contracted_scientific_runtime_raw.v1"
RUNTIME_FEASIBILITY_PASS = "PASS_CONTRACTED_SCIENTIFIC_RUNTIME_FEASIBILITY"
INCONCLUSIVE_ORACLE = "INCONCLUSIVE_DELTA1_MSM_ORACLE"
IMPLEMENTATION_STATUS = "NUMERICAL_KERNEL_READY_RAW_ONLY"
DATA_PREREQUISITE_FAILURE = "DATA_PREREQUISITE_FAILED_NOT_MODEL_FAILURE"
NUMERICAL_KERNEL_IMPLEMENTED = True
_HEX = frozenset("0123456789abcdef")
STATE_ARCHIVE_SCHEMA = "deepjump.contracted_scientific_state_archive.v1"
GLOBAL_CLAIM_SCHEMA = "deepjump.contracted_scientific_global_claim.v1"
GLOBAL_CLAIM_RECEIPT_SCHEMA = "deepjump.obs_conditional_create_receipt.v1"
OBS_CONDITIONAL_CREATE_HELPER_SOURCE = '''from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from obs import ObsClient

ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
SDK_VERSION = "3.26.6"

def write_new(path, raw):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())

if importlib.metadata.version("esdk-obs-python") != SDK_VERSION:
    raise SystemExit("unexpected OBS SDK version")
descriptor = json.loads(Path(sys.argv[1]).read_bytes().decode("utf-8"))
uri = descriptor["uri"]
if not uri.startswith("obs://"):
    raise SystemExit("invalid OBS URI")
bucket, key = uri[6:].split("/", 1)
payload = (json.dumps(descriptor["payload"], indent=2, sort_keys=True) + "\\n").encode()
if hashlib.sha256(payload).hexdigest() != descriptor["payload_sha256"]:
    raise SystemExit("payload SHA mismatch")
client = ObsClient(server=ENDPOINT, security_provider_policy="ECS")
try:
    response = client.putContent(
        bucket,
        key,
        payload,
        extensionHeaders={"If-None-Match": "*"},
    )
    if response.status < 200 or response.status >= 300:
        raise SystemExit(f"conditional create failed: HTTP {response.status}")
    observed = client.getObject(bucket, key, loadStreamInMemory=True)
    if observed.status < 200 or observed.status >= 300:
        raise SystemExit(f"readback failed: HTTP {observed.status}")
    readback = observed.body.buffer
finally:
    client.close()
if readback != payload:
    raise SystemExit("OBS readback differs from payload")
receipt = {
    "schema": "deepjump.obs_conditional_create_receipt.v1",
    "created": True,
    "condition": "If-None-Match:*",
    "uri": uri,
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
    "payload_size_bytes": len(payload),
    "helper_sha256": descriptor["helper_sha256"],
}
write_new(sys.argv[3], readback)
write_new(sys.argv[2], (json.dumps(receipt, sort_keys=True) + "\\n").encode())
'''
OBS_CONDITIONAL_CREATE_HELPER_SHA256 = hashlib.sha256(
    OBS_CONDITIONAL_CREATE_HELPER_SOURCE.encode("utf-8")
).hexdigest()

SESSION_KEYS = frozenset({
    "schema",
    "session_id",
    "repo_commit",
    "protocol_sha256",
    "phase",
    "authorization_id",
    "checkpoint_sha256",
    "checkpoint_step",
    "checkpoint_delta",
    "data_root",
    "full_training_contract_sha256",
    "data_manifest_sha256",
    "evaluation_exclusion_registry_sha256",
    "panel_name",
    "panel_sha256",
    "panel_domains",
    "msm_oracle_status",
    "msm_oracle_prerequisite_path",
    "msm_oracle_prerequisite_sha256",
    "msm_oracle_raw_path",
    "msm_oracle_raw_sha256",
    "runtime_feasibility_path",
    "runtime_feasibility_sha256",
    "runtime_probe_output",
    "raw_output",
    "decision_output",
    "state_archive_output",
    "obs_prefix",
    "global_claim_descriptor_path",
    "global_claim_receipt_path",
    "global_claim_readback_path",
    "formal_training_authorized",
})


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def _require_absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} must be canonical and absolute")
    return str(path)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_json_value(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[Any, str, Path]:
    expected_sha256 = _require_sha256(expected_sha256, f"{label} SHA256")
    configured = Path(path).expanduser()
    if configured.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = configured.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"cannot open {label}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"{label} changed while it was being read")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    return payload, actual, resolved


def _read_regular_json(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], str, Path]:
    payload, actual, resolved = _read_regular_json_value(path, expected_sha256, label)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, actual, resolved


def validate_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the frozen semantics without treating schema validity as a PASS."""

    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("scientific protocol schema mismatch")
    if set(payload) != {
        "schema",
        "seed",
        "panel",
        "conditional_transition",
        "geometry",
        "adjudication",
        "runtime_feasibility",
        "oracle_prerequisite",
        "data_prerequisite",
        "qualification_trust_boundary",
        "raw_evidence_schema",
        "state_archive_schema",
        "untouched_global_claim",
        "formal_training_authorized",
    }:
        raise ValueError("scientific protocol exact top-level schema mismatch")
    if payload.get("formal_training_authorized") is not False:
        raise ValueError("scientific protocol must keep formal training unauthorized")
    if payload.get("seed") != 20260723:
        raise ValueError("scientific protocol seed is not frozen")
    panel = payload.get("panel", {})
    if panel != {
        "cells_per_domain": 25,
        "phase_domain_counts": {
            "development": 20,
            "external": 20,
            "untouched": 100,
        },
        "replicas": [0, 1, 2, 3, 4],
        "temperatures_kelvin": [320, 348, 379, 413, 450],
    }:
        raise ValueError("scientific protocol panel is not exactly frozen")

    expected_guard = {
        "bond_max_lt_angstrom": 5.5,
        "bond_mean_gt_angstrom": 3.2,
        "bond_mean_lt_angstrom": 4.5,
        "fallback": "exact_source_position_and_vector_per_draw",
        "report_raw_and_guarded": True,
    }
    transition = payload.get("conditional_transition", {})
    if transition != {
        "draws_per_start": 16,
        "guard": expected_guard,
        "methods": ["ode_1"],
        "msm": {
            "clusters": 32,
            "lag_frames": 1,
            "log_base": 2,
            "pseudocount": 1e-8,
            "row_weighting": "shared_observed_origin_counts",
        },
        "real_frames_per_cell": 500,
        "starts_per_cell": 3,
        "tica": {
            "components": 4,
            "crossfit": "next_replica_leave_one_replica_out",
            "lag_ns": 10,
            "max_pair_features": 512,
        },
    }:
        raise ValueError("conditional transition settings are not exactly frozen")

    geometry = payload.get("geometry", {})
    geometry_guard = dict(expected_guard)
    geometry_guard["fallback"] = "exact_current_source_position_and_vector_per_step"
    if geometry != {
        "calibration_draws": 10000,
        "collision_distance_angstrom": 2.5,
        "derive_h20_from_h100_prefix": True,
        "guard": geometry_guard,
        "horizons": [20, 100],
        "method": "mean",
        "real_envelope_alpha": 0.01,
        "reference_frames_per_cell": 500,
        "starts_per_cell": 3,
    }:
        raise ValueError("geometry settings are not exactly frozen")

    adjudication = payload.get("adjudication", {})
    if adjudication != {
        "bootstrap_draws": 10000,
        "delta1_msm_oracle_required": True,
        "domain_is_outer_statistical_unit": True,
        "energy_gain_ci95_lower_gt": 0.0,
        "formal_training_authorized": False,
        "geometry_domain_mean_one_sided_alpha": 0.05,
        "geometry_domain_mean_one_sided_ci95_upper_lte": 0.0,
        "geometry_hard_envelope_all_domain_cell_step_lte": 0.0,
        "msm_gain_ci95_lower_gt": 0.0,
        "paired_ci_quantiles": [0.025, 0.975],
        "result_mode": "independent_recomputation_from_raw_evidence",
    }:
        raise ValueError("scientific adjudication settings are not exactly frozen")
    if payload.get("runtime_feasibility") != {
        "execution_mode": "single_gpu_sequential_domains",
        "max_projected_seconds_by_phase": {
            "development": 8400.0,
            "external": 8400.0,
            "untouched": 8400.0,
        },
        "projection_safety_factor": 1.25,
        "gpu_environment_required": True,
        "peak_memory_required": True,
        "probe_source_role": "non_reserved_representative_domains_including_largest_payload",
        "raw_per_cell_timings_required": True,
        "raw_schema": RUNTIME_RAW_SCHEMA,
        "representative_cells_per_domain": 20,
        "representative_domains": 5,
        "representative_selection": "train_payload_size_quantiles_0_25_50_75_100_lexical_tiebreak",
        "required_probe_includes_largest_payload": True,
        "schema": RUNTIME_FEASIBILITY_SCHEMA,
        "status": RUNTIME_FEASIBILITY_PASS,
        "workload": "five_non_reserved_domains_20_cells_each_conditional16_and_guarded_h100",
    }:
        raise ValueError("scientific runtime-feasibility settings are not exactly frozen")
    if payload.get("raw_evidence_schema") != RAW_EVIDENCE_SCHEMA:
        raise ValueError("scientific raw-evidence schema mismatch")
    if payload.get("qualification_trust_boundary") != (
        "frozen_evaluator_source_and_public_measure_cli;"
        "arbitrary_local_python_or_source_modification_out_of_scope"
    ):
        raise ValueError("scientific qualification trust boundary is not frozen")
    if payload.get("state_archive_schema") != STATE_ARCHIVE_SCHEMA:
        raise ValueError("scientific state-archive schema mismatch")
    if payload.get("untouched_global_claim") != {
        "condition": "If-None-Match:*",
        "helper_id": "embedded_obs_sdk_conditional_create_v1",
        "helper_sha256": OBS_CONDITIONAL_CREATE_HELPER_SHA256,
        "obs_endpoint": "https://obs.cn-north-4.myhuaweicloud.com",
        "obs_sdk_python": "/data/venvs/obs-claim/bin/python",
        "obs_sdk_version": "3.26.6",
    }:
        raise ValueError("scientific untouched global-claim helper is not frozen")
    if payload.get("oracle_prerequisite") != {
        "cell_evidence": "reference_transition_rows_plus_noop_model_raw_counts_and_shared_origins",
        "cells_per_domain": 25,
        "decision_schema": ORACLE_SCHEMA,
        "development_only_producer": True,
        "domain_is_outer_statistical_unit": True,
        "raw_schema": ORACLE_RAW_SCHEMA,
        "source_evaluator": "in_process_conditional_ode1_16draws_to_raw_msm_counts",
        "source_requires_consumed_development_evaluator": True,
    }:
        raise ValueError("scientific oracle-prerequisite settings are not exactly frozen")
    if payload.get("data_prerequisite") != {
        "fail_status": DATA_PREREQUISITE_FAILURE,
        "full_training_contract_required": True,
        "hdf5_readability_required": True,
        "live_panel_payload_sha256_rehash": True,
        "panel_domains_must_exist_in_manifest": True,
        "zero_unresolved_failures": True,
    }:
        raise ValueError("scientific data-prerequisite semantics mismatch")
    return payload


def load_protocol(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    payload, actual, resolved = _read_regular_json(
        path, expected_sha256, "scientific protocol"
    )
    validate_protocol(payload)
    return {**payload, "path": str(resolved), "sha256": actual}


def load_session(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    payload, actual, resolved = _read_regular_json(path, expected_sha256, "scientific session")
    if set(payload) != SESSION_KEYS or payload.get("schema") != SESSION_SCHEMA:
        raise ValueError("scientific session exact schema mismatch")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(payload.get("session_id"))) is None:
        raise ValueError("scientific session_id must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}", str(payload.get("repo_commit"))) is None:
        raise ValueError("scientific session repo_commit must be a full lowercase SHA")
    for key in (
        "protocol_sha256",
        "checkpoint_sha256",
        "full_training_contract_sha256",
        "data_manifest_sha256",
        "evaluation_exclusion_registry_sha256",
        "panel_sha256",
    ):
        _require_sha256(payload.get(key), f"scientific session {key}")
    if payload.get("phase") not in PREREQUISITE_STATUS:
        raise ValueError("scientific session phase is invalid")
    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(payload.get("authorization_id"))
    ) is None:
        raise ValueError("scientific session authorization_id is invalid")
    if type(payload.get("checkpoint_step")) is not int or payload["checkpoint_step"] <= 0:
        raise ValueError("scientific session checkpoint_step is invalid")
    if type(payload.get("checkpoint_delta")) is not int or payload["checkpoint_delta"] <= 0:
        raise ValueError("scientific session checkpoint_delta is invalid")
    _require_absolute(payload.get("data_root"), "scientific session data_root")
    if not isinstance(payload.get("panel_name"), str) or not payload["panel_name"]:
        raise ValueError("scientific session panel_name is invalid")
    if payload.get("panel_domains") not in {20, 100}:
        raise ValueError("scientific session panel domain count is invalid")
    oracle_status = payload.get("msm_oracle_status")
    oracle_path = payload.get("msm_oracle_prerequisite_path")
    oracle_sha = payload.get("msm_oracle_prerequisite_sha256")
    oracle_raw_path = payload.get("msm_oracle_raw_path")
    oracle_raw_sha = payload.get("msm_oracle_raw_sha256")
    if oracle_status == ORACLE_PASS:
        _require_absolute(oracle_path, "MSM oracle prerequisite path")
        _require_sha256(oracle_sha, "MSM oracle prerequisite SHA256")
        _require_absolute(oracle_raw_path, "MSM oracle raw path")
        _require_sha256(oracle_raw_sha, "MSM oracle raw SHA256")
    elif oracle_status == ORACLE_UNRESOLVED:
        if any(
            value is not None
            for value in (oracle_path, oracle_sha, oracle_raw_path, oracle_raw_sha)
        ):
            raise ValueError("unresolved MSM oracle must not claim an artifact")
    else:
        raise ValueError("scientific session MSM oracle status is invalid")
    _require_absolute(payload.get("runtime_feasibility_path"), "runtime feasibility path")
    _require_sha256(payload.get("runtime_feasibility_sha256"), "runtime feasibility SHA256")
    for key in (
        "runtime_probe_output",
        "raw_output",
        "decision_output",
        "state_archive_output",
    ):
        _require_absolute(payload.get(key), f"scientific session {key}")
    obs_prefix = payload.get("obs_prefix")
    if not isinstance(obs_prefix, str) or not obs_prefix.startswith("obs://"):
        raise ValueError("scientific session obs_prefix must use obs://")
    global_keys = (
        "global_claim_descriptor_path",
        "global_claim_receipt_path",
        "global_claim_readback_path",
    )
    if payload["phase"] == "untouched":
        for key in global_keys:
            _require_absolute(payload[key], f"scientific session {key}")
    elif any(payload[key] is not None for key in global_keys):
        raise ValueError("non-untouched session must not claim a global OBS one-shot helper")
    if payload.get("formal_training_authorized") is not False:
        raise ValueError("scientific session must keep formal training unauthorized")
    return {**payload, "path": str(resolved), "sha256": actual}


def load_scientific_prerequisite(
    path: str | Path,
    expected_sha256: str,
    *,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Load the final v2 authorization plus scientific-session bindings once."""

    payload, actual, resolved = _read_regular_json(
        path, expected_sha256, "scientific prerequisite"
    )
    required = {
        "schema": PREREQUISITE_SCHEMA,
        "authorization_id": session["authorization_id"],
        "status": PREREQUISITE_STATUS[session["phase"]],
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
        "scientific_session_sha256": session["sha256"],
        "msm_oracle_status": session["msm_oracle_status"],
        "msm_oracle_prerequisite_path": session[
            "msm_oracle_prerequisite_path"
        ],
        "msm_oracle_prerequisite_sha256": session[
            "msm_oracle_prerequisite_sha256"
        ],
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
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("scientific prerequisite does not authorize this exact session")
    ledger = payload.get("consumption_ledger_root")
    if not isinstance(ledger, str) or not Path(ledger).is_absolute():
        raise ValueError("scientific prerequisite consumption ledger is invalid")
    return {
        **required,
        "consumption_ledger_root": ledger,
        "path": str(resolved),
        "sha256": actual,
    }


def _bootstrap_domain_gain(
    gains: list[float], *, draws: int, seed: int, quantiles: list[float]
) -> dict[str, Any]:
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("oracle bootstrap requires finite multi-domain raw gains")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, quantiles)
    return {
        "domain_mean_gain": float(values.mean()),
        "paired_msm_gain_ci95_lower": float(low),
        "paired_msm_gain_ci95_upper": float(high),
        "oracle_domains": len(values),
    }


def _row_probabilities_from_raw_counts(
    value: object, *, states: int, pseudocount: float, label: str
) -> np.ndarray:
    counts = np.asarray(value, dtype=np.float64)
    if (
        counts.shape != (states, states)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or not np.equal(counts, np.floor(counts)).all()
    ):
        raise ValueError(f"{label} must be a non-negative integer MSM count matrix")
    smoothed = counts + pseudocount
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def _oracle_cell_gain(cell: dict[str, Any], protocol: dict[str, Any]) -> float:
    msm = protocol["conditional_transition"]["msm"]
    states = msm["clusters"]
    required = {
        "temperature",
        "replica",
        "reference_transition_rows",
        "noop_transition_counts",
        "model_transition_counts",
        "shared_origin_counts",
    }
    if not isinstance(cell, dict) or set(cell) != required:
        raise ValueError("delta1 MSM oracle raw cell schema mismatch")
    reference = np.asarray(cell["reference_transition_rows"], dtype=np.float64)
    if (
        reference.shape != (states, states)
        or not np.isfinite(reference).all()
        or (reference < 0).any()
        or not np.allclose(reference.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("delta1 MSM oracle reference row distributions are invalid")
    noop = _row_probabilities_from_raw_counts(
        cell["noop_transition_counts"],
        states=states,
        pseudocount=msm["pseudocount"],
        label="oracle no-op",
    )
    model = _row_probabilities_from_raw_counts(
        cell["model_transition_counts"],
        states=states,
        pseudocount=msm["pseudocount"],
        label="oracle model",
    )
    weights = np.asarray(cell["shared_origin_counts"], dtype=np.float64)
    if (
        weights.shape != (states,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.equal(weights, np.floor(weights)).all()
        or weights.sum() <= 0
    ):
        raise ValueError("delta1 MSM oracle shared origin counts are invalid")
    noop_jsd, _ = weighted_row_jsd_bits(reference, noop, weights)
    model_jsd, _ = weighted_row_jsd_bits(reference, model, weights)
    return float(noop_jsd - model_jsd)


def _raw_transition_counts(
    origins: np.ndarray, destinations: np.ndarray, *, states: int
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    destinations = np.asarray(destinations, dtype=np.int64)
    if (
        origins.ndim != 1
        or destinations.shape != origins.shape
        or (origins < 0).any()
        or (destinations < 0).any()
        or (origins >= states).any()
        or (destinations >= states).any()
    ):
        raise ValueError("oracle transition assignments are invalid")
    counts = np.zeros((states, states), dtype=np.int64)
    np.add.at(counts, (origins, destinations), 1)
    return counts


def _oracle_count_cell_from_measured_conditional(
    cell: dict[str, Any], protocol: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    """Derive recomputable MSM count evidence from an in-process model cell."""

    settings = protocol["conditional_transition"]
    starts = settings["starts_per_cell"]
    draws = settings["draws_per_start"]
    components = settings["tica"]["components"]
    states = settings["msm"]["clusters"]
    real = np.asarray(cell.get("real_tic"), dtype=np.float64)
    source = np.asarray(cell.get("source_tic"), dtype=np.float64)
    guarded = np.asarray(cell.get("guarded_predicted_tic"), dtype=np.float64)
    if (
        real.ndim != 2
        or real.shape[1] != components
        or len(real) < states
        or source.shape != (starts, components)
        or guarded.shape != (starts, draws, components)
        or not np.isfinite(real).all()
        or not np.isfinite(source).all()
        or not np.isfinite(guarded).all()
    ):
        raise ValueError("measured oracle conditional TIC evidence is invalid")
    if cell.get("draws") != draws or cell.get("delta_frames") != 1:
        raise ValueError("measured oracle cell is not the frozen delta=1 workload")
    centers, labels = fit_kmeans(real, states, seed=seed)
    reference_rows, _ = transition_matrix(
        labels,
        n_states=states,
        lag=settings["msm"]["lag_frames"],
        pseudocount=settings["msm"]["pseudocount"],
    )
    origins = assign_clusters(source, centers)
    repeated_origins = np.repeat(origins, draws)
    noop_counts = _raw_transition_counts(
        repeated_origins, repeated_origins, states=states
    )
    model_destinations = assign_clusters(
        guarded.reshape(-1, components), centers
    )
    model_counts = _raw_transition_counts(
        repeated_origins, model_destinations, states=states
    )
    shared_origins = np.bincount(repeated_origins, minlength=states).astype(np.int64)
    return {
        "temperature": int(cell["temperature"]),
        "replica": int(cell["replica"]),
        "reference_transition_rows": reference_rows.tolist(),
        "noop_transition_counts": noop_counts.tolist(),
        "model_transition_counts": model_counts.tolist(),
        "shared_origin_counts": shared_origins.tolist(),
    }


def _load_oracle_raw(
    protocol: dict[str, Any], session: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, actual, resolved = _read_regular_json(
        session["msm_oracle_raw_path"],
        session["msm_oracle_raw_sha256"],
        "delta1 MSM oracle raw draws",
    )
    required = {
        "schema": ORACLE_RAW_SCHEMA,
        "phase": "development",
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "protocol_sha256": protocol["sha256"],
        "seed": protocol["seed"],
        "formal_training_authorized": False,
    }
    extra = {
        "oracle_panel_name",
        "oracle_panel_sha256",
        "source_session_path",
        "source_session_sha256",
        "source_prerequisite_path",
        "source_prerequisite_sha256",
        "consumption_claim",
        "domains",
    }
    if set(raw) != set(required) | extra or any(
        raw.get(key) != value for key, value in required.items()
    ):
        raise ValueError("delta1 MSM oracle raw identity mismatch")
    if not isinstance(raw.get("oracle_panel_name"), str) or not raw["oracle_panel_name"]:
        raise ValueError("delta1 MSM oracle development panel is invalid")
    _require_sha256(raw.get("oracle_panel_sha256"), "delta1 MSM oracle panel SHA256")
    source_session = load_session(
        raw.get("source_session_path"), raw.get("source_session_sha256")
    )
    source_prerequisite = load_scientific_prerequisite(
        raw.get("source_prerequisite_path"),
        raw.get("source_prerequisite_sha256"),
        session=source_session,
    )
    if (
        source_session["phase"] != "development"
        or source_session["checkpoint_delta"] != 1
        or source_session["msm_oracle_status"] != ORACLE_UNRESOLVED
        or source_session["checkpoint_sha256"] != session["checkpoint_sha256"]
        or source_session["checkpoint_step"] != session["checkpoint_step"]
        or source_session["full_training_contract_sha256"]
        != session["full_training_contract_sha256"]
        or source_session["panel_name"] != raw["oracle_panel_name"]
        or source_session["panel_sha256"] != raw["oracle_panel_sha256"]
        or source_session["panel_domains"] != 20
        or source_session["raw_output"] != str(resolved)
    ):
        raise ValueError("delta1 MSM oracle source is not the exact development evaluator")
    verify_reserved_evaluation_claim(
        source_prerequisite,
        raw["source_prerequisite_sha256"],
        raw.get("consumption_claim"),
        runtime_probe_output=source_session["runtime_probe_output"],
        output=source_session["raw_output"],
    )
    domains = raw.get("domains")
    if not isinstance(domains, list) or len(domains) != 20:
        raise ValueError("delta1 MSM oracle requires exactly 20 development domains")
    expected_grid = {
        (temperature, replica)
        for temperature in protocol["panel"]["temperatures_kelvin"]
        for replica in protocol["panel"]["replicas"]
    }
    domain_ids: list[str] = []
    domain_gains: list[float] = []
    for row in domains:
        if not isinstance(row, dict) or set(row) != {"domain", "cells"}:
            raise ValueError("delta1 MSM oracle raw domain schema mismatch")
        if not isinstance(row["domain"], str) or not row["domain"]:
            raise ValueError("delta1 MSM oracle raw domain id is invalid")
        cells = row["cells"]
        if not isinstance(cells, list) or len(cells) != 25:
            raise ValueError("delta1 MSM oracle raw domain needs exact 5x5 cells")
        actual_grid = []
        cell_gains = []
        for cell in cells:
            if not isinstance(cell, dict):
                raise ValueError("delta1 MSM oracle raw cell schema mismatch")
            actual_grid.append((cell["temperature"], cell["replica"]))
            cell_gains.append(_oracle_cell_gain(cell, protocol))
        if set(actual_grid) != expected_grid or len(set(actual_grid)) != 25:
            raise ValueError("delta1 MSM oracle raw cell grid mismatch")
        domain_ids.append(row["domain"])
        domain_gains.append(float(np.mean(cell_gains)))
    if len(set(domain_ids)) != 20:
        raise ValueError("delta1 MSM oracle development domains are not unique")
    recomputed = _bootstrap_domain_gain(
        domain_gains,
        draws=protocol["adjudication"]["bootstrap_draws"],
        seed=protocol["seed"] + 2,
        quantiles=protocol["adjudication"]["paired_ci_quantiles"],
    )
    return {**raw, "path": str(resolved), "sha256": actual}, recomputed


def load_delta1_oracle_artifact(
    protocol: dict[str, Any], session: dict[str, Any]
) -> dict[str, Any] | None:
    """Open and verify the substantive delta=1 oracle before authorization use."""

    if session["checkpoint_delta"] != 1:
        return None
    if session["msm_oracle_status"] == ORACLE_UNRESOLVED:
        return None
    payload, actual, resolved = _read_regular_json(
        session["msm_oracle_prerequisite_path"],
        session["msm_oracle_prerequisite_sha256"],
        "delta1 MSM oracle artifact",
    )
    raw, recomputed = _load_oracle_raw(protocol, session)
    required = {
        "schema": ORACLE_SCHEMA,
        "status": ORACLE_PASS,
        "decision": "PASS",
        "evidence_type": "measured_delta1_msm_oracle",
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "protocol_sha256": protocol["sha256"],
        "seed": protocol["seed"],
        "raw_draws_path": raw["path"],
        "raw_draws_sha256": raw["sha256"],
        "decision_rule": "domain_bootstrap_ci95_lower_gt_zero",
        "formal_training_authorized": False,
    }
    if set(payload) != set(required):
        raise ValueError("delta1 MSM oracle artifact is not an exact bound PASS")
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("delta1 MSM oracle artifact is not an exact bound PASS")
    if recomputed["paired_msm_gain_ci95_lower"] <= 0:
        raise ValueError("delta1 MSM oracle recomputed CI does not substantiate PASS")
    return {
        **payload,
        "path": str(resolved),
        "sha256": actual,
        "raw_draws": raw,
        "recomputed": recomputed,
    }


def _load_runtime_raw(
    protocol: dict[str, Any],
    session: dict[str, Any],
    path: str | Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], float]:
    raw, actual, resolved = _read_regular_json(
        path, expected_sha256, "scientific runtime raw measurement"
    )
    settings = protocol["runtime_feasibility"]
    required = {
        "schema": settings["raw_schema"],
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
        "protocol_sha256": protocol["sha256"],
        "projected_domains": session["panel_domains"],
        "probe_source_role": settings["probe_source_role"],
        "workload": settings["workload"],
        "formal_training_authorized": False,
    }
    if set(raw) != set(required) | {"probe_domains", "gpu_environment", "peak_memory_bytes"}:
        raise ValueError("runtime raw measurement exact schema mismatch")
    if any(raw.get(key) != value for key, value in required.items()):
        raise ValueError("runtime raw measurement identity mismatch")
    probe_domains = raw.get("probe_domains")
    if not isinstance(probe_domains, list) or len(probe_domains) != settings["representative_domains"]:
        raise ValueError("runtime-feasibility requires exact multi-domain probes")
    domain_ids: list[str] = []
    elapsed_by_domain: list[float] = []
    largest_flags = []
    for row in probe_domains:
        if not isinstance(row, dict) or set(row) != {
            "domain", "payload_sha256", "payload_size_bytes",
            "is_largest_non_reserved_payload", "cells"
        }:
            raise ValueError("runtime-feasibility probe-domain schema mismatch")
        if not isinstance(row["domain"], str) or not row["domain"]:
            raise ValueError("runtime-feasibility probe domain is invalid")
        _require_sha256(row["payload_sha256"], "runtime probe payload SHA256")
        if type(row["payload_size_bytes"]) is not int or row["payload_size_bytes"] <= 0:
            raise ValueError("runtime probe payload size is invalid")
        if type(row["is_largest_non_reserved_payload"]) is not bool:
            raise ValueError("runtime largest-payload flag is invalid")
        cells = row["cells"]
        if not isinstance(cells, list) or len(cells) != settings["representative_cells_per_domain"]:
            raise ValueError("runtime probe requires exactly 20 raw cell timings per domain")
        coordinates = []
        elapsed_values = []
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != {
                "temperature", "replica", "elapsed_seconds"
            }:
                raise ValueError("runtime raw cell timing schema mismatch")
            coordinates.append((cell["temperature"], cell["replica"]))
            value = cell["elapsed_seconds"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError("runtime raw cell timing is invalid")
            elapsed_values.append(float(value))
        allowed = {
            (temperature, replica)
            for temperature in protocol["panel"]["temperatures_kelvin"]
            for replica in protocol["panel"]["replicas"]
        }
        if len(set(coordinates)) != len(coordinates) or not set(coordinates).issubset(allowed):
            raise ValueError("runtime raw cell identities are invalid")
        domain_ids.append(row["domain"])
        elapsed_by_domain.append(float(sum(elapsed_values)))
        largest_flags.append(row["is_largest_non_reserved_payload"])
    if len(set(domain_ids)) != settings["representative_domains"]:
        raise ValueError("runtime probe domains are not unique")
    if largest_flags.count(True) != 1:
        raise ValueError("runtime probe must include exactly one largest non-reserved payload")
    gpu = raw.get("gpu_environment")
    if not isinstance(gpu, dict) or set(gpu) != {
        "device_name", "device_uuid", "driver_version", "cuda_runtime_version",
        "compute_capability"
    } or any(not isinstance(value, str) or not value for value in gpu.values()):
        raise ValueError("runtime GPU environment evidence is incomplete")
    peak_memory = raw.get("peak_memory_bytes")
    if type(peak_memory) is not int or peak_memory <= 0:
        raise ValueError("runtime peak-memory evidence is invalid")
    elapsed = float(sum(elapsed_by_domain))
    expected_projected = (
        elapsed
        * session["panel_domains"]
        / settings["representative_domains"]
        * settings["projection_safety_factor"]
    )
    return {**raw, "path": str(resolved), "sha256": actual}, expected_projected


def load_runtime_feasibility_artifact(
    protocol: dict[str, Any], session: dict[str, Any]
) -> dict[str, Any]:
    """Recompute projection from a hash-bound five-domain raw measurement."""

    payload, actual, resolved = _read_regular_json(
        session["runtime_feasibility_path"],
        session["runtime_feasibility_sha256"],
        "scientific runtime-feasibility artifact",
    )
    settings = protocol["runtime_feasibility"]
    required = {
        "schema": settings["schema"],
        "status": settings["status"],
        "execution_mode": settings["execution_mode"],
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
        "protocol_sha256": protocol["sha256"],
        "measured_domains": settings["representative_domains"],
        "projected_domains": session["panel_domains"],
        "projection_safety_factor": settings["projection_safety_factor"],
        "probe_source_role": settings["probe_source_role"],
        "workload": settings["workload"],
        "max_projected_seconds": settings["max_projected_seconds_by_phase"][
            session["phase"]
        ],
        "formal_training_authorized": False,
    }
    if set(payload) != set(required) | {"raw_measurement_path", "raw_measurement_sha256"}:
        raise ValueError("runtime-feasibility artifact exact schema mismatch")
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("runtime-feasibility artifact identity mismatch")
    raw, expected_projected = _load_runtime_raw(
        protocol,
        session,
        payload["raw_measurement_path"],
        payload["raw_measurement_sha256"],
    )
    if expected_projected > required["max_projected_seconds"]:
        raise ValueError("STOP_PROJECTED_SCIENTIFIC_RUNTIME")
    return {
        **payload,
        "path": str(resolved),
        "sha256": actual,
        "raw_measurement": raw,
        "probe_domains": raw["probe_domains"],
        "gpu_environment": raw["gpu_environment"],
        "peak_memory_bytes": raw["peak_memory_bytes"],
        "recomputed_projected_total_seconds": expected_projected,
        "measured_elapsed_seconds": expected_projected
        / session["panel_domains"]
        * settings["representative_domains"]
        / settings["projection_safety_factor"],
    }


def load_bundle_prerequisites_for_mode(
    protocol: dict[str, Any],
    session: dict[str, Any],
    qualification_mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load existing qualification artifacts only for the final bundle mode."""

    if qualification_mode == "delta1_oracle":
        if (
            session["checkpoint_delta"] != 1
            or session["msm_oracle_status"] != ORACLE_UNRESOLVED
        ):
            raise ValueError("delta1 oracle qualification requires an unresolved delta=1 session")
        return None, None
    if qualification_mode == "runtime":
        return None, None
    if qualification_mode != "bundle":
        raise ValueError("scientific qualification mode is invalid")
    if (
        session["checkpoint_delta"] == 1
        and session["msm_oracle_status"] == ORACLE_UNRESOLVED
    ):
        raise ValueError("delta1 MSM oracle unresolved; authorization preserved")
    return (
        load_delta1_oracle_artifact(protocol, session),
        load_runtime_feasibility_artifact(protocol, session),
    )


def verify_runtime_probe_bindings(
    runtime: dict[str, Any], contract: str | Path, session: dict[str, Any]
) -> None:
    """Bind every runtime probe to contracted payloads and the exclusion set."""

    contract_payload, _, contract_path = _read_regular_json(
        contract,
        session["full_training_contract_sha256"],
        "runtime full-training contract",
    )
    artifacts = contract_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("runtime contract artifact set is invalid")

    def artifact(name: str, label: str) -> tuple[object, str]:
        row = artifacts.get(name)
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError(f"runtime contract {label} identity is invalid")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"runtime contract {label} path is invalid")
        payload, actual, _ = _read_regular_json_value(
            contract_path.parent / relative, row["sha256"], label
        )
        return payload, actual

    manifest, manifest_sha = artifact("manifest", "runtime contracted manifest")
    registry, registry_sha = artifact(
        "panel_registry", "runtime contracted exclusion registry"
    )
    if manifest_sha != session["data_manifest_sha256"]:
        raise ValueError("runtime manifest binding mismatch")
    if registry_sha != session["evaluation_exclusion_registry_sha256"]:
        raise ValueError("runtime exclusion-registry binding mismatch")
    train_row = artifacts.get("train_list")
    if not isinstance(train_row, dict) or set(train_row) != {"path", "sha256"}:
        raise ValueError("runtime contracted train-list identity is invalid")
    relative = Path(train_row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("runtime contracted train-list path is invalid")
    train_path = contract_path.parent / relative
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(train_path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        train_raw = handle.read()
        after = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(before.st_mode)
        or _stat_identity(before) != _stat_identity(after)
        or hashlib.sha256(train_raw).hexdigest() != train_row["sha256"]
    ):
        raise ValueError("runtime contracted train-list changed or mismatched")
    try:
        train_domains = train_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("runtime contracted train-list is invalid UTF-8") from exc
    if not isinstance(manifest, list) or not isinstance(registry, dict):
        raise ValueError("runtime manifest or exclusion registry schema mismatch")
    by_domain = {
        row.get("domain"): row
        for row in manifest
        if isinstance(row, dict) and isinstance(row.get("domain"), str)
    }
    if len(by_domain) != len(manifest):
        raise ValueError("runtime contracted manifest domain identity mismatch")
    train_set = set(train_domains)
    if not train_set or not train_set.issubset(by_domain):
        raise ValueError("runtime train-list differs from contracted manifest")
    largest_size = max(int(by_domain[domain]["size"]) for domain in train_domains)
    ordered = sorted(train_domains, key=lambda domain: (int(by_domain[domain]["size"]), domain))
    expected_domains = [
        ordered[int((len(ordered) - 1) * numerator / 4)]
        for numerator in range(5)
    ]
    if [probe["domain"] for probe in runtime["probe_domains"]] != expected_domains:
        raise ValueError("runtime probes do not match the frozen size-quantile selection")
    for probe in runtime["probe_domains"]:
        domain = probe["domain"]
        if domain not in train_set:
            raise ValueError("runtime probe domain is reserved or uncontracted")
        row = by_domain[domain]
        if (
            probe["payload_sha256"] != row.get("sha256")
            or probe["payload_size_bytes"] != row.get("size")
        ):
            raise ValueError("runtime probe payload identity mismatch")
        expected_largest = probe["payload_size_bytes"] == largest_size
        if probe["is_largest_non_reserved_payload"] is not expected_largest:
            raise ValueError("runtime largest non-reserved payload claim mismatch")


def _gpu_environment(device: torch.device) -> dict[str, str]:
    properties = torch.cuda.get_device_properties(device)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
            f"--id={device.index or 0}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip().split(",", 1)
    if len(query) != 2 or not all(value.strip() for value in query):
        raise ValueError("runtime GPU UUID/driver query is incomplete")
    return {
        "device_name": properties.name,
        "device_uuid": query[0].strip(),
        "driver_version": query[1].strip(),
        "cuda_runtime_version": str(torch.version.cuda),
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


@torch.no_grad()
def execute_runtime_feasibility_measurement(
    *,
    protocol: dict[str, Any],
    session: dict[str, Any],
    checkpoint: str | Path,
    contract: str | Path,
    data_root: str | Path,
    probe_plan: list[dict[str, Any]],
    raw_output: str | Path,
    decision_output: str | Path,
) -> dict[str, Any]:
    """Actually execute the frozen 5-domain x 20-cell workload on CUDA."""

    settings = protocol["runtime_feasibility"]
    if Path(raw_output).exists() or Path(decision_output).exists():
        raise FileExistsError("runtime measurement outputs must both be fresh")
    if len(probe_plan) != settings["representative_domains"]:
        raise ValueError("runtime executor requires exactly five probe domains")
    identities = []
    paths: dict[str, Path] = {}
    root = Path(data_root).expanduser().resolve()
    for row in probe_plan:
        if not isinstance(row, dict) or set(row) != {
            "domain", "path", "payload_sha256", "payload_size_bytes",
            "is_largest_non_reserved_payload"
        }:
            raise ValueError("runtime executor probe-plan schema mismatch")
        path = Path(row["path"]).expanduser().resolve()
        if root not in path.parents:
            raise ValueError("runtime probe payload is outside the contracted data root")
        _, actual_sha, resolved = _read_stable_regular_bytes(
            path, f"runtime probe payload {row['domain']}", row["payload_sha256"]
        )
        if resolved.stat().st_size != row["payload_size_bytes"]:
            raise ValueError("runtime probe payload size mismatch")
        paths[row["domain"]] = resolved
        identities.append({
            key: row[key]
            for key in (
                "domain", "payload_sha256", "payload_size_bytes",
                "is_largest_non_reserved_payload"
            )
        } | {"cells": []})
    verify_runtime_probe_bindings({"probe_domains": identities}, contract, session)

    checkpoint_payload, _ = _load_verified_checkpoint(
        Path(checkpoint).expanduser().resolve(), session["checkpoint_sha256"]
    )
    data_cfg = checkpoint_payload["cfg"]["data"]
    model_cfg = checkpoint_payload["cfg"]["model"]
    device = resolve_device(checkpoint_payload["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("runtime feasibility measurement requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    grid = [
        (temperature, replica)
        for temperature in protocol["panel"]["temperatures_kelvin"]
        for replica in protocol["panel"]["replicas"]
    ][: settings["representative_cells_per_domain"]]
    measured_domains = []
    for domain_index, identity in enumerate(identities):
        handle = _DomainHandle(paths[identity["domain"]])
        try:
            cells = []
            for cell_index, (temperature, replica) in enumerate(grid):
                torch.cuda.synchronize(device)
                started = time.monotonic_ns()
                evaluate_conditional_cell(
                    handle=handle,
                    layout=handle.layout,
                    model=model,
                    device=device,
                    data_cfg=data_cfg,
                    delta=session["checkpoint_delta"],
                    temperature=temperature,
                    replica=replica,
                    replicas=protocol["panel"]["replicas"],
                    protocol=protocol,
                    seed_offset=domain_index * 10000 + cell_index,
                )
                evaluate_geometry_cell(
                    handle=handle,
                    layout=handle.layout,
                    model=model,
                    device=device,
                    data_cfg=data_cfg,
                    delta=session["checkpoint_delta"],
                    temperature=temperature,
                    replica=replica,
                    protocol=protocol,
                )
                torch.cuda.synchronize(device)
                elapsed = (time.monotonic_ns() - started) / 1_000_000_000.0
                if not np.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("runtime monotonic cell timing is invalid")
                cells.append({
                    "temperature": temperature,
                    "replica": replica,
                    "elapsed_seconds": elapsed,
                })
        finally:
            handle.close()
        _read_stable_regular_bytes(
            paths[identity["domain"]],
            f"runtime probe payload post-run {identity['domain']}",
            identity["payload_sha256"],
        )
        measured_domains.append({**{k: v for k, v in identity.items() if k != "cells"}, "cells": cells})
    raw = {
        "schema": settings["raw_schema"],
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
        "protocol_sha256": protocol["sha256"],
        "projected_domains": session["panel_domains"],
        "probe_source_role": settings["probe_source_role"],
        "workload": settings["workload"],
        "probe_domains": measured_domains,
        "gpu_environment": _gpu_environment(device),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "formal_training_authorized": False,
    }
    _write_new_json(raw_output, raw)
    _, sha256, resolved = _read_stable_regular_bytes(
        raw_output, "runtime executor raw output"
    )
    reopened, projected = _load_runtime_raw(
        protocol, session, resolved, sha256
    )
    maximum = settings["max_projected_seconds_by_phase"].get(reopened["phase"])
    if not isinstance(maximum, (int, float)) or projected > maximum:
        raise ValueError("STOP_PROJECTED_SCIENTIFIC_RUNTIME")
    decision = {
        "schema": settings["schema"],
        "status": settings["status"],
        "execution_mode": settings["execution_mode"],
        "phase": reopened["phase"],
        "checkpoint_sha256": reopened["checkpoint_sha256"],
        "checkpoint_step": reopened["checkpoint_step"],
        "full_training_contract_sha256": reopened["full_training_contract_sha256"],
        "data_manifest_sha256": reopened["data_manifest_sha256"],
        "evaluation_exclusion_registry_sha256": reopened[
            "evaluation_exclusion_registry_sha256"
        ],
        "panel_name": reopened["panel_name"],
        "panel_sha256": reopened["panel_sha256"],
        "protocol_sha256": protocol["sha256"],
        "measured_domains": settings["representative_domains"],
        "projected_domains": reopened["projected_domains"],
        "projection_safety_factor": settings["projection_safety_factor"],
        "probe_source_role": settings["probe_source_role"],
        "workload": settings["workload"],
        "max_projected_seconds": maximum,
        "raw_measurement_path": reopened["path"],
        "raw_measurement_sha256": reopened["sha256"],
        "formal_training_authorized": False,
    }
    _write_new_json(decision_output, decision)
    _, decision_sha, decision_path = _read_stable_regular_bytes(
        decision_output, "scientific runtime-feasibility decision"
    )
    return {
        "status": settings["status"],
        "decision_path": str(decision_path),
        "decision_sha256": decision_sha,
        "raw_measurement_sha256": reopened["sha256"],
        "recomputed_projected_total_seconds": projected,
        "formal_training_authorized": False,
    }


def validate_bundle_bindings(
    protocol: dict[str, Any],
    session: dict[str, Any],
    prerequisite: dict[str, Any],
    *,
    expected_repo_commit: str,
) -> None:
    if session["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("scientific session protocol SHA256 mismatch")
    if session["repo_commit"] != expected_repo_commit:
        raise ValueError("scientific session repository commit mismatch")
    expected_domains = protocol["panel"]["phase_domain_counts"][session["phase"]]
    if session["panel_domains"] != expected_domains:
        raise ValueError("scientific session panel count does not match its phase")
    if prerequisite["scientific_session_sha256"] != session["sha256"]:
        raise ValueError("scientific prerequisite session SHA256 mismatch")
    if prerequisite["scientific_protocol_sha256"] != protocol["sha256"]:
        raise ValueError("scientific prerequisite protocol SHA256 mismatch")


def preflight_output_paths(session: dict[str, Any]) -> None:
    paths = [
        Path(session["runtime_probe_output"]),
        Path(session["raw_output"]),
        Path(session["decision_output"]),
        Path(session["state_archive_output"]),
    ]
    if session["phase"] == "untouched":
        paths.extend(
            Path(session[key])
            for key in (
                "global_claim_descriptor_path",
                "global_claim_receipt_path",
                "global_claim_readback_path",
            )
        )
    if len({str(path) for path in paths}) != len(paths):
        raise ValueError("scientific session output paths must be distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing scientific output: {path}")


def verify_scientific_data_prerequisite(
    contract: str | Path, expected_contract_sha256: str
) -> dict[str, Any]:
    """Independently require the complete live data audit used by science.

    A missing or incomplete audit is a data-prerequisite failure.  It is never
    evidence that the model failed a scientific metric.
    """

    contract_payload, _, contract_path = _read_regular_json(
        contract, expected_contract_sha256, "scientific full-data contract"
    )
    row = contract_payload.get("artifacts", {}).get("data_audit")
    if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    audit, audit_sha256, audit_path = _read_regular_json(
        contract_path.parent / relative, row["sha256"], "scientific full-data audit"
    )
    required = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "domains": EXPECTED_DOMAINS,
        "h5_files": EXPECTED_DOMAINS,
        "h5_bytes": EXPECTED_H5_BYTES,
        "trajectories": EXPECTED_TRAJECTORIES,
        "hdf5_files_structurally_verified": EXPECTED_DOMAINS,
        "payload_hash_verification_mode": "full_rehash",
        "payload_hashes_verified": EXPECTED_DOMAINS,
        "data_gate_passed": True,
        "live_payload_bytes_rehashed": True,
        "coordinate_finiteness_scope": "first_and_last_frame_per_trajectory",
        "finite_endpoint_frames_verified": EXPECTED_TRAJECTORIES * 2,
        "formal_training_authorized": False,
    }
    if any(audit.get(key) != value for key, value in required.items()):
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    return {
        "status": "PASS_CONTRACTED_DATA_PREREQUISITE",
        "data_audit_path": str(audit_path),
        "data_audit_sha256": audit_sha256,
        "domains": EXPECTED_DOMAINS,
        "h5_files": EXPECTED_DOMAINS,
        "h5_bytes": EXPECTED_H5_BYTES,
        "trajectories": EXPECTED_TRAJECTORIES,
        "hdf5_files_structurally_verified": EXPECTED_DOMAINS,
        "formal_training_authorized": False,
    }


def establish_claimed_inputs(
    *,
    protocol: dict[str, Any],
    session: dict[str, Any],
    prerequisite: dict[str, Any],
    prerequisite_sha256: str,
    checkpoint: str | Path,
    contract: str | Path,
    panel_file: str | Path,
    data_root: str | Path,
    oracle_artifact: dict[str, Any] | None,
    runtime_feasibility: dict[str, Any],
) -> dict[str, Any]:
    """Consume once, then verify and pin all protected inputs.

    This function is ready for the later numerical evaluator.  The public CLI
    never reaches it while ``NUMERICAL_KERNEL_IMPLEMENTED`` is false.
    """

    claim = claim_reserved_evaluation(
        prerequisite,
        prerequisite_sha256,
        runtime_probe_output=session["runtime_probe_output"],
        output=session["raw_output"],
    )
    identity = verify_frozen_evaluation_identity(
        checkpoint,
        contract,
        session["full_training_contract_sha256"],
        expected_checkpoint_sha256=session["checkpoint_sha256"],
        expected_checkpoint_step=session["checkpoint_step"],
        phase=session["phase"],
        panel_name=session["panel_name"],
        panel_file=panel_file,
    )
    if identity["panel_sha256"] != session["panel_sha256"]:
        raise ValueError("claimed scientific panel identity mismatch")
    data_prerequisite = verify_scientific_data_prerequisite(
        contract, session["full_training_contract_sha256"]
    )
    checkpoint_payload, checkpoint_sha256 = _load_verified_checkpoint(
        Path(checkpoint).expanduser().resolve(), session["checkpoint_sha256"]
    )
    if checkpoint_sha256 != identity["checkpoint_sha256"]:
        raise ValueError("scientific checkpoint changed during claimed identity verification")
    data_cfg = checkpoint_payload.get("cfg", {}).get("data", {})
    if require_single_delta(data_cfg.get("delta_frames")) != session["checkpoint_delta"]:
        raise ValueError("scientific session delta differs from the checkpoint")
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg.get("temperatures"), data_cfg.get("replicas")
    )
    if temperatures != protocol["panel"]["temperatures_kelvin"]:
        raise ValueError("scientific checkpoint temperature grid differs from the protocol")
    if replicas != protocol["panel"]["replicas"]:
        raise ValueError("scientific checkpoint replica grid differs from the protocol")
    canonical_data_root = str(Path(data_root).expanduser().resolve())
    if canonical_data_root != session["data_root"]:
        raise ValueError("claimed scientific data root differs from the session")
    if str(Path(data_cfg.get("root", "")).expanduser().resolve()) != canonical_data_root:
        raise ValueError("scientific session data root differs from the checkpoint")
    payload = rehash_contracted_panel_payloads(
        contract, panel_file, canonical_data_root, keep_open=True
    )
    if payload["panel_domains"] != session["panel_domains"]:
        payload["pins"].close()
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    try:
        global_obs_claim = establish_global_obs_claim(protocol, session)
    except BaseException:
        payload["pins"].close()
        raise
    return {
        "protocol": protocol,
        "session": session,
        "prerequisite": prerequisite,
        "consumption_claim": claim,
        "identity": identity,
        "payload_verification": payload,
        "data_prerequisite": data_prerequisite,
        "oracle_artifact": oracle_artifact,
        "runtime_feasibility": runtime_feasibility,
        "global_obs_claim": global_obs_claim,
        "formal_training_authorized": False,
    }


def _write_new_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _read_stable_regular_bytes(
    path: str | Path, label: str, expected_sha256: str | None = None
) -> tuple[bytes, str, Path]:
    configured = Path(path).expanduser()
    if configured.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = configured.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"cannot open {label}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(before.st_mode)
        or _stat_identity(before) != _stat_identity(after)
        or len(raw) != before.st_size
    ):
        raise ValueError(f"{label} changed while it was being read")
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    return raw, actual, resolved


def _run_reviewed_obs_conditional_create(
    protocol: dict[str, Any], descriptor_path: str, receipt_path: str, readback_path: str
) -> None:
    settings = protocol["untouched_global_claim"]
    helper_raw = OBS_CONDITIONAL_CREATE_HELPER_SOURCE.encode("utf-8")
    helper_sha = hashlib.sha256(helper_raw).hexdigest()
    if helper_sha != settings["helper_sha256"]:
        raise ValueError("embedded OBS conditional-create helper SHA256 mismatch")
    descriptor_parent = Path(descriptor_path).parent
    descriptor_fd, verified_helper_name = tempfile.mkstemp(
        prefix=".reviewed-obs-conditional-create.", dir=descriptor_parent
    )
    verified_helper = Path(verified_helper_name)
    try:
        with os.fdopen(descriptor_fd, "wb") as handle:
            handle.write(helper_raw)
            handle.flush()
            os.fsync(handle.fileno())
        subprocess.run(
            [
                settings["obs_sdk_python"],
                str(verified_helper),
                descriptor_path,
                receipt_path,
                readback_path,
            ],
            check=True,
            timeout=120,
        )
    finally:
        verified_helper.unlink(missing_ok=True)


def establish_global_obs_claim(
    protocol: dict[str, Any], session: dict[str, Any]
) -> dict[str, Any]:
    """Perform the untouched one-shot conditional create inside the core gate."""

    if session["phase"] != "untouched":
        return {"required": False, "completed": False}
    helper_sha = protocol["untouched_global_claim"]["helper_sha256"]
    payload = {
        "schema": GLOBAL_CLAIM_SCHEMA,
        "repo_commit": session["repo_commit"],
        "protocol_sha256": protocol["sha256"],
        "session_sha256": session["sha256"],
        "authorization_id": session["authorization_id"],
        "panel_name": session["panel_name"],
        "panel_sha256": session["panel_sha256"],
        "phase": "untouched",
        "formal_training_authorized": False,
    }
    payload_raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    payload_sha = hashlib.sha256(payload_raw).hexdigest()
    uri = (
        f"{session['obs_prefix'].split('/deepjump-scientific/', 1)[0]}"
        "/deepjump-scientific/one-shot-claims/"
        f"{session['authorization_id']}/{session['panel_sha256']}.json"
    )
    descriptor = {
        "schema": "deepjump.contracted_scientific_global_claim_descriptor.v1",
        "uri": uri,
        "payload": payload,
        "payload_sha256": payload_sha,
        "payload_size_bytes": len(payload_raw),
        "helper_sha256": helper_sha,
    }
    _write_new_json(session["global_claim_descriptor_path"], descriptor)
    _, descriptor_sha, descriptor_path = _read_stable_regular_bytes(
        session["global_claim_descriptor_path"], "global OBS claim descriptor"
    )
    _run_reviewed_obs_conditional_create(
        protocol,
        session["global_claim_descriptor_path"],
        session["global_claim_receipt_path"],
        session["global_claim_readback_path"],
    )
    receipt_raw, receipt_sha, receipt_path = _read_stable_regular_bytes(
        session["global_claim_receipt_path"], "global OBS claim receipt"
    )
    readback_raw, readback_sha, readback_path = _read_stable_regular_bytes(
        session["global_claim_readback_path"], "global OBS claim readback"
    )
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("global OBS claim receipt is invalid JSON") from exc
    expected_receipt = {
        "schema": GLOBAL_CLAIM_RECEIPT_SCHEMA,
        "created": True,
        "condition": "If-None-Match:*",
        "uri": uri,
        "payload_sha256": payload_sha,
        "payload_size_bytes": len(payload_raw),
        "helper_sha256": helper_sha,
    }
    if receipt != expected_receipt:
        raise ValueError("global OBS conditional-create receipt mismatch")
    if readback_raw != payload_raw or readback_sha != payload_sha:
        raise ValueError("global OBS conditional-create readback bytes mismatch")
    return {
        "required": True,
        "completed": True,
        "helper_sha256": helper_sha,
        "descriptor_path": str(descriptor_path),
        "descriptor_sha256": descriptor_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "readback_path": str(readback_path),
        "readback_sha256": readback_sha,
        "payload_sha256": payload_sha,
        "payload_size_bytes": len(payload_raw),
        "uri": uri,
    }


def _repeat_batch(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {
        key: value.repeat(count, *([1] * (value.ndim - 1)))
        for key, value in batch.items()
    }


def _guard_observations(
    positions: torch.Tensor,
    vectors: torch.Tensor,
    bond_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    distances = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)
    rows = []
    for index in range(len(positions)):
        p_finite = bool(torch.isfinite(positions[index]).all().item())
        v_finite = bool(torch.isfinite(vectors[index]).all().item())
        selected = distances[index][bond_mask[index]]
        geometry_finite = bool(torch.isfinite(selected).all().item())
        mean = float(selected.mean().item()) if geometry_finite else None
        maximum = float(selected.max().item()) if geometry_finite else None
        rows.append({
            "position_finite": p_finite,
            "vector_finite": v_finite,
            "bond_mean": mean,
            "bond_max": maximum,
        })
    return rows


def _json_vectors(values: np.ndarray) -> list:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("scientific vector evidence must have at least one dimension")

    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None

    return convert(values.tolist())


def _state_sha256(positions: torch.Tensor, vectors: torch.Tensor) -> str:
    """Hash exact canonical P/V tensor bytes, including dtype and shape."""

    digest = hashlib.sha256()
    for label, tensor in ((b"P", positions), (b"V", vectors)):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(label)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_hashes(positions: torch.Tensor, vectors: torch.Tensor) -> list[str]:
    if len(positions) != len(vectors):
        raise ValueError("P/V state batches must have equal length")
    return [_state_sha256(positions[i], vectors[i]) for i in range(len(positions))]


class StateArchiveWriter:
    """Collect canonical P/V arrays and publish one immutable NPZ companion."""

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}

    def add(self, name: str, positions: torch.Tensor, vectors: torch.Tensor) -> list[str]:
        if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
            raise ValueError("state archive key is invalid")
        if name + ".P" in self._arrays or len(positions) != len(vectors):
            raise ValueError("state archive key is duplicated or P/V shape mismatched")
        p = np.ascontiguousarray(positions.detach().cpu().numpy())
        v = np.ascontiguousarray(vectors.detach().cpu().numpy())
        self._arrays[name + ".P"] = p
        self._arrays[name + ".V"] = v
        return _state_hashes(positions, vectors)

    def publish(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **self._arrays)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        _, sha256, resolved = _read_stable_regular_bytes(
            destination, "scientific state archive"
        )
        return {
            "schema": STATE_ARCHIVE_SCHEMA,
            "path": str(resolved),
            "sha256": sha256,
            "arrays": len(self._arrays),
        }


def _trajectory_features(
    handle: _DomainHandle,
    layout,
    temperature: int,
    replica: int,
    frames: np.ndarray,
    residue_slice: slice,
    pairs: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    rows = []
    for frame in frames:
        coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, int(frame)))
        )
        positions, _ = apply_layout(coordinates, layout)
        positions = positions[residue_slice]
        positions = positions - positions.mean(0, keepdim=True)
        rows.append(pairdist_features(positions, pairs))
    return np.stack(rows)


def _model_batch(
    layout,
    residue_slice: slice,
    positions: torch.Tensor,
    vectors: torch.Tensor,
    delta: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count, n = positions.shape[:2]
    topology = torch.as_tensor(
        layout.bond_mask[residue_slice.start : residue_slice.stop - 1],
        dtype=torch.bool,
        device=device,
    )
    return {
        "P_t": positions.to(device),
        "V_t": vectors.to(device),
        "res_index": torch.as_tensor(
            layout.res_index[residue_slice], device=device
        )[None].repeat(count, 1),
        "bond_mask": topology[None].repeat(count, 1),
        "delta_ns": torch.full((count,), float(delta), device=device),
        "residue_mask": torch.ones(count, n, dtype=torch.bool, device=device),
        "atom_mask": torch.as_tensor(
            layout.atom_mask[residue_slice], device=device
        )[None].repeat(count, 1, 1),
    }


def evaluate_conditional_cell(
    *,
    handle: _DomainHandle,
    layout,
    model: DeepJumpLite,
    device: torch.device,
    data_cfg: dict[str, Any],
    delta: int,
    temperature: int,
    replica: int,
    replicas: list[int],
    protocol: dict[str, Any],
    seed_offset: int,
    state_archive: StateArchiveWriter | None = None,
    state_archive_prefix: str | None = None,
) -> dict[str, Any]:
    """Emit raw 4-TIC coordinates and guard facts, never a score or verdict."""

    settings = protocol["conditional_transition"]
    reference_replica = replicas[(replicas.index(replica) + 1) % len(replicas)]
    reference = handle.replicas(temperature, [reference_replica])
    evaluation = handle.replicas(temperature, [replica])
    if len(reference) != 1 or len(evaluation) != 1:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    n = min(layout.num_residues, int(data_cfg["crop_length"]))
    offset = max(0, (layout.num_residues - n) // 2)
    residue_slice = slice(offset, offset + n)
    pairs = selected_pair_indices(n, settings["tica"]["max_pair_features"])
    fit_frames = contiguous_frame_ids(reference[0][2], settings["real_frames_per_cell"])
    fit_features = _trajectory_features(
        handle, layout, temperature, reference_replica, fit_frames, residue_slice, pairs
    )
    feature_mean, projection = fit_tica(
        fit_features,
        lag=settings["tica"]["lag_ns"],
        n_components=settings["tica"]["components"],
    )
    real_tic = (fit_features - feature_mean) @ projection

    eval_frames = contiguous_frame_ids(evaluation[0][2], settings["real_frames_per_cell"])
    possible = eval_frames[:-delta]
    if len(possible) < settings["starts_per_cell"]:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    starts = possible[
        np.linspace(0, len(possible) - 1, settings["starts_per_cell"], dtype=int)
    ]
    source_positions, source_vectors, target_positions = [], [], []
    for frame in starts:
        source_coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, int(frame)))
        )
        target_coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, int(frame + delta)))
        )
        source_p, source_v = apply_model_layout(
            source_coordinates,
            layout,
            canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
        )
        target_p, _ = apply_layout(target_coordinates, layout)
        source_p = source_p[residue_slice]
        target_p = target_p[residue_slice]
        source_positions.append(source_p - source_p.mean(0, keepdim=True))
        source_vectors.append(source_v[residue_slice])
        target_positions.append(target_p - target_p.mean(0, keepdim=True))
    source_positions_t = torch.stack(source_positions).to(device)
    source_vectors_t = torch.stack(source_vectors).to(device)
    target_positions_t = torch.stack(target_positions).to(device)
    base_batch = _model_batch(
        layout,
        residue_slice,
        source_positions_t,
        source_vectors_t,
        delta,
        device,
    )
    draws = settings["draws_per_start"]
    raw_tic, guarded_tic, accepted_rows, guard_rows = [], [], [], []
    raw_state_hashes, selected_state_hashes = [], []
    source_state_hashes = (
        state_archive.add(
            f"{state_archive_prefix}.source", source_positions_t, source_vectors_t
        )
        if state_archive is not None and state_archive_prefix is not None
        else _state_hashes(source_positions_t, source_vectors_t)
    )
    for start_index in range(settings["starts_per_cell"]):
        one = {key: value[start_index : start_index + 1] for key, value in base_batch.items()}
        expanded = _repeat_batch(one, draws)
        generator = torch.Generator().manual_seed(
            protocol["seed"] + seed_offset * 100 + start_index
        )
        raw_p, raw_v = model.sample(
            expanded, mode="ode", steps=1, generator=generator
        )
        guarded_p, guarded_v, accepted = reject_to_source(
            raw_p,
            raw_v,
            expanded["P_t"],
            expanded["V_t"],
            expanded["bond_mask"],
            lo=settings["guard"]["bond_mean_gt_angstrom"],
            hi=settings["guard"]["bond_mean_lt_angstrom"],
            max_bond=settings["guard"]["bond_max_lt_angstrom"],
        )
        raw_tic.append((pairdist_features(raw_p, pairs) - feature_mean) @ projection)
        guarded_tic.append(
            (pairdist_features(guarded_p, pairs) - feature_mean) @ projection
        )
        accepted_rows.append([bool(value) for value in accepted.tolist()])
        guard_rows.append(_guard_observations(raw_p, raw_v, expanded["bond_mask"]))
        raw_state_hashes.append(
            state_archive.add(f"{state_archive_prefix}.raw{start_index}", raw_p, raw_v)
            if state_archive is not None and state_archive_prefix is not None
            else _state_hashes(raw_p, raw_v)
        )
        selected_state_hashes.append(
            state_archive.add(
                f"{state_archive_prefix}.selected{start_index}", guarded_p, guarded_v
            )
            if state_archive is not None and state_archive_prefix is not None
            else _state_hashes(guarded_p, guarded_v)
        )

    source_tic = (pairdist_features(source_positions_t, pairs) - feature_mean) @ projection
    target_tic = (pairdist_features(target_positions_t, pairs) - feature_mean) @ projection
    source_guard = _guard_observations(
        source_positions_t, source_vectors_t, base_batch["bond_mask"]
    )
    return {
        "temperature": int(temperature),
        "replica": int(replica),
        "reference_replica": int(reference_replica),
        "start_frames": [int(value) for value in starts],
        "delta_frames": int(delta),
        "draws": int(draws),
        "tica_components": int(settings["tica"]["components"]),
        "real_tic": _json_vectors(real_tic),
        "source_tic": _json_vectors(source_tic),
        "target_tic": _json_vectors(target_tic),
        "raw_predicted_tic": _json_vectors(np.stack(raw_tic)),
        "guarded_predicted_tic": _json_vectors(np.stack(guarded_tic)),
        "accepted": accepted_rows,
        "state_archive_prefix": state_archive_prefix,
        "source_state_sha256": source_state_hashes,
        "raw_state_sha256": raw_state_hashes,
        "selected_state_sha256": selected_state_hashes,
        "source_guard_observations": source_guard,
        "raw_guard_observations": guard_rows,
    }


@torch.no_grad()
def execute_delta1_oracle_measurement(
    *,
    inputs: dict[str, Any],
    checkpoint_path: str | Path,
    raw_output: str | Path,
    decision_output: str | Path,
) -> dict[str, Any]:
    """Run the consumed 20-domain evaluator and emit only derived MSM counts."""

    protocol = inputs["protocol"]
    session = inputs["session"]
    if Path(raw_output).exists() or Path(decision_output).exists():
        raise FileExistsError("oracle measurement outputs must both be fresh")
    if (
        session["phase"] != "development"
        or session["checkpoint_delta"] != 1
        or session["msm_oracle_status"] != ORACLE_UNRESOLVED
        or session["panel_domains"] != 20
        or str(Path(raw_output).expanduser().resolve()) != session["raw_output"]
    ):
        raise ValueError("oracle executor requires the exact unresolved development session")
    checkpoint, _ = _load_verified_checkpoint(
        Path(checkpoint_path).expanduser().resolve(), session["checkpoint_sha256"]
    )
    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("delta1 MSM oracle measurement requires CUDA")
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    if require_single_delta(data_cfg["delta_frames"]) != 1:
        raise ValueError("delta1 MSM oracle checkpoint is not delta=1")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    pins = inputs["payload_verification"]["pins"]
    domains = []
    try:
        for domain_index, pin in enumerate(pins):
            handle = _pinned_domain_handle(pin)
            try:
                cells = []
                for temperature_index, temperature in enumerate(temperatures):
                    for replica_index, replica in enumerate(replicas):
                        cell_index = temperature_index * len(replicas) + replica_index
                        measured = evaluate_conditional_cell(
                            handle=handle,
                            layout=handle.layout,
                            model=model,
                            device=device,
                            data_cfg=data_cfg,
                            delta=1,
                            temperature=temperature,
                            replica=replica,
                            replicas=replicas,
                            protocol=protocol,
                            seed_offset=(
                                domain_index * 10000
                                + temperature_index * 100
                                + replica_index
                            ),
                        )
                        cells.append(_oracle_count_cell_from_measured_conditional(
                            measured,
                            protocol,
                            seed=protocol["seed"] + domain_index * 1000 + cell_index,
                        ))
                domains.append({"domain": pin["domain"], "cells": cells})
            finally:
                handle.close()
    finally:
        pins.close()
    if len(domains) != 20:
        raise ValueError("delta1 MSM oracle executor did not measure exactly 20 domains")
    raw = {
        "schema": ORACLE_RAW_SCHEMA,
        "phase": "development",
        "checkpoint_sha256": session["checkpoint_sha256"],
        "checkpoint_step": session["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": session["full_training_contract_sha256"],
        "protocol_sha256": protocol["sha256"],
        "seed": protocol["seed"],
        "oracle_panel_name": session["panel_name"],
        "oracle_panel_sha256": session["panel_sha256"],
        "source_session_path": session["path"],
        "source_session_sha256": session["sha256"],
        "source_prerequisite_path": inputs["prerequisite"]["path"],
        "source_prerequisite_sha256": inputs["prerequisite"]["sha256"],
        "consumption_claim": inputs["consumption_claim"],
        "domains": domains,
        "formal_training_authorized": False,
    }
    _write_new_json(raw_output, raw)
    _, sha256, resolved = _read_stable_regular_bytes(
        raw_output, "delta1 MSM oracle evaluator raw output"
    )
    synthetic_session = {
        "msm_oracle_raw_path": str(resolved),
        "msm_oracle_raw_sha256": sha256,
        "checkpoint_sha256": raw["checkpoint_sha256"],
        "checkpoint_step": raw["checkpoint_step"],
        "full_training_contract_sha256": raw["full_training_contract_sha256"],
    }
    reopened, recomputed = _load_oracle_raw(protocol, synthetic_session)
    if recomputed["paired_msm_gain_ci95_lower"] <= 0:
        raise ValueError("delta1 MSM oracle recomputed CI does not substantiate PASS")
    decision = {
        "schema": ORACLE_SCHEMA,
        "status": ORACLE_PASS,
        "decision": "PASS",
        "evidence_type": "measured_delta1_msm_oracle",
        "checkpoint_sha256": reopened["checkpoint_sha256"],
        "checkpoint_step": reopened["checkpoint_step"],
        "checkpoint_delta": 1,
        "full_training_contract_sha256": reopened["full_training_contract_sha256"],
        "protocol_sha256": protocol["sha256"],
        "seed": protocol["seed"],
        "raw_draws_path": reopened["path"],
        "raw_draws_sha256": reopened["sha256"],
        "decision_rule": "domain_bootstrap_ci95_lower_gt_zero",
        "formal_training_authorized": False,
    }
    _write_new_json(decision_output, decision)
    _, decision_sha, decision_path = _read_stable_regular_bytes(
        decision_output, "delta1 MSM oracle decision"
    )
    return {
        "status": ORACLE_PASS,
        "decision_path": str(decision_path),
        "decision_sha256": decision_sha,
        "raw_draws_sha256": reopened["sha256"],
        "recomputed": recomputed,
        "formal_training_authorized": False,
    }


def _geometry_rows(
    positions: torch.Tensor,
    bond_mask: np.ndarray,
    *,
    collision_distance: float,
) -> dict[str, list[float | None]]:
    from deepjump.evaluation import geometry_frame_statistics

    names = (
        "bond_mean",
        "bond_p99",
        "bond_max",
        "angle_cos_mean",
        "angle_cos_p01",
        "angle_cos_p99",
        "collision_fraction",
    )
    result = {name: [] for name in names}
    for row in positions:
        if not torch.isfinite(row).all():
            for name in names:
                result[name].append(None)
            continue
        stats = geometry_frame_statistics(
            row[None].float().cpu().numpy(),
            bond_mask,
            collision_distance=collision_distance,
        )
        for name in names:
            result[name].append(float(stats[name][0]))
    return result


def evaluate_geometry_cell(
    *,
    handle: _DomainHandle,
    layout,
    model: DeepJumpLite,
    device: torch.device,
    data_cfg: dict[str, Any],
    delta: int,
    temperature: int,
    replica: int,
    protocol: dict[str, Any],
    state_archive: StateArchiveWriter | None = None,
    state_archive_prefix: str | None = None,
) -> dict[str, Any]:
    """Run one guarded H100 trajectory; H20 is only a prefix in adjudication."""

    from deepjump.evaluation import geometry_frame_statistics

    settings = protocol["geometry"]
    available = handle.replicas(temperature, [replica])
    if len(available) != 1:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    frames = available[0][2]
    n = min(layout.num_residues, int(data_cfg["crop_length"]))
    offset = max(0, (layout.num_residues - n) // 2)
    residue_slice = slice(offset, offset + n)
    bond_mask_np = np.asarray(
        layout.bond_mask[residue_slice.start : residue_slice.stop - 1], dtype=bool
    )
    reference_ids = np.linspace(
        0,
        frames - 1,
        min(settings["reference_frames_per_cell"], frames),
        dtype=int,
    )
    reference_positions = []
    for frame in reference_ids:
        coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, int(frame)))
        )
        positions, _ = apply_layout(coordinates, layout)
        positions = positions[residue_slice]
        reference_positions.append(positions - positions.mean(0, keepdim=True))
    reference_statistics = geometry_frame_statistics(
        torch.stack(reference_positions).numpy(),
        bond_mask_np,
        collision_distance=settings["collision_distance_angstrom"],
    )
    if frames < settings["starts_per_cell"]:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    starts = np.linspace(0, frames - 1, settings["starts_per_cell"], dtype=int)
    start_positions, start_vectors = [], []
    for frame in starts:
        coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, int(frame)))
        )
        positions, vectors = apply_model_layout(
            coordinates,
            layout,
            canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
        )
        positions = positions[residue_slice]
        start_positions.append(positions - positions.mean(0, keepdim=True))
        start_vectors.append(vectors[residue_slice])
    current_p = torch.stack(start_positions).to(device)
    current_v = torch.stack(start_vectors).to(device)
    initial_statistics = _geometry_rows(
        current_p,
        bond_mask_np,
        collision_distance=settings["collision_distance_angstrom"],
    )
    initial_state_hashes = (
        state_archive.add(f"{state_archive_prefix}.initial", current_p, current_v)
        if state_archive is not None and state_archive_prefix is not None
        else _state_hashes(current_p, current_v)
    )
    steps = []
    for step in range(1, max(settings["horizons"]) + 1):
        batch = _model_batch(
            layout, residue_slice, current_p, current_v, delta, device
        )
        raw_p, raw_v = model.sample(batch, mode="mean", steps=1)
        guarded_p, guarded_v, accepted = reject_to_source(
            raw_p,
            raw_v,
            current_p,
            current_v,
            batch["bond_mask"],
            lo=settings["guard"]["bond_mean_gt_angstrom"],
            hi=settings["guard"]["bond_mean_lt_angstrom"],
            max_bond=settings["guard"]["bond_max_lt_angstrom"],
        )
        source_guard = _guard_observations(current_p, current_v, batch["bond_mask"])
        raw_guard = _guard_observations(raw_p, raw_v, batch["bond_mask"])
        source_state_hashes = (
            state_archive.add(f"{state_archive_prefix}.source{step}", current_p, current_v)
            if state_archive is not None and state_archive_prefix is not None
            else _state_hashes(current_p, current_v)
        )
        raw_state_hashes = (
            state_archive.add(f"{state_archive_prefix}.raw{step}", raw_p, raw_v)
            if state_archive is not None and state_archive_prefix is not None
            else _state_hashes(raw_p, raw_v)
        )
        selected_state_hashes = (
            state_archive.add(
                f"{state_archive_prefix}.selected{step}", guarded_p, guarded_v
            )
            if state_archive is not None and state_archive_prefix is not None
            else _state_hashes(guarded_p, guarded_v)
        )
        steps.append({
            "step": step,
            "accepted": [bool(value) for value in accepted.tolist()],
            "selected_position_exact": [
                bool(torch.equal(guarded_p[i], raw_p[i] if accepted[i] else current_p[i]))
                for i in range(len(accepted))
            ],
            "selected_vector_exact": [
                bool(torch.equal(guarded_v[i], raw_v[i] if accepted[i] else current_v[i]))
                for i in range(len(accepted))
            ],
            "source_state_sha256": source_state_hashes,
            "raw_state_sha256": raw_state_hashes,
            "selected_state_sha256": selected_state_hashes,
            "source_guard_observations": source_guard,
            "raw_guard_observations": raw_guard,
            "raw_statistics": _geometry_rows(
                raw_p,
                bond_mask_np,
                collision_distance=settings["collision_distance_angstrom"],
            ),
            "guarded_statistics": _geometry_rows(
                guarded_p,
                bond_mask_np,
                collision_distance=settings["collision_distance_angstrom"],
            ),
        })
        current_p, current_v = guarded_p, guarded_v
    return {
        "temperature": int(temperature),
        "replica": int(replica),
        "start_frames": [int(value) for value in starts],
        "state_archive_prefix": state_archive_prefix,
        "reference_statistics": {
            key: _json_vectors(value) for key, value in reference_statistics.items()
        },
        "initial_statistics": initial_statistics,
        "initial_state_sha256": initial_state_hashes,
        "steps_h100": steps,
    }


@torch.no_grad()
def evaluate_scientific_bundle(inputs: dict[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
    protocol = inputs["protocol"]
    session = inputs["session"]
    checkpoint, _ = _load_verified_checkpoint(
        Path(checkpoint_path).expanduser().resolve(), session["checkpoint_sha256"]
    )
    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("contracted scientific evaluation requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    delta = require_single_delta(data_cfg["delta_frames"])
    pins = inputs["payload_verification"]["pins"]
    conditional_domains, geometry_domains = [], []
    state_archive = StateArchiveWriter()
    try:
        for domain_index, pin in enumerate(pins):
            handle = _pinned_domain_handle(pin)
            try:
                conditional_cells, geometry_cells = [], []
                for temperature_index, temperature in enumerate(temperatures):
                    for replica_index, replica in enumerate(replicas):
                        seed_offset = (
                            domain_index * 10000
                            + temperature_index * 100
                            + replica_index
                        )
                        cell_index = temperature_index * len(replicas) + replica_index
                        conditional_cells.append(evaluate_conditional_cell(
                            handle=handle,
                            layout=handle.layout,
                            model=model,
                            device=device,
                            data_cfg=data_cfg,
                            delta=delta,
                            temperature=temperature,
                            replica=replica,
                            replicas=replicas,
                            protocol=protocol,
                            seed_offset=seed_offset,
                            state_archive=state_archive,
                            state_archive_prefix=f"conditional.d{domain_index}.c{cell_index}",
                        ))
                        geometry_cells.append(evaluate_geometry_cell(
                            handle=handle,
                            layout=handle.layout,
                            model=model,
                            device=device,
                            data_cfg=data_cfg,
                            delta=delta,
                            temperature=temperature,
                            replica=replica,
                            protocol=protocol,
                            state_archive=state_archive,
                            state_archive_prefix=f"geometry.d{domain_index}.c{cell_index}",
                        ))
                conditional_domains.append({"domain": pin["domain"], "cells": conditional_cells})
                geometry_domains.append({"domain": pin["domain"], "cells": geometry_cells})
            finally:
                handle.close()
    finally:
        pins.close()
    payload_verification = {
        key: value
        for key, value in inputs["payload_verification"].items()
        if key not in {"paths", "pins"}
    }
    state_archive_identity = state_archive.publish(session["state_archive_output"])
    _write_new_json(
        session["runtime_probe_output"],
        {
            "status": "PASS_CONTRACTED_SCIENTIFIC_RUNTIME",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "cuda_runtime_version": str(torch.version.cuda),
            "cuda_device_count": torch.cuda.device_count(),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint_sha256": session["checkpoint_sha256"],
            "protocol_sha256": protocol["sha256"],
            "session_sha256": session["sha256"],
            "formal_training_authorized": False,
        },
    )
    return {
        "schema": RAW_EVIDENCE_SCHEMA,
        "status": EVALUATOR_STATUS,
        "numerical_kernel_implemented": True,
        "protocol_sha256": protocol["sha256"],
        "session_sha256": session["sha256"],
        "prerequisite_sha256": inputs["prerequisite"]["sha256"],
        "identity": inputs["identity"],
        "payload_verification": payload_verification,
        "data_prerequisite_status": inputs["data_prerequisite"]["status"],
        "oracle_artifact": inputs["oracle_artifact"],
        "runtime_feasibility": inputs["runtime_feasibility"],
        "global_obs_claim": inputs["global_obs_claim"],
        "state_archive": state_archive_identity,
        "consumption_claim": inputs["consumption_claim"],
        "conditional_raw": {"domains": conditional_domains},
        "geometry_raw": {"domains": geometry_domains},
        "scientific_adjudication_performed": False,
        "formal_training_authorized": False,
    }


def implementation_report() -> dict[str, Any]:
    return {
        "status": IMPLEMENTATION_STATUS,
        "numerical_kernel_implemented": NUMERICAL_KERNEL_IMPLEMENTED,
        "required_semantics": {
            "conditional": "ODE1 >=16 draws with per-draw reject_to_source",
            "geometry": "mean guarded H100 with H20 derived from its exact prefix",
            "adjudication": "independent recomputation from raw evidence",
            "data": "exact contract, complete manifest, live SHA256, readable HDF5",
        },
        "formal_training_authorized": False,
    }


def main() -> None:
    if sys.argv[1:] == ["--implementation-status"]:
        print(json.dumps(implementation_report(), sort_keys=True))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--sha256-file":
        print(_read_stable_regular_bytes(sys.argv[2], "runner artifact")[1])
        return
    if sys.argv[1:2] == ["--measure-delta1-oracle"]:
        oracle_parser = argparse.ArgumentParser()
        oracle_parser.add_argument("--measure-delta1-oracle", action="store_true")
        oracle_parser.add_argument("--protocol", required=True)
        oracle_parser.add_argument("--expected-protocol-sha256", required=True)
        oracle_parser.add_argument("--session", required=True)
        oracle_parser.add_argument("--expected-session-sha256", required=True)
        oracle_parser.add_argument("--prerequisite-decision", required=True)
        oracle_parser.add_argument(
            "--expected-prerequisite-decision-sha256", required=True
        )
        oracle_parser.add_argument("--checkpoint", required=True)
        oracle_parser.add_argument("--contract", required=True)
        oracle_parser.add_argument("--panel-file", required=True)
        oracle_parser.add_argument("--data-root", required=True)
        oracle_parser.add_argument("--expected-repo-commit", required=True)
        oracle_parser.add_argument("--raw-output", required=True)
        oracle_parser.add_argument("--decision-output", required=True)
        oracle_args = oracle_parser.parse_args()
        oracle_protocol = load_protocol(
            oracle_args.protocol, oracle_args.expected_protocol_sha256
        )
        oracle_session = load_session(
            oracle_args.session, oracle_args.expected_session_sha256
        )
        oracle_prerequisite = load_scientific_prerequisite(
            oracle_args.prerequisite_decision,
            oracle_args.expected_prerequisite_decision_sha256,
            session=oracle_session,
        )
        validate_bundle_bindings(
            oracle_protocol,
            oracle_session,
            oracle_prerequisite,
            expected_repo_commit=oracle_args.expected_repo_commit,
        )
        if (
            oracle_session["phase"] != "development"
            or oracle_session["checkpoint_delta"] != 1
            or oracle_session["msm_oracle_status"] != ORACLE_UNRESOLVED
            or str(Path(oracle_args.raw_output).expanduser().resolve())
            != oracle_session["raw_output"]
            or Path(oracle_args.raw_output).exists()
            or Path(oracle_args.decision_output).exists()
        ):
            raise ValueError("oracle CLI requires fresh exact development outputs")
        oracle_runtime = load_runtime_feasibility_artifact(
            oracle_protocol, oracle_session
        )
        verify_runtime_probe_bindings(
            oracle_runtime, oracle_args.contract, oracle_session
        )
        oracle_inputs = establish_claimed_inputs(
            protocol=oracle_protocol,
            session=oracle_session,
            prerequisite=oracle_prerequisite,
            prerequisite_sha256=oracle_args.expected_prerequisite_decision_sha256,
            checkpoint=oracle_args.checkpoint,
            contract=oracle_args.contract,
            panel_file=oracle_args.panel_file,
            data_root=oracle_args.data_root,
            oracle_artifact=None,
            runtime_feasibility=oracle_runtime,
        )
        print(json.dumps(execute_delta1_oracle_measurement(
            inputs=oracle_inputs,
            checkpoint_path=oracle_args.checkpoint,
            raw_output=oracle_args.raw_output,
            decision_output=oracle_args.decision_output,
        ), sort_keys=True))
        return
    if sys.argv[1:2] == ["--measure-runtime"]:
        runtime_parser = argparse.ArgumentParser()
        runtime_parser.add_argument("--measure-runtime", action="store_true")
        runtime_parser.add_argument("--protocol", required=True)
        runtime_parser.add_argument("--expected-protocol-sha256", required=True)
        runtime_parser.add_argument("--session", required=True)
        runtime_parser.add_argument("--expected-session-sha256", required=True)
        runtime_parser.add_argument("--checkpoint", required=True)
        runtime_parser.add_argument("--contract", required=True)
        runtime_parser.add_argument("--data-root", required=True)
        runtime_parser.add_argument("--probe-plan", required=True)
        runtime_parser.add_argument("--expected-probe-plan-sha256", required=True)
        runtime_parser.add_argument("--raw-output", required=True)
        runtime_parser.add_argument("--decision-output", required=True)
        runtime_args = runtime_parser.parse_args()
        runtime_protocol = load_protocol(
            runtime_args.protocol, runtime_args.expected_protocol_sha256
        )
        runtime_session = load_session(
            runtime_args.session, runtime_args.expected_session_sha256
        )
        probe_plan, _, _ = _read_regular_json_value(
            runtime_args.probe_plan,
            runtime_args.expected_probe_plan_sha256,
            "runtime probe plan",
        )
        if not isinstance(probe_plan, list):
            raise ValueError("runtime probe plan must be a JSON list")
        print(json.dumps(execute_runtime_feasibility_measurement(
            protocol=runtime_protocol,
            session=runtime_session,
            checkpoint=runtime_args.checkpoint,
            contract=runtime_args.contract,
            data_root=runtime_args.data_root,
            probe_plan=probe_plan,
            raw_output=runtime_args.raw_output,
            decision_output=runtime_args.decision_output,
        ), sort_keys=True))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--expected-session-sha256", required=True)
    parser.add_argument("--prerequisite-decision", required=True)
    parser.add_argument("--expected-prerequisite-decision-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol, args.expected_protocol_sha256)
    session = load_session(args.session, args.expected_session_sha256)
    prerequisite = load_scientific_prerequisite(
        args.prerequisite_decision,
        args.expected_prerequisite_decision_sha256,
        session=session,
    )
    validate_bundle_bindings(
        protocol,
        session,
        prerequisite,
        expected_repo_commit=args.expected_repo_commit,
    )
    preflight_output_paths(session)
    if (
        session["checkpoint_delta"] == 1
        and session["msm_oracle_status"] == ORACLE_UNRESOLVED
    ):
        _write_new_json(
            session["decision_output"],
            {
                "schema": "deepjump.contracted_scientific_decision.v1",
                "status": INCONCLUSIVE_ORACLE,
                "reason": "delta=1 requires a verified bound MSM oracle artifact",
                "session_sha256": session["sha256"],
                "authorization_consumed": False,
                "panel_opened": False,
                "model_opened": False,
                "formal_training_authorized": False,
            },
        )
        print(json.dumps({
            "status": INCONCLUSIVE_ORACLE,
            "decision_output": session["decision_output"],
            "authorization_consumed": False,
            "formal_training_authorized": False,
        }, sort_keys=True))
        return
    oracle_artifact = load_delta1_oracle_artifact(protocol, session)
    runtime_feasibility = load_runtime_feasibility_artifact(protocol, session)
    verify_runtime_probe_bindings(runtime_feasibility, args.contract, session)
    data_root = _require_absolute(args.data_root, "command-line data_root")
    if data_root != session["data_root"]:
        raise ValueError("command-line data root differs from the scientific session")
    inputs = establish_claimed_inputs(
        protocol=protocol,
        session=session,
        prerequisite=prerequisite,
        prerequisite_sha256=args.expected_prerequisite_decision_sha256,
        checkpoint=args.checkpoint,
        contract=args.contract,
        panel_file=args.panel_file,
        data_root=data_root,
        oracle_artifact=oracle_artifact,
        runtime_feasibility=runtime_feasibility,
    )
    result = evaluate_scientific_bundle(inputs, args.checkpoint)
    _write_new_json(session["raw_output"], result)
    print(json.dumps({
        "status": result["status"],
        "raw_output": session["raw_output"],
        "domains": len(result["conditional_raw"]["domains"]),
        "formal_training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
