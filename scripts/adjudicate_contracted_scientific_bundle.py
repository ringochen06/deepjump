#!/usr/bin/env python
"""Independent, fail-closed adjudicator for contracted scientific evidence.

This reader reopens every protected identity and recomputes all metrics from
draw/step-level evidence. It never trusts evaluator summaries and never grants
formal-training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from deepjump.evaluation import (
    aggregate_geometry_panel,
    assign_clusters,
    bootstrap_domain_mean_upper,
    calibrate_geometry_worst_envelope,
    fit_kmeans,
    geometry_worst_excess,
    reference_transition_deltas,
    transition_matrix,
    weighted_row_jsd_bits,
    require_mdcath_full_grid,
    require_single_delta,
)
from deepjump.evaluation_consumption import verify_reserved_evaluation_claim
from deepjump.evaluation_contract import (
    _load_verified_checkpoint,
    verify_frozen_evaluation_identity,
)
from scripts.contracted_guarded_endpoint_panel_eval import rehash_contracted_panel_payloads
from scripts.transition_robustness_eval import energy_distance, energy_score
from scripts.contracted_scientific_bundle_eval import (
    DATA_PREREQUISITE_FAILURE,
    EVALUATOR_STATUS,
    NUMERICAL_KERNEL_IMPLEMENTED,
    ORACLE_PASS,
    ORACLE_UNRESOLVED,
    RAW_EVIDENCE_SCHEMA,
    STATE_ARCHIVE_SCHEMA,
    GLOBAL_CLAIM_SCHEMA,
    GLOBAL_CLAIM_RECEIPT_SCHEMA,
    _read_regular_json,
    _read_stable_regular_bytes,
    load_protocol,
    load_scientific_prerequisite,
    load_session,
    validate_bundle_bindings,
    verify_scientific_data_prerequisite,
    verify_runtime_probe_bindings,
)


DECISION_SCHEMA = "deepjump.contracted_scientific_decision.v1"
INCONCLUSIVE_ORACLE = "INCONCLUSIVE_DELTA1_MSM_ORACLE"
READY_FOR_RECOMPUTATION = "READY_FOR_INDEPENDENT_NUMERICAL_RECOMPUTATION"
IMPLEMENTATION_STATUS = "INDEPENDENT_ADJUDICATION_KERNEL_READY"
DATA_PASS = "PASS_CONTRACTED_DATA_PREREQUISITE"

RAW_KEYS = frozenset({
    "schema",
    "status",
    "numerical_kernel_implemented",
    "protocol_sha256",
    "session_sha256",
    "prerequisite_sha256",
    "identity",
    "payload_verification",
    "data_prerequisite_status",
    "oracle_artifact",
    "runtime_feasibility",
    "global_obs_claim",
    "state_archive",
    "consumption_claim",
    "conditional_raw",
    "geometry_raw",
    "scientific_adjudication_performed",
    "formal_training_authorized",
})


def validate_raw_evidence(payload: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    """Require complete raw evidence; never translate missing data into model FAIL."""

    if set(payload) != RAW_KEYS or payload.get("schema") != RAW_EVIDENCE_SCHEMA:
        raise ValueError("scientific raw evidence exact schema mismatch")
    if payload.get("status") != EVALUATOR_STATUS:
        raise ValueError("scientific evaluator did not complete raw evidence")
    if payload.get("numerical_kernel_implemented") is not True:
        raise ValueError("scientific raw evidence was not produced by the complete kernel")
    if payload.get("protocol_sha256") != session["protocol_sha256"]:
        raise ValueError("scientific raw protocol binding mismatch")
    if payload.get("session_sha256") != session["sha256"]:
        raise ValueError("scientific raw session binding mismatch")
    if payload.get("data_prerequisite_status") == DATA_PREREQUISITE_FAILURE:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    if payload.get("data_prerequisite_status") != DATA_PASS:
        raise ValueError("scientific data prerequisite has no exact PASS evidence")
    identity = payload.get("identity")
    if not isinstance(identity, dict) or any(
        identity.get(key) != session[session_key]
        for key, session_key in (
            ("checkpoint_sha256", "checkpoint_sha256"),
            ("checkpoint_step", "checkpoint_step"),
            ("full_training_contract_sha256", "full_training_contract_sha256"),
            ("panel_name", "panel_name"),
            ("panel_sha256", "panel_sha256"),
            ("panel_domains", "panel_domains"),
        )
    ):
        raise ValueError("scientific raw identity differs from the session")
    payload_verification = payload.get("payload_verification")
    if (
        not isinstance(payload_verification, dict)
        or payload_verification.get("status")
        != "PASS_CONTRACTED_PANEL_LIVE_PAYLOAD_REHASH"
        or payload_verification.get("panel_domains") != session["panel_domains"]
    ):
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    if set(payload.get("conditional_raw", {})) != {"domains"}:
        raise ValueError("conditional raw evidence is absent")
    if set(payload.get("geometry_raw", {})) != {"domains"}:
        raise ValueError("geometry raw evidence is absent")
    conditional_domains = payload["conditional_raw"]["domains"]
    geometry_domains = payload["geometry_raw"]["domains"]
    if (
        not isinstance(conditional_domains, list)
        or not isinstance(geometry_domains, list)
        or len(conditional_domains) != session["panel_domains"]
        or len(geometry_domains) != session["panel_domains"]
    ):
        raise ValueError("scientific raw domain count mismatch")
    conditional_ids = [row.get("domain") for row in conditional_domains if isinstance(row, dict)]
    geometry_ids = [row.get("domain") for row in geometry_domains if isinstance(row, dict)]
    if (
        len(conditional_ids) != session["panel_domains"]
        or conditional_ids != geometry_ids
        or len(set(conditional_ids)) != session["panel_domains"]
    ):
        raise ValueError("scientific raw domain identity mismatch")
    payload_ids = [
        row.get("domain")
        for row in payload_verification.get("payloads", [])
        if isinstance(row, dict)
    ]
    if conditional_ids != payload_ids:
        raise ValueError("scientific raw domains differ from live-rehashed payloads")
    if any(len(row.get("cells", [])) != 25 for row in conditional_domains + geometry_domains):
        raise ValueError("scientific raw evidence requires the exact 5x5 cell grid")
    if payload.get("scientific_adjudication_performed") is not False:
        raise ValueError("evaluator must not adjudicate its own raw evidence")
    if payload.get("formal_training_authorized") is not False:
        raise ValueError("raw evidence cannot authorize formal training")
    state_archive = payload.get("state_archive")
    if (
        not isinstance(state_archive, dict)
        or set(state_archive) != {"schema", "path", "sha256", "arrays"}
        or state_archive.get("schema") != STATE_ARCHIVE_SCHEMA
        or state_archive.get("path") != session["state_archive_output"]
        or type(state_archive.get("arrays")) is not int
        or state_archive["arrays"] <= 0
    ):
        raise ValueError("scientific state archive identity is absent")
    global_claim = payload.get("global_obs_claim")
    if session.get("phase") == "untouched":
        if not isinstance(global_claim, dict) or global_claim.get("completed") is not True:
            raise ValueError("untouched scientific raw evidence lacks the global OBS claim")
    elif global_claim != {"required": False, "completed": False}:
        raise ValueError("non-untouched raw evidence has an unexpected global OBS claim")
    return payload


def delta1_oracle_disposition(session: dict[str, Any]) -> dict[str, Any]:
    """Return the preregistered non-PASS/non-FAIL oracle disposition."""

    if session["checkpoint_delta"] == 1 and session["msm_oracle_status"] == ORACLE_UNRESOLVED:
        return {
            "status": INCONCLUSIVE_ORACLE,
            "reason": "delta=1 requires a resolved MSM oracle prerequisite",
            "formal_training_authorized": False,
        }
    if session["checkpoint_delta"] == 1 and session["msm_oracle_status"] != ORACLE_PASS:
        raise ValueError("delta=1 MSM oracle status is invalid")
    return {
        "status": READY_FOR_RECOMPUTATION,
        "formal_training_authorized": False,
    }


def _numpy_state_sha256(positions: np.ndarray, vectors: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, array in ((b"P", positions), (b"V", vectors)):
        value = np.ascontiguousarray(array)
        digest.update(label)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_state_archive(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    identity = raw["state_archive"]
    archive_raw, actual, resolved = _read_stable_regular_bytes(
        identity["path"], "scientific state archive", identity["sha256"]
    )
    if actual != identity["sha256"] or str(resolved) != identity["path"]:
        raise ValueError("scientific state archive identity mismatch")
    try:
        with np.load(io.BytesIO(archive_raw), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError("scientific state archive is not a safe NPZ") from exc
    if len(arrays) != identity["arrays"] or len(arrays) != len(set(arrays)):
        raise ValueError("scientific state archive array count mismatch")
    expected: set[str] = set()
    for domain in raw.get("conditional_raw", {}).get("domains", []):
        for cell in domain.get("cells", []):
            prefix = cell.get("state_archive_prefix")
            if isinstance(prefix, str):
                expected.update({prefix + ".source.P", prefix + ".source.V"})
                for start_index in range(len(cell.get("raw_state_sha256", []))):
                    for kind in ("raw", "selected"):
                        expected.update(
                            {
                                f"{prefix}.{kind}{start_index}.P",
                                f"{prefix}.{kind}{start_index}.V",
                            }
                        )
    for domain in raw.get("geometry_raw", {}).get("domains", []):
        for cell in domain.get("cells", []):
            prefix = cell.get("state_archive_prefix")
            if isinstance(prefix, str):
                expected.update({prefix + ".initial.P", prefix + ".initial.V"})
                for step_index in range(1, len(cell.get("steps_h100", [])) + 1):
                    for kind in ("source", "raw", "selected"):
                        expected.update(
                            {
                                f"{prefix}.{kind}{step_index}.P",
                                f"{prefix}.{kind}{step_index}.V",
                            }
                        )
    if set(arrays) != expected:
        raise ValueError("scientific state archive is not the exact raw-evidence union")
    return arrays


def _archive_pair(
    archive: dict[str, np.ndarray], name: str, expected_hashes: object, label: str
) -> tuple[np.ndarray, np.ndarray]:
    try:
        positions = archive[name + ".P"]
        vectors = archive[name + ".V"]
    except KeyError as exc:
        raise ValueError(f"{label} state archive arrays are missing") from exc
    if positions.ndim < 2 or vectors.ndim < 2 or len(positions) != len(vectors):
        raise ValueError(f"{label} state archive P/V shapes mismatch")
    hashes = [_numpy_state_sha256(positions[i], vectors[i]) for i in range(len(positions))]
    if hashes != expected_hashes:
        raise ValueError(f"{label} state hashes differ from canonical archive bytes")
    return positions, vectors


def _verify_conditional_archive(
    cell: dict[str, Any], archive: dict[str, np.ndarray]
) -> None:
    prefix = cell.get("state_archive_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("conditional state archive prefix is absent")
    source_p, source_v = _archive_pair(
        archive, prefix + ".source", cell.get("source_state_sha256"), "conditional source"
    )
    accepted = cell.get("accepted")
    for start_index in range(len(source_p)):
        raw_p, raw_v = _archive_pair(
            archive,
            f"{prefix}.raw{start_index}",
            cell.get("raw_state_sha256", [])[start_index],
            "conditional raw",
        )
        selected_p, selected_v = _archive_pair(
            archive,
            f"{prefix}.selected{start_index}",
            cell.get("selected_state_sha256", [])[start_index],
            "conditional selected",
        )
        if not isinstance(accepted, list) or len(accepted[start_index]) != len(raw_p):
            raise ValueError("conditional archive accepted shape mismatch")
        for draw_index, is_accepted in enumerate(accepted[start_index]):
            expected_p = raw_p[draw_index] if is_accepted else source_p[start_index]
            expected_v = raw_v[draw_index] if is_accepted else source_v[start_index]
            if not np.array_equal(selected_p[draw_index], expected_p, equal_nan=True) or not np.array_equal(
                selected_v[draw_index], expected_v, equal_nan=True
            ):
                raise ValueError("conditional archive does not contain exact raw-or-source fallback")


def _verify_geometry_archive(cell: dict[str, Any], archive: dict[str, np.ndarray]) -> None:
    prefix = cell.get("state_archive_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("geometry state archive prefix is absent")
    previous_p, previous_v = _archive_pair(
        archive, prefix + ".initial", cell.get("initial_state_sha256"), "geometry initial"
    )
    for step_index, step in enumerate(cell.get("steps_h100", []), 1):
        source_p, source_v = _archive_pair(
            archive,
            f"{prefix}.source{step_index}",
            step.get("source_state_sha256"),
            "geometry source",
        )
        raw_p, raw_v = _archive_pair(
            archive,
            f"{prefix}.raw{step_index}",
            step.get("raw_state_sha256"),
            "geometry raw",
        )
        selected_p, selected_v = _archive_pair(
            archive,
            f"{prefix}.selected{step_index}",
            step.get("selected_state_sha256"),
            "geometry selected",
        )
        if not np.array_equal(source_p, previous_p, equal_nan=True) or not np.array_equal(
            source_v, previous_v, equal_nan=True
        ):
            raise ValueError("geometry archive source is not previous selected state")
        accepted = step.get("accepted")
        if not isinstance(accepted, list) or len(accepted) != len(source_p):
            raise ValueError("geometry archive accepted shape mismatch")
        for index, is_accepted in enumerate(accepted):
            expected_p = raw_p[index] if is_accepted else source_p[index]
            expected_v = raw_v[index] if is_accepted else source_v[index]
            if not np.array_equal(selected_p[index], expected_p, equal_nan=True) or not np.array_equal(
                selected_v[index], expected_v, equal_nan=True
            ):
                raise ValueError("geometry archive does not contain exact raw-or-source fallback")
        previous_p, previous_v = selected_p, selected_v


def _verify_global_obs_claim(
    protocol: dict[str, Any], session: dict[str, Any], claim: dict[str, Any]
) -> None:
    if session["phase"] != "untouched":
        if claim != {"required": False, "completed": False}:
            raise ValueError("unexpected global OBS claim")
        return
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
    receipt_raw, receipt_sha, receipt_path = _read_stable_regular_bytes(
        session["global_claim_receipt_path"], "global OBS claim receipt"
    )
    readback_raw, readback_sha, readback_path = _read_stable_regular_bytes(
        session["global_claim_readback_path"], "global OBS claim readback"
    )
    try:
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("global OBS claim receipt is invalid") from exc
    uri = (
        f"{session['obs_prefix'].split('/deepjump-scientific/', 1)[0]}"
        "/deepjump-scientific/one-shot-claims/"
        f"{session['authorization_id']}/{session['panel_sha256']}.json"
    )
    descriptor, descriptor_sha, descriptor_path = _read_regular_json(
        session["global_claim_descriptor_path"],
        claim.get("descriptor_sha256"),
        "global OBS claim descriptor",
    )
    expected_descriptor = {
        "schema": "deepjump.contracted_scientific_global_claim_descriptor.v1",
        "uri": uri,
        "payload": payload,
        "payload_sha256": payload_sha,
        "payload_size_bytes": len(payload_raw),
        "helper_sha256": helper_sha,
    }
    if descriptor != expected_descriptor or str(descriptor_path) != session[
        "global_claim_descriptor_path"
    ]:
        raise ValueError("global OBS claim descriptor mismatch")
    expected_receipt = {
        "schema": GLOBAL_CLAIM_RECEIPT_SCHEMA,
        "created": True,
        "condition": "If-None-Match:*",
        "uri": uri,
        "payload_sha256": payload_sha,
        "payload_size_bytes": len(payload_raw),
        "helper_sha256": helper_sha,
    }
    if receipt != expected_receipt or readback_raw != payload_raw or readback_sha != payload_sha:
        raise ValueError("global OBS conditional-create evidence mismatch")
    expected_claim = {
        "required": True,
        "completed": True,
        "helper_sha256": helper_sha,
        "descriptor_path": session["global_claim_descriptor_path"],
        "descriptor_sha256": descriptor_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "readback_path": str(readback_path),
        "readback_sha256": readback_sha,
        "payload_sha256": payload_sha,
        "payload_size_bytes": len(payload_raw),
        "uri": uri,
    }
    if claim != expected_claim:
        raise ValueError("global OBS claim does not bind the exact session")


def _write_new_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def implementation_report() -> dict[str, Any]:
    return {
        "status": IMPLEMENTATION_STATUS,
        "independent_numerical_recomputation_implemented": True,
        "delta1_oracle_inconclusive_rule_implemented": True,
        "formal_training_authorized": False,
    }


def independently_verify_bound_inputs(
    *,
    raw: dict[str, Any],
    protocol: dict[str, Any],
    session: dict[str, Any],
    checkpoint: str | Path,
    contract: str | Path,
    panel_file: str | Path,
    data_root: str | Path,
) -> dict[str, Any]:
    """Re-open every protected input rather than trusting evaluator summaries."""

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
    if raw["identity"] != identity:
        raise ValueError("adjudicator identity recomputation differs from raw evidence")
    data_prerequisite = verify_scientific_data_prerequisite(
        contract, session["full_training_contract_sha256"]
    )
    if raw["data_prerequisite_status"] != data_prerequisite["status"]:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    canonical_root = str(Path(data_root).expanduser().resolve())
    if canonical_root != session["data_root"]:
        raise ValueError("adjudicator data root differs from the session")
    payload = rehash_contracted_panel_payloads(
        contract, panel_file, canonical_root, keep_open=False
    )
    payload = {key: value for key, value in payload.items() if key not in {"paths", "pins"}}
    if raw["payload_verification"] != payload:
        raise ValueError(DATA_PREREQUISITE_FAILURE)
    checkpoint_payload, checkpoint_sha = _load_verified_checkpoint(
        Path(checkpoint).expanduser().resolve(), session["checkpoint_sha256"]
    )
    if checkpoint_sha != identity["checkpoint_sha256"]:
        raise ValueError("checkpoint changed during adjudicator verification")
    data_cfg = checkpoint_payload.get("cfg", {}).get("data", {})
    if require_single_delta(data_cfg.get("delta_frames")) != session["checkpoint_delta"]:
        raise ValueError("adjudicator checkpoint delta differs from the session")
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg.get("temperatures"), data_cfg.get("replicas")
    )
    if temperatures != protocol["panel"]["temperatures_kelvin"]:
        raise ValueError("adjudicator temperature grid differs from the protocol")
    if replicas != protocol["panel"]["replicas"]:
        raise ValueError("adjudicator replica grid differs from the protocol")
    if str(Path(data_cfg.get("root", "")).expanduser().resolve()) != canonical_root:
        raise ValueError("adjudicator checkpoint data root differs from the session")
    from scripts.contracted_scientific_bundle_eval import (
        load_delta1_oracle_artifact,
        load_runtime_feasibility_artifact,
    )

    oracle_artifact = load_delta1_oracle_artifact(protocol, session)
    runtime_feasibility = load_runtime_feasibility_artifact(protocol, session)
    verify_runtime_probe_bindings(runtime_feasibility, contract, session)
    if raw["oracle_artifact"] != oracle_artifact:
        raise ValueError("adjudicator oracle artifact differs from raw evidence")
    if raw["runtime_feasibility"] != runtime_feasibility:
        raise ValueError("adjudicator runtime feasibility differs from raw evidence")
    return {
        "identity": identity,
        "data_prerequisite": data_prerequisite,
        "payload_verification": payload,
        "oracle_artifact": oracle_artifact,
        "runtime_feasibility": runtime_feasibility,
        "formal_training_authorized": False,
    }


def _as_finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} shape or finiteness mismatch")
    return array


def _guard_accepts(observation: dict[str, Any], guard: dict[str, Any]) -> bool:
    if not isinstance(observation, dict):
        raise ValueError("guard observation must be an object")
    finite = observation.get("position_finite") is True and observation.get("vector_finite") is True
    mean = observation.get("bond_mean")
    maximum = observation.get("bond_max")
    if not finite:
        return False
    if not isinstance(mean, (int, float)) or not isinstance(maximum, (int, float)):
        raise ValueError("finite guard observation lacks bond metrics")
    return bool(
        np.isfinite(mean)
        and np.isfinite(maximum)
        and mean > guard["bond_mean_gt_angstrom"]
        and mean < guard["bond_mean_lt_angstrom"]
        and maximum < guard["bond_max_lt_angstrom"]
    )


def _require_state_hashes(value: object, count: int, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} state-hash count mismatch")
    for digest in value:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} contains an invalid state SHA256")
    return value


def recompute_conditional_cell(
    cell: dict[str, Any], protocol: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    settings = protocol["conditional_transition"]
    starts = settings["starts_per_cell"]
    draws = settings["draws_per_start"]
    components = settings["tica"]["components"]
    if cell.get("draws") != draws or cell.get("tica_components") != components:
        raise ValueError("conditional draw/component metadata mismatch")
    if not isinstance(cell.get("start_frames"), list) or len(cell["start_frames"]) != starts:
        raise ValueError("conditional start-frame evidence mismatch")
    real = np.asarray(cell.get("real_tic"), dtype=np.float64)
    if real.ndim != 2 or real.shape[1] != components or len(real) < 2:
        raise ValueError("real TIC evidence shape mismatch")
    if not np.isfinite(real).all():
        raise ValueError("real TIC evidence is non-finite")
    source = _as_finite_array(cell.get("source_tic"), (starts, components), "source TIC")
    target = _as_finite_array(cell.get("target_tic"), (starts, components), "target TIC")
    guarded = _as_finite_array(
        cell.get("guarded_predicted_tic"),
        (starts, draws, components),
        "guarded predicted TIC",
    )
    raw_value = cell.get("raw_predicted_tic")
    raw = np.asarray(
        [[[np.nan if value is None else value for value in vector] for vector in row]
         for row in raw_value],
        dtype=np.float64,
    )
    if raw.shape != (starts, draws, components):
        raise ValueError("raw predicted TIC shape mismatch")
    accepted = cell.get("accepted")
    source_state_hashes = _require_state_hashes(
        cell.get("source_state_sha256"), starts, "conditional source"
    )
    raw_state_hashes = cell.get("raw_state_sha256")
    selected_state_hashes = cell.get("selected_state_sha256")
    observations = cell.get("raw_guard_observations")
    source_observations = cell.get("source_guard_observations")
    if (
        not isinstance(accepted, list)
        or not isinstance(observations, list)
        or not isinstance(source_observations, list)
        or len(accepted) != starts
        or len(observations) != starts
        or len(source_observations) != starts
        or not isinstance(raw_state_hashes, list)
        or not isinstance(selected_state_hashes, list)
        or len(raw_state_hashes) != starts
        or len(selected_state_hashes) != starts
    ):
        raise ValueError("conditional guard evidence shape mismatch")
    for start_index in range(starts):
        if not _guard_accepts(source_observations[start_index], settings["guard"]):
            raise ValueError("conditional source is not a valid safeguard source")
        if len(accepted[start_index]) != draws or len(observations[start_index]) != draws:
            raise ValueError("conditional draw count mismatch")
        raw_hashes = _require_state_hashes(
            raw_state_hashes[start_index], draws, "conditional raw"
        )
        selected_hashes = _require_state_hashes(
            selected_state_hashes[start_index], draws, "conditional selected"
        )
        for draw_index in range(draws):
            observation = observations[start_index][draw_index]
            expected_accept = _guard_accepts(
                observation, settings["guard"]
            )
            if accepted[start_index][draw_index] is not expected_accept:
                raise ValueError("conditional accepted flag differs from raw guard evidence")
            expected_hash = (
                raw_hashes[draw_index]
                if expected_accept
                else source_state_hashes[start_index]
            )
            if selected_hashes[draw_index] != expected_hash:
                raise ValueError("conditional selected P/V is not exact raw-or-source")
            expected = raw[start_index, draw_index] if expected_accept else source[start_index]
            if not np.array_equal(guarded[start_index, draw_index], expected):
                raise ValueError("conditional guarded TIC is not exact raw-or-source")
            raw_tic_finite = bool(np.isfinite(raw[start_index, draw_index]).all())
            if observation.get("position_finite") is True and not raw_tic_finite:
                raise ValueError("finite raw positions produced non-finite TIC evidence")
            if observation.get("position_finite") is not True and raw_tic_finite:
                raise ValueError("non-finite raw positions produced finite TIC evidence")

    centers, labels = fit_kmeans(real, settings["msm"]["clusters"], seed=seed)
    one_step, _ = transition_matrix(
        labels,
        n_states=settings["msm"]["clusters"],
        lag=settings["msm"]["lag_frames"],
        pseudocount=settings["msm"]["pseudocount"],
    )
    delta = int(cell.get("delta_frames"))
    if delta < 1 or delta % settings["msm"]["lag_frames"]:
        raise ValueError("conditional delta/MSM lag mismatch")
    target_msm = np.linalg.matrix_power(
        one_step, delta // settings["msm"]["lag_frames"]
    )
    origins = assign_clusters(source, centers)
    repeated_origins = np.repeat(origins, draws)
    reference_delta = reference_transition_deltas(real, delta)

    variants = {
        "noop": np.repeat(source[:, None, :], draws, axis=1),
        "guarded": guarded,
    }
    if np.isfinite(raw).all():
        variants["raw"] = raw
    metrics = {}
    for name, predictions in variants.items():
        energy_values = [
            energy_score(predictions[index], target[index])
            for index in range(starts)
        ]
        destinations = assign_clusters(predictions.reshape(-1, components), centers)
        predicted_msm, origin_counts = transition_matrix(
            repeated_origins,
            destinations,
            n_states=settings["msm"]["clusters"],
            pseudocount=settings["msm"]["pseudocount"],
        )
        row_jsd, _ = weighted_row_jsd_bits(target_msm, predicted_msm, origin_counts)
        increments = predictions - source[:, None, :]
        metrics[name] = {
            "mean_energy_score": float(np.mean(energy_values)),
            "transition_energy_distance": energy_distance(
                reference_delta, increments.reshape(-1, components)
            ),
            "msm_row_jsd_bits": float(row_jsd),
        }
    if "raw" not in metrics:
        metrics["raw"] = None
    return metrics


def _statistics_from_step(step: dict[str, Any], name: str, starts: int) -> dict[str, np.ndarray]:
    payload = step.get(name)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"missing geometry {name}")
    result = {}
    for metric, values in payload.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (starts,) or not np.isfinite(array).all():
            raise ValueError(f"geometry {name}/{metric} is incomplete or non-finite")
        result[metric] = array
    return result


def _optional_statistics_from_step(
    step: dict[str, Any], name: str, starts: int
) -> dict[str, np.ndarray] | None:
    payload = step.get(name)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"missing geometry {name}")
    result = {}
    any_missing = False
    for metric, values in payload.items():
        if not isinstance(values, list) or len(values) != starts:
            raise ValueError(f"geometry {name}/{metric} shape mismatch")
        any_missing |= any(value is None for value in values)
        result[metric] = np.asarray(
            [np.nan if value is None else value for value in values], dtype=np.float64
        )
    if any_missing:
        return None
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"geometry {name} contains invalid numeric values")
    return result


def recompute_geometry_cell(
    cell: dict[str, Any], protocol: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    settings = protocol["geometry"]
    starts = settings["starts_per_cell"]
    if not isinstance(cell.get("start_frames"), list) or len(cell["start_frames"]) != starts:
        raise ValueError("geometry start-frame evidence mismatch")
    reference = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in cell.get("reference_statistics", {}).items()
    }
    if (
        not reference
        or any(value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all()
               for value in reference.values())
    ):
        raise ValueError("geometry reference statistics are incomplete")
    initial = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in cell.get("initial_statistics", {}).items()
    }
    if set(initial) != set(reference) or any(value.shape != (starts,) for value in initial.values()):
        raise ValueError("geometry initial statistics mismatch")
    previous_state_hashes = _require_state_hashes(
        cell.get("initial_state_sha256"), starts, "geometry initial"
    )
    steps = cell.get("steps_h100")
    if not isinstance(steps, list) or len(steps) != 100:
        raise ValueError("geometry evidence must contain exactly H100")
    guarded_by_step = []
    raw_by_step: list[dict[str, np.ndarray] | None] = []
    previous = initial
    for index, step in enumerate(steps, start=1):
        if step.get("step") != index:
            raise ValueError("geometry step sequence mismatch")
        accepted = step.get("accepted")
        observations = step.get("raw_guard_observations")
        source_observations = step.get("source_guard_observations")
        source_state_hashes = _require_state_hashes(
            step.get("source_state_sha256"), starts, "geometry source"
        )
        raw_state_hashes = _require_state_hashes(
            step.get("raw_state_sha256"), starts, "geometry raw"
        )
        selected_state_hashes = _require_state_hashes(
            step.get("selected_state_sha256"), starts, "geometry selected"
        )
        if source_state_hashes != previous_state_hashes:
            raise ValueError("geometry source P/V is not the exact current source")
        if not all(
            isinstance(value, list) and len(value) == starts
            for value in (accepted, observations, source_observations)
        ):
            raise ValueError("geometry guard evidence shape mismatch")
        if step.get("selected_position_exact") != [True] * starts:
            raise ValueError("geometry selected positions are not exact")
        if step.get("selected_vector_exact") != [True] * starts:
            raise ValueError("geometry selected vectors are not exact")
        guarded = _statistics_from_step(step, "guarded_statistics", starts)
        raw = _optional_statistics_from_step(step, "raw_statistics", starts)
        if raw is not None and set(raw) != set(guarded):
            raise ValueError("raw and guarded geometry metrics differ")
        for start_index in range(starts):
            expected_accept = _guard_accepts(observations[start_index], settings["guard"])
            if accepted[start_index] is not expected_accept:
                raise ValueError("geometry accepted flag differs from raw guard evidence")
            expected_hash = (
                raw_state_hashes[start_index]
                if expected_accept
                else source_state_hashes[start_index]
            )
            if selected_state_hashes[start_index] != expected_hash:
                raise ValueError("geometry selected P/V is not exact raw-or-current-source")
            source_obs = source_observations[start_index]
            if not _guard_accepts(source_obs, settings["guard"]):
                raise ValueError("geometry current source is not a valid safeguard source")
            if (
                source_obs.get("bond_mean") != previous["bond_mean"][start_index]
                or source_obs.get("bond_max") != previous["bond_max"][start_index]
            ):
                raise ValueError("geometry source guard facts differ from current source")
            if raw is not None:
                if (
                    observations[start_index].get("bond_mean")
                    != raw["bond_mean"][start_index]
                    or observations[start_index].get("bond_max")
                    != raw["bond_max"][start_index]
                ):
                    raise ValueError("geometry raw guard facts differ from raw statistics")
            if not expected_accept:
                for metric in guarded:
                    if guarded[metric][start_index] != previous[metric][start_index]:
                        raise ValueError("rejected geometry step is not exact current source")
            else:
                if raw is None:
                    raise ValueError("accepted geometry step has incomplete raw statistics")
                for metric in guarded:
                    if guarded[metric][start_index] != raw[metric][start_index]:
                        raise ValueError("accepted geometry step differs from raw proposal")
        guarded_by_step.append(guarded)
        raw_by_step.append(raw)
        previous = guarded
        previous_state_hashes = selected_state_hashes

    result = {}
    for horizon in settings["horizons"]:
        envelope = calibrate_geometry_worst_envelope(
            reference,
            starts,
            horizon,
            draws=settings["calibration_draws"],
            alpha=settings["real_envelope_alpha"],
            seed=seed,
        )
        guarded_panels = [
            aggregate_geometry_panel(guarded_by_step[index])
            for index in range(horizon)
        ]
        noop_panel = aggregate_geometry_panel(initial)
        result[f"h{horizon}"] = {
            "guarded_worst_excess": geometry_worst_excess(guarded_panels, envelope),
            "guarded_step_excess": [
                geometry_worst_excess([panel], envelope)
                for panel in guarded_panels
            ],
            "noop_worst_excess": geometry_worst_excess(
                [noop_panel for _ in range(horizon)], envelope
            ),
            "raw_worst_excess": (
                geometry_worst_excess(
                    [aggregate_geometry_panel(value) for value in raw_by_step[:horizon]],
                    envelope,
                )
                if all(value is not None for value in raw_by_step[:horizon])
                else None
            ),
        }
    return result


def _paired_gain(
    model: list[float],
    baseline: list[float],
    *,
    draws: int,
    seed: int,
    quantiles: tuple[float, float] = (0.025, 0.975),
    lower_gt: float = 0.0,
) -> dict:
    model_array = np.asarray(model, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if model_array.shape != baseline_array.shape or model_array.ndim != 1:
        raise ValueError("paired scientific gains require matching domain vectors")
    gains = baseline_array - model_array
    if len(gains) < 2 or not np.isfinite(gains).all():
        raise ValueError("paired scientific gains need finite multi-domain evidence")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gains), size=(draws, len(gains)))
    samples = gains[indices].mean(axis=1)
    low, high = np.quantile(samples, quantiles)
    return {
        "mean_baseline_minus_model": float(gains.mean()),
        "ci95": [float(low), float(high)],
        "domains": len(gains),
        "passes": bool(low > lower_gt),
    }


def independently_adjudicate(
    raw: dict[str, Any], protocol: dict[str, Any], session: dict[str, Any]
) -> dict[str, Any]:
    """Recompute every reported statistic from raw cells; trust no evaluator summary."""

    bootstrap_draws = protocol["adjudication"]["bootstrap_draws"]
    paired_quantiles = tuple(protocol["adjudication"]["paired_ci_quantiles"])
    seed = protocol["seed"]
    state_archive = _load_state_archive(raw)
    conditional_domains = []
    geometry_domains = []
    expected_cells = {
        (temperature, replica)
        for temperature in protocol["panel"]["temperatures_kelvin"]
        for replica in protocol["panel"]["replicas"]
    }

    def exact_cells(domain: dict[str, Any], label: str) -> list[dict[str, Any]]:
        cells = domain["cells"]
        actual = [(cell.get("temperature"), cell.get("replica")) for cell in cells]
        if len(actual) != len(expected_cells) or set(actual) != expected_cells:
            raise ValueError(f"{label} does not contain the exact 5x5 grid")
        if label.startswith("conditional"):
            replicas = protocol["panel"]["replicas"]
            for cell in cells:
                expected_reference = replicas[
                    (replicas.index(cell["replica"]) + 1) % len(replicas)
                ]
                if cell.get("reference_replica") != expected_reference:
                    raise ValueError("conditional cross-fit replica mismatch")
                if cell.get("delta_frames") != session["checkpoint_delta"]:
                    raise ValueError("conditional cell delta differs from the session")
        return sorted(cells, key=lambda cell: (cell["temperature"], cell["replica"]))

    for domain_index, domain in enumerate(raw["conditional_raw"]["domains"]):
        source_cells = exact_cells(domain, "conditional domain")
        for cell in source_cells:
            _verify_conditional_archive(cell, state_archive)
        cells = [
            recompute_conditional_cell(cell, protocol, seed=seed + domain_index * 1000 + i)
            for i, cell in enumerate(source_cells)
        ]
        conditional_domains.append({
            "domain": domain["domain"],
            "noop_energy": float(np.mean([row["noop"]["mean_energy_score"] for row in cells])),
            "guarded_energy": float(np.mean([row["guarded"]["mean_energy_score"] for row in cells])),
            "noop_msm": float(np.mean([row["noop"]["msm_row_jsd_bits"] for row in cells])),
            "guarded_msm": float(np.mean([row["guarded"]["msm_row_jsd_bits"] for row in cells])),
        })
    conditional = {
        "paired_energy_score_gain": _paired_gain(
            [row["guarded_energy"] for row in conditional_domains],
            [row["noop_energy"] for row in conditional_domains],
            draws=bootstrap_draws,
            seed=seed,
            quantiles=paired_quantiles,
            lower_gt=protocol["adjudication"]["energy_gain_ci95_lower_gt"],
        ),
        "paired_msm_row_jsd_gain": _paired_gain(
            [row["guarded_msm"] for row in conditional_domains],
            [row["noop_msm"] for row in conditional_domains],
            draws=bootstrap_draws,
            seed=seed + 1,
            quantiles=paired_quantiles,
            lower_gt=protocol["adjudication"]["msm_gain_ci95_lower_gt"],
        ),
        "domains": conditional_domains,
    }

    hard_threshold = protocol["adjudication"][
        "geometry_hard_envelope_all_domain_cell_step_lte"
    ]
    hard_violations = []
    for domain_index, domain in enumerate(raw["geometry_raw"]["domains"]):
        source_cells = exact_cells(domain, "geometry domain")
        for cell in source_cells:
            _verify_geometry_archive(cell, state_archive)
        cells = [
            recompute_geometry_cell(cell, protocol, seed=seed + domain_index * 1000 + i)
            for i, cell in enumerate(source_cells)
        ]
        row = {"domain": domain["domain"]}
        for horizon in protocol["geometry"]["horizons"]:
            key = f"h{horizon}"
            metrics = cells[0][key]["guarded_worst_excess"]
            row[key] = {
                metric: float(np.mean([
                    cell[key]["guarded_worst_excess"][metric] for cell in cells
                ]))
                for metric in metrics
            }
            ordered_source_cells = exact_cells(domain, "geometry domain")
            for cell_index, cell in enumerate(cells):
                step_rows = cell[key].get("guarded_step_excess")
                if not isinstance(step_rows, list) or len(step_rows) != horizon:
                    raise ValueError("geometry hard-envelope step evidence is incomplete")
                for step_index, step_metrics in enumerate(step_rows, start=1):
                    if set(step_metrics) != set(metrics):
                        raise ValueError("geometry hard-envelope metric set mismatch")
                    for metric, value in step_metrics.items():
                        if value > hard_threshold:
                            hard_violations.append({
                                "domain": domain["domain"],
                                "cell_index": cell_index,
                                "temperature": ordered_source_cells[cell_index]["temperature"],
                                "replica": ordered_source_cells[cell_index]["replica"],
                                "horizon": horizon,
                                "step": step_index,
                                "metric": metric,
                                "excess": float(value),
                            })
        geometry_domains.append(row)
    geometry = {
        "domains": geometry_domains,
        "horizons": {},
        "hard_envelope": {
            "threshold_lte": hard_threshold,
            "violations": hard_violations,
            "passes": not hard_violations,
        },
    }
    for horizon_index, horizon in enumerate(protocol["geometry"]["horizons"]):
        key = f"h{horizon}"
        geometry["horizons"][key] = {
            metric: bootstrap_domain_mean_upper(
                [row[key][metric] for row in geometry_domains],
                draws=bootstrap_draws,
                alpha=protocol["adjudication"][
                    "geometry_domain_mean_one_sided_alpha"
                ],
                seed=seed + 100 + horizon_index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(geometry_domains[0][key])
        }

    conditional_pass = bool(
        conditional["paired_energy_score_gain"]["passes"]
        and conditional["paired_msm_row_jsd_gain"]["passes"]
    )
    geometry_domain_mean_pass = all(
        metric["passes"]
        for horizon in geometry["horizons"].values()
        for metric in horizon.values()
    )
    geometry_pass = bool(
        geometry_domain_mean_pass and geometry["hard_envelope"]["passes"]
    )
    oracle = delta1_oracle_disposition(session)
    if oracle["status"] == INCONCLUSIVE_ORACLE:
        status = INCONCLUSIVE_ORACLE
    elif conditional_pass and geometry_pass:
        status = "PASS_CONTRACTED_SCIENTIFIC_BUNDLE"
    else:
        status = "FAIL_CONTRACTED_SCIENTIFIC_BUNDLE"
    return {
        "status": status,
        "conditional": conditional,
        "geometry": geometry,
        "conditional_pass": conditional_pass,
        "geometry_pass": geometry_pass,
        "geometry_domain_mean_pass": geometry_domain_mean_pass,
        "oracle": oracle,
        "formal_training_authorized": False,
    }


def main() -> None:
    import sys

    if sys.argv[1:] == ["--implementation-status"]:
        print(json.dumps(implementation_report(), sort_keys=True))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--expected-session-sha256", required=True)
    parser.add_argument("--prerequisite-decision", required=True)
    parser.add_argument("--expected-prerequisite-decision-sha256", required=True)
    parser.add_argument("--raw-evidence", required=True)
    parser.add_argument("--expected-raw-evidence-sha256", required=True)
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
    raw, _, raw_path = _read_regular_json(
        args.raw_evidence,
        args.expected_raw_evidence_sha256,
        "scientific raw evidence",
    )
    if str(raw_path) != session["raw_output"]:
        raise ValueError("scientific raw evidence path differs from the session")
    validate_raw_evidence(raw, session)
    if raw["prerequisite_sha256"] != prerequisite["sha256"]:
        raise ValueError("scientific raw prerequisite binding mismatch")
    verify_reserved_evaluation_claim(
        prerequisite,
        prerequisite["sha256"],
        raw["consumption_claim"],
        runtime_probe_output=session["runtime_probe_output"],
        output=session["raw_output"],
    )
    independently_verify_bound_inputs(
        raw=raw,
        protocol=protocol,
        session=session,
        checkpoint=args.checkpoint,
        contract=args.contract,
        panel_file=args.panel_file,
        data_root=args.data_root,
    )
    _verify_global_obs_claim(protocol, session, raw["global_obs_claim"])

    adjudication = independently_adjudicate(raw, protocol, session)
    _write_new_json(
        session["decision_output"],
        {
            "schema": DECISION_SCHEMA,
            **adjudication,
            "session_sha256": session["sha256"],
            "raw_evidence_sha256": args.expected_raw_evidence_sha256,
        },
    )
    print(json.dumps({
        "status": adjudication["status"],
        "decision_output": session["decision_output"],
        "formal_training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
