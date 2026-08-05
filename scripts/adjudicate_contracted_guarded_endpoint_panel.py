#!/usr/bin/env python
"""Fail-closed adjudication for a contracted guarded endpoint panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from deepjump.data_contract import _read_regular_bytes
from deepjump.evaluation import (
    MDCATH_REPLICAS,
    MDCATH_TEMPERATURES,
    require_mdcath_full_grid,
    require_single_delta,
)
from deepjump.evaluation_consumption import verify_reserved_evaluation_claim
from deepjump.evaluation_contract import (
    _load_verified_checkpoint,
    verify_frozen_evaluation_identity,
)
from scripts.adjudicate_endpoint_panel import _t_summary
from scripts.contracted_guarded_endpoint_panel_eval import (
    PREREQUISITE_STATUS,
    SCOPE,
    _atomic_json_new,
    rehash_contracted_panel_payloads,
    verify_reserved_evaluation_prerequisite,
)
from scripts.endpoint_panel_eval import EXPECTED_STARTS
from scripts.guarded_endpoint_panel_eval import (
    BOND_MAX,
    BOND_MEAN_HI,
    BOND_MEAN_LO,
    MAX_FALLBACK_CELLS,
    MAX_FALLBACK_STARTS,
)


EXPECTED_CELLS = {
    (temperature, replica)
    for temperature in MDCATH_TEMPERATURES
    for replica in MDCATH_REPLICAS
}
EXPECTED_DOMAIN_COUNTS = {"development": 20, "external": 20, "untouched": 100}
T_CRITICAL = {
    20: (2.093024054408263, 2.10092204024096),
    100: (1.9842169515086827, 1.984467454426692),
}
FP64_MAX_ABS_DIFF = 1e-12
ZERO_WIDTH_EPS = 1e-12


def _load_exact_json(path: str | Path, expected_sha256: str, label: str) -> tuple[dict, str]:
    raw = _read_regular_bytes(Path(path).expanduser().resolve(), label)
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


def _finite(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _close(actual: object, expected: float, *, label: str) -> float:
    value = _finite(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{label} mismatch")
    return value


def _physical(metrics: dict) -> bool:
    mean = metrics.get("bond_mean")
    maximum = metrics.get("bond_max")
    return bool(
        mean is not None
        and maximum is not None
        and math.isfinite(float(mean))
        and math.isfinite(float(maximum))
        and BOND_MEAN_LO < float(mean) < BOND_MEAN_HI
        and float(maximum) < BOND_MAX
    )


def _verify_runtime_probe(
    result: dict,
    runtime_probe_path: str | Path,
    expected_runtime_probe_sha256: str,
    *,
    phase: str,
    panel_ids: list[str],
) -> dict:
    probe, probe_sha256 = _load_exact_json(
        runtime_probe_path, expected_runtime_probe_sha256, "runtime probe"
    )
    if result.get("runtime_probe_sha256") != probe_sha256:
        raise ValueError("result runtime-probe SHA256 binding mismatch")
    if result.get("runtime_probe") != probe:
        raise ValueError("result runtime-probe payload binding mismatch")
    maximum_minutes = 50.0 if phase == "external" else 250.0
    if probe.get("status") != "PASS_RUNTIME_PROBE":
        raise ValueError("runtime probe did not pass")
    if probe.get("domain") not in panel_ids:
        raise ValueError("runtime probe domain is outside the panel")
    if int(probe.get("residues", -1)) <= 1:
        raise ValueError("runtime probe residue count is invalid")
    cell_seconds = _finite(probe.get("cell_seconds"), label="runtime cell seconds")
    projected = _finite(
        probe.get("projected_panel_minutes"), label="runtime projected minutes"
    )
    peak_bytes = int(probe.get("peak_memory_bytes", -1))
    total_bytes = int(probe.get("total_memory_bytes", -1))
    peak_fraction = _finite(
        probe.get("peak_memory_fraction"), label="runtime peak-memory fraction"
    )
    if cell_seconds <= 0 or not 0 <= projected <= maximum_minutes:
        raise ValueError("runtime probe exceeded the frozen time limit")
    if total_bytes <= 0 or not 0 <= peak_bytes <= total_bytes:
        raise ValueError("runtime probe memory byte counts are invalid")
    if not math.isclose(peak_fraction, peak_bytes / total_bytes, rel_tol=0, abs_tol=1e-12):
        raise ValueError("runtime probe memory fraction mismatch")
    if not 0 <= peak_fraction <= 0.8:
        raise ValueError("runtime probe exceeded the frozen memory limit")
    if probe.get("limits") != {
        "max_peak_memory_fraction": 0.8,
        "max_projected_minutes": maximum_minutes,
    }:
        raise ValueError("runtime probe limit identity mismatch")
    return {**probe, "sha256": probe_sha256}


def _mechanism_passes(mechanism: dict, first_domain: str) -> bool:
    return bool(
        mechanism.get("domain") == first_domain
        and int(mechanism.get("temperature", -1)) == MDCATH_TEMPERATURES[0]
        and int(mechanism.get("replica", -1)) == MDCATH_REPLICAS[0]
        and int(mechanism.get("target_slot", -1)) == 0
        and int(mechanism.get("target_start", -1)) >= 0
        and mechanism.get("same_shape_peer_position_bitwise_equal") is True
        and mechanism.get("same_shape_peer_vector_bitwise_equal") is True
        and _finite(
            mechanism.get("fp32_b1_b3_position_max_abs_diff"),
            label="FP32 position difference",
        )
        >= 0
        and _finite(
            mechanism.get("fp32_b1_b3_vector_max_abs_diff"),
            label="FP32 vector difference",
        )
        >= 0
        and 0
        <= _finite(
            mechanism.get("fp64_b1_b3_position_max_abs_diff"),
            label="FP64 position difference",
        )
        <= FP64_MAX_ABS_DIFF
        and 0
        <= _finite(
            mechanism.get("fp64_b1_b3_vector_max_abs_diff"),
            label="FP64 vector difference",
        )
        <= FP64_MAX_ABS_DIFF
        and type(mechanism.get("fp64_accept_b1")) is bool
        and type(mechanism.get("fp64_accept_b3")) is bool
        and mechanism.get("fp64_accept_b1") == mechanism.get("fp64_accept_b3")
    )


def adjudicate(
    result_path: str | Path,
    expected_result_sha256: str,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    expected_checkpoint_step: int,
    contract_path: str | Path,
    expected_contract_sha256: str,
    phase: str,
    panel_name: str,
    panel_file: str | Path,
    prerequisite_decision: str | Path,
    expected_prerequisite_decision_sha256: str,
    runtime_probe_path: str | Path,
    expected_runtime_probe_sha256: str,
) -> dict:
    """Re-verify and scientifically adjudicate one immutable evaluator result."""

    if phase not in EXPECTED_DOMAIN_COUNTS:
        raise ValueError("phase must be 'development', 'external', or 'untouched'")
    expected_domains = EXPECTED_DOMAIN_COUNTS[phase]
    result, result_sha256 = _load_exact_json(
        result_path, expected_result_sha256, "contracted evaluator result"
    )
    if result.get("status") != "EVALUATION_COMPLETE_NOT_ADJUDICATED":
        raise ValueError("contracted evaluator result status mismatch")
    if result.get("scope") != SCOPE:
        raise ValueError("contracted evaluator result scope mismatch")
    if result.get("formal_training_authorized") is not False:
        raise ValueError("evaluator result must not authorize formal training")

    identity = verify_frozen_evaluation_identity(
        checkpoint_path,
        contract_path,
        expected_contract_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_checkpoint_step=expected_checkpoint_step,
        phase=phase,
        panel_name=panel_name,
        panel_file=panel_file,
    )
    if identity.get("panel_domains") != expected_domains:
        raise ValueError(f"{phase} phase requires exactly {expected_domains} domains")
    if result.get("identity") != identity:
        raise ValueError("contracted evaluator identity binding mismatch")
    prerequisite = verify_reserved_evaluation_prerequisite(
        prerequisite_decision,
        expected_prerequisite_decision_sha256,
        phase=phase,
        checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_step=expected_checkpoint_step,
        contract_sha256=expected_contract_sha256,
        panel_name=panel_name,
        panel_sha256=identity["panel_sha256"],
    )
    if prerequisite.get("status") != PREREQUISITE_STATUS[phase]:
        raise ValueError("reserved evaluation prerequisite status mismatch")
    if result.get("prerequisite") != prerequisite:
        raise ValueError("contracted evaluator prerequisite binding mismatch")
    consumption_claim = verify_reserved_evaluation_claim(
        prerequisite,
        expected_prerequisite_decision_sha256,
        result.get("consumption_claim"),
        runtime_probe_output=runtime_probe_path,
        output=result_path,
    )

    checkpoint, checkpoint_sha256 = _load_verified_checkpoint(
        Path(checkpoint_path).expanduser().resolve(), expected_checkpoint_sha256
    )
    if checkpoint_sha256 != identity["checkpoint_sha256"]:
        raise ValueError("checkpoint identity changed during adjudication")
    data_cfg = checkpoint.get("cfg", {}).get("data", {})
    delta = require_single_delta(data_cfg.get("delta_frames"))
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg.get("temperatures"), data_cfg.get("replicas")
    )
    if result.get("delta_frames") != delta:
        raise ValueError("evaluator delta differs from the checkpoint contract")
    if result.get("grid") != {
        "temperatures": temperatures,
        "replicas": replicas,
    }:
        raise ValueError("evaluator grid differs from the checkpoint contract")

    panel_raw = _read_regular_bytes(Path(panel_file).expanduser().resolve(), "panel file")
    if hashlib.sha256(panel_raw).hexdigest() != identity["panel_sha256"]:
        raise ValueError("panel changed during adjudication")
    try:
        panel_ids = panel_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("panel file is not valid UTF-8") from exc
    if len(panel_ids) != expected_domains or len(set(panel_ids)) != expected_domains:
        raise ValueError("panel count or uniqueness mismatch")

    payload = rehash_contracted_panel_payloads(
        contract_path, panel_file, data_cfg.get("root", "")
    )
    payload.pop("paths")
    payload.pop("pins")
    if result.get("payload_verification") != payload:
        raise ValueError("contracted panel payload report mismatch")
    runtime_probe = _verify_runtime_probe(
        result,
        runtime_probe_path,
        expected_runtime_probe_sha256,
        phase=phase,
        panel_ids=panel_ids,
    )

    expected_settings = {
        "starts": EXPECTED_STARTS,
        "start_strategy": "valid_source_linspace",
        "method": "mean",
        "source_noise": False,
        "policy": "reject_to_exact_source_per_start",
        "strict_thresholds": {
            "bond_mean_gt": BOND_MEAN_LO,
            "bond_mean_lt": BOND_MEAN_HI,
            "bond_max_lt": BOND_MAX,
        },
        "fallback_caps": {
            "max_starts": MAX_FALLBACK_STARTS,
            "max_cells": MAX_FALLBACK_CELLS,
        },
    }
    if result.get("settings") != expected_settings:
        raise ValueError("contracted evaluator settings mismatch")
    expected_completeness = {
        "status": "PASS_RESERVED_EVALUATION_EVIDENCE_COMPLETENESS",
        "domains": expected_domains,
        "cells": expected_domains * len(EXPECTED_CELLS),
        "starts": expected_domains * len(EXPECTED_CELLS) * EXPECTED_STARTS,
        "scientific_adjudication_performed": False,
    }
    if result.get("evidence_completeness") != expected_completeness:
        raise ValueError("evaluator evidence-completeness report mismatch")

    mechanism = result.get("mechanism_probe")
    if not isinstance(mechanism, dict):
        raise ValueError("mechanism probe is missing")
    mechanism_passes = _mechanism_passes(mechanism, panel_ids[0])
    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != expected_domains:
        raise ValueError(f"{phase} result requires exactly {expected_domains} domains")
    if [domain.get("domain") for domain in domains] != panel_ids:
        raise ValueError("result domain identity or order mismatch")

    domain_deltas: list[float] = []
    raw_finite_starts = source_physical_starts = guarded_physical_starts = 0
    source_physical_cells = guarded_physical_cells = 0
    fallback_starts = fallback_cells = guarded_cells_better = 0
    total_starts = expected_domains * len(EXPECTED_CELLS) * EXPECTED_STARTS
    for domain in domains:
        domain_id = domain["domain"]
        preprocessing = domain.get("preprocessing", {})
        residues = int(preprocessing.get("residues_total", -1))
        if preprocessing.get("canon_symmetric") is not True:
            raise ValueError("contracted panel requires canonical symmetric preprocessing")
        if residues <= 1 or int(preprocessing.get("residues_evaluated", -1)) != residues:
            raise ValueError("contracted panel must evaluate every residue")
        cells = domain.get("cells")
        if not isinstance(cells, list) or len(cells) != len(EXPECTED_CELLS):
            raise ValueError(f"domain {domain_id} requires exactly 25 cells")
        cell_identities = [
            (int(cell.get("temperature", -1)), int(cell.get("replica", -1)))
            for cell in cells
        ]
        if set(cell_identities) != EXPECTED_CELLS or len(set(cell_identities)) != len(cells):
            raise ValueError(f"domain {domain_id} has missing, duplicate, or extra cells")
        cell_deltas: list[float] = []
        domain_fallback_starts = domain_fallback_cells = 0
        for cell in cells:
            if cell.get("domain") != domain_id:
                raise ValueError("cell domain mismatch")
            frames = int(cell.get("frames", -1))
            if frames <= 1:
                raise ValueError("cell frame count is invalid")
            last = frames - delta - 1
            expected_starts = [0, last // 2, last]
            if len(set(expected_starts)) != EXPECTED_STARTS:
                raise ValueError("cell cannot provide three distinct starts")
            if cell.get("starts") != expected_starts:
                raise ValueError("cell start panel mismatch")
            rows = cell.get("by_start")
            if not isinstance(rows, list) or len(rows) != EXPECTED_STARTS:
                raise ValueError("cell requires three per-start records")
            cell_deltas_by_start: list[float] = []
            cell_source_physical = cell_raw_physical = cell_guarded_physical = True
            cell_fallbacks = 0
            for index, row in enumerate(rows):
                if int(row.get("start_index", -1)) != index:
                    raise ValueError("start index mismatch")
                if int(row.get("start_frame", -1)) != expected_starts[index]:
                    raise ValueError("start frame mismatch")
                if row.get("target_position_finite") is not True:
                    raise ValueError("target position must be finite")
                source = row.get("source", {})
                raw = row.get("raw", {})
                guarded = row.get("guarded", {})
                source_finite = bool(
                    source.get("position_finite") is True
                    and source.get("vector_finite") is True
                )
                raw_finite = bool(
                    raw.get("position_finite") is True
                    and raw.get("vector_finite") is True
                )
                guarded_finite = bool(
                    guarded.get("position_finite") is True
                    and guarded.get("vector_finite") is True
                )
                source_geometry_physical = _physical(source)
                raw_geometry_physical = _physical(raw)
                guarded_geometry_physical = _physical(guarded)
                if source.get("physical") is not source_geometry_physical:
                    raise ValueError("source physical flag mismatch")
                if raw.get("physical") is not raw_geometry_physical:
                    raise ValueError("raw physical flag mismatch")
                if guarded.get("physical") is not guarded_geometry_physical:
                    raise ValueError("guarded physical flag mismatch")
                source_physical = source_finite and source_geometry_physical
                raw_physical = raw_finite and raw_geometry_physical
                guarded_physical = guarded_finite and guarded_geometry_physical
                expected_accept = raw_finite and raw_physical
                if row.get("accepted") is not expected_accept:
                    raise ValueError("guard acceptance does not match strict raw predicate")
                if row.get("fallback") is not (not expected_accept):
                    raise ValueError("fallback flag mismatch")
                if row.get("selected_position_exact") is not True:
                    raise ValueError("guarded position is not the exact selected branch")
                if row.get("selected_vector_exact") is not True:
                    raise ValueError("guarded vector is not the exact selected branch")
                noop = _finite(row.get("noop_rmsd"), label="no-op RMSD")
                guarded_rmsd = _finite(guarded.get("rmsd"), label="guarded RMSD")
                guarded_delta = _close(
                    guarded.get("minus_noop"),
                    guarded_rmsd - noop,
                    label="guarded-minus-noop",
                )
                if expected_accept:
                    raw_rmsd = _finite(raw.get("rmsd"), label="raw RMSD")
                    _close(raw.get("minus_noop"), raw_rmsd - noop, label="raw-minus-noop")
                    if not math.isclose(guarded_rmsd, raw_rmsd, rel_tol=0, abs_tol=1e-9):
                        raise ValueError("accepted guarded RMSD differs from raw RMSD")
                    for metric in ("bond_mean", "bond_max"):
                        _close(guarded.get(metric), float(raw[metric]), label=f"accepted {metric}")
                else:
                    if raw_finite:
                        raw_rmsd = _finite(raw.get("rmsd"), label="raw RMSD")
                        _close(
                            raw.get("minus_noop"), raw_rmsd - noop, label="raw-minus-noop"
                        )
                    elif raw.get("rmsd") is not None or raw.get("minus_noop") is not None:
                        raise ValueError("non-finite raw output must not report finite RMSD")
                    if not math.isclose(guarded_rmsd, noop, rel_tol=0, abs_tol=1e-9):
                        raise ValueError("fallback guarded RMSD differs from no-op RMSD")
                    for metric in ("bond_mean", "bond_max"):
                        _close(
                            guarded.get(metric), float(source[metric]), label=f"fallback {metric}"
                        )
                raw_finite_starts += int(raw_finite)
                source_physical_starts += int(source_physical)
                guarded_physical_starts += int(guarded_physical)
                cell_source_physical &= source_physical
                cell_raw_physical &= raw_physical
                cell_guarded_physical &= guarded_physical
                cell_fallbacks += int(not expected_accept)
                cell_deltas_by_start.append(guarded_delta)
            cell_delta = statistics.fmean(cell_deltas_by_start)
            _close(
                cell.get("mean_guarded_minus_noop"),
                cell_delta,
                label="cell mean guarded-minus-noop",
            )
            if cell.get("source_cell_physical") is not cell_source_physical:
                raise ValueError("source cell physical flag mismatch")
            if cell.get("raw_cell_physical") is not cell_raw_physical:
                raise ValueError("raw cell physical flag mismatch")
            if cell.get("guarded_cell_physical") is not cell_guarded_physical:
                raise ValueError("guarded cell physical flag mismatch")
            if int(cell.get("fallback_starts", -1)) != cell_fallbacks:
                raise ValueError("cell fallback count mismatch")
            source_physical_cells += int(cell_source_physical)
            guarded_physical_cells += int(cell_guarded_physical)
            fallback_starts += cell_fallbacks
            fallback_cells += int(cell_fallbacks > 0)
            domain_fallback_starts += cell_fallbacks
            domain_fallback_cells += int(cell_fallbacks > 0)
            guarded_cells_better += int(cell_delta < 0)
            cell_deltas.append(cell_delta)
        domain_delta = statistics.fmean(cell_deltas)
        summary = domain.get("summary", {})
        if int(summary.get("cells", -1)) != len(EXPECTED_CELLS):
            raise ValueError("domain summary cell count mismatch")
        _close(
            summary.get("mean_guarded_minus_noop"),
            domain_delta,
            label="domain mean guarded-minus-noop",
        )
        if int(summary.get("cells_better_than_noop", -1)) != sum(
            value < 0 for value in cell_deltas
        ):
            raise ValueError("domain cell win count mismatch")
        if int(summary.get("fallback_starts", -1)) != domain_fallback_starts:
            raise ValueError("domain fallback start count mismatch")
        if int(summary.get("fallback_cells", -1)) != domain_fallback_cells:
            raise ValueError("domain fallback cell count mismatch")
        domain_deltas.append(domain_delta)

    primary_critical, loo_critical = T_CRITICAL[expected_domains]
    primary = _t_summary(domain_deltas, primary_critical)
    domains_better = sum(value < 0 for value in domain_deltas)
    minimum_domains_better = math.ceil(0.7 * expected_domains)
    leave_one_out = []
    for index, excluded_domain in enumerate(panel_ids):
        summary = _t_summary(domain_deltas[:index] + domain_deltas[index + 1 :], loo_critical)
        summary["excluded_domain"] = excluded_domain
        summary["passes_negative"] = bool(
            summary["standard_error"] > ZERO_WIDTH_EPS
            and summary["ci95_model_minus_noop"][1] < 0
        )
        leave_one_out.append(summary)
    statistical_pass = bool(
        primary["standard_error"] > ZERO_WIDTH_EPS
        and primary["ci95_model_minus_noop"][1] < 0
        and domains_better >= minimum_domains_better
        and all(row["passes_negative"] for row in leave_one_out)
    )
    all_raw_finite = raw_finite_starts == total_starts
    all_source_physical = source_physical_starts == total_starts
    all_guarded_physical = guarded_physical_starts == total_starts
    fallback_within_cap = bool(
        fallback_starts <= MAX_FALLBACK_STARTS and fallback_cells <= MAX_FALLBACK_CELLS
    )
    if not mechanism_passes:
        gate_status = "STOP_CONTRACTED_GUARD_MECHANISM"
    elif not all_raw_finite:
        gate_status = "STOP_CONTRACTED_GUARD_RAW_NONFINITE"
    elif not all_source_physical:
        gate_status = "STOP_CONTRACTED_GUARD_SOURCE_INVALID"
    elif not all_guarded_physical:
        gate_status = "STOP_CONTRACTED_GUARD_GUARDED_NONPHYSICAL"
    elif not fallback_within_cap:
        gate_status = "STOP_CONTRACTED_GUARD_FALLBACK_CAP"
    elif primary["standard_error"] <= ZERO_WIDTH_EPS:
        gate_status = "STOP_CONTRACTED_GUARD_ZERO_VARIANCE"
    elif statistical_pass:
        gate_status = {
            "development": "PASS_CONTRACTED_GUARD_DEVELOPMENT20",
            "external": "PASS_CONTRACTED_GUARD_EXTERNAL20",
            "untouched": "PASS_CONTRACTED_GUARD_UNTOUCHED100",
        }[phase]
    else:
        gate_status = "STOP_CONTRACTED_GUARD_NO_ADVANTAGE"

    status = (
        "ADVANCE_EXPANDED_DATA_EXTERNAL"
        if phase == "development"
        and gate_status == "PASS_CONTRACTED_GUARD_DEVELOPMENT20"
        else gate_status
    )

    return {
        "status": status,
        "gate_status": gate_status,
        "scope": f"contracted guarded {phase} {expected_domains}x5x5x3 adjudication",
        "phase": phase,
        "result_sha256": result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": expected_checkpoint_step,
        "full_training_contract_sha256": expected_contract_sha256,
        "panel_name": panel_name,
        "panel_sha256": identity["panel_sha256"],
        "prerequisite_decision_sha256": expected_prerequisite_decision_sha256,
        "identity": identity,
        "prerequisite": prerequisite,
        "consumption_claim": consumption_claim,
        "payload_verification": payload,
        "runtime_probe": runtime_probe,
        "mechanism_passes": mechanism_passes,
        "domains": expected_domains,
        "cells": expected_domains * len(EXPECTED_CELLS),
        "starts": total_starts,
        "raw_finite_starts": raw_finite_starts,
        "source_physical_starts": source_physical_starts,
        "source_physical_cells": source_physical_cells,
        "guarded_physical_starts": guarded_physical_starts,
        "guarded_physical_cells": guarded_physical_cells,
        "fallback_starts": fallback_starts,
        "fallback_cells": fallback_cells,
        "fallback_within_cap": fallback_within_cap,
        "domain_mean_guarded_minus_noop": domain_deltas,
        "primary": primary,
        "primary_domains_better_than_noop": domains_better,
        "minimum_domains_better_than_noop": minimum_domains_better,
        "leave_one_domain_out": leave_one_out,
        "guarded_cells_better_than_noop": guarded_cells_better,
        "decision_rule": expected_settings,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument(
        "--phase", choices=("development", "external", "untouched"), required=True
    )
    parser.add_argument("--panel-name", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--prerequisite-decision", required=True)
    parser.add_argument("--expected-prerequisite-decision-sha256", required=True)
    parser.add_argument("--runtime-probe", required=True)
    parser.add_argument("--expected-runtime-probe-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.expected_result_sha256,
        args.checkpoint,
        args.expected_checkpoint_sha256,
        args.expected_checkpoint_step,
        args.contract,
        args.expected_contract_sha256,
        args.phase,
        args.panel_name,
        args.panel_file,
        args.prerequisite_decision,
        args.expected_prerequisite_decision_sha256,
        args.runtime_probe,
        args.expected_runtime_probe_sha256,
    )
    _atomic_json_new(Path(args.output), report)
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
