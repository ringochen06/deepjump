#!/usr/bin/env python
"""Fail-closed adjudication for the training-domain safeguard panel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from deepjump.evaluation import (
    MDCATH_REPLICAS,
    MDCATH_TEMPERATURES,
    load_frozen_domain_ids,
)
from scripts.adjudicate_endpoint_panel import _t_summary
from scripts.endpoint_panel_eval import EXPECTED_DOMAINS, EXPECTED_STARTS
from scripts.external_endpoint_identity import (
    _sha256,
    load_fresh_external_panels,
    load_paper_horizon_external_panels,
    verify_guarded_training_prerequisite,
    verify_multidomain_checkpoint,
    verify_paper_horizon_ab_prerequisite,
    verify_paper_vector_ab_prerequisite,
    verify_paper_vector_external_evidence,
)
from scripts.guarded_endpoint_panel_eval import (
    BOND_MAX,
    BOND_MEAN_HI,
    BOND_MEAN_LO,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_EXTERNAL_BYTES,
    EXPECTED_EXTERNAL_PANEL_SHA256,
    EXPECTED_PANEL_SHA256,
    EXPECTED_PAPER_HORIZON_EXTERNAL_BYTES,
    EXPECTED_PAPER_HORIZON_EXTERNAL_PANEL_SHA256,
    EXPECTED_PRIOR_EXTERNAL_SHA256,
    EXPECTED_TRAINING_SHA256,
    EXPECTED_TRAINING_DECISION_SHA256,
    EXPECTED_UNTOUCHED_SHA256,
    EXTERNAL_SCOPE,
    HORIZON_AB_BASELINE_PROFILE,
    MAX_FALLBACK_CELLS,
    MAX_FALLBACK_STARTS,
    CHECKPOINT_PROFILES,
    FROZEN_BASELINE_PROFILE,
    PAPER_HORIZON_EXTERNAL_SCOPE,
    PAPER_HORIZON_PROFILE,
    PAPER_VECTOR_EXTERNAL_SCOPE,
    PAPER_VECTOR_PROFILE,
    SCOPE,
    checkpoint_profile_requirements,
)


EXPECTED_CELLS = {
    (temperature, replica)
    for temperature in MDCATH_TEMPERATURES
    for replica in MDCATH_REPLICAS
}
MIN_DOMAINS_BETTER = 14
T_CRITICAL_19 = 2.093024054408263
T_CRITICAL_18 = 2.10092204024096
ZERO_WIDTH_EPS = 1e-12
FP64_MAX_ABS_DIFF = 1e-12


def _finite(value: object, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


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


def _close(actual: object, expected: float, *, label: str) -> float:
    value = _finite(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{label} mismatch")
    return value


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_domain_list: str | Path,
    training_domain_list_sha256: str,
    domain_list: str | Path,
    domain_list_sha256: str,
    *,
    panel_kind: str = "training",
    prior_external_domain_list: str | Path | None = None,
    prior_external_domain_list_sha256: str | None = None,
    prior_fresh_external_domain_list: str | Path | None = None,
    prior_fresh_external_domain_list_sha256: str | None = None,
    untouched_domain_list: str | Path | None = None,
    untouched_domain_list_sha256: str | None = None,
    prerequisite_decision: str | Path | None = None,
    prerequisite_decision_sha256: str | None = None,
    baseline_checkpoint_sha256: str | None = None,
    candidate_checkpoint_sha256: str | None = None,
    external_claim: str | Path | None = None,
    external_claim_sha256: str | None = None,
    external_download_manifest: str | Path | None = None,
    external_download_manifest_sha256: str | None = None,
    source_proof: str | Path | None = None,
    source_proof_sha256: str | None = None,
    checkpoint_profile: str = FROZEN_BASELINE_PROFILE,
) -> dict:
    if panel_kind not in {
        "training", "fresh-external", "paper-horizon-external",
        "paper-vector-external",
    }:
        raise ValueError("unknown guarded panel kind")
    expected_data, expected_model, expected_train = checkpoint_profile_requirements(
        checkpoint_profile, checkpoint_sha256
    )
    if panel_kind == "fresh-external" and checkpoint_profile != FROZEN_BASELINE_PROFILE:
        raise ValueError("legacy fresh-external requires the frozen baseline profile")
    if panel_kind == "paper-horizon-external" and checkpoint_profile not in {
        HORIZON_AB_BASELINE_PROFILE, PAPER_HORIZON_PROFILE
    }:
        raise ValueError("paper-horizon external requires a matched A/B checkpoint profile")
    if panel_kind == "paper-vector-external" and checkpoint_profile not in {
        PAPER_HORIZON_PROFILE, PAPER_VECTOR_PROFILE
    }:
        raise ValueError("paper-vector external requires a matched A/B checkpoint profile")
    if training_domain_list_sha256 != EXPECTED_TRAINING_SHA256:
        raise ValueError("training subset identity mismatch")
    expected_panel_sha = {
        "training": EXPECTED_PANEL_SHA256,
        "fresh-external": EXPECTED_EXTERNAL_PANEL_SHA256,
        "paper-horizon-external": EXPECTED_PAPER_HORIZON_EXTERNAL_PANEL_SHA256,
        "paper-vector-external": EXPECTED_PAPER_HORIZON_EXTERNAL_PANEL_SHA256,
    }[panel_kind]
    if domain_list_sha256 != expected_panel_sha:
        raise ValueError(f"{panel_kind} panel identity mismatch")
    checkpoint, train_fingerprint = verify_multidomain_checkpoint(
        checkpoint_path,
        checkpoint_sha256,
        expected_step=EXPECTED_CHECKPOINT_STEP,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )
    training_ids, training_sha = load_frozen_domain_ids(
        training_domain_list, training_domain_list_sha256
    )
    prerequisite = None
    external_evidence = None
    if panel_kind == "fresh-external":
        if prior_external_domain_list_sha256 != EXPECTED_PRIOR_EXTERNAL_SHA256:
            raise ValueError("prior external panel identity mismatch")
        if untouched_domain_list_sha256 != EXPECTED_UNTOUCHED_SHA256:
            raise ValueError("untouched panel identity mismatch")
        if prerequisite_decision_sha256 != EXPECTED_TRAINING_DECISION_SHA256:
            raise ValueError("training prerequisite decision identity mismatch")
        if not all((prior_external_domain_list, untouched_domain_list, prerequisite_decision)):
            raise ValueError("fresh external prerequisite paths are missing")
        contract = load_fresh_external_panels(
            training_domain_list, training_domain_list_sha256,
            prior_external_domain_list, prior_external_domain_list_sha256,
            untouched_domain_list, untouched_domain_list_sha256,
            domain_list, domain_list_sha256,
        )
        panel_ids = contract["fresh_external"]["ids"]
        panel_sha = contract["fresh_external"]["sha256"]
        prerequisite = verify_guarded_training_prerequisite(
            prerequisite_decision,
            prerequisite_decision_sha256,
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            expected_training_sha256=EXPECTED_TRAINING_SHA256,
        )
    elif panel_kind in {"paper-horizon-external", "paper-vector-external"}:
        if prior_external_domain_list_sha256 != EXPECTED_PRIOR_EXTERNAL_SHA256:
            raise ValueError("prior external panel identity mismatch")
        if prior_fresh_external_domain_list_sha256 != EXPECTED_EXTERNAL_PANEL_SHA256:
            raise ValueError("prior fresh external panel identity mismatch")
        if untouched_domain_list_sha256 != EXPECTED_UNTOUCHED_SHA256:
            raise ValueError("untouched panel identity mismatch")
        if not all((
            prior_external_domain_list,
            prior_fresh_external_domain_list,
            untouched_domain_list,
            prerequisite_decision,
            prerequisite_decision_sha256,
            baseline_checkpoint_sha256
            if panel_kind == "paper-vector-external" else True,
            candidate_checkpoint_sha256,
            external_claim if panel_kind == "paper-vector-external" else True,
            external_claim_sha256 if panel_kind == "paper-vector-external" else True,
            (
                external_download_manifest
                if panel_kind == "paper-vector-external" else True
            ),
            (
                external_download_manifest_sha256
                if panel_kind == "paper-vector-external" else True
            ),
            source_proof if panel_kind == "paper-vector-external" else True,
            source_proof_sha256 if panel_kind == "paper-vector-external" else True,
        )):
            raise ValueError(f"{panel_kind} prerequisite paths are missing")
        contract = load_paper_horizon_external_panels(
            training_domain_list, training_domain_list_sha256,
            prior_external_domain_list, prior_external_domain_list_sha256,
            prior_fresh_external_domain_list, prior_fresh_external_domain_list_sha256,
            untouched_domain_list, untouched_domain_list_sha256,
            domain_list, domain_list_sha256,
        )
        panel_ids = contract["paper_horizon_external"]["ids"]
        panel_sha = contract["paper_horizon_external"]["sha256"]
        if panel_kind == "paper-horizon-external":
            prerequisite = verify_paper_horizon_ab_prerequisite(
                prerequisite_decision,
                prerequisite_decision_sha256,
                expected_candidate_checkpoint_sha256=candidate_checkpoint_sha256,
                expected_training_sha256=EXPECTED_TRAINING_SHA256,
                expected_training_panel_sha256=EXPECTED_PANEL_SHA256,
            )
        else:
            prerequisite = verify_paper_vector_ab_prerequisite(
                prerequisite_decision,
                prerequisite_decision_sha256,
                expected_baseline_checkpoint_sha256=(
                    baseline_checkpoint_sha256
                ),
                expected_candidate_checkpoint_sha256=(
                    candidate_checkpoint_sha256
                ),
                expected_training_sha256=EXPECTED_TRAINING_SHA256,
                expected_training_panel_sha256=EXPECTED_PANEL_SHA256,
            )
            external_evidence = verify_paper_vector_external_evidence(
                external_claim,
                external_claim_sha256,
                external_download_manifest,
                external_download_manifest_sha256,
                expected_panel_sha256=panel_sha,
                expected_prerequisite_decision_sha256=(
                    prerequisite_decision_sha256
                ),
                expected_baseline_checkpoint_sha256=(
                    baseline_checkpoint_sha256
                ),
                expected_candidate_checkpoint_sha256=(
                    candidate_checkpoint_sha256
                ),
                source_proof_path=source_proof,
                expected_source_proof_sha256=source_proof_sha256,
            )
    else:
        panel_ids, panel_sha = load_frozen_domain_ids(domain_list, domain_list_sha256)
    if len(training_ids) != 1000 or len(set(training_ids)) != 1000:
        raise ValueError("training subset must contain 1000 unique domains")
    if len(panel_ids) != EXPECTED_DOMAINS or len(set(panel_ids)) != EXPECTED_DOMAINS:
        raise ValueError("training development panel must contain 20 unique domains")
    if panel_kind == "training" and not set(panel_ids).issubset(training_ids):
        raise ValueError("training development panel must be contained in training1000")

    result = json.loads(Path(result_path).read_text())
    expected_scope = {
        "training": SCOPE,
        "fresh-external": EXTERNAL_SCOPE,
        "paper-horizon-external": PAPER_HORIZON_EXTERNAL_SCOPE,
        "paper-vector-external": PAPER_VECTOR_EXTERNAL_SCOPE,
    }[panel_kind]
    if result.get("scope") != expected_scope:
        raise ValueError("guarded panel scope mismatch")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("result checkpoint SHA256 mismatch")
    if result.get("checkpoint_profile", FROZEN_BASELINE_PROFILE) != checkpoint_profile:
        raise ValueError("result checkpoint profile mismatch")
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("result checkpoint step mismatch")
    if int(result.get("checkpoint_schema", -1)) != 2:
        raise ValueError("result checkpoint schema mismatch")
    if int(result.get("checkpoint_train_seed", -1)) != 0:
        raise ValueError("result checkpoint seed mismatch")
    if result.get("checkpoint_train_fingerprint") != train_fingerprint:
        raise ValueError("result checkpoint fingerprint mismatch")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("result checkpoint path mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("guarded panel requires delta=1")
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
        raise ValueError("guarded panel settings mismatch")

    training = result.get("training_subset", {})
    if training.get("sha256") != training_sha or training.get("ids") != training_ids:
        raise ValueError("training subset identity or order mismatch")
    if training.get("train_fingerprint") != train_fingerprint:
        raise ValueError("training fingerprint mismatch")
    if int(training.get("domains_total", -1)) != 1000:
        raise ValueError("training data audit domain count mismatch")
    if int(training.get("train_domains", -1)) != 980:
        raise ValueError("checkpoint train split count mismatch")
    if int(training.get("validation_domains", -1)) != 20:
        raise ValueError("checkpoint validation split count mismatch")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != panel_sha or panel.get("ids") != panel_ids:
        raise ValueError("training development panel identity or order mismatch")
    if panel.get("subset_of_training1000") is not (panel_kind == "training"):
        raise ValueError("panel inclusion proof is missing")
    if panel_kind in {
        "fresh-external", "paper-horizon-external", "paper-vector-external"
    } and panel.get(
        "fresh_external"
    ) is not True:
        raise ValueError("fresh external panel identity flag mismatch")
    if panel_kind == "training" and panel.get("fresh_external") not in (None, False):
        raise ValueError("training panel is incorrectly marked fresh external")
    if panel_kind == "fresh-external":
        if int(panel.get("exclusion_union_count", -1)) != 1120:
            raise ValueError("fresh external exclusion-union proof is missing")
        if int(panel.get("total_bytes", -1)) != EXPECTED_EXTERNAL_BYTES:
            raise ValueError("fresh external panel byte count mismatch")
        if result.get("prerequisite") != prerequisite:
            raise ValueError("fresh external prerequisite binding mismatch")
    if panel_kind in {"paper-horizon-external", "paper-vector-external"}:
        if panel.get("paper_horizon_external") is not True:
            raise ValueError("paper-horizon external panel identity flag mismatch")
        if int(panel.get("exclusion_union_count", -1)) != 1140:
            raise ValueError("paper-horizon exclusion-union proof is missing")
        if int(panel.get("total_bytes", -1)) != EXPECTED_PAPER_HORIZON_EXTERNAL_BYTES:
            raise ValueError("paper-horizon external panel byte count mismatch")
        if result.get("prerequisite") != prerequisite:
            raise ValueError("paper-horizon prerequisite binding mismatch")
        if panel_kind == "paper-vector-external" and result.get(
            "external_evidence"
        ) != external_evidence:
            raise ValueError("paper-vector external evidence binding mismatch")
        expected_vector_flag = panel_kind == "paper-vector-external"
        if panel.get("paper_vector_external", False) is not expected_vector_flag:
            raise ValueError("paper-vector external panel identity flag mismatch")
    if int(panel.get("h5_files", -1)) != EXPECTED_DOMAINS:
        raise ValueError("training development panel HDF5 count mismatch")
    if int(panel.get("total_bytes", -1)) <= 0:
        raise ValueError("training development panel byte count is invalid")
    if result.get("grid") != {
        "temperatures": list(MDCATH_TEMPERATURES),
        "replicas": list(MDCATH_REPLICAS),
    }:
        raise ValueError("guarded panel must use the canonical 5x5 grid")

    mechanism = result.get("mechanism_probe", {})
    mechanism_passes = bool(
        mechanism.get("domain") == panel_ids[0]
        and int(mechanism.get("temperature", -1)) == MDCATH_TEMPERATURES[0]
        and int(mechanism.get("replica", -1)) == MDCATH_REPLICAS[0]
        and int(mechanism.get("target_slot", -1)) == 0
        and int(mechanism.get("target_start", -1)) >= 0
        and mechanism.get("same_shape_peer_position_bitwise_equal") is True
        and mechanism.get("same_shape_peer_vector_bitwise_equal") is True
        and _finite(
            mechanism.get("fp32_b1_b3_position_max_abs_diff"),
            label="FP32 position difference",
        ) >= 0
        and _finite(
            mechanism.get("fp32_b1_b3_vector_max_abs_diff"),
            label="FP32 vector difference",
        ) >= 0
        and 0 <= _finite(
            mechanism.get("fp64_b1_b3_position_max_abs_diff"),
            label="FP64 position difference",
        ) <= FP64_MAX_ABS_DIFF
        and 0 <= _finite(
            mechanism.get("fp64_b1_b3_vector_max_abs_diff"),
            label="FP64 vector difference",
        ) <= FP64_MAX_ABS_DIFF
        and type(mechanism.get("fp64_accept_b1")) is bool
        and type(mechanism.get("fp64_accept_b3")) is bool
        and mechanism.get("fp64_accept_b1") == mechanism.get("fp64_accept_b3")
    )

    probe = result.get("runtime_probe", {})
    if probe.get("status") != "PASS_RUNTIME_PROBE":
        raise ValueError("runtime probe did not pass")
    if probe.get("domain") not in panel_ids:
        raise ValueError("runtime probe domain is outside the panel")
    if int(probe.get("batch_size", -1)) != EXPECTED_STARTS:
        raise ValueError("runtime probe batch size mismatch")
    peak_fraction = _finite(probe.get("peak_memory_fraction"), label="peak fraction")
    projected_minutes = _finite(
        probe.get("projected_500_cell_minutes"), label="projected minutes"
    )
    if not 0 <= peak_fraction <= 0.8 or not 0 <= projected_minutes <= 50.0:
        raise ValueError("runtime probe exceeded frozen limits")
    if probe.get("limits") != {
        "max_peak_memory_fraction": 0.8,
        "max_projected_minutes": 50.0,
    }:
        raise ValueError("runtime probe limit identity mismatch")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != EXPECTED_DOMAINS:
        raise ValueError("guarded panel requires exactly 20 complete domains")
    if [domain.get("domain") for domain in domains] != panel_ids:
        raise ValueError("guarded panel domain identity or order mismatch")

    domain_deltas = []
    raw_finite_starts = source_physical_starts = guarded_physical_starts = 0
    source_physical_cells = guarded_physical_cells = 0
    fallback_starts = fallback_cells = 0
    guarded_cells_better = 0
    total_starts = EXPECTED_DOMAINS * len(EXPECTED_CELLS) * EXPECTED_STARTS
    for domain in domains:
        domain_id = domain["domain"]
        preprocessing = domain.get("preprocessing", {})
        residues = int(preprocessing.get("residues_total", -1))
        if preprocessing.get("canon_symmetric") is not True:
            raise ValueError("guarded panel requires canonical symmetric preprocessing")
        if residues <= 1 or int(preprocessing.get("residues_evaluated", -1)) != residues:
            raise ValueError("guarded panel must evaluate every residue")
        cells = domain.get("cells")
        if not isinstance(cells, list) or len(cells) != len(EXPECTED_CELLS):
            raise ValueError(f"domain {domain_id} requires exactly 25 cells")
        identities = {
            (int(cell.get("temperature", -1)), int(cell.get("replica", -1)))
            for cell in cells
        }
        if identities != EXPECTED_CELLS:
            raise ValueError(f"domain {domain_id} has missing, duplicate, or extra cells")
        cell_deltas = []
        domain_fallback_starts = domain_fallback_cells = 0
        for cell in cells:
            if cell.get("domain") != domain_id:
                raise ValueError("cell domain mismatch")
            frames = int(cell.get("frames", -1))
            if frames <= 1:
                raise ValueError("cell frame count is invalid")
            last = frames - 2
            expected_starts = [0, last // 2, last]
            if len(set(expected_starts)) != EXPECTED_STARTS:
                raise ValueError("cell cannot provide three distinct starts")
            if cell.get("starts") != expected_starts:
                raise ValueError("cell start panel mismatch")
            rows = cell.get("by_start")
            if not isinstance(rows, list) or len(rows) != EXPECTED_STARTS:
                raise ValueError("cell requires three per-start records")
            cell_guarded_deltas = []
            cell_source_physical = cell_guarded_physical = True
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
                source_physical = source_finite and source_geometry_physical
                raw_physical = raw_finite and raw_geometry_physical
                guarded_physical = guarded_finite and guarded_geometry_physical
                if source.get("physical") is not source_geometry_physical:
                    raise ValueError("source physical flag mismatch")
                if raw.get("physical") is not raw_geometry_physical:
                    raise ValueError("raw physical flag mismatch")
                if guarded.get("physical") is not guarded_geometry_physical:
                    raise ValueError("guarded physical flag mismatch")
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
                    if raw.get("position_finite") is True:
                        raw_rmsd = _finite(raw.get("rmsd"), label="raw RMSD")
                        _close(raw.get("minus_noop"), raw_rmsd - noop, label="raw-minus-noop")
                    elif raw.get("rmsd") is not None or raw.get("minus_noop") is not None:
                        raise ValueError("non-finite raw output must not report finite RMSD")
                    if not math.isclose(guarded_rmsd, noop, rel_tol=0, abs_tol=1e-9):
                        raise ValueError("fallback guarded RMSD differs from no-op RMSD")
                    for metric in ("bond_mean", "bond_max"):
                        _close(guarded.get(metric), float(source[metric]), label=f"fallback {metric}")
                raw_finite_starts += int(raw_finite)
                source_physical_starts += int(source_physical)
                guarded_physical_starts += int(guarded_physical)
                cell_source_physical &= source_physical
                cell_guarded_physical &= guarded_physical
                cell_fallbacks += int(not expected_accept)
                cell_guarded_deltas.append(guarded_delta)
            cell_delta = statistics.fmean(cell_guarded_deltas)
            _close(
                cell.get("mean_guarded_minus_noop"),
                cell_delta,
                label="cell mean guarded-minus-noop",
            )
            if cell.get("source_cell_physical") is not cell_source_physical:
                raise ValueError("source cell physical flag mismatch")
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
        if int(summary.get("cells_better_than_noop", -1)) != sum(v < 0 for v in cell_deltas):
            raise ValueError("domain cell win count mismatch")
        if int(summary.get("fallback_starts", -1)) != domain_fallback_starts:
            raise ValueError("domain fallback start count mismatch")
        if int(summary.get("fallback_cells", -1)) != domain_fallback_cells:
            raise ValueError("domain fallback cell count mismatch")
        domain_deltas.append(domain_delta)

    primary = _t_summary(domain_deltas, T_CRITICAL_19)
    domains_better = sum(value < 0 for value in domain_deltas)
    leave_one_out = []
    for index, excluded_domain in enumerate(panel_ids):
        summary = _t_summary(
            domain_deltas[:index] + domain_deltas[index + 1:], T_CRITICAL_18
        )
        summary["excluded_domain"] = excluded_domain
        summary["passes_negative"] = bool(
            summary["standard_error"] > ZERO_WIDTH_EPS
            and summary["ci95_model_minus_noop"][1] < 0
        )
        leave_one_out.append(summary)
    statistical_pass = bool(
        primary["standard_error"] > ZERO_WIDTH_EPS
        and primary["ci95_model_minus_noop"][1] < 0
        and domains_better >= MIN_DOMAINS_BETTER
        and all(row["passes_negative"] for row in leave_one_out)
    )
    all_raw_finite = raw_finite_starts == total_starts
    all_source_physical = source_physical_starts == total_starts
    all_guarded_physical = guarded_physical_starts == total_starts
    fallback_within_cap = bool(
        fallback_starts <= MAX_FALLBACK_STARTS
        and fallback_cells <= MAX_FALLBACK_CELLS
    )
    if not mechanism_passes:
        status = "STOP_CONDITIONAL_SAFEGUARD_MECHANISM"
    elif not all_raw_finite:
        status = "STOP_CONDITIONAL_SAFEGUARD_RAW_NONFINITE"
    elif not all_source_physical:
        status = "STOP_CONDITIONAL_SAFEGUARD_SOURCE_INVALID"
    elif not all_guarded_physical:
        status = "STOP_CONDITIONAL_SAFEGUARD_GUARDED_NONPHYSICAL"
    elif not fallback_within_cap:
        status = "STOP_CONDITIONAL_SAFEGUARD_FALLBACK_CAP"
    elif primary["standard_error"] <= ZERO_WIDTH_EPS:
        status = "INCONCLUSIVE_CONDITIONAL_SAFEGUARD_ZERO_VARIANCE"
    elif statistical_pass:
        status = {
            "training": "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20",
            "fresh-external": "PASS_CONDITIONAL_SAFEGUARD_EXTERNAL_DEV20",
            "paper-horizon-external": (
                "PASS_CONDITIONAL_SAFEGUARD_PAPER_HORIZON_EXTERNAL20"
            ),
            "paper-vector-external": (
                "PASS_CONDITIONAL_SAFEGUARD_PAPER_VECTOR_EXTERNAL20"
            ),
        }[panel_kind]
    else:
        status = "STOP_CONDITIONAL_SAFEGUARD_NO_ADVANTAGE"

    return {
        "status": status,
        "scope": {
            "training": "conditional reject-to-source training-domain 20x5x5x3 gate",
            "fresh-external": "conditional reject-to-source fresh-external 20x5x5x3 gate",
            "paper-horizon-external": (
                "conditional reject-to-source paper-horizon external 20x5x5x3 gate"
            ),
            "paper-vector-external": (
                "conditional reject-to-source paper-vector external 20x5x5x3 gate"
            ),
        }[panel_kind],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_profile": checkpoint_profile,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "checkpoint_train_fingerprint": train_fingerprint,
        "training_domain_list_sha256": training_sha,
        "domain_list_sha256": panel_sha,
        "result_sha256": _sha256(result_path),
        "external_evidence": external_evidence,
        "mechanism_passes": mechanism_passes,
        "domains": EXPECTED_DOMAINS,
        "cells": EXPECTED_DOMAINS * len(EXPECTED_CELLS),
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
        "leave_one_domain_out": leave_one_out,
        "guarded_cells_better_than_noop": guarded_cells_better,
        "decision_rule": expected_settings,
        "training_development_gate_completed": panel_kind == "training",
        "external_development_gate_completed": panel_kind in {
            "fresh-external", "paper-horizon-external", "paper-vector-external"
        },
        "external_development_authorized": (
            panel_kind == "training"
            and status == "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"
        ),
        "second_seed_authorized": (
            panel_kind == "fresh-external"
            and status == "PASS_CONDITIONAL_SAFEGUARD_EXTERNAL_DEV20"
        ),
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-profile", choices=CHECKPOINT_PROFILES,
                        default=FROZEN_BASELINE_PROFILE)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument(
        "--panel-kind",
        choices=(
            "training", "fresh-external", "paper-horizon-external",
            "paper-vector-external",
        ),
        default="training",
    )
    parser.add_argument("--prior-external-domain-list")
    parser.add_argument("--prior-external-domain-list-sha256")
    parser.add_argument("--prior-fresh-external-domain-list")
    parser.add_argument("--prior-fresh-external-domain-list-sha256")
    parser.add_argument("--untouched-domain-list")
    parser.add_argument("--untouched-domain-list-sha256")
    parser.add_argument("--prerequisite-decision")
    parser.add_argument("--prerequisite-decision-sha256")
    parser.add_argument("--baseline-checkpoint-sha256")
    parser.add_argument("--candidate-checkpoint-sha256")
    parser.add_argument("--external-claim")
    parser.add_argument("--external-claim-sha256")
    parser.add_argument("--external-download-manifest")
    parser.add_argument("--external-download-manifest-sha256")
    parser.add_argument("--source-proof")
    parser.add_argument("--source-proof-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.checkpoint,
        args.checkpoint_sha256,
        args.training_domain_list,
        args.training_domain_list_sha256,
        args.domain_list,
        args.domain_list_sha256,
        panel_kind=args.panel_kind,
        prior_external_domain_list=args.prior_external_domain_list,
        prior_external_domain_list_sha256=args.prior_external_domain_list_sha256,
        prior_fresh_external_domain_list=args.prior_fresh_external_domain_list,
        prior_fresh_external_domain_list_sha256=(
            args.prior_fresh_external_domain_list_sha256
        ),
        untouched_domain_list=args.untouched_domain_list,
        untouched_domain_list_sha256=args.untouched_domain_list_sha256,
        prerequisite_decision=args.prerequisite_decision,
        prerequisite_decision_sha256=args.prerequisite_decision_sha256,
        baseline_checkpoint_sha256=args.baseline_checkpoint_sha256,
        candidate_checkpoint_sha256=args.candidate_checkpoint_sha256,
        external_claim=args.external_claim,
        external_claim_sha256=args.external_claim_sha256,
        external_download_manifest=args.external_download_manifest,
        external_download_manifest_sha256=(
            args.external_download_manifest_sha256
        ),
        source_proof=args.source_proof,
        source_proof_sha256=args.source_proof_sha256,
        checkpoint_profile=args.checkpoint_profile,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
