#!/usr/bin/env python
"""Fail-closed adjudication for the external multi-domain H1 panel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from deepjump.evaluation import MDCATH_REPLICAS, MDCATH_TEMPERATURES
from scripts.adjudicate_endpoint_panel import _finite_vector, _t_summary
from scripts.adjudicate_source_law_candidate import BOND_MEAN_RANGE, MAX_BOND_MAX
from scripts.endpoint_panel_eval import (
    EXPECTED_DOMAINS,
    EXPECTED_STARTS,
    MAX_PEAK_MEMORY_FRACTION,
    MAX_PROJECTED_MINUTES,
)
from scripts.external_endpoint_identity import (
    EXPECTED_CHECKPOINT_SCHEMA,
    EXPECTED_CHECKPOINT_STEP,
    _sha256,
    load_disjoint_panels,
    verify_multidomain_checkpoint,
)


EXPECTED_EXTERNAL_BYTES = 13_778_143_616
EXPECTED_CELLS = {
    (temperature, replica)
    for temperature in MDCATH_TEMPERATURES
    for replica in MDCATH_REPLICAS
}
MIN_DOMAINS_BETTER = 14
T_CRITICAL_19 = 2.093024054408263
T_CRITICAL_18 = 2.10092204024096
ZERO_WIDTH_EPS = 1e-12


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_domain_list: str | Path,
    training_domain_list_sha256: str,
    external_domain_list: str | Path,
    external_domain_list_sha256: str,
) -> dict:
    _, train_fingerprint = verify_multidomain_checkpoint(
        checkpoint_path, checkpoint_sha256
    )
    training_ids, training_sha256, domain_ids, domain_sha256 = load_disjoint_panels(
        training_domain_list,
        training_domain_list_sha256,
        external_domain_list,
        external_domain_list_sha256,
    )

    result = json.loads(Path(result_path).read_text())
    if result.get("scope") != "external_multidomain_fp32_pilot_h1":
        raise ValueError("external endpoint result scope mismatch")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("result checkpoint SHA256 mismatch")
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("external endpoint gate requires checkpoint step 1000")
    if int(result.get("checkpoint_schema", -1)) != EXPECTED_CHECKPOINT_SCHEMA:
        raise ValueError("external endpoint checkpoint schema mismatch")
    if result.get("checkpoint_train_fingerprint") != train_fingerprint:
        raise ValueError("external endpoint train fingerprint mismatch")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("external endpoint gate requires delta=1")
    if result.get("settings") != {
        "starts": EXPECTED_STARTS,
        "start_strategy": "valid_source_linspace",
        "method": "mean",
        "source_noise": False,
    }:
        raise ValueError("external endpoint settings mismatch")

    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_sha256:
        raise ValueError("external domain panel SHA256 mismatch")
    if panel.get("ids") != domain_ids:
        raise ValueError("external domain panel identity or order mismatch")
    if int(panel.get("h5_files", -1)) != EXPECTED_DOMAINS:
        raise ValueError("external domain panel HDF5 count mismatch")
    if int(panel.get("total_bytes", -1)) != EXPECTED_EXTERNAL_BYTES:
        raise ValueError("external domain panel byte count mismatch")
    training = result.get("training_subset", {})
    if training.get("sha256") != training_sha256 or training.get("ids") != training_ids:
        raise ValueError("training subset identity or order mismatch")
    if training.get("train_fingerprint") != train_fingerprint:
        raise ValueError("training subset fingerprint mismatch")
    if int(training.get("domains_total", -1)) != 1000:
        raise ValueError("training subset domain count mismatch")
    if int(training.get("train_domains", -1)) != 980:
        raise ValueError("checkpoint training split count mismatch")
    if int(training.get("validation_domains", -1)) != 20:
        raise ValueError("checkpoint validation split count mismatch")
    if result.get("grid") != {
        "temperatures": list(MDCATH_TEMPERATURES),
        "replicas": list(MDCATH_REPLICAS),
    }:
        raise ValueError("external endpoint panel must use the canonical 5x5 grid")

    runtime_probe = result.get("runtime_probe", {})
    if runtime_probe.get("status") != "PASS_RUNTIME_PROBE":
        raise ValueError("external endpoint runtime probe did not pass")
    if runtime_probe.get("domain") not in domain_ids:
        raise ValueError("runtime probe domain is outside the external panel")
    if int(runtime_probe.get("residues", -1)) <= 0:
        raise ValueError("runtime probe residue count is invalid")
    if int(runtime_probe.get("batch_size", -1)) != EXPECTED_STARTS:
        raise ValueError("runtime probe batch size mismatch")
    cell_seconds = float(runtime_probe.get("cell_seconds", math.nan))
    peak_bytes = int(runtime_probe.get("peak_memory_bytes", -1))
    total_bytes = int(runtime_probe.get("total_memory_bytes", -1))
    peak_fraction = float(runtime_probe.get("peak_memory_fraction", math.nan))
    projected_minutes = float(runtime_probe.get("projected_500_cell_minutes", math.nan))
    if (
        not math.isfinite(cell_seconds) or cell_seconds < 0
        or peak_bytes < 0 or total_bytes <= 0
        or not math.isfinite(peak_fraction)
        or peak_fraction < 0 or peak_fraction > MAX_PEAK_MEMORY_FRACTION
        or not math.isfinite(projected_minutes)
        or projected_minutes < 0 or projected_minutes > MAX_PROJECTED_MINUTES
        or runtime_probe.get("limits") != {
            "max_peak_memory_fraction": MAX_PEAK_MEMORY_FRACTION,
            "max_projected_minutes": MAX_PROJECTED_MINUTES,
        }
    ):
        raise ValueError("runtime probe limits mismatch")
    if not math.isclose(peak_fraction, peak_bytes / total_bytes, rel_tol=0, abs_tol=1e-12):
        raise ValueError("runtime probe memory fraction mismatch")
    if not math.isclose(
        projected_minutes,
        cell_seconds * EXPECTED_DOMAINS * len(EXPECTED_CELLS) / 60,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("runtime probe projected duration mismatch")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != EXPECTED_DOMAINS:
        raise ValueError("external endpoint panel requires exactly 20 complete domains")
    if [domain.get("domain") for domain in domains] != domain_ids:
        raise ValueError("external domain result identity or order mismatch")

    domain_deltas = []
    physical_cells = 0
    cells_better = 0
    for domain in domains:
        domain_id = domain["domain"]
        preprocessing = domain.get("preprocessing", {})
        if preprocessing.get("canon_symmetric") is not True:
            raise ValueError("external endpoint gate requires canonical symmetric atom slots")
        residues_total = int(preprocessing.get("residues_total", -1))
        if residues_total <= 0 or int(preprocessing.get("residues_evaluated", -1)) != residues_total:
            raise ValueError(f"domain {domain_id} must evaluate all residues")
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
        for cell in cells:
            if cell.get("domain") != domain_id:
                raise ValueError("cell domain mismatch")
            frames = int(cell.get("frames", -1))
            if frames <= 1:
                raise ValueError("cell frame count is invalid")
            last = frames - 2
            expected_starts = [0, last // 2, last]
            if len(set(expected_starts)) != EXPECTED_STARTS or cell.get("starts") != expected_starts:
                raise ValueError("cell start panel mismatch")
            model = _finite_vector(cell.get("model_rmsd_by_start"), label="model_rmsd_by_start")
            noop = _finite_vector(cell.get("noop_rmsd_by_start"), label="noop_rmsd_by_start")
            recorded = _finite_vector(
                cell.get("model_minus_noop_by_start"), label="model_minus_noop_by_start"
            )
            paired = [m - n for m, n in zip(model, noop)]
            if any(not math.isclose(a, e, rel_tol=0, abs_tol=1e-9) for a, e in zip(recorded, paired)):
                raise ValueError("recorded paired RMSD does not match model-minus-noop")
            cell_delta = statistics.fmean(paired)
            if not math.isclose(
                float(cell.get("mean_model_minus_noop", math.nan)),
                cell_delta,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("cell mean paired RMSD mismatch")
            bond_mean = float(cell.get("bond_mean", math.nan))
            bond_max = float(cell.get("bond_max", math.nan))
            if not math.isfinite(bond_mean) or not math.isfinite(bond_max):
                raise ValueError("cell bond metrics must be finite")
            physical_cells += int(
                BOND_MEAN_RANGE[0] <= bond_mean <= BOND_MEAN_RANGE[1]
                and bond_max <= MAX_BOND_MAX
            )
            cells_better += int(cell_delta < 0)
            cell_deltas.append(cell_delta)
        domain_delta = statistics.fmean(cell_deltas)
        summary = domain.get("summary", {})
        if int(summary.get("cells", -1)) != len(EXPECTED_CELLS):
            raise ValueError("domain summary cell count mismatch")
        if not math.isclose(
            float(summary.get("mean_model_minus_noop", math.nan)),
            domain_delta,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("domain mean paired RMSD mismatch")
        if int(summary.get("cells_better_than_noop", -1)) != sum(v < 0 for v in cell_deltas):
            raise ValueError("domain cell win count mismatch")
        domain_deltas.append(domain_delta)

    primary = _t_summary(domain_deltas, T_CRITICAL_19)
    primary_low, primary_high = primary["ci95_model_minus_noop"]
    domains_better = sum(value < 0 for value in domain_deltas)
    domains_worse = sum(value > 0 for value in domain_deltas)
    leave_one_out = []
    for index, excluded_domain in enumerate(domain_ids):
        subset = domain_deltas[:index] + domain_deltas[index + 1:]
        summary = _t_summary(subset, T_CRITICAL_18)
        summary["excluded_domain"] = excluded_domain
        summary["passes_negative"] = (
            summary["standard_error"] > ZERO_WIDTH_EPS
            and summary["ci95_model_minus_noop"][1] < 0
        )
        leave_one_out.append(summary)
    t_negative = primary["standard_error"] > ZERO_WIDTH_EPS and primary_high < 0
    sign_negative = domains_better >= MIN_DOMAINS_BETTER
    t_positive = primary["standard_error"] > ZERO_WIDTH_EPS and primary_low > 0
    sign_positive = domains_worse >= MIN_DOMAINS_BETTER
    jackknife_negative = all(item["passes_negative"] for item in leave_one_out)
    all_physical = physical_cells == EXPECTED_DOMAINS * len(EXPECTED_CELLS)
    if not all_physical:
        status = "STOP_NONPHYSICAL_EXTERNAL_DEV20_ENDPOINT"
    elif primary["standard_error"] <= ZERO_WIDTH_EPS:
        status = "INCONCLUSIVE_ZERO_VARIANCE_EXTERNAL_DEV20_ENDPOINT"
    elif t_negative and sign_negative and jackknife_negative:
        status = "PASS_EXTERNAL_DEV20_ENDPOINT"
    elif t_positive and sign_positive:
        status = "STOP_DOMAIN_DISADVANTAGE_EXTERNAL_DEV20_ENDPOINT"
    elif t_negative != sign_negative or t_positive != sign_positive or (t_negative and not jackknife_negative):
        status = "INCONCLUSIVE_EXTERNAL_DEV20_ENDPOINT"
    else:
        status = "STOP_NULL_EXTERNAL_DEV20_ENDPOINT"

    return {
        "status": status,
        "scope": "external 20-domain 5x5-cell clean-source H1 gate for multi-domain FP32 pilot",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_train_fingerprint": train_fingerprint,
        "training_domain_list_sha256": training_sha256,
        "external_domain_list_sha256": domain_sha256,
        "result_sha256": _sha256(result_path),
        "domains": EXPECTED_DOMAINS,
        "cells": EXPECTED_DOMAINS * len(EXPECTED_CELLS),
        "starts": EXPECTED_DOMAINS * len(EXPECTED_CELLS) * EXPECTED_STARTS,
        "domain_mean_model_minus_noop": domain_deltas,
        "primary": primary,
        "primary_domains_better_than_noop": domains_better,
        "leave_one_domain_out": leave_one_out,
        "cells_better_than_noop": cells_better,
        "physical_cells": physical_cells,
        "decision_rule": {
            "PASS_EXTERNAL_DEV20_ENDPOINT": (
                "all 20 external domains and 500 cells are complete and physical; the 95% t "
                "interval upper bound for domain-balanced mean(model-noop) is below zero; at "
                "least 14/20 domain means are negative; every leave-one-domain-out interval passes"
            ),
            "STOP_NONPHYSICAL_EXTERNAL_DEV20_ENDPOINT": "at least one of 500 cells violates the bond gate",
            "INCONCLUSIVE": "all other outcomes fail closed and do not authorize training",
        },
        "external_gate_completed": True,
        "second_seed_authorized": status == "PASS_EXTERNAL_DEV20_ENDPOINT",
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
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
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
