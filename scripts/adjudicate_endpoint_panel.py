#!/usr/bin/env python
"""Fail-closed domain-level adjudication for the frozen 20-domain H1 panel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from deepjump.evaluation import MDCATH_REPLICAS, MDCATH_TEMPERATURES
from scripts.adjudicate_source_law_candidate import (
    BOND_MEAN_RANGE,
    EXPECTED_CHECKPOINT_STEP,
    MAX_BOND_MAX,
    _sha256,
    _verify_checkpoint_source_law,
)
from scripts.endpoint_panel_eval import (
    EXPECTED_DOMAINS,
    EXPECTED_STARTS,
    MAX_PEAK_MEMORY_FRACTION,
    MAX_PROJECTED_MINUTES,
)


EXPECTED_DOMAIN_IDS = (
    "1gxlA02", "1nh8A02", "1qhdA01", "1qu3A05", "1s5lH00",
    "1vddA03", "1zcaA02", "1zu2A00", "2b5eA03", "2dgmA02",
    "2e9xD02", "2kl5A00", "2nluA02", "2ogyA00", "2xhgA02",
    "3fk5A01", "3ha4B00", "3k6yA01", "4agrB00", "4i9cA01",
)
EXPECTED_CELLS = {
    (temperature, replica)
    for temperature in MDCATH_TEMPERATURES
    for replica in MDCATH_REPLICAS
}
PRIMARY_DOMAINS = EXPECTED_DOMAINS - 1
MIN_PRIMARY_DOMAINS_BETTER = 14
T_CRITICAL_18 = 2.10092204024096
T_CRITICAL_17 = 2.10981557783318
ZERO_WIDTH_EPS = 1e-12


def _finite_vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != EXPECTED_STARTS:
        raise ValueError(f"{label} must contain exactly three starts")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _t_summary(values: list[float], critical: float) -> dict:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("t summary requires at least two finite domain effects")
    mean = statistics.fmean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "domains": len(values),
        "mean_model_minus_noop": mean,
        "standard_error": standard_error,
        "t_critical": critical,
        "ci95_model_minus_noop": [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
    }


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> dict:
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    _verify_checkpoint_source_law(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["cfg"]["data"].get("domains") != ["1a0hA01"]:
        raise ValueError("checkpoint training domain mismatch")
    result = json.loads(Path(result_path).read_text())
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("result checkpoint SHA256 mismatch")
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("endpoint panel requires checkpoint step 1000")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("endpoint panel requires delta=1")
    if result.get("settings") != {
        "starts": EXPECTED_STARTS,
        "start_strategy": "valid_source_linspace",
        "method": "mean",
        "source_noise": False,
    }:
        raise ValueError("endpoint panel settings mismatch")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256:
        raise ValueError("domain panel SHA256 mismatch")
    if panel.get("ids") != list(EXPECTED_DOMAIN_IDS):
        raise ValueError("endpoint panel identity or order mismatch")
    if "1a0hA01" in panel["ids"]:
        raise ValueError("endpoint panel contains the checkpoint training domain")
    if result.get("grid") != {
        "temperatures": list(MDCATH_TEMPERATURES),
        "replicas": list(MDCATH_REPLICAS),
    }:
        raise ValueError("endpoint panel must use the canonical 5x5 grid")
    runtime_probe = result.get("runtime_probe", {})
    if runtime_probe.get("status") != "PASS_RUNTIME_PROBE":
        raise ValueError("endpoint panel runtime probe did not pass")
    if runtime_probe.get("domain") not in EXPECTED_DOMAIN_IDS:
        raise ValueError("runtime probe domain is outside the frozen panel")
    if int(runtime_probe.get("residues", -1)) <= 0:
        raise ValueError("runtime probe residue count is invalid")
    if int(runtime_probe.get("batch_size", -1)) != EXPECTED_STARTS:
        raise ValueError("runtime probe batch size mismatch")
    cell_seconds = float(runtime_probe.get("cell_seconds", math.nan))
    peak_bytes = int(runtime_probe.get("peak_memory_bytes", -1))
    total_bytes = int(runtime_probe.get("total_memory_bytes", -1))
    peak_fraction = float(runtime_probe.get("peak_memory_fraction", math.nan))
    projected_minutes = float(
        runtime_probe.get("projected_500_cell_minutes", math.nan)
    )
    if (
        not math.isfinite(cell_seconds)
        or cell_seconds < 0
        or peak_bytes < 0
        or total_bytes <= 0
        or not math.isfinite(peak_fraction)
        or not math.isfinite(projected_minutes)
        or peak_fraction < 0
        or peak_fraction > MAX_PEAK_MEMORY_FRACTION
        or projected_minutes < 0
        or projected_minutes > MAX_PROJECTED_MINUTES
        or runtime_probe.get("limits") != {
            "max_peak_memory_fraction": MAX_PEAK_MEMORY_FRACTION,
            "max_projected_minutes": MAX_PROJECTED_MINUTES,
        }
    ):
        raise ValueError("runtime probe limits mismatch")
    if not math.isclose(
        peak_fraction,
        peak_bytes / total_bytes,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        projected_minutes,
        cell_seconds * EXPECTED_DOMAINS * len(EXPECTED_CELLS) / 60,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("runtime probe derived metrics mismatch")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != EXPECTED_DOMAINS:
        raise ValueError("endpoint panel requires exactly 20 complete domains")
    if [domain.get("domain") for domain in domains] != list(EXPECTED_DOMAIN_IDS):
        raise ValueError("domain result identity or order mismatch")

    domain_deltas = []
    physical_cells = 0
    cells_better = 0
    for domain in domains:
        domain_id = domain["domain"]
        preprocessing = domain.get("preprocessing", {})
        if preprocessing.get("canon_symmetric") is not True:
            raise ValueError("endpoint panel requires canonical symmetric atom slots")
        residues_total = int(preprocessing.get("residues_total", -1))
        residues_evaluated = int(preprocessing.get("residues_evaluated", -1))
        if residues_total <= 0 or residues_evaluated != residues_total:
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
            if len(set(expected_starts)) != EXPECTED_STARTS:
                raise ValueError("cell cannot provide three distinct starts")
            if cell.get("starts") != expected_starts:
                raise ValueError("cell start panel mismatch")
            model = _finite_vector(
                cell.get("model_rmsd_by_start"), label="model_rmsd_by_start"
            )
            noop = _finite_vector(
                cell.get("noop_rmsd_by_start"), label="noop_rmsd_by_start"
            )
            recorded = _finite_vector(
                cell.get("model_minus_noop_by_start"),
                label="model_minus_noop_by_start",
            )
            paired = [
                model_value - noop_value
                for model_value, noop_value in zip(model, noop)
            ]
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
                for actual, expected in zip(recorded, paired)
            ):
                raise ValueError("recorded paired RMSD does not match model-minus-noop")
            cell_delta = statistics.fmean(paired)
            if not math.isclose(
                float(cell.get("mean_model_minus_noop", math.nan)),
                cell_delta,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("cell mean paired RMSD mismatch")
            bond_mean = float(cell.get("bond_mean", math.nan))
            bond_max = float(cell.get("bond_max", math.nan))
            if not math.isfinite(bond_mean) or not math.isfinite(bond_max):
                raise ValueError("cell bond metrics must be finite")
            physical = (
                BOND_MEAN_RANGE[0] <= bond_mean <= BOND_MEAN_RANGE[1]
                and bond_max <= MAX_BOND_MAX
            )
            physical_cells += int(physical)
            cells_better += int(cell_delta < 0)
            cell_deltas.append(cell_delta)
        domain_delta = statistics.fmean(cell_deltas)
        summary = domain.get("summary", {})
        if int(summary.get("cells", -1)) != len(EXPECTED_CELLS):
            raise ValueError("domain summary cell count mismatch")
        if not math.isclose(
            float(summary.get("mean_model_minus_noop", math.nan)),
            domain_delta,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("domain mean paired RMSD mismatch")
        if int(summary.get("cells_better_than_noop", -1)) != sum(
            value < 0 for value in cell_deltas
        ):
            raise ValueError("domain cell win count mismatch")
        domain_deltas.append(domain_delta)

    anchor_delta = domain_deltas[0]
    primary_deltas = domain_deltas[1:]
    primary = _t_summary(primary_deltas, T_CRITICAL_18)
    primary_low, primary_high = primary["ci95_model_minus_noop"]
    primary_better = sum(value < 0 for value in primary_deltas)
    primary_worse = sum(value > 0 for value in primary_deltas)
    leave_one_out = []
    for index in range(len(primary_deltas)):
        subset = primary_deltas[:index] + primary_deltas[index + 1:]
        summary = _t_summary(subset, T_CRITICAL_17)
        summary["excluded_domain"] = EXPECTED_DOMAIN_IDS[index + 1]
        summary["passes_negative"] = (
            summary["standard_error"] > ZERO_WIDTH_EPS
            and summary["ci95_model_minus_noop"][1] < 0
        )
        leave_one_out.append(summary)
    t_negative = primary["standard_error"] > ZERO_WIDTH_EPS and primary_high < 0
    sign_negative = primary_better >= MIN_PRIMARY_DOMAINS_BETTER
    t_positive = primary["standard_error"] > ZERO_WIDTH_EPS and primary_low > 0
    sign_positive = primary_worse >= MIN_PRIMARY_DOMAINS_BETTER
    jackknife_negative = all(item["passes_negative"] for item in leave_one_out)
    all_physical = physical_cells == EXPECTED_DOMAINS * len(EXPECTED_CELLS)
    if not all_physical:
        status = "STOP_NONPHYSICAL_DEV20_ENDPOINT"
    elif primary["standard_error"] <= ZERO_WIDTH_EPS:
        status = "INCONCLUSIVE_ZERO_VARIANCE_DEV20_ENDPOINT"
    elif t_negative and sign_negative and jackknife_negative:
        status = "PASS_DEV20_ENDPOINT"
    elif t_positive and sign_positive:
        status = "STOP_DOMAIN_DISADVANTAGE_DEV20_ENDPOINT"
    elif (
        t_negative != sign_negative
        or t_positive != sign_positive
        or (t_negative and not jackknife_negative)
    ):
        status = "INCONCLUSIVE_DEV20_ENDPOINT"
    else:
        status = "STOP_NULL_DEV20_ENDPOINT"

    return {
        "status": status,
        "scope": "20-domain 5x5-cell clean-source H1 endpoint development gate only",
        "checkpoint_sha256": checkpoint_sha256,
        "domain_list_sha256": domain_list_sha256,
        "result_sha256": _sha256(result_path),
        "domains": EXPECTED_DOMAINS,
        "cells": EXPECTED_DOMAINS * len(EXPECTED_CELLS),
        "starts": EXPECTED_DOMAINS * len(EXPECTED_CELLS) * EXPECTED_STARTS,
        "domain_mean_model_minus_noop": domain_deltas,
        "observed_anchor": {
            "domain": EXPECTED_DOMAIN_IDS[0],
            "mean_model_minus_noop": anchor_delta,
            "included_in_primary_inference": False,
        },
        "primary_unseen_domains": list(EXPECTED_DOMAIN_IDS[1:]),
        "primary": primary,
        "primary_domains_better_than_noop": primary_better,
        "leave_one_domain_out": leave_one_out,
        "cells_better_than_noop": cells_better,
        "physical_cells": physical_cells,
        "decision_rule": {
            "PASS_DEV20_ENDPOINT": (
                "all 20 domains and 500 cells are complete and physical; on the 19 previously "
                "unobserved domains the two-sided 95% t interval upper bound for mean(model-noop) "
                "is below zero, at least 14/19 domain means are negative, and the t criterion "
                "survives every leave-one-domain-out subset"
            ),
            "STOP_NONPHYSICAL_DEV20_ENDPOINT": "at least one of 500 cells violates the bond gate",
            "STOP_DOMAIN_DISADVANTAGE_DEV20_ENDPOINT": (
                "on the 19 previously unobserved domains the 95% t interval lower bound is "
                "above zero and at least 14/19 domain means are positive"
            ),
            "INCONCLUSIVE": (
                "all other outcomes, including zero between-domain variance, do not advance"
            ),
        },
        "twenty_domain_completed": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "recursive_evaluation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.checkpoint,
        args.checkpoint_sha256,
        args.domain_list_sha256,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
