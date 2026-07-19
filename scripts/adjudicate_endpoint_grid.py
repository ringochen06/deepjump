#!/usr/bin/env python
"""Fail-closed adjudication for the single-domain 5x5 clean endpoint grid."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from deepjump.evaluation import MDCATH_REPLICAS, MDCATH_TEMPERATURES
from scripts.adjudicate_source_law_candidate import (
    BOND_MEAN_RANGE,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_STARTS,
    MAX_BOND_MAX,
    _sha256,
    _verify_checkpoint_source_law,
)


EXPECTED_CELLS = {
    (temperature, replica)
    for temperature in MDCATH_TEMPERATURES
    for replica in MDCATH_REPLICAS
}


def _finite_vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != EXPECTED_STARTS:
        raise ValueError(f"{label} must contain exactly five starts")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def _adjudicate_grid(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
    *,
    expected_domain_id: str,
    expected_residue_count: int | None,
    pass_status: str,
    null_status: str,
    nonphysical_status: str,
    scope: str,
) -> dict:
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    _verify_checkpoint_source_law(checkpoint_path)
    result = json.loads(Path(result_path).read_text())
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("result checkpoint SHA256 mismatch")
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("endpoint grid requires checkpoint step 1000")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("endpoint grid requires delta=1")
    if result.get("settings") != {
        "starts": EXPECTED_STARTS,
        "method": "mean",
        "source_noise": False,
    }:
        raise ValueError("endpoint grid settings mismatch")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256:
        raise ValueError("domain panel SHA256 mismatch")
    domain_ids = panel.get("ids")
    if domain_ids != [expected_domain_id]:
        raise ValueError("endpoint grid domain identity mismatch")
    if result.get("grid") != {
        "temperatures": list(MDCATH_TEMPERATURES),
        "replicas": list(MDCATH_REPLICAS),
    }:
        raise ValueError("endpoint grid must be the canonical 5x5 grid")
    preprocessing = result.get("preprocessing", {})
    if preprocessing.get("canon_symmetric") is not True:
        raise ValueError("endpoint grid requires canonical symmetric atom slots")
    residues_total = int(preprocessing.get("residues_total", -1))
    residues_evaluated = int(preprocessing.get("residues_evaluated", -1))
    if expected_residue_count is None:
        if residues_total <= 0 or residues_evaluated != residues_total:
            raise ValueError("endpoint grid requires all residues")
    elif (
        residues_total != expected_residue_count
        or residues_evaluated != expected_residue_count
    ):
        raise ValueError(
            f"endpoint grid requires all {expected_residue_count} residues"
        )

    cells = result.get("cells")
    if not isinstance(cells, list) or len(cells) != len(EXPECTED_CELLS):
        raise ValueError("endpoint grid requires exactly 25 cells")
    identities = {
        (int(cell.get("temperature", -1)), int(cell.get("replica", -1)))
        for cell in cells
    }
    if identities != EXPECTED_CELLS:
        raise ValueError("endpoint grid has missing, duplicate, or extra cells")

    cell_deltas = []
    physical_cells = 0
    starts_better = 0
    for cell in cells:
        if cell.get("domain") != domain_ids[0]:
            raise ValueError("cell domain mismatch")
        frames = int(cell.get("frames", -1))
        if frames <= 1:
            raise ValueError("cell frame count is invalid")
        expected_starts = np.linspace(0, frames - 2, EXPECTED_STARTS, dtype=int).tolist()
        if len(set(expected_starts)) != EXPECTED_STARTS:
            raise ValueError("cell cannot provide five distinct starts")
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
        paired = [model_value - noop_value for model_value, noop_value in zip(model, noop)]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            for actual, expected in zip(recorded, paired)
        ):
            raise ValueError("recorded paired RMSD does not match model-minus-noop")
        mean_delta = statistics.fmean(paired)
        if not math.isclose(
            float(cell.get("mean_model_minus_noop", math.nan)),
            mean_delta,
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
        starts_better += sum(delta < 0 for delta in paired)
        cell_deltas.append(mean_delta)

    mean_delta = statistics.fmean(cell_deltas)
    standard_error = statistics.stdev(cell_deltas) / math.sqrt(len(cell_deltas))
    robust_advantage = (
        mean_delta < 0
        and standard_error > 0
        and abs(mean_delta) >= 2 * standard_error
    )
    physical = physical_cells == len(EXPECTED_CELLS)
    if not physical:
        status = nonphysical_status
    elif robust_advantage:
        status = pass_status
    else:
        status = null_status

    return {
        "status": status,
        "scope": scope,
        "checkpoint_sha256": checkpoint_sha256,
        "domain_list_sha256": domain_list_sha256,
        "result_sha256": _sha256(result_path),
        "cells": len(cell_deltas),
        "starts": len(cell_deltas) * EXPECTED_STARTS,
        "cell_mean_model_minus_noop": cell_deltas,
        "mean_model_minus_noop": mean_delta,
        "standard_error": standard_error,
        "absolute_mean_over_standard_error": (
            abs(mean_delta) / standard_error if standard_error > 0 else None
        ),
        "cells_better_than_noop": sum(delta < 0 for delta in cell_deltas),
        "starts_better_than_noop": starts_better,
        "physical_cells": physical_cells,
        "robust_endpoint_advantage": robust_advantage,
        "decision_rule": {
            pass_status: (
                "all 25 cells are physical and cell-balanced mean(model-noop)<0 "
                "with |mean|>=2SE"
            ),
            null_status: (
                "the physical clean endpoint does not beat no-op by the preregistered 2SE rule"
            ),
            nonphysical_status: "at least one cell violates the frozen bond gate",
        },
        "twenty_domain_authorized": False,
        "second_seed_authorized": False,
        "confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> dict:
    return _adjudicate_grid(
        result_path,
        checkpoint_path,
        checkpoint_sha256,
        domain_list_sha256,
        expected_domain_id="1a0hA01",
        expected_residue_count=89,
        pass_status="PASS_CLEAN_ENDPOINT_GRID",
        null_status="STOP_NULL_ENDPOINT_GRID",
        nonphysical_status="STOP_NONPHYSICAL_ENDPOINT_GRID",
        scope="single-domain 5x5-cell clean-source H1 endpoint discriminator only",
    )


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
