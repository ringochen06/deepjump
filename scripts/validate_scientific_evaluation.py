#!/usr/bin/env python
"""Validate the scientific DeepJump development gate against a no-op baseline.

This validator is intentionally separate from the full-grid integration gate.
The integration gate proves that the evaluation pipeline ran on the exact 5x5
grid; this gate additionally requires domain-level statistical improvement and
geometry safety on the frozen development panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.validate_full_grid_evaluation import (
    _load,
    validate_geometry,
    validate_transition,
)


def _validate_identity(
    result: dict[str, Any],
    *,
    expected_delta: int,
    expected_domain_list_sha256: str,
) -> None:
    if result.get("delta_frames") != expected_delta:
        raise ValueError("evaluation delta_frames does not match the frozen contract")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != expected_domain_list_sha256:
        raise ValueError("evaluation domain-list SHA256 does not match the frozen panel")


def _validate_paired_gain(
    value: Any,
    *,
    name: str,
    expected_domains: int,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"missing paired domain statistic: {name}")
    if value.get("domains") != expected_domains:
        raise ValueError(f"{name} does not use the expected number of domains")
    ci95 = value.get("ci95")
    if not isinstance(ci95, list) or len(ci95) != 2:
        raise ValueError(f"{name} is missing its 95% confidence interval")
    if value.get("passes") is not True or ci95[0] <= 0:
        raise ValueError(f"{name} does not have a strictly positive 95% lower bound")
    if value.get("mean_baseline_minus_model", 0) <= 0:
        raise ValueError(f"{name} does not improve the domain-balanced mean")


def validate_scientific_transition(
    result: dict[str, Any],
    *,
    expected_checkpoint: str,
    expected_step: int,
    expected_domains: int,
    expected_delta: int,
    expected_domain_list_sha256: str,
) -> None:
    validate_transition(
        result,
        expected_checkpoint=expected_checkpoint,
        expected_step=expected_step,
        expected_domains=expected_domains,
    )
    _validate_identity(
        result,
        expected_delta=expected_delta,
        expected_domain_list_sha256=expected_domain_list_sha256,
    )
    model = result["summary"].get("mean", {})
    _validate_paired_gain(
        model.get("paired_energy_score_gain"),
        name="conditional energy-score gain",
        expected_domains=expected_domains,
    )
    _validate_paired_gain(
        model.get("paired_msm_row_jsd_gain"),
        name="MSM row-JSD gain",
        expected_domains=expected_domains,
    )


def validate_scientific_geometry(
    result: dict[str, Any],
    *,
    expected_checkpoint: str,
    expected_step: int,
    expected_domains: int,
    expected_delta: int,
    expected_domain_list_sha256: str,
    expected_rollout_steps: int,
) -> None:
    validate_geometry(
        result,
        expected_checkpoint=expected_checkpoint,
        expected_step=expected_step,
        expected_domains=expected_domains,
        expected_rollout_steps=expected_rollout_steps,
    )
    _validate_identity(
        result,
        expected_delta=expected_delta,
        expected_domain_list_sha256=expected_domain_list_sha256,
    )
    model = result["summary"].get("mean", {})
    if model.get("domain_count") != expected_domains:
        raise ValueError("geometry summary does not use the expected number of domains")
    if model.get("domains_all_cells_all_steps_pass") != expected_domains:
        raise ValueError("at least one domain/cell/step violates the geometry envelope")
    if model.get("hard_envelope_pass") is not True or model.get("passes") is not True:
        raise ValueError("geometry scientific envelope does not pass")
    metric_cis = model.get("domain_mean_worst_excess")
    if not isinstance(metric_cis, dict) or not metric_cis:
        raise ValueError("geometry summary is missing domain-level confidence bounds")
    for name, statistic in metric_cis.items():
        if not isinstance(statistic, dict):
            raise ValueError(f"invalid geometry statistic: {name}")
        if statistic.get("domains") != expected_domains:
            raise ValueError(f"geometry statistic {name} has the wrong domain count")
        if statistic.get("passes") is not True or statistic.get("one_sided_upper", 1) > 0:
            raise ValueError(f"geometry statistic {name} exceeds the real-data envelope")


def _write_summary(path: str, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True)
    parser.add_argument("--geometry-20", required=True)
    parser.add_argument("--geometry-100", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-domains", type=int, default=20)
    parser.add_argument("--expected-delta", type=int, required=True)
    parser.add_argument("--expected-domain-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    common = {
        "expected_checkpoint": args.expected_checkpoint,
        "expected_step": args.expected_step,
        "expected_domains": args.expected_domains,
        "expected_delta": args.expected_delta,
        "expected_domain_list_sha256": args.expected_domain_list_sha256,
    }
    summary: dict[str, Any] = {
        "status": "FAIL",
        "scope": "development_scientific_gate",
        "checkpoint": args.expected_checkpoint,
        "checkpoint_step": args.expected_step,
        "delta_frames": args.expected_delta,
        "domain_list_sha256": args.expected_domain_list_sha256,
        "evaluated_domains": args.expected_domains,
        "cells_per_domain": 25,
        "requirements": {
            "conditional_energy_paired_ci95_lower_gt_zero": True,
            "msm_row_jsd_paired_ci95_lower_gt_zero": True,
            "geometry_h20_envelope": True,
            "geometry_h100_envelope": True,
        },
        "artifacts": {
            "transition": args.transition,
            "geometry_20": args.geometry_20,
            "geometry_100": args.geometry_100,
        },
        "errors": [],
    }
    try:
        transition = _load(args.transition)
        geometry_20 = _load(args.geometry_20)
        geometry_100 = _load(args.geometry_100)
        validate_scientific_transition(transition, **common)
        validate_scientific_geometry(
            geometry_20, **common, expected_rollout_steps=20
        )
        validate_scientific_geometry(
            geometry_100, **common, expected_rollout_steps=100
        )
    except Exception as exc:  # noqa: BLE001 - emit a durable fail-closed report
        summary["errors"] = [str(exc)]
    else:
        summary["status"] = "PASS"
    _write_summary(args.output, summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        print(f"ERROR: {summary['errors'][0]}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
