#!/usr/bin/env python
"""Validate strict 5x5 model-level evaluation integration artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TEMPERATURES = [320, 348, 379, 413, 450]
REPLICAS = [0, 1, 2, 3, 4]
EXPECTED_CELLS = {(temperature, replica) for temperature in TEMPERATURES for replica in REPLICAS}


def _require_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite numeric value at {path}: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _validate_common(
    result: dict[str, Any],
    *,
    expected_checkpoint: str,
    expected_step: int,
    expected_domains: int,
) -> None:
    _require_finite(result)
    if result.get("checkpoint") != expected_checkpoint:
        raise ValueError("evaluation checkpoint path does not match the reviewed checkpoint")
    if result.get("checkpoint_step") != expected_step:
        raise ValueError("evaluation checkpoint step does not match the expected step")
    panel = result.get("domain_panel", {})
    if panel.get("evaluated_count") != expected_domains:
        raise ValueError("evaluated domain count does not match the integration contract")
    grid = result.get("trajectory_grid", {})
    if grid.get("formal_full_grid") is not True:
        raise ValueError("evaluation is not marked as a formal full grid")
    if grid.get("temperatures") != TEMPERATURES or grid.get("replicas") != REPLICAS:
        raise ValueError("evaluation does not use the frozen 5x5 temperature/replica grid")
    if grid.get("required_cells_per_domain") != 25:
        raise ValueError("evaluation does not require exactly 25 cells per domain")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != expected_domains:
        raise ValueError("domain result count does not match the integration contract")
    for domain in domains:
        if domain.get("grid", {}).get("cells") != 25:
            raise ValueError(f"domain {domain.get('domain')} does not contain 25 cells")
        trajectories = domain.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != 25:
            raise ValueError(f"domain {domain.get('domain')} does not expose 25 trajectories")
        cells = {(cell.get("temperature"), cell.get("replica")) for cell in trajectories}
        if cells != EXPECTED_CELLS:
            raise ValueError(f"domain {domain.get('domain')} has an incomplete or duplicate grid")


def validate_transition(
    result: dict[str, Any],
    *,
    expected_checkpoint: str,
    expected_step: int,
    expected_domains: int,
) -> None:
    _validate_common(
        result,
        expected_checkpoint=expected_checkpoint,
        expected_step=expected_step,
        expected_domains=expected_domains,
    )
    for domain in result["domains"]:
        for cell in domain["trajectories"]:
            if cell.get("reference_replica") == cell.get("replica"):
                raise ValueError("transition cell reuses its evaluation replica for fitting")
    summary = result.get("summary", {})
    if not {"noop", "mean"}.issubset(summary):
        raise ValueError("transition summary is missing noop or mean")


def validate_geometry(
    result: dict[str, Any],
    *,
    expected_checkpoint: str,
    expected_step: int,
    expected_domains: int,
    expected_rollout_steps: int,
) -> None:
    _validate_common(
        result,
        expected_checkpoint=expected_checkpoint,
        expected_step=expected_step,
        expected_domains=expected_domains,
    )
    if result.get("settings", {}).get("steps") != expected_rollout_steps:
        raise ValueError("geometry rollout length does not match the requested gate")
    summary = result.get("summary", {})
    if not {"noop", "mean"}.issubset(summary):
        raise ValueError("geometry summary is missing noop or mean")


def _load(path: str) -> dict[str, Any]:
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", required=True)
    parser.add_argument("--geometry-20", required=True)
    parser.add_argument("--geometry-100", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-domains", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validate_transition(
        _load(args.transition),
        expected_checkpoint=args.expected_checkpoint,
        expected_step=args.expected_step,
        expected_domains=args.expected_domains,
    )
    validate_geometry(
        _load(args.geometry_20),
        expected_checkpoint=args.expected_checkpoint,
        expected_step=args.expected_step,
        expected_domains=args.expected_domains,
        expected_rollout_steps=20,
    )
    validate_geometry(
        _load(args.geometry_100),
        expected_checkpoint=args.expected_checkpoint,
        expected_step=args.expected_step,
        expected_domains=args.expected_domains,
        expected_rollout_steps=100,
    )
    summary = {
        "status": "PASS",
        "checkpoint": args.expected_checkpoint,
        "checkpoint_step": args.expected_step,
        "evaluated_domains": args.expected_domains,
        "cells_per_domain": 25,
        "artifacts": {
            "transition": args.transition,
            "geometry_20": args.geometry_20,
            "geometry_100": args.geometry_100,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
