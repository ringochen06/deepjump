#!/usr/bin/env python
"""Adjudicate the frozen H1-H6 teacher-forced versus feedback discriminator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


BOND_MEAN_RANGE = (3.2, 4.5)
MAX_BOND_MAX = 5.5
EXPECTED_STEPS = 6


def _finite_series(values: object, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
        raise ValueError(f"{name} must contain steps 0..{EXPECTED_STEPS}")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{name} must be finite")
    return numbers


def _method_series(domain: dict, method: str) -> dict[str, list[float]]:
    methods = domain.get("methods", {})
    if method not in methods:
        raise ValueError(f"missing method: {method}")
    row = methods[method]
    return {
        name: _finite_series(row.get(name), f"{method}.{name}")
        for name in ("rmsd", "bond_mean", "bond_max")
    }


def _physical_steps(series: dict[str, list[float]]) -> list[bool]:
    return [
        BOND_MEAN_RANGE[0] <= mean <= BOND_MEAN_RANGE[1] and maximum <= MAX_BOND_MAX
        for mean, maximum in zip(series["bond_mean"], series["bond_max"])
    ]


def _first_failure(physical: list[bool]) -> int | None:
    return next((step for step in range(1, len(physical)) if not physical[step]), None)


def adjudicate(result_path: str | Path, domain_list_sha256: str) -> dict:
    result = json.loads(Path(result_path).read_text())
    if int(result.get("checkpoint_step", -1)) != 5000:
        raise ValueError("feedback discriminator requires checkpoint step 5000")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("feedback discriminator requires delta=1")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256:
        raise ValueError("domain panel SHA256 mismatch")
    if int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("feedback discriminator requires exactly one domain")
    settings = result.get("settings", {})
    expected = {
        "domains": 1,
        "starts": 5,
        "steps": EXPECTED_STEPS,
        "methods": "mean",
        "teacher_forced_mean": True,
    }
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"feedback settings mismatch: {mismatches}")
    if result.get("preprocessing", {}).get("canon_symmetric") is not True:
        raise ValueError("feedback discriminator requires canonical symmetric atom slots")
    domains = result.get("domains", [])
    if len(domains) != 1:
        raise ValueError("feedback discriminator requires one domain row")

    noop = _method_series(domains[0], "noop")
    autoregressive = _method_series(domains[0], "mean")
    teacher_forced = _method_series(domains[0], "teacher_forced_mean")
    autoregressive_physical = _physical_steps(autoregressive)
    teacher_forced_physical = _physical_steps(teacher_forced)
    teacher_wins = sum(
        teacher_forced["rmsd"][step] < noop["rmsd"][step]
        for step in range(1, EXPECTED_STEPS + 1)
    )
    auto_first_failure = _first_failure(autoregressive_physical)
    teacher_first_failure = _first_failure(teacher_forced_physical)

    feedback_shift = bool(
        all(teacher_forced_physical[1:])
        and teacher_wins >= 4
        and auto_first_failure is not None
        and auto_first_failure <= 4
    )
    endpoint_failure = bool(
        (teacher_first_failure is not None and teacher_first_failure <= 4)
        or teacher_wins <= 1
    )
    if feedback_shift:
        status = "FEEDBACK_DISTRIBUTION_SHIFT"
    elif endpoint_failure:
        status = "ENDPOINT_OPERATOR_FAILURE"
    else:
        status = "INCONCLUSIVE_FEEDBACK_DISCRIMINATOR"

    return {
        "status": status,
        "scope": "same-domain no-training mechanism discriminator",
        "autoregressive_first_nonphysical_step": auto_first_failure,
        "teacher_forced_first_nonphysical_step": teacher_first_failure,
        "teacher_forced_steps_better_than_noop": teacher_wins,
        "decision_rule": {
            "FEEDBACK_DISTRIBUTION_SHIFT": (
                "teacher-forced is physical through H6 and beats no-op at least 4/6, "
                "while autoregressive becomes nonphysical by H4"
            ),
            "ENDPOINT_OPERATOR_FAILURE": (
                "teacher-forced becomes nonphysical by H4 or beats no-op at most 1/6"
            ),
            "INCONCLUSIVE_FEEDBACK_DISCRIMINATOR": "neither preregistered condition is met",
        },
        "formal_training_authorized": False,
        "twenty_domain_authorized": False,
        "confirmation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(args.result, args.domain_list_sha256)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
