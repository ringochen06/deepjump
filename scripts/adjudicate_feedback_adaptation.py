#!/usr/bin/env python
"""Adjudicate the bounded same-domain feedback-adaptation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


BOND_MEAN_RANGE = (3.2, 4.5)
MAX_BOND_MAX = 5.5
EXPECTED_STEPS = 6
EXPECTED_CHECKPOINT_STEP = 250


def _finite_series(values: object, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
        raise ValueError(f"{name} must contain steps 0..{EXPECTED_STEPS}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result


def _method(domain: dict, name: str) -> dict[str, list[float]]:
    try:
        row = domain["methods"][name]
    except KeyError as exc:
        raise ValueError(f"missing method: {name}") from exc
    return {
        metric: _finite_series(row.get(metric), f"{name}.{metric}")
        for metric in ("rmsd", "bond_mean", "bond_max")
    }


def _physical(row: dict[str, list[float]]) -> list[bool]:
    return [
        BOND_MEAN_RANGE[0] <= mean <= BOND_MEAN_RANGE[1] and maximum <= MAX_BOND_MAX
        for mean, maximum in zip(row["bond_mean"], row["bond_max"])
    ]


def _first_failure(values: list[bool]) -> int | None:
    return next((step for step in range(1, len(values)) if not values[step]), None)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjudicate(
    result_path: str | Path,
    domain_list_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> dict:
    result = json.loads(Path(result_path).read_text())
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("feedback adaptation requires checkpoint step 250")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("feedback adaptation requires delta=1")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256 or int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("domain panel identity mismatch")
    settings = result.get("settings", {})
    expected = {
        "domains": 1,
        "starts": 5,
        "steps": EXPECTED_STEPS,
        "methods": "mean",
        "teacher_forced_mean": True,
        "integrator": "euler",
        "tau_max": 1.0,
        "drift_anchor": "state",
    }
    mismatches = {
        key: (settings.get(key), expected_value)
        for key, expected_value in expected.items()
        if settings.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"feedback adaptation settings mismatch: {mismatches}")
    if result.get("preprocessing", {}).get("canon_symmetric") is not True:
        raise ValueError("feedback adaptation requires canonical symmetric atom slots")
    domains = result.get("domains", [])
    if len(domains) != 1:
        raise ValueError("feedback adaptation requires one domain row")

    noop = _method(domains[0], "noop")
    persistence = _method(domains[0], "one_step_persistence")
    autoregressive = _method(domains[0], "mean")
    teacher = _method(domains[0], "teacher_forced_mean")
    auto_failure = _first_failure(_physical(autoregressive))
    teacher_failure = _first_failure(_physical(teacher))
    teacher_wins = sum(
        teacher["rmsd"][step] < persistence["rmsd"][step]
        for step in range(1, EXPECTED_STEPS + 1)
    )
    auto_beats_noop_final = autoregressive["rmsd"][-1] < noop["rmsd"][-1]

    passed = (
        auto_failure is None
        and auto_beats_noop_final
        and teacher_failure is None
        and teacher_wins >= 4
    )
    failed = (
        (auto_failure is not None and auto_failure <= 4)
        or teacher_failure is not None
        or teacher_wins <= 2
        or not auto_beats_noop_final
    )
    status = (
        "PASS_FEEDBACK_ADAPTATION"
        if passed
        else "FAIL_FEEDBACK_ADAPTATION"
        if failed
        else "INCONCLUSIVE_FEEDBACK_ADAPTATION"
    )
    return {
        "status": status,
        "scope": "single-domain 250-step warm-start feedback adaptation only",
        "autoregressive_first_nonphysical_step": auto_failure,
        "teacher_forced_first_nonphysical_step": teacher_failure,
        "teacher_forced_steps_better_than_one_step_persistence": teacher_wins,
        "autoregressive_final_rmsd": autoregressive["rmsd"][-1],
        "noop_final_rmsd": noop["rmsd"][-1],
        "autoregressive_beats_noop_final": auto_beats_noop_final,
        "decision_rule": {
            "PASS": "autoregressive is physical through H6 and beats no-op at H6; teacher-forced is physical and beats one-step persistence at least 4/6",
            "FAIL": "autoregressive fails by H4, teacher-forced is nonphysical, teacher-forced beats persistence at most 2/6, or autoregressive does not beat no-op at H6",
            "INCONCLUSIVE": "all remaining outcomes, including first autoregressive failure at H5",
        },
        "formal_training_authorized": False,
        "twenty_domain_authorized": False,
        "confirmation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.domain_list_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
