#!/usr/bin/env python
"""Fail-closed adjudication for the same-checkpoint H1 ODE step-count scan."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Mapping

from scripts.adjudicate_source_law_candidate import (
    BOND_MEAN_RANGE,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_STARTS,
    MAX_BOND_MAX,
    _sha256,
    _verify_checkpoint_source_law,
)


EXPECTED_METHODS = (
    "mean",
    "ode_1",
    "ode_2",
    "ode_5",
    "ode_10",
    "ode_20",
    "ode_40",
    "ode_75",
    "ode_150",
)
ODE_METHODS = tuple(name for name in EXPECTED_METHODS if name.startswith("ode_"))


def _validate_method(method: dict, *, label: str) -> None:
    for name in ("rmsd_by_start", "bond_mean", "bond_max"):
        values = method.get(name)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{label}.{name} must contain H0 and H1")
        if name == "rmsd_by_start" and any(
            not isinstance(horizon, list) or len(horizon) != EXPECTED_STARTS
            for horizon in values
        ):
            raise ValueError(f"{label}.rmsd_by_start must contain five starts")
        flat = (
            [value for horizon in values for value in horizon]
            if name == "rmsd_by_start"
            else values
        )
        if not all(math.isfinite(float(value)) for value in flat):
            raise ValueError(f"{label}.{name} must be finite")


def _load_result(
    path: str | Path,
    *,
    method: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> tuple[dict, dict]:
    result = json.loads(Path(path).read_text())
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("ODE step scan requires checkpoint step 1000")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("ODE step scan requires delta=1")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256 or int(
        panel.get("evaluated_count", -1)
    ) != 1:
        raise ValueError("domain panel identity mismatch")
    expected = {
        "domains": 1,
        "starts": EXPECTED_STARTS,
        "steps": 1,
        "methods": method,
        "seed": 20260718,
        "noise_sigma": None,
        "integrator": "euler",
        "tau_max": 1.0,
        "terminal_denoise": False,
        "drift_anchor": "state",
        "project_v_atom_mask": False,
        "teacher_forced_mean": False,
    }
    settings = result.get("settings", {})
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"ODE step scan settings mismatch: {mismatches}")
    if result.get("preprocessing", {}).get("canon_symmetric") is not True:
        raise ValueError("ODE step scan requires canonical symmetric atom slots")
    domains = result.get("domains", [])
    if len(domains) != 1:
        raise ValueError("ODE step scan requires one domain row")
    methods = domains[0].get("methods", {})
    if set(methods) != {"noop", "one_step_persistence", method}:
        raise ValueError(f"unexpected method set for {method}")
    for name, payload in methods.items():
        _validate_method(payload, label=f"{method}.{name}")
    return result, methods


def _common_settings(result: dict) -> dict:
    return {
        key: value
        for key, value in result["settings"].items()
        if key not in {"methods", "output"}
    }


def _verify_identity(
    baseline_result: dict,
    baseline_methods: dict,
    result: dict,
    methods: dict,
    *,
    method: str,
) -> None:
    if _common_settings(result) != _common_settings(baseline_result):
        raise ValueError(f"non-method settings mismatch for {method}")
    if result["domain_panel"] != baseline_result["domain_panel"]:
        raise ValueError(f"domain panel payload mismatch for {method}")
    if result.get("preprocessing") != baseline_result.get("preprocessing"):
        raise ValueError(f"preprocessing mismatch for {method}")
    domain = result["domains"][0]
    baseline_domain = baseline_result["domains"][0]
    for key in ("domain", "residues_total", "residues_evaluated", "frames", "starts"):
        if domain.get(key) != baseline_domain.get(key):
            raise ValueError(f"domain identity mismatch for {method}: {key}")
    for name in ("noop", "one_step_persistence"):
        if methods[name] != baseline_methods[name]:
            raise ValueError(f"{name} baseline mismatch for {method}")
    baseline_model = baseline_methods[baseline_result["settings"]["methods"]]
    model = methods[method]
    for name in ("rmsd_by_start", "bond_mean", "bond_max"):
        if model[name][0] != baseline_model[name][0]:
            raise ValueError(f"H0 model identity mismatch for {method}: {name}")


def _paired_metrics(model: dict, noop: dict) -> dict:
    model_final = [float(value) for value in model["rmsd_by_start"][-1]]
    noop_final = [float(value) for value in noop["rmsd_by_start"][-1]]
    deltas = [a - b for a, b in zip(model_final, noop_final)]
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(EXPECTED_STARTS)
    physical = (
        BOND_MEAN_RANGE[0] <= float(model["bond_mean"][-1]) <= BOND_MEAN_RANGE[1]
        and float(model["bond_max"][-1]) <= MAX_BOND_MAX
    )
    robust_advantage = (
        mean_delta < 0
        and standard_error > 0
        and abs(mean_delta) >= 2 * standard_error
    )
    return {
        "model_rmsd_by_start": model_final,
        "noop_rmsd_by_start": noop_final,
        "model_minus_noop_by_start": deltas,
        "mean_model_minus_noop": mean_delta,
        "standard_error": standard_error,
        "absolute_mean_over_standard_error": (
            abs(mean_delta) / standard_error if standard_error > 0 else None
        ),
        "bond_mean": float(model["bond_mean"][-1]),
        "bond_max": float(model["bond_max"][-1]),
        "physical": physical,
        "robust_advantage": robust_advantage,
        "passes": physical and robust_advantage,
    }


def adjudicate(
    result_paths: Mapping[str, str | Path],
    reference_ode150_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> dict:
    if set(result_paths) != set(EXPECTED_METHODS):
        raise ValueError("ODE step scan requires exactly the preregistered methods")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    _verify_checkpoint_source_law(checkpoint_path)

    loaded: dict[str, tuple[dict, dict]] = {}
    for method in EXPECTED_METHODS:
        loaded[method] = _load_result(
            result_paths[method],
            method=method,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            domain_list_sha256=domain_list_sha256,
        )

    baseline_result, baseline_methods = loaded["ode_1"]
    for method in EXPECTED_METHODS:
        result, methods = loaded[method]
        _verify_identity(
            baseline_result,
            baseline_methods,
            result,
            methods,
            method=method,
        )

    reference_result, reference_methods = _load_result(
        reference_ode150_path,
        method="ode_150",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        domain_list_sha256=domain_list_sha256,
    )
    _verify_identity(
        baseline_result,
        baseline_methods,
        reference_result,
        reference_methods,
        method="ode_150",
    )
    current_ode150 = loaded["ode_150"][1]
    if current_ode150["ode_150"] != reference_methods["ode_150"]:
        raise ValueError("ode_150 did not reproduce the frozen H1 method payload")

    metrics = {
        method: _paired_metrics(methods[method], methods["noop"])
        for method, (_, methods) in loaded.items()
    }
    if not metrics["ode_1"]["passes"]:
        status = "STOP_SOURCE_ENDPOINT_H1"
        first_failed = "ode_1"
    else:
        first_failed = next(
            (method for method in ODE_METHODS[1:] if not metrics[method]["passes"]),
            None,
        )
        status = "STOP_INTERNAL_ODE_FEEDBACK" if first_failed else "PASS_ODE_STEP_SCAN"

    return {
        "status": status,
        "scope": "single-domain same-checkpoint H1 ODE step-count discriminator only",
        "checkpoint_sha256": checkpoint_sha256,
        "domain_list_sha256": domain_list_sha256,
        "reference_ode150_sha256": _sha256(reference_ode150_path),
        "result_sha256": {
            method: _sha256(result_paths[method]) for method in EXPECTED_METHODS
        },
        "ode150_reproduced": True,
        "first_failed_ode_method": first_failed,
        "metrics": metrics,
        "decision_rule": {
            "STOP_SOURCE_ENDPOINT_H1": (
                "ode_1 is nonphysical or does not beat no-op by at least 2SE; "
                "multi-substep feedback is not required for failure"
            ),
            "STOP_INTERNAL_ODE_FEEDBACK": (
                "ode_1 passes but the first later preregistered ODE step count fails"
            ),
            "PASS_ODE_STEP_SCAN": "every preregistered ODE step count passes",
        },
        "twenty_domain_authorized": False,
        "second_seed_authorized": False,
        "confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Preregistered METHOD=PATH result; provide each method exactly once.",
    )
    parser.add_argument("--reference-ode150", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = {}
    for item in args.result:
        method, separator, path = item.partition("=")
        if not separator or method in results:
            parser.error(f"invalid or duplicate --result: {item}")
        results[method] = path
    report = adjudicate(
        results,
        args.reference_ode150,
        args.checkpoint,
        args.checkpoint_sha256,
        args.domain_list_sha256,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
