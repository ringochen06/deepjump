"""Fail-closed paired adjudication for the masked-V sampler discriminator."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.adjudicate_source_law_candidate import (
    BOND_MEAN_RANGE,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_METHOD,
    EXPECTED_STARTS,
    MAX_BOND_MAX,
    _sha256,
    _verify_checkpoint_source_law,
)


def _load_result(
    path: str | Path,
    *,
    steps: int,
    projected: bool,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> tuple[dict, dict]:
    result = json.loads(Path(path).read_text())
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("V-mask discriminator requires checkpoint step 1000")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("V-mask discriminator requires delta=1")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256 or int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("domain panel identity mismatch")
    settings = result.get("settings", {})
    expected = {
        "domains": 1,
        "starts": EXPECTED_STARTS,
        "steps": steps,
        "methods": EXPECTED_METHOD,
        "seed": 20260718,
        "noise_sigma": None,
        "integrator": "euler",
        "tau_max": 1.0,
        "terminal_denoise": False,
        "drift_anchor": "state",
        "teacher_forced_mean": False,
        "project_v_atom_mask": projected,
    }
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"V-mask settings mismatch: {mismatches}")
    if result.get("preprocessing", {}).get("canon_symmetric") is not True:
        raise ValueError("V-mask discriminator requires canonical symmetric atom slots")
    domains = result.get("domains", [])
    if len(domains) != 1:
        raise ValueError("V-mask discriminator requires one domain row")
    methods = domains[0].get("methods", {})
    if set(methods) != {"noop", "one_step_persistence", EXPECTED_METHOD}:
        raise ValueError("unexpected method set")
    return result, methods


def _validate_method(method: dict, *, steps: int, label: str) -> None:
    for name in ("rmsd_by_start", "bond_mean", "bond_max"):
        values = method.get(name)
        if not isinstance(values, list) or len(values) != steps + 1:
            raise ValueError(f"{label}.{name} must contain H0..H{steps}")
        if name == "rmsd_by_start" and any(
            not isinstance(horizon, list) or len(horizon) != EXPECTED_STARTS
            for horizon in values
        ):
            raise ValueError(f"{label}.rmsd_by_start must contain five starts per horizon")
        flat = (
            [value for horizon in values for value in horizon]
            if name == "rmsd_by_start"
            else values
        )
        if not all(math.isfinite(float(value)) for value in flat):
            raise ValueError(f"{label}.{name} must be finite")


def _paired_metrics(masked: list[float], baseline: list[float]) -> dict:
    deltas = [float(a) - float(b) for a, b in zip(masked, baseline)]
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(EXPECTED_STARTS)
    robust = mean_delta < 0 and standard_error > 0 and abs(mean_delta) >= 2 * standard_error
    return {
        "deltas": deltas,
        "mean": mean_delta,
        "standard_error": standard_error,
        "absolute_mean_over_standard_error": (
            abs(mean_delta) / standard_error if standard_error > 0 else None
        ),
        "robust_advantage": robust,
    }


def _same_identity(current: dict, masked: dict, current_methods: dict, masked_methods: dict) -> None:
    current_settings = {k: v for k, v in current["settings"].items() if k not in {"output", "project_v_atom_mask"}}
    masked_settings = {k: v for k, v in masked["settings"].items() if k not in {"output", "project_v_atom_mask"}}
    if current_settings != masked_settings:
        raise ValueError("non-projection settings mismatch")
    if current["domain_panel"] != masked["domain_panel"]:
        raise ValueError("domain panel payload mismatch")
    current_domain, masked_domain = current["domains"][0], masked["domains"][0]
    for key in ("domain", "residues_total", "residues_evaluated", "frames", "starts"):
        if current_domain.get(key) != masked_domain.get(key):
            raise ValueError(f"domain identity mismatch: {key}")
    for baseline in ("noop", "one_step_persistence"):
        if current_methods[baseline] != masked_methods[baseline]:
            raise ValueError(f"{baseline} baseline mismatch")
    if current_methods[EXPECTED_METHOD]["rmsd_by_start"][0] != masked_methods[EXPECTED_METHOD]["rmsd_by_start"][0]:
        raise ValueError("H0 model identity mismatch")


def adjudicate(
    current_path: str | Path,
    masked_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
    *,
    steps: int,
    h1_decision_path: str | Path | None = None,
) -> dict:
    if steps not in {1, 6}:
        raise ValueError("V-mask discriminator steps must be 1 or 6")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    _verify_checkpoint_source_law(checkpoint_path)
    if steps == 1 and h1_decision_path is not None:
        raise ValueError("H1 adjudication must not receive an H1 decision")
    if steps == 6:
        if h1_decision_path is None:
            raise ValueError("H6 adjudication requires the H1 decision")
        h1 = json.loads(Path(h1_decision_path).read_text())
        if h1.get("status") != "ADVANCE_MASKED_H6":
            raise ValueError("H6 requires ADVANCE_MASKED_H6")
        if h1.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("H1 checkpoint identity mismatch")
        if h1.get("domain_list_sha256") != domain_list_sha256:
            raise ValueError("H1 domain list identity mismatch")
        if h1.get("steps") != 1:
            raise ValueError("H1 decision horizon mismatch")
        for key in (
            "twenty_domain_authorized",
            "second_seed_authorized",
            "confirmation_authorized",
            "formal_training_authorized",
        ):
            if h1.get(key) is not False:
                raise ValueError(f"H1 decision must keep {key}=false")

    current, current_methods = _load_result(
        current_path,
        steps=steps,
        projected=False,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        domain_list_sha256=domain_list_sha256,
    )
    masked, masked_methods = _load_result(
        masked_path,
        steps=steps,
        projected=True,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        domain_list_sha256=domain_list_sha256,
    )
    _same_identity(current, masked, current_methods, masked_methods)
    for prefix, methods in (("current", current_methods), ("masked", masked_methods)):
        for name, method in methods.items():
            _validate_method(method, steps=steps, label=f"{prefix}.{name}")

    current_final = current_methods[EXPECTED_METHOD]["rmsd_by_start"][-1]
    masked_final = masked_methods[EXPECTED_METHOD]["rmsd_by_start"][-1]
    noop_final = masked_methods["noop"]["rmsd_by_start"][-1]
    masked_vs_current = _paired_metrics(masked_final, current_final)
    masked_vs_noop = _paired_metrics(masked_final, noop_final)
    candidate = masked_methods[EXPECTED_METHOD]
    physical = all(
        BOND_MEAN_RANGE[0] <= float(mean) <= BOND_MEAN_RANGE[1]
        and float(maximum) <= MAX_BOND_MAX
        for mean, maximum in zip(candidate["bond_mean"][1:], candidate["bond_max"][1:])
    )
    passed = (
        physical
        and masked_vs_current["robust_advantage"]
        and masked_vs_noop["robust_advantage"]
    )
    status = (
        "ADVANCE_MASKED_H6" if steps == 1 and passed
        else "STOP_MASKED_H1" if steps == 1
        else "PASS_MASKED_H6" if passed
        else "STOP_MASKED_H6"
    )
    return {
        "status": status,
        "scope": "single-domain same-checkpoint masked-V inference discriminator only",
        "steps": steps,
        "checkpoint_sha256": checkpoint_sha256,
        "domain_list_sha256": domain_list_sha256,
        "h1_decision_sha256": (
            _sha256(h1_decision_path) if h1_decision_path is not None else None
        ),
        "current_result_sha256": _sha256(current_path),
        "masked_result_sha256": _sha256(masked_path),
        "physical_through_horizon": physical,
        "masked_vs_current": masked_vs_current,
        "masked_vs_noop": masked_vs_noop,
        "twenty_domain_authorized": False,
        "second_seed_authorized": False,
        "confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--masked", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--steps", type=int, choices=(1, 6), required=True)
    parser.add_argument("--h1-decision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.current,
        args.masked,
        args.checkpoint,
        args.checkpoint_sha256,
        args.domain_list_sha256,
        steps=args.steps,
        h1_decision_path=args.h1_decision,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
