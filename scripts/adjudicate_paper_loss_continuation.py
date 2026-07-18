#!/usr/bin/env python
"""Adjudicate the preregistered bounded step-1000 to step-2000 continuation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_STEPS = list(range(1100, 2001, 100))
METHODS = ("mean", "ode_1")
MIN_BOND_MEAN = 3.2
MAX_BOND_MEAN = 4.5
MAX_BOND_MAX = 5.5


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _geometry_excess(bond_mean: float, bond_max: float) -> float:
    return max(
        MIN_BOND_MEAN - bond_mean,
        bond_mean - MAX_BOND_MEAN,
        bond_max - MAX_BOND_MAX,
        0.0,
    )


def _load_history(path: Path) -> tuple[list[dict], bool]:
    history = json.loads(path.read_text())
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    steps = [row.get("step") for row in history]
    if steps != EXPECTED_STEPS:
        raise ValueError(f"history steps {steps} != {EXPECTED_STEPS}")
    losses = [
        _finite_number(row.get("val_loss"), f"history step {step} val_loss")
        for step, row in zip(EXPECTED_STEPS, history)
    ]
    # A 0.5% end-to-start decrease is treated as a real downward trend rather
    # than validation-panel noise. It classifies failure; it is not a PASS.
    materially_falling = losses[-1] <= 0.995 * losses[0]
    return history, materially_falling


def _load_rollouts(directory: Path, domain_sha256: str) -> dict[str, list[dict]]:
    traces = {method: [] for method in METHODS}
    for step in EXPECTED_STEPS:
        path = directory / f"rollout_{step}.json"
        result = json.loads(path.read_text())
        if result.get("checkpoint_step") != step:
            raise ValueError(f"{path}: checkpoint_step mismatch")
        if result.get("delta_frames") != 1:
            raise ValueError(f"{path}: delta_frames must be 1")
        panel = result.get("domain_panel", {})
        if panel.get("sha256") != domain_sha256 or panel.get("evaluated_count") != 3:
            raise ValueError(f"{path}: frozen domain panel mismatch")
        settings = result.get("settings", {})
        expected_settings = {
            "domains": 3,
            "starts": 2,
            "steps": 20,
            "methods": "mean,ode_1",
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
        }
        for name, expected in expected_settings.items():
            if settings.get(name) != expected:
                raise ValueError(
                    f"{path}: setting {name}={settings.get(name)!r} != {expected!r}"
                )

        summary = result.get("summary", {})
        if set(summary) != {"noop", *METHODS}:
            raise ValueError(f"{path}: unexpected summary methods")
        noop_rmsd = _finite_number(
            summary["noop"].get("mean_final_rmsd"), f"{path} noop RMSD"
        )
        for method in METHODS:
            metrics = summary[method]
            rmsd = _finite_number(
                metrics.get("mean_final_rmsd"), f"{path} {method} RMSD"
            )
            bond_mean = _finite_number(
                metrics.get("mean_final_bond_mean"), f"{path} {method} bond mean"
            )
            bond_max = _finite_number(
                metrics.get("mean_final_bond_max"), f"{path} {method} bond max"
            )
            if metrics.get("finite") is not True:
                raise ValueError(f"{path}: {method} summary is not finite")
            traces[method].append(
                {
                    "step": step,
                    "rmsd": rmsd,
                    "noop_rmsd": noop_rmsd,
                    "bond_mean": bond_mean,
                    "bond_max": bond_max,
                    "geometry_excess": _geometry_excess(bond_mean, bond_max),
                    "passes": (
                        rmsd < noop_rmsd
                        and MIN_BOND_MEAN <= bond_mean <= MAX_BOND_MEAN
                        and bond_max <= MAX_BOND_MAX
                    ),
                }
            )
    return traces


def adjudicate(history_path: Path, rollout_dir: Path, domain_sha256: str) -> dict:
    history, val_loss_materially_falling = _load_history(history_path)
    traces = _load_rollouts(rollout_dir, domain_sha256)
    eligible = []
    for method, rows in traces.items():
        last_three = rows[-3:]
        favorable_last_three = (
            last_three[-1]["rmsd"] < last_three[0]["rmsd"]
            and last_three[-1]["geometry_excess"]
            <= last_three[0]["geometry_excess"]
        )
        final_two_pass = rows[-2]["passes"] and rows[-1]["passes"]
        if final_two_pass and favorable_last_three:
            eligible.append(method)

    if eligible:
        selected = min(eligible, key=lambda name: traces[name][-1]["rmsd"])
        status = "GO_BOUNDED_EXTENSION"
        reason = "final two checkpoints pass and the preregistered last-three trend is favorable"
    elif val_loss_materially_falling:
        selected = None
        status = "STOP_OBJECTIVE_MISMATCH"
        reason = "validation loss falls materially but no rollout method satisfies the gate"
    else:
        selected = None
        status = "STOP_OPTIMIZATION_INCONCLUSIVE"
        reason = "neither rollout recovery nor a material validation-loss decline is established"

    return {
        "status": status,
        "scope": "bounded_paper_loss_continuation_only",
        "formal_training_authorized": False,
        "selected_method": selected,
        "reason": reason,
        "decision_rule": {
            "required_final_consecutive_passes": 2,
            "trend_steps": [1800, 1900, 2000],
            "rmsd": "model < no-op at steps 1900 and 2000; step2000 < step1800",
            "bond_mean_angstrom": [MIN_BOND_MEAN, MAX_BOND_MEAN],
            "bond_max_angstrom": f"<= {MAX_BOND_MAX}",
            "geometry_excess": "step2000 <= step1800",
            "material_val_loss_decline": "val_loss_2000 <= 0.995 * val_loss_1100",
        },
        "val_loss": {
            "step_1100": history[0]["val_loss"],
            "step_2000": history[-1]["val_loss"],
            "materially_falling": val_loss_materially_falling,
        },
        "methods": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(args.history, args.rollout_dir, args.domain_list_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
