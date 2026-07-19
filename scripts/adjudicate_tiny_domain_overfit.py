#!/usr/bin/env python
"""Adjudicate the bounded full-tensor same-domain learnability diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_STEPS = tuple(range(500, 5001, 500))
SELECTION_WINDOW = (4000, 4500, 5000)
MAX_PLATEAU_RELATIVE_SPAN = 0.02
MAX_LOSS_RATIO = 0.50
BOND_MEAN_RANGE = (3.2, 4.5)
MAX_BOND_MAX = 5.5


def _finite_number(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def select_checkpoint(history_path: str | Path) -> dict:
    history = json.loads(Path(history_path).read_text())
    by_step: dict[int, dict] = {}
    for row in history:
        step = int(row["step"])
        if step in by_step:
            raise ValueError(f"duplicate history step: {step}")
        by_step[step] = row
    if tuple(sorted(by_step)) != EXPECTED_STEPS:
        raise ValueError(
            f"history steps mismatch: expected {EXPECTED_STEPS}, got {tuple(sorted(by_step))}"
        )

    losses = {
        step: _finite_number(by_step[step]["val_loss"], f"val_loss[{step}]")
        for step in EXPECTED_STEPS
    }
    if any(loss < 0 for loss in losses.values()):
        raise ValueError("validation losses must be non-negative")
    noop_rmsds = []
    for step in EXPECTED_STEPS:
        _finite_number(by_step[step]["val_rmsd"], f"val_rmsd[{step}]")
        noop_rmsds.append(
            _finite_number(by_step[step]["noop_rmsd"], f"noop_rmsd[{step}]")
        )
    if max(noop_rmsds) - min(noop_rmsds) > 1e-6:
        raise ValueError("validation no-op RMSD changed across checkpoints")

    selected_step = min(SELECTION_WINDOW, key=lambda step: (losses[step], step))
    window_losses = [losses[step] for step in SELECTION_WINDOW]
    plateau_relative_span = (
        max(window_losses) - min(window_losses)
    ) / max(abs(min(window_losses)), 1e-12)
    loss_ratio = losses[selected_step] / max(abs(losses[EXPECTED_STEPS[0]]), 1e-12)
    selected = by_step[selected_step]
    val_beats_noop = float(selected["val_rmsd"]) < float(selected["noop_rmsd"])
    converged = bool(
        plateau_relative_span <= MAX_PLATEAU_RELATIVE_SPAN
        and loss_ratio <= MAX_LOSS_RATIO
        and val_beats_noop
    )
    return {
        "selected_step": selected_step,
        "selection_source": "minimum val_loss within fixed final window",
        "selection_window": list(SELECTION_WINDOW),
        "selected_val_loss": losses[selected_step],
        "selected_val_rmsd": float(selected["val_rmsd"]),
        "selected_noop_rmsd": float(selected["noop_rmsd"]),
        "first_val_loss": losses[EXPECTED_STEPS[0]],
        "loss_ratio_to_step500": loss_ratio,
        "plateau_relative_span": plateau_relative_span,
        "val_beats_noop": val_beats_noop,
        "converged": converged,
        "convergence_rule": {
            "max_plateau_relative_span": MAX_PLATEAU_RELATIVE_SPAN,
            "max_loss_ratio_to_step500": MAX_LOSS_RATIO,
            "require_val_rmsd_below_noop": True,
        },
    }


def _validate_rollout(rollout: dict, selected_step: int, domain_list_sha256: str) -> None:
    if int(rollout.get("checkpoint_step", -1)) != selected_step:
        raise ValueError("rollout checkpoint does not match validation-only selection")
    if int(rollout.get("delta_frames", -1)) != 1:
        raise ValueError("rollout delta must be 1")
    panel = rollout.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256:
        raise ValueError("rollout domain panel mismatch")
    if int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("rollout must evaluate exactly one frozen domain")
    settings = rollout.get("settings", {})
    expected = {
        "domains": 1,
        "starts": 5,
        "steps": 20,
        "methods": "mean,ode_1",
        "integrator": "euler",
        "tau_max": 1.0,
        "terminal_denoise": False,
        "drift_anchor": "state",
    }
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"rollout settings mismatch: {mismatches}")


def adjudicate(
    history_path: str | Path,
    rollout_path: str | Path,
    domain_list_sha256: str,
) -> dict:
    selection = select_checkpoint(history_path)
    rollout = json.loads(Path(rollout_path).read_text())
    _validate_rollout(rollout, selection["selected_step"], domain_list_sha256)

    summary = rollout.get("summary", {})
    if "noop" not in summary or "mean" not in summary or "ode_1" not in summary:
        raise ValueError("rollout summary must contain noop, mean, and ode_1")
    noop_rmsd = _finite_number(summary["noop"]["mean_final_rmsd"], "noop rmsd")

    method_reports = {}
    for method in ("mean", "ode_1"):
        row = summary[method]
        rmsd = _finite_number(row["mean_final_rmsd"], f"{method} rmsd")
        bond_mean = _finite_number(row["mean_final_bond_mean"], f"{method} bond mean")
        bond_max = _finite_number(row["mean_final_bond_max"], f"{method} bond max")
        finite = bool(row.get("finite", False))
        physical = bool(
            finite
            and BOND_MEAN_RANGE[0] <= bond_mean <= BOND_MEAN_RANGE[1]
            and bond_max <= MAX_BOND_MAX
        )
        method_reports[method] = {
            "finite": finite,
            "mean_final_rmsd": rmsd,
            "noop_mean_final_rmsd": noop_rmsd,
            "beats_noop": rmsd < noop_rmsd,
            "mean_final_bond_mean": bond_mean,
            "mean_final_bond_max": bond_max,
            "physical": physical,
            "passes": bool(physical and rmsd < noop_rmsd),
        }

    if not selection["converged"]:
        status = "INCONCLUSIVE_NOT_CONVERGED"
    elif method_reports["mean"]["passes"]:
        status = "IN_DOMAIN_OPERATOR_LEARNABLE"
    else:
        status = "IN_DOMAIN_RECURRENCE_FAILURE"

    return {
        "status": status,
        "scope": "same-domain mechanism diagnostic only",
        "selection": selection,
        "methods": method_reports,
        "primary_method": "mean",
        "decision_rule": {
            "IN_DOMAIN_OPERATOR_LEARNABLE": (
                "converged and deterministic mean H20 is finite, has bond mean in "
                "[3.2,4.5] A, bond max <=5.5 A, and beats no-op RMSD"
            ),
            "IN_DOMAIN_RECURRENCE_FAILURE": (
                "converged but deterministic mean H20 fails physicality or no-op RMSD"
            ),
            "INCONCLUSIVE_NOT_CONVERGED": "the frozen convergence rule is not met",
        },
        "interpretation_limit": (
            "A pass proves only in-domain operator learnability. It does not prove that "
            "undertraining caused the full-panel failure or authorize broader evaluation."
        ),
        "domain_list_sha256": domain_list_sha256,
        "formal_training_authorized": False,
        "twenty_domain_authorized": False,
        "confirmation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--rollout")
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.selection_only:
        if args.rollout:
            parser.error("--selection-only cannot be combined with --rollout")
        report = select_checkpoint(args.history)
    else:
        if not args.rollout:
            parser.error("--rollout is required unless --selection-only is used")
        report = adjudicate(args.history, args.rollout, args.domain_list_sha256)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
