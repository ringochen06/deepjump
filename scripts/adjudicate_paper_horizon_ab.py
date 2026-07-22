#!/usr/bin/env python
"""Fail-closed adjudication for the matched 1000-vs-500k LR-horizon A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.adjudicate_endpoint_panel import _t_summary
from scripts.adjudicate_guarded_endpoint_panel import (
    T_CRITICAL_18,
    T_CRITICAL_19,
    ZERO_WIDTH_EPS,
)
from scripts.guarded_endpoint_panel_eval import (
    HORIZON_AB_BASELINE_PROFILE,
    PAPER_HORIZON_PROFILE,
)

EXPECTED_STEPS = list(range(100, 2001, 100))
ABSOLUTE_PASS = "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"
EXTERNAL_ABSOLUTE_PASS = "PASS_CONDITIONAL_SAFEGUARD_PAPER_HORIZON_EXTERNAL20"
MIN_PAIRED_DOMAINS_BETTER = 14
MATERIAL_OBJECTIVE_FACTOR = 0.995


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _history(path: Path) -> list[dict]:
    payload = _load_json(path)
    if not isinstance(payload, list) or [row.get("step") for row in payload] != EXPECTED_STEPS:
        raise ValueError(f"{path}: expected validation steps {EXPECTED_STEPS}")
    for row in payload:
        for key in ("val_loss", "val_rmsd", "noop_rmsd"):
            _finite(row.get(key), f"{path} step {row['step']} {key}")
    return payload


def _decision(path: Path, profile: str) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: decision must be an object")
    if payload.get("checkpoint_profile") != profile:
        raise ValueError(f"{path}: checkpoint profile mismatch")
    if int(payload.get("checkpoint_step", -1)) != 2000:
        raise ValueError(f"{path}: checkpoint step mismatch")
    if int(payload.get("domains", -1)) != 20 or int(payload.get("starts", -1)) != 1500:
        raise ValueError(f"{path}: panel size mismatch")
    values = payload.get("domain_mean_guarded_minus_noop")
    if not isinstance(values, list) or len(values) != 20:
        raise ValueError(f"{path}: expected 20 domain means")
    for index, value in enumerate(values):
        _finite(value, f"{path} domain mean {index}")
    return payload


def adjudicate(
    baseline_decision_path: Path,
    candidate_decision_path: Path,
    baseline_history_path: Path,
    candidate_history_path: Path,
    *,
    panel_kind: str = "training",
) -> dict:
    if panel_kind not in {"training", "paper-horizon-external"}:
        raise ValueError("unknown paper-horizon A/B panel kind")
    baseline = _decision(baseline_decision_path, HORIZON_AB_BASELINE_PROFILE)
    candidate = _decision(candidate_decision_path, PAPER_HORIZON_PROFILE)
    for key in ("training_domain_list_sha256", "domain_list_sha256"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"A/B {key} mismatch")
    if baseline.get("checkpoint_sha256") == candidate.get("checkpoint_sha256"):
        raise ValueError("A/B checkpoints must be distinct")

    baseline_history = _history(baseline_history_path)
    candidate_history = _history(candidate_history_path)
    baseline_noop = [row["noop_rmsd"] for row in baseline_history]
    candidate_noop = [row["noop_rmsd"] for row in candidate_history]
    if baseline_noop != candidate_noop:
        raise ValueError("A/B frozen validation no-op histories differ")

    baseline_loss = _finite(baseline_history[-1]["val_loss"], "baseline final val_loss")
    candidate_loss = _finite(candidate_history[-1]["val_loss"], "candidate final val_loss")
    objective_improved = candidate_loss <= MATERIAL_OBJECTIVE_FACTOR * baseline_loss

    paired = [
        _finite(candidate_value, f"candidate domain {index}")
        - _finite(baseline_value, f"baseline domain {index}")
        for index, (candidate_value, baseline_value) in enumerate(zip(
            candidate["domain_mean_guarded_minus_noop"],
            baseline["domain_mean_guarded_minus_noop"],
        ))
    ]
    primary = _t_summary(paired, T_CRITICAL_19)
    domains_better = sum(value < 0 for value in paired)
    leave_one_out = []
    for index, excluded_domain in enumerate(range(20)):
        row = _t_summary(paired[:index] + paired[index + 1 :], T_CRITICAL_18)
        row["excluded_domain_index"] = excluded_domain
        row["passes_negative"] = bool(
            row["standard_error"] > ZERO_WIDTH_EPS
            and row["ci95_model_minus_noop"][1] < 0
        )
        leave_one_out.append(row)
    paired_pass = bool(
        primary["standard_error"] > ZERO_WIDTH_EPS
        and primary["ci95_model_minus_noop"][1] < 0
        and domains_better >= MIN_PAIRED_DOMAINS_BETTER
        and all(row["passes_negative"] for row in leave_one_out)
    )
    required_absolute_status = (
        ABSOLUTE_PASS if panel_kind == "training" else EXTERNAL_ABSOLUTE_PASS
    )
    absolute_pass = candidate.get("status") == required_absolute_status

    if not absolute_pass:
        status = "STOP_PAPER_HORIZON_ABSOLUTE_GATE"
    elif not objective_improved:
        status = "STOP_PAPER_HORIZON_OBJECTIVE_GAIN"
    elif not paired_pass:
        status = "STOP_PAPER_HORIZON_PAIRED_ADVANTAGE"
    else:
        status = (
            "ADVANCE_PAPER_HORIZON_EXTERNAL20"
            if panel_kind == "training"
            else "PASS_PAPER_HORIZON_EXTERNAL20"
        )

    return {
        "status": status,
        "scope": (
            "matched_fresh_continuous_0_to_2000_lr_horizon_training_dev_ab"
            if panel_kind == "training"
            else "matched_lr_horizon_paper_external20_ab"
        ),
        "panel_kind": panel_kind,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "baseline_history_sha256": _sha256(baseline_history_path),
        "candidate_history_sha256": _sha256(candidate_history_path),
        "training_domain_list_sha256": candidate["training_domain_list_sha256"],
        "domain_list_sha256": candidate["domain_list_sha256"],
        "candidate_absolute_status": candidate["status"],
        "candidate_absolute_pass": absolute_pass,
        "objective": {
            "baseline_step2000_val_loss": baseline_loss,
            "candidate_step2000_val_loss": candidate_loss,
            "required_candidate_factor": MATERIAL_OBJECTIVE_FACTOR,
            "passes": objective_improved,
        },
        "paired_candidate_minus_baseline": paired,
        "paired_primary": primary,
        "paired_domains_better": domains_better,
        "paired_leave_one_out": leave_one_out,
        "paired_pass": paired_pass,
        "decision_rule": {
            "fixed_checkpoint_step": 2000,
            "candidate_absolute_gate": required_absolute_status,
            "objective": "candidate val_loss <= 0.995 * matched baseline val_loss",
            "paired_metric": "candidate guarded-minus-noop minus baseline guarded-minus-noop",
            "paired_ci": "95% t-CI upper < 0",
            "paired_domains_better_min": MIN_PAIRED_DOMAINS_BETTER,
            "paired_leave_one_out": "20/20 95% t-CI upper < 0",
        },
        "external_development_authorized": (
            status == "ADVANCE_PAPER_HORIZON_EXTERNAL20"
        ),
        "second_seed_scientifically_eligible": (
            status == "PASS_PAPER_HORIZON_EXTERNAL20"
        ),
        # Authorization is deliberately withheld until an independent OBS
        # double-readback marker is verified by the downstream runner.
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-decision", required=True, type=Path)
    parser.add_argument("--candidate-decision", required=True, type=Path)
    parser.add_argument("--baseline-history", required=True, type=Path)
    parser.add_argument("--candidate-history", required=True, type=Path)
    parser.add_argument(
        "--panel-kind",
        choices=("training", "paper-horizon-external"),
        default="training",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.baseline_decision,
        args.candidate_decision,
        args.baseline_history,
        args.candidate_history,
        panel_kind=args.panel_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
