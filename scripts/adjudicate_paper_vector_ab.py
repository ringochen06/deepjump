#!/usr/bin/env python
"""Fail-closed adjudication for the paper-horizon vector-attention arm."""

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
    PAPER_HORIZON_PROFILE,
    PAPER_VECTOR_PROFILE,
)


EXPECTED_STEPS = list(range(100, 2001, 100))
EXPECTED_BASELINE_CHECKPOINT_SHA256 = (
    "fb12d776b106867ca14a8f56476daf776a6296b6dca640f03c2188a75a69bb47"
)
EXPECTED_BASELINE_HISTORY_SHA256 = (
    "868e3e44386163e61e61f6c0da60c160e3cb9f282e20c3ba7a9198208c64fa3f"
)
EXPECTED_BASELINE_FINAL_VAL_LOSS = 4.176195051670074
ABSOLUTE_PASS = "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"
EXTERNAL_ABSOLUTE_PASS = "PASS_CONDITIONAL_SAFEGUARD_PAPER_VECTOR_EXTERNAL20"
MIN_PAIRED_DOMAINS_BETTER = 14
MATERIAL_OBJECTIVE_FACTOR = 0.995
MIN_BOND_MEAN = 3.2
MAX_BOND_MEAN = 4.5
MAX_BOND_MAX = 5.5
ROLLOUT_METHODS = ("mean", "ode_1")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
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


def _h20_gate(path: Path, checkpoint_sha256: str) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("H20 result must be an object")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("H20 checkpoint SHA256 mismatch")
    if int(payload.get("checkpoint_step", -1)) != 2000:
        raise ValueError("H20 checkpoint step mismatch")
    if int(payload.get("delta_frames", -1)) != 1:
        raise ValueError("H20 delta must be 1")
    panel = payload.get("domain_panel", {})
    if panel.get("sha256") != (
        "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
    ) or int(panel.get("evaluated_count", -1)) != 3:
        raise ValueError("H20 frozen panel mismatch")
    settings = payload.get("settings", {})
    expected_settings = {
        "domains": 3,
        "starts": 2,
        "steps": 20,
        "methods": "mean,ode_1",
        "seed": 20260718,
        "integrator": "euler",
        "tau_max": 1.0,
        "terminal_denoise": False,
        "drift_anchor": "state",
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(f"H20 setting mismatch: {key}")
    summary = payload.get("summary", {})
    noop = summary.get("noop", {})
    if noop.get("finite") is not True:
        raise ValueError("H20 no-op summary is not finite")
    noop_rmsd = _finite(noop.get("mean_final_rmsd"), "H20 no-op RMSD")
    methods = {}
    for method in ROLLOUT_METHODS:
        row = summary.get(method, {})
        if row.get("finite") is not True:
            raise ValueError(f"H20 {method} summary is not finite")
        rmsd = _finite(row.get("mean_final_rmsd"), f"H20 {method} RMSD")
        bond_mean = _finite(
            row.get("mean_final_bond_mean"), f"H20 {method} bond mean"
        )
        bond_max = _finite(
            row.get("mean_final_bond_max"), f"H20 {method} bond max"
        )
        methods[method] = {
            "rmsd": rmsd,
            "noop_rmsd": noop_rmsd,
            "bond_mean": bond_mean,
            "bond_max": bond_max,
            "passes": bool(
                rmsd < noop_rmsd
                and MIN_BOND_MEAN <= bond_mean <= MAX_BOND_MEAN
                and bond_max <= MAX_BOND_MAX
            ),
        }
    eligible = [name for name, row in methods.items() if row["passes"]]
    selected = min(eligible, key=lambda name: methods[name]["rmsd"]) if eligible else None
    return {
        "passes": bool(eligible),
        "selected_method": selected,
        "methods": methods,
        "decision_rule": {
            "fixed_checkpoint_step": 2000,
            "fixed_rollout_steps": 20,
            "eligible_methods": list(ROLLOUT_METHODS),
            "requires_at_least_one_method": True,
            "rmsd": "model < no-op",
            "bond_mean_angstrom": [MIN_BOND_MEAN, MAX_BOND_MEAN],
            "bond_max_angstrom": f"<= {MAX_BOND_MAX}",
        },
    }


def _rename_summary(summary: dict) -> dict:
    return {
        "mean_candidate_minus_baseline": summary["mean_model_minus_noop"],
        "standard_error": summary["standard_error"],
        "ci95_candidate_minus_baseline": summary["ci95_model_minus_noop"],
        "domains": summary["domains"],
    }


def adjudicate(
    baseline_decision_path: Path,
    candidate_decision_path: Path,
    baseline_history_path: Path,
    candidate_history_path: Path,
    *,
    candidate_h20_path: Path | None = None,
    panel_kind: str = "training",
) -> dict:
    if panel_kind not in {"training", "paper-vector-external"}:
        raise ValueError("unknown paper-vector A/B panel kind")
    baseline = _decision(baseline_decision_path, PAPER_HORIZON_PROFILE)
    candidate = _decision(candidate_decision_path, PAPER_VECTOR_PROFILE)
    if baseline.get("checkpoint_sha256") != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise ValueError("baseline checkpoint is not the sealed paper-horizon artifact")
    for key in ("training_domain_list_sha256", "domain_list_sha256"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"A/B {key} mismatch")
    if baseline.get("checkpoint_sha256") == candidate.get("checkpoint_sha256"):
        raise ValueError("A/B checkpoints must be distinct")
    external_evidence = None
    if panel_kind == "paper-vector-external":
        external_evidence = baseline.get("external_evidence")
        if not isinstance(external_evidence, dict) or not external_evidence:
            raise ValueError("external A/B evidence binding is missing")
        if candidate.get("external_evidence") != external_evidence:
            raise ValueError("external A/B evidence bindings differ")

    if _sha256(baseline_history_path) != EXPECTED_BASELINE_HISTORY_SHA256:
        raise ValueError("baseline history SHA256 mismatch")
    baseline_history = _history(baseline_history_path)
    candidate_history = _history(candidate_history_path)
    if [row["noop_rmsd"] for row in baseline_history] != [
        row["noop_rmsd"] for row in candidate_history
    ]:
        raise ValueError("A/B frozen validation no-op histories differ")
    baseline_loss = _finite(baseline_history[-1]["val_loss"], "baseline final val_loss")
    if baseline_loss != EXPECTED_BASELINE_FINAL_VAL_LOSS:
        raise ValueError("baseline final val_loss mismatch")
    candidate_loss = _finite(candidate_history[-1]["val_loss"], "candidate final val_loss")
    objective_improved = candidate_loss <= MATERIAL_OBJECTIVE_FACTOR * baseline_loss

    paired = [
        _finite(candidate_value, f"candidate domain {index}")
        - _finite(baseline_value, f"baseline domain {index}")
        for index, (candidate_value, baseline_value) in enumerate(
            zip(
                candidate["domain_mean_guarded_minus_noop"],
                baseline["domain_mean_guarded_minus_noop"],
                strict=True,
            )
        )
    ]
    primary_raw = _t_summary(paired, T_CRITICAL_19)
    primary = _rename_summary(primary_raw)
    domains_better = sum(value < 0 for value in paired)
    leave_one_out = []
    for index in range(20):
        raw = _t_summary(paired[:index] + paired[index + 1 :], T_CRITICAL_18)
        renamed = _rename_summary(raw)
        renamed["excluded_domain_index"] = index
        renamed["passes_negative"] = bool(
            renamed["standard_error"] > ZERO_WIDTH_EPS
            and renamed["ci95_candidate_minus_baseline"][1] < 0
        )
        leave_one_out.append(renamed)
    paired_pass = bool(
        primary["standard_error"] > ZERO_WIDTH_EPS
        and primary["ci95_candidate_minus_baseline"][1] < 0
        and domains_better >= MIN_PAIRED_DOMAINS_BETTER
        and all(row["passes_negative"] for row in leave_one_out)
    )
    required_absolute_status = (
        ABSOLUTE_PASS if panel_kind == "training" else EXTERNAL_ABSOLUTE_PASS
    )
    baseline_required_status = (
        ABSOLUTE_PASS if panel_kind == "training" else EXTERNAL_ABSOLUTE_PASS
    )
    baseline_reproduced = baseline.get("status") == baseline_required_status
    absolute_pass = candidate.get("status") == required_absolute_status
    if panel_kind == "training":
        if candidate_h20_path is None:
            raise ValueError("training A/B requires the fixed H20 gate")
        h20 = _h20_gate(candidate_h20_path, candidate["checkpoint_sha256"])
    else:
        if candidate_h20_path is not None:
            raise ValueError("external A/B must not replace the training H20 prerequisite")
        h20 = None

    if not baseline_reproduced:
        status = "STOP_PAPER_VECTOR_BASELINE_REPRODUCIBILITY"
    elif not absolute_pass:
        status = "STOP_PAPER_VECTOR_ABSOLUTE_GATE"
    elif not objective_improved:
        status = "STOP_PAPER_VECTOR_OBJECTIVE_GAIN"
    elif not paired_pass:
        status = "STOP_PAPER_VECTOR_PAIRED_ADVANTAGE"
    elif h20 is not None and not h20["passes"]:
        status = "STOP_PAPER_VECTOR_H20_GATE"
    else:
        status = (
            "ADVANCE_PAPER_VECTOR_EXTERNAL20"
            if panel_kind == "training"
            else "PASS_PAPER_VECTOR_EXTERNAL20"
        )

    return {
        "status": status,
        "scope": (
            "matched_fresh_continuous_0_to_2000_paper_vector_attention_training_dev_ab"
            if panel_kind == "training"
            else "matched_paper_vector_attention_external20_ab"
        ),
        "panel_kind": panel_kind,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "baseline_history_sha256": _sha256(baseline_history_path),
        "candidate_history_sha256": _sha256(candidate_history_path),
        "training_domain_list_sha256": candidate["training_domain_list_sha256"],
        "domain_list_sha256": candidate["domain_list_sha256"],
        "external_evidence": external_evidence,
        "baseline_absolute_status": baseline["status"],
        "baseline_required_absolute_status": baseline_required_status,
        "baseline_reproduced": baseline_reproduced,
        "candidate_absolute_status": candidate["status"],
        "candidate_absolute_pass": absolute_pass,
        "objective": {
            "baseline_step2000_val_loss": baseline_loss,
            "candidate_step2000_val_loss": candidate_loss,
            "required_candidate_factor": MATERIAL_OBJECTIVE_FACTOR,
            "required_candidate_max_val_loss": (
                MATERIAL_OBJECTIVE_FACTOR * baseline_loss
            ),
            "passes": objective_improved,
        },
        "paired_candidate_minus_baseline": paired,
        "paired_primary": primary,
        "paired_domains_better": domains_better,
        "paired_leave_one_out": leave_one_out,
        "paired_pass": paired_pass,
        "candidate_h20_gate": h20,
        "decision_rule": {
            "fixed_checkpoint_step": 2000,
            "baseline_absolute_gate": baseline_required_status,
            "candidate_absolute_gate": required_absolute_status,
            "objective": "candidate val_loss <= 0.995 * sealed full-tensor paper-horizon val_loss",
            "paired_metric": "candidate guarded-minus-noop minus baseline guarded-minus-noop",
            "paired_ci": "95% t-CI upper < 0",
            "paired_domains_better_min": MIN_PAIRED_DOMAINS_BETTER,
            "paired_leave_one_out": "20/20 95% t-CI upper < 0",
            "training_h20_gate": "fixed step2000; at least one frozen method passes existing RMSD/geometry bounds",
        },
        "external_development_authorized": (
            status == "ADVANCE_PAPER_VECTOR_EXTERNAL20"
        ),
        "second_seed_scientifically_eligible": (
            status == "PASS_PAPER_VECTOR_EXTERNAL20"
        ),
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
    parser.add_argument("--candidate-h20", type=Path)
    parser.add_argument(
        "--panel-kind",
        choices=("training", "paper-vector-external"),
        default="training",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.baseline_decision,
        args.candidate_decision,
        args.baseline_history,
        args.candidate_history,
        candidate_h20_path=args.candidate_h20,
        panel_kind=args.panel_kind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
