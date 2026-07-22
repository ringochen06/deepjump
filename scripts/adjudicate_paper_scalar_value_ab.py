#!/usr/bin/env python
"""Fail-closed seed-0 adjudication for the scalar-value architecture arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.adjudicate_endpoint_panel import _t_summary
from scripts.adjudicate_guarded_endpoint_panel import (
    T_CRITICAL_18,
    T_CRITICAL_19,
    ZERO_WIDTH_EPS,
)
from scripts.adjudicate_paper_vector_ab import (
    ABSOLUTE_PASS,
    MATERIAL_OBJECTIVE_FACTOR,
    MIN_PAIRED_DOMAINS_BETTER,
    _decision,
    _finite,
    _h20_gate,
    _history,
    _rename_summary,
    _sha256,
)
from scripts.guarded_endpoint_panel_eval import (
    PAPER_SCALAR_VALUE_PROFILE,
    PAPER_VECTOR_PROFILE,
)


EXPECTED_BASELINE_CHECKPOINT_SHA256 = (
    "19d960826938419e1bf494701a09b395ece729e1c0dc2c8a5d1e6bf36d73053b"
)
EXPECTED_BASELINE_HISTORY_SHA256 = (
    "36f8850ba4e9c094526850370b22371d10df76765eead3e39adf051e68d0d80e"
)
EXPECTED_BASELINE_DECISION_SHA256 = (
    "0816f94b01bf8b434086677d59c913193a70aa8b802f79b46378590f772af7bf"
)
EXPECTED_BASELINE_FINAL_VAL_LOSS = 4.150860345363617
EXPECTED_BASELINE_ABSOLUTE_STATUS = "STOP_CONDITIONAL_SAFEGUARD_FALLBACK_CAP"


def _evidence(
    path: Path,
    *,
    baseline_decision_path: Path,
    baseline_replay_decision_path: Path,
    candidate_decision_path: Path,
    baseline_history_path: Path,
    candidate_history_path: Path,
    candidate_h20_path: Path,
    baseline_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
    training_domain_list_sha256: str,
    domain_list_sha256: str,
) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != (
        "deepjump.scalar_value_training_evidence.v1"
    ):
        raise ValueError("scalar-value training evidence schema mismatch")
    expected = {
        "baseline_decision_sha256": _sha256(baseline_decision_path),
        "baseline_replay_decision_sha256": _sha256(
            baseline_replay_decision_path
        ),
        "candidate_decision_sha256": _sha256(candidate_decision_path),
        "baseline_history_sha256": _sha256(baseline_history_path),
        "candidate_history_sha256": _sha256(candidate_history_path),
        "candidate_h20_sha256": _sha256(candidate_h20_path),
        "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "training_domain_list_sha256": training_domain_list_sha256,
        "domain_list_sha256": domain_list_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"scalar-value training evidence {key} mismatch")
    for key in ("run_id", "commit", "candidate_config_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"scalar-value training evidence {key} is missing")
    return payload


def adjudicate(
    baseline_decision_path: Path,
    baseline_replay_decision_path: Path,
    candidate_decision_path: Path,
    baseline_history_path: Path,
    candidate_history_path: Path,
    candidate_h20_path: Path,
    evidence_manifest_path: Path,
    *,
    expected_baseline_checkpoint_sha256: str | None = None,
    expected_baseline_history_sha256: str | None = None,
    expected_baseline_decision_sha256: str | None = None,
    expected_baseline_final_val_loss: float | None = None,
    expected_baseline_absolute_status: str | None = None,
) -> dict:
    expected_baseline_checkpoint_sha256 = (
        EXPECTED_BASELINE_CHECKPOINT_SHA256
        if expected_baseline_checkpoint_sha256 is None
        else expected_baseline_checkpoint_sha256
    )
    expected_baseline_history_sha256 = (
        EXPECTED_BASELINE_HISTORY_SHA256
        if expected_baseline_history_sha256 is None
        else expected_baseline_history_sha256
    )
    expected_baseline_decision_sha256 = (
        EXPECTED_BASELINE_DECISION_SHA256
        if expected_baseline_decision_sha256 is None
        else expected_baseline_decision_sha256
    )
    expected_baseline_final_val_loss = (
        EXPECTED_BASELINE_FINAL_VAL_LOSS
        if expected_baseline_final_val_loss is None
        else expected_baseline_final_val_loss
    )
    expected_baseline_absolute_status = (
        EXPECTED_BASELINE_ABSOLUTE_STATUS
        if expected_baseline_absolute_status is None
        else expected_baseline_absolute_status
    )
    baseline = _decision(baseline_decision_path, PAPER_VECTOR_PROFILE)
    baseline_replay = _decision(
        baseline_replay_decision_path, PAPER_VECTOR_PROFILE
    )
    candidate = _decision(candidate_decision_path, PAPER_SCALAR_VALUE_PROFILE)
    if baseline.get("checkpoint_sha256") != expected_baseline_checkpoint_sha256:
        raise ValueError("baseline checkpoint is not the sealed vector-only artifact")
    if not expected_baseline_decision_sha256 or (
        _sha256(baseline_decision_path) != expected_baseline_decision_sha256
    ):
        raise ValueError("baseline absolute decision SHA256 mismatch")
    if candidate.get("checkpoint_sha256") == baseline.get("checkpoint_sha256"):
        raise ValueError("A/B checkpoints must be distinct")
    for key in ("training_domain_list_sha256", "domain_list_sha256"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"A/B {key} mismatch")

    if _sha256(baseline_history_path) != expected_baseline_history_sha256:
        raise ValueError("baseline history SHA256 mismatch")
    baseline_history = _history(baseline_history_path)
    candidate_history = _history(candidate_history_path)
    if [row["noop_rmsd"] for row in baseline_history] != [
        row["noop_rmsd"] for row in candidate_history
    ]:
        raise ValueError("A/B frozen validation no-op histories differ")
    baseline_loss = _finite(
        baseline_history[-1]["val_loss"], "baseline final val_loss"
    )
    if baseline_loss != expected_baseline_final_val_loss:
        raise ValueError("baseline final val_loss mismatch")
    candidate_loss = _finite(
        candidate_history[-1]["val_loss"], "candidate final val_loss"
    )
    objective_improved = (
        candidate_loss <= MATERIAL_OBJECTIVE_FACTOR * baseline_loss
    )

    paired = [
        _finite(candidate_value, f"candidate domain {index}")
        - _finite(baseline_value, f"baseline domain {index}")
        for index, (candidate_value, baseline_value) in enumerate(zip(
            candidate["domain_mean_guarded_minus_noop"],
            baseline["domain_mean_guarded_minus_noop"],
            strict=True,
        ))
    ]
    primary = _rename_summary(_t_summary(paired, T_CRITICAL_19))
    domains_better = sum(value < 0 for value in paired)
    leave_one_out = []
    for index in range(20):
        summary = _rename_summary(
            _t_summary(paired[:index] + paired[index + 1 :], T_CRITICAL_18)
        )
        summary["excluded_domain_index"] = index
        summary["passes_negative"] = bool(
            summary["standard_error"] > ZERO_WIDTH_EPS
            and summary["ci95_candidate_minus_baseline"][1] < 0
        )
        leave_one_out.append(summary)
    paired_pass = bool(
        primary["standard_error"] > ZERO_WIDTH_EPS
        and primary["ci95_candidate_minus_baseline"][1] < 0
        and domains_better >= MIN_PAIRED_DOMAINS_BETTER
        and all(row["passes_negative"] for row in leave_one_out)
    )

    replay_fields = (
        "status",
        "checkpoint_step",
        "checkpoint_sha256",
        "training_domain_list_sha256",
        "domain_list_sha256",
        "domains",
        "starts",
        "domain_mean_guarded_minus_noop",
    )
    baseline_replay_mismatches = [
        key for key in replay_fields if baseline_replay.get(key) != baseline.get(key)
    ]
    baseline_reproduced = bool(
        baseline.get("status") == expected_baseline_absolute_status
        and not baseline_replay_mismatches
    )
    absolute_pass = candidate.get("status") == ABSOLUTE_PASS
    h20 = _h20_gate(candidate_h20_path, candidate["checkpoint_sha256"])
    evidence = _evidence(
        evidence_manifest_path,
        baseline_decision_path=baseline_decision_path,
        baseline_replay_decision_path=baseline_replay_decision_path,
        candidate_decision_path=candidate_decision_path,
        baseline_history_path=baseline_history_path,
        candidate_history_path=candidate_history_path,
        candidate_h20_path=candidate_h20_path,
        baseline_checkpoint_sha256=baseline["checkpoint_sha256"],
        candidate_checkpoint_sha256=candidate["checkpoint_sha256"],
        training_domain_list_sha256=candidate["training_domain_list_sha256"],
        domain_list_sha256=candidate["domain_list_sha256"],
    )
    if not baseline_reproduced:
        status = "STOP_SCALAR_VALUE_BASELINE_REPRODUCIBILITY"
    elif not absolute_pass:
        status = "STOP_SCALAR_VALUE_ABSOLUTE_GATE"
    elif not objective_improved:
        status = "STOP_SCALAR_VALUE_OBJECTIVE_GAIN"
    elif not paired_pass:
        status = "STOP_SCALAR_VALUE_PAIRED_ADVANTAGE"
    elif not h20["passes"]:
        status = "STOP_SCALAR_VALUE_H20_GATE"
    else:
        status = "ADVANCE_SCALAR_VALUE_EXTERNAL20"

    return {
        "status": status,
        "scope": (
            "matched_fresh_seed0_0_to_2000_normalized_scalar_value_training_dev_ab"
        ),
        "panel_kind": "training",
        "paper_equivalence": "architecture_hypothesis_not_paper_verified",
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "baseline_history_sha256": _sha256(baseline_history_path),
        "candidate_history_sha256": _sha256(candidate_history_path),
        "training_evidence_manifest_sha256": _sha256(evidence_manifest_path),
        "training_evidence": evidence,
        "training_domain_list_sha256": candidate["training_domain_list_sha256"],
        "domain_list_sha256": candidate["domain_list_sha256"],
        "baseline_absolute_status": baseline["status"],
        "baseline_replay_absolute_status": baseline_replay["status"],
        "baseline_required_absolute_status": expected_baseline_absolute_status,
        "baseline_reproduced": baseline_reproduced,
        "baseline_replay_mismatches": baseline_replay_mismatches,
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
            "baseline_identity": expected_baseline_absolute_status,
            "candidate_absolute_gate": ABSOLUTE_PASS,
            "objective": (
                "candidate val_loss <= 0.995 * sealed vector-only val_loss"
            ),
            "paired_metric": (
                "candidate guarded-minus-noop minus vector-only guarded-minus-noop"
            ),
            "paired_ci": "95% t-CI upper < 0",
            "paired_domains_better_min": MIN_PAIRED_DOMAINS_BETTER,
            "paired_leave_one_out": "20/20 95% t-CI upper < 0",
            "training_h20_gate": (
                "fixed step2000; at least one frozen raw method passes existing "
                "RMSD/no-op and geometry bounds"
            ),
        },
        "external_development_scientifically_eligible": (
            status == "ADVANCE_SCALAR_VALUE_EXTERNAL20"
        ),
        # The dedicated scalar-value external contract is intentionally a
        # separate implementation milestone and is not yet authorized here.
        "external_development_authorized": False,
        "second_seed_scientifically_eligible": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-decision", required=True, type=Path)
    parser.add_argument("--baseline-replay-decision", required=True, type=Path)
    parser.add_argument("--candidate-decision", required=True, type=Path)
    parser.add_argument("--baseline-history", required=True, type=Path)
    parser.add_argument("--candidate-history", required=True, type=Path)
    parser.add_argument("--candidate-h20", required=True, type=Path)
    parser.add_argument("--evidence-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.baseline_decision,
        args.baseline_replay_decision,
        args.candidate_decision,
        args.baseline_history,
        args.candidate_history,
        args.candidate_h20,
        args.evidence_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
