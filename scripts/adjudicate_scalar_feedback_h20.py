#!/usr/bin/env python
"""Fail-closed H1-H20 scalar feedback mechanism discriminator."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path

from scripts.adjudicate_scalar_feedback_h6 import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_DOMAIN_LIST_SHA256,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_SOURCE_EVIDENCE_SHA256,
)
from scripts.external_endpoint_identity import verify_multidomain_checkpoint
from scripts.guarded_endpoint_panel_eval import (
    PAPER_SCALAR_VALUE_PROFILE,
    checkpoint_profile_requirements,
)
from scripts.rollout_robustness_eval import summarize_domains


EXPECTED_H6_DECISION_SHA256 = (
    "ace2b577efadb0c81f4d793ae07708f27c68f100fbd666fba4dabe47ac805e09"
)
EXPECTED_H6_COMPLETION_SHA256 = (
    "90a03a75c53f93291b2c841597080a049b66022f764405193a7fc01b3149c510"
)
EXPECTED_H6_STATUS = "INCONCLUSIVE_SCALAR_FEEDBACK_H6"
EXPECTED_DOMAINS = ("1gxlA02", "2dgmA02", "4i9cA01")
EXPECTED_STEPS = 20
EXPECTED_STARTS = 2
BOND_MEAN_RANGE = (3.2, 4.5)
MAX_BOND_MAX = 5.5
TEACHER_STRONG_WINS = 14
TEACHER_WEAK_WINS = 6


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: str | Path, label: str) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_source_evidence(path: str | Path) -> dict:
    if not hmac.compare_digest(_sha256(path), EXPECTED_SOURCE_EVIDENCE_SHA256):
        raise ValueError("source training evidence SHA256 mismatch")
    payload = _load_object(path, "source training evidence")
    expected = {
        "schema": "deepjump.scalar_value_training_evidence.v1",
        "commit": EXPECTED_SOURCE_COMMIT,
        "candidate_config_sha256": EXPECTED_CONFIG_SHA256,
        "candidate_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"source training evidence {key} mismatch")
    return payload


def _require_h6_evidence(
    decision_path: str | Path, completion_path: str | Path
) -> tuple[dict, dict]:
    if not hmac.compare_digest(_sha256(decision_path), EXPECTED_H6_DECISION_SHA256):
        raise ValueError("H6 decision SHA256 mismatch")
    if not hmac.compare_digest(_sha256(completion_path), EXPECTED_H6_COMPLETION_SHA256):
        raise ValueError("H6 completion SHA256 mismatch")
    payload = _load_object(decision_path, "H6 decision")
    expected = {
        "status": EXPECTED_H6_STATUS,
        "scope": "scalar_step2000_teacher_forced_vs_autoregressive_h1_h6_no_training",
        "checkpoint_step": 2000,
        "checkpoint_profile": PAPER_SCALAR_VALUE_PROFILE,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_source_commit": EXPECTED_SOURCE_COMMIT,
        "source_evidence_sha256": EXPECTED_SOURCE_EVIDENCE_SHA256,
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
        "domain": EXPECTED_DOMAINS[0],
        "starts": 5,
        "steps": 6,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"H6 decision {key} mismatch")
    for key in (
        "external_development_scientifically_eligible",
        "external_development_authorized",
        "second_seed_scientifically_eligible",
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"H6 decision {key} must be false")
    completion = _load_object(completion_path, "H6 completion")
    expected_completion = {
        "status": "OBS_DOUBLE_READBACK_PASS",
        "decision_status": EXPECTED_H6_STATUS,
        "run_id": "20260722T115322Z",
        "commit": "279b9fd628725f36cd2d1508e7222110ba0fa461",
        "archived_decision_sha256": EXPECTED_H6_DECISION_SHA256,
        "recomputed_decision_sha256": EXPECTED_H6_DECISION_SHA256,
        "independent_readbacks_verified": 2,
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    for key, value in expected_completion.items():
        if completion.get(key) != value or (
            isinstance(value, bool) and type(completion.get(key)) is not bool
        ):
            raise ValueError(f"H6 completion {key} mismatch")
    return payload, completion


def _finite_series(values: object, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
        raise ValueError(f"{label} must contain H0..H{EXPECTED_STEPS}")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{label} must be finite")
    return numbers


def _method(methods: dict, name: str) -> dict[str, list]:
    payload = methods.get(name)
    if not isinstance(payload, dict):
        raise ValueError(f"missing method: {name}")
    expected_metrics = {
        "rmsd",
        "rmsd_by_start",
        "fnc",
        "bond_mean",
        "bond_p95",
        "bond_p99",
        "bond_max",
        "bond_mae_real",
        "angle_cos_mae_real",
        "bond_mean_by_start",
        "bond_max_by_start",
    }
    if set(payload) != expected_metrics:
        raise ValueError(f"{name} has missing or extra metrics")
    validated: dict[str, list] = {}
    for metric in expected_metrics - {
        "rmsd_by_start", "bond_mean_by_start", "bond_max_by_start"
    }:
        validated[metric] = _finite_series(payload.get(metric), f"{name}.{metric}")
    by_start = payload.get("rmsd_by_start")
    if not isinstance(by_start, list) or len(by_start) != EXPECTED_STEPS + 1:
        raise ValueError(f"{name}.rmsd_by_start must contain H0..H{EXPECTED_STEPS}")
    if any(not isinstance(row, list) or len(row) != EXPECTED_STARTS for row in by_start):
        raise ValueError(f"{name}.rmsd_by_start must contain two starts per horizon")
    converted = [[float(value) for value in row] for row in by_start]
    if not all(math.isfinite(value) for row in converted for value in row):
        raise ValueError(f"{name}.rmsd_by_start must be finite")
    for step, (aggregate, row) in enumerate(
        zip(validated["rmsd"], converted, strict=True)
    ):
        recomputed = math.fsum(row) / EXPECTED_STARTS
        if not math.isclose(float(aggregate), recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name}.rmsd H{step} does not match rmsd_by_start")
    validated["rmsd_by_start"] = converted
    for metric in ("bond_mean_by_start", "bond_max_by_start"):
        values = payload.get(metric)
        if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
            raise ValueError(f"{name}.{metric} must contain H0..H{EXPECTED_STEPS}")
        if any(not isinstance(row, list) or len(row) != EXPECTED_STARTS for row in values):
            raise ValueError(f"{name}.{metric} must contain two starts per horizon")
        matrix = [[float(value) for value in row] for row in values]
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ValueError(f"{name}.{metric} must be finite")
        validated[metric] = matrix
    for step in range(EXPECTED_STEPS + 1):
        mean_from_starts = math.fsum(validated["bond_mean_by_start"][step]) / EXPECTED_STARTS
        max_from_starts = max(validated["bond_max_by_start"][step])
        if not math.isclose(
            validated["bond_mean"][step], mean_from_starts, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"{name}.bond_mean H{step} does not match per-start geometry")
        if not math.isclose(
            validated["bond_max"][step], max_from_starts, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(f"{name}.bond_max H{step} does not match per-start geometry")
    return validated


def _first_nonphysical(method: dict[str, list]) -> int | None:
    for step in range(1, EXPECTED_STEPS + 1):
        for start in range(EXPECTED_STARTS):
            if not (
                BOND_MEAN_RANGE[0]
                <= float(method["bond_mean_by_start"][step][start])
                <= BOND_MEAN_RANGE[1]
                and float(method["bond_max_by_start"][step][start]) <= MAX_BOND_MAX
            ):
                return step
    return None


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    domain_list_path: str | Path,
    source_evidence_path: str | Path,
    h6_decision_path: str | Path,
    h6_completion_path: str | Path,
) -> dict:
    """Verify frozen identities and classify H20 endpoint versus feedback failure."""
    _require_source_evidence(source_evidence_path)
    _require_h6_evidence(h6_decision_path, h6_completion_path)
    if not hmac.compare_digest(_sha256(checkpoint_path), EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("checkpoint SHA256 mismatch")
    if not hmac.compare_digest(_sha256(domain_list_path), EXPECTED_DOMAIN_LIST_SHA256):
        raise ValueError("domain list SHA256 mismatch")
    expected_data, expected_model, expected_train = checkpoint_profile_requirements(
        PAPER_SCALAR_VALUE_PROFILE, EXPECTED_CHECKPOINT_SHA256
    )
    verify_multidomain_checkpoint(
        checkpoint_path,
        EXPECTED_CHECKPOINT_SHA256,
        expected_step=2000,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )

    result = _load_object(result_path, "feedback result")
    expected_result_keys = {
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_step",
        "settings",
        "preprocessing",
        "delta_frames",
        "domain_panel",
        "summary",
        "domains",
    }
    if set(result) != expected_result_keys:
        raise ValueError("feedback result has missing or extra fields")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("result checkpoint path mismatch")
    if result.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("result checkpoint SHA256 mismatch")
    if int(result.get("checkpoint_step", -1)) != 2000:
        raise ValueError("scalar feedback H20 discriminator requires checkpoint step 2000")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("scalar feedback H20 discriminator requires delta=1")
    if result.get("preprocessing") != {"canon_symmetric": True}:
        raise ValueError("scalar feedback H20 discriminator requires canonical symmetric atom slots")

    panel = result.get("domain_panel", {})
    if set(panel) != {"path", "sha256", "count", "evaluated_count"}:
        raise ValueError("feedback domain panel has missing or extra fields")
    if Path(panel.get("path", "")).resolve() != Path(domain_list_path).resolve():
        raise ValueError("result domain list path mismatch")
    if panel.get("sha256") != EXPECTED_DOMAIN_LIST_SHA256:
        raise ValueError("result domain list SHA256 mismatch")
    if int(panel.get("count", -1)) != 20 or int(panel.get("evaluated_count", -1)) != 3:
        raise ValueError("scalar feedback H20 discriminator requires frozen dev20 and three domains")

    settings = result.get("settings", {})
    expected_settings = {
        "ckpt": str(checkpoint_path),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "domain_list": str(domain_list_path),
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
        "domains": 3,
        "starts": EXPECTED_STARTS,
        "steps": EXPECTED_STEPS,
        "methods": "mean",
        "seed": 20260718,
        "noise_sigma": None,
        "integrator": "euler",
        "tau_max": 1.0,
        "terminal_denoise": False,
        "drift_anchor": "state",
        "project_v_atom_mask": False,
        "teacher_forced_mean": True,
        "per_start_geometry": True,
    }
    if set(settings) != set(expected_settings) | {"output"}:
        raise ValueError("scalar feedback H20 settings have missing or extra fields")
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected_settings.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"scalar feedback H20 settings mismatch: {mismatches}")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != len(EXPECTED_DOMAINS):
        raise ValueError("scalar feedback H20 discriminator requires three domain rows")
    validated_domains: list[dict[str, dict[str, list]]] = []
    expected_methods = {"noop", "one_step_persistence", "mean", "teacher_forced_mean"}
    for row, expected_domain in zip(domains, EXPECTED_DOMAINS, strict=True):
        if set(row) != {
            "domain", "residues_total", "residues_evaluated", "frames", "starts", "methods"
        }:
            raise ValueError("scalar feedback H20 domain row has missing or extra fields")
        if row.get("domain") != expected_domain:
            raise ValueError("scalar feedback H20 domain identity mismatch")
        residues_total = _positive_int(row.get("residues_total"), "residues_total")
        residues_evaluated = _positive_int(
            row.get("residues_evaluated"), "residues_evaluated"
        )
        if residues_evaluated != min(residues_total, 256):
            raise ValueError("residues_evaluated does not match frozen crop length")
        frames = _positive_int(row.get("frames"), "frames")
        last = frames - 1 - EXPECTED_STEPS
        if last < EXPECTED_STARTS - 1:
            raise ValueError("scalar feedback H20 frame count is too small")
        expected_starts = [0, last]
        if row.get("starts") != expected_starts:
            raise ValueError("scalar feedback H20 start identity mismatch")
        methods = row.get("methods", {})
        if set(methods) != expected_methods:
            raise ValueError("scalar feedback H20 method set mismatch")
        validated_domains.append({name: _method(methods, name) for name in expected_methods})

    expected_summary = {
        name: summarize_domains(domains, name, EXPECTED_STEPS)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    if result.get("summary") != expected_summary:
        raise ValueError("scalar feedback H20 summary mismatch")

    domain_evidence = []
    for domain, validated in zip(EXPECTED_DOMAINS, validated_domains, strict=True):
        teacher = validated["teacher_forced_mean"]
        autoregressive = validated["mean"]
        persistence = validated["one_step_persistence"]
        teacher_wins = sum(
            float(teacher["rmsd"][step]) < float(persistence["rmsd"][step])
            for step in range(1, EXPECTED_STEPS + 1)
        )
        domain_evidence.append({
            "domain": domain,
            "teacher_forced_first_nonphysical_step": _first_nonphysical(teacher),
            "autoregressive_first_nonphysical_step": _first_nonphysical(autoregressive),
            "teacher_forced_steps_better_than_one_step_persistence": teacher_wins,
        })

    teacher_strong_domains = sum(
        row["teacher_forced_first_nonphysical_step"] is None
        and row["teacher_forced_steps_better_than_one_step_persistence"] >= TEACHER_STRONG_WINS
        for row in domain_evidence
    )
    teacher_failure_domains = sum(
        row["teacher_forced_first_nonphysical_step"] is not None
        or row["teacher_forced_steps_better_than_one_step_persistence"] <= TEACHER_WEAK_WINS
        for row in domain_evidence
    )
    autoregressive_failure_domains = sum(
        row["autoregressive_first_nonphysical_step"] is not None
        for row in domain_evidence
    )
    if teacher_strong_domains == 3 and autoregressive_failure_domains >= 2:
        status = "FEEDBACK_DISTRIBUTION_SHIFT_H20"
    elif teacher_failure_domains >= 2:
        status = "ENDPOINT_OPERATOR_FAILURE_H20"
    else:
        status = "INCONCLUSIVE_SCALAR_FEEDBACK_H20"

    return {
        "status": status,
        "scope": "scalar_step2000_teacher_forced_vs_autoregressive_h1_h20_three_domain_no_training",
        "extends_h6_status": EXPECTED_H6_STATUS,
        "h6_decision_sha256": EXPECTED_H6_DECISION_SHA256,
        "h6_completion_sha256": EXPECTED_H6_COMPLETION_SHA256,
        "checkpoint_step": 2000,
        "checkpoint_profile": PAPER_SCALAR_VALUE_PROFILE,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_source_commit": EXPECTED_SOURCE_COMMIT,
        "source_evidence_sha256": EXPECTED_SOURCE_EVIDENCE_SHA256,
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
        "domains": list(EXPECTED_DOMAINS),
        "starts_per_domain": EXPECTED_STARTS,
        "steps": EXPECTED_STEPS,
        "domain_evidence": domain_evidence,
        "teacher_strong_domains": teacher_strong_domains,
        "teacher_failure_domains": teacher_failure_domains,
        "autoregressive_failure_domains": autoregressive_failure_domains,
        "decision_rule": {
            "physical_bounds": {
                "bond_mean_inclusive": list(BOND_MEAN_RANGE),
                "bond_max_inclusive_upper": MAX_BOND_MAX,
            },
            "FEEDBACK_DISTRIBUTION_SHIFT_H20": (
                "all three sets of teacher-forced endpoints remain physical and each strictly beats "
                "one-step persistence on at least 14/20 horizons, while autoregressive "
                "becomes nonphysical by H20 in at least two domains"
            ),
            "ENDPOINT_OPERATOR_FAILURE_H20": (
                "in at least two domains teacher becomes nonphysical by H20 or strictly "
                "beats one-step persistence on at most 6/20 horizons"
            ),
            "INCONCLUSIVE_SCALAR_FEEDBACK_H20": "neither preregistered condition is met",
        },
        "external_development_scientifically_eligible": False,
        "external_development_authorized": False,
        "second_seed_scientifically_eligible": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--domain-list", required=True, type=Path)
    parser.add_argument("--source-evidence", required=True, type=Path)
    parser.add_argument("--h6-decision", required=True, type=Path)
    parser.add_argument("--h6-completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.checkpoint,
        args.domain_list,
        args.source_evidence,
        args.h6_decision,
        args.h6_completion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
