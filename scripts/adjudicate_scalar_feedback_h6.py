#!/usr/bin/env python
"""Fail-closed H1-H6 scalar-value feedback mechanism discriminator."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path

from scripts.external_endpoint_identity import verify_multidomain_checkpoint
from scripts.guarded_endpoint_panel_eval import (
    PAPER_SCALAR_VALUE_PROFILE,
    checkpoint_profile_requirements,
)
from scripts.rollout_robustness_eval import summarize_domains


EXPECTED_CHECKPOINT_SHA256 = (
    "fc5f1e7b5188af4911e518ac0e3d44c2aba4a22431360bde704465c9c1889a73"
)
EXPECTED_DOMAIN_LIST_SHA256 = (
    "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
)
EXPECTED_DOMAIN = "1gxlA02"
EXPECTED_SOURCE_COMMIT = "9af7125cab0badd3b4e3ef94de37d8a996d4c532"
EXPECTED_SOURCE_EVIDENCE_SHA256 = (
    "63c4e1027bd03722ab335bd61cff458aa7d3c562ee6adee22c75bbff461691da"
)
EXPECTED_CONFIG_SHA256 = (
    "2b6d96b647d386fe942bcfdc85dac29f6428b6fe685ce5b10a3f9117cdc48832"
)
EXPECTED_STEPS = 6
EXPECTED_STARTS = 5
BOND_MEAN_RANGE = (3.2, 4.5)
MAX_BOND_MAX = 5.5


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


def _finite_series(values: object, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
        raise ValueError(f"{label} must contain H0..H{EXPECTED_STEPS}")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{label} must be finite")
    return numbers


def _method(methods: dict, name: str) -> dict[str, list[float] | list[list[float]]]:
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
    }
    if set(payload) != expected_metrics:
        raise ValueError(f"{name} has missing or extra metrics")
    validated: dict[str, list[float] | list[list[float]]] = {}
    for metric in (
        "rmsd",
        "fnc",
        "bond_mean",
        "bond_p95",
        "bond_p99",
        "bond_max",
        "bond_mae_real",
        "angle_cos_mae_real",
    ):
        validated[metric] = _finite_series(payload.get(metric), f"{name}.{metric}")
    by_start = payload.get("rmsd_by_start")
    if not isinstance(by_start, list) or len(by_start) != EXPECTED_STEPS + 1:
        raise ValueError(f"{name}.rmsd_by_start must contain H0..H{EXPECTED_STEPS}")
    if any(not isinstance(row, list) or len(row) != EXPECTED_STARTS for row in by_start):
        raise ValueError(f"{name}.rmsd_by_start must contain five starts per horizon")
    converted = [[float(value) for value in row] for row in by_start]
    if not all(math.isfinite(value) for row in converted for value in row):
        raise ValueError(f"{name}.rmsd_by_start must be finite")
    rmsd = validated["rmsd"]
    assert isinstance(rmsd, list)
    for step, (aggregate, row) in enumerate(zip(rmsd, converted, strict=True)):
        recomputed = math.fsum(row) / EXPECTED_STARTS
        if not math.isclose(float(aggregate), recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name}.rmsd H{step} does not match rmsd_by_start")
    validated["rmsd_by_start"] = converted
    return validated


def _first_nonphysical(method: dict[str, list[float] | list[list[float]]]) -> int | None:
    means = method["bond_mean"]
    maxima = method["bond_max"]
    assert isinstance(means, list) and isinstance(maxima, list)
    for step in range(1, EXPECTED_STEPS + 1):
        if not (
            BOND_MEAN_RANGE[0] <= float(means[step]) <= BOND_MEAN_RANGE[1]
            and float(maxima[step]) <= MAX_BOND_MAX
        ):
            return step
    return None


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    domain_list_path: str | Path,
    source_evidence_path: str | Path,
) -> dict:
    """Verify frozen identities and classify endpoint versus feedback failure."""
    _require_source_evidence(source_evidence_path)
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
        raise ValueError("scalar feedback discriminator requires checkpoint step 2000")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("scalar feedback discriminator requires delta=1")
    preprocessing = result.get("preprocessing", {})
    if set(preprocessing) != {"canon_symmetric"}:
        raise ValueError("feedback preprocessing has missing or extra fields")
    if preprocessing.get("canon_symmetric") is not True:
        raise ValueError("scalar feedback discriminator requires canonical symmetric atom slots")

    panel = result.get("domain_panel", {})
    if set(panel) != {"path", "sha256", "count", "evaluated_count"}:
        raise ValueError("feedback domain panel has missing or extra fields")
    if Path(panel.get("path", "")).resolve() != Path(domain_list_path).resolve():
        raise ValueError("result domain list path mismatch")
    if panel.get("sha256") != EXPECTED_DOMAIN_LIST_SHA256:
        raise ValueError("result domain list SHA256 mismatch")
    if int(panel.get("count", -1)) != 20 or int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("scalar feedback discriminator requires the frozen dev20 list and one domain")

    settings = result.get("settings", {})
    expected_settings = {
        "ckpt": str(checkpoint_path),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "domain_list": str(domain_list_path),
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
        "domains": 1,
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
    }
    if set(settings) != set(expected_settings) | {"output"}:
        raise ValueError("scalar feedback settings have missing or extra fields")
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected_settings.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"scalar feedback settings mismatch: {mismatches}")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError("scalar feedback discriminator requires one domain row")
    domain = domains[0]
    if set(domain) != {
        "domain",
        "residues_total",
        "residues_evaluated",
        "frames",
        "starts",
        "methods",
    }:
        raise ValueError("scalar feedback domain row has missing or extra fields")
    if domain.get("domain") != EXPECTED_DOMAIN:
        raise ValueError("scalar feedback discriminator domain mismatch")
    frames = int(domain.get("frames", -1))
    last = frames - 1 - EXPECTED_STEPS
    if last < EXPECTED_STARTS - 1:
        raise ValueError("scalar feedback discriminator frame count is too small")
    starts = domain.get("starts")
    expected_starts = [index * last // (EXPECTED_STARTS - 1) for index in range(EXPECTED_STARTS)]
    if starts != expected_starts:
        raise ValueError("scalar feedback discriminator start identity mismatch")
    methods = domain.get("methods", {})
    expected_methods = {
        "noop", "one_step_persistence", "mean", "teacher_forced_mean"
    }
    if set(methods) != expected_methods:
        raise ValueError("scalar feedback discriminator method set mismatch")
    validated = {name: _method(methods, name) for name in expected_methods}
    expected_summary = {
        name: summarize_domains(domains, name, EXPECTED_STEPS)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    if result.get("summary") != expected_summary:
        raise ValueError("scalar feedback discriminator summary mismatch")

    teacher = validated["teacher_forced_mean"]
    autoregressive = validated["mean"]
    persistence = validated["one_step_persistence"]
    teacher_rmsd = teacher["rmsd"]
    persistence_rmsd = persistence["rmsd"]
    assert isinstance(teacher_rmsd, list) and isinstance(persistence_rmsd, list)
    teacher_wins = sum(
        float(teacher_rmsd[step]) < float(persistence_rmsd[step])
        for step in range(1, EXPECTED_STEPS + 1)
    )
    teacher_first_nonphysical = _first_nonphysical(teacher)
    autoregressive_first_nonphysical = _first_nonphysical(autoregressive)

    feedback_shift = bool(
        teacher_first_nonphysical is None
        and teacher_wins >= 4
        and autoregressive_first_nonphysical is not None
        and autoregressive_first_nonphysical <= 4
    )
    endpoint_failure = bool(
        (
            teacher_first_nonphysical is not None
            and teacher_first_nonphysical <= 4
        )
        or teacher_wins <= 2
    )
    if feedback_shift:
        status = "FEEDBACK_DISTRIBUTION_SHIFT"
    elif endpoint_failure:
        status = "ENDPOINT_OPERATOR_FAILURE"
    else:
        status = "INCONCLUSIVE_SCALAR_FEEDBACK_H6"

    return {
        "status": status,
        "scope": "scalar_step2000_teacher_forced_vs_autoregressive_h1_h6_no_training",
        "checkpoint_step": 2000,
        "checkpoint_profile": PAPER_SCALAR_VALUE_PROFILE,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_source_commit": EXPECTED_SOURCE_COMMIT,
        "source_evidence_sha256": EXPECTED_SOURCE_EVIDENCE_SHA256,
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
        "domain": EXPECTED_DOMAIN,
        "starts": EXPECTED_STARTS,
        "steps": EXPECTED_STEPS,
        "teacher_forced_first_nonphysical_step": teacher_first_nonphysical,
        "autoregressive_first_nonphysical_step": autoregressive_first_nonphysical,
        "teacher_forced_steps_better_than_one_step_persistence": teacher_wins,
        "decision_rule": {
            "physical_bounds": {
                "bond_mean_inclusive": list(BOND_MEAN_RANGE),
                "bond_max_inclusive_upper": MAX_BOND_MAX,
            },
            "FEEDBACK_DISTRIBUTION_SHIFT": (
                "teacher H1-H6 all finite and physical, teacher strictly beats "
                "one-step persistence at least 4/6, and autoregressive becomes "
                "nonphysical by H4"
            ),
            "ENDPOINT_OPERATOR_FAILURE": (
                "teacher becomes nonphysical by H4 or strictly beats one-step "
                "persistence at most 2/6"
            ),
            "INCONCLUSIVE_SCALAR_FEEDBACK_H6": "neither preregistered condition is met",
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.result, args.checkpoint, args.domain_list, args.source_evidence
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
