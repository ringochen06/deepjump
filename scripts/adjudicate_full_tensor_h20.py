#!/usr/bin/env python
"""Fail-closed H1-H20 discriminator for the exact full-tensor paper-horizon checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path

from scripts.external_endpoint_identity import verify_multidomain_checkpoint
from scripts.guarded_endpoint_panel_eval import (
    PAPER_HORIZON_PROFILE,
    checkpoint_profile_requirements,
)
from scripts.rollout_robustness_eval import summarize_domains


EXPECTED_CHECKPOINT_SHA256 = (
    "fb12d776b106867ca14a8f56476daf776a6296b6dca640f03c2188a75a69bb47"
)
EXPECTED_SOURCE_DECISION_SHA256 = (
    "2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38"
)
EXPECTED_SOURCE_RUNNER_SHA256 = (
    "2c8eedad191a814080303b6a30204fbb9bee522937c3a0cb5087e3439b6bd75f"
)
EXPECTED_SOURCE_STATUS = "STOP_PAPER_HORIZON_OBJECTIVE_GAIN"
EXPECTED_SOURCE_COMMIT = "dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b"
EXPECTED_CONFIG_SHA256 = (
    "506237167a3921bdf4dfe795fccee1b79bd8628aa89f3c28b5677301097a9898"
)
EXPECTED_HISTORY_SHA256 = (
    "868e3e44386163e61e61f6c0da60c160e3cb9f282e20c3ba7a9198208c64fa3f"
)
EXPECTED_DOMAIN_LIST_SHA256 = (
    "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
)
EXPECTED_DOMAINS = ("1gxlA02", "2dgmA02", "4i9cA01")
EXPECTED_STEPS = 20
EXPECTED_STARTS = 2
STRONG_WINS = 14
WEAK_WINS = 6
RMSD_MARGIN = 1e-6
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


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _exact_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be the integer {expected}")


def _same_json_primitive(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


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
        "rmsd", "rmsd_by_start", "fnc", "bond_mean", "bond_p95", "bond_p99",
        "bond_max", "bond_mae_real", "angle_cos_mae_real", "bond_mean_by_start",
        "bond_max_by_start",
    }
    if set(payload) != expected_metrics:
        raise ValueError(f"{name} has missing or extra metrics")
    validated: dict[str, list] = {}
    matrix_metrics = {"rmsd_by_start", "bond_mean_by_start", "bond_max_by_start"}
    for metric in expected_metrics - matrix_metrics:
        series = _finite_series(payload.get(metric), f"{name}.{metric}")
        if metric == "fnc":
            if any(value < 0.0 or value > 1.0 for value in series):
                raise ValueError(f"{name}.{metric} must be in [0, 1]")
        elif any(value < 0.0 for value in series):
            raise ValueError(f"{name}.{metric} must be nonnegative")
        validated[metric] = series
    for metric in matrix_metrics:
        values = payload.get(metric)
        if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
            raise ValueError(f"{name}.{metric} must contain H0..H{EXPECTED_STEPS}")
        if any(not isinstance(row, list) or len(row) != EXPECTED_STARTS for row in values):
            raise ValueError(f"{name}.{metric} must contain two starts per horizon")
        matrix = [[float(value) for value in row] for row in values]
        if not all(math.isfinite(value) and value >= 0.0 for row in matrix for value in row):
            raise ValueError(f"{name}.{metric} must be finite and nonnegative")
        validated[metric] = matrix
    for step in range(EXPECTED_STEPS + 1):
        rmsd = math.fsum(validated["rmsd_by_start"][step]) / EXPECTED_STARTS
        bond_mean = math.fsum(validated["bond_mean_by_start"][step]) / EXPECTED_STARTS
        bond_max = max(validated["bond_max_by_start"][step])
        if not math.isclose(validated["rmsd"][step], rmsd, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name}.rmsd H{step} does not match rmsd_by_start")
        if not math.isclose(validated["bond_mean"][step], bond_mean, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{name}.bond_mean H{step} does not match per-start geometry")
        if not math.isclose(validated["bond_max"][step], bond_max, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{name}.bond_max H{step} does not match per-start geometry")
        for start in range(EXPECTED_STARTS):
            if validated["bond_max_by_start"][step][start] < validated["bond_mean_by_start"][step][start]:
                raise ValueError(f"{name} bond_max must be >= bond_mean per start")
    return validated


def _require_source_decision(path: str | Path) -> dict:
    if not hmac.compare_digest(_sha256(path), EXPECTED_SOURCE_DECISION_SHA256):
        raise ValueError("source decision SHA256 mismatch")
    payload = _load_object(path, "source decision")
    expected = {
        "status": EXPECTED_SOURCE_STATUS,
        "candidate_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"source decision {key} mismatch")
    for key in (
        "external_development_authorized",
        "second_seed_scientifically_eligible",
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"source decision {key} must be false")
    return payload


def _require_source_runner(path: str | Path) -> None:
    if not hmac.compare_digest(_sha256(path), EXPECTED_SOURCE_RUNNER_SHA256):
        raise ValueError("source runner SHA256 mismatch")
    runner = Path(path).read_text()
    gate = 'if [[ "$training_ab_status" == ADVANCE_PAPER_HORIZON_EXTERNAL20 ]]; then'
    mkdir = 'mkdir -p "$EXTERNAL_DATA_ROOT"'
    download = '"$PYTHON" scripts/download_mdcath.py'
    stop_copy = 'cp "$RUN_DIR/training_ab_decision.json" "$RUN_DIR/decision.json"'
    positions = [runner.index(fragment) for fragment in (gate, mkdir, download, stop_copy)]
    if positions != sorted(positions):
        raise ValueError("source runner external gate order mismatch")


def _first_nonphysical(method: dict[str, list]) -> int | None:
    for step in range(1, EXPECTED_STEPS + 1):
        for start in range(EXPECTED_STARTS):
            mean = float(method["bond_mean_by_start"][step][start])
            maximum = float(method["bond_max_by_start"][step][start])
            if not (
                math.isfinite(mean)
                and math.isfinite(maximum)
                and BOND_MEAN_RANGE[0] <= mean <= BOND_MEAN_RANGE[1]
                and 0.0 <= maximum <= MAX_BOND_MAX
                and maximum >= mean
            ):
                return step
    return None


def _validate_result(
    result_path: str | Path,
    checkpoint_path: str | Path,
    domain_list_path: str | Path,
    expected_result_output_path: str | Path | None = None,
) -> tuple[dict, list[dict[str, dict[str, list]]]]:
    result = _load_object(result_path, "full-tensor H20 result")
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
        raise ValueError("full-tensor H20 result has missing or extra fields")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("result checkpoint path mismatch")
    if result.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("result checkpoint SHA256 mismatch")
    _exact_int(result.get("checkpoint_step"), 2000, "checkpoint_step")
    _exact_int(result.get("delta_frames"), 1, "delta_frames")
    preprocessing = result.get("preprocessing")
    if (
        not isinstance(preprocessing, dict)
        or set(preprocessing) != {"canon_symmetric"}
        or preprocessing.get("canon_symmetric") is not True
    ):
        raise ValueError("full-tensor H20 requires canonical symmetric atom slots")

    panel = result.get("domain_panel", {})
    if set(panel) != {"path", "sha256", "count", "evaluated_count"}:
        raise ValueError("domain panel has missing or extra fields")
    if Path(panel.get("path", "")).resolve() != Path(domain_list_path).resolve():
        raise ValueError("result domain list path mismatch")
    if panel.get("sha256") != EXPECTED_DOMAIN_LIST_SHA256:
        raise ValueError("result domain list SHA256 mismatch")
    _exact_int(panel.get("count"), 20, "domain_panel.count")
    _exact_int(panel.get("evaluated_count"), 3, "domain_panel.evaluated_count")

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
        raise ValueError("full-tensor H20 settings have missing or extra fields")
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected_settings.items()
        if not _same_json_primitive(settings.get(key), value)
    }
    if mismatches:
        raise ValueError(f"full-tensor H20 settings mismatch: {mismatches}")
    output = settings.get("output")
    expected_output = (
        Path(expected_result_output_path)
        if expected_result_output_path is not None
        else Path(result_path)
    )
    if not isinstance(output, str) or Path(output).resolve() != expected_output.resolve():
        raise ValueError("full-tensor H20 settings output path mismatch")

    domains = result.get("domains")
    if not isinstance(domains, list) or len(domains) != len(EXPECTED_DOMAINS):
        raise ValueError("full-tensor H20 requires three domain rows")
    expected_methods = {"noop", "one_step_persistence", "mean", "teacher_forced_mean"}
    validated_domains: list[dict[str, dict[str, list]]] = []
    for row, expected_domain in zip(domains, EXPECTED_DOMAINS, strict=True):
        if set(row) != {
            "domain", "residues_total", "residues_evaluated", "frames", "starts", "methods"
        }:
            raise ValueError("full-tensor H20 domain row has missing or extra fields")
        if row.get("domain") != expected_domain:
            raise ValueError("full-tensor H20 domain identity mismatch")
        residues_total = _positive_int(row.get("residues_total"), "residues_total")
        residues_evaluated = _positive_int(row.get("residues_evaluated"), "residues_evaluated")
        if residues_evaluated != min(residues_total, 256):
            raise ValueError("residues_evaluated does not match frozen crop length")
        frames = _positive_int(row.get("frames"), "frames")
        last = frames - 1 - EXPECTED_STEPS
        if last < EXPECTED_STARTS - 1 or row.get("starts") != [0, last]:
            raise ValueError("full-tensor H20 start identity mismatch")
        methods = row.get("methods", {})
        if set(methods) != expected_methods:
            raise ValueError("full-tensor H20 method set mismatch")
        validated_domains.append({name: _method(methods, name) for name in expected_methods})

    expected_summary = {
        name: summarize_domains(domains, name, EXPECTED_STEPS)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    if result.get("summary") != expected_summary:
        raise ValueError("full-tensor H20 summary mismatch")
    return result, validated_domains


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    domain_list_path: str | Path,
    source_decision_path: str | Path,
    source_runner_path: str | Path,
    *,
    expected_result_output_path: str | Path | None = None,
) -> dict:
    """Verify frozen identities and classify exact full-tensor H20 endpoints."""
    _require_source_decision(source_decision_path)
    _require_source_runner(source_runner_path)
    if not hmac.compare_digest(_sha256(checkpoint_path), EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("checkpoint SHA256 mismatch")
    if not hmac.compare_digest(_sha256(domain_list_path), EXPECTED_DOMAIN_LIST_SHA256):
        raise ValueError("domain list SHA256 mismatch")
    expected_data, expected_model, expected_train = checkpoint_profile_requirements(
        PAPER_HORIZON_PROFILE, EXPECTED_CHECKPOINT_SHA256
    )
    verify_multidomain_checkpoint(
        checkpoint_path,
        EXPECTED_CHECKPOINT_SHA256,
        expected_step=2000,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )
    _, validated_domains = _validate_result(
        result_path,
        checkpoint_path,
        domain_list_path,
        expected_result_output_path,
    )

    domain_evidence = []
    for domain, validated in zip(EXPECTED_DOMAINS, validated_domains, strict=True):
        teacher = validated["teacher_forced_mean"]
        autoregressive = validated["mean"]
        persistence = validated["one_step_persistence"]
        wins_by_start = [
            sum(
                float(teacher["rmsd_by_start"][step][start]) + RMSD_MARGIN
                < float(persistence["rmsd_by_start"][step][start])
                for step in range(1, EXPECTED_STEPS + 1)
            )
            for start in range(EXPECTED_STARTS)
        ]
        aggregate_wins = sum(
            float(teacher["rmsd"][step]) + RMSD_MARGIN
            < float(persistence["rmsd"][step])
            for step in range(1, EXPECTED_STEPS + 1)
        )
        teacher_nonphysical = _first_nonphysical(teacher)
        domain_evidence.append({
            "domain": domain,
            "teacher_forced_first_nonphysical_step": teacher_nonphysical,
            "autoregressive_first_nonphysical_step": _first_nonphysical(autoregressive),
            "teacher_forced_steps_better_than_one_step_persistence_by_start": wins_by_start,
            "teacher_forced_steps_better_than_one_step_persistence_aggregate": aggregate_wins,
            "teacher_forced_strong": teacher_nonphysical is None
            and aggregate_wins >= STRONG_WINS
            and all(wins >= STRONG_WINS for wins in wins_by_start),
            "teacher_forced_failure": teacher_nonphysical is not None
            or aggregate_wins <= WEAK_WINS
            or any(wins <= WEAK_WINS for wins in wins_by_start),
        })

    strong_domains = sum(row["teacher_forced_strong"] for row in domain_evidence)
    failure_domains = sum(row["teacher_forced_failure"] for row in domain_evidence)
    if strong_domains == 3:
        status = "FULL_TENSOR_ENDPOINT_SIGNAL_H20"
    elif failure_domains >= 2:
        status = "FULL_TENSOR_ENDPOINT_FAILURE_H20"
    else:
        status = "INCONCLUSIVE_FULL_TENSOR_H20"

    return {
        "status": status,
        "scope": "exact_full_tensor_paper_horizon_step2000_teacher_forced_h1_h20_three_domain_no_training",
        "source_status": EXPECTED_SOURCE_STATUS,
        "source_decision_sha256": EXPECTED_SOURCE_DECISION_SHA256,
        "source_runner_sha256": EXPECTED_SOURCE_RUNNER_SHA256,
        "checkpoint_step": 2000,
        "checkpoint_profile": PAPER_HORIZON_PROFILE,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_source_commit": EXPECTED_SOURCE_COMMIT,
        "checkpoint_config_sha256": EXPECTED_CONFIG_SHA256,
        "checkpoint_history_sha256": EXPECTED_HISTORY_SHA256,
        "domain_list_sha256": EXPECTED_DOMAIN_LIST_SHA256,
        "domains": list(EXPECTED_DOMAINS),
        "starts_per_domain": EXPECTED_STARTS,
        "steps": EXPECTED_STEPS,
        "domain_evidence": domain_evidence,
        "teacher_strong_domains": strong_domains,
        "teacher_failure_domains": failure_domains,
        "decision_rule": {
            "rmsd_strict_margin_angstrom": RMSD_MARGIN,
            "physical_bounds": {
                "bond_mean_inclusive": list(BOND_MEAN_RANGE),
                "bond_max_inclusive_upper": MAX_BOND_MAX,
            },
            "FULL_TENSOR_ENDPOINT_SIGNAL_H20": (
                "all three domains remain teacher-forced physical and both starts plus their "
                "aggregate in each strictly beat one-step persistence on at least 14/20 horizons"
            ),
            "FULL_TENSOR_ENDPOINT_FAILURE_H20": (
                "at least two domains become teacher-forced nonphysical or have a start or "
                "aggregate that strictly beats one-step persistence on at most 6/20 horizons"
            ),
            "INCONCLUSIVE_FULL_TENSOR_H20": "neither preregistered condition is met",
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
    parser.add_argument("--source-decision", required=True, type=Path)
    parser.add_argument("--source-runner", required=True, type=Path)
    parser.add_argument("--expected-result-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.checkpoint,
        args.domain_list,
        args.source_decision,
        args.source_runner,
        expected_result_output_path=args.expected_result_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
