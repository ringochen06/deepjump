#!/usr/bin/env python
"""Adjudicate the frozen scalar teacher-update projection discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CHECKPOINT_SHA256 = "fc5f1e7b5188af4911e518ac0e3d44c2aba4a22431360bde704465c9c1889a73"
DOMAIN_LIST_SHA256 = "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
H20_RESULT_SHA256 = "bacf07bdd93119a0b793b67335a520c468c5749d5d9da71d887d5a5fe8aa7753"
H20_DECISION_SHA256 = "03a953b4bda5e45391f7a06311eceeb84485a1a4b4a54f01edcd8aa7aea2609d"
H20_COMPLETION_SHA256 = "70a84d0e6e1bb4491ce51d89bcaf7fccab090ded2dc0415a267792e659794512"
EXPECTED_DOMAINS = ["1gxlA02", "2dgmA02", "4i9cA01"]
CALIBRATION_DOMAIN = "1gxlA02"
HELD_OUT_DOMAINS = ["2dgmA02", "4i9cA01"]
STEPS = 20
STARTS = 2
MIN_WINNING_HORIZONS = 14
MIN_MATERIAL_ALPHA_DELTA = 0.05
GAIN_MARGIN = 1e-8
RMSD_MARGIN = 1e-6
ATOL = 2e-5
MATRIX_METRICS = {
    "dot_uv_by_start", "u_sq_by_start", "v_sq_by_start",
    "cosine_by_start", "rho_by_start", "raw_gain_by_start",
    "scaled_gain_by_start", "teacher_aligned_rmsd_by_start",
    "persistence_aligned_rmsd_by_start", "scaled_aligned_rmsd_by_start",
    "scaled_bond_mean_by_start", "scaled_bond_max_by_start",
}
VECTOR_METRICS = {name.removesuffix("_by_start") for name in MATRIX_METRICS}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_bound_json(path: str | Path, expected_sha: str, name: str) -> dict:
    actual = _sha256(path)
    if actual != expected_sha:
        raise ValueError(f"{name} SHA256 mismatch")
    return json.loads(Path(path).read_text())


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _matrix(metrics: dict, name: str) -> list[list[float]]:
    rows = metrics.get(name)
    if not isinstance(rows, list) or len(rows) != STEPS:
        raise ValueError(f"{name} must contain {STEPS} horizons")
    out = []
    for horizon, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != STARTS:
            raise ValueError(f"{name} H{horizon} must contain {STARTS} starts")
        out.append([
            _finite_number(value, f"{name} H{horizon} start {start}")
            for start, value in enumerate(row)
        ])
    return out


def _vector(metrics: dict, name: str) -> list[float]:
    values = metrics.get(name)
    if not isinstance(values, list) or len(values) != STEPS:
        raise ValueError(f"{name} must contain {STEPS} horizons")
    return [_finite_number(value, f"{name} H{index}") for index, value in enumerate(values, 1)]


def _assert_close(actual: float, expected: float, name: str, atol: float = ATOL) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=atol):
        raise ValueError(f"{name} mismatch: {actual} != {expected}")


def _validate_identity(result: dict, checkpoint: str, domain_list: str) -> None:
    if set(result) != {
        "checkpoint", "checkpoint_sha256", "checkpoint_step", "settings",
        "preprocessing", "delta_frames", "domain_panel", "calibration", "domains",
    }:
        raise ValueError("result has missing or extra fields")
    if result.get("checkpoint") != checkpoint:
        raise ValueError("checkpoint path mismatch")
    if result.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("checkpoint identity mismatch")
    if result.get("checkpoint_step") != 2000:
        raise ValueError("checkpoint step mismatch")
    settings = result.get("settings")
    expected_settings = {
        "ckpt": checkpoint,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "domain_list": domain_list,
        "domain_list_sha256": DOMAIN_LIST_SHA256,
        "domains": 3,
        "starts": STARTS,
        "steps": STEPS,
        "calibration_domain": CALIBRATION_DOMAIN,
    }
    if not isinstance(settings, dict):
        raise ValueError("settings missing")
    if set(settings) != set(expected_settings) | {"output"}:
        raise ValueError("settings has missing or extra fields")
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(f"settings.{key} mismatch")
    if result.get("delta_frames") != 1:
        raise ValueError("delta mismatch")
    if result.get("preprocessing") != {
        "canon_symmetric": True,
        "target_alignment": "full_structure_target_to_source_before_crop",
        "update_translation": "per_crop_update_mean_removed",
    }:
        raise ValueError("preprocessing mismatch")
    panel = result.get("domain_panel")
    if panel != {
        "path": domain_list,
        "sha256": DOMAIN_LIST_SHA256,
        "count": 20,
        "evaluated_count": 3,
    }:
        raise ValueError("domain panel mismatch")
    calibration = result.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != {
        "domain", "alpha", "formula",
    }:
        raise ValueError("calibration has missing or extra fields")


def adjudicate(
    result_path: str | Path,
    checkpoint: str | Path,
    domain_list: str | Path,
    h20_result_path: str | Path,
    h20_decision_path: str | Path,
    h20_completion_path: str | Path,
) -> dict:
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if _sha256(domain_list) != DOMAIN_LIST_SHA256:
        raise ValueError("domain list SHA256 mismatch")
    h20_result = _load_bound_json(h20_result_path, H20_RESULT_SHA256, "H20 result")
    h20_decision = _load_bound_json(h20_decision_path, H20_DECISION_SHA256, "H20 decision")
    h20_completion = _load_bound_json(
        h20_completion_path, H20_COMPLETION_SHA256, "H20 completion"
    )
    if h20_decision.get("status") != "ENDPOINT_OPERATOR_FAILURE_H20":
        raise ValueError("unexpected H20 decision status")
    if h20_completion.get("status") != "OBS_DOUBLE_READBACK_PASS":
        raise ValueError("H20 readback is not complete")
    if h20_completion.get("archived_decision_sha256") != H20_DECISION_SHA256:
        raise ValueError("H20 completion does not bind the archived decision")

    result = json.loads(Path(result_path).read_text())
    _validate_identity(result, str(checkpoint), str(domain_list))
    rows = result.get("domains")
    if not isinstance(rows, list) or [row.get("domain") for row in rows] != EXPECTED_DOMAINS:
        raise ValueError("frozen domain order mismatch")
    old_rows = h20_result.get("domains")
    if [row.get("domain") for row in old_rows] != EXPECTED_DOMAINS:
        raise ValueError("bound H20 domain order mismatch")

    domain_reports = []
    all_domain_dots = []
    calibration_dot = []
    calibration_u_sq = []
    alpha_reported = _finite_number(
        result.get("calibration", {}).get("alpha"), "calibration alpha"
    )
    if result.get("calibration", {}).get("domain") != CALIBRATION_DOMAIN:
        raise ValueError("calibration domain mismatch")
    if result.get("calibration", {}).get("formula") != "sum(dot_uv)/sum(u_sq)":
        raise ValueError("calibration formula mismatch")

    validated = []
    for row, old_row in zip(rows, old_rows):
        if set(row) != {
            "domain", "residues_total", "residues_evaluated", "frames", "starts",
            "metrics",
        }:
            raise ValueError(f"{row.get('domain')} row has missing or extra fields")
        if row.get("starts") != old_row.get("starts"):
            raise ValueError(f"{row['domain']} start identities drifted from H20")
        for key in ("residues_total", "residues_evaluated", "frames"):
            if row.get(key) != old_row.get(key):
                raise ValueError(f"{row['domain']} {key} drifted from H20")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{row['domain']} metrics missing")
        if set(metrics) != MATRIX_METRICS | VECTOR_METRICS:
            raise ValueError(f"{row['domain']} metrics have missing or extra fields")
        matrices = {
            name: _matrix(metrics, name)
            for name in MATRIX_METRICS
        }
        vectors = {
            name: _vector(metrics, name)
            for name in VECTOR_METRICS
        }
        for name, matrix in matrices.items():
            aggregate_name = name.removesuffix("_by_start")
            if aggregate_name not in vectors:
                continue
            for horizon, values in enumerate(matrix):
                expected = max(values) if aggregate_name == "scaled_bond_max" else float(np.mean(values))
                _assert_close(
                    vectors[aggregate_name][horizon], expected,
                    f"{row['domain']} {aggregate_name} H{horizon + 1}",
                )
        for horizon in range(STEPS):
            for start in range(STARTS):
                dot = matrices["dot_uv_by_start"][horizon][start]
                u_sq = matrices["u_sq_by_start"][horizon][start]
                v_sq = matrices["v_sq_by_start"][horizon][start]
                if u_sq < 0 or v_sq <= 0:
                    raise ValueError("update squared norms must be non-negative and target-positive")
                if abs(dot) > math.sqrt(u_sq * v_sq) + ATOL:
                    raise ValueError("update dot product violates the Cauchy bound")
                expected_raw = 2 * dot - u_sq
                _assert_close(
                    matrices["raw_gain_by_start"][horizon][start], expected_raw,
                    f"{row['domain']} raw gain identity",
                )
                expected_scaled = 2 * alpha_reported * dot - alpha_reported ** 2 * u_sq
                _assert_close(
                    matrices["scaled_gain_by_start"][horizon][start], expected_scaled,
                    f"{row['domain']} scaled gain identity",
                )
                expected_cosine = (
                    dot / math.sqrt(u_sq * v_sq) if u_sq > 1e-12 else 0.0
                )
                expected_rho = math.sqrt(u_sq / v_sq)
                _assert_close(
                    matrices["cosine_by_start"][horizon][start], expected_cosine,
                    f"{row['domain']} cosine identity",
                )
                _assert_close(
                    matrices["rho_by_start"][horizon][start], expected_rho,
                    f"{row['domain']} rho identity",
                )
                cosine = matrices["cosine_by_start"][horizon][start]
                rho = matrices["rho_by_start"][horizon][start]
                if not -1.0 - ATOL <= cosine <= 1.0 + ATOL or rho < 0:
                    raise ValueError("cosine or norm ratio is outside its physical range")
                for name in (
                    "teacher_aligned_rmsd_by_start",
                    "persistence_aligned_rmsd_by_start",
                    "scaled_aligned_rmsd_by_start",
                    "scaled_bond_mean_by_start",
                    "scaled_bond_max_by_start",
                ):
                    if matrices[name][horizon][start] < 0:
                        raise ValueError(f"{name} must be non-negative")
                if (
                    matrices["scaled_bond_max_by_start"][horizon][start]
                    < matrices["scaled_bond_mean_by_start"][horizon][start]
                ):
                    raise ValueError("scaled bond maximum cannot be below its mean")
                old_teacher = old_row["methods"]["teacher_forced_mean"]["rmsd_by_start"][horizon + 1][start]
                old_persistence = old_row["methods"]["one_step_persistence"]["rmsd_by_start"][horizon + 1][start]
                _assert_close(
                    matrices["teacher_aligned_rmsd_by_start"][horizon][start],
                    old_teacher,
                    f"{row['domain']} teacher RMSD H{horizon + 1} start {start}",
                )
                _assert_close(
                    matrices["persistence_aligned_rmsd_by_start"][horizon][start],
                    old_persistence,
                    f"{row['domain']} persistence RMSD H{horizon + 1} start {start}",
                )
        flattened_dot = [value for horizon in matrices["dot_uv_by_start"] for value in horizon]
        all_domain_dots.append(float(np.mean(flattened_dot)))
        if row["domain"] == CALIBRATION_DOMAIN:
            calibration_dot = flattened_dot
            calibration_u_sq = [
                value for horizon in matrices["u_sq_by_start"] for value in horizon
            ]
        validated.append((row, matrices, vectors))

    denominator = float(sum(calibration_u_sq))
    numerator = float(sum(calibration_dot))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("calibration denominator is not informative")
    alpha = numerator / denominator
    _assert_close(alpha_reported, alpha, "calibration alpha", atol=1e-8)

    held_out_pass = True
    all_geometry_physical = True
    for row, matrices, vectors in validated:
        gain_matrix = np.asarray(matrices["scaled_gain_by_start"], dtype=float)
        scaled_rmsd = np.asarray(matrices["scaled_aligned_rmsd_by_start"], dtype=float)
        teacher_rmsd = np.asarray(
            matrices["teacher_aligned_rmsd_by_start"], dtype=float
        )
        persistence_rmsd = np.asarray(
            matrices["persistence_aligned_rmsd_by_start"], dtype=float
        )
        start_mean_gain = gain_matrix.mean(axis=0).tolist()
        gain_wins_by_start = (gain_matrix > GAIN_MARGIN).sum(axis=0).tolist()
        winning_gain_horizons = int(
            (gain_matrix.mean(axis=1) > GAIN_MARGIN).sum()
        )
        persistence_wins_by_start = (
            scaled_rmsd + RMSD_MARGIN < persistence_rmsd
        ).sum(axis=0).tolist()
        winning_rmsd_horizons = int(
            (
                scaled_rmsd.mean(axis=1) + RMSD_MARGIN
                < persistence_rmsd.mean(axis=1)
            ).sum()
        )
        raw_improvement_by_start = (
            scaled_rmsd + RMSD_MARGIN < teacher_rmsd
        ).sum(axis=0).tolist()
        raw_improvement_horizons = int(
            (
                scaled_rmsd.mean(axis=1) + RMSD_MARGIN
                < teacher_rmsd.mean(axis=1)
            ).sum()
        )
        raw_improves_by_start_mean = (
            scaled_rmsd.mean(axis=0) + RMSD_MARGIN < teacher_rmsd.mean(axis=0)
        ).tolist()
        raw_improves_domain_mean = bool(
            scaled_rmsd.mean() + RMSD_MARGIN < teacher_rmsd.mean()
        )
        domain_mean_gain = float(gain_matrix.mean())
        geometry_physical = all(
            3.2 <= mean <= 4.5 and maximum <= 5.5
            for means, maxima in zip(
                matrices["scaled_bond_mean_by_start"],
                matrices["scaled_bond_max_by_start"],
            )
            for mean, maximum in zip(means, maxima)
        )
        all_geometry_physical = all_geometry_physical and geometry_physical
        is_held_out = row["domain"] in HELD_OUT_DOMAINS
        domain_pass = (
            all(value > GAIN_MARGIN for value in start_mean_gain)
            and domain_mean_gain > GAIN_MARGIN
            and all(value >= MIN_WINNING_HORIZONS for value in gain_wins_by_start)
            and winning_gain_horizons >= MIN_WINNING_HORIZONS
            and all(value >= MIN_WINNING_HORIZONS for value in persistence_wins_by_start)
            and winning_rmsd_horizons >= MIN_WINNING_HORIZONS
            and all(value >= MIN_WINNING_HORIZONS for value in raw_improvement_by_start)
            and raw_improvement_horizons >= MIN_WINNING_HORIZONS
            and all(raw_improves_by_start_mean)
            and raw_improves_domain_mean
            and geometry_physical
        )
        if is_held_out:
            held_out_pass = held_out_pass and domain_pass
        domain_reports.append({
            "domain": row["domain"],
            "role": "calibration" if row["domain"] == CALIBRATION_DOMAIN else "held_out",
            "mean_dot_uv": all_domain_dots[len(domain_reports)],
            "mean_scaled_gain": domain_mean_gain,
            "mean_scaled_gain_by_start": start_mean_gain,
            "scaled_gain_winning_horizons_by_start": gain_wins_by_start,
            "scaled_gain_winning_horizons": winning_gain_horizons,
            "scaled_aligned_rmsd_winning_horizons_by_start": persistence_wins_by_start,
            "scaled_aligned_rmsd_winning_horizons": winning_rmsd_horizons,
            "scaled_improves_raw_horizons_by_start": raw_improvement_by_start,
            "scaled_improves_raw_horizons": raw_improvement_horizons,
            "scaled_improves_raw_by_start_mean": raw_improves_by_start_mean,
            "scaled_improves_raw_domain_mean": raw_improves_domain_mean,
            "scaled_geometry_physical": geometry_physical,
            "held_out_pass": domain_pass if is_held_out else None,
        })

    directional_domains = int(sum(value <= 0 for value in all_domain_dots))
    if directional_domains >= 2:
        status = "DIRECTIONAL_CA_ENDPOINT_FAILURE"
    elif (
        alpha > 0
        and abs(alpha - 1.0) >= MIN_MATERIAL_ALPHA_DELTA
        and held_out_pass
        and all_geometry_physical
    ):
        status = "POSITIVE_SCALAR_RESCALE_SIGNAL"
    else:
        status = "INCONCLUSIVE_PROJECTION"
    return {
        "status": status,
        "scope": "scalar_step2000_teacher_update_projection_three_domain_h20_no_training",
        "checkpoint_step": 2000,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "domain_list_sha256": DOMAIN_LIST_SHA256,
        "h20_result_sha256": H20_RESULT_SHA256,
        "h20_decision_sha256": H20_DECISION_SHA256,
        "h20_completion_sha256": H20_COMPLETION_SHA256,
        "calibration_domain": CALIBRATION_DOMAIN,
        "held_out_domains": HELD_OUT_DOMAINS,
        "alpha": alpha,
        "directional_failure_domains": directional_domains,
        "domain_evidence": domain_reports,
        "decision_rule": {
            "POSITIVE_SCALAR_RESCALE_SIGNAL": "alpha > 0 and differs from 1 by at least 0.05; both held-out domains exceed the frozen 1e-8 gain and 1e-6 A RMSD margins in mean and at least 14/20 horizons for each start and their mean versus persistence and raw teacher; all scaled endpoints remain physical",
            "DIRECTIONAL_CA_ENDPOINT_FAILURE": "at least two domains have mean CA dot(update_pred, update_true) <= 0",
            "INCONCLUSIVE_PROJECTION": "neither preregistered condition is met",
        },
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--domain-list", required=True)
    ap.add_argument("--h20-result", required=True)
    ap.add_argument("--h20-decision", required=True)
    ap.add_argument("--h20-completion", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report = adjudicate(
        args.result,
        args.checkpoint,
        args.domain_list,
        args.h20_result,
        args.h20_decision,
        args.h20_completion,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
