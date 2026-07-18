#!/usr/bin/env python
"""Adjudicate the matched full-tensor versus frozen vector-only discriminator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.adjudicate_paper_loss_continuation import (
    MAX_BOND_MAX,
    MAX_BOND_MEAN,
    MIN_BOND_MEAN,
    adjudicate as adjudicate_absolute_gate,
)


K1_MIN_RMSD_IMPROVEMENT_ANGSTROM = 0.01
H20_MIN_RELATIVE_RMSD_IMPROVEMENT = 0.20
H20_MIN_RELATIVE_GEOMETRY_EXCESS_IMPROVEMENT = 0.50
MATCHED_SETTING_KEYS = (
    "domain_list_sha256",
    "domains",
    "starts",
    "steps",
    "methods",
    "seed",
    "noise_sigma",
    "integrator",
    "tau_max",
    "terminal_denoise",
    "drift_anchor",
)


def _finite(value: object, label: str) -> float:
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


def _validate_matched_panel(full: dict, vector: dict) -> None:
    for label, result in (("full", full), ("vector", vector)):
        if result.get("checkpoint_step") != 2000:
            raise ValueError(f"{label}: checkpoint_step must be 2000")
        if result.get("delta_frames") != 1:
            raise ValueError(f"{label}: delta_frames must be 1")
    if full.get("domain_panel") != vector.get("domain_panel"):
        raise ValueError("full/vector domain_panel mismatch")
    full_settings = full.get("settings")
    vector_settings = vector.get("settings")
    if not isinstance(full_settings, dict) or not isinstance(vector_settings, dict):
        raise ValueError("full/vector settings must be dictionaries")
    full_matched = {key: full_settings.get(key) for key in MATCHED_SETTING_KEYS}
    vector_matched = {key: vector_settings.get(key) for key in MATCHED_SETTING_KEYS}
    if full_matched != vector_matched:
        raise ValueError("full/vector matched settings mismatch")

    full_domains = full.get("domains")
    vector_domains = vector.get("domains")
    if not isinstance(full_domains, list) or len(full_domains) != 3:
        raise ValueError("full: expected exactly 3 domain records")
    if not isinstance(vector_domains, list) or len(vector_domains) != 3:
        raise ValueError("vector: expected exactly 3 domain records")
    for full_domain, vector_domain in zip(full_domains, vector_domains, strict=True):
        if full_domain.get("domain") != vector_domain.get("domain"):
            raise ValueError("full/vector domain order mismatch")
        if full_domain.get("starts") != vector_domain.get("starts"):
            raise ValueError("full/vector starts mismatch")


def _mean_curve(result: dict, method: str, metric: str) -> list[float]:
    curves = []
    for domain in result["domains"]:
        values = domain.get("methods", {}).get(method, {}).get(metric)
        if not isinstance(values, list) or len(values) != 21:
            raise ValueError(f"{domain.get('domain')} {method} {metric}: expected 21 values")
        curves.append(
            [_finite(value, f"{domain.get('domain')} {method} {metric}") for value in values]
        )
    return [sum(values) / len(values) for values in zip(*curves, strict=True)]


def adjudicate(
    history_path: Path,
    rollout_dir: Path,
    vector_baseline_path: Path,
    domain_sha256: str,
) -> dict:
    absolute = adjudicate_absolute_gate(history_path, rollout_dir, domain_sha256)
    full = json.loads((rollout_dir / "rollout_2000.json").read_text())
    vector = json.loads(vector_baseline_path.read_text())
    _validate_matched_panel(full, vector)

    for metric in ("rmsd", "bond_mean", "bond_max"):
        full_noop = _mean_curve(full, "noop", metric)
        vector_noop = _mean_curve(vector, "noop", metric)
        if any(abs(left - right) > 1e-6 for left, right in zip(full_noop, vector_noop)):
            raise ValueError(f"full/vector no-op {metric} curves mismatch")

    selected = absolute.get("selected_method")
    comparison = None
    relative_pass = False
    if selected in ("mean", "ode_1"):
        full_rmsd = _mean_curve(full, selected, "rmsd")
        vector_rmsd = _mean_curve(vector, selected, "rmsd")
        full_bond_mean = _mean_curve(full, selected, "bond_mean")
        vector_bond_mean = _mean_curve(vector, selected, "bond_mean")
        full_bond_max = _mean_curve(full, selected, "bond_max")
        vector_bond_max = _mean_curve(vector, selected, "bond_max")
        full_excess = _geometry_excess(full_bond_mean[-1], full_bond_max[-1])
        vector_excess = _geometry_excess(vector_bond_mean[-1], vector_bond_max[-1])
        k1_improvement = vector_rmsd[1] - full_rmsd[1]
        if vector_rmsd[-1] <= 0:
            raise ValueError("vector H20 RMSD must be positive")
        h20_relative_rmsd_improvement = (
            vector_rmsd[-1] - full_rmsd[-1]
        ) / vector_rmsd[-1]
        if vector_excess > 0:
            h20_relative_geometry_improvement = (vector_excess - full_excess) / vector_excess
        else:
            h20_relative_geometry_improvement = 1.0 if full_excess == 0 else -math.inf
        relative_pass = (
            k1_improvement >= K1_MIN_RMSD_IMPROVEMENT_ANGSTROM
            and h20_relative_rmsd_improvement >= H20_MIN_RELATIVE_RMSD_IMPROVEMENT
            and h20_relative_geometry_improvement
            >= H20_MIN_RELATIVE_GEOMETRY_EXCESS_IMPROVEMENT
        )
        comparison = {
            "method": selected,
            "k1_rmsd": {"full": full_rmsd[1], "vector": vector_rmsd[1]},
            "k1_rmsd_improvement_angstrom": k1_improvement,
            "h20_rmsd": {"full": full_rmsd[-1], "vector": vector_rmsd[-1]},
            "h20_relative_rmsd_improvement": h20_relative_rmsd_improvement,
            "h20_geometry_excess": {"full": full_excess, "vector": vector_excess},
            "h20_relative_geometry_excess_improvement": h20_relative_geometry_improvement,
            "passes": relative_pass,
        }

    if absolute.get("status") == "GO_BOUNDED_EXTENSION" and relative_pass:
        status = "GO_STRICT_INTEGRATION"
        reason = "absolute H20 gate and preregistered vector-only comparison both pass"
    else:
        status = "STOP_ARCHITECTURE_PAIR"
        reason = "absolute H20 gate or preregistered vector-only comparison did not pass"

    return {
        "status": status,
        "scope": "bounded_full_tensor_discriminator_only",
        "formal_training_authorized": False,
        "reason": reason,
        "absolute_gate": absolute,
        "relative_vector_only_comparison": comparison,
        "relative_decision_rule": {
            "step": 2000,
            "selected_method": "method selected by the absolute gate",
            "k1_min_rmsd_improvement_angstrom": K1_MIN_RMSD_IMPROVEMENT_ANGSTROM,
            "h20_min_relative_rmsd_improvement": H20_MIN_RELATIVE_RMSD_IMPROVEMENT,
            "h20_min_relative_geometry_excess_improvement": H20_MIN_RELATIVE_GEOMETRY_EXCESS_IMPROVEMENT,
            "all_conditions_required": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--vector-baseline", required=True, type=Path)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = adjudicate(
        args.history,
        args.rollout_dir,
        args.vector_baseline,
        args.domain_list_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
