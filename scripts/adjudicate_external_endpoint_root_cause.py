#!/usr/bin/env python
"""Fail-closed adjudication for the external endpoint root-cause probe."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scripts.adjudicate_endpoint_panel import _t_summary
from scripts.external_endpoint_identity import _sha256, verify_multidomain_checkpoint


CONTEXT_DOMAINS = [
    "1jq5A02", "1sqgA01", "2kinA00", "2q0xA01", "3u75A01",
    "3ubrA02", "4clcA00", "4uejA02", "5dwzC01",
]
LOSER_DOMAINS = {"1jq5A02", "1sqgA01", "2kinA00", "2q0xA01", "3u75A01", "4clcA00", "4uejA02"}
WITHIN_TRAIN_CROP_LOSERS = {"1jq5A02", "1sqgA01", "2kinA00", "4clcA00", "4uejA02"}
OVER_TRAIN_CROP_LOSERS = {"2q0xA01", "3u75A01"}
T_CRITICAL_6 = 2.4469118511449692
IDENTITY_ATOL = 1e-5
MASK_ATOL = 1e-5
OUTLIER_LIMIT = 5.5
CONTEXT_PANEL_SHA256 = "7ec4af135d80c94764099c201ed1e3283f8bf17579fba34aa208e477bb484573"


def classify_context(domain_shifts: dict[str, float], standard_error: float, ci_high: float) -> str:
    """Classify the preregistered context-length counterfactual."""
    loser_wins = sum(domain_shifts[domain] < 0 for domain in LOSER_DOMAINS)
    within_wins = sum(domain_shifts[domain] < 0 for domain in WITHIN_TRAIN_CROP_LOSERS)
    over_wins = sum(domain_shifts[domain] < 0 for domain in OVER_TRAIN_CROP_LOSERS)
    if standard_error > 1e-12 and ci_high < 0 and loser_wins >= 6 and within_wins >= 4:
        return "SUPPORT_DENSE_CONTEXT_LENGTH_MECHANISM"
    if over_wins == len(OVER_TRAIN_CROP_LOSERS) and within_wins <= 3:
        return "SUPPORT_TRAIN_CROP_EXTRAPOLATION_ONLY"
    return "REJECT_CONTEXT_LENGTH_MECHANISM"


def classify_outlier(
    bond_max: float,
    repeat_difference: float,
    batch_difference: float,
    fp32_fp64_difference: float,
) -> str:
    """Classify exact single-cell provenance without changing the physical limit."""
    if repeat_difference > IDENTITY_ATOL:
        return "STOP_NONDETERMINISTIC_OUTLIER_REPLAY"
    if batch_difference > IDENTITY_ATOL:
        return "STOP_BATCH_CONTEXT_COUPLING"
    if fp32_fp64_difference > IDENTITY_ATOL:
        return "STOP_BOND_PRECISION_MISMATCH"
    if bond_max > OUTLIER_LIMIT:
        return "CONFIRM_MODEL_OUTPUT_OUTLIER"
    return "STOP_OUTLIER_IDENTITY_MISMATCH"


def _close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=IDENTITY_ATOL):
        raise ValueError(f"{label} does not reproduce the frozen panel")


def _finite_triplet(values, label):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{label} must contain exactly three starts")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must be finite")
    return result


def _reference_cells(panel: dict) -> dict[tuple[str, int, int], dict]:
    return {
        (domain["domain"], int(cell["temperature"]), int(cell["replica"])): cell
        for domain in panel["domains"] for cell in domain["cells"]
    }


def adjudicate(result_path, checkpoint_path, checkpoint_sha256, reference_panel_path, reference_panel_sha256):
    checkpoint, train_fingerprint = verify_multidomain_checkpoint(checkpoint_path, checkpoint_sha256)
    if _sha256(reference_panel_path) != reference_panel_sha256:
        raise ValueError("reference panel SHA256 mismatch")
    result = json.loads(Path(result_path).read_text())
    reference = json.loads(Path(reference_panel_path).read_text())
    if result.get("scope") != "external_endpoint_root_cause_v1":
        raise ValueError("root-cause scope mismatch")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("root-cause checkpoint SHA256 mismatch")
    if result.get("checkpoint_train_fingerprint") != train_fingerprint:
        raise ValueError("root-cause train fingerprint mismatch")
    if int(result.get("checkpoint_step", -1)) != 1000 or int(checkpoint["step"]) != 1000:
        raise ValueError("root-cause probe requires checkpoint step 1000")
    if result.get("reference_panel", {}).get("sha256") != reference_panel_sha256:
        raise ValueError("recorded reference panel SHA256 mismatch")
    if reference.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reference panel checkpoint identity mismatch")
    if result.get("external_panel", {}).get("sha256") != reference.get("domain_panel", {}).get("sha256"):
        raise ValueError("external panel identity mismatch")
    if (
        result.get("context_panel", {}).get("ids") != CONTEXT_DOMAINS
        or result.get("context_panel", {}).get("sha256") != CONTEXT_PANEL_SHA256
    ):
        raise ValueError("context panel identity or order mismatch")
    settings = result.get("settings", {})
    if settings.get("context_crop") != 128 or settings.get("evaluation_crop") != 64:
        raise ValueError("context/evaluation crop mismatch")
    if settings.get("starts") != 3 or settings.get("method") != "mean" or settings.get("source_noise") is not False:
        raise ValueError("root-cause sampling settings mismatch")

    reference_cells = _reference_cells(reference)
    domains = result.get("domains")
    if not isinstance(domains, list) or [domain.get("domain") for domain in domains] != CONTEXT_DOMAINS:
        raise ValueError("root-cause domain results mismatch")
    domain_shifts = {}
    max_padding_difference = 0.0
    for domain in domains:
        domain_id = domain["domain"]
        cells = domain.get("cells")
        if not isinstance(cells, list) or len(cells) != 25:
            raise ValueError(f"domain {domain_id} requires 25 cells")
        shifts = []
        for cell in cells:
            key = (domain_id, int(cell["temperature"]), int(cell["replica"]))
            if key not in reference_cells:
                raise ValueError("root-cause cell is absent from reference panel")
            frozen = reference_cells[key]
            full_model = _finite_triplet(cell.get("full_model_rmsd_by_start"), "full model RMSD")
            full_noop = _finite_triplet(cell.get("full_noop_rmsd_by_start"), "full no-op RMSD")
            for actual, expected in zip(full_model, frozen["model_rmsd_by_start"]):
                _close(actual, expected, "full model RMSD")
            for actual, expected in zip(full_noop, frozen["noop_rmsd_by_start"]):
                _close(actual, expected, "full no-op RMSD")
            _close(cell["full_bond_mean"], frozen["bond_mean"], "full bond mean")
            _close(cell["full_bond_max"], frozen["bond_max"], "full bond max")
            full = _finite_triplet(
                cell.get("central_full_context_rmsd_by_start"), "central full-context RMSD"
            )
            crop = _finite_triplet(
                cell.get("central_crop_context_rmsd_by_start"), "central crop-context RMSD"
            )
            recorded = _finite_triplet(cell.get("crop_minus_full_by_start"), "crop-minus-full shift")
            expected_shift = [c - f for c, f in zip(crop, full)]
            if any(not math.isclose(a, e, rel_tol=0, abs_tol=1e-9) for a, e in zip(recorded, expected_shift)):
                raise ValueError("recorded crop-minus-full shift mismatch")
            shifts.extend(recorded)
            padding_difference = float(cell["padding_max_abs_prediction_difference"])
            if not math.isfinite(padding_difference) or padding_difference < 0:
                raise ValueError("padding invariance metric must be finite and non-negative")
            max_padding_difference = max(max_padding_difference, padding_difference)
        domain_shifts[domain_id] = statistics.fmean(shifts)

    loser_values = [domain_shifts[domain] for domain in CONTEXT_DOMAINS if domain in LOSER_DOMAINS]
    context_summary = _t_summary(loser_values, T_CRITICAL_6)
    context_status = classify_context(
        domain_shifts,
        float(context_summary["standard_error"]),
        float(context_summary["ci95_model_minus_noop"][1]),
    )
    if max_padding_difference > MASK_ATOL:
        context_status = "STOP_MASK_INVARIANCE_FAILURE"

    outlier = result.get("outlier", {})
    frozen_outlier = reference_cells[("1neiA00", 450, 2)]
    outlier_model = _finite_triplet(outlier.get("model_rmsd_by_start"), "outlier model RMSD")
    outlier_noop = _finite_triplet(outlier.get("noop_rmsd_by_start"), "outlier no-op RMSD")
    for actual, expected in zip(outlier_model, frozen_outlier["model_rmsd_by_start"]):
        _close(actual, expected, "outlier model RMSD")
    for actual, expected in zip(outlier_noop, frozen_outlier["noop_rmsd_by_start"]):
        _close(actual, expected, "outlier no-op RMSD")
    _close(outlier.get("bond_mean"), frozen_outlier["bond_mean"], "outlier bond mean")
    _close(outlier.get("bond_max"), frozen_outlier["bond_max"], "outlier bond max")
    per_start = outlier.get("per_start")
    if not isinstance(per_start, list) or len(per_start) != 3:
        raise ValueError("outlier probe requires three per-start records")
    fp_differences = []
    for record in per_start:
        values = [
            float(record[name]) for name in (
                "predicted_length_fp32", "predicted_length_fp64", "source_length", "target_length"
            )
        ]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("outlier bond provenance must be finite and non-negative")
        fp_differences.append(abs(values[0] - values[1]))
    outlier_status = classify_outlier(
        float(outlier["bond_max"]),
        float(outlier["repeat_max_abs_prediction_difference"]),
        float(outlier["batched_vs_individual_max_abs_prediction_difference"]),
        max(fp_differences),
    )

    return {
        "status": "ROOT_CAUSE_EVIDENCE_COMPLETE",
        "context_status": context_status,
        "outlier_status": outlier_status,
        "checkpoint_sha256": checkpoint_sha256,
        "reference_panel_sha256": reference_panel_sha256,
        "result_sha256": _sha256(result_path),
        "domain_crop_minus_full_rmsd": domain_shifts,
        "loser_domain_summary": context_summary,
        "max_padding_prediction_difference": max_padding_difference,
        "outlier": {
            "bond_max": float(outlier["bond_max"]),
            "repeat_max_abs_prediction_difference": float(outlier["repeat_max_abs_prediction_difference"]),
            "batched_vs_individual_max_abs_prediction_difference": float(
                outlier["batched_vs_individual_max_abs_prediction_difference"]
            ),
            "per_start": per_start,
        },
        "formal_training_authorized": False,
        "decision_rule": {
            "mask": "all masked-padding prediction differences must be <=1e-5",
            "dense_context": (
                "7-loser domain-balanced crop-minus-full CI upper<0, >=6/7 loser shifts<0, "
                "and >=4/5 <=256-residue loser shifts<0"
            ),
            "outlier": (
                "frozen 5.712279A aggregate reproduced within 1e-5; repeated and batched-vs-single "
                "predictions and FP32-vs-FP64 bond lengths agree within 1e-5"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--reference-panel", required=True)
    parser.add_argument("--reference-panel-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = adjudicate(
        args.result, args.checkpoint, args.checkpoint_sha256,
        args.reference_panel, args.reference_panel_sha256,
    )
    Path(args.output).write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
