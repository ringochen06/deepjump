#!/usr/bin/env python
"""Fail-closed adjudication for the frozen step-2000 single-cell replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from deepjump.atom_constants import NUM_RESIDUE_TYPES
from scripts.adjudicate_source_law_candidate import MAX_BOND_MAX
from scripts.external_endpoint_identity import _sha256, load_disjoint_panels, verify_multidomain_checkpoint
from scripts.external_step2000_outlier_replay import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_STEP,
    EXPECTED_EXTERNAL_PANEL_SHA256,
    EXPECTED_REFERENCE_PANEL_SHA256,
    EXPECTED_TRAIN_SEED,
    OUTLIER_DOMAIN,
    OUTLIER_REPLICA,
    OUTLIER_TEMPERATURE,
    SCOPE,
    _json_sha256,
    _reference_cell,
)


IDENTITY_ATOL = 1e-5


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _close(actual: object, expected: object, label: str, *, atol: float = IDENTITY_ATOL) -> None:
    if not math.isclose(_finite(actual, label), _finite(expected, label), rel_tol=0, abs_tol=atol):
        raise ValueError(f"{label} does not reproduce the frozen panel")


def _vector(value: object, label: str, length: int = 3) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} finite values")
    return [_finite(item, label) for item in value]


def _distance(points: object, label: str) -> float:
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError(f"{label} must contain two endpoint positions")
    endpoints = [_vector(point, label, length=3) for point in points]
    return math.sqrt(sum((right - left) ** 2 for left, right in zip(*endpoints)))


def _validate_topology(topology: dict) -> tuple[list[int], list[bool], set[int]]:
    residues = int(topology.get("residues", -1))
    residue_type_ids = topology.get("residue_type_ids")
    bond_mask = topology.get("bond_mask")
    if (
        residues < 2
        or not isinstance(residue_type_ids, list)
        or len(residue_type_ids) != residues
    ):
        raise ValueError("residue type topology shape mismatch")
    if not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < NUM_RESIDUE_TYPES
        for value in residue_type_ids
    ):
        raise ValueError("residue type ids must be valid integers")
    if not isinstance(bond_mask, list) or len(bond_mask) != residues - 1:
        raise ValueError("bond mask topology shape mismatch")
    if not all(isinstance(value, bool) for value in bond_mask):
        raise ValueError("bond mask must contain booleans")
    if topology.get("residue_type_ids_sha256") != _json_sha256(residue_type_ids):
        raise ValueError("residue type provenance SHA256 mismatch")
    if topology.get("bond_mask_sha256") != _json_sha256(bond_mask):
        raise ValueError("bond mask provenance SHA256 mismatch")
    expected_indices = {index for index, valid in enumerate(bond_mask) if valid}
    records = topology.get("valid_bonds")
    if not isinstance(records, list) or int(topology.get("valid_bond_count", -1)) != len(records):
        raise ValueError("valid bond topology count mismatch")
    recorded_indices = set()
    for record in records:
        index = int(record.get("bond_index", -1))
        if index < 0 or index >= residues - 1 or index in recorded_indices:
            raise ValueError("invalid or duplicate topology bond index")
        if record.get("residue_position_pair") != [index, index + 1]:
            raise ValueError("topology residue position pair mismatch")
        if record.get("residue_type_pair") != [
            residue_type_ids[index], residue_type_ids[index + 1]
        ]:
            raise ValueError("topology residue type pair mismatch")
        if record.get("bond_mask_value") is not True or not bond_mask[index]:
            raise ValueError("topology record is not a mask-true bond")
        recorded_indices.add(index)
    if recorded_indices != expected_indices or not expected_indices:
        raise ValueError("valid bond indices do not reproduce bond_mask")
    return residue_type_ids, bond_mask, expected_indices


def _validate_per_start(
    records: object,
    starts: list[int],
    residue_type_ids: list[int],
    valid_indices: set[int],
) -> tuple[list[float], list[float], list[float]]:
    if not isinstance(records, list) or len(records) != len(starts):
        raise ValueError("per-start provenance count mismatch")
    all_predicted, all_source, all_target = [], [], []
    expected_order = sorted(valid_indices)
    for start_index, record in enumerate(records):
        if record.get("start_index") != start_index or record.get("start_frame") != starts[start_index]:
            raise ValueError("per-start identity mismatch")
        for name in (
            "source_positions_sha256", "target_positions_sha256", "predicted_positions_sha256"
        ):
            digest = record.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name} is not a SHA256 digest")
        lengths = record.get("valid_bond_lengths")
        if not isinstance(lengths, list) or [int(item.get("bond_index", -1)) for item in lengths] != expected_order:
            raise ValueError("per-start valid bond order or identity mismatch")
        by_index = {}
        for item in lengths:
            index = int(item["bond_index"])
            values = {
                name: _finite(item.get(name), name)
                for name in ("source_length", "target_length", "predicted_length")
            }
            if any(value < 0 for value in values.values()):
                raise ValueError("bond lengths must be non-negative")
            by_index[index] = values
            all_source.append(values["source_length"])
            all_target.append(values["target_length"])
            all_predicted.append(values["predicted_length"])
        maximum = record.get("max_predicted_bond", {})
        max_index = int(maximum.get("bond_index", -1))
        if max_index not in valid_indices:
            raise ValueError("maximum predicted bond is masked out")
        if maximum.get("residue_position_pair") != [max_index, max_index + 1]:
            raise ValueError("maximum predicted bond residue position pair mismatch")
        if maximum.get("residue_type_pair") != [
            residue_type_ids[max_index], residue_type_ids[max_index + 1]
        ]:
            raise ValueError("maximum predicted bond residue type pair mismatch")
        if max_index != max(by_index, key=lambda index: by_index[index]["predicted_length"]):
            raise ValueError("recorded maximum is not the longest valid predicted bond")
        for prefix in ("source", "target", "predicted"):
            derived = _distance(maximum.get(f"{prefix}_positions"), f"{prefix} endpoints")
            recorded_name = (
                "predicted_length_fp64" if prefix == "predicted" else f"{prefix}_length"
            )
            _close(derived, maximum.get(recorded_name), f"{prefix} endpoint length", atol=1e-9)
            comparison_name = "predicted_length" if prefix == "predicted" else f"{prefix}_length"
            _close(
                maximum.get(recorded_name),
                by_index[max_index][comparison_name],
                f"{prefix} maximum bond provenance",
                atol=1e-9,
            )
        _close(
            maximum.get("predicted_length_fp32"),
            maximum.get("predicted_length_fp64"),
            "FP32/FP64 maximum bond length",
        )
    return all_predicted, all_source, all_target


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    training_domain_list: str | Path,
    training_domain_list_sha256: str,
    external_domain_list: str | Path,
    external_domain_list_sha256: str,
    reference_panel_path: str | Path,
    reference_panel_sha256: str,
) -> dict:
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("step2000 replay checkpoint identity mismatch")
    if reference_panel_sha256 != EXPECTED_REFERENCE_PANEL_SHA256:
        raise ValueError("step2000 replay reference panel identity mismatch")
    if external_domain_list_sha256 != EXPECTED_EXTERNAL_PANEL_SHA256:
        raise ValueError("step2000 replay external panel identity mismatch")
    checkpoint, train_fingerprint = verify_multidomain_checkpoint(
        checkpoint_path,
        checkpoint_sha256,
        expected_step=EXPECTED_CHECKPOINT_STEP,
    )
    _, training_sha, external_ids, external_sha = load_disjoint_panels(
        training_domain_list,
        training_domain_list_sha256,
        external_domain_list,
        external_domain_list_sha256,
    )
    if _sha256(reference_panel_path) != reference_panel_sha256:
        raise ValueError("reference panel SHA256 mismatch")
    reference = json.loads(Path(reference_panel_path).read_text())
    frozen = _reference_cell(reference)
    result = json.loads(Path(result_path).read_text())

    if result.get("scope") != SCOPE:
        raise ValueError("step2000 outlier replay scope mismatch")
    if result.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("replay checkpoint SHA256 mismatch")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("replay checkpoint path mismatch")
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("replay checkpoint step mismatch")
    if int(result.get("checkpoint_train_seed", -1)) != EXPECTED_TRAIN_SEED:
        raise ValueError("replay checkpoint training seed mismatch")
    if result.get("checkpoint_train_fingerprint") != train_fingerprint:
        raise ValueError("replay checkpoint train fingerprint mismatch")
    if result.get("training_domain_list_sha256") != training_sha:
        raise ValueError("replay training panel identity mismatch")
    if result.get("external_panel") != {"sha256": external_sha, "ids": external_ids}:
        raise ValueError("replay external panel identity or order mismatch")
    if result.get("reference_panel", {}).get("sha256") != reference_panel_sha256:
        raise ValueError("recorded reference panel SHA256 mismatch")
    if reference.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("reference panel checkpoint identity mismatch")
    if result.get("settings") != {
        "delta_frames": 1, "steps": 1, "method": "mean", "source_noise": False
    }:
        raise ValueError("replay sampling settings mismatch")
    cell = result.get("cell", {})
    expected_cell = {
        "domain": OUTLIER_DOMAIN,
        "temperature": OUTLIER_TEMPERATURE,
        "replica": OUTLIER_REPLICA,
        "frames": int(frozen["frames"]),
        "starts": frozen["starts"],
    }
    if cell != expected_cell:
        raise ValueError("replay cell identity or starts mismatch")
    if result.get("frozen_panel_cell") != {
        name: frozen[name]
        for name in (
            "domain", "temperature", "replica", "frames", "starts",
            "model_rmsd_by_start", "noop_rmsd_by_start", "model_minus_noop_by_start",
            "mean_model_minus_noop", "bond_mean", "bond_max",
        )
    }:
        raise ValueError("embedded frozen panel cell mismatch")

    replay = result.get("replay", {})
    model = _vector(replay.get("model_rmsd_by_start"), "model RMSD")
    noop = _vector(replay.get("noop_rmsd_by_start"), "no-op RMSD")
    paired = _vector(replay.get("model_minus_noop_by_start"), "paired RMSD")
    for actual, expected in zip(model, frozen["model_rmsd_by_start"]):
        _close(actual, expected, "model RMSD")
    for actual, expected in zip(noop, frozen["noop_rmsd_by_start"]):
        _close(actual, expected, "no-op RMSD")
    for actual, expected in zip(paired, [left - right for left, right in zip(model, noop)]):
        _close(actual, expected, "paired RMSD", atol=1e-9)
    _close(replay.get("mean_model_minus_noop"), statistics.fmean(paired), "mean paired RMSD", atol=1e-9)
    _close(replay.get("mean_model_minus_noop"), frozen["mean_model_minus_noop"], "panel mean paired RMSD")
    _close(replay.get("bond_mean"), frozen["bond_mean"], "panel bond mean")
    _close(replay.get("bond_max"), frozen["bond_max"], "panel bond max")

    residue_type_ids, _, valid_indices = _validate_topology(replay.get("topology", {}))
    predicted, source, target = _validate_per_start(
        replay.get("per_start"), frozen["starts"], residue_type_ids, valid_indices
    )
    _close(statistics.fmean(predicted), replay["bond_mean"], "provenance bond mean")
    _close(max(predicted), replay["bond_max"], "provenance bond max")
    repeat_difference = _finite(
        replay.get("repeat_max_abs_prediction_difference"), "repeat difference"
    )
    batch_difference = _finite(
        replay.get("batched_vs_individual_max_abs_prediction_difference"),
        "batched-vs-individual difference",
    )
    if repeat_difference < 0 or batch_difference < 0:
        raise ValueError("prediction differences must be non-negative")
    if repeat_difference > IDENTITY_ATOL:
        status = "STOP_NONDETERMINISTIC_STEP2000_OUTLIER_REPLAY"
    elif batch_difference > IDENTITY_ATOL:
        status = "STOP_BATCH_CONTEXT_STEP2000_OUTLIER_REPLAY"
    elif max(source) > MAX_BOND_MAX or max(target) > MAX_BOND_MAX:
        status = "STOP_SOURCE_TARGET_GEOMETRY_ANOMALY"
    elif max(predicted) > MAX_BOND_MAX:
        status = "CONFIRM_STEP2000_MODEL_OUTPUT_OUTLIER"
    else:
        status = "STOP_STEP2000_OUTLIER_NOT_REPRODUCED"

    return {
        "status": status,
        "scope": "fixed single-cell provenance replay; no training and no gate changes",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "reference_panel_sha256": reference_panel_sha256,
        "result_sha256": _sha256(result_path),
        "cell": expected_cell,
        "bond_mean": float(replay["bond_mean"]),
        "bond_max": float(replay["bond_max"]),
        "source_bond_max": max(source),
        "target_bond_max": max(target),
        "repeat_max_abs_prediction_difference": repeat_difference,
        "batched_vs_individual_max_abs_prediction_difference": batch_difference,
        "valid_bond_count_per_start": len(valid_indices),
        "physical_limit_angstrom": MAX_BOND_MAX,
        "formal_training_authorized": False,
        "decision_rule": {
            "identity": "checkpoint, panel, cell, starts, RMSD, and aggregate bond metrics reproduce frozen evidence within 1e-5",
            "topology": "every counted bond is mask-true, adjacent in residue position, and carries valid residue-type ids",
            "model_outlier": "repeat and batched-vs-single differences <=1e-5; source/target bond maxima <=5.5A; predicted bond max >5.5A",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--external-domain-list", required=True)
    parser.add_argument("--external-domain-list-sha256", required=True)
    parser.add_argument("--reference-panel", required=True)
    parser.add_argument("--reference-panel-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = adjudicate(
        args.result,
        args.checkpoint,
        args.checkpoint_sha256,
        args.training_domain_list,
        args.training_domain_list_sha256,
        args.external_domain_list,
        args.external_domain_list_sha256,
        args.reference_panel,
        args.reference_panel_sha256,
    )
    Path(args.output).write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
