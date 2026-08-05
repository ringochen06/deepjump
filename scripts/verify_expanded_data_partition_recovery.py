#!/usr/bin/env python
"""Verify that a regenerated partition audit matches sealed semantics.

The partition builder records absolute provenance and output paths. Those paths
legitimately change when a reviewed recovery commit is deployed in a new clean
repository or writes through a new atomic temporary directory. Recovery may
therefore ignore directory prefixes, but it must preserve every field and every
referenced basename.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "deepjump.expanded_data_partition.v1"
EXPECTED_STATUS = "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_report(path: str | Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"{label} schema mismatch")
    if payload.get("status") != EXPECTED_STATUS:
        raise ValueError(f"{label} status mismatch")
    return payload


def _replace_path_with_basename(container: dict[str, Any], key: str, label: str) -> None:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be an absolute file path")
    container[key] = {"basename": path.name}


def _normalized_identity(payload: dict[str, Any], label: str) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)

    panels = normalized.get("held_out_panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError(f"{label} held_out_panels must be a non-empty list")
    for index, panel in enumerate(panels):
        if not isinstance(panel, dict):
            raise ValueError(f"{label} held_out_panels[{index}] must be an object")
        _replace_path_with_basename(
            panel, "path", f"{label} held_out_panels[{index}].path"
        )

    outputs = normalized.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{label} outputs must be an object")
    train_output = outputs.get("train_eligible_list")
    audit_output = outputs.get("audit")
    if not isinstance(train_output, dict) or not isinstance(audit_output, dict):
        raise ValueError(f"{label} outputs are malformed")
    _replace_path_with_basename(
        train_output, "path", f"{label} outputs.train_eligible_list.path"
    )
    _replace_path_with_basename(
        audit_output, "path", f"{label} outputs.audit.path"
    )

    panel_registry = normalized.get("panel_registry")
    if not isinstance(panel_registry, dict):
        raise ValueError(f"{label} panel_registry must be an object")
    _replace_path_with_basename(
        panel_registry, "path", f"{label} panel_registry.path"
    )
    return normalized


def verify_recovery_equivalence(
    sealed_path: str | Path, candidate_path: str | Path
) -> None:
    sealed = _normalized_identity(_load_report(sealed_path, "sealed audit"), "sealed")
    candidate = _normalized_identity(
        _load_report(candidate_path, "candidate audit"), "candidate"
    )
    if candidate != sealed:
        raise ValueError(
            "candidate partition audit differs from sealed semantics after "
            "deployment/output path normalization"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    try:
        verify_recovery_equivalence(args.sealed, args.candidate)
    except ValueError as exc:
        parser.error(str(exc))
    print("PASS_EXPANDED_DATA_PARTITION_RECOVERY_EQUIVALENCE")


if __name__ == "__main__":
    main()
