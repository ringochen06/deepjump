"""Identity gate for external and untouched evaluation of contracted checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch

from deepjump.data_contract import (
    ContractExpectations,
    _read_regular_bytes,
    verify_full_training_data_contract,
)


PHASE_ROLES = {
    "development": "development_seen",
    "external": "external_reserved",
    "untouched": "untouched_confirmation_reserved",
}
_HEX = frozenset("0123456789abcdef")


def _load_json_bytes(raw: bytes, label: str):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a lowercase 64-hex SHA256")
    return value


def _load_verified_checkpoint(path: Path, expected_sha256: str) -> tuple[dict, str]:
    """Hash and load one immutable byte snapshot of a non-symlink checkpoint."""

    expected_sha256 = _require_sha256(expected_sha256, "checkpoint SHA256")
    raw = _read_regular_bytes(path, "checkpoint")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "checkpoint SHA256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload, actual_sha256


def _verify_model_state(model_state: object) -> None:
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("checkpoint model state must be a non-empty dictionary")
    invalid = [
        name
        for name, value in model_state.items()
        if not isinstance(name, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise ValueError(f"checkpoint model state contains non-tensor entries: {invalid[:5]}")
    nonfinite = [
        name
        for name, value in model_state.items()
        if (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all().item()
    ]
    if nonfinite:
        raise ValueError(f"checkpoint model state contains non-finite tensors: {nonfinite[:5]}")


def verify_frozen_evaluation_identity(
    checkpoint: str | Path,
    contract: str | Path,
    expected_contract_sha256: str,
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_step: int,
    phase: str,
    panel_name: str,
    panel_file: str | Path,
    contract_expectations: ContractExpectations | None = None,
) -> dict:
    """Verify that a checkpoint and evaluation panel share one sealed data contract."""

    if phase not in PHASE_ROLES:
        raise ValueError("phase must be 'development', 'external', or 'untouched'")
    if type(expected_checkpoint_step) is not int or expected_checkpoint_step <= 0:
        raise ValueError("expected checkpoint step must be a positive integer")
    checkpoint_input = Path(checkpoint).expanduser()
    contract_input = Path(contract).expanduser()
    if checkpoint_input.is_symlink() or not checkpoint_input.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    if contract_input.is_symlink() or not contract_input.is_file():
        raise ValueError("contract must be a regular non-symlink file")
    checkpoint_path = checkpoint_input.resolve()
    contract_path = contract_input.resolve()

    checkpoint_payload, actual_checkpoint_sha256 = _load_verified_checkpoint(
        checkpoint_path, expected_checkpoint_sha256
    )
    if not isinstance(checkpoint_payload.get("cfg"), dict):
        raise ValueError("checkpoint has no configuration identity")
    model_state = checkpoint_payload.get("model")
    _verify_model_state(model_state)
    step = checkpoint_payload.get("step")
    if type(step) is not int or step != expected_checkpoint_step:
        raise ValueError(
            f"checkpoint step mismatch: {step!r} != {expected_checkpoint_step}"
        )
    cfg = checkpoint_payload["cfg"]
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    if train_cfg.get("run_class") not in {"full_data_stage", "formal"}:
        raise ValueError("checkpoint was not produced by a contracted full-data run")
    if data_cfg.get("full_training_contract_sha256") != expected_contract_sha256:
        raise ValueError("checkpoint configuration contract SHA256 mismatch")
    if Path(data_cfg.get("full_training_contract", "")).expanduser().resolve() != contract_path:
        raise ValueError("checkpoint configuration contract path mismatch")

    verification = verify_full_training_data_contract(
        contract_path,
        expected_contract_sha256,
        configured_root=data_cfg.get("root", ""),
        configured_manifest=data_cfg.get("manifest", ""),
        configured_domains_file=data_cfg.get("domains_file", ""),
        expectations=contract_expectations,
    )
    state = checkpoint_payload.get("train_state")
    if not isinstance(state, dict) or state.get("full_training_data_contract") != verification:
        raise ValueError("checkpoint train state does not bind the exact full-data contract")

    contract_raw = _read_regular_bytes(contract_path, "contract")
    if hashlib.sha256(contract_raw).hexdigest() != expected_contract_sha256:
        raise ValueError("training data contract changed during evaluation identity verification")
    contract_payload = _load_json_bytes(contract_raw, "contract")
    registry_row = contract_payload["artifacts"]["panel_registry"]
    registry_path = contract_path.parent / registry_row["path"]
    registry_raw = _read_regular_bytes(registry_path, "panel registry")
    if hashlib.sha256(registry_raw).hexdigest() != registry_row["sha256"]:
        raise ValueError("contracted panel registry SHA256 mismatch")
    registry = _load_json_bytes(registry_raw, "panel registry")
    matches = [row for row in registry.get("panels", []) if row.get("name") == panel_name]
    if len(matches) != 1:
        raise ValueError("panel name is absent or duplicated in the frozen registry")
    panel = matches[0]
    required_role = PHASE_ROLES[phase]
    if panel.get("role") != required_role:
        raise ValueError(
            f"panel role {panel.get('role')!r} cannot serve as {phase}; "
            f"required {required_role!r}"
        )
    expected_panel_input = registry_path.parent / panel["path"]
    configured_panel_input = Path(panel_file).expanduser()
    if configured_panel_input.is_symlink() or not configured_panel_input.is_file():
        raise ValueError("panel file must be a regular non-symlink file")
    if expected_panel_input.is_symlink() or not expected_panel_input.is_file():
        raise ValueError("frozen panel must be a regular non-symlink file")
    configured_panel_path = configured_panel_input.resolve()
    expected_panel_path = expected_panel_input.resolve()
    if configured_panel_path != expected_panel_path:
        raise ValueError("configured panel file differs from the frozen registry")
    panel_raw = _read_regular_bytes(configured_panel_path, "evaluation panel")
    if hashlib.sha256(panel_raw).hexdigest() != panel["sha256"]:
        raise ValueError("evaluation panel SHA256 mismatch")
    try:
        panel_domains = panel_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("evaluation panel is not valid UTF-8") from exc
    if len(panel_domains) != panel["domains"] or len(set(panel_domains)) != panel["domains"]:
        raise ValueError("evaluation panel count or uniqueness mismatch")

    train_row = contract_payload["artifacts"]["train_list"]
    train_path = contract_path.parent / train_row["path"]
    train_raw = _read_regular_bytes(train_path, "contracted train list")
    if hashlib.sha256(train_raw).hexdigest() != train_row["sha256"]:
        raise ValueError("contracted train-list SHA256 mismatch")
    try:
        train_domains = set(train_raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as exc:
        raise ValueError("contracted train list is not valid UTF-8") from exc
    overlap = sorted(train_domains & set(panel_domains))
    if overlap:
        raise ValueError(f"evaluation panel overlaps contracted training domains: {overlap[:10]}")

    return {
        "status": "PASS_FROZEN_EVALUATION_IDENTITY",
        "phase": phase,
        "panel_name": panel_name,
        "panel_role": required_role,
        "panel_sha256": panel["sha256"],
        "panel_domains": panel["domains"],
        "checkpoint_sha256": actual_checkpoint_sha256,
        "checkpoint_step": step,
        "full_training_contract_sha256": expected_contract_sha256,
        "train_list_sha256": verification["train_list_sha256"],
        "formal_training_authorized": False,
    }
