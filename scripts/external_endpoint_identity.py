#!/usr/bin/env python
"""Fail-closed identity checks for the external mdCATH endpoint panel."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path

import torch

from deepjump.data import discover_domains
from deepjump.evaluation import domain_id, load_frozen_domain_ids
from deepjump.utils import split_domains
from scripts.train_ddp import dataset_fingerprint


EXPECTED_CHECKPOINT_STEP = 1000
EXPECTED_CHECKPOINT_SCHEMA = 2
EXPECTED_TRAINING_DOMAINS = 1000
EXPECTED_EXTERNAL_DOMAINS = 20
EXPECTED_UNTOUCHED_DOMAINS = 100
EXPECTED_EXCLUSION_UNION_DOMAINS = 1120


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return digest


def verify_multidomain_checkpoint(
    checkpoint_path: str | Path,
    expected_sha256: str,
    *,
    expected_step: int = EXPECTED_CHECKPOINT_STEP,
) -> tuple[dict, str]:
    """Verify the frozen FP32 pilot checkpoint without trusting its filename."""
    actual_sha256 = _sha256(checkpoint_path)
    expected_sha256 = _require_digest(expected_sha256, label="checkpoint SHA256")
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError("checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if expected_step < 1:
        raise ValueError("expected checkpoint step must be positive")
    if checkpoint.get("step") != expected_step:
        raise ValueError(
            f"external endpoint gate requires checkpoint step {expected_step}"
        )
    if checkpoint.get("checkpoint_schema") != EXPECTED_CHECKPOINT_SCHEMA:
        raise ValueError("external endpoint gate requires checkpoint schema 2")

    cfg = checkpoint.get("cfg") or {}
    data_cfg = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    train_cfg = cfg.get("train") or {}
    required_data = {
        "domains": [],
        "val_fraction": 0.02,
        "seed": 0,
        "delta_frames": 1,
        "canon_symmetric": True,
    }
    for key, expected in required_data.items():
        if data_cfg.get(key) != expected:
            raise ValueError(f"checkpoint data.{key} mismatch")
    if model_cfg.get("tensor_cloud01") is not True:
        raise ValueError("checkpoint is not TensorCloud01")
    if model_cfg.get("tensor_cloud01_vector_only_attention", False) is not False:
        raise ValueError("checkpoint is not the full-tensor candidate")
    if train_cfg.get("seed") != 0 or train_cfg.get("amp") is not False:
        raise ValueError("checkpoint is not the frozen seed-0 FP32 pilot")

    train_state = checkpoint.get("train_state") or {}
    if train_state.get("world_size") != 8:
        raise ValueError("checkpoint world size mismatch")
    train_fingerprint = _require_digest(
        train_state.get("train_fingerprint"), label="checkpoint train fingerprint"
    )
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("checkpoint model state is missing")
    nonfinite = [
        name
        for name, value in model_state.items()
        if torch.is_tensor(value) and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f"checkpoint contains non-finite model tensors: {nonfinite[:5]}")
    return checkpoint, train_fingerprint


def load_disjoint_panels(
    training_domain_list: str | Path,
    training_domain_list_sha256: str,
    external_domain_list: str | Path,
    external_domain_list_sha256: str,
) -> tuple[list[str], str, list[str], str]:
    """Load exact training/external panels and prove their disjointness."""
    training_ids, training_sha256 = load_frozen_domain_ids(
        training_domain_list, training_domain_list_sha256
    )
    external_ids, external_sha256 = load_frozen_domain_ids(
        external_domain_list, external_domain_list_sha256
    )
    if len(training_ids) != EXPECTED_TRAINING_DOMAINS:
        raise ValueError("training subset must contain exactly 1000 domains")
    if len(external_ids) != EXPECTED_EXTERNAL_DOMAINS:
        raise ValueError("external panel must contain exactly 20 domains")
    overlap = sorted(set(training_ids) & set(external_ids))
    if overlap:
        raise ValueError(f"external panel overlaps the training subset: {overlap}")
    return training_ids, training_sha256, external_ids, external_sha256


def load_fresh_external_panels(
    training_domain_list: str | Path,
    training_domain_list_sha256: str,
    prior_external_domain_list: str | Path,
    prior_external_domain_list_sha256: str,
    untouched_domain_list: str | Path,
    untouched_domain_list_sha256: str,
    fresh_external_domain_list: str | Path,
    fresh_external_domain_list_sha256: str,
) -> dict[str, object]:
    """Load the four frozen panels and prove fresh external data was never reused."""
    specs = (
        ("training", training_domain_list, training_domain_list_sha256, EXPECTED_TRAINING_DOMAINS),
        ("prior_external", prior_external_domain_list, prior_external_domain_list_sha256, EXPECTED_EXTERNAL_DOMAINS),
        ("untouched", untouched_domain_list, untouched_domain_list_sha256, EXPECTED_UNTOUCHED_DOMAINS),
        ("fresh_external", fresh_external_domain_list, fresh_external_domain_list_sha256, EXPECTED_EXTERNAL_DOMAINS),
    )
    loaded: dict[str, dict[str, object]] = {}
    for label, path, expected_sha256, expected_count in specs:
        ids, actual_sha256 = load_frozen_domain_ids(path, expected_sha256)
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ValueError(f"{label} panel must contain exactly {expected_count} unique domains")
        loaded[label] = {"ids": ids, "sha256": actual_sha256}
    excluded = (
        set(loaded["training"]["ids"])
        | set(loaded["prior_external"]["ids"])
        | set(loaded["untouched"]["ids"])
    )
    if len(excluded) != EXPECTED_EXCLUSION_UNION_DOMAINS:
        raise ValueError("frozen exclusion panels are not mutually disjoint")
    overlap = sorted(excluded & set(loaded["fresh_external"]["ids"]))
    if overlap:
        raise ValueError(f"fresh external panel overlaps the frozen exclusion union: {overlap}")
    return {**loaded, "exclusion_union_count": len(excluded)}


def verify_guarded_training_prerequisite(
    decision_path: str | Path,
    expected_sha256: str,
    *,
    expected_checkpoint_sha256: str,
    expected_training_sha256: str,
) -> dict:
    """Bind an external gate to the exact successful guarded training decision."""
    expected_sha256 = _require_digest(expected_sha256, label="prerequisite decision SHA256")
    actual_sha256 = _sha256(decision_path)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError("prerequisite decision SHA256 mismatch")
    decision = json.loads(Path(decision_path).read_text())
    if decision.get("status") != "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20":
        raise ValueError("guarded training prerequisite did not pass")
    if decision.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("prerequisite checkpoint SHA256 mismatch")
    if decision.get("training_domain_list_sha256") != expected_training_sha256:
        raise ValueError("prerequisite training subset SHA256 mismatch")
    expected_flags = {
        "external_development_authorized": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    for key, expected in expected_flags.items():
        if type(decision.get(key)) is not bool or decision.get(key) is not expected:
            raise ValueError(f"prerequisite authorization flag mismatch: {key}")
    return {"sha256": actual_sha256, "status": decision["status"]}


def verify_training_fingerprint(
    checkpoint: dict,
    checkpoint_train_fingerprint: str,
    training_data_root: str | Path,
    training_ids: list[str],
) -> dict:
    """Reproduce the ordered training split and its checkpoint fingerprint."""
    root = Path(training_data_root).expanduser().resolve()
    data_cfg = checkpoint["cfg"]["data"]
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("training manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list):
        raise ValueError("training manifest root must be a list")

    discovered = discover_domains(root)
    by_name = {Path(path).name: Path(path) for path in discovered}
    if len(by_name) != len(discovered):
        raise ValueError("training data contains duplicate HDF5 filenames")
    files = []
    for entry in manifest:
        name = Path(entry["file"]).name
        if name not in by_name:
            raise FileNotFoundError(f"training manifest file is missing: {name}")
        files.append(by_name[name])
    discovered_ids = [domain_id(path) for path in files]
    if len(files) != EXPECTED_TRAINING_DOMAINS or set(discovered_ids) != set(training_ids):
        raise ValueError("training HDF5 identity does not match the frozen 1000-domain subset")

    train_files, val_files = split_domains(
        files, float(data_cfg["val_fraction"]), int(data_cfg["seed"])
    )
    fingerprint = dataset_fingerprint(train_files)
    if not hmac.compare_digest(fingerprint, checkpoint_train_fingerprint):
        raise ValueError("reconstructed training fingerprint does not match checkpoint")
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "domains_total": len(files),
        "train_domains": len(train_files),
        "validation_domains": len(val_files),
        "train_fingerprint": fingerprint,
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def finite_number(value: object, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
