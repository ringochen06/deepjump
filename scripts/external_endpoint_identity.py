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
) -> tuple[dict, str]:
    """Verify the frozen FP32 pilot checkpoint without trusting its filename."""
    actual_sha256 = _sha256(checkpoint_path)
    expected_sha256 = _require_digest(expected_sha256, label="checkpoint SHA256")
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError("checkpoint SHA256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("external endpoint gate requires checkpoint step 1000")
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
