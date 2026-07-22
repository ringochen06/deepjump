#!/usr/bin/env python
"""Fail-closed identity checks for the external mdCATH endpoint panel."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Mapping

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
    expected_data_config: Mapping[str, object] | None = None,
    expected_model_config: Mapping[str, object] | None = None,
    expected_train_config: Mapping[str, object] | None = None,
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
    if expected_data_config is not None:
        for key, expected in expected_data_config.items():
            if data_cfg.get(key) != expected:
                raise ValueError(f"checkpoint data.{key} mismatch")
    if model_cfg.get("tensor_cloud01") is not True:
        raise ValueError("checkpoint is not TensorCloud01")
    if expected_model_config is None:
        if model_cfg.get("tensor_cloud01_vector_only_attention", False) is not False:
            raise ValueError("checkpoint is not the full-tensor candidate")
    else:
        expected_vector_only = expected_model_config.get(
            "tensor_cloud01_vector_only_attention"
        )
        if type(expected_vector_only) is not bool or type(
            model_cfg.get("tensor_cloud01_vector_only_attention")
        ) is not bool:
            raise ValueError("checkpoint vector-only architecture flag is not boolean")
        if (
            model_cfg.get("tensor_cloud01_vector_only_attention")
            is not expected_vector_only
        ):
            raise ValueError("checkpoint vector-only architecture flag mismatch")
        for key, expected in expected_model_config.items():
            actual = (
                model_cfg.get(key, False)
                if key == "tensor_cloud01_vector_only_scalar_value"
                else model_cfg.get(key)
            )
            if key == "tensor_cloud01_vector_only_scalar_value" and (
                type(actual) is not bool or type(expected) is not bool
            ):
                raise ValueError(
                    "checkpoint scalar-value architecture flag is not boolean"
                )
            if actual != expected:
                raise ValueError(f"checkpoint model.{key} mismatch")
    if train_cfg.get("seed") != 0 or train_cfg.get("amp") is not False:
        raise ValueError("checkpoint is not the frozen seed-0 FP32 pilot")
    if expected_train_config is not None:
        for key, expected in expected_train_config.items():
            if train_cfg.get(key) != expected:
                raise ValueError(f"checkpoint train.{key} mismatch")

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


def load_paper_horizon_external_panels(
    training_domain_list: str | Path,
    training_domain_list_sha256: str,
    prior_external_domain_list: str | Path,
    prior_external_domain_list_sha256: str,
    prior_fresh_external_domain_list: str | Path,
    prior_fresh_external_domain_list_sha256: str,
    untouched_domain_list: str | Path,
    untouched_domain_list_sha256: str,
    paper_horizon_domain_list: str | Path,
    paper_horizon_domain_list_sha256: str,
) -> dict[str, object]:
    """Prove the paper-horizon external panel excludes all 1140 seen/reserved domains."""
    specs = (
        ("training", training_domain_list, training_domain_list_sha256, 1000),
        ("prior_external", prior_external_domain_list, prior_external_domain_list_sha256, 20),
        ("prior_fresh_external", prior_fresh_external_domain_list,
         prior_fresh_external_domain_list_sha256, 20),
        ("untouched", untouched_domain_list, untouched_domain_list_sha256, 100),
        ("paper_horizon_external", paper_horizon_domain_list,
         paper_horizon_domain_list_sha256, 20),
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
        | set(loaded["prior_fresh_external"]["ids"])
        | set(loaded["untouched"]["ids"])
    )
    if len(excluded) != 1140:
        raise ValueError("paper-horizon exclusion panels are not mutually disjoint")
    overlap = sorted(excluded & set(loaded["paper_horizon_external"]["ids"]))
    if overlap:
        raise ValueError(
            "paper-horizon external panel overlaps the frozen exclusion union: "
            f"{overlap}"
        )
    return {**loaded, "exclusion_union_count": len(excluded)}


def verify_paper_horizon_ab_prerequisite(
    decision_path: str | Path,
    expected_sha256: str,
    *,
    expected_candidate_checkpoint_sha256: str,
    expected_training_sha256: str,
    expected_training_panel_sha256: str,
) -> dict:
    """Bind the external A/B to the exact successful training-dev A/B decision."""
    expected_sha256 = _require_digest(expected_sha256, label="A/B decision SHA256")
    actual_sha256 = _sha256(decision_path)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError("A/B prerequisite decision SHA256 mismatch")
    decision = json.loads(Path(decision_path).read_text())
    expected = {
        "status": "ADVANCE_PAPER_HORIZON_EXTERNAL20",
        "candidate_checkpoint_sha256": expected_candidate_checkpoint_sha256,
        "training_domain_list_sha256": expected_training_sha256,
        "domain_list_sha256": expected_training_panel_sha256,
        "external_development_authorized": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    for key, value in expected.items():
        if decision.get(key) != value or (
            isinstance(value, bool) and type(decision.get(key)) is not bool
        ):
            raise ValueError(f"A/B prerequisite mismatch: {key}")
    return {"sha256": actual_sha256, "status": decision["status"]}


def verify_paper_vector_ab_prerequisite(
    decision_path: str | Path,
    expected_sha256: str,
    *,
    expected_baseline_checkpoint_sha256: str,
    expected_candidate_checkpoint_sha256: str,
    expected_training_sha256: str,
    expected_training_panel_sha256: str,
) -> dict:
    """Bind external evaluation to the exact successful paper-vector A/B."""
    expected_sha256 = _require_digest(
        expected_sha256, label="paper-vector A/B decision SHA256"
    )
    actual_sha256 = _sha256(decision_path)
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise ValueError("paper-vector A/B prerequisite decision SHA256 mismatch")
    decision = json.loads(Path(decision_path).read_text())
    expected = {
        "status": "ADVANCE_PAPER_VECTOR_EXTERNAL20",
        "scope": (
            "matched_fresh_continuous_0_to_2000_"
            "paper_vector_attention_training_dev_ab"
        ),
        "panel_kind": "training",
        "baseline_checkpoint_sha256": expected_baseline_checkpoint_sha256,
        "candidate_checkpoint_sha256": expected_candidate_checkpoint_sha256,
        "training_domain_list_sha256": expected_training_sha256,
        "domain_list_sha256": expected_training_panel_sha256,
        "baseline_reproduced": True,
        "candidate_absolute_pass": True,
        "paired_pass": True,
        "external_development_authorized": True,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    for key, value in expected.items():
        actual = decision.get(key)
        if actual != value or (isinstance(value, bool) and type(actual) is not bool):
            raise ValueError(f"paper-vector A/B prerequisite mismatch: {key}")
    objective = decision.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("paper-vector A/B prerequisite objective is missing")
    if objective.get("required_candidate_factor") != 0.995:
        raise ValueError("paper-vector A/B prerequisite objective factor mismatch")
    if type(objective.get("passes")) is not bool or objective.get("passes") is not True:
        raise ValueError("paper-vector A/B prerequisite objective did not pass")
    h20 = decision.get("candidate_h20_gate")
    if not isinstance(h20, dict):
        raise ValueError("paper-vector A/B prerequisite H20 gate is missing")
    if type(h20.get("passes")) is not bool or h20.get("passes") is not True:
        raise ValueError("paper-vector A/B prerequisite H20 gate did not pass")
    return {
        "sha256": actual_sha256,
        "status": decision["status"],
        "baseline_checkpoint_sha256": decision["baseline_checkpoint_sha256"],
        "candidate_checkpoint_sha256": decision["candidate_checkpoint_sha256"],
    }


def verify_paper_vector_external_evidence(
    claim_path: str | Path,
    expected_claim_sha256: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    *,
    expected_panel_sha256: str,
    expected_prerequisite_decision_sha256: str,
    expected_baseline_checkpoint_sha256: str,
    expected_candidate_checkpoint_sha256: str,
    source_proof_path: str | Path,
    expected_source_proof_sha256: str,
    panel_data_root: str | Path | None = None,
) -> dict:
    """Bind a paper-vector external evaluation to its one-shot claim and HDF5s."""
    claim_sha = _require_digest(expected_claim_sha256, label="external claim SHA256")
    manifest_sha = _require_digest(
        expected_manifest_sha256, label="external download manifest SHA256"
    )
    source_proof_sha = _require_digest(
        expected_source_proof_sha256, label="source proof SHA256"
    )
    if not hmac.compare_digest(_sha256(claim_path), claim_sha):
        raise ValueError("external claim SHA256 mismatch")
    if not hmac.compare_digest(_sha256(manifest_path), manifest_sha):
        raise ValueError("external download manifest SHA256 mismatch")
    if not hmac.compare_digest(_sha256(source_proof_path), source_proof_sha):
        raise ValueError("source proof SHA256 mismatch")
    source_proof = json.loads(Path(source_proof_path).read_text())
    expected_source_proof = {
        "schema": "deepjump.prior_source_control_flow_proof.v1",
        "status": "PASS_PRIOR_AUTHORITATIVE_RUN_EXTERNAL_UNCONSUMED",
        "source_run_id": "20260722T012922Z",
        "source_commit": "dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b",
        "source_audit_obs_uri": (
            "obs://deepjump-mdcath-cn4-ringochen/deepjump-calibration/"
            "paper-horizon-ab2000/20260722T012922Z/audit"
        ),
        "source_decision_sha256": (
            "2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38"
        ),
        "source_runner_sha256": (
            "2c8eedad191a814080303b6a30204fbb9bee522937c3a0cb5087e3439b6bd75f"
        ),
        "source_status": "STOP_PAPER_HORIZON_OBJECTIVE_GAIN",
        "required_advance_status": "ADVANCE_PAPER_HORIZON_EXTERNAL20",
        "proof_basis": "fixed_decision_and_fixed_runner_control_flow",
        "prior_authoritative_run_consumed": False,
    }
    if not isinstance(source_proof, dict) or source_proof != expected_source_proof:
        raise ValueError("source proof exact schema or fixed evidence mismatch")
    claim = json.loads(Path(claim_path).read_text())
    expected_claim = {
        "schema": "deepjump.external_panel_claim.v1",
        "status": "CLAIMED_FOR_SINGLE_USE",
        "panel_sha256": expected_panel_sha256,
        "panel_count": 20,
        "expected_total_bytes": 14236836972,
        "source_stop_decision_sha256": (
            "2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38"
        ),
        "source_proof_sha256": source_proof_sha,
        "training_ab_decision_sha256": expected_prerequisite_decision_sha256,
        "baseline_checkpoint_sha256": expected_baseline_checkpoint_sha256,
        "candidate_checkpoint_sha256": expected_candidate_checkpoint_sha256,
        "prior_authoritative_run_consumed": False,
    }
    if not isinstance(claim, dict) or set(claim) != {
        *expected_claim,
        "run_id",
        "commit",
        "claimed_at",
    }:
        raise ValueError("external claim exact schema mismatch")
    for key, expected in expected_claim.items():
        actual = claim.get(key)
        if actual != expected or (isinstance(expected, bool) and type(actual) is not bool):
            raise ValueError(f"external claim mismatch: {key}")
    manifest = json.loads(Path(manifest_path).read_text())
    expected_manifest_keys = {
        "schema", "status", "panel_sha256", "claim_sha256", "run_id",
        "commit", "root", "files_count", "total_bytes", "trajectories",
        "unresolved_failures", "inventory_sha256", "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("external download manifest exact schema mismatch")
    checks = {
        "schema": "deepjump.external_download_inventory.v1",
        "status": "PASS",
        "panel_sha256": expected_panel_sha256,
        "claim_sha256": claim_sha,
        "run_id": claim.get("run_id"),
        "commit": claim.get("commit"),
        "files_count": 20,
        "total_bytes": 14236836972,
        "trajectories": 500,
        "unresolved_failures": 0,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"external download manifest mismatch: {key}")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 20:
        raise ValueError("external download manifest must contain 20 files")
    file_keys = {
        "domain", "relative_path", "bytes", "sha256", "residues",
        "trajectories", "min_frames",
    }
    if any(not isinstance(row, dict) or set(row) != file_keys for row in files):
        raise ValueError("external download manifest file schema mismatch")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest.get("inventory_sha256"):
        raise ValueError("external download inventory SHA256 mismatch")
    if panel_data_root is not None:
        root = Path(panel_data_root).resolve()
        if manifest.get("root") != str(root):
            raise ValueError("external download root mismatch")
        expected_paths = set()
        for row in files:
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe external manifest path")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("external manifest file missing or symlink")
            if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
                raise ValueError("external manifest file identity mismatch")
            expected_paths.add(path.resolve())
        if set(path.resolve() for path in root.rglob("*.h5")) != expected_paths:
            raise ValueError("external manifest HDF5 exact inventory mismatch")
    return {
        "claim_sha256": claim_sha,
        "download_manifest_sha256": manifest_sha,
        "inventory_sha256": manifest["inventory_sha256"],
        "source_proof_sha256": source_proof_sha,
        "panel_sha256": expected_panel_sha256,
        "run_id": claim["run_id"],
        "commit": claim["commit"],
    }


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
