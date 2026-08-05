#!/usr/bin/env python
"""Evaluate a reserved panel only through the sealed full-data identity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import statistics
import time
from pathlib import Path

import torch

from deepjump.config import ModelConfig
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import require_mdcath_full_grid, require_single_delta
from deepjump.evaluation_consumption import claim_reserved_evaluation
from deepjump.evaluation_contract import (
    _load_verified_checkpoint,
    verify_frozen_evaluation_identity,
)
from deepjump.model import DeepJumpLite
from deepjump.utils import resolve_device
from scripts.endpoint_panel_eval import EXPECTED_STARTS
from scripts.guarded_endpoint_panel_eval import (
    BOND_MAX,
    BOND_MEAN_HI,
    BOND_MEAN_LO,
    MAX_FALLBACK_CELLS,
    MAX_FALLBACK_STARTS,
    _evaluate_cell,
    _mechanism_probe,
)


SCOPE = "contracted_guarded_reserved_endpoint_panel_v1"
PREREQUISITE_SCHEMA = "deepjump.reserved_evaluation_authorization.v2"
PREREQUISITE_STATUS = {
    "development": "ADVANCE_EXPANDED_DATA_DEVELOPMENT",
    "external": "ADVANCE_EXPANDED_DATA_EXTERNAL",
    "untouched": "ADVANCE_SECOND_SEED_UNTOUCHED",
}


class _PinnedPayloads(list):
    """Own panel descriptors and close them on every normal or exceptional path."""

    def close(self) -> None:
        for pin in self:
            descriptor = pin.get("descriptor", -1)
            if descriptor >= 0:
                os.close(descriptor)
                pin["descriptor"] = -1

    def __del__(self):
        self.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reserved_evaluation_prerequisite(
    path: str | Path,
    expected_sha256: str,
    *,
    phase: str,
    checkpoint_sha256: str,
    checkpoint_step: int,
    contract_sha256: str,
    panel_name: str,
    panel_sha256: str | None = None,
) -> dict:
    """Require an exact prior decision before a reserved panel can be opened."""

    configured_path = Path(path).expanduser()
    if configured_path.is_symlink():
        raise ValueError("evaluation prerequisite must be a regular non-symlink file")
    decision_path = configured_path.resolve()
    if not decision_path.is_file():
        raise ValueError("evaluation prerequisite must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(decision_path, flags)
    except OSError as exc:
        raise ValueError("cannot open evaluation prerequisite") from exc
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    if not stat.S_ISREG(before.st_mode) or _stat_fingerprint(before) != _stat_fingerprint(after):
        raise ValueError("evaluation prerequisite changed while it was being read")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("evaluation prerequisite SHA256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation prerequisite is not valid UTF-8 JSON") from exc
    bound_panel_sha256 = payload.get("panel_sha256") if isinstance(payload, dict) else None
    if (
        not isinstance(bound_panel_sha256, str)
        or len(bound_panel_sha256) != 64
        or set(bound_panel_sha256) - frozenset("0123456789abcdef")
    ):
        raise ValueError("evaluation prerequisite panel SHA256 is invalid")
    if panel_sha256 is not None and bound_panel_sha256 != panel_sha256:
        raise ValueError("evaluation prerequisite panel SHA256 mismatch")
    authorization_id = payload.get("authorization_id") if isinstance(payload, dict) else None
    ledger_root = payload.get("consumption_ledger_root") if isinstance(payload, dict) else None
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("evaluation prerequisite authorization_id is invalid")
    if not isinstance(ledger_root, str) or not Path(ledger_root).is_absolute():
        raise ValueError("evaluation prerequisite ledger root must be absolute")
    required = {
        "schema": PREREQUISITE_SCHEMA,
        "authorization_id": authorization_id,
        "consumption_ledger_root": ledger_root,
        "status": PREREQUISITE_STATUS[phase],
        "phase": phase,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "full_training_contract_sha256": contract_sha256,
        "panel_name": panel_name,
        "panel_sha256": bound_panel_sha256,
        "reserved_panel_authorized": True,
        "formal_training_authorized": False,
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("evaluation prerequisite does not authorize this exact reserved-panel run")
    run_binding = payload.get("run_binding")
    if run_binding is not None and not isinstance(run_binding, dict):
        raise ValueError("evaluation prerequisite run binding is invalid")
    return {
        **required,
        **({"run_binding": run_binding} if run_binding is not None else {}),
        "path": str(decision_path),
        "sha256": actual_sha256,
    }


def _stat_fingerprint(value: os.stat_result) -> dict[str, int]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _rehash_regular_file(
    path: Path,
    expected_sha256: str,
    expected_fingerprint: dict,
    *,
    keep_open: bool = False,
) -> tuple[str, int | None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot open contracted panel payload: {path}") from exc
    try:
        digest = hashlib.sha256()
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"contracted panel payload is not regular: {path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _stat_fingerprint(before) != expected_fingerprint:
            raise ValueError(f"contracted panel payload fingerprint mismatch: {path}")
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise ValueError(f"contracted panel payload changed during rehash: {path}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"contracted panel payload SHA256 mismatch: {path}")
        if keep_open:
            return actual_sha256, descriptor
        os.close(descriptor)
        return actual_sha256, None
    except BaseException:
        os.close(descriptor)
        raise


def rehash_contracted_panel_payloads(
    contract: str | Path,
    panel_file: str | Path,
    data_root: str | Path,
    *,
    keep_open: bool = False,
) -> dict:
    """Live-rehash every panel H5 against the qualifying full-data manifest."""

    contract_path = Path(contract).expanduser().resolve()
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_row = contract_payload["artifacts"]["manifest"]
    manifest_path = (contract_path.parent / manifest_row["path"]).resolve()
    if manifest_path.is_symlink() or _sha256(manifest_path) != manifest_row["sha256"]:
        raise ValueError("contracted manifest identity drifted before evaluation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_domain = {row["domain"]: row for row in manifest}
    panel_domains = Path(panel_file).read_text(encoding="utf-8").splitlines()
    root = Path(data_root).expanduser().resolve()
    paths = []
    pins = _PinnedPayloads()
    rows = []
    try:
        for domain in panel_domains:
            entry = by_domain.get(domain)
            if entry is None:
                raise ValueError(
                    f"reserved panel domain is absent from the contracted manifest: {domain}"
                )
            path = root / "data" / entry["file"]
            digest, descriptor = _rehash_regular_file(
                path,
                entry["sha256"],
                entry["local_fingerprint"],
                keep_open=keep_open,
            )
            paths.append(path)
            if descriptor is not None:
                pins.append({
                    "domain": domain,
                    "path": path,
                    "descriptor": descriptor,
                    "fd_path": Path(f"/proc/self/fd/{descriptor}"),
                })
            rows.append({
                "domain": domain,
                "file": entry["file"],
                "bytes": entry["local_fingerprint"]["size"],
                "sha256": digest,
            })
    except BaseException:
        pins.close()
        raise
    return {
        "status": "PASS_CONTRACTED_PANEL_LIVE_PAYLOAD_REHASH",
        "manifest_sha256": manifest_row["sha256"],
        "panel_domains": len(rows),
        "panel_bytes": sum(row["bytes"] for row in rows),
        "payloads": rows,
        "paths": paths,
        "pins": pins,
    }


def _pinned_domain_handle(pin: dict) -> _DomainHandle:
    try:
        fingerprint = _stat_fingerprint(os.fstat(pin["descriptor"]))
    except OSError as exc:
        raise RuntimeError(
            "pinned panel descriptor is unavailable on this host"
        ) from exc
    handle = _DomainHandle(
        pin["path"],
        fingerprint,
        descriptor=pin["descriptor"],
    )
    handle.name = pin["domain"]
    return handle


def _pinned_mechanism_probe(
    checkpoint: dict,
    model,
    pin: dict,
    temperature: int,
    replica: int,
    delta: int,
    canon_symmetric: bool,
    device,
) -> dict:
    return _mechanism_probe(
        checkpoint,
        model,
        pin["path"],
        temperature,
        replica,
        delta,
        canon_symmetric,
        device,
        domain_name=pin["domain"],
        descriptor=pin["descriptor"],
    )


def _atomic_json_new(path: Path, payload: dict) -> str:
    """Publish complete JSON without overwriting or exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    content = (json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _preflight_output_paths(runtime_probe_output: str | Path, output: str | Path) -> None:
    paths = [Path(runtime_probe_output).expanduser(), Path(output).expanduser()]
    resolved = [path.resolve() for path in paths]
    if resolved[0] == resolved[1]:
        raise ValueError("runtime probe and final output paths must be distinct")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to consume a panel with an existing output: {path}")


def validate_evaluation_completeness(domains: list[dict], expected_domains: int) -> dict:
    """Reject incomplete/non-finite evidence without pretending to adjudicate it."""
    if len(domains) != expected_domains:
        raise ValueError("reserved evaluation domain count is incomplete")
    total_cells = 0
    total_starts = 0
    for domain in domains:
        cells = domain.get("cells")
        if not isinstance(cells, list) or len(cells) != 25:
            raise ValueError("reserved evaluation must contain exactly 25 cells per domain")
        for cell in cells:
            total_cells += 1
            starts = cell.get("by_start")
            if not isinstance(starts, list) or len(starts) != EXPECTED_STARTS:
                raise ValueError("reserved evaluation cell has an incomplete start set")
            if not all(
                row.get("target_position_finite") is True
                and row.get("selected_position_exact") is True
                and row.get("selected_vector_exact") is True
                and row.get("source", {}).get("position_finite") is True
                and row.get("source", {}).get("vector_finite") is True
                and row.get("guarded", {}).get("position_finite") is True
                and row.get("guarded", {}).get("vector_finite") is True
                and row.get("noop_rmsd") is not None
                and row.get("guarded", {}).get("rmsd") is not None
                and row.get("guarded", {}).get("minus_noop") is not None
                for row in starts
            ):
                raise ValueError("reserved evaluation contains non-finite or incomplete metrics")
            if cell.get("source_cell_physical") is not True or cell.get("guarded_cell_physical") is not True:
                raise ValueError("reserved evaluation contains a non-physical source or guarded cell")
            total_starts += len(starts)
    return {
        "status": "PASS_RESERVED_EVALUATION_EVIDENCE_COMPLETENESS",
        "domains": expected_domains,
        "cells": total_cells,
        "starts": total_starts,
        "scientific_adjudication_performed": False,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument(
        "--phase", choices=("development", "external", "untouched"), required=True
    )
    parser.add_argument("--panel-name", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--prerequisite-decision", required=True)
    parser.add_argument("--expected-prerequisite-decision-sha256", required=True)
    parser.add_argument("--runtime-probe-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _preflight_output_paths(args.runtime_probe_output, args.output)

    prerequisite = verify_reserved_evaluation_prerequisite(
        args.prerequisite_decision,
        args.expected_prerequisite_decision_sha256,
        phase=args.phase,
        checkpoint_sha256=args.expected_checkpoint_sha256,
        checkpoint_step=args.expected_checkpoint_step,
        contract_sha256=args.expected_contract_sha256,
        panel_name=args.panel_name,
    )
    consumption_claim = claim_reserved_evaluation(
        prerequisite,
        args.expected_prerequisite_decision_sha256,
        runtime_probe_output=args.runtime_probe_output,
        output=args.output,
    )
    identity = verify_frozen_evaluation_identity(
        args.checkpoint,
        args.contract,
        args.expected_contract_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_step=args.expected_checkpoint_step,
        phase=args.phase,
        panel_name=args.panel_name,
        panel_file=args.panel_file,
    )
    if prerequisite["panel_sha256"] != identity["panel_sha256"]:
        raise ValueError("consumed authorization panel SHA256 mismatch")
    checkpoint, _ = _load_verified_checkpoint(
        Path(args.checkpoint).expanduser().resolve(), args.expected_checkpoint_sha256
    )
    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    delta = require_single_delta(data_cfg["delta_frames"])
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    payload_pre = rehash_contracted_panel_payloads(
        args.contract,
        args.panel_file,
        data_cfg["root"],
        keep_open=True,
    )
    payload_pre.pop("paths")
    pins = payload_pre.pop("pins")

    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("contracted reserved-panel evaluation requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    mechanism = _pinned_mechanism_probe(
        checkpoint,
        model,
        pins[0],
        temperatures[0],
        replicas[0],
        delta,
        bool(data_cfg.get("canon_symmetric", False)),
        device,
    )
    residue_counts = []
    for pin in pins:
        handle = _pinned_domain_handle(pin)
        try:
            residue_counts.append((handle.layout.num_residues, pin))
        finally:
            handle.close()
    largest_residues, largest_pin = max(residue_counts, key=lambda item: item[0])
    handle = _pinned_domain_handle(largest_pin)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        _evaluate_cell(
            model=model,
            handle=handle,
            layout=handle.layout,
            temperature=temperatures[0],
            replica=replicas[0],
            delta=delta,
            canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
            device=device,
        )
        torch.cuda.synchronize(device)
        cell_seconds = time.perf_counter() - started
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    finally:
        handle.close()
    peak_fraction = peak_bytes / total_bytes
    projected_minutes = cell_seconds * len(pins) * len(temperatures) * len(replicas) / 60
    max_projected_minutes = 50.0 if args.phase == "external" else 250.0
    probe_status = (
        "PASS_RUNTIME_PROBE"
        if peak_fraction <= 0.8 and projected_minutes <= max_projected_minutes
        else "STOP_RUNTIME_PROBE"
    )
    runtime_probe = {
        "status": probe_status,
        "domain": largest_pin["domain"],
        "residues": int(largest_residues),
        "cell_seconds": cell_seconds,
        "projected_panel_minutes": projected_minutes,
        "peak_memory_bytes": peak_bytes,
        "total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_fraction,
        "limits": {
            "max_peak_memory_fraction": 0.8,
            "max_projected_minutes": max_projected_minutes,
        },
    }
    runtime_probe_sha256 = _atomic_json_new(Path(args.runtime_probe_output), runtime_probe)
    if probe_status != "PASS_RUNTIME_PROBE":
        raise RuntimeError(f"runtime probe failed with {probe_status}")
    torch.cuda.empty_cache()

    domains = []
    for pin in pins:
        handle = _pinned_domain_handle(pin)
        try:
            cells = [
                _evaluate_cell(
                    model=model,
                    handle=handle,
                    layout=handle.layout,
                    temperature=temperature,
                    replica=replica,
                    delta=delta,
                    canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                    device=device,
                )
                for temperature in temperatures
                for replica in replicas
            ]
            values = [cell["mean_guarded_minus_noop"] for cell in cells]
            domains.append({
                "domain": handle.name,
                "preprocessing": {
                    "canon_symmetric": bool(data_cfg.get("canon_symmetric", False)),
                    "residues_total": int(handle.layout.num_residues),
                    "residues_evaluated": int(handle.layout.num_residues),
                },
                "summary": {
                    "cells": len(cells),
                    "mean_guarded_minus_noop": (
                        statistics.fmean(values)
                        if all(value is not None for value in values)
                        else None
                    ),
                    "cells_better_than_noop": sum(
                        value is not None and value < 0 for value in values
                    ),
                    "fallback_starts": sum(cell["fallback_starts"] for cell in cells),
                    "fallback_cells": sum(cell["fallback_starts"] > 0 for cell in cells),
                },
                "cells": cells,
            })
        finally:
            handle.close()

    completeness = validate_evaluation_completeness(domains, identity["panel_domains"])
    pins.close()

    payload_post = rehash_contracted_panel_payloads(
        args.contract, args.panel_file, data_cfg["root"]
    )
    payload_post.pop("paths")
    payload_post.pop("pins")
    identity_post = verify_frozen_evaluation_identity(
        args.checkpoint,
        args.contract,
        args.expected_contract_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_step=args.expected_checkpoint_step,
        phase=args.phase,
        panel_name=args.panel_name,
        panel_file=args.panel_file,
    )
    if identity_post != identity or payload_post != payload_pre:
        raise RuntimeError("reserved evaluation identity changed during execution")
    result = {
        "status": "EVALUATION_COMPLETE_NOT_ADJUDICATED",
        "scope": SCOPE,
        "identity": identity,
        "prerequisite": prerequisite,
        "consumption_claim": consumption_claim,
        "payload_verification": payload_post,
        "runtime_probe": runtime_probe,
        "runtime_probe_sha256": runtime_probe_sha256,
        "mechanism_probe": mechanism,
        "evidence_completeness": completeness,
        "delta_frames": delta,
        "settings": {
            "starts": EXPECTED_STARTS,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
            "policy": "reject_to_exact_source_per_start",
            "strict_thresholds": {
                "bond_mean_gt": BOND_MEAN_LO,
                "bond_mean_lt": BOND_MEAN_HI,
                "bond_max_lt": BOND_MAX,
            },
            "fallback_caps": {
                "max_starts": MAX_FALLBACK_STARTS,
                "max_cells": MAX_FALLBACK_CELLS,
            },
        },
        "grid": {"temperatures": temperatures, "replicas": replicas},
        "domains": domains,
        "formal_training_authorized": False,
    }
    _atomic_json_new(Path(args.output), result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
