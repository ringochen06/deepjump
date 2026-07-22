#!/usr/bin/env python
"""Evaluate a frozen external H1 panel with the multi-domain FP32 pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deepjump.config import ModelConfig
from deepjump.data import discover_domains
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import (
    require_mdcath_full_grid,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.model import DeepJumpLite
from deepjump.utils import resolve_device
from scripts.endpoint_panel_eval import (
    EXPECTED_DOMAINS,
    EXPECTED_STARTS,
    _domain_summary,
    _evaluate_cell,
    _runtime_probe_status,
)
from scripts.external_endpoint_identity import (
    EXPECTED_CHECKPOINT_STEP,
    load_disjoint_panels,
    verify_multidomain_checkpoint,
    verify_training_fingerprint,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--expected-checkpoint-step", type=int, default=EXPECTED_CHECKPOINT_STEP
    )
    parser.add_argument("--training-data-root", required=True)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--external-data-root", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--starts", type=int, default=EXPECTED_STARTS)
    parser.add_argument("--runtime-probe-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.starts != EXPECTED_STARTS:
        parser.error("the frozen external panel requires three starts per cell")

    checkpoint, train_fingerprint = verify_multidomain_checkpoint(
        args.ckpt,
        args.checkpoint_sha256,
        expected_step=args.expected_checkpoint_step,
    )
    model_cfg = checkpoint["cfg"]["model"]
    data_cfg = checkpoint["cfg"]["data"]
    training_ids, training_sha256, domain_ids, domain_sha256 = load_disjoint_panels(
        args.training_domain_list,
        args.training_domain_list_sha256,
        args.domain_list,
        args.domain_list_sha256,
    )
    training_identity = verify_training_fingerprint(
        checkpoint,
        train_fingerprint,
        args.training_data_root,
        training_ids,
    )

    delta = require_single_delta(data_cfg["delta_frames"])
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    external_root = Path(args.external_data_root).expanduser().resolve()
    paths = resolve_frozen_domains(discover_domains(external_root), domain_ids)
    external_total_bytes = sum(path.stat().st_size for path in paths)

    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("external endpoint gate requires a CUDA device")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    residue_counts = []
    for path in paths:
        handle = _DomainHandle(path)
        try:
            residue_counts.append((handle.layout.num_residues, path))
        finally:
            handle.close()
    largest_residues, largest_path = max(residue_counts, key=lambda item: item[0])
    handle = _DomainHandle(largest_path)
    try:
        layout = handle.layout
        residue_index = torch.as_tensor(layout.res_index, device=device)
        atom_mask = torch.as_tensor(layout.atom_mask, device=device)
        bond_mask = torch.as_tensor(layout.bond_mask, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        _evaluate_cell(
            model=model,
            handle=handle,
            layout=layout,
            residue_index=residue_index,
            atom_mask=atom_mask,
            bond_mask=bond_mask,
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
    projected_minutes = cell_seconds * EXPECTED_DOMAINS * 25 / 60
    probe_status = _runtime_probe_status(peak_fraction, projected_minutes)
    runtime_probe = {
        "status": probe_status,
        "domain": Path(largest_path).stem.replace("mdcath_dataset_", ""),
        "residues": largest_residues,
        "batch_size": EXPECTED_STARTS,
        "cell_seconds": cell_seconds,
        "projected_500_cell_minutes": projected_minutes,
        "peak_memory_bytes": peak_bytes,
        "total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_fraction,
        "limits": {"max_peak_memory_fraction": 0.8, "max_projected_minutes": 50.0},
    }
    probe_output = Path(args.runtime_probe_output)
    probe_output.parent.mkdir(parents=True, exist_ok=True)
    probe_output.write_text(json.dumps(runtime_probe, indent=2) + "\n")
    if probe_status != "PASS_RUNTIME_PROBE":
        raise RuntimeError(f"runtime probe failed with {probe_status}")
    torch.cuda.empty_cache()

    domains = []
    for path in paths:
        handle = _DomainHandle(path)
        try:
            layout = handle.layout
            residue_index = torch.as_tensor(layout.res_index, device=device)
            atom_mask = torch.as_tensor(layout.atom_mask, device=device)
            bond_mask = torch.as_tensor(layout.bond_mask, device=device)
            cells = []
            for temperature in temperatures:
                for replica in replicas:
                    cells.append(_evaluate_cell(
                        model=model,
                        handle=handle,
                        layout=layout,
                        residue_index=residue_index,
                        atom_mask=atom_mask,
                        bond_mask=bond_mask,
                        temperature=temperature,
                        replica=replica,
                        delta=delta,
                        canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                        device=device,
                    ))
            domains.append({
                "domain": handle.name,
                "preprocessing": {
                    "canon_symmetric": bool(data_cfg.get("canon_symmetric", False)),
                    "residues_total": layout.num_residues,
                    "residues_evaluated": layout.num_residues,
                },
                "summary": _domain_summary(cells),
                "cells": cells,
            })
        finally:
            handle.close()

    result = {
        "scope": "external_multidomain_fp32_pilot_h1",
        "checkpoint": args.ckpt,
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_schema": int(checkpoint["checkpoint_schema"]),
        "checkpoint_train_fingerprint": train_fingerprint,
        "delta_frames": delta,
        "settings": {
            "starts": EXPECTED_STARTS,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
        },
        "training_subset": {
            "path": args.training_domain_list,
            "sha256": training_sha256,
            "ids": training_ids,
            **training_identity,
        },
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_sha256,
            "ids": domain_ids,
            "data_root": str(external_root),
            "h5_files": len(paths),
            "total_bytes": external_total_bytes,
        },
        "grid": {"temperatures": temperatures, "replicas": replicas},
        "runtime_probe": runtime_probe,
        "domains": domains,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
