#!/usr/bin/env python
"""Evaluate clean H1 endpoints over the frozen 20-domain mdCATH panel."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from deepjump.config import ModelConfig
from deepjump.data import discover_domains
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import (
    load_frozen_domain_ids,
    require_mdcath_full_grid,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.metrics import aligned_ca_rmsd
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_model_layout
from deepjump.utils import resolve_device
from scripts.endpoint_grid_eval import _center, _sha256
from scripts.rollout_robustness_eval import _local_geometry


EXPECTED_DOMAINS = 20
EXPECTED_STARTS = 3
MAX_PEAK_MEMORY_FRACTION = 0.80
MAX_PROJECTED_MINUTES = 50.0


def _panel_starts(frames: int, delta: int) -> list[int]:
    """Return three fixed, distinct starts spanning all valid source frames."""
    last = int(frames) - 1 - int(delta)
    if last < 0:
        raise ValueError("trajectory has no valid H1 pair")
    starts = np.linspace(0, last, EXPECTED_STARTS, dtype=int).tolist()
    if len(set(starts)) != EXPECTED_STARTS:
        raise ValueError("trajectory cannot provide three distinct H1 starts")
    return starts


def _domain_summary(cells: list[dict]) -> dict:
    deltas = [float(cell["mean_model_minus_noop"]) for cell in cells]
    return {
        "cells": len(deltas),
        "mean_model_minus_noop": statistics.fmean(deltas),
        "cells_better_than_noop": sum(value < 0 for value in deltas),
    }


def _runtime_probe_status(peak_memory_fraction: float, projected_minutes: float) -> str:
    values = (float(peak_memory_fraction), float(projected_minutes))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("runtime probe metrics must be finite and non-negative")
    if peak_memory_fraction > MAX_PEAK_MEMORY_FRACTION:
        return "STOP_MEMORY_HEADROOM"
    if projected_minutes > MAX_PROJECTED_MINUTES:
        return "STOP_PROJECTED_RUNTIME"
    return "PASS_RUNTIME_PROBE"


def _evaluate_cell(
    *,
    model: DeepJumpLite,
    handle: _DomainHandle,
    layout,
    residue_index: torch.Tensor,
    atom_mask: torch.Tensor,
    bond_mask: torch.Tensor,
    temperature: int,
    replica: int,
    delta: int,
    canon_symmetric: bool,
    device: torch.device,
) -> dict:
    available = handle.replicas(temperature, [replica])
    if len(available) != 1:
        raise ValueError(f"missing trajectory {handle.name}/{temperature}/{replica}")
    frames = int(available[0][2])
    starts = _panel_starts(frames, delta)
    source_positions = []
    source_vectors = []
    target_positions = []
    for start in starts:
        source_coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, start))
        )
        target_coordinates = torch.from_numpy(
            np.asarray(handle.coords(temperature, replica, start + delta))
        )
        P_t, V_t = apply_model_layout(
            source_coordinates, layout, canon_symmetric=canon_symmetric
        )
        P_1, _ = apply_model_layout(
            target_coordinates, layout, canon_symmetric=canon_symmetric
        )
        source_positions.append(_center(P_t))
        source_vectors.append(V_t)
        target_positions.append(_center(P_1))
    P_t = torch.stack(source_positions).to(device)
    V_t = torch.stack(source_vectors).to(device)
    P_1 = torch.stack(target_positions).to(device)
    n = layout.num_residues
    batch = {
        "P_t": P_t,
        "V_t": V_t,
        "res_index": residue_index.unsqueeze(0).repeat(EXPECTED_STARTS, 1),
        "delta_ns": torch.full((EXPECTED_STARTS,), float(delta), device=device),
        "residue_mask": torch.ones(EXPECTED_STARTS, n, dtype=torch.bool, device=device),
        "atom_mask": atom_mask.unsqueeze(0).repeat(EXPECTED_STARTS, 1, 1),
        "bond_mask": bond_mask.unsqueeze(0).repeat(EXPECTED_STARTS, 1),
    }
    prediction, _ = model.sample(batch, steps=1, mode="mean")
    model_rmsd = [
        float(aligned_ca_rmsd(prediction[i], P_1[i]).item())
        for i in range(EXPECTED_STARTS)
    ]
    noop_rmsd = [
        float(aligned_ca_rmsd(P_t[i], P_1[i]).item())
        for i in range(EXPECTED_STARTS)
    ]
    paired = [
        model_value - noop_value
        for model_value, noop_value in zip(model_rmsd, noop_rmsd)
    ]
    geometry = _local_geometry(prediction, P_1, batch["bond_mask"])
    return {
        "domain": handle.name,
        "temperature": int(temperature),
        "replica": int(replica),
        "frames": frames,
        "starts": starts,
        "model_rmsd_by_start": model_rmsd,
        "noop_rmsd_by_start": noop_rmsd,
        "model_minus_noop_by_start": paired,
        "mean_model_minus_noop": statistics.fmean(paired),
        "bond_mean": geometry["bond_mean"],
        "bond_max": geometry["bond_max"],
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--starts", type=int, default=EXPECTED_STARTS)
    parser.add_argument("--runtime-probe-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.starts != EXPECTED_STARTS:
        parser.error("the frozen 20-domain endpoint panel requires three starts per cell")

    checkpoint_sha256 = _sha256(args.ckpt)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["cfg"]["model"]
    data_cfg = checkpoint["cfg"]["data"]
    if data_cfg.get("domains") != ["1a0hA01"]:
        raise ValueError("checkpoint training domain mismatch")
    delta = require_single_delta(data_cfg["delta_frames"])
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    if len(domain_ids) != EXPECTED_DOMAINS or len(set(domain_ids)) != EXPECTED_DOMAINS:
        raise ValueError("endpoint panel requires exactly 20 unique frozen domains")
    if "1a0hA01" in domain_ids:
        raise ValueError("endpoint panel must exclude the checkpoint training domain")
    paths = resolve_frozen_domains(discover_domains(data_cfg["root"]), domain_ids)

    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if device.type != "cuda":
        raise ValueError("runtime probe requires a CUDA device")
    residue_counts = []
    for path in paths:
        probe_handle = _DomainHandle(path)
        try:
            residue_counts.append((probe_handle.layout.num_residues, path))
        finally:
            probe_handle.close()
    largest_residues, largest_path = max(residue_counts, key=lambda item: item[0])
    probe_handle = _DomainHandle(largest_path)
    try:
        probe_layout = probe_handle.layout
        probe_residue_index = torch.as_tensor(probe_layout.res_index, device=device)
        probe_atom_mask = torch.as_tensor(probe_layout.atom_mask, device=device)
        probe_bond_mask = torch.as_tensor(probe_layout.bond_mask, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        probe_started = time.perf_counter()
        _evaluate_cell(
            model=model,
            handle=probe_handle,
            layout=probe_layout,
            residue_index=probe_residue_index,
            atom_mask=probe_atom_mask,
            bond_mask=probe_bond_mask,
            temperature=temperatures[0],
            replica=replicas[0],
            delta=delta,
            canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
            device=device,
        )
        torch.cuda.synchronize(device)
        probe_seconds = time.perf_counter() - probe_started
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    finally:
        probe_handle.close()
    peak_fraction = peak_bytes / total_bytes
    projected_minutes = probe_seconds * EXPECTED_DOMAINS * 25 / 60
    probe_status = _runtime_probe_status(peak_fraction, projected_minutes)
    runtime_probe = {
        "status": probe_status,
        "domain": Path(largest_path).stem.replace("mdcath_dataset_", ""),
        "residues": largest_residues,
        "batch_size": EXPECTED_STARTS,
        "cell_seconds": probe_seconds,
        "projected_500_cell_minutes": projected_minutes,
        "peak_memory_bytes": peak_bytes,
        "total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_fraction,
        "limits": {
            "max_peak_memory_fraction": MAX_PEAK_MEMORY_FRACTION,
            "max_projected_minutes": MAX_PROJECTED_MINUTES,
        },
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
            n = layout.num_residues
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
                    "residues_evaluated": n,
                },
                "summary": _domain_summary(cells),
                "cells": cells,
            })
        finally:
            handle.close()

    result = {
        "checkpoint": args.ckpt,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "delta_frames": delta,
        "settings": {
            "starts": args.starts,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
        },
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_list_sha256,
            "ids": domain_ids,
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
