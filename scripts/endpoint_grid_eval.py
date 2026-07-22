#!/usr/bin/env python
"""Evaluate a deterministic H1 endpoint over one domain's complete mdCATH grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
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

try:
    from scripts.rollout_robustness_eval import _local_geometry
except ModuleNotFoundError:  # direct `python scripts/endpoint_grid_eval.py`
    from rollout_robustness_eval import _local_geometry


def _center(positions: torch.Tensor) -> torch.Tensor:
    return positions - positions.mean(dim=0, keepdim=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_cells(cells: list[dict]) -> dict:
    """Summarize cell-balanced paired model-minus-no-op RMSD."""
    deltas = [float(cell["mean_model_minus_noop"]) for cell in cells]
    if len(deltas) < 2 or not all(math.isfinite(value) for value in deltas):
        raise ValueError("endpoint grid requires at least two finite cell deltas")
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return {
        "cells": len(cells),
        "mean_model_minus_noop": mean_delta,
        "standard_error": standard_error,
        "absolute_mean_over_standard_error": (
            abs(mean_delta) / standard_error if standard_error > 0 else None
        ),
        "cells_better_than_noop": sum(value < 0 for value in deltas),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.starts != 5:
        parser.error("the frozen endpoint grid requires exactly five starts per cell")
    checkpoint_sha256 = _sha256(args.ckpt)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["cfg"]["model"]
    data_cfg = checkpoint["cfg"]["data"]
    delta = require_single_delta(data_cfg["delta_frames"])
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    if len(domain_ids) != 1:
        raise ValueError("endpoint grid requires exactly one frozen domain")
    paths = resolve_frozen_domains(discover_domains(data_cfg["root"]), domain_ids)

    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    handle = _DomainHandle(paths[0])
    layout = handle.layout
    n = min(layout.num_residues, int(data_cfg["crop_length"]))
    offset = max(0, (layout.num_residues - n) // 2)
    residue_slice = slice(offset, offset + n)
    residue_index = torch.as_tensor(layout.res_index[residue_slice], device=device)
    atom_mask = torch.as_tensor(layout.atom_mask[residue_slice], device=device)
    bond_mask = torch.as_tensor(
        layout.bond_mask[residue_slice.start:residue_slice.stop - 1], device=device
    )
    cells = []
    try:
        for temperature in temperatures:
            for replica in replicas:
                available = handle.replicas(temperature, [replica])
                if len(available) != 1:
                    raise ValueError(
                        f"missing trajectory {handle.name}/{temperature}/{replica}"
                    )
                frames = int(available[0][2])
                last = frames - 1 - delta
                if last < 0:
                    raise ValueError(
                        f"trajectory {handle.name}/{temperature}/{replica} has no H1 pair"
                    )
                starts = np.linspace(0, last, args.starts, dtype=int)
                if len(set(starts.tolist())) != args.starts:
                    raise ValueError(
                        f"trajectory {handle.name}/{temperature}/{replica} "
                        "cannot provide five distinct starts"
                    )
                source_positions = []
                source_vectors = []
                target_positions = []
                for start in starts:
                    source_coordinates = torch.from_numpy(
                        np.asarray(handle.coords(temperature, replica, int(start)))
                    )
                    target_coordinates = torch.from_numpy(
                        np.asarray(handle.coords(temperature, replica, int(start + delta)))
                    )
                    P_t, V_t = apply_model_layout(
                        source_coordinates,
                        layout,
                        canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                    )
                    P_1, _ = apply_model_layout(
                        target_coordinates,
                        layout,
                        canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                    )
                    source_positions.append(_center(P_t[residue_slice]))
                    source_vectors.append(V_t[residue_slice])
                    target_positions.append(_center(P_1[residue_slice]))

                P_t = torch.stack(source_positions).to(device)
                V_t = torch.stack(source_vectors).to(device)
                P_1 = torch.stack(target_positions).to(device)
                batch = {
                    "P_t": P_t,
                    "V_t": V_t,
                    "res_index": residue_index[None].repeat(args.starts, 1),
                    "delta_ns": torch.full(
                        (args.starts,), float(delta), device=device
                    ),
                    "residue_mask": torch.ones(
                        args.starts, n, dtype=torch.bool, device=device
                    ),
                    "atom_mask": atom_mask[None].repeat(args.starts, 1, 1),
                    "bond_mask": bond_mask[None].repeat(args.starts, 1),
                }
                prediction, _ = model.sample(batch, steps=1, mode="mean")
                model_rmsd = [
                    float(aligned_ca_rmsd(prediction[i], P_1[i]).item())
                    for i in range(args.starts)
                ]
                noop_rmsd = [
                    float(aligned_ca_rmsd(P_t[i], P_1[i]).item())
                    for i in range(args.starts)
                ]
                paired = [model - noop for model, noop in zip(model_rmsd, noop_rmsd)]
                geometry = _local_geometry(prediction, P_1, batch["bond_mask"])
                cells.append({
                    "domain": handle.name,
                    "temperature": int(temperature),
                    "replica": int(replica),
                    "frames": frames,
                    "starts": starts.tolist(),
                    "model_rmsd_by_start": model_rmsd,
                    "noop_rmsd_by_start": noop_rmsd,
                    "model_minus_noop_by_start": paired,
                    "mean_model_minus_noop": statistics.fmean(paired),
                    "bond_mean": geometry["bond_mean"],
                    "bond_max": geometry["bond_max"],
                })
    finally:
        handle.close()

    result = {
        "checkpoint": args.ckpt,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "delta_frames": delta,
        "settings": {"starts": args.starts, "method": "mean", "source_noise": False},
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_list_sha256,
            "ids": domain_ids,
        },
        "grid": {"temperatures": temperatures, "replicas": replicas},
        "preprocessing": {
            "canon_symmetric": bool(data_cfg.get("canon_symmetric", False)),
            "residues_total": layout.num_residues,
            "residues_evaluated": n,
        },
        "summary": summarize_cells(cells),
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
