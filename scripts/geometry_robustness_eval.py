#!/usr/bin/env python
"""Geometry-only rollout gate calibrated against real-vs-real mdCATH panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepjump.config import ModelConfig
from deepjump.data import discover_domains
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import (
    aggregate_geometry_panel,
    calibrate_geometry_envelope,
    geometry_frame_statistics,
    geometry_panel_passes,
    load_frozen_domain_ids,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_layout
from deepjump.sampling import rollout
from deepjump.utils import resolve_device


def _center(positions: torch.Tensor) -> torch.Tensor:
    return positions - positions.mean(dim=0, keepdim=True)


def _spread(paths: list[Path], count: int) -> list[Path]:
    positions = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=int)
    return [paths[index] for index in positions]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--domains", type=int, default=5)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--methods", default="mean,ode_1")
    parser.add_argument("--reference-frames", type=int, default=500)
    parser.add_argument("--calibration-draws", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--collision-distance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg, data_cfg = checkpoint["cfg"]["model"], checkpoint["cfg"]["data"]
    delta = require_single_delta(data_cfg["delta_frames"])
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=data_cfg["noise_sigma"],
        predict_heavy=model_cfg["predict_heavy"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    domain_ids, panel_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    paths = resolve_frozen_domains(discover_domains(data_cfg["root"]), domain_ids)
    chosen = _spread(paths, args.domains)
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    invalid = [value for value in methods if value != "mean" and not value.startswith("ode_")]
    if invalid:
        raise ValueError(f"unsupported methods: {invalid}")

    rows = []
    for domain_index, path in enumerate(chosen):
        handle = _DomainHandle(path)
        layout = handle.layout
        temperature, replica = data_cfg["temperatures"][0], data_cfg["replicas"][0]
        n_frames = handle.replicas(temperature, [replica])[0][2]
        n = min(layout.num_residues, int(data_cfg["crop_length"]))
        offset = max(0, (layout.num_residues - n) // 2)
        residue_slice = slice(offset, offset + n)
        bond_mask = np.asarray(
            layout.bond_mask[residue_slice.start:residue_slice.stop - 1], dtype=bool
        )

        reference_ids = np.linspace(
            0, n_frames - 1, min(args.reference_frames, n_frames), dtype=int
        )
        reference_positions = []
        for frame in reference_ids:
            coordinates = torch.from_numpy(
                np.asarray(handle.coords(temperature, replica, int(frame)))
            )
            positions, _ = apply_layout(coordinates, layout)
            reference_positions.append(_center(positions[residue_slice]).numpy())
        reference_statistics = geometry_frame_statistics(
            np.stack(reference_positions),
            bond_mask,
            collision_distance=args.collision_distance,
        )
        envelope = calibrate_geometry_envelope(
            reference_statistics,
            args.starts,
            draws=args.calibration_draws,
            alpha=args.alpha,
            seed=args.seed + domain_index,
        )

        start_ids = np.linspace(0, n_frames - 1, min(args.starts, n_frames), dtype=int)
        start_positions, start_velocities = [], []
        for frame in start_ids:
            coordinates = torch.from_numpy(
                np.asarray(handle.coords(temperature, replica, int(frame)))
            )
            positions, velocities = apply_layout(coordinates, layout)
            start_positions.append(_center(positions[residue_slice]))
            start_velocities.append(velocities[residue_slice])
        initial_positions = torch.stack(start_positions).to(device)
        initial_velocities = torch.stack(start_velocities).to(device)
        batch = {
            "P_t": initial_positions,
            "V_t": initial_velocities,
            "res_index": torch.as_tensor(
                layout.res_index[residue_slice], device=device
            )[None].repeat(len(start_ids), 1),
            "delta_ns": torch.full((len(start_ids),), float(delta), device=device),
            "residue_mask": torch.ones(
                len(start_ids), n, dtype=torch.bool, device=device
            ),
            "atom_mask": torch.as_tensor(
                layout.atom_mask[residue_slice], device=device
            )[None].repeat(len(start_ids), 1, 1),
            "bond_mask": torch.as_tensor(bond_mask, device=device)[None].repeat(
                len(start_ids), 1
            ),
        }

        trajectories = {"noop": [(initial_positions, initial_velocities)] * (args.steps + 1)}
        for method_index, method in enumerate(methods):
            if method == "mean":
                mode, ode_steps, generator = "mean", 1, None
            else:
                mode, ode_steps = "ode", int(method.split("_", 1)[1])
                generator = torch.Generator().manual_seed(
                    args.seed + domain_index * 1000 + method_index * 100
                )
            trajectories[method], _ = rollout(
                model,
                batch,
                n_steps=args.steps,
                ode_steps=ode_steps,
                mode=mode,
                gate=False,
                generator=generator,
            )

        domain_methods = {}
        for method, trajectory in trajectories.items():
            panels, checks, passes = [], [], []
            for positions, _ in trajectory:
                statistics = geometry_frame_statistics(
                    positions.float().cpu().numpy(),
                    bond_mask,
                    collision_distance=args.collision_distance,
                )
                panel = aggregate_geometry_panel(statistics)
                passed, metric_checks = geometry_panel_passes(panel, envelope)
                panels.append(panel)
                checks.append(metric_checks)
                passes.append(passed)
            domain_methods[method] = {
                "all_steps_pass": all(passes),
                "failed_steps": [index for index, passed in enumerate(passes) if not passed],
                "panels": panels,
                "checks": checks,
            }
        rows.append({
            "domain": handle.name,
            "temperature": temperature,
            "replica": replica,
            "frames": n_frames,
            "residues_total": layout.num_residues,
            "residues_evaluated": n,
            "reference_frames": len(reference_ids),
            "start_frames": start_ids.tolist(),
            "envelope": envelope,
            "methods": domain_methods,
        })
        handle.close()

    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint["step"],
        "delta_frames": delta,
        "domain_panel": {
            "path": args.domain_list,
            "sha256": panel_sha256,
            "count": len(domain_ids),
            "evaluated_count": len(chosen),
        },
        "settings": vars(args),
        "summary": {
            method: {
                "domains_all_steps_pass": sum(
                    row["methods"][method]["all_steps_pass"] for row in rows
                ),
                "domain_count": len(rows),
            }
            for method in ["noop", *methods]
        },
        "domains": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
