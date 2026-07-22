#!/usr/bin/env python
"""Measure deterministic teacher updates in the training-aligned source frame."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path

import numpy as np
import torch

from deepjump.config import ModelConfig
from deepjump.data import discover_domains
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import (
    load_frozen_domain_ids,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.model import DeepJumpLite
from deepjump.metrics import aligned_ca_rmsd
from deepjump.representation import apply_model_layout, kabsch_rotation
from deepjump.utils import resolve_device
from scripts.rollout_robustness_eval import (
    _local_geometry,
    _local_geometry_by_start,
    select_validation_domains,
)


EPS = 1e-12


def _center_update(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=-2, keepdim=True)


def teacher_update_statistics(
    source: torch.Tensor,
    prediction: torch.Tensor,
    aligned_target: torch.Tensor,
) -> dict[str, list[float]]:
    """Return per-start update statistics after removing residual translation."""
    if source.shape != prediction.shape or source.shape != aligned_target.shape:
        raise ValueError("source, prediction, and aligned target shapes must match")
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("teacher update tensors must have shape [B, N, 3]")
    u = _center_update(prediction - source)
    v = _center_update(aligned_target - source)
    dot = (u * v).sum(dim=(-2, -1)) / source.shape[-2]
    u_sq = u.square().sum(dim=(-2, -1)) / source.shape[-2]
    v_sq = v.square().sum(dim=(-2, -1)) / source.shape[-2]
    if bool((v_sq <= EPS).any()):
        raise ValueError("true transition update is too small for projection analysis")
    denom = (u_sq * v_sq).clamp_min(EPS).sqrt()
    cosine = torch.where(u_sq > EPS, dot / denom, torch.zeros_like(dot))
    rho = (u_sq / v_sq).clamp_min(0).sqrt()
    raw_gain = 2 * dot - u_sq
    return {
        "dot_uv_by_start": dot.cpu().tolist(),
        "u_sq_by_start": u_sq.cpu().tolist(),
        "v_sq_by_start": v_sq.cpu().tolist(),
        "cosine_by_start": cosine.cpu().tolist(),
        "rho_by_start": rho.cpu().tolist(),
        "raw_gain_by_start": raw_gain.cpu().tolist(),
    }


def _full_align_target_to_source(
    source_full: torch.Tensor,
    target_full: torch.Tensor,
) -> torch.Tensor:
    """Align a full target structure to its source before any residue crop."""
    rotation = kabsch_rotation(target_full, source_full)
    return (rotation @ (target_full - target_full.mean(0, keepdim=True)).T).T


def _mean_rows(rows: list[list[float]]) -> list[float]:
    return [float(np.mean(row)) for row in rows]


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be 64 lowercase hex characters")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--domain-list", required=True)
    ap.add_argument("--domain-list-sha256", required=True)
    ap.add_argument("--domains", type=int, default=3)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--calibration-domain", default="1gxlA02")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if (args.domains, args.starts, args.steps) != (3, 2, 20):
        ap.error("the frozen projection panel requires 3 domains, 2 starts, and 20 steps")
    _validate_sha256(args.checkpoint_sha256, "checkpoint SHA256")
    checkpoint_sha256 = hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()
    if not hmac.compare_digest(checkpoint_sha256, args.checkpoint_sha256):
        raise ValueError("checkpoint SHA256 mismatch")

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["cfg"]["model"]
    data_cfg = checkpoint["cfg"]["data"]
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=data_cfg["noise_sigma"],
        predict_heavy=model_cfg["predict_heavy"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    delta = require_single_delta(data_cfg["delta_frames"])
    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    panel = resolve_frozen_domains(discover_domains(data_cfg["root"]), domain_ids)
    chosen = select_validation_domains(panel, args.domains)

    domain_rows: list[dict] = []
    cached: list[dict] = []
    for path in chosen:
        handle = _DomainHandle(path)
        layout = handle.layout
        temp, replica = data_cfg["temperatures"][0], data_cfg["replicas"][0]
        n_frames = handle.replicas(temp, [replica])[0][2]
        last = n_frames - 1 - args.steps * delta
        if last < 0:
            handle.close()
            raise ValueError(f"{handle.name} has too few frames")
        starts = np.linspace(0, last, min(args.starts, last + 1), dtype=int)
        if len(starts) != args.starts:
            handle.close()
            raise ValueError(f"{handle.name} does not provide two distinct starts")
        n = min(layout.num_residues, int(data_cfg["crop_length"]))
        offset = max(0, (layout.num_residues - n) // 2)
        residue_slice = slice(offset, offset + n)
        static = {
            "res_index": torch.as_tensor(
                layout.res_index[residue_slice], device=device
            )[None].repeat(args.starts, 1),
            "delta_ns": torch.full((args.starts,), float(delta), device=device),
            "residue_mask": torch.ones(args.starts, n, dtype=torch.bool, device=device),
            "atom_mask": torch.as_tensor(
                layout.atom_mask[residue_slice], device=device
            )[None].repeat(args.starts, 1, 1),
            "bond_mask": torch.as_tensor(
                layout.bond_mask[residue_slice.start:residue_slice.stop - 1],
                device=device,
            )[None].repeat(args.starts, 1),
        }
        metric_rows = {name: [] for name in (
            "dot_uv_by_start", "u_sq_by_start", "v_sq_by_start",
            "cosine_by_start", "rho_by_start", "raw_gain_by_start",
            "teacher_aligned_rmsd_by_start",
            "persistence_aligned_rmsd_by_start",
        )}
        transition_cache = []
        for horizon in range(args.steps):
            sources = []
            source_vectors = []
            aligned_targets = []
            for start in starts:
                source_coords = torch.from_numpy(np.asarray(
                    handle.coords(temp, replica, int(start + horizon * delta))
                ))
                target_coords = torch.from_numpy(np.asarray(
                    handle.coords(temp, replica, int(start + (horizon + 1) * delta))
                ))
                source_full, source_v_full = apply_model_layout(
                    source_coords,
                    layout,
                    canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                )
                target_full, _ = apply_model_layout(
                    target_coords,
                    layout,
                    canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                )
                aligned_full = _full_align_target_to_source(source_full, target_full)
                source_crop = source_full[residue_slice]
                source_crop = source_crop - source_crop.mean(0, keepdim=True)
                target_crop = aligned_full[residue_slice]
                target_crop = target_crop - target_crop.mean(0, keepdim=True)
                sources.append(source_crop)
                source_vectors.append(source_v_full[residue_slice])
                aligned_targets.append(target_crop)
            source = torch.stack(sources).to(device)
            source_vectors_tensor = torch.stack(source_vectors).to(device)
            target = torch.stack(aligned_targets).to(device)
            prediction, _ = model.sample(
                {"P_t": source, "V_t": source_vectors_tensor, **static},
                steps=1,
                mode="mean",
            )
            stats = teacher_update_statistics(source, prediction, target)
            stats["teacher_aligned_rmsd_by_start"] = [
                float(aligned_ca_rmsd(prediction[index], target[index]).item())
                for index in range(args.starts)
            ]
            stats["persistence_aligned_rmsd_by_start"] = [
                float(aligned_ca_rmsd(source[index], target[index]).item())
                for index in range(args.starts)
            ]
            for name, values in stats.items():
                metric_rows[name].append([float(value) for value in values])
            transition_cache.append((source.cpu(), prediction.cpu(), target.cpu()))
        cached.append({
            "domain": handle.name,
            "transitions": transition_cache,
            "bond_mask": static["bond_mask"].cpu(),
        })
        domain_rows.append({
            "domain": handle.name,
            "residues_total": layout.num_residues,
            "residues_evaluated": n,
            "frames": n_frames,
            "starts": starts.tolist(),
            "metrics": {
                **metric_rows,
                **{
                    name.removesuffix("_by_start"): _mean_rows(values)
                    for name, values in metric_rows.items()
                },
            },
        })
        handle.close()

    calibration = next(
        (row for row in domain_rows if row["domain"] == args.calibration_domain), None
    )
    if calibration is None:
        raise ValueError("calibration domain is absent from the frozen spread panel")
    calibration_metrics = calibration["metrics"]
    numerator = sum(sum(row) for row in calibration_metrics["dot_uv_by_start"])
    denominator = sum(sum(row) for row in calibration_metrics["u_sq_by_start"])
    if denominator <= EPS:
        raise ValueError("calibration teacher updates have zero norm")
    alpha = float(numerator / denominator)

    for row, cache in zip(domain_rows, cached):
        metrics = row["metrics"]
        scaled_gain_by_start = [
            [2 * alpha * dot - alpha * alpha * u_sq for dot, u_sq in zip(dots, norms)]
            for dots, norms in zip(
                metrics["dot_uv_by_start"], metrics["u_sq_by_start"]
            )
        ]
        metrics["scaled_gain_by_start"] = scaled_gain_by_start
        metrics["scaled_gain"] = _mean_rows(scaled_gain_by_start)
        bond_mean_by_start = []
        bond_max_by_start = []
        scaled_aligned_rmsd_by_start = []
        bond_mean = []
        bond_max = []
        for source, prediction, target in cache["transitions"]:
            scaled = source + alpha * (prediction - source)
            geometry = _local_geometry(scaled, target, cache["bond_mask"])
            geometry_by_start = _local_geometry_by_start(
                scaled, target, cache["bond_mask"]
            )
            scaled_aligned_rmsd_by_start.append([
                float(aligned_ca_rmsd(scaled[index], target[index]).item())
                for index in range(args.starts)
            ])
            bond_mean.append(geometry["bond_mean"])
            bond_max.append(geometry["bond_max"])
            bond_mean_by_start.append(geometry_by_start["bond_mean_by_start"])
            bond_max_by_start.append(geometry_by_start["bond_max_by_start"])
        metrics["scaled_bond_mean"] = bond_mean
        metrics["scaled_bond_max"] = bond_max
        metrics["scaled_bond_mean_by_start"] = bond_mean_by_start
        metrics["scaled_bond_max_by_start"] = bond_max_by_start
        metrics["scaled_aligned_rmsd_by_start"] = scaled_aligned_rmsd_by_start
        metrics["scaled_aligned_rmsd"] = _mean_rows(scaled_aligned_rmsd_by_start)

    result = {
        "checkpoint": args.ckpt,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint["step"],
        "settings": vars(args),
        "preprocessing": {
            "canon_symmetric": bool(data_cfg.get("canon_symmetric", False)),
            "target_alignment": "full_structure_target_to_source_before_crop",
            "update_translation": "per_crop_update_mean_removed",
        },
        "delta_frames": delta,
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_list_sha256,
            "count": len(domain_ids),
            "evaluated_count": len(chosen),
        },
        "calibration": {
            "domain": args.calibration_domain,
            "alpha": alpha,
            "formula": "sum(dot_uv)/sum(u_sq)",
        },
        "domains": domain_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
