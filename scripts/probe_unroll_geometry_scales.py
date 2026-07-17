#!/usr/bin/env python
"""Calibrate local-geometry loss weights from honest step-2 parameter gradients."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from deepjump.config import Config, DataConfig, ModelConfig, TrainConfig
from deepjump.data import MdcathPairDataset, collate_pairs, discover_domains
from deepjump.losses import allatom_pairwise_huber_loss, ca_local_geometry_huber_losses
from deepjump.model import DeepJumpLite
from deepjump.utils import move_batch, resolve_device, split_domains


def _cfg_from_ckpt(raw: dict) -> Config:
    return Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )


def _flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(param) if grad is None else grad).detach().float().flatten().cpu()
        for param, grad in zip(params, grads)
    ])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    return float((a @ b / denom.clamp_min(1e-12)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--length-fraction", type=float, default=0.02)
    parser.add_argument("--angle-fraction", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _cfg_from_ckpt(checkpoint["cfg"])
    if cfg.data.unroll < 2:
        raise ValueError("checkpoint config must expose an honest step-2 target")
    device = resolve_device(cfg.train.device)
    model = DeepJumpLite(
        cfg.model, noise_sigma=cfg.data.noise_sigma,
        predict_heavy=cfg.model.predict_heavy,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    files = discover_domains(cfg.data.root)
    _, val_files = split_domains(files, cfg.data.val_fraction, cfg.data.seed)
    manifest = json.loads(Path(cfg.data.manifest).read_text()) if cfg.data.manifest else None
    dataset = MdcathPairDataset(
        val_files, cfg.data.temperatures, cfg.data.replicas, cfg.data.delta_frames,
        cfg.data.crop_length, align=True, canon_symmetric=cfg.data.canon_symmetric,
        manifest=manifest, seed=cfg.data.seed + 99, unroll=cfg.data.unroll,
    )
    # One trajectory from each distinct domain.  Taking the first entries from
    # ``stratified_indices`` would still exhaust all temperature/replica
    # trajectories of the first domain before reaching the next one.
    rng = np.random.default_rng(cfg.data.seed + 2)
    indices: list[int] = []
    seen_files: set[int] = set()
    start = 0
    for trajectory, stop_value in zip(dataset._traj, dataset._cum):
        stop = int(stop_value)
        file_index = trajectory[0]
        if file_index not in seen_files:
            indices.append(start + int(rng.integers(0, stop - start)))
            seen_files.add(file_index)
            if len(indices) == args.samples:
                break
        start = stop
    if len(indices) < args.samples:
        raise ValueError(f"requested {args.samples} domains, found only {len(indices)}")
    cpu_batch = next(iter(DataLoader(
        Subset(dataset, indices), batch_size=len(indices), collate_fn=collate_pairs, shuffle=False,
    )))
    batch = move_batch(cpu_batch, device)
    tau0 = torch.zeros(len(indices), device=device)
    with torch.no_grad():
        first = model(batch, tau=tau0)
    step2_batch = {
        **batch,
        "P_t": first["P_hat_1"].detach(),
        "V_t": first["V_hat_1"].detach(),
        "P_1": batch["P_2"],
        "V_1": batch["V_2"],
    }
    second = model(step2_batch, tau=tau0)
    main_loss = allatom_pairwise_huber_loss(
        second["P_hat_1"], second["V_hat_1"], batch["P_2"], batch["V_2"],
        batch["atom_mask"], batch["residue_mask"], cfg.model.dist_cutoff,
        cfg.train.huber_delta,
    )
    length_loss, angle_loss = ca_local_geometry_huber_losses(
        second["P_hat_1"], batch["P_2"], batch["residue_mask"], batch["bond_mask"],
        getattr(cfg.train, "geom_huber_delta", 0.05),
    )
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    main_grad = _flat_grad(main_loss, params, retain_graph=True)
    length_grad = _flat_grad(length_loss, params, retain_graph=True)
    angle_grad = _flat_grad(angle_loss, params, retain_graph=False)
    main_norm = float(main_grad.norm().item())
    length_norm = float(length_grad.norm().item())
    angle_norm = float(angle_grad.norm().item())
    length_weight = args.length_fraction * main_norm / max(length_norm, 1e-12)
    angle_weight = args.angle_fraction * main_norm / max(angle_norm, 1e-12)
    combined = length_weight * length_grad + angle_weight * angle_grad
    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint["step"],
        "sample_indices": indices,
        "domains": cpu_batch["domains"],
        "main_loss": float(main_loss.detach().item()),
        "length_loss": float(length_loss.detach().item()),
        "angle_loss": float(angle_loss.detach().item()),
        "main_grad_norm": main_norm,
        "length_grad_norm": length_norm,
        "angle_grad_norm": angle_norm,
        "main_length_grad_cosine": _cosine(main_grad, length_grad),
        "main_angle_grad_cosine": _cosine(main_grad, angle_grad),
        "length_angle_grad_cosine": _cosine(length_grad, angle_grad),
        "length_weight": length_weight,
        "angle_weight": angle_weight,
        "combined_to_main_grad_ratio": float(combined.norm().item() / max(main_norm, 1e-12)),
        "main_combined_grad_cosine": _cosine(main_grad, combined),
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise RuntimeError("non-finite loss or gradient diagnostic")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()
