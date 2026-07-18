#!/usr/bin/env python
"""Measure loss values and parameter-gradient scales for one fixed validation batch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from deepjump.config import Config, DataConfig, ModelConfig, TrainConfig
from deepjump.data import MdcathPairDataset, collate_pairs, discover_domains
from deepjump.losses import allatom_pairwise_huber_loss, ca_bond_length_huber_loss
from deepjump.model import DeepJumpLite
from deepjump.utils import move_batch, resolve_device, split_domains


def _cfg_from_ckpt(d: dict) -> Config:
    return Config(
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        train=TrainConfig(**d["train"]),
    )


def _filter_checkpoint_domains(files: list[Path], domains: list[str]) -> list[Path]:
    if not domains:
        return list(files)
    wanted = set(domains)
    filtered = [
        path
        for path in files
        if path.stem.replace("mdcath_dataset_", "") in wanted
    ]
    if not filtered:
        raise ValueError(f"checkpoint domains not found: {sorted(wanted)}")
    return filtered


def _gradient_norm(model: torch.nn.Module, loss: torch.Tensor, retain_graph: bool) -> float:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    squared = sum(
        parameter.grad.detach().float().square().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _cfg_from_ckpt(ck["cfg"])
    device = resolve_device(cfg.train.device)
    model = DeepJumpLite(
        cfg.model, noise_sigma=cfg.data.noise_sigma,
        predict_heavy=cfg.model.predict_heavy,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.train()

    files = _filter_checkpoint_domains(discover_domains(cfg.data.root), cfg.data.domains)
    _, val_files = split_domains(files, cfg.data.val_fraction, cfg.data.seed)
    manifest = json.loads(Path(cfg.data.manifest).read_text()) if cfg.data.manifest else None
    ds = MdcathPairDataset(
        val_files, cfg.data.temperatures, cfg.data.replicas, cfg.data.delta_frames,
        cfg.data.crop_length, align=True, canon_symmetric=cfg.data.canon_symmetric,
        manifest=manifest, seed=cfg.data.seed + 99,
    )
    indices = ds.stratified_indices(1, seed=cfg.data.seed + 2)[:args.samples]
    cpu_batch = next(iter(DataLoader(
        Subset(ds, indices), batch_size=args.samples, collate_fn=collate_pairs, shuffle=False,
    )))
    batch = move_batch(cpu_batch, device)
    tau0 = torch.zeros(len(indices), device=device)
    out = model(batch, tau=tau0)
    aa = allatom_pairwise_huber_loss(
        out["P_hat_1"], out["V_hat_1"], batch["P_1"], batch["V_1"],
        batch["atom_mask"], batch["residue_mask"],
        cutoff=cfg.model.dist_cutoff, delta=cfg.train.huber_delta,
    )
    bond = ca_bond_length_huber_loss(
        out["P_hat_1"], batch["P_1"], batch["residue_mask"], batch["bond_mask"],
        delta=cfg.train.huber_delta,
    )
    noop_aa = allatom_pairwise_huber_loss(
        batch["P_t"], batch["V_t"], batch["P_1"], batch["V_1"],
        batch["atom_mask"], batch["residue_mask"],
        cutoff=cfg.model.dist_cutoff, delta=cfg.train.huber_delta,
    )
    noop_bond = ca_bond_length_huber_loss(
        batch["P_t"], batch["P_1"], batch["residue_mask"], batch["bond_mask"],
        delta=cfg.train.huber_delta,
    )
    aa_grad = _gradient_norm(model, aa, retain_graph=True)
    bond_grad = _gradient_norm(model, bond, retain_graph=False)
    aa_value = float(aa.detach().item())
    bond_value = float(bond.detach().item())
    noop_aa_value = float(noop_aa.detach().item())
    noop_bond_value = float(noop_bond.detach().item())
    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": ck["step"],
        "sample_indices": indices,
        "domains": cpu_batch["domains"],
        "allatom_loss": aa_value,
        "bond_loss": bond_value,
        "noop_allatom_loss": noop_aa_value,
        "noop_bond_loss": noop_bond_value,
        "model_to_noop_allatom_loss_ratio": aa_value / max(noop_aa_value, 1e-12),
        "model_to_noop_bond_loss_ratio": bond_value / max(noop_bond_value, 1e-12),
        "bond_to_allatom_loss_ratio": bond_value / max(aa_value, 1e-12),
        "allatom_grad_norm": aa_grad,
        "bond_grad_norm": bond_grad,
        "bond_to_allatom_grad_norm_ratio": bond_grad / max(aa_grad, 1e-12),
        "equal_gradient_bond_weight": aa_grad / max(bond_grad, 1e-12),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    ds.close()


if __name__ == "__main__":
    main()
