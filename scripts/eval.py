#!/usr/bin/env python
"""Single-step evaluation of a trained DeepJump-lite checkpoint vs baselines.

Reports CA RMSD, pairwise-distance MAE, and native-contact recovery (FNC) for:
  * model   : x1 prediction (deterministic, noise off)
  * no-op   : X_hat = X_t          (structure-preserving lower bound on motion)
  * noise   : X_hat = X_t + eps    (negative reference)

    python scripts/eval.py --ckpt runs/ca_delta1/last.ckpt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from deepjump.config import Config, DataConfig, ModelConfig, TrainConfig
from deepjump.data import MdcathPairDataset, collate_pairs, discover_domains
from deepjump.metrics import (
    contact_fraction_native,
    masked_ca_rmsd,
    masked_pair_distance_mae,
)
from deepjump.model import DeepJumpLite
from deepjump.utils import move_batch, resolve_device, split_domains


def _cfg_from_ckpt(d: dict) -> Config:
    return Config(
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        train=TrainConfig(**d["train"]),
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--max-batches", type=int, default=50)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _cfg_from_ckpt(ck["cfg"])
    device = resolve_device(cfg.train.device)

    model = DeepJumpLite(
        cfg.model, noise_sigma=cfg.data.noise_sigma, predict_heavy=cfg.model.predict_heavy
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    model.noise_sigma = 0.0  # deterministic single-step prediction

    files = discover_domains(cfg.data.root)
    train_files, val_files = split_domains(files, cfg.data.val_fraction, cfg.data.seed)
    files = val_files if args.split == "val" else train_files
    ds = MdcathPairDataset(
        files, cfg.data.temperatures, cfg.data.replicas, cfg.data.delta_frames,
        cfg.data.crop_length, align=True, seed=99,
    )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=collate_pairs)

    agg = {k: [] for k in
           ["model_rmsd", "noop_rmsd", "noise_rmsd",
            "model_pdmae", "noop_pdmae", "model_fnc", "noop_fnc"]}
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        batch = move_batch(batch, device)
        m = batch["residue_mask"]
        P_t, P_1 = batch["P_t"], batch["P_1"]
        P_hat = model(batch, tau=torch.zeros(P_t.shape[0], device=device))["P_hat_1"]
        P_noise = P_t + cfg.data.noise_sigma * torch.randn_like(P_t)

        agg["model_rmsd"].append(masked_ca_rmsd(P_hat, P_1, m).mean().item())
        agg["noop_rmsd"].append(masked_ca_rmsd(P_t, P_1, m).mean().item())
        agg["noise_rmsd"].append(masked_ca_rmsd(P_noise, P_1, m).mean().item())
        agg["model_pdmae"].append(masked_pair_distance_mae(P_hat, P_1, m).mean().item())
        agg["noop_pdmae"].append(masked_pair_distance_mae(P_t, P_1, m).mean().item())
        agg["model_fnc"].append(contact_fraction_native(P_hat, P_1, m).mean().item())
        agg["noop_fnc"].append(contact_fraction_native(P_t, P_1, m).mean().item())

    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"\nsplit={args.split}  domains={[f.stem.replace('mdcath_dataset_','') for f in files]}")
    print(f"{'metric':<22}{'model':>10}{'no-op':>10}{'noise':>10}")
    print("-" * 52)
    print(f"{'CA RMSD (A) down':<22}{mean['model_rmsd']:>10.3f}{mean['noop_rmsd']:>10.3f}{mean['noise_rmsd']:>10.3f}")
    print(f"{'pair-dist MAE (A) down':<22}{mean['model_pdmae']:>10.3f}{mean['noop_pdmae']:>10.3f}{'-':>10}")
    print(f"{'contact FNC up':<22}{mean['model_fnc']:>10.3f}{mean['noop_fnc']:>10.3f}{'-':>10}")
    delta = mean["noop_rmsd"] - mean["model_rmsd"]
    print(
        f"\nmodel single-step CA RMSD vs no-op: {delta:+.3f} A "
        f"({'better' if delta > 0 else 'not better'} than no-op)."
    )
    print(
        "Note: a deterministic x1 predictor approximates E[X_{t+d}|X_t] ~ X_t for\n"
        "diffusive 1 ns dynamics, so single-step RMSD near no-op is expected;\n"
        "distributional/sampled evaluation (stage 2) is the meaningful test."
    )


if __name__ == "__main__":
    main()
