#!/usr/bin/env python
"""Diagnose what the model learned: CA RMSD as a function of latent time tau.

A flow-matching x1-predictor should get MORE accurate as the interpolant input
X^tau approaches X_1 (large tau). The generative-critical regimes are tau=0
(predict X_1 from X_t alone) and the integrated ODE sample. This script prints
all of them next to the no-op baseline so the learning curve over tau is explicit.

    python scripts/diagnose_tau.py --ckpt runs/ca_delta1/last.ckpt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from deepjump.config import ModelConfig
from deepjump.data import MdcathPairDataset, collate_pairs, discover_domains
from deepjump.metrics import masked_ca_rmsd
from deepjump.model import DeepJumpLite
from deepjump.utils import move_batch, resolve_device, split_domains


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-batches", type=int, default=15)
    ap.add_argument("--sample-steps", type=int, default=20)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_model, cfg_data, cfg_train = ck["cfg"]["model"], ck["cfg"]["data"], ck["cfg"]["train"]
    device = resolve_device(cfg_train["device"])
    model = DeepJumpLite(
        ModelConfig(**cfg_model), noise_sigma=cfg_data["noise_sigma"],
        predict_heavy=cfg_model.get("predict_heavy", False),
    ).to(device)
    model.load_state_dict(ck["model"]); model.eval(); model.noise_sigma = 0.0

    files = discover_domains(cfg_data["root"])
    _, val_files = split_domains(files, cfg_data["val_fraction"], cfg_data["seed"])
    ds = MdcathPairDataset(val_files, cfg_data["temperatures"], cfg_data["replicas"],
                           cfg_data["delta_frames"], cfg_data["crop_length"], align=True, seed=99)
    loader = list(DataLoader(ds, batch_size=cfg_train["batch_size"], collate_fn=collate_pairs))
    batches = [move_batch(b, device) for b in loader[: args.max_batches]]

    def avg(pred):
        return sum(masked_ca_rmsd(pred(b), b["P_1"], b["residue_mask"]).mean().item()
                   for b in batches) / len(batches)

    print(f"{'query':<26}{'CA RMSD (A)':>12}")
    print("-" * 38)
    print(f"{'no-op (X_t)':<26}{avg(lambda b: b['P_t']):>12.3f}")
    for tau in [0.0, 0.25, 0.5, 0.75, 0.9]:
        def f(b, tau=tau):
            t = torch.full((b["P_t"].shape[0],), tau, device=device)
            return model(b, tau=t)["P_hat_1"]
        print(f"{'one-shot x1 @ tau=' + format(tau, '.2f'):<26}{avg(f):>12.3f}")
    print(f"{'ODE sample (' + str(args.sample_steps) + ' steps)':<26}"
          f"{avg(lambda b: model.sample(b, steps=args.sample_steps)[0]):>12.3f}")
    ds.close()


if __name__ == "__main__":
    main()
