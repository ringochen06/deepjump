#!/usr/bin/env python
"""Deterministic, domain-stratified robustness evaluation for one checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from deepjump.config import Config, DataConfig, ModelConfig, TrainConfig
from deepjump.data import MdcathPairDataset, collate_pairs, discover_domains
from deepjump.evaluation import (
    load_frozen_domain_ids,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.metrics import contact_fraction_native, masked_ca_rmsd, masked_pair_distance_mae
from deepjump.model import DeepJumpLite
from deepjump.utils import move_batch, resolve_device


def _cfg_from_ckpt(d: dict) -> Config:
    return Config(
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        train=TrainConfig(**d["train"]),
    )


def _summarize(rows: list[dict], noop_by_domain: dict[str, float]) -> dict:
    metrics = ("rmsd", "pdmae", "fnc")
    result = {key: sum(row[key] for row in rows) / len(rows) for key in metrics}
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row["rmsd"])
    means = {domain: sum(values) / len(values) for domain, values in by_domain.items()}
    result["domains_better_than_noop"] = sum(
        means[domain] < noop_by_domain[domain] for domain in means
    )
    result["domain_count"] = len(means)
    return result


def _paired_domain_bootstrap(
    rows: list[dict], noop_by_domain: dict[str, float], seed: int = 20260717,
    draws: int = 10000,
) -> dict:
    """Bootstrap the domain-balanced RMSD gain (no-op minus model)."""
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row["rmsd"])
    domains = sorted(by_domain)
    gains = np.asarray([
        noop_by_domain[d] - sum(by_domain[d]) / len(by_domain[d]) for d in domains
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = gains[rng.integers(0, len(gains), size=(draws, len(gains)))].mean(1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_noop_minus_model": float(gains.mean()),
        "ci95": [float(lo), float(hi)],
        "domains": len(domains),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain-list", required=True)
    ap.add_argument("--domain-list-sha256", required=True)
    ap.add_argument("--samples-per-trajectory", type=int, default=1)
    ap.add_argument("--ode-steps", default="1,2,5")
    ap.add_argument("--sample-seed", type=int, default=20260716)
    ap.add_argument(
        "--noise-sigma", type=float, default=None,
        help="Override checkpoint source-noise sigma for inference-only calibration.",
    )
    ap.add_argument("--integrator", choices=("euler", "heun"), default="euler")
    ap.add_argument("--tau-max", type=float, default=1.0)
    ap.add_argument("--terminal-denoise", action="store_true")
    ap.add_argument("--drift-anchor", choices=("state", "conditioner"), default="state")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.noise_sigma is not None and args.noise_sigma < 0:
        ap.error("--noise-sigma must be non-negative")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = _cfg_from_ckpt(ck["cfg"])
    delta = require_single_delta(cfg.data.delta_frames)
    device = resolve_device(cfg.train.device)
    noise_sigma = cfg.data.noise_sigma if args.noise_sigma is None else args.noise_sigma
    model = DeepJumpLite(
        cfg.model, noise_sigma=noise_sigma,
        predict_heavy=cfg.model.predict_heavy,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    val_files = resolve_frozen_domains(discover_domains(cfg.data.root), domain_ids)
    manifest = None
    if cfg.data.manifest:
        manifest = json.loads(Path(cfg.data.manifest).expanduser().read_text())
    ds = MdcathPairDataset(
        val_files, cfg.data.temperatures, cfg.data.replicas, delta,
        cfg.data.crop_length, align=True, canon_symmetric=cfg.data.canon_symmetric,
        manifest=manifest, seed=99,
    )
    indices = ds.stratified_indices(args.samples_per_trajectory, seed=cfg.data.seed + 2)
    # Materialize once on CPU: every method sees identical frames and crops.
    batches = list(DataLoader(Subset(ds, indices), batch_size=cfg.train.batch_size,
                              collate_fn=collate_pairs, shuffle=False))

    methods = ["noop", "mean"] + [f"ode_{n}" for n in map(int, args.ode_steps.split(","))]
    all_rows: dict[str, list[dict]] = {name: [] for name in methods}
    for method in methods:
        generator = torch.Generator().manual_seed(args.sample_seed)
        for cpu_batch in batches:
            batch = move_batch(cpu_batch, device)
            P_t, P_1, mask = batch["P_t"], batch["P_1"], batch["residue_mask"]
            if method == "noop":
                pred = P_t
            elif method == "mean":
                saved = model.noise_sigma
                model.noise_sigma = 0.0
                pred = model.sample(batch, mode="mean")[0]
                model.noise_sigma = saved
            else:
                steps = int(method.split("_")[1])
                pred = model.sample(
                    batch, mode="ode", steps=steps, generator=generator,
                    integrator=args.integrator, tau_max=args.tau_max,
                    terminal_denoise=args.terminal_denoise,
                    drift_anchor=args.drift_anchor,
                )[0]
            values = {
                "rmsd": masked_ca_rmsd(pred, P_1, mask).float().cpu().tolist(),
                "pdmae": masked_pair_distance_mae(pred, P_1, mask).float().cpu().tolist(),
                "fnc": contact_fraction_native(pred, P_1, mask).float().cpu().tolist(),
            }
            for i, domain in enumerate(cpu_batch["domains"]):
                all_rows[method].append({"domain": domain, **{k: v[i] for k, v in values.items()}})

    noop_domain_values = defaultdict(list)
    for row in all_rows["noop"]:
        noop_domain_values[row["domain"]].append(row["rmsd"])
    noop_by_domain = {
        domain: sum(values) / len(values) for domain, values in noop_domain_values.items()
    }
    result = {
        "checkpoint": str(Path(args.ckpt)),
        "checkpoint_step": ck["step"],
        "delta_frames": delta,
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_list_sha256,
            "count": len(domain_ids),
            "evaluated_count": len(val_files),
        },
        "sample_seed": args.sample_seed,
        "sample_count": len(indices),
        "sampling": {
            "integrator": args.integrator, "tau_max": args.tau_max,
            "terminal_denoise": args.terminal_denoise,
            "drift_anchor": args.drift_anchor,
        },
        "metrics": {},
    }
    for method_index, (method, rows) in enumerate(all_rows.items()):
        summary = _summarize(rows, noop_by_domain)
        summary["paired_rmsd_gain"] = _paired_domain_bootstrap(
            rows, noop_by_domain, seed=args.sample_seed + method_index
        )
        result["metrics"][method] = summary
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    ds.close()


if __name__ == "__main__":
    main()
