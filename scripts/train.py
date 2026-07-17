#!/usr/bin/env python
"""Train DeepJump-lite (CA, delta=1 ns) on an mdCATH subset.

    python scripts/train.py --config configs/ca_delta1.yaml
    python scripts/train.py --config configs/ca_delta1.yaml --fast-dev   # overfit 1 batch
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepjump.config import load_config, to_dict
from deepjump.data import MdcathPairDataset, collate_pairs, discover_domains
from deepjump.losses import pairwise_vector_huber_loss
from deepjump.metrics import masked_ca_rmsd
from deepjump.model import DeepJumpLite
from deepjump.model.deepjump import count_parameters
from deepjump.training import total_loss
from deepjump.utils import move_batch, resolve_device, split_domains


def build_loaders(cfg):
    files = discover_domains(cfg.data.root)
    if cfg.data.domains:
        wanted = set(cfg.data.domains)
        files = [f for f in files if f.stem.replace("mdcath_dataset_", "") in wanted]
    if not files:
        raise SystemExit(f"no mdCATH files under {cfg.data.root}; run download_mdcath.py")
    train_files, val_files = split_domains(files, cfg.data.val_fraction, cfg.data.seed)

    def make(fs, seed):
        return MdcathPairDataset(
            fs, cfg.data.temperatures, cfg.data.replicas, cfg.data.delta_frames,
            cfg.data.crop_length, align=True, unroll=cfg.data.unroll,
            canon_symmetric=cfg.data.canon_symmetric, seed=seed,
        )

    train_ds, val_ds = make(train_files, cfg.data.seed), make(val_files, cfg.data.seed + 1)
    dl_kw = dict(batch_size=cfg.train.batch_size, collate_fn=collate_pairs, num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw),
        DataLoader(val_ds, shuffle=False, **dl_kw),
        [f.stem.replace("mdcath_dataset_", "") for f in train_files],
        [f.stem.replace("mdcath_dataset_", "") for f in val_files],
    )


@torch.no_grad()
def evaluate(model, loader, device, cfg, max_batches=20):
    """Honest single-step evaluation: predict x1 from X_t alone (tau=0, no noise).

    NOTE: we deliberately query at tau=0 (the true generative starting point).
    Averaging over random tau would leak the answer (at high tau the interpolant
    input is already close to X_1), giving a misleadingly low RMSD.
    """
    model.eval()
    saved_sigma = model.noise_sigma
    model.noise_sigma = 0.0
    losses, rmsds, noop_rmsds = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        tau0 = torch.zeros(batch["P_t"].shape[0], device=device)
        out = model(batch, tau=tau0)
        loss = pairwise_vector_huber_loss(
            out["P_hat_1"], batch["P_1"], batch["residue_mask"], cfg.train.huber_delta
        )
        losses.append(loss.item())
        rmsds.append(masked_ca_rmsd(out["P_hat_1"], batch["P_1"], batch["residue_mask"]).mean().item())
        noop_rmsds.append(masked_ca_rmsd(batch["P_t"], batch["P_1"], batch["residue_mask"]).mean().item())
    model.noise_sigma = saved_sigma
    model.train()
    return {
        "val_loss": sum(losses) / len(losses),
        "val_rmsd": sum(rmsds) / len(rmsds),
        "noop_rmsd": sum(noop_rmsds) / len(noop_rmsds),
    }


@torch.no_grad()
def fast_dev_metrics(model, batch, cfg):
    """Measure the one-batch gate at the honest generative endpoint (tau=0)."""
    tau0 = torch.zeros(batch["P_t"].shape[0], device=batch["P_t"].device)
    out = model(batch, tau=tau0)
    loss, _ = total_loss(out, batch, cfg, model)
    rmsd = masked_ca_rmsd(
        out["P_hat_1"], batch["P_1"], batch["residue_mask"]
    ).mean()
    return {"loss": loss.item(), "rmsd": rmsd.item()}


def fast_dev_gate_errors(report, max_loss_ratio=None, max_rmsd_ratio=None):
    """Return fail-closed reasons for a machine-readable fast-dev report."""
    errors = []
    for phase in ("initial", "final"):
        for metric in ("loss", "rmsd"):
            value = report[phase][metric]
            if not math.isfinite(value):
                errors.append(f"{phase} {metric} is non-finite: {value}")
    if errors:
        return errors
    if max_loss_ratio is not None and report["loss_ratio"] > max_loss_ratio:
        errors.append(
            f"loss ratio {report['loss_ratio']:.6g} exceeds {max_loss_ratio:.6g}"
        )
    if max_rmsd_ratio is not None and report["rmsd_ratio"] > max_rmsd_ratio:
        errors.append(
            f"RMSD ratio {report['rmsd_ratio']:.6g} exceeds {max_rmsd_ratio:.6g}"
        )
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fast-dev", action="store_true", help="overfit a single batch")
    ap.add_argument(
        "--fast-dev-max-loss-ratio", type=float, default=None,
        help="with --fast-dev, fail unless final tau=0 loss / initial loss is at most this",
    )
    ap.add_argument(
        "--fast-dev-max-rmsd-ratio", type=float, default=None,
        help="with --fast-dev, fail unless final tau=0 RMSD / initial RMSD is at most this",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--lr", type=float, default=None, help="override train.lr")
    args = ap.parse_args()
    if not args.fast_dev and (
        args.fast_dev_max_loss_ratio is not None
        or args.fast_dev_max_rmsd_ratio is not None
    ):
        ap.error("fast-dev thresholds require --fast-dev")

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.lr is not None:
        cfg.train.lr = args.lr
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"device: {device}")

    train_loader, val_loader, train_doms, val_doms = build_loaders(cfg)
    print(f"train domains: {train_doms}  |  val domains: {val_doms}")
    print(f"train pairs: {len(train_loader.dataset)}  val pairs: {len(val_loader.dataset)}")

    model = DeepJumpLite(
        cfg.model, noise_sigma=cfg.data.noise_sigma, predict_heavy=cfg.model.predict_heavy
    ).to(device)
    print(f"model params: {count_parameters(model):,}  predict_heavy={cfg.model.predict_heavy}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    history = []

    if args.fast_dev:
        # Sanity: a single batch, no noise -> loss must collapse toward 0.
        model.noise_sigma = 0.0
        batch = move_batch(next(iter(train_loader)), device)
        initial = fast_dev_metrics(model, batch, cfg)
        print(
            "fast-dev: overfitting one batch "
            f"(initial tau=0 loss={initial['loss']:.6f}, rmsd={initial['rmsd']:.6f})"
        )
        for step in range(cfg.train.max_steps):
            out = model(batch, tau=torch.rand(batch["P_t"].shape[0], device=device))
            loss, _ = total_loss(out, batch, cfg, model)
            if not torch.isfinite(loss):
                raise RuntimeError(f"fast-dev loss became non-finite at step {step}: {loss}")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if step % cfg.train.log_every == 0 or step == cfg.train.max_steps - 1:
                rmsd = masked_ca_rmsd(out["P_hat_1"], batch["P_1"], batch["residue_mask"]).mean().item()
                print(f"  step {step:4d}  loss {loss.item():.4f}  rmsd {rmsd:.3f}")
        final = fast_dev_metrics(model, batch, cfg)
        loss_ratio = final["loss"] / initial["loss"] if initial["loss"] > 0 else math.inf
        rmsd_ratio = final["rmsd"] / initial["rmsd"] if initial["rmsd"] > 0 else math.inf
        report = {
            "status": "PENDING",
            "steps": cfg.train.max_steps,
            "evaluation_tau": 0.0,
            "initial": initial,
            "final": final,
            "loss_ratio": loss_ratio,
            "rmsd_ratio": rmsd_ratio,
            "thresholds": {
                "max_loss_ratio": args.fast_dev_max_loss_ratio,
                "max_rmsd_ratio": args.fast_dev_max_rmsd_ratio,
            },
        }
        errors = fast_dev_gate_errors(
            report, args.fast_dev_max_loss_ratio, args.fast_dev_max_rmsd_ratio
        )
        report["status"] = "PASS" if not errors else "FAIL"
        report["errors"] = errors
        (out_dir / "fast_dev.json").write_text(json.dumps(report, indent=2))
        print(
            f"fast-dev {report['status']}: final tau=0 loss={final['loss']:.6f} "
            f"({loss_ratio:.4f}x), rmsd={final['rmsd']:.6f} ({rmsd_ratio:.4f}x)"
        )
        if errors:
            raise SystemExit("fast-dev gate failed: " + "; ".join(errors))
        return

    step = 0
    train_iter = iter(train_loader)
    model.train()
    while step < cfg.train.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch(batch, device)
        out = model(batch)
        loss, comps = total_loss(out, batch, cfg, model)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        step += 1

        if step % cfg.train.log_every == 0:
            extra = f"  offset {comps['offset']:.4f}" if "offset" in comps else ""
            print(f"step {step:5d}  train_loss {loss.item():.4f}  ca {comps['ca']:.4f}{extra}")
        if step % cfg.train.val_every == 0 or step == cfg.train.max_steps:
            metrics = evaluate(model, val_loader, device, cfg)
            metrics["step"] = step
            metrics["train_loss"] = loss.item()
            history.append(metrics)
            print(
                f"  [val] step {step:5d}  loss {metrics['val_loss']:.4f}  "
                f"rmsd {metrics['val_rmsd']:.3f}  (no-op {metrics['noop_rmsd']:.3f})"
            )
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            torch.save({"model": model.state_dict(), "cfg": to_dict(cfg), "step": step},
                       out_dir / "last.ckpt")

    print(f"done. artifacts in {out_dir}")


if __name__ == "__main__":
    main()
