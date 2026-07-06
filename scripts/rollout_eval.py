#!/usr/bin/env python
"""Roll the jump operator along a real mdCATH trajectory and evaluate stability.

Starting from a real frame X(t0), autoregressively predict X_hat(t0 + k*delta) and
compare, per step k, against:
  * ground truth  X(t0 + k*delta)         -> model-vs-truth RMSD (Kabsch-aligned)
  * the start     X(t0)                    -> model drift  vs  true drift
The no-op baseline predicts X(t0) forever, so its error to truth == true drift.
Also tracks CA-CA bond geometry (chemical validity) and native-contact recovery.

    python scripts/rollout_eval.py --ckpt runs/full_delta1/last.ckpt --steps 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deepjump.config import ModelConfig  # noqa: E402
from deepjump.data.mdcath import _DomainHandle  # noqa: E402
from deepjump.data import discover_domains  # noqa: E402
from deepjump.metrics import aligned_ca_rmsd, ca_bond_stats, contact_fraction_native  # noqa: E402
from deepjump.model import DeepJumpLite  # noqa: E402
from deepjump.representation import apply_layout  # noqa: E402
from deepjump.utils import resolve_device, split_domains  # noqa: E402


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cm, cd = ck["cfg"]["model"], ck["cfg"]["data"]
    model = DeepJumpLite(ModelConfig(**cm), noise_sigma=cd["noise_sigma"],
                         predict_heavy=cm["predict_heavy"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck["cfg"]


def real_frames(handle, temp, rep, t0, n_steps, delta, layout, device):
    """Return list of (P,V) for real frames t0, t0+delta, ..., centered per frame."""
    out = []
    for k in range(n_steps + 1):
        c = torch.from_numpy(np.asarray(handle.coords(temp, rep, t0 + k * delta)))
        P, V = apply_layout(c, layout)
        P = P - P.mean(0, keepdim=True)
        out.append((P.to(device), V.to(device)))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=10, help="rollout length")
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--mode", choices=["ode", "mean"], default="ode")
    ap.add_argument("--gate", action="store_true", help="geometry acceptance gate")
    ap.add_argument("--starts", type=int, default=5, help="number of start frames to average")
    ap.add_argument("--out", default="runs/rollout")
    args = ap.parse_args()

    from deepjump.sampling import rollout

    device = resolve_device("auto")
    model, cfg = load_model(args.ckpt, device)
    cd = cfg["data"]
    delta = cd["delta_frames"]

    files = discover_domains(cd["root"])
    _, val = split_domains(files, cd["val_fraction"], cd["seed"])
    handle = _DomainHandle(sorted(val)[0])
    temp, rep = cd["temperatures"][0], cd["replicas"][0]
    n_frames = handle.replicas(temp, [rep])[0][2]
    layout = handle.layout
    N = layout.num_residues
    print(f"domain {handle.name}  N={N} residues  frames={n_frames}")

    # start frames spread across the trajectory, leaving room for the rollout
    last = n_frames - 1 - args.steps * delta
    starts = np.linspace(0, max(last, 0), args.starts, dtype=int).tolist()

    K = args.steps + 1
    acc = {k: {"rmsd": [], "drift": [], "true_drift": [], "fnc": [], "bond": []} for k in range(K)}
    accept_rates = []
    gen = torch.Generator(device="cpu")

    for si, t0 in enumerate(starts):
        real = real_frames(handle, temp, rep, t0, args.steps, delta, layout, device)
        P0, V0 = real[0]
        mask = torch.ones(1, N, dtype=torch.bool, device=device)
        init = {
            "P_t": P0[None], "V_t": V0[None],
            "res_index": torch.as_tensor(layout.res_index, device=device)[None],
            "delta_ns": torch.tensor([float(delta)], device=device),
            "residue_mask": mask,
        }
        gen.manual_seed(si)
        traj, accepts = rollout(model, init, n_steps=args.steps, ode_steps=args.ode_steps,
                                mode=args.mode, gate=args.gate, generator=gen)
        if accepts:
            accept_rates.append(float(torch.stack(accepts).float().mean().item()))
        for k in range(K):
            Pk = traj[k][0][0]
            Preal_k, _ = real[k]
            bmean, _ = ca_bond_stats(Pk)
            acc[k]["rmsd"].append(aligned_ca_rmsd(Pk, Preal_k).item())
            acc[k]["drift"].append(aligned_ca_rmsd(Pk, P0).item())
            acc[k]["true_drift"].append(aligned_ca_rmsd(Preal_k, P0).item())
            acc[k]["fnc"].append(contact_fraction_native(Pk[None], Preal_k[None], mask).item())
            acc[k]["bond"].append(bmean.item())

    steps = list(range(K))
    mean = lambda k, key: float(np.mean(acc[k][key]))
    rmsd = [mean(k, "rmsd") for k in steps]
    drift = [mean(k, "drift") for k in steps]
    true_drift = [mean(k, "true_drift") for k in steps]
    fnc = [mean(k, "fnc") for k in steps]
    bond = [mean(k, "bond") for k in steps]

    gate_note = f"  gate={'on' if args.gate else 'off'}"
    if accept_rates:
        gate_note += f"  mean accept rate={sum(accept_rates)/len(accept_rates):.2f}"
    print(f"\nmode={args.mode}{gate_note}")
    print(f"{'step':>4}{'model-vs-truth':>16}{'no-op(truedrift)':>18}{'model drift':>13}{'FNC':>7}{'bond(A)':>9}")
    for k in steps:
        print(f"{k:>4}{rmsd[k]:>16.3f}{true_drift[k]:>18.3f}{drift[k]:>13.3f}{fnc[k]:>7.2f}{bond[k]:>9.2f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.mode + ("_gated" if args.gate else "")
    json.dump({"mode": args.mode, "steps": steps, "rmsd": rmsd, "true_drift": true_drift,
               "drift": drift, "fnc": fnc, "bond": bond, "domain": handle.name},
              open(out / f"rollout_{tag}.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(steps, rmsd, "o-", label="model vs truth")
    ax[0].plot(steps, true_drift, "s--", label="no-op (true drift)")
    ax[0].set(xlabel="rollout step (ns)", ylabel="CA RMSD (A)", title="accuracy vs truth")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(steps, drift, "o-", label="model drift")
    ax[1].plot(steps, true_drift, "s--", label="true drift")
    ax[1].set(xlabel="rollout step (ns)", ylabel="RMSD from start (A)", title="drift"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].axhline(3.8, color="gray", ls=":", label="ideal 3.8 A")
    ax[2].plot(steps, bond, "o-", label="model")
    ax[2].set(xlabel="rollout step (ns)", ylabel="mean CA-CA (A)", title="bond geometry"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle(f"rollout mode={args.mode}  domain={handle.name}", y=1.02)
    fig.tight_layout(); fig.savefig(out / f"rollout_{tag}.png", dpi=110, bbox_inches="tight")
    print(f"\nsaved {out/('rollout_'+tag+'.json')} and {out/('rollout_'+tag+'.png')}")
    handle.close()


if __name__ == "__main__":
    main()
