#!/usr/bin/env python
"""Distributional evaluation via TICA (the DeepJump-style eval philosophy).

Fit TICA on a real mdCATH trajectory (features = CA-CA pairwise distances, which are
SE(3)-invariant), then project both the real frames and a model-generated rollout ensemble
onto the top-2 TICs and compare their distributions (2D-histogram JSD). This measures
whether the model *samples the right conformational landscape*, not single-frame RMSD.

    python scripts/tica_eval.py --ckpt runs/full_delta1_unroll/last.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deepjump.config import ModelConfig  # noqa: E402
from deepjump.data import discover_domains  # noqa: E402
from deepjump.data.mdcath import _DomainHandle  # noqa: E402
from deepjump.model import DeepJumpLite  # noqa: E402
from deepjump.representation import apply_layout  # noqa: E402
from deepjump.sampling import rollout  # noqa: E402
from deepjump.utils import resolve_device, split_domains  # noqa: E402


def pairdist_features(P):  # P [N,3] tensor -> [N*(N-1)/2] numpy
    d = torch.cdist(P, P)
    iu = torch.triu_indices(P.shape[0], P.shape[0], offset=1)
    return d[iu[0], iu[1]].cpu().numpy()


def fit_tica(feats, lag=1, n=2):
    """Return projection matrix [D, n] (top-n slowest TICs). Pure-numpy generalized eig."""
    X = feats - feats.mean(0, keepdims=True)
    T = len(X)
    C0 = X.T @ X / T
    Xa, Xb = X[:-lag], X[lag:]
    Ctau = (Xa.T @ Xb + Xb.T @ Xa) / (2 * (T - lag))
    # whiten by C0 (keep positive eigenvalues), then diagonalise Ctau
    ev, U = np.linalg.eigh(C0)
    keep = ev > 1e-6 * ev.max()
    W = U[:, keep] / np.sqrt(ev[keep])
    w2, U2 = np.linalg.eigh(W.T @ Ctau @ W)
    idx = np.argsort(w2)[::-1][:n]
    return W @ U2[:, idx]  # [D, n]


def hist2d_jsd(a, b, bins=24):
    lo = np.minimum(a.min(0), b.min(0))
    hi = np.maximum(a.max(0), b.max(0))
    rng = [[lo[0], hi[0]], [lo[1], hi[1]]]
    Ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=rng, density=True)
    Hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=rng, density=True)
    pa = Ha.ravel() + 1e-9; pa /= pa.sum()
    pb = Hb.ravel() + 1e-9; pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log(p / q))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gen", choices=["conditional", "rollout"], default="conditional",
                    help="conditional = K stochastic ODE single-jumps per start (DeepJump-native)")
    ap.add_argument("--starts", type=int, default=40, help="number of start frames")
    ap.add_argument("--K", type=int, default=5, help="stochastic draws per start (conditional)")
    ap.add_argument("--sigma", type=float, default=None, help="override tau=0 noise sigma")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--domain", default=None, help="evaluate on this domain id (for fair cross-model compare)")
    ap.add_argument("--out", default="docs/tica.png")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cm, cd = ck["cfg"]["model"], ck["cfg"]["data"]
    device = resolve_device(cd["device"] if "device" in cd else "auto")
    model = DeepJumpLite(ModelConfig(**cm), noise_sigma=cd["noise_sigma"],
                         predict_heavy=cm["predict_heavy"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    files = discover_domains(cd["root"])
    if args.domain:
        match = [f for f in files if args.domain in f.name]
        if not match:
            raise SystemExit(f"domain {args.domain} not found under {cd['root']}")
        chosen = match[0]
    else:
        _, val = split_domains(files, cd["val_fraction"], cd["seed"])
        chosen = sorted(val)[0]
    h = _DomainHandle(chosen); layout = h.layout
    temp, rep = cd["temperatures"][0], cd["replicas"][0]
    nf = h.replicas(temp, [rep])[0][2]
    N = layout.num_residues
    print(f"domain {h.name}  N={N}  frames={nf}")

    # real ensemble: all frames -> (P centered, pairdist features)
    reals_P, real_feats = [], []
    for t in range(nf):
        c = torch.from_numpy(np.asarray(h.coords(temp, rep, t)))
        P, _ = apply_layout(c, layout)
        P = P - P.mean(0, keepdim=True)
        reals_P.append(P)
        real_feats.append(pairdist_features(P))
    real_feats = np.stack(real_feats)
    proj = fit_tica(real_feats, lag=args.lag)  # [D,2]
    real_tic = (real_feats - real_feats.mean(0)) @ proj

    # model ensemble. conditional (default): DeepJump-native diversity = K stochastic ODE
    # single-jumps of X_{t+delta} per start frame (different tau=0 noise eps each). A single jump
    # is geometrically stable (only rollout compounding explodes), so this populates the model's
    # p(X_{t+delta}|X_t) aggregated over the trajectory -- the right distributional test.
    if args.sigma is not None:
        model.noise_sigma = args.sigma
    stride = max(1, nf // args.starts)
    starts = list(range(0, nf - 1 - args.steps, stride))
    model_feats, start_feats = [], []
    for si, t0 in enumerate(starts):
        c = torch.from_numpy(np.asarray(h.coords(temp, rep, int(t0))))
        P0, V0 = apply_layout(c, layout)
        P0 = (P0 - P0.mean(0, keepdim=True)).to(device)
        init = {"P_t": P0[None], "V_t": V0[None].to(device),
                "res_index": torch.as_tensor(layout.res_index, device=device)[None],
                "delta_ns": torch.tensor([1.0], device=device),
                "residue_mask": torch.ones(1, N, dtype=torch.bool, device=device)}
        start_feats.append(pairdist_features(P0.cpu()))
        if args.gen == "rollout":
            gen = torch.Generator().manual_seed(si)
            traj, _ = rollout(model, init, n_steps=args.steps, mode="mean", gate=True, generator=gen)
            for (P, _V) in traj[1:]:
                model_feats.append(pairdist_features(P[0].cpu()))
        else:
            for k in range(args.K):
                gen = torch.Generator().manual_seed(si * 1000 + k)
                P, _V = model.sample(init, steps=args.steps, mode="ode", generator=gen)
                model_feats.append(pairdist_features(P[0].cpu()))
    model_feats = np.stack(model_feats)
    print(f"gen={args.gen}  sigma={model.noise_sigma}  ensemble={len(model_feats)}")
    start_feats = np.stack(start_feats)
    model_tic = (model_feats - real_feats.mean(0)) @ proj
    start_tic = (start_feats - real_feats.mean(0)) @ proj

    jsd_model = hist2d_jsd(real_tic, model_tic)
    jsd_start = hist2d_jsd(real_tic, start_tic)  # no-dynamics reference
    print(f"\nTIC-space 2D-histogram JSD (lower = better distributional match):")
    print(f"  model rollout ensemble vs real : {jsd_model:.3f}")
    print(f"  start-frames-only    vs real   : {jsd_start:.3f}  (no-dynamics reference)")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(real_tic[:, 0], real_tic[:, 1], s=6, c="lightgray", label="real MD")
    ax.scatter(model_tic[:, 0], model_tic[:, 1], s=6, c="C0", alpha=.6, label="model rollout")
    ax.set(xlabel="TIC 1", ylabel="TIC 2",
           title=f"{h.name}  TICA landscape\nJSD(model,real)={jsd_model:.3f}  (start-only {jsd_start:.3f})")
    ax.legend()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    h.close()


if __name__ == "__main__":
    main()
