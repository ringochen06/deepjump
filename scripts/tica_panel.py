#!/usr/bin/env python
"""Paper-style TICA panel figure (structure overlay + 2D free-energy heatmaps + marginals).

Reproduces the *layout* of the DeepJump equilibrium-ensemble figure, but honestly on OUR
scale/data: mdCATH domains (NOT the DESRES fast folders), a DeepJump-native
stochastic conditional ensemble (K ODE single-jumps per start frame), no folding.

Per domain (one block) we draw 4 sub-panels, exactly like the paper:
    [ CA structure overlay | real-MD free-energy | model free-energy | TIC1/TIC2 marginals ]
Free energy F = -ln p (kT units), shared TIC binning between real & model so the two
heatmaps are directly comparable. Marginals: real (black) vs model (blue), native basin
marked with a dashed line.

    python scripts/tica_panel.py --ckpt runs/faithful_scaled/last.ckpt --n 4
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deepjump.config import ModelConfig  # noqa: E402
from deepjump.data import discover_domains  # noqa: E402
from deepjump.data.mdcath import _DomainHandle  # noqa: E402
from deepjump.model import DeepJumpLite  # noqa: E402
from deepjump.representation import apply_layout, apply_model_layout  # noqa: E402
from deepjump.utils import resolve_device, split_domains  # noqa: E402

# ---- TICA helpers (shared with tica_eval) ----------------------------------

def pairdist_features(P):  # P [N,3] tensor -> [N*(N-1)/2] numpy
    d = torch.cdist(P, P)
    iu = torch.triu_indices(P.shape[0], P.shape[0], offset=1)
    return d[iu[0], iu[1]].cpu().numpy()


def fit_tica(feats, lag=1, n=2):
    X = feats - feats.mean(0, keepdims=True)
    T = len(X)
    C0 = X.T @ X / T
    Xa, Xb = X[:-lag], X[lag:]
    Ctau = (Xa.T @ Xb + Xb.T @ Xa) / (2 * (T - lag))
    ev, U = np.linalg.eigh(C0)
    keep = ev > 1e-6 * ev.max()
    W = U[:, keep] / np.sqrt(ev[keep])
    w2, U2 = np.linalg.eigh(W.T @ Ctau @ W)
    idx = np.argsort(w2)[::-1][:n]
    return W @ U2[:, idx]  # [D, n]


def hist2d_jsd(a, b, rng, bins=24):
    """JSD on a grid the caller fixes from the reference; strays are clipped in.

    The grid must not be sized to the union of both samples: a diverging model
    would stretch the extent, collapse the reference into a couple of bins, and
    drive the JSD down. See scripts/tica_eval.py:reference_range.
    """
    edges = [np.linspace(rng[0][0], rng[0][1], bins + 1),
             np.linspace(rng[1][0], rng[1][1], bins + 1)]

    def density(sample):
        clipped = np.stack([np.clip(sample[:, 0], edges[0][0], edges[0][-1]),
                            np.clip(sample[:, 1], edges[1][0], edges[1][-1])], axis=1)
        H, _, _ = np.histogram2d(clipped[:, 0], clipped[:, 1], bins=edges)
        p = H.ravel() + 1e-9
        return p / p.sum()

    pa, pb = density(a), density(b)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log(p / q))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


# ---- free-energy rendering --------------------------------------------------

FE_MAX = 8.0  # kT cap for colour scale / y-axis


def _gauss_blur(H, sigma=0.9, half=2):
    """Tiny separable Gaussian blur (numpy-only) so sparse histograms read as smooth basins."""
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, H)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


def free_energy_2d(tic, rng, bins=32, smooth=True):
    """2D free energy F=-ln p on a shared grid; empty bins -> NaN (drawn white)."""
    H, xe, ye = np.histogram2d(tic[:, 0], tic[:, 1], bins=bins, range=rng)
    occ = H > 0
    if smooth:
        H = _gauss_blur(H)
        occ = _gauss_blur(occ.astype(float)) > 1e-3  # keep a small halo, not the whole plane
    p = H / H.sum()
    with np.errstate(divide="ignore"):
        F = -np.log(p)
    F = F - np.nanmin(F[np.isfinite(F)])
    F[(~np.isfinite(F)) | (~occ)] = np.nan
    return F, xe, ye


def free_energy_1d(x, edges):
    H, _ = np.histogram(x, bins=edges, density=True)
    p = H / H.sum()
    with np.errstate(divide="ignore"):
        F = -np.log(p)
    F[p > 0] -= F[p > 0].min()
    F[p == 0] = np.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, F


def draw_fe_heatmap(ax, F, xe, ye, title):
    cmap = plt.cm.Spectral_r.copy()
    cmap.set_bad("white")
    ax.imshow(F.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
              aspect="auto", cmap=cmap, vmin=0, vmax=FE_MAX)
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("TIC 1", fontsize=7)


def draw_overlay(ax, ref_P, ens_P):
    """CA backbone traces projected to 2D via the reference structure's principal axes.

    Faint rainbow cloud = model-generated conformers, bold rainbow = one real reference.
    """
    Xc = ref_P - ref_P.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    axes2 = Vt[:2].T  # [3,2]

    def seglines(P2, lw, alpha):
        pts = P2.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="rainbow", lw=lw, alpha=alpha)
        lc.set_array(np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)

    for P in ens_P:
        seglines((P - P.mean(0, keepdims=True)) @ axes2, lw=0.5, alpha=0.12)
    seglines(Xc @ axes2, lw=1.8, alpha=0.95)
    allp = np.concatenate([(ref_P - ref_P.mean(0)) @ axes2] +
                          [(P - P.mean(0)) @ axes2 for P in ens_P])
    m = np.abs(allp).max() * 1.1
    ax.set(xlim=(-m, m), ylim=(-m, m))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def draw_marginals(ax_top, ax_bot, real_tic, model_tic, xe, ye):
    for ax, dim, edges, lbl in ((ax_top, 0, xe, "TIC 1"), (ax_bot, 1, ye, "TIC 2")):
        cr, fr = free_energy_1d(real_tic[:, dim], edges)
        cm, fm = free_energy_1d(model_tic[:, dim], edges)
        ax.plot(cr, fr, color="black", lw=1.3, label="real MD")
        ax.plot(cm, fm, color="C0", lw=1.3, label="model")
        native = cr[np.nanargmin(fr)]
        ax.axvline(native, color="0.4", ls="--", lw=0.8)
        ax.set_ylim(0, FE_MAX + 1)
        ax.set_xlabel(f"{lbl}", fontsize=7)
        ax.set_ylabel("F (kT)", fontsize=7)
        ax.tick_params(labelsize=6)


# ---- ensemble generation ----------------------------------------------------

@torch.no_grad()
def build_domain(
    model, handle, temp, rep, device, starts, K, steps, sigma, *, canon_symmetric,
    sample_kwargs=None,
):
    layout = handle.layout
    N = layout.num_residues
    nf = handle.replicas(temp, [rep])[0][2]

    reals_P, real_feats = [], []
    for t in range(nf):
        c = torch.from_numpy(np.asarray(handle.coords(temp, rep, t)))
        P, _ = apply_layout(c, layout)
        P = P - P.mean(0, keepdim=True)
        reals_P.append(P.cpu().numpy())
        real_feats.append(pairdist_features(P))
    real_feats = np.stack(real_feats)
    proj = fit_tica(real_feats)
    real_tic = (real_feats - real_feats.mean(0)) @ proj

    if sigma is not None:
        model.noise_sigma = sigma
    stride = max(1, nf // starts)
    start_idx = list(range(0, nf - 1 - steps, stride))
    model_feats, ens_P = [], []
    for si, t0 in enumerate(start_idx):
        c = torch.from_numpy(np.asarray(handle.coords(temp, rep, int(t0))))
        P0, V0 = apply_model_layout(
            c, layout, canon_symmetric=canon_symmetric
        )
        P0 = (P0 - P0.mean(0, keepdim=True)).to(device)
        # atom_mask is required both by source_noise_v checkpoints and by the
        # per-step V re-masking that keeps the sampled state on the training
        # manifold; omitting it made the ODE path unusable here.
        init = {"P_t": P0[None], "V_t": V0[None].to(device),
                "res_index": torch.as_tensor(layout.res_index, device=device)[None],
                "atom_mask": torch.as_tensor(layout.atom_mask, device=device)[None],
                "delta_ns": torch.tensor([1.0], device=device),
                "residue_mask": torch.ones(1, N, dtype=torch.bool, device=device)}
        for k in range(K):
            gen = torch.Generator().manual_seed(si * 1000 + k)
            P, _V = model.sample(init, steps=steps, mode="ode", generator=gen,
                                 **(sample_kwargs or {}))
            Pn = P[0].cpu()
            model_feats.append(pairdist_features(Pn))
            if len(ens_P) < 30:
                ens_P.append(Pn.numpy())
    model_feats = np.stack(model_feats)
    model_tic = (model_feats - real_feats.mean(0)) @ proj
    return dict(name=handle.name, real_tic=real_tic, model_tic=model_tic,
                ref_P=reals_P[0], ens_P=ens_P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/faithful_scaled/last.ckpt")
    ap.add_argument("--n", type=int, default=4, help="number of domains (rows)")
    ap.add_argument("--domains", nargs="*", default=None, help="explicit domain ids")
    ap.add_argument("--cols", type=int, default=1, help="protein blocks per figure row (1 or 2)")
    ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--root", default=None, help="override the checkpoint's data root")
    ap.add_argument("--tau-max", type=float, default=1.0,
                    help="truncate the interpolant before the singular endpoint; <1 needs --terminal-denoise")
    ap.add_argument("--terminal-denoise", action="store_true",
                    help="emit one endpoint prediction at tau_max instead of integrating into the singularity")
    ap.add_argument("--integrator", choices=["euler", "heun"], default="euler")
    ap.add_argument("--held-out-list", default=None,
                    help="file of domain ids the checkpoint did NOT train on; only with this "
                         "does the figure claim the rows are held out")
    ap.add_argument("--out", default="docs/tica_panel.png")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cm, cd = ck["cfg"]["model"], ck["cfg"]["data"]
    device = resolve_device(cd.get("device", "auto"))
    # Checkpoints predating the current tree may carry model options ModelConfig no
    # longer declares. Dropping one silently would change the architecture, so only
    # falsy unknowns are dropped and the state dict is still loaded strictly.
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    for key, value in cm.items():
        if key not in known and value:
            raise SystemExit(f"checkpoint sets unknown model option {key}={value!r}")
    model = DeepJumpLite(ModelConfig(**{k: v for k, v in cm.items() if k in known}),
                         noise_sigma=cd["noise_sigma"],
                         predict_heavy=cm["predict_heavy"]).to(device)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    print(f"loaded {args.ckpt}  H={cm['hidden']}  device={device}")

    held_out_ids = None
    if args.held_out_list:
        held_out_ids = {line.strip() for line in Path(args.held_out_list).read_text().split()
                        if line.strip()}
        print(f"held-out list: {len(held_out_ids)} domain ids from {args.held_out_list}")

    files = discover_domains(args.root or cd["root"])
    if args.domains:
        chosen = [f for f in files if any(d in f.name for d in args.domains)]
    else:
        _, val = split_domains(files, cd["val_fraction"], cd["seed"])
        chosen = sorted(val)[: args.n]
    temp, rep = cd["temperatures"][0], cd["replicas"][0]

    blocks = []
    for f in chosen:
        h = _DomainHandle(f)
        print(f"  building {h.name} ...", flush=True)
        blocks.append(build_domain(
            model,
            h,
            temp,
            rep,
            device,
            args.starts,
            args.K,
            args.steps,
            args.sigma,
            canon_symmetric=bool(cd.get("canon_symmetric", False)),
            sample_kwargs=dict(integrator=args.integrator,
                               tau_max=args.tau_max,
                               terminal_denoise=args.terminal_denoise),
        ))
        h.close()

    # ---- layout: mimic the paper (blocks arranged in a grid) ----
    nb = len(blocks)
    ncol = max(1, args.cols)
    nrow = (nb + ncol - 1) // ncol
    fig = plt.figure(figsize=(6.6 * ncol, 2.5 * nrow))
    outer = fig.add_gridspec(nrow, ncol, wspace=0.28, hspace=0.75)

    for bi, blk in enumerate(blocks):
        r, c = divmod(bi, ncol)
        inner = outer[r, c].subgridspec(1, 4, width_ratios=[1.0, 1.25, 1.25, 0.95], wspace=0.55)
        ax_ov = fig.add_subplot(inner[0, 0])
        ax_r = fig.add_subplot(inner[0, 1])
        ax_m = fig.add_subplot(inner[0, 2])
        marg = inner[0, 3].subgridspec(2, 1, hspace=0.9)
        ax_t1 = fig.add_subplot(marg[0, 0])
        ax_t2 = fig.add_subplot(marg[1, 0])

        # TIC range fixed by real MD alone. Including the model would let a
        # diverging ensemble stretch the axes until real MD occupies one bin,
        # which both flatters the JSD and renders the reference unreadable.
        lo, hi = blk["real_tic"].min(0), blk["real_tic"].max(0)
        pad = 0.05 * (hi - lo + 1e-6)
        rng = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
        jsd = hist2d_jsd(blk["real_tic"], blk["model_tic"], rng, bins=24)

        Fr, xe, ye = free_energy_2d(blk["real_tic"], rng, bins=args.bins)
        Fm, _, _ = free_energy_2d(blk["model_tic"], rng, bins=args.bins)

        draw_overlay(ax_ov, blk["ref_P"], blk["ens_P"])
        ax_ov.set_ylabel(blk["name"], fontsize=9, rotation=90, labelpad=6)
        draw_fe_heatmap(ax_r, Fr, xe, ye, "real MD")
        draw_fe_heatmap(ax_m, Fm, xe, ye, f"DeepJump-lite  (JSD {jsd:.2f})")
        draw_marginals(ax_t1, ax_t2, blk["real_tic"], blk["model_tic"], xe, ye)
        if bi == 0:
            ax_t1.legend(fontsize=5, loc="upper center", frameon=False)

    # Provenance must be demonstrated, not asserted. Selecting domains by
    # split_domains() over whatever happens to be on local disk says nothing about
    # what the checkpoint trained on: this checkpoint's contract lists 5218 train
    # domains out of mdCATH's 5398, so a domain drawn without checking the real
    # list is ~97% likely to have been trained on.
    if held_out_ids is None:
        provenance = "training status unverified — no --held-out-list given"
    else:
        seen = [b["name"] for b in blocks if b["name"] not in held_out_ids]
        provenance = ("held-out domains" if not seen
                      else f"WARNING: trained-on domains present ({', '.join(seen)})")
    fig.suptitle(f"TICA equilibrium ensemble — mdCATH domains, {provenance}\n"
                 "(DeepJump-lite, conditional stochastic ensemble)", fontsize=9, y=1.0)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
