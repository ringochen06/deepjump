#!/usr/bin/env python
"""Deterministic multi-domain TICA/JSD gate for checkpoint comparison."""

from __future__ import annotations

import argparse
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
from deepjump.representation import apply_layout
from deepjump.utils import resolve_device


def selected_pair_indices(n_residues: int, max_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically select an approximately uniform subset of CA distance pairs."""
    i, j = np.triu_indices(n_residues, k=1)
    if len(i) <= max_features:
        return i, j
    keep = np.linspace(0, len(i) - 1, max_features, dtype=np.int64)
    return i[keep], j[keep]


def contiguous_frame_ids(n_frames: int, max_frames: int) -> np.ndarray:
    """Choose a centered contiguous window so the configured TICA lag stays physical."""
    count = min(max_frames, n_frames)
    start = max(0, (n_frames - count) // 2)
    return np.arange(start, start + count, dtype=np.int64)


def pairdist_features(P: torch.Tensor, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    i = torch.as_tensor(pairs[0], device=P.device)
    j = torch.as_tensor(pairs[1], device=P.device)
    return (P[..., i, :] - P[..., j, :]).norm(dim=-1).float().cpu().numpy()


def fit_tica(feats: np.ndarray, lag: int = 1, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Fit a regularized symmetric TICA projection without a generalized eigensolver."""
    if lag < 1 or lag >= len(feats):
        raise ValueError("lag must be in [1, n_frames)")
    mean = feats.mean(axis=0, keepdims=True)
    X = feats - mean
    C0 = X.T @ X / len(X)
    Xa, Xb = X[:-lag], X[lag:]
    Ctau = (Xa.T @ Xb + Xb.T @ Xa) / (2 * len(Xa))
    ev, U = np.linalg.eigh(C0)
    cutoff = max(float(ev.max()) * 1e-6, 1e-10)
    keep = ev > cutoff
    if int(keep.sum()) < n_components:
        raise ValueError("reference trajectory has insufficient TICA rank")
    W = U[:, keep] / np.sqrt(ev[keep])
    vals, vecs = np.linalg.eigh(W.T @ Ctau @ W)
    order = np.argsort(vals)[::-1][:n_components]
    return mean[0], W @ vecs[:, order]


def reference_histogram_jsd(reference: np.ndarray, sample: np.ndarray, bins: int = 24) -> float:
    """JSD using reference-fixed bounds; outliers accumulate in edge bins."""
    lo = reference.min(axis=0).copy()
    hi = reference.max(axis=0).copy()
    flat = hi <= lo
    lo[flat] -= 5e-9
    hi[flat] += 5e-9
    clipped = np.clip(sample, lo, hi)
    rng = [[lo[0], hi[0]], [lo[1], hi[1]]]
    hr, _, _ = np.histogram2d(reference[:, 0], reference[:, 1], bins=bins, range=rng)
    hs, _, _ = np.histogram2d(clipped[:, 0], clipped[:, 1], bins=bins, range=rng)
    pr = hr.ravel().astype(np.float64) + 1e-12
    ps = hs.ravel().astype(np.float64) + 1e-12
    pr /= pr.sum()
    ps /= ps.sum()
    mid = 0.5 * (pr + ps)
    return float(0.5 * np.sum(pr * np.log(pr / mid)) + 0.5 * np.sum(ps * np.log(ps / mid)))


def _repeat_batch(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {key: value.repeat(count, *([1] * (value.ndim - 1))) for key, value in batch.items()}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain-list", required=True)
    ap.add_argument("--domain-list-sha256", required=True)
    ap.add_argument("--domains", type=int, default=5)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--ode-steps", default="1,2,5")
    ap.add_argument("--real-frames", type=int, default=500)
    ap.add_argument("--max-features", type=int, default=512)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--integrator", choices=("euler", "heun"), default="euler")
    ap.add_argument("--tau-max", type=float, default=1.0)
    ap.add_argument("--terminal-denoise", action="store_true")
    ap.add_argument("--drift-anchor", choices=("state", "conditioner"), default="state")
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cm, cd = ck["cfg"]["model"], ck["cfg"]["data"]
    delta = require_single_delta(cd["delta_frames"])
    device = resolve_device(ck["cfg"]["train"]["device"])
    model = DeepJumpLite(ModelConfig(**cm), noise_sigma=cd["noise_sigma"],
                         predict_heavy=cm["predict_heavy"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    ordered = resolve_frozen_domains(discover_domains(cd["root"]), domain_ids)
    positions = np.linspace(0, len(ordered) - 1, min(args.domains, len(ordered)), dtype=int)
    chosen = [ordered[i] for i in positions]
    methods = [f"ode_{n}" for n in map(int, args.ode_steps.split(","))]
    rows = []

    for domain_index, path in enumerate(chosen):
        h = _DomainHandle(path)
        layout = h.layout
        temp, rep = cd["temperatures"][0], cd["replicas"][0]
        nf = h.replicas(temp, [rep])[0][2]
        n = min(layout.num_residues, int(cd["crop_length"]))
        offset = max(0, (layout.num_residues - n) // 2)
        residue_slice = slice(offset, offset + n)
        pairs = selected_pair_indices(n, args.max_features)
        frame_ids = contiguous_frame_ids(nf, args.real_frames)
        real_features = []
        for frame in frame_ids:
            coords = torch.from_numpy(np.asarray(h.coords(temp, rep, int(frame))))
            P, _ = apply_layout(coords, layout)
            P = P[residue_slice]
            P = P - P.mean(dim=0, keepdim=True)
            real_features.append(pairdist_features(P, pairs))
        real_features = np.stack(real_features)
        feature_mean, projection = fit_tica(real_features, lag=args.lag)
        real_tic = (real_features - feature_mean) @ projection

        last = nf - 1 - delta
        if last < 0:
            h.close()
            raise ValueError(f"{h.name} has too few frames for delta={delta}")
        starts = np.linspace(0, last, min(args.starts, last + 1), dtype=int)
        method_features = {name: [] for name in methods}
        noop_features = []
        for start_index, frame in enumerate(starts):
            coords = torch.from_numpy(np.asarray(h.coords(temp, rep, int(frame))))
            P, V = apply_layout(coords, layout)
            P = P[residue_slice]
            V = V[residue_slice]
            P = P - P.mean(dim=0, keepdim=True)
            batch = {
                "P_t": P[None].to(device),
                "V_t": V[None].to(device),
                "res_index": torch.as_tensor(layout.res_index[residue_slice], device=device)[None],
                "delta_ns": torch.tensor([float(delta)], device=device),
                "residue_mask": torch.ones(1, n, dtype=torch.bool, device=device),
                "atom_mask": torch.as_tensor(layout.atom_mask[residue_slice], device=device)[None],
            }
            expanded = _repeat_batch(batch, args.draws)
            noop_features.append(np.repeat(pairdist_features(P, pairs)[None], args.draws, axis=0))
            for method in methods:
                steps = int(method.split("_")[1])
                # DeepJumpLite.sample draws source noise on CPU before moving it
                # to the model device, so the generator must remain CPU-backed.
                generator = torch.Generator().manual_seed(
                    args.seed + domain_index * 100000 + start_index * 100 + steps
                )
                pred, _ = model.sample(
                    expanded, mode="ode", steps=steps, generator=generator,
                    integrator=args.integrator, tau_max=args.tau_max,
                    terminal_denoise=args.terminal_denoise,
                    drift_anchor=args.drift_anchor,
                )
                method_features[method].append(pairdist_features(pred, pairs))

        noop = np.concatenate(noop_features)
        domain_result = {
            "domain": h.name,
            "residues_total": layout.num_residues,
            "residues_evaluated": n,
            "frames": nf,
            "reference_frames": len(real_features),
            "sample_count": len(noop),
            "jsd": {
                "noop": reference_histogram_jsd(real_tic, (noop - feature_mean) @ projection, args.bins)
            },
        }
        for method, chunks in method_features.items():
            feats = np.concatenate(chunks)
            domain_result["jsd"][method] = reference_histogram_jsd(
                real_tic, (feats - feature_mean) @ projection, args.bins
            )
        rows.append(domain_result)
        h.close()

    summary = {
        method: {
            "mean_jsd": float(np.mean([row["jsd"][method] for row in rows])),
            "median_jsd": float(np.median([row["jsd"][method] for row in rows])),
            "domains_better_than_noop": sum(
                row["jsd"][method] < row["jsd"]["noop"] for row in rows
            ) if method != "noop" else 0,
        }
        for method in ["noop", *methods]
    }
    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": ck["step"],
        "delta_frames": delta,
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_list_sha256,
            "count": len(domain_ids),
            "evaluated_count": len(chosen),
        },
        "seed": args.seed,
        "settings": vars(args),
        "summary": summary,
        "domains": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
