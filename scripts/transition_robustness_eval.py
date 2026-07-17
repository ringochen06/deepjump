#!/usr/bin/env python
"""Conditional TICA-transition gate with a proper ensemble energy score."""

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
    assign_clusters,
    fit_kmeans,
    load_frozen_domain_ids,
    paired_domain_bootstrap_gain,
    reference_transition_deltas,
    require_single_delta,
    resolve_frozen_domains,
    transition_matrix,
    weighted_row_jsd_bits,
)
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_layout
from deepjump.utils import resolve_device
try:
    from scripts.tica_robustness_eval import (
        contiguous_frame_ids, fit_tica, pairdist_features, selected_pair_indices,
    )
except ModuleNotFoundError:  # direct `python scripts/transition_robustness_eval.py`
    from tica_robustness_eval import (
        contiguous_frame_ids, fit_tica, pairdist_features, selected_pair_indices,
    )


def energy_score(samples: np.ndarray, observation: np.ndarray) -> float:
    """Fair multivariate ensemble energy score; lower is better.

    The ensemble-pair term uses the off-diagonal U-statistic.  Including the
    zero diagonal systematically penalizes finite ensembles relative to a
    deterministic forecast and made the earlier four-draw gate unfair.
    """
    samples = np.asarray(samples, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    first = np.linalg.norm(samples - observation[None], axis=-1).mean()
    if len(samples) > 1:
        distances = np.linalg.norm(samples[:, None] - samples[None, :], axis=-1)
        pair = distances[~np.eye(len(samples), dtype=bool)].mean()
    else:
        pair = 0.0
    return float(first - 0.5 * pair)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """V-statistic energy distance between two transition distributions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = np.linalg.norm(x[:, None] - y[None, :], axis=-1).mean()
    xx = np.linalg.norm(x[:, None] - x[None, :], axis=-1).mean()
    yy = np.linalg.norm(y[:, None] - y[None, :], axis=-1).mean()
    return float(max(0.0, 2 * xy - xx - yy))


def repeat_batch(batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    return {key: value.repeat(count, *([1] * (value.ndim - 1))) for key, value in batch.items()}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain-list", required=True)
    ap.add_argument("--domain-list-sha256", required=True)
    ap.add_argument("--domains", type=int, default=10)
    ap.add_argument("--starts", type=int, default=50)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--methods", default="mean,ode_1")
    ap.add_argument("--real-frames", type=int, default=500)
    ap.add_argument("--max-features", type=int, default=512)
    ap.add_argument("--lag", type=int, default=10)
    ap.add_argument("--tica-components", type=int, default=4)
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--msm-lag", type=int, default=1)
    ap.add_argument("--msm-pseudocount", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument(
        "--noise-sigma", type=float, default=None,
        help="Override checkpoint source-noise sigma for inference-only calibration.",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.noise_sigma is not None and args.noise_sigma < 0:
        ap.error("--noise-sigma must be non-negative")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cm, cd = ck["cfg"]["model"], ck["cfg"]["data"]
    delta = require_single_delta(cd["delta_frames"])
    if args.msm_lag < 1:
        raise ValueError("msm_lag must be positive")
    if delta % args.msm_lag:
        raise ValueError(
            f"delta={delta} must be divisible by msm_lag={args.msm_lag}"
        )
    device = resolve_device(ck["cfg"]["train"]["device"])
    noise_sigma = cd["noise_sigma"] if args.noise_sigma is None else args.noise_sigma
    model = DeepJumpLite(
        ModelConfig(**cm), noise_sigma=noise_sigma, predict_heavy=cm["predict_heavy"]
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    ordered = resolve_frozen_domains(discover_domains(cd["root"]), domain_ids)
    positions = np.linspace(0, len(ordered) - 1, min(args.domains, len(ordered)), dtype=int)
    chosen = [ordered[i] for i in positions]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if any(m != "mean" and not m.startswith("ode_") for m in methods):
        raise ValueError("methods must be mean or ode_N")
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
            P = P - P.mean(0, keepdim=True)
            real_features.append(pairdist_features(P, pairs))
        real_features = np.stack(real_features)
        feature_mean, projection = fit_tica(
            real_features, lag=args.lag, n_components=args.tica_components
        )
        real_tic = (real_features - feature_mean) @ projection
        reference_delta = reference_transition_deltas(real_tic, delta)
        self_floor = energy_distance(reference_delta[::2], reference_delta[1::2])
        cluster_centers, real_cluster_labels = fit_kmeans(
            real_tic, args.clusters, seed=args.seed + domain_index
        )
        one_step_msm, _ = transition_matrix(
            real_cluster_labels,
            n_states=args.clusters,
            lag=args.msm_lag,
            pseudocount=args.msm_pseudocount,
        )
        target_msm = np.linalg.matrix_power(one_step_msm, delta // args.msm_lag)

        possible = frame_ids[:-delta]
        starts = possible[np.linspace(0, len(possible) - 1, min(args.starts, len(possible)), dtype=int)]
        scores = {"noop": []} | {m: [] for m in methods}
        predicted_delta = {"noop": []} | {m: [] for m in methods}
        transition_origins = []
        transition_destinations = {"noop": []} | {m: [] for m in methods}

        for start_index, frame in enumerate(starts):
            coords0 = torch.from_numpy(np.asarray(h.coords(temp, rep, int(frame))))
            coords1 = torch.from_numpy(np.asarray(h.coords(temp, rep, int(frame + delta))))
            P0, V0 = apply_layout(coords0, layout)
            P1, _ = apply_layout(coords1, layout)
            P0, V0, P1 = P0[residue_slice], V0[residue_slice], P1[residue_slice]
            P0 = P0 - P0.mean(0, keepdim=True)
            P1 = P1 - P1.mean(0, keepdim=True)
            tic0 = (pairdist_features(P0, pairs) - feature_mean) @ projection
            tic1 = (pairdist_features(P1, pairs) - feature_mean) @ projection
            scores["noop"].append(energy_score(tic0[None], tic1))
            predicted_delta["noop"].append(np.zeros((1, len(tic0))))
            origin_cluster = int(assign_clusters(tic0[None], cluster_centers)[0])
            transition_origins.extend([origin_cluster] * args.draws)
            transition_destinations["noop"].extend([origin_cluster] * args.draws)

            batch = {
                "P_t": P0[None].to(device),
                "V_t": V0[None].to(device),
                "res_index": torch.as_tensor(layout.res_index[residue_slice], device=device)[None],
                "bond_mask": torch.as_tensor(
                    layout.bond_mask[residue_slice.start:residue_slice.stop - 1], device=device
                )[None],
                "delta_ns": torch.tensor([float(delta)], device=device),
                "residue_mask": torch.ones(1, n, dtype=torch.bool, device=device),
                "atom_mask": torch.as_tensor(layout.atom_mask[residue_slice], device=device)[None],
            }
            expanded = repeat_batch(batch, args.draws)
            for method_index, method in enumerate(methods):
                if method == "mean":
                    pred, _ = model.sample(expanded, mode="mean")
                else:
                    ode_steps = int(method.split("_", 1)[1])
                    generator = torch.Generator().manual_seed(
                        args.seed + domain_index * 100000 + start_index * 100 + method_index
                    )
                    pred, _ = model.sample(
                        expanded, mode="ode", steps=ode_steps, generator=generator
                    )
                pred_tic = (pairdist_features(pred, pairs) - feature_mean) @ projection
                scores[method].append(energy_score(pred_tic, tic1))
                predicted_delta[method].append(pred_tic - tic0[None])
                transition_destinations[method].extend(
                    assign_clusters(pred_tic, cluster_centers).tolist()
                )

        domain = {
            "domain": h.name,
            "temperature": temp,
            "replica": rep,
            "starts": len(starts),
            "draws": args.draws,
            "reference_transitions": len(reference_delta),
            "self_energy_distance_floor": self_floor,
            "methods": {},
        }
        for method in ["noop", *methods]:
            predicted_increment = np.concatenate(predicted_delta[method])
            predicted_msm, origin_counts = transition_matrix(
                np.asarray(transition_origins),
                np.asarray(transition_destinations[method]),
                n_states=args.clusters,
                pseudocount=args.msm_pseudocount,
            )
            row_jsd, row_values = weighted_row_jsd_bits(
                target_msm, predicted_msm, origin_counts
            )
            domain["methods"][method] = {
                "mean_energy_score": float(np.mean(scores[method])),
                "median_energy_score": float(np.median(scores[method])),
                "transition_energy_distance": energy_distance(
                    reference_delta, predicted_increment
                ),
                "msm_row_jsd_bits": row_jsd,
                "msm_observed_origin_states": int((origin_counts > 0).sum()),
                "msm_max_observed_row_jsd_bits": float(row_values[origin_counts > 0].max()),
            }
        rows.append(domain)
        h.close()

    summary = {}
    for method in ["noop", *methods]:
        summary[method] = {
            "mean_energy_score": float(np.mean([
                row["methods"][method]["mean_energy_score"] for row in rows
            ])),
            "mean_transition_energy_distance": float(np.mean([
                row["methods"][method]["transition_energy_distance"] for row in rows
            ])),
            "mean_msm_row_jsd_bits": float(np.mean([
                row["methods"][method]["msm_row_jsd_bits"] for row in rows
            ])),
            "domains_better_than_noop_score": sum(
                row["methods"][method]["mean_energy_score"]
                < row["methods"]["noop"]["mean_energy_score"] for row in rows
            ) if method != "noop" else 0,
            "domains_better_than_noop_transition": sum(
                row["methods"][method]["transition_energy_distance"]
                < row["methods"]["noop"]["transition_energy_distance"] for row in rows
            ) if method != "noop" else 0,
            "domains_better_than_noop_msm": sum(
                row["methods"][method]["msm_row_jsd_bits"]
                < row["methods"]["noop"]["msm_row_jsd_bits"] for row in rows
            ) if method != "noop" else 0,
        }
        if method != "noop":
            summary[method]["paired_energy_score_gain"] = paired_domain_bootstrap_gain(
                np.asarray([
                    row["methods"][method]["mean_energy_score"] for row in rows
                ]),
                np.asarray([
                    row["methods"]["noop"]["mean_energy_score"] for row in rows
                ]),
                seed=args.seed,
            )
            summary[method]["paired_msm_row_jsd_gain"] = paired_domain_bootstrap_gain(
                np.asarray([
                    row["methods"][method]["msm_row_jsd_bits"] for row in rows
                ]),
                np.asarray([
                    row["methods"]["noop"]["msm_row_jsd_bits"] for row in rows
                ]),
                seed=args.seed + 1,
            )
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
