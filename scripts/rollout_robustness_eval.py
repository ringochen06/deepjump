#!/usr/bin/env python
"""Deterministic multi-domain long-rollout gate for checkpoint comparison."""

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
from deepjump.metrics import aligned_ca_rmsd, contact_fraction_native
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_model_layout
from deepjump.sampling import rollout
from deepjump.utils import resolve_device


def select_validation_domains(paths: list[str], count: int) -> list[str]:
    """Select deterministic domains spread over the sorted validation set."""
    ordered = sorted(paths)
    positions = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return [ordered[i] for i in positions]


def summarize_domains(rows: list[dict], method: str, steps: int) -> dict:
    """Summarize final and trajectory-integrated RMSD across domains."""
    final = [row["methods"][method]["rmsd"][steps] for row in rows]
    auc = [float(np.mean(row["methods"][method]["rmsd"][1:])) for row in rows]
    noop_final = [row["methods"]["noop"]["rmsd"][steps] for row in rows]
    summary = {
        "mean_final_rmsd": float(np.mean(final)),
        "median_final_rmsd": float(np.median(final)),
        "mean_rollout_rmsd": float(np.mean(auc)),
        "domains_better_than_noop_final": int(sum(a < b for a, b in zip(final, noop_final))),
        "finite": bool(np.isfinite(final).all() and np.isfinite(auc).all()),
    }
    metrics = rows[0]["methods"][method]
    for name in ("bond_mean", "bond_p95", "bond_p99", "bond_max",
                 "bond_mae_real", "angle_cos_mae_real"):
        if name in metrics:
            summary[f"mean_final_{name}"] = float(np.mean([
                row["methods"][method][name][steps] for row in rows
            ]))
    return summary


def _center(P: torch.Tensor) -> torch.Tensor:
    return P - P.mean(dim=0, keepdim=True)


def _local_geometry(pred: torch.Tensor, target: torch.Tensor, bond_mask: torch.Tensor) -> dict:
    """Aggregate topology-valid bond and angle diagnostics over a start batch."""
    pred_bond = pred[:, 1:] - pred[:, :-1]
    target_bond = target[:, 1:] - target[:, :-1]
    pred_len = pred_bond.norm(dim=-1)
    target_len = target_bond.norm(dim=-1)
    valid_bonds = bond_mask.bool()
    lengths = pred_len[valid_bonds]
    if lengths.numel() == 0:
        return {name: float("nan") for name in (
            "bond_mean", "bond_p95", "bond_p99", "bond_max",
            "bond_mae_real", "angle_cos_mae_real",
        )}
    pred_unit = pred_bond / pred_len.clamp_min(1e-6).unsqueeze(-1)
    target_unit = target_bond / target_len.clamp_min(1e-6).unsqueeze(-1)
    pred_cos = (pred_unit[:, :-1] * pred_unit[:, 1:]).sum(-1)
    target_cos = (target_unit[:, :-1] * target_unit[:, 1:]).sum(-1)
    valid_angles = valid_bonds[:, :-1] & valid_bonds[:, 1:]
    return {
        "bond_mean": float(lengths.mean().item()),
        "bond_p95": float(torch.quantile(lengths, 0.95).item()),
        "bond_p99": float(torch.quantile(lengths, 0.99).item()),
        "bond_max": float(lengths.max().item()),
        "bond_mae_real": float((pred_len[valid_bonds] - target_len[valid_bonds]).abs().mean().item()),
        "angle_cos_mae_real": float(
            (pred_cos[valid_angles] - target_cos[valid_angles]).abs().mean().item()
        ) if valid_angles.any() else float("nan"),
    }


def teacher_forced_mean_trajectory(
    model,
    real_positions: list[torch.Tensor],
    real_vectors: list[torch.Tensor],
    static: dict[str, torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Predict every horizon from its real preceding state, never model feedback."""
    if len(real_positions) != len(real_vectors) or len(real_positions) < 2:
        raise ValueError("teacher-forced inputs must have matching length >= 2")
    trajectory = [(real_positions[0], real_vectors[0])]
    for step in range(len(real_positions) - 1):
        batch = {
            "P_t": real_positions[step],
            "V_t": real_vectors[step],
            **static,
        }
        trajectory.append(model.sample(batch, steps=1, mode="mean"))
    return trajectory


def one_step_persistence_trajectory(
    real_positions: list[torch.Tensor],
    real_vectors: list[torch.Tensor],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Use the preceding real frame as the prediction for each one-step target."""
    if len(real_positions) != len(real_vectors) or len(real_positions) < 2:
        raise ValueError("persistence inputs must have matching length >= 2")
    return [
        (real_positions[0], real_vectors[0]),
        *list(zip(real_positions[:-1], real_vectors[:-1])),
    ]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain-list", required=True)
    ap.add_argument("--domain-list-sha256", required=True)
    ap.add_argument("--domains", type=int, default=5)
    ap.add_argument("--starts", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--methods", default="mean,ode_1")
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument(
        "--noise-sigma", type=float, default=None,
        help="Override checkpoint source-noise sigma for inference-only calibration.",
    )
    ap.add_argument("--integrator", choices=("euler", "heun"), default="euler")
    ap.add_argument("--tau-max", type=float, default=1.0)
    ap.add_argument("--terminal-denoise", action="store_true")
    ap.add_argument("--drift-anchor", choices=("state", "conditioner"), default="state")
    ap.add_argument(
        "--teacher-forced-mean", action="store_true",
        help="Also evaluate deterministic one-step predictions from each real preceding frame.",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.noise_sigma is not None and args.noise_sigma < 0:
        ap.error("--noise-sigma must be non-negative")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cm, cd = ck["cfg"]["model"], ck["cfg"]["data"]
    device = resolve_device(ck["cfg"]["train"]["device"])
    noise_sigma = cd["noise_sigma"] if args.noise_sigma is None else args.noise_sigma
    model = DeepJumpLite(ModelConfig(**cm), noise_sigma=noise_sigma,
                         predict_heavy=cm["predict_heavy"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    delta = require_single_delta(cd["delta_frames"])
    domain_ids, domain_list_sha256 = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    panel = resolve_frozen_domains(discover_domains(cd["root"]), domain_ids)
    chosen = select_validation_domains(panel, args.domains)
    requested_methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    invalid = [name for name in requested_methods if name != "mean" and not name.startswith("ode_")]
    if invalid:
        raise ValueError(f"unsupported methods: {invalid}")

    rows = []
    for domain_index, path in enumerate(chosen):
        h = _DomainHandle(path)
        layout = h.layout
        temp, rep = cd["temperatures"][0], cd["replicas"][0]
        n_frames = h.replicas(temp, [rep])[0][2]
        last = n_frames - 1 - args.steps * delta
        if last < 0:
            h.close()
            raise ValueError(f"{h.name} has too few frames for {args.steps} rollout steps")
        starts = np.linspace(0, last, min(args.starts, last + 1), dtype=int)
        n = min(layout.num_residues, int(cd["crop_length"]))
        offset = max(0, (layout.num_residues - n) // 2)
        residue_slice = slice(offset, offset + n)

        real_by_step: list[list[torch.Tensor]] = [[] for _ in range(args.steps + 1)]
        vectors_by_step: list[list[torch.Tensor]] = [[] for _ in range(args.steps + 1)]
        for start in starts:
            for step in range(args.steps + 1):
                coords = torch.from_numpy(np.asarray(h.coords(temp, rep, int(start + step * delta))))
                P, V = apply_model_layout(
                    coords,
                    layout,
                    canon_symmetric=bool(cd.get("canon_symmetric", False)),
                )
                P = _center(P[residue_slice])
                real_by_step[step].append(P)
                vectors_by_step[step].append(V[residue_slice])
        real = [torch.stack(values).to(device) for values in real_by_step]
        real_vectors = [torch.stack(values).to(device) for values in vectors_by_step]
        P0 = real[0]
        V0 = real_vectors[0]
        batch = {
            "P_t": P0,
            "V_t": V0,
            "res_index": torch.as_tensor(layout.res_index[residue_slice], device=device)[None].repeat(len(starts), 1),
            "delta_ns": torch.full((len(starts),), float(delta), device=device),
            "residue_mask": torch.ones(len(starts), n, dtype=torch.bool, device=device),
            "atom_mask": torch.as_tensor(
                layout.atom_mask[residue_slice], device=device
            )[None].repeat(len(starts), 1, 1),
            "bond_mask": torch.as_tensor(
                layout.bond_mask[residue_slice.start:residue_slice.stop - 1], device=device
            )[None].repeat(len(starts), 1),
        }

        trajectories = {
            "noop": [(P0, V0)] * (args.steps + 1),
            "one_step_persistence": one_step_persistence_trajectory(real, real_vectors),
        }
        for method_index, method in enumerate(requested_methods):
            if method == "mean":
                mode, ode_steps, generator = "mean", 1, None
            else:
                mode, ode_steps = "ode", int(method.split("_", 1)[1])
                generator = torch.Generator().manual_seed(
                    args.seed + domain_index * 1000 + method_index * 100
                )
            trajectories[method], _ = rollout(
                model, batch, n_steps=args.steps, ode_steps=ode_steps,
                mode=mode, gate=False, generator=generator,
                sample_kwargs={
                    "integrator": args.integrator,
                    "tau_max": args.tau_max,
                    "terminal_denoise": args.terminal_denoise,
                    "drift_anchor": args.drift_anchor,
                },
            )
        if args.teacher_forced_mean:
            static = {
                key: value for key, value in batch.items() if key not in {"P_t", "V_t"}
            }
            trajectories["teacher_forced_mean"] = teacher_forced_mean_trajectory(
                model, real, real_vectors, static
            )

        domain_methods = {}
        for method, trajectory in trajectories.items():
            metrics = {name: [] for name in (
                "rmsd", "fnc", "bond_mean", "bond_p95", "bond_p99", "bond_max",
                "bond_mae_real", "angle_cos_mae_real",
            )}
            for step, (pred, _) in enumerate(trajectory):
                rmsd = [aligned_ca_rmsd(pred[i], real[step][i]).item() for i in range(len(starts))]
                fnc = contact_fraction_native(pred, real[step], batch["residue_mask"])
                metrics["rmsd"].append(float(np.mean(rmsd)))
                metrics["fnc"].append(float(fnc.mean().item()))
                geometry = _local_geometry(pred, real[step], batch["bond_mask"])
                for name, value in geometry.items():
                    metrics[name].append(value)
            domain_methods[method] = metrics
        rows.append({
            "domain": h.name,
            "residues_total": layout.num_residues,
            "residues_evaluated": n,
            "frames": n_frames,
            "starts": starts.tolist(),
            "methods": domain_methods,
        })
        h.close()

    methods = ["noop", "one_step_persistence", *requested_methods]
    if args.teacher_forced_mean:
        methods.append("teacher_forced_mean")
    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": ck["step"],
        "settings": vars(args),
        "preprocessing": {
            "canon_symmetric": bool(cd.get("canon_symmetric", False)),
        },
        "delta_frames": delta,
        "domain_panel": {
            "path": args.domain_list,
            "sha256": domain_list_sha256,
            "count": len(domain_ids),
            "evaluated_count": len(chosen),
        },
        "summary": {method: summarize_domains(rows, method, args.steps) for method in methods},
        "domains": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
