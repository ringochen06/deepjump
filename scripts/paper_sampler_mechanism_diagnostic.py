#!/usr/bin/env python
"""Diagnostic-only comparison of one-shot mean and paper-style ODE sampling.

This script is deliberately not an authorization gate.  It evaluates only a
fixed subset of already-consumed development domains and emits mechanism
evidence that cannot authorize external, untouched, or formal-training access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from deepjump.config import ModelConfig
from deepjump.data import discover_domains
from deepjump.data.mdcath import _DomainHandle
from deepjump.evaluation import (
    load_frozen_domain_ids,
    require_mdcath_full_grid,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.model import DeepJumpLite
from deepjump.sampling import reject_to_source
from deepjump.utils import resolve_device
from scripts.external_endpoint_identity import verify_multidomain_checkpoint
from scripts.external_endpoint_root_cause import _batch, _cell_tensors
from scripts.guarded_endpoint_panel_eval import (
    BOND_MAX,
    BOND_MEAN_HI,
    BOND_MEAN_LO,
    EXPECTED_STARTS,
    _bond_metrics_by_start,
    _finite_by_start,
    _rmsd_or_none,
)


SCOPE_BY_DRIFT_ANCHOR = {
    "state": "diagnostic_only_mean_vs_fixed_ode20_consumed_development_v1",
    "conditioner": (
        "diagnostic_only_mean_vs_literal_paper_conditioner_ode20_"
        "consumed_development_v1"
    ),
}
METHODS = ("mean", "ode_20")
EXPECTED_DOMAIN_IDS = (
    "1qu3A05",
    "1s5lH00",
    "2dgmA02",
    "2e9xD02",
    "2nluA02",
    "4i9cA01",
)
EXPECTED_CELLS_PER_DOMAIN = 25


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method_cell(
    *,
    model: DeepJumpLite,
    handle: _DomainHandle,
    temperature: int,
    replica: int,
    delta: int,
    canon_symmetric: bool,
    device: torch.device,
    method: str,
    seed: int,
    ode_drift_anchor: str,
) -> dict:
    source_p, source_v, target_p, frames, starts = _cell_tensors(
        handle,
        handle.layout,
        temperature,
        replica,
        delta,
        canon_symmetric,
        device,
    )
    batch = _batch(
        source_p,
        source_v,
        torch.as_tensor(handle.layout.res_index, device=device),
        torch.as_tensor(handle.layout.atom_mask, device=device),
        torch.as_tensor(handle.layout.bond_mask, dtype=torch.bool, device=device),
        delta,
    )
    if method == "mean":
        raw_p, raw_v = model.sample(batch, steps=1, mode="mean")
    elif method == "ode_20":
        generator = torch.Generator().manual_seed(seed)
        raw_p, raw_v = model.sample(
            batch,
            steps=20,
            mode="ode",
            generator=generator,
            drift_anchor=ode_drift_anchor,
        )
    else:  # pragma: no cover - protected by frozen METHODS
        raise ValueError(f"unsupported diagnostic method: {method}")

    guarded_p, guarded_v, accepted = reject_to_source(
        raw_p,
        raw_v,
        source_p,
        source_v,
        batch["bond_mask"],
        lo=BOND_MEAN_LO,
        hi=BOND_MEAN_HI,
        max_bond=BOND_MAX,
    )
    raw_geometry = _bond_metrics_by_start(raw_p, batch["bond_mask"])
    raw_p_finite = _finite_by_start(raw_p)
    raw_v_finite = _finite_by_start(raw_v)
    noop_rmsd = _rmsd_or_none(source_p, target_p)
    raw_rmsd = _rmsd_or_none(raw_p, target_p)
    guarded_rmsd = _rmsd_or_none(guarded_p, target_p)

    rows = []
    for index, start in enumerate(starts):
        rows.append({
            "start_index": index,
            "start_frame": int(start),
            "accepted": bool(accepted[index].item()),
            "fallback": not bool(accepted[index].item()),
            "raw_position_finite": raw_p_finite[index],
            "raw_vector_finite": raw_v_finite[index],
            "raw_geometry": raw_geometry[index],
            "noop_rmsd": noop_rmsd[index],
            "raw_rmsd": raw_rmsd[index],
            "guarded_rmsd": guarded_rmsd[index],
            "raw_minus_noop": (
                None
                if raw_rmsd[index] is None or noop_rmsd[index] is None
                else raw_rmsd[index] - noop_rmsd[index]
            ),
            "guarded_minus_noop": (
                None
                if guarded_rmsd[index] is None or noop_rmsd[index] is None
                else guarded_rmsd[index] - noop_rmsd[index]
            ),
        })
    return {
        "temperature": int(temperature),
        "replica": int(replica),
        "frames": int(frames),
        "starts": [int(value) for value in starts],
        "fallback_starts": sum(row["fallback"] for row in rows),
        "nonfinite_starts": sum(
            not row["raw_position_finite"] or not row["raw_vector_finite"]
            for row in rows
        ),
        "by_start": rows,
    }


def _summarize(domains: list[dict], method: str) -> dict:
    cells = [
        cell
        for domain in domains
        for cell in domain["methods"][method]["cells"]
    ]
    rows = [row for cell in cells for row in cell["by_start"]]
    raw = [row["raw_minus_noop"] for row in rows]
    guarded = [row["guarded_minus_noop"] for row in rows]
    if any(value is None for value in raw + guarded):
        raise ValueError(f"{method} produced an incomplete RMSD set")
    return {
        "domains": len(domains),
        "cells": len(cells),
        "starts": len(rows),
        "fallback_cells": sum(cell["fallback_starts"] > 0 for cell in cells),
        "fallback_starts": sum(row["fallback"] for row in rows),
        "nonfinite_starts": sum(
            not row["raw_position_finite"] or not row["raw_vector_finite"]
            for row in rows
        ),
        "mean_raw_minus_noop": statistics.fmean(raw),
        "mean_guarded_minus_noop": statistics.fmean(guarded),
    }


def diagnostic_decision(
    summaries: dict[str, dict], *, ode_drift_anchor: str = "state"
) -> dict:
    """Return a conservative mechanism decision with no authorization effect."""
    if ode_drift_anchor not in SCOPE_BY_DRIFT_ANCHOR:
        raise ValueError("unsupported ODE drift anchor")
    mean = summaries["mean"]
    ode = summaries["ode_20"]
    exact_grid = all(
        row["domains"] == len(EXPECTED_DOMAIN_IDS)
        and row["cells"] == len(EXPECTED_DOMAIN_IDS) * EXPECTED_CELLS_PER_DOMAIN
        and row["starts"] == (
            len(EXPECTED_DOMAIN_IDS) * EXPECTED_CELLS_PER_DOMAIN * EXPECTED_STARTS
        )
        for row in (mean, ode)
    )
    supports = (
        exact_grid
        and mean["fallback_starts"] > 0
        and ode["fallback_starts"] == 0
        and ode["nonfinite_starts"] == 0
        and ode["mean_raw_minus_noop"] <= mean["mean_raw_minus_noop"]
    )
    return {
        "status": (
            (
                "SUPPORT_FIXED_ODE20_MECHANISM"
                if ode_drift_anchor == "state"
                else "SUPPORT_LITERAL_PAPER_CONDITIONER_ODE20_MECHANISM"
            )
            if supports
            else (
                "STOP_FIXED_ODE20_MECHANISM"
                if ode_drift_anchor == "state"
                else "STOP_LITERAL_PAPER_CONDITIONER_ODE20_MECHANISM"
            )
        ),
        "exact_grid": exact_grid,
        "ode_drift_anchor": ode_drift_anchor,
        "criteria": {
            "mean_reproduces_compression": mean["fallback_starts"] > 0,
            "ode20_zero_fallbacks": ode["fallback_starts"] == 0,
            "ode20_zero_nonfinite": ode["nonfinite_starts"] == 0,
            "ode20_raw_mean_not_worse_than_mean": (
                ode["mean_raw_minus_noop"] <= mean["mean_raw_minus_noop"]
            ),
        },
        "formal_training_authorized": False,
        "external_authorized": False,
        "untouched_authorized": False,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--ode-drift-anchor",
        choices=tuple(SCOPE_BY_DRIFT_ANCHOR),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint, train_fingerprint = verify_multidomain_checkpoint(
        args.ckpt,
        args.checkpoint_sha256,
        expected_step=args.checkpoint_step,
    )
    domain_ids, panel_sha = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    if tuple(sorted(domain_ids)) != EXPECTED_DOMAIN_IDS:
        raise ValueError("mechanism panel identity mismatch")
    if panel_sha != args.domain_list_sha256:
        raise ValueError("mechanism panel SHA256 mismatch")

    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    delta = require_single_delta(data_cfg["delta_frames"])
    paths = resolve_frozen_domains(
        discover_domains(Path(args.data_root).expanduser().resolve()), domain_ids
    )
    payloads = [{
        "domain": path.stem.replace("mdcath_dataset_", ""),
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in paths]

    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("paper sampler mechanism diagnostic requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    domains = []
    for domain_index, path in enumerate(paths):
        handle = _DomainHandle(path)
        try:
            methods = {}
            for method_index, method in enumerate(METHODS):
                cells = []
                for temperature_index, temperature in enumerate(temperatures):
                    for replica_index, replica in enumerate(replicas):
                        seed = (
                            args.seed
                            + domain_index * 100_000
                            + method_index * 10_000
                            + temperature_index * 100
                            + replica_index
                        )
                        cells.append(_method_cell(
                            model=model,
                            handle=handle,
                            temperature=temperature,
                            replica=replica,
                            delta=delta,
                            canon_symmetric=bool(
                                data_cfg.get("canon_symmetric", False)
                            ),
                            device=device,
                            method=method,
                            seed=seed,
                            ode_drift_anchor=args.ode_drift_anchor,
                        ))
                methods[method] = {"cells": cells}
            domains.append({"domain": handle.name, "methods": methods})
        finally:
            handle.close()

    summaries = {
        method: _summarize(domains, method)
        for method in METHODS
    }
    result = {
        "scope": SCOPE_BY_DRIFT_ANCHOR[args.ode_drift_anchor],
        "scientific_credit": False,
        "checkpoint": {
            "path": str(Path(args.ckpt).resolve()),
            "sha256": args.checkpoint_sha256,
            "step": int(checkpoint["step"]),
            "train_fingerprint": train_fingerprint,
        },
        "panel": {
            "path": str(Path(args.domain_list).resolve()),
            "sha256": panel_sha,
            "ids": domain_ids,
            "already_consumed_development_only": True,
            "payloads": payloads,
        },
        "settings": {
            "methods": list(METHODS),
            "ode_steps": 20,
            "ode_integrator": "euler",
            "ode_drift_anchor": args.ode_drift_anchor,
            "seed": args.seed,
            "temperatures": temperatures,
            "replicas": replicas,
            "starts": EXPECTED_STARTS,
            "fallback_thresholds": {
                "bond_mean_gt": BOND_MEAN_LO,
                "bond_mean_lt": BOND_MEAN_HI,
                "bond_max_lt": BOND_MAX,
            },
        },
        "summaries": summaries,
        "decision": diagnostic_decision(
            summaries, ode_drift_anchor=args.ode_drift_anchor
        ),
        "domains": domains,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "scope": result["scope"],
        "summaries": summaries,
        "decision": result["decision"],
        "output": str(output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
