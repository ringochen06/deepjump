#!/usr/bin/env python
"""Run the frozen external-H1 context and bond-outlier root-cause probe."""

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
    require_mdcath_full_grid,
    require_single_delta,
    resolve_frozen_domains,
)
from deepjump.metrics import aligned_ca_rmsd
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_model_layout
from deepjump.utils import resolve_device
from scripts.endpoint_grid_eval import _center
from scripts.endpoint_panel_eval import _panel_starts
from scripts.external_endpoint_identity import (
    _sha256,
    load_disjoint_panels,
    verify_multidomain_checkpoint,
    verify_training_fingerprint,
)
from scripts.rollout_robustness_eval import _local_geometry


CONTEXT_CROP = 128
EVAL_CROP = 64
OUTLIER_DOMAIN = "1neiA00"
OUTLIER_TEMPERATURE = 450
OUTLIER_REPLICA = 2


def centered_slice(length: int, width: int) -> slice:
    """Return a deterministic centered slice with exactly ``width`` entries."""
    if width < 1 or length < width:
        raise ValueError("centered slice width must be in [1, length]")
    start = (length - width) // 2
    return slice(start, start + width)


def _cell_tensors(handle, layout, temperature, replica, delta, canon_symmetric, device):
    available = handle.replicas(temperature, [replica])
    if len(available) != 1:
        raise ValueError(f"missing trajectory {handle.name}/{temperature}/{replica}")
    frames = int(available[0][2])
    starts = _panel_starts(frames, delta)
    sources_p, sources_v, targets_p = [], [], []
    for start in starts:
        source = torch.from_numpy(np.asarray(handle.coords(temperature, replica, start)))
        target = torch.from_numpy(np.asarray(handle.coords(temperature, replica, start + delta)))
        p0, v0 = apply_model_layout(source, layout, canon_symmetric=canon_symmetric)
        p1, _ = apply_model_layout(target, layout, canon_symmetric=canon_symmetric)
        sources_p.append(_center(p0))
        sources_v.append(v0)
        targets_p.append(_center(p1))
    return (
        torch.stack(sources_p).to(device),
        torch.stack(sources_v).to(device),
        torch.stack(targets_p).to(device),
        frames,
        starts,
    )


def _batch(p, v, res_index, atom_mask, bond_mask, delta):
    batch_size, residues = p.shape[:2]
    return {
        "P_t": p,
        "V_t": v,
        "res_index": res_index.unsqueeze(0).repeat(batch_size, 1),
        "delta_ns": torch.full((batch_size,), float(delta), device=p.device),
        "residue_mask": torch.ones(batch_size, residues, dtype=torch.bool, device=p.device),
        "atom_mask": atom_mask.unsqueeze(0).repeat(batch_size, 1, 1),
        "bond_mask": bond_mask.unsqueeze(0).repeat(batch_size, 1),
    }


def _padded_crop_batch(crop_batch: dict[str, torch.Tensor], full_length: int) -> dict[str, torch.Tensor]:
    """Pad a real crop to the original length while masking every added residue."""
    batch_size, crop_length = crop_batch["P_t"].shape[:2]
    if full_length < crop_length:
        raise ValueError("full length cannot be smaller than crop length")
    device = crop_batch["P_t"].device
    vector_shape = crop_batch["V_t"].shape[2:]
    padded = {
        "P_t": torch.zeros(batch_size, full_length, 3, device=device),
        "V_t": torch.zeros(batch_size, full_length, *vector_shape, device=device),
        "res_index": torch.zeros(batch_size, full_length, dtype=crop_batch["res_index"].dtype, device=device),
        "delta_ns": crop_batch["delta_ns"].clone(),
        "residue_mask": torch.zeros(batch_size, full_length, dtype=torch.bool, device=device),
        "atom_mask": torch.zeros(
            batch_size, full_length, crop_batch["atom_mask"].shape[-1], dtype=torch.bool, device=device
        ),
        "bond_mask": torch.zeros(batch_size, full_length - 1, dtype=torch.bool, device=device),
    }
    for key in ("P_t", "V_t", "res_index", "residue_mask", "atom_mask"):
        padded[key][:, :crop_length] = crop_batch[key]
    padded["bond_mask"][:, : crop_length - 1] = crop_batch["bond_mask"]
    return padded


def _rmsd_by_start(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    return [float(aligned_ca_rmsd(prediction[i], target[i]).item()) for i in range(len(prediction))]


def _context_cell(model, handle, layout, temperature, replica, delta, canon_symmetric, device):
    p0, v0, p1, frames, starts = _cell_tensors(
        handle, layout, temperature, replica, delta, canon_symmetric, device
    )
    res_index = torch.as_tensor(layout.res_index, device=device)
    atom_mask = torch.as_tensor(layout.atom_mask, device=device)
    bond_mask = torch.as_tensor(layout.bond_mask, device=device)
    full_batch = _batch(p0, v0, res_index, atom_mask, bond_mask, delta)
    pred_full, _ = model.sample(full_batch, steps=1, mode="mean")

    crop = centered_slice(layout.num_residues, CONTEXT_CROP)
    eval_within_crop = centered_slice(CONTEXT_CROP, EVAL_CROP)
    eval_start = crop.start + eval_within_crop.start
    eval_full = slice(eval_start, eval_start + EVAL_CROP)
    crop_p = p0[:, crop] - p0[:, crop].mean(dim=1, keepdim=True)
    crop_batch = _batch(
        crop_p,
        v0[:, crop],
        res_index[crop],
        atom_mask[crop],
        bond_mask[crop.start : crop.stop - 1],
        delta,
    )
    pred_crop, vec_crop = model.sample(crop_batch, steps=1, mode="mean")
    padded_batch = _padded_crop_batch(crop_batch, layout.num_residues)
    pred_padded, vec_padded = model.sample(padded_batch, steps=1, mode="mean")

    target_eval = p1[:, eval_full]
    full_eval = pred_full[:, eval_full]
    crop_eval = pred_crop[:, eval_within_crop]
    noop_eval = p0[:, eval_full]
    full_context = _rmsd_by_start(full_eval, target_eval)
    crop_context = _rmsd_by_start(crop_eval, target_eval)
    noop = _rmsd_by_start(noop_eval, target_eval)
    padding_max_abs = max(
        float((pred_crop - pred_padded[:, :CONTEXT_CROP]).abs().max().item()),
        float((vec_crop - vec_padded[:, :CONTEXT_CROP]).abs().max().item()),
    )
    geometry = _local_geometry(pred_full, p1, full_batch["bond_mask"])
    return {
        "domain": handle.name,
        "temperature": int(temperature),
        "replica": int(replica),
        "frames": frames,
        "starts": starts,
        "full_model_rmsd_by_start": _rmsd_by_start(pred_full, p1),
        "full_noop_rmsd_by_start": _rmsd_by_start(p0, p1),
        "full_bond_mean": geometry["bond_mean"],
        "full_bond_max": geometry["bond_max"],
        "central_full_context_rmsd_by_start": full_context,
        "central_crop_context_rmsd_by_start": crop_context,
        "central_noop_rmsd_by_start": noop,
        "crop_minus_full_by_start": [c - f for c, f in zip(crop_context, full_context)],
        "padding_max_abs_prediction_difference": padding_max_abs,
    }


def _bond_provenance(pred, source, target, bond_mask, res_index):
    records = []
    for start_index in range(pred.shape[0]):
        pred_len = (pred[start_index, 1:] - pred[start_index, :-1]).norm(dim=-1)
        source_len = (source[start_index, 1:] - source[start_index, :-1]).norm(dim=-1)
        target_len = (target[start_index, 1:] - target[start_index, :-1]).norm(dim=-1)
        valid_indices = torch.nonzero(bond_mask[start_index], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("outlier cell has no valid bonds")
        local = int(torch.argmax(pred_len[valid_indices]).item())
        index = int(valid_indices[local].item())
        records.append({
            "start_index": start_index,
            "bond_index": index,
            "res_index_pair": [int(res_index[index].item()), int(res_index[index + 1].item())],
            "predicted_length_fp32": float(pred_len[index].item()),
            "predicted_length_fp64": float(
                (pred[start_index, index + 1].double() - pred[start_index, index].double()).norm().item()
            ),
            "source_length": float(source_len[index].item()),
            "target_length": float(target_len[index].item()),
        })
    return records


def _outlier_probe(model, handle, layout, delta, canon_symmetric, device):
    p0, v0, p1, frames, starts = _cell_tensors(
        handle, layout, OUTLIER_TEMPERATURE, OUTLIER_REPLICA, delta, canon_symmetric, device
    )
    res_index = torch.as_tensor(layout.res_index, device=device)
    atom_mask = torch.as_tensor(layout.atom_mask, device=device)
    bond_mask = torch.as_tensor(layout.bond_mask, device=device)
    batch = _batch(p0, v0, res_index, atom_mask, bond_mask, delta)
    pred_a, _ = model.sample(batch, steps=1, mode="mean")
    pred_b, _ = model.sample(batch, steps=1, mode="mean")
    individual = []
    for index in range(len(starts)):
        single = {key: value[index : index + 1] for key, value in batch.items()}
        prediction, _ = model.sample(single, steps=1, mode="mean")
        individual.append(prediction[0])
    pred_individual = torch.stack(individual)
    geometry = _local_geometry(pred_a, p1, batch["bond_mask"])
    return {
        "domain": handle.name,
        "temperature": OUTLIER_TEMPERATURE,
        "replica": OUTLIER_REPLICA,
        "frames": frames,
        "starts": starts,
        "model_rmsd_by_start": _rmsd_by_start(pred_a, p1),
        "noop_rmsd_by_start": _rmsd_by_start(p0, p1),
        "bond_mean": geometry["bond_mean"],
        "bond_max": geometry["bond_max"],
        "repeat_max_abs_prediction_difference": float((pred_a - pred_b).abs().max().item()),
        "batched_vs_individual_max_abs_prediction_difference": float(
            (pred_a - pred_individual).abs().max().item()
        ),
        "per_start": _bond_provenance(pred_a, p0, p1, batch["bond_mask"], res_index),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-data-root", required=True)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--external-data-root", required=True)
    parser.add_argument("--external-domain-list", required=True)
    parser.add_argument("--external-domain-list-sha256", required=True)
    parser.add_argument("--context-domain-list", required=True)
    parser.add_argument("--context-domain-list-sha256", required=True)
    parser.add_argument("--reference-panel", required=True)
    parser.add_argument("--reference-panel-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint, train_fingerprint = verify_multidomain_checkpoint(args.ckpt, args.checkpoint_sha256)
    training_ids, training_sha, external_ids, external_sha = load_disjoint_panels(
        args.training_domain_list,
        args.training_domain_list_sha256,
        args.external_domain_list,
        args.external_domain_list_sha256,
    )
    context_ids, context_sha = load_frozen_domain_ids(
        args.context_domain_list, args.context_domain_list_sha256
    )
    if len(context_ids) != 9 or not set(context_ids) < set(external_ids):
        raise ValueError("context panel must contain nine unique external domains")
    if _sha256(args.reference_panel) != args.reference_panel_sha256:
        raise ValueError("reference panel SHA256 mismatch")
    training_identity = verify_training_fingerprint(
        checkpoint, train_fingerprint, args.training_data_root, training_ids
    )
    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    delta = require_single_delta(data_cfg["delta_frames"])
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    root = Path(args.external_data_root).expanduser().resolve()
    paths = resolve_frozen_domains(discover_domains(root), context_ids + [OUTLIER_DOMAIN])
    by_domain = {path.stem.replace("mdcath_dataset_", ""): path for path in paths}
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("root-cause probe requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    domains = []
    for domain_id in context_ids:
        handle = _DomainHandle(by_domain[domain_id])
        try:
            if handle.layout.num_residues < CONTEXT_CROP:
                raise ValueError(f"context domain {domain_id} is shorter than {CONTEXT_CROP}")
            cells = [
                _context_cell(
                    model, handle, handle.layout, temperature, replica, delta,
                    bool(data_cfg.get("canon_symmetric", False)), device,
                )
                for temperature in temperatures for replica in replicas
            ]
            domains.append({
                "domain": domain_id,
                "residues": handle.layout.num_residues,
                "cells": cells,
            })
        finally:
            handle.close()

    outlier_handle = _DomainHandle(by_domain[OUTLIER_DOMAIN])
    try:
        outlier = _outlier_probe(
            model, outlier_handle, outlier_handle.layout, delta,
            bool(data_cfg.get("canon_symmetric", False)), device,
        )
    finally:
        outlier_handle.close()

    result = {
        "scope": "external_endpoint_root_cause_v1",
        "checkpoint": args.ckpt,
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_train_fingerprint": train_fingerprint,
        "training_subset": {"sha256": training_sha, "ids": training_ids, **training_identity},
        "external_panel": {"sha256": external_sha, "ids": external_ids, "data_root": str(root)},
        "context_panel": {"sha256": context_sha, "ids": context_ids},
        "reference_panel": {"path": args.reference_panel, "sha256": args.reference_panel_sha256},
        "settings": {
            "delta_frames": delta,
            "starts": 3,
            "method": "mean",
            "source_noise": False,
            "context_crop": CONTEXT_CROP,
            "evaluation_crop": EVAL_CROP,
            "temperatures": temperatures,
            "replicas": replicas,
        },
        "domains": domains,
        "outlier": outlier,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
