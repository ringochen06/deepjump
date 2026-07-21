#!/usr/bin/env python
"""Replay the frozen step-2000 external-panel outlier without training."""

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
from deepjump.evaluation import require_single_delta, resolve_frozen_domains
from deepjump.model import DeepJumpLite
from deepjump.utils import resolve_device
from scripts.external_endpoint_identity import (
    _sha256,
    load_disjoint_panels,
    verify_multidomain_checkpoint,
)
from scripts.external_endpoint_root_cause import _batch, _cell_tensors, _rmsd_by_start
from scripts.rollout_robustness_eval import _local_geometry


SCOPE = "external_step2000_outlier_provenance_v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "f3b5965303794e14059f2b67b6b81a538fadb1303c44e1d7c640af44ea690222"
)
EXPECTED_REFERENCE_PANEL_SHA256 = (
    "20dc1b20f5dad02323493d4a24466a35157ff5ff751ce5d7631fd91ceb5fe97e"
)
EXPECTED_EXTERNAL_PANEL_SHA256 = (
    "9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245"
)
EXPECTED_CHECKPOINT_STEP = 2000
EXPECTED_TRAIN_SEED = 0
OUTLIER_DOMAIN = "3ubrA02"
OUTLIER_TEMPERATURE = 450
OUTLIER_REPLICA = 3


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _positions(value: torch.Tensor) -> list[list[float]]:
    return [[float(axis) for axis in row] for row in value.detach().double().cpu().tolist()]


def build_bond_provenance(
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    bond_mask: torch.Tensor,
    res_index: torch.Tensor,
    starts: list[int],
) -> tuple[dict, list[dict]]:
    """Record every topology-valid bond and the coordinates of each per-start maximum."""
    if prediction.shape != source.shape or prediction.shape != target.shape:
        raise ValueError("source, target, and prediction shapes must match")
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("position tensors must have shape [starts, residues, 3]")
    if len(starts) != prediction.shape[0]:
        raise ValueError("start count does not match the prediction batch")
    if bond_mask.shape != (prediction.shape[0], prediction.shape[1] - 1):
        raise ValueError("bond mask shape mismatch")
    if res_index.shape != (prediction.shape[1],):
        raise ValueError("res_index shape mismatch")

    mask = bond_mask[0].bool()
    if not torch.equal(bond_mask.bool(), mask.unsqueeze(0).expand_as(bond_mask)):
        raise ValueError("bond mask differs across starts")
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("outlier cell has no topology-valid bonds")
    res_values = [int(value) for value in res_index.detach().cpu().tolist()]
    mask_values = [bool(value) for value in mask.detach().cpu().tolist()]
    valid_bonds = []
    for index_value in indices.detach().cpu().tolist():
        index = int(index_value)
        valid_bonds.append({
            "bond_index": index,
            "res_index_pair": [res_values[index], res_values[index + 1]],
            "consecutive_res_index": res_values[index + 1] == res_values[index] + 1,
        })
    topology = {
        "residues": prediction.shape[1],
        "res_index": res_values,
        "res_index_sha256": _json_sha256(res_values),
        "bond_mask": mask_values,
        "bond_mask_sha256": _json_sha256(mask_values),
        "valid_bond_count": len(valid_bonds),
        "valid_bonds": valid_bonds,
    }

    records = []
    for start_index, start_frame in enumerate(starts):
        pred_lengths_fp32 = (
            prediction[start_index, 1:] - prediction[start_index, :-1]
        ).norm(dim=-1)
        pred_lengths_fp64 = (
            prediction[start_index, 1:].double()
            - prediction[start_index, :-1].double()
        ).norm(dim=-1)
        source_lengths_fp64 = (
            source[start_index, 1:].double() - source[start_index, :-1].double()
        ).norm(dim=-1)
        target_lengths_fp64 = (
            target[start_index, 1:].double() - target[start_index, :-1].double()
        ).norm(dim=-1)
        valid_indices = indices.to(pred_lengths_fp32.device)
        local_max = int(torch.argmax(pred_lengths_fp32[valid_indices]).item())
        max_index = int(valid_indices[local_max].item())
        all_valid = [
            {
                "bond_index": int(index),
                "source_length": float(source_lengths_fp64[index].item()),
                "target_length": float(target_lengths_fp64[index].item()),
                "predicted_length": float(pred_lengths_fp64[index].item()),
            }
            for index in indices.detach().cpu().tolist()
        ]
        records.append({
            "start_index": start_index,
            "start_frame": int(start_frame),
            "source_positions_sha256": _json_sha256(
                _positions(source[start_index])
            ),
            "target_positions_sha256": _json_sha256(
                _positions(target[start_index])
            ),
            "predicted_positions_sha256": _json_sha256(
                _positions(prediction[start_index])
            ),
            "valid_bond_lengths": all_valid,
            "max_predicted_bond": {
                "bond_index": max_index,
                "res_index_pair": [res_values[max_index], res_values[max_index + 1]],
                "source_positions": _positions(source[start_index, max_index : max_index + 2]),
                "target_positions": _positions(target[start_index, max_index : max_index + 2]),
                "predicted_positions": _positions(
                    prediction[start_index, max_index : max_index + 2]
                ),
                "source_length": float(source_lengths_fp64[max_index].item()),
                "target_length": float(target_lengths_fp64[max_index].item()),
                "predicted_length_fp32": float(pred_lengths_fp32[max_index].item()),
                "predicted_length_fp64": float(pred_lengths_fp64[max_index].item()),
            },
        })
    return topology, records


def _reference_cell(panel: dict) -> dict:
    matches = [
        cell
        for domain in panel.get("domains", [])
        if domain.get("domain") == OUTLIER_DOMAIN
        for cell in domain.get("cells", [])
        if int(cell.get("temperature", -1)) == OUTLIER_TEMPERATURE
        and int(cell.get("replica", -1)) == OUTLIER_REPLICA
    ]
    if len(matches) != 1:
        raise ValueError("reference panel does not contain exactly one frozen outlier cell")
    return matches[0]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--external-data-root", required=True)
    parser.add_argument("--external-domain-list", required=True)
    parser.add_argument("--external-domain-list-sha256", required=True)
    parser.add_argument("--reference-panel", required=True)
    parser.add_argument("--reference-panel-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("step2000 replay checkpoint identity mismatch")
    if args.reference_panel_sha256 != EXPECTED_REFERENCE_PANEL_SHA256:
        raise ValueError("step2000 replay reference panel identity mismatch")
    if args.external_domain_list_sha256 != EXPECTED_EXTERNAL_PANEL_SHA256:
        raise ValueError("step2000 replay external panel identity mismatch")
    checkpoint, train_fingerprint = verify_multidomain_checkpoint(
        args.ckpt,
        args.checkpoint_sha256,
        expected_step=EXPECTED_CHECKPOINT_STEP,
        expected_train_seed=EXPECTED_TRAIN_SEED,
    )
    _, training_sha, external_ids, external_sha = load_disjoint_panels(
        args.training_domain_list,
        args.training_domain_list_sha256,
        args.external_domain_list,
        args.external_domain_list_sha256,
    )
    if OUTLIER_DOMAIN not in external_ids:
        raise ValueError("frozen outlier domain is absent from the external panel")
    if _sha256(args.reference_panel) != args.reference_panel_sha256:
        raise ValueError("reference panel SHA256 mismatch")
    reference = json.loads(Path(args.reference_panel).read_text())
    frozen_cell = _reference_cell(reference)

    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    delta = require_single_delta(data_cfg["delta_frames"])
    root = Path(args.external_data_root).expanduser().resolve()
    path = resolve_frozen_domains(discover_domains(root), [OUTLIER_DOMAIN])[0]
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("step2000 outlier replay requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    handle = _DomainHandle(path)
    try:
        layout = handle.layout
        source, vectors, target, frames, starts = _cell_tensors(
            handle,
            layout,
            OUTLIER_TEMPERATURE,
            OUTLIER_REPLICA,
            delta,
            bool(data_cfg.get("canon_symmetric", False)),
            device,
        )
        res_index = torch.as_tensor(layout.res_index, device=device)
        atom_mask = torch.as_tensor(layout.atom_mask, device=device)
        bond_mask = torch.as_tensor(layout.bond_mask, device=device)
        batch = _batch(source, vectors, res_index, atom_mask, bond_mask, delta)
        prediction_a, _ = model.sample(batch, steps=1, mode="mean")
        prediction_b, _ = model.sample(batch, steps=1, mode="mean")
        individual = []
        for index in range(len(starts)):
            single = {name: value[index : index + 1] for name, value in batch.items()}
            prediction, _ = model.sample(single, steps=1, mode="mean")
            individual.append(prediction[0])
        prediction_individual = torch.stack(individual)
        geometry = _local_geometry(prediction_a, target, batch["bond_mask"])
        topology, per_start = build_bond_provenance(
            prediction_a, source, target, batch["bond_mask"], res_index, starts
        )
    finally:
        handle.close()

    model_rmsd = _rmsd_by_start(prediction_a, target)
    noop_rmsd = _rmsd_by_start(source, target)
    paired = [model - noop for model, noop in zip(model_rmsd, noop_rmsd)]
    result = {
        "scope": SCOPE,
        "checkpoint": args.ckpt,
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_train_seed": int(checkpoint["cfg"]["train"]["seed"]),
        "checkpoint_train_fingerprint": train_fingerprint,
        "training_domain_list_sha256": training_sha,
        "external_panel": {"sha256": external_sha, "ids": external_ids},
        "reference_panel": {
            "path": args.reference_panel,
            "sha256": args.reference_panel_sha256,
        },
        "cell": {
            "domain": OUTLIER_DOMAIN,
            "temperature": OUTLIER_TEMPERATURE,
            "replica": OUTLIER_REPLICA,
            "frames": frames,
            "starts": starts,
        },
        "settings": {"delta_frames": delta, "steps": 1, "method": "mean", "source_noise": False},
        "replay": {
            "model_rmsd_by_start": model_rmsd,
            "noop_rmsd_by_start": noop_rmsd,
            "model_minus_noop_by_start": paired,
            "mean_model_minus_noop": statistics.fmean(paired),
            "bond_mean": geometry["bond_mean"],
            "bond_max": geometry["bond_max"],
            "repeat_max_abs_prediction_difference": float(
                (prediction_a - prediction_b).abs().max().item()
            ),
            "batched_vs_individual_max_abs_prediction_difference": float(
                (prediction_a - prediction_individual).abs().max().item()
            ),
            "topology": topology,
            "per_start": per_start,
        },
        "frozen_panel_cell": {
            name: frozen_cell[name]
            for name in (
                "domain", "temperature", "replica", "frames", "starts",
                "model_rmsd_by_start", "noop_rmsd_by_start", "model_minus_noop_by_start",
                "mean_model_minus_noop", "bond_mean", "bond_max",
            )
        },
        "formal_training_authorized": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
