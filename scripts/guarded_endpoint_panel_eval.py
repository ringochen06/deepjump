#!/usr/bin/env python
"""Evaluate the frozen training-domain panel with a reject-to-source safeguard."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
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
from deepjump.metrics import aligned_ca_rmsd
from deepjump.model import DeepJumpLite
from deepjump.sampling import reject_to_source
from deepjump.utils import resolve_device
from scripts.endpoint_panel_eval import (
    EXPECTED_DOMAINS,
    EXPECTED_STARTS,
    _panel_starts,
    _runtime_probe_status,
)
from scripts.external_endpoint_identity import (
    load_fresh_external_panels,
    load_paper_horizon_external_panels,
    verify_guarded_training_prerequisite,
    verify_multidomain_checkpoint,
    verify_paper_horizon_ab_prerequisite,
    verify_paper_vector_ab_prerequisite,
    verify_paper_vector_external_evidence,
    verify_training_fingerprint,
)
from scripts.external_endpoint_root_cause import _batch, _cell_tensors


SCOPE = "conditional_reject_to_source_training_dev20_v1"
EXTERNAL_SCOPE = "conditional_reject_to_source_fresh_external_dev20_v1"
PAPER_HORIZON_EXTERNAL_SCOPE = (
    "conditional_reject_to_source_paper_horizon_external_dev20_v1"
)
PAPER_VECTOR_EXTERNAL_SCOPE = (
    "conditional_reject_to_source_paper_vector_external_dev20_v1"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "f3b5965303794e14059f2b67b6b81a538fadb1303c44e1d7c640af44ea690222"
)
EXPECTED_CHECKPOINT_STEP = 2000
EXPECTED_TRAINING_SHA256 = (
    "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734"
)
EXPECTED_PANEL_SHA256 = (
    "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af"
)
EXPECTED_EXTERNAL_PANEL_SHA256 = (
    "9bae11fa0e6336e7451c372efa25ca55af77aa9cb27f91e1fd241612531a920f"
)
EXPECTED_PRIOR_EXTERNAL_SHA256 = (
    "9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245"
)
EXPECTED_UNTOUCHED_SHA256 = (
    "e56ed7de735db542f4e20fb73f2654a6c1bcf67f3082849f63f0ab74f4208c38"
)
EXPECTED_TRAINING_DECISION_SHA256 = (
    "b234f31db96c2f461ea0abd056aa6e724d2d94aa52930bbc990c43cfc302000b"
)
EXPECTED_EXTERNAL_BYTES = 13_354_825_648
EXPECTED_PAPER_HORIZON_EXTERNAL_PANEL_SHA256 = (
    "9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb"
)
EXPECTED_PAPER_HORIZON_EXTERNAL_BYTES = 14_236_836_972
BOND_MEAN_LO = 3.2
BOND_MEAN_HI = 4.5
BOND_MAX = 5.5
MAX_FALLBACK_STARTS = 3
MAX_FALLBACK_CELLS = 1
FROZEN_BASELINE_PROFILE = "frozen-baseline"
HORIZON_AB_BASELINE_PROFILE = "paper-horizon-ab-baseline1000"
PAPER_HORIZON_PROFILE = "paper-horizon-500k"
PAPER_VECTOR_PROFILE = "paper-horizon-vector-only-500k"
PAPER_SCALAR_VALUE_PROFILE = "paper-horizon-vector-scalar-value-500k"
CHECKPOINT_PROFILES = (
    FROZEN_BASELINE_PROFILE,
    HORIZON_AB_BASELINE_PROFILE,
    PAPER_HORIZON_PROFILE,
    PAPER_VECTOR_PROFILE,
    PAPER_SCALAR_VALUE_PROFILE,
)

_PAPER_HORIZON_DATA_CONFIG = {
    "root": "/data/mdcath",
    "manifest": "/data/mdcath/manifest.json",
    "domains": [],
    "temperatures": [320, 348, 379, 413, 450],
    "replicas": [0, 1, 2, 3, 4],
    "delta_frames": 1,
    "crop_length": 256,
    "val_fraction": 0.02,
    "noise_sigma": 0.1,
    "unroll": 1,
    "canon_symmetric": True,
    "max_open_files": 96,
    "seed": 0,
}
_PAPER_HORIZON_MODEL_CONFIG = {
    "hidden": 128,
    "vector_channels": 128,
    "num_heads": 4,
    "cond_layers": 6,
    "transport_layers": 6,
    "seq_embed_ks": 32,
    "num_dist_basis": 16,
    "dist_cutoff": 25.0,
    "predict_heavy": True,
    "input_aug_sigma": 0.0,
    "source_noise_v": True,
    "source_noise_sigma_v": None,
    "vector_qk": False,
    "tensor_qkv": False,
    "paper_ff": False,
    "tensor_cloud01": True,
    "tensor_cloud01_vector_only_attention": False,
    "tensor_cloud01_vector_only_scalar_value": False,
}
_PAPER_HORIZON_TRAIN_CONFIG = {
    "batch_size": 2,
    "grad_accum": 8,
    "lr": 5.0e-3,
    "lr_final": 3.0e-3,
    "warmup_steps": 200,
    "grad_clip": 0.1,
    "max_steps": 2000,
    "val_every": 100,
    "log_every": 50,
    "ckpt_every": 1000,
    "keep_last_k": 2,
    "huber_delta": 1.0,
    "geom_huber_delta": 0.05,
    "w_ca": 0.0,
    "w_bond": 0.0,
    "w_bond_unroll": 0.0,
    "w_geom_length_unroll": 0.0,
    "w_geom_angle_unroll": 0.0,
    "w_offset": 0.0,
    "w_allatom": 1.0,
    "w_unroll": 0.0,
    "amp": False,
    "amp_dtype": "fp16",
    "num_workers": 8,
    "device": "cuda",
    "resume": "",
    "seed": 0,
}


def checkpoint_profile_requirements(
    profile: str, checkpoint_sha256: str
) -> tuple[dict | None, dict | None, dict | None]:
    """Return exact checkpoint recipe constraints for a guarded-panel profile."""
    if profile == FROZEN_BASELINE_PROFILE:
        if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("checkpoint is not the frozen full-tensor step2000 artifact")
        return None, None, None
    if profile not in {
        HORIZON_AB_BASELINE_PROFILE,
        PAPER_HORIZON_PROFILE,
        PAPER_VECTOR_PROFILE,
        PAPER_SCALAR_VALUE_PROFILE,
    }:
        raise ValueError("unknown checkpoint profile")
    if len(checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA256 must be 64 lowercase hex characters")
    horizon = 1000 if profile == HORIZON_AB_BASELINE_PROFILE else 500000
    if profile == HORIZON_AB_BASELINE_PROFILE:
        out_dir = "runs/v100_tensorcloud01_full_d1_fp32_horizon_ab_baseline2000"
    elif profile == PAPER_HORIZON_PROFILE:
        out_dir = "runs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000"
    elif profile == PAPER_VECTOR_PROFILE:
        out_dir = (
            "runs/v100_tensorcloud01_vector_only_d1_fp32_"
            "paper_horizon500k_2000"
        )
    else:
        out_dir = (
            "runs/v100_tensorcloud01_vector_scalar_value_d1_fp32_"
            "paper_horizon500k_2000"
        )
    model = dict(_PAPER_HORIZON_MODEL_CONFIG)
    model["tensor_cloud01_vector_only_attention"] = (
        profile in {PAPER_VECTOR_PROFILE, PAPER_SCALAR_VALUE_PROFILE}
    )
    model["tensor_cloud01_vector_only_scalar_value"] = (
        profile == PAPER_SCALAR_VALUE_PROFILE
    )
    train = {
        **_PAPER_HORIZON_TRAIN_CONFIG,
        "lr_horizon_steps": horizon,
        "out_dir": out_dir,
    }
    return _PAPER_HORIZON_DATA_CONFIG, model, train


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _bond_metrics_by_start(
    positions: torch.Tensor, bond_mask: torch.Tensor
) -> list[dict]:
    lengths = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)
    rows = []
    for index in range(positions.shape[0]):
        valid = bond_mask[index].bool()
        if not valid.any():
            raise ValueError("every panel start must have a topology-valid bond")
        values = lengths[index][valid]
        mean = _finite_or_none(values.mean().item())
        maximum = _finite_or_none(values.max().item())
        rows.append({
            "bond_mean": mean,
            "bond_max": maximum,
            "physical": bool(
                mean is not None
                and maximum is not None
                and BOND_MEAN_LO < mean < BOND_MEAN_HI
                and maximum < BOND_MAX
            ),
        })
    return rows


def _finite_by_start(value: torch.Tensor) -> list[bool]:
    return [bool(row) for row in torch.isfinite(value).flatten(1).all(1).tolist()]


def _rmsd_or_none(prediction: torch.Tensor, target: torch.Tensor) -> list[float | None]:
    rows = []
    for index in range(prediction.shape[0]):
        if not torch.isfinite(prediction[index]).all():
            rows.append(None)
        else:
            rows.append(_finite_or_none(aligned_ca_rmsd(
                prediction[index], target[index]
            ).item()))
    return rows


def _evaluate_cell(
    *,
    model: DeepJumpLite,
    handle: _DomainHandle,
    layout,
    temperature: int,
    replica: int,
    delta: int,
    canon_symmetric: bool,
    device: torch.device,
) -> dict:
    source_P, source_V, target_P, frames, starts = _cell_tensors(
        handle,
        layout,
        temperature,
        replica,
        delta,
        canon_symmetric,
        device,
    )
    residue_index = torch.as_tensor(layout.res_index, device=device)
    atom_mask = torch.as_tensor(layout.atom_mask, device=device)
    topology = torch.as_tensor(layout.bond_mask, dtype=torch.bool, device=device)
    batch = _batch(
        source_P, source_V, residue_index, atom_mask, topology, delta
    )
    raw_P, raw_V = model.sample(batch, steps=1, mode="mean")
    guarded_P, guarded_V, accepted = reject_to_source(
        raw_P,
        raw_V,
        source_P,
        source_V,
        batch["bond_mask"],
        lo=BOND_MEAN_LO,
        hi=BOND_MEAN_HI,
        max_bond=BOND_MAX,
    )

    source_geometry = _bond_metrics_by_start(source_P, batch["bond_mask"])
    raw_geometry = _bond_metrics_by_start(raw_P, batch["bond_mask"])
    guarded_geometry = _bond_metrics_by_start(guarded_P, batch["bond_mask"])
    source_p_finite = _finite_by_start(source_P)
    source_v_finite = _finite_by_start(source_V)
    raw_p_finite = _finite_by_start(raw_P)
    raw_v_finite = _finite_by_start(raw_V)
    guarded_p_finite = _finite_by_start(guarded_P)
    guarded_v_finite = _finite_by_start(guarded_V)
    noop_rmsd = _rmsd_or_none(source_P, target_P)
    raw_rmsd = _rmsd_or_none(raw_P, target_P)
    guarded_rmsd = _rmsd_or_none(guarded_P, target_P)

    by_start = []
    for index, start in enumerate(starts):
        raw_delta = (
            None if raw_rmsd[index] is None or noop_rmsd[index] is None
            else raw_rmsd[index] - noop_rmsd[index]
        )
        guarded_delta = (
            None if guarded_rmsd[index] is None or noop_rmsd[index] is None
            else guarded_rmsd[index] - noop_rmsd[index]
        )
        is_accepted = bool(accepted[index].item())
        selected_P = raw_P[index] if is_accepted else source_P[index]
        selected_V = raw_V[index] if is_accepted else source_V[index]
        by_start.append({
            "start_index": index,
            "start_frame": int(start),
            "target_position_finite": bool(torch.isfinite(target_P[index]).all()),
            "noop_rmsd": noop_rmsd[index],
            "accepted": is_accepted,
            "fallback": not is_accepted,
            "selected_position_exact": bool(torch.equal(guarded_P[index], selected_P)),
            "selected_vector_exact": bool(torch.equal(guarded_V[index], selected_V)),
            "source": {
                "position_finite": source_p_finite[index],
                "vector_finite": source_v_finite[index],
                **source_geometry[index],
            },
            "raw": {
                "position_finite": raw_p_finite[index],
                "vector_finite": raw_v_finite[index],
                "rmsd": raw_rmsd[index],
                "minus_noop": raw_delta,
                **raw_geometry[index],
            },
            "guarded": {
                "position_finite": guarded_p_finite[index],
                "vector_finite": guarded_v_finite[index],
                "rmsd": guarded_rmsd[index],
                "minus_noop": guarded_delta,
                **guarded_geometry[index],
            },
        })

    guarded_deltas = [row["guarded"]["minus_noop"] for row in by_start]
    mean_guarded_delta = (
        statistics.fmean(guarded_deltas)
        if all(value is not None for value in guarded_deltas)
        else None
    )
    return {
        "domain": handle.name,
        "temperature": int(temperature),
        "replica": int(replica),
        "frames": int(frames),
        "starts": [int(value) for value in starts],
        "by_start": by_start,
        "mean_guarded_minus_noop": mean_guarded_delta,
        "source_cell_physical": all(row["source"]["physical"] for row in by_start),
        "raw_cell_physical": all(row["raw"]["physical"] for row in by_start),
        "guarded_cell_physical": all(row["guarded"]["physical"] for row in by_start),
        "fallback_starts": sum(row["fallback"] for row in by_start),
    }


def _convert_batch_dtype(batch: dict[str, torch.Tensor], dtype: torch.dtype) -> dict:
    return {
        key: value.to(dtype=dtype) if value.is_floating_point() else value
        for key, value in batch.items()
    }


def _mechanism_probe(
    checkpoint: dict,
    model: DeepJumpLite,
    path: Path,
    temperature: int,
    replica: int,
    delta: int,
    canon_symmetric: bool,
    device: torch.device,
) -> dict:
    handle = _DomainHandle(path)
    try:
        layout = handle.layout
        source_P, source_V, _, _, starts = _cell_tensors(
            handle, layout, temperature, replica, delta, canon_symmetric, device
        )
        residue_index = torch.as_tensor(layout.res_index, device=device)
        atom_mask = torch.as_tensor(layout.atom_mask, device=device)
        topology = torch.as_tensor(layout.bond_mask, dtype=torch.bool, device=device)
        batch = _batch(
            source_P, source_V, residue_index, atom_mask, topology, delta
        )
        raw_P, raw_V = model.sample(batch, steps=1, mode="mean")

        peer_batch = {key: value.clone() for key, value in batch.items()}
        for value in peer_batch.values():
            value[1:] = value[0:1]
        peer_P, peer_V = model.sample(peer_batch, steps=1, mode="mean")

        single = {key: value[:1] for key, value in batch.items()}
        single_P, single_V = model.sample(single, steps=1, mode="mean")
        _, _, accept_b3 = reject_to_source(
            raw_P, raw_V, source_P, source_V, batch["bond_mask"]
        )
        _, _, accept_b1 = reject_to_source(
            single_P, single_V, single["P_t"], single["V_t"], single["bond_mask"]
        )

        model_cfg = checkpoint["cfg"]["model"]
        data_cfg = checkpoint["cfg"]["data"]
        model64 = DeepJumpLite(
            ModelConfig(**model_cfg),
            noise_sigma=float(data_cfg["noise_sigma"]),
            predict_heavy=bool(model_cfg["predict_heavy"]),
        ).to(device)
        model64.load_state_dict(checkpoint["model"])
        model64.double().eval()
        batch64 = _convert_batch_dtype(batch, torch.float64)
        single64 = {key: value[:1] for key, value in batch64.items()}
        raw64_P, raw64_V = model64.sample(batch64, steps=1, mode="mean")
        single64_P, single64_V = model64.sample(single64, steps=1, mode="mean")
        _, _, accept64_b3 = reject_to_source(
            raw64_P,
            raw64_V,
            batch64["P_t"],
            batch64["V_t"],
            batch64["bond_mask"],
        )
        _, _, accept64_b1 = reject_to_source(
            single64_P,
            single64_V,
            single64["P_t"],
            single64["V_t"],
            single64["bond_mask"],
        )
        return {
            "domain": handle.name,
            "temperature": int(temperature),
            "replica": int(replica),
            "target_slot": 0,
            "target_start": int(starts[0]),
            "same_shape_peer_position_bitwise_equal": bool(torch.equal(raw_P[0], peer_P[0])),
            "same_shape_peer_vector_bitwise_equal": bool(torch.equal(raw_V[0], peer_V[0])),
            "fp32_b1_b3_position_max_abs_diff": float(
                (single_P[0] - raw_P[0]).abs().max().item()
            ),
            "fp32_b1_b3_vector_max_abs_diff": float(
                (single_V[0] - raw_V[0]).abs().max().item()
            ),
            "fp32_accept_b1": bool(accept_b1[0].item()),
            "fp32_accept_b3": bool(accept_b3[0].item()),
            "fp64_b1_b3_position_max_abs_diff": float(
                (single64_P[0] - raw64_P[0]).abs().max().item()
            ),
            "fp64_b1_b3_vector_max_abs_diff": float(
                (single64_V[0] - raw64_V[0]).abs().max().item()
            ),
            "fp64_accept_b1": bool(accept64_b1[0].item()),
            "fp64_accept_b3": bool(accept64_b3[0].item()),
        }
    finally:
        handle.close()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-profile", choices=CHECKPOINT_PROFILES,
                        default=FROZEN_BASELINE_PROFILE)
    parser.add_argument("--training-data-root", required=True)
    parser.add_argument("--training-domain-list", required=True)
    parser.add_argument("--training-domain-list-sha256", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument(
        "--panel-kind",
        choices=(
            "training",
            "fresh-external",
            "paper-horizon-external",
            "paper-vector-external",
        ),
        default="training",
    )
    parser.add_argument("--panel-data-root")
    parser.add_argument("--prior-external-domain-list")
    parser.add_argument("--prior-external-domain-list-sha256")
    parser.add_argument("--prior-fresh-external-domain-list")
    parser.add_argument("--prior-fresh-external-domain-list-sha256")
    parser.add_argument("--untouched-domain-list")
    parser.add_argument("--untouched-domain-list-sha256")
    parser.add_argument("--prerequisite-decision")
    parser.add_argument("--prerequisite-decision-sha256")
    parser.add_argument("--baseline-checkpoint-sha256")
    parser.add_argument("--candidate-checkpoint-sha256")
    parser.add_argument("--external-claim")
    parser.add_argument("--external-claim-sha256")
    parser.add_argument("--external-download-manifest")
    parser.add_argument("--external-download-manifest-sha256")
    parser.add_argument("--source-proof")
    parser.add_argument("--source-proof-sha256")
    parser.add_argument("--runtime-probe-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        expected_data, expected_model, expected_train = checkpoint_profile_requirements(
            args.checkpoint_profile, args.checkpoint_sha256
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.panel_kind == "fresh-external" and args.checkpoint_profile != FROZEN_BASELINE_PROFILE:
        parser.error("legacy fresh-external requires the frozen baseline profile")
    if args.panel_kind == "paper-horizon-external" and args.checkpoint_profile not in {
        HORIZON_AB_BASELINE_PROFILE, PAPER_HORIZON_PROFILE
    }:
        parser.error("paper-horizon external requires a matched A/B checkpoint profile")
    if args.panel_kind == "paper-vector-external" and args.checkpoint_profile not in {
        PAPER_HORIZON_PROFILE, PAPER_VECTOR_PROFILE
    }:
        parser.error("paper-vector external requires a matched A/B checkpoint profile")
    if args.training_domain_list_sha256 != EXPECTED_TRAINING_SHA256:
        parser.error("training subset identity mismatch")
    expected_panel_sha = {
        "training": EXPECTED_PANEL_SHA256,
        "fresh-external": EXPECTED_EXTERNAL_PANEL_SHA256,
        "paper-horizon-external": EXPECTED_PAPER_HORIZON_EXTERNAL_PANEL_SHA256,
        "paper-vector-external": EXPECTED_PAPER_HORIZON_EXTERNAL_PANEL_SHA256,
    }[args.panel_kind]
    if args.domain_list_sha256 != expected_panel_sha:
        parser.error(f"{args.panel_kind} panel identity mismatch")
    checkpoint, train_fingerprint = verify_multidomain_checkpoint(
        args.ckpt,
        args.checkpoint_sha256,
        expected_step=EXPECTED_CHECKPOINT_STEP,
        expected_data_config=expected_data,
        expected_model_config=expected_model,
        expected_train_config=expected_train,
    )
    training_ids, training_sha = load_frozen_domain_ids(
        args.training_domain_list, args.training_domain_list_sha256
    )
    prerequisite = None
    external_evidence = None
    panel_contract = None
    if args.panel_kind == "fresh-external":
        required = {
            "prior external list": args.prior_external_domain_list,
            "prior external SHA256": args.prior_external_domain_list_sha256,
            "untouched list": args.untouched_domain_list,
            "untouched SHA256": args.untouched_domain_list_sha256,
            "prerequisite decision": args.prerequisite_decision,
            "prerequisite decision SHA256": args.prerequisite_decision_sha256,
            "panel data root": args.panel_data_root,
        }
        missing = [label for label, value in required.items() if not value]
        if missing:
            parser.error(f"fresh external panel is missing: {', '.join(missing)}")
        if args.prior_external_domain_list_sha256 != EXPECTED_PRIOR_EXTERNAL_SHA256:
            parser.error("prior external panel identity mismatch")
        if args.untouched_domain_list_sha256 != EXPECTED_UNTOUCHED_SHA256:
            parser.error("untouched panel identity mismatch")
        if args.prerequisite_decision_sha256 != EXPECTED_TRAINING_DECISION_SHA256:
            parser.error("training prerequisite decision identity mismatch")
        panel_contract = load_fresh_external_panels(
            args.training_domain_list, args.training_domain_list_sha256,
            args.prior_external_domain_list, args.prior_external_domain_list_sha256,
            args.untouched_domain_list, args.untouched_domain_list_sha256,
            args.domain_list, args.domain_list_sha256,
        )
        panel_ids = panel_contract["fresh_external"]["ids"]
        panel_sha = panel_contract["fresh_external"]["sha256"]
        prerequisite = verify_guarded_training_prerequisite(
            args.prerequisite_decision,
            args.prerequisite_decision_sha256,
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            expected_training_sha256=EXPECTED_TRAINING_SHA256,
        )
    elif args.panel_kind in {"paper-horizon-external", "paper-vector-external"}:
        required = {
            "prior external list": args.prior_external_domain_list,
            "prior external SHA256": args.prior_external_domain_list_sha256,
            "prior fresh external list": args.prior_fresh_external_domain_list,
            "prior fresh external SHA256": args.prior_fresh_external_domain_list_sha256,
            "untouched list": args.untouched_domain_list,
            "untouched SHA256": args.untouched_domain_list_sha256,
            "A/B prerequisite decision": args.prerequisite_decision,
            "A/B prerequisite decision SHA256": args.prerequisite_decision_sha256,
            "baseline checkpoint SHA256": (
                args.baseline_checkpoint_sha256
                if args.panel_kind == "paper-vector-external" else True
            ),
            "candidate checkpoint SHA256": args.candidate_checkpoint_sha256,
            "external claim": (
                args.external_claim
                if args.panel_kind == "paper-vector-external" else True
            ),
            "external claim SHA256": (
                args.external_claim_sha256
                if args.panel_kind == "paper-vector-external" else True
            ),
            "external download manifest": (
                args.external_download_manifest
                if args.panel_kind == "paper-vector-external" else True
            ),
            "external download manifest SHA256": (
                args.external_download_manifest_sha256
                if args.panel_kind == "paper-vector-external" else True
            ),
            "source proof": (
                args.source_proof
                if args.panel_kind == "paper-vector-external" else True
            ),
            "source proof SHA256": (
                args.source_proof_sha256
                if args.panel_kind == "paper-vector-external" else True
            ),
            "panel data root": args.panel_data_root,
        }
        missing = [label for label, value in required.items() if not value]
        if missing:
            parser.error(f"{args.panel_kind} panel is missing: {', '.join(missing)}")
        if args.prior_external_domain_list_sha256 != EXPECTED_PRIOR_EXTERNAL_SHA256:
            parser.error("prior external panel identity mismatch")
        if args.prior_fresh_external_domain_list_sha256 != EXPECTED_EXTERNAL_PANEL_SHA256:
            parser.error("prior fresh external panel identity mismatch")
        if args.untouched_domain_list_sha256 != EXPECTED_UNTOUCHED_SHA256:
            parser.error("untouched panel identity mismatch")
        panel_contract = load_paper_horizon_external_panels(
            args.training_domain_list, args.training_domain_list_sha256,
            args.prior_external_domain_list, args.prior_external_domain_list_sha256,
            args.prior_fresh_external_domain_list,
            args.prior_fresh_external_domain_list_sha256,
            args.untouched_domain_list, args.untouched_domain_list_sha256,
            args.domain_list, args.domain_list_sha256,
        )
        panel_ids = panel_contract["paper_horizon_external"]["ids"]
        panel_sha = panel_contract["paper_horizon_external"]["sha256"]
        if args.panel_kind == "paper-horizon-external":
            prerequisite = verify_paper_horizon_ab_prerequisite(
                args.prerequisite_decision,
                args.prerequisite_decision_sha256,
                expected_candidate_checkpoint_sha256=(
                    args.candidate_checkpoint_sha256
                ),
                expected_training_sha256=EXPECTED_TRAINING_SHA256,
                expected_training_panel_sha256=EXPECTED_PANEL_SHA256,
            )
        else:
            prerequisite = verify_paper_vector_ab_prerequisite(
                args.prerequisite_decision,
                args.prerequisite_decision_sha256,
                expected_baseline_checkpoint_sha256=(
                    args.baseline_checkpoint_sha256
                ),
                expected_candidate_checkpoint_sha256=(
                    args.candidate_checkpoint_sha256
                ),
                expected_training_sha256=EXPECTED_TRAINING_SHA256,
                expected_training_panel_sha256=EXPECTED_PANEL_SHA256,
            )
            external_evidence = verify_paper_vector_external_evidence(
                args.external_claim,
                args.external_claim_sha256,
                args.external_download_manifest,
                args.external_download_manifest_sha256,
                expected_panel_sha256=panel_sha,
                expected_prerequisite_decision_sha256=(
                    args.prerequisite_decision_sha256
                ),
                expected_baseline_checkpoint_sha256=(
                    args.baseline_checkpoint_sha256
                ),
                expected_candidate_checkpoint_sha256=(
                    args.candidate_checkpoint_sha256
                ),
                source_proof_path=args.source_proof,
                expected_source_proof_sha256=args.source_proof_sha256,
                panel_data_root=args.panel_data_root,
            )
    else:
        panel_ids, panel_sha = load_frozen_domain_ids(
            args.domain_list, args.domain_list_sha256
        )
    if len(training_ids) != 1000 or len(set(training_ids)) != 1000:
        raise ValueError("training subset must contain 1000 unique domains")
    if len(panel_ids) != EXPECTED_DOMAINS or len(set(panel_ids)) != EXPECTED_DOMAINS:
        raise ValueError("training development panel must contain 20 unique domains")
    if args.panel_kind == "training" and not set(panel_ids).issubset(training_ids):
        raise ValueError("training development panel must be a subset of training1000")
    training_identity = verify_training_fingerprint(
        checkpoint,
        train_fingerprint,
        args.training_data_root,
        training_ids,
    )

    data_cfg = checkpoint["cfg"]["data"]
    model_cfg = checkpoint["cfg"]["model"]
    delta = require_single_delta(data_cfg["delta_frames"])
    temperatures, replicas = require_mdcath_full_grid(
        data_cfg["temperatures"], data_cfg["replicas"]
    )
    root = Path(args.panel_data_root or args.training_data_root).expanduser().resolve()
    paths = resolve_frozen_domains(discover_domains(root), panel_ids)
    panel_total_bytes = sum(path.stat().st_size for path in paths)
    if args.panel_kind == "fresh-external" and panel_total_bytes != EXPECTED_EXTERNAL_BYTES:
        raise ValueError("fresh external panel byte count mismatch")
    if (
        args.panel_kind in {"paper-horizon-external", "paper-vector-external"}
        and panel_total_bytes != EXPECTED_PAPER_HORIZON_EXTERNAL_BYTES
    ):
        raise ValueError("paper-horizon external panel byte count mismatch")
    device = resolve_device(checkpoint["cfg"]["train"]["device"])
    if device.type != "cuda":
        raise ValueError("conditional safeguard panel requires CUDA")
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=float(data_cfg["noise_sigma"]),
        predict_heavy=bool(model_cfg["predict_heavy"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mechanism = _mechanism_probe(
        checkpoint,
        model,
        paths[0],
        temperatures[0],
        replicas[0],
        delta,
        bool(data_cfg.get("canon_symmetric", False)),
        device,
    )

    residue_counts = []
    for path in paths:
        handle = _DomainHandle(path)
        try:
            residue_counts.append((handle.layout.num_residues, path))
        finally:
            handle.close()
    largest_residues, largest_path = max(residue_counts, key=lambda item: item[0])
    handle = _DomainHandle(largest_path)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        _evaluate_cell(
            model=model,
            handle=handle,
            layout=handle.layout,
            temperature=temperatures[0],
            replica=replicas[0],
            delta=delta,
            canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
            device=device,
        )
        torch.cuda.synchronize(device)
        cell_seconds = time.perf_counter() - started
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    finally:
        handle.close()
    peak_fraction = peak_bytes / total_bytes
    projected_minutes = cell_seconds * EXPECTED_DOMAINS * 25 / 60
    probe_status = _runtime_probe_status(peak_fraction, projected_minutes)
    runtime_probe = {
        "status": probe_status,
        "domain": Path(largest_path).stem.replace("mdcath_dataset_", ""),
        "residues": int(largest_residues),
        "batch_size": EXPECTED_STARTS,
        "cell_seconds": cell_seconds,
        "projected_500_cell_minutes": projected_minutes,
        "peak_memory_bytes": peak_bytes,
        "total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_fraction,
        "limits": {"max_peak_memory_fraction": 0.8, "max_projected_minutes": 50.0},
    }
    probe_output = Path(args.runtime_probe_output)
    probe_output.parent.mkdir(parents=True, exist_ok=True)
    probe_output.write_text(json.dumps(runtime_probe, indent=2) + "\n")
    if probe_status != "PASS_RUNTIME_PROBE":
        raise RuntimeError(f"runtime probe failed with {probe_status}")
    torch.cuda.empty_cache()

    domains = []
    for path in paths:
        handle = _DomainHandle(path)
        try:
            cells = [
                _evaluate_cell(
                    model=model,
                    handle=handle,
                    layout=handle.layout,
                    temperature=temperature,
                    replica=replica,
                    delta=delta,
                    canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
                    device=device,
                )
                for temperature in temperatures
                for replica in replicas
            ]
            values = [cell["mean_guarded_minus_noop"] for cell in cells]
            domains.append({
                "domain": handle.name,
                "preprocessing": {
                    "canon_symmetric": bool(data_cfg.get("canon_symmetric", False)),
                    "residues_total": int(handle.layout.num_residues),
                    "residues_evaluated": int(handle.layout.num_residues),
                },
                "summary": {
                    "cells": len(cells),
                    "mean_guarded_minus_noop": (
                        statistics.fmean(values)
                        if all(value is not None for value in values)
                        else None
                    ),
                    "cells_better_than_noop": sum(
                        value is not None and value < 0 for value in values
                    ),
                    "fallback_starts": sum(cell["fallback_starts"] for cell in cells),
                    "fallback_cells": sum(cell["fallback_starts"] > 0 for cell in cells),
                },
                "cells": cells,
            })
        finally:
            handle.close()

    result = {
        "scope": {
            "training": SCOPE,
            "fresh-external": EXTERNAL_SCOPE,
            "paper-horizon-external": PAPER_HORIZON_EXTERNAL_SCOPE,
            "paper-vector-external": PAPER_VECTOR_EXTERNAL_SCOPE,
        }[args.panel_kind],
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_profile": args.checkpoint_profile,
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_schema": int(checkpoint["checkpoint_schema"]),
        "checkpoint_train_seed": int(checkpoint["cfg"]["train"]["seed"]),
        "checkpoint_train_fingerprint": train_fingerprint,
        "delta_frames": delta,
        "settings": {
            "starts": EXPECTED_STARTS,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
            "policy": "reject_to_exact_source_per_start",
            "strict_thresholds": {
                "bond_mean_gt": BOND_MEAN_LO,
                "bond_mean_lt": BOND_MEAN_HI,
                "bond_max_lt": BOND_MAX,
            },
            "fallback_caps": {
                "max_starts": MAX_FALLBACK_STARTS,
                "max_cells": MAX_FALLBACK_CELLS,
            },
        },
        "training_subset": {
            "path": str(Path(args.training_domain_list).resolve()),
            "sha256": training_sha,
            "ids": training_ids,
            **training_identity,
        },
        "domain_panel": {
            "path": str(Path(args.domain_list).resolve()),
            "sha256": panel_sha,
            "ids": panel_ids,
            "subset_of_training1000": args.panel_kind == "training",
            "fresh_external": args.panel_kind in {
                "fresh-external", "paper-horizon-external", "paper-vector-external"
            },
            "paper_horizon_external": args.panel_kind in {
                "paper-horizon-external", "paper-vector-external"
            },
            "paper_vector_external": args.panel_kind == "paper-vector-external",
            "exclusion_union_count": (
                panel_contract["exclusion_union_count"] if panel_contract else None
            ),
            "h5_files": len(paths),
            "total_bytes": panel_total_bytes,
        },
        "prerequisite": prerequisite,
        "external_evidence": external_evidence,
        "grid": {"temperatures": temperatures, "replicas": replicas},
        "mechanism_probe": mechanism,
        "runtime_probe": runtime_probe,
        "domains": domains,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
