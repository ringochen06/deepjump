#!/usr/bin/env python
"""Adjudicate the bounded first-party source-law H6 then H20 discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


EXPECTED_CHECKPOINT_STEP = 1000
EXPECTED_STARTS = 5
EXPECTED_METHOD = "ode_150"
BOND_MEAN_RANGE = (3.2, 4.5)
MAX_BOND_MAX = 5.5


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_result(
    path: str | Path,
    *,
    steps: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> tuple[dict, dict]:
    result = json.loads(Path(path).read_text())
    if int(result.get("checkpoint_step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("source-law candidate requires checkpoint step 1000")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("source-law candidate requires delta=1")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256 or int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("domain panel identity mismatch")
    settings = result.get("settings", {})
    expected = {
        "domains": 1,
        "starts": EXPECTED_STARTS,
        "steps": steps,
        "methods": EXPECTED_METHOD,
        "seed": 20260718,
        "noise_sigma": None,
        "integrator": "euler",
        "tau_max": 1.0,
        "terminal_denoise": False,
        "drift_anchor": "state",
        "teacher_forced_mean": False,
    }
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"source-law settings mismatch: {mismatches}")
    if result.get("preprocessing", {}).get("canon_symmetric") is not True:
        raise ValueError("source-law candidate requires canonical symmetric atom slots")
    domains = result.get("domains", [])
    if len(domains) != 1:
        raise ValueError("source-law candidate requires one domain row")
    methods = domains[0].get("methods", {})
    if set(methods) != {"noop", "one_step_persistence", EXPECTED_METHOD}:
        raise ValueError("unexpected method set")
    return result, methods


def _verify_checkpoint_source_law(path: str | Path) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != EXPECTED_CHECKPOINT_STEP:
        raise ValueError("source-law checkpoint must be at step 1000")
    data = checkpoint.get("cfg", {}).get("data", {})
    model = checkpoint.get("cfg", {}).get("model", {})
    expected = {
        "data.noise_sigma": (data.get("noise_sigma"), 1.5),
        "data.unroll": (data.get("unroll"), 1),
        "data.canon_symmetric": (data.get("canon_symmetric"), True),
        "model.source_noise_v": (model.get("source_noise_v"), True),
        "model.source_noise_sigma_v": (model.get("source_noise_sigma_v"), 1.0),
        "model.tensor_cloud01": (model.get("tensor_cloud01"), True),
        "model.tensor_cloud01_vector_only_attention": (
            model.get("tensor_cloud01_vector_only_attention"), False
        ),
    }
    mismatches = {
        key: values for key, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"checkpoint source-law identity mismatch: {mismatches}")


def _metrics(methods: dict, steps: int) -> dict:
    model = methods[EXPECTED_METHOD]
    noop = methods["noop"]
    for name in ("rmsd_by_start", "bond_mean", "bond_max"):
        for method_name, row in ((EXPECTED_METHOD, model), ("noop", noop)):
            values = row.get(name)
            if not isinstance(values, list) or len(values) != steps + 1:
                raise ValueError(f"{method_name}.{name} must contain H0..H{steps}")
            if name == "rmsd_by_start" and any(
                not isinstance(horizon, list) or len(horizon) != EXPECTED_STARTS
                for horizon in values
            ):
                raise ValueError(
                    f"{method_name}.rmsd_by_start must contain five starts at every horizon"
                )
            flat_values = (
                [value for horizon in values for value in horizon]
                if name == "rmsd_by_start"
                else values
            )
            if not all(math.isfinite(float(value)) for value in flat_values):
                raise ValueError(f"{method_name}.{name} must be finite")
    model_final = [float(value) for value in model["rmsd_by_start"][-1]]
    noop_final = [float(value) for value in noop["rmsd_by_start"][-1]]
    if len(model_final) != EXPECTED_STARTS or len(noop_final) != EXPECTED_STARTS:
        raise ValueError("final RMSD must contain five starts")
    deltas = [a - b for a, b in zip(model_final, noop_final)]
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(EXPECTED_STARTS)
    physical = all(
        BOND_MEAN_RANGE[0] <= float(mean) <= BOND_MEAN_RANGE[1]
        and float(maximum) <= MAX_BOND_MAX
        for mean, maximum in zip(model["bond_mean"][1:], model["bond_max"][1:])
    )
    # A zero-width empirical interval has no standing as a five-start robustness claim.
    robust_advantage = (
        mean_delta < 0
        and standard_error > 0
        and abs(mean_delta) >= 2 * standard_error
    )
    return {
        "steps": steps,
        "model_minus_noop_by_start": deltas,
        "mean_model_minus_noop": mean_delta,
        "standard_error": standard_error,
        "absolute_mean_over_standard_error": (
            abs(mean_delta) / standard_error if standard_error > 0 else None
        ),
        "physical_through_horizon": physical,
        "robust_advantage": robust_advantage,
    }


def adjudicate(
    h6_path: str | Path,
    h20_path: str | Path | None,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> dict:
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    _verify_checkpoint_source_law(checkpoint_path)
    _, h6_methods = _load_result(
        h6_path,
        steps=6,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        domain_list_sha256=domain_list_sha256,
    )
    h6 = _metrics(h6_methods, 6)
    if not (h6["physical_through_horizon"] and h6["robust_advantage"]):
        status = "STOP_SOURCE_LAW_H6"
        h20 = None
    elif h20_path is None:
        status = "ADVANCE_SOURCE_LAW_H20"
        h20 = None
    else:
        _, h20_methods = _load_result(
            h20_path,
            steps=20,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            domain_list_sha256=domain_list_sha256,
        )
        h20 = _metrics(h20_methods, 20)
        status = (
            "PASS_SOURCE_LAW_H20"
            if h20["physical_through_horizon"] and h20["robust_advantage"]
            else "STOP_SOURCE_LAW_H20"
        )
    return {
        "status": status,
        "scope": "single-domain 1000-step first-party source-law discriminator only",
        "checkpoint_sha256": checkpoint_sha256,
        "source_law": {"coordinate_sigma": 1.5, "vector_sigma": 1.0},
        "sampling": {"method": EXPECTED_METHOD, "seed": 20260718},
        "h6": h6,
        "h20": h20,
        "decision_rule": {
            "ADVANCE_H20": "H1-H6 all physical and H6 mean(model-noop)<0 with |mean|>=2SE",
            "PASS_H20": "H1-H20 all physical and H20 mean(model-noop)<0 with |mean|>=2SE",
            "STOP": "any nonphysical horizon or failure of the paired robust-advantage rule",
        },
        "twenty_domain_authorized": status == "PASS_SOURCE_LAW_H20",
        "second_seed_authorized": False,
        "confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h6", required=True)
    parser.add_argument("--h20")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.h6,
        args.h20,
        args.checkpoint,
        args.checkpoint_sha256,
        args.domain_list_sha256,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
