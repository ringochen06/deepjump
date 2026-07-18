#!/usr/bin/env python
"""Adjudicate whether a five-start H6 no-op margin exceeds its noise floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


EXPECTED_STARTS = 5
EXPECTED_STEPS = 6


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_start_values(domain: dict, method: str) -> list[float]:
    try:
        values = domain["methods"][method]["rmsd_by_start"]
    except KeyError as exc:
        raise ValueError(f"missing {method}.rmsd_by_start") from exc
    if not isinstance(values, list) or len(values) != EXPECTED_STEPS + 1:
        raise ValueError(f"{method}.rmsd_by_start must contain H0..H{EXPECTED_STEPS}")
    final = [float(value) for value in values[-1]]
    if len(final) != EXPECTED_STARTS or not all(math.isfinite(value) for value in final):
        raise ValueError(f"{method} H6 must contain five finite per-start values")
    return final


def adjudicate(
    result_path: str | Path,
    domain_list_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> dict:
    result = json.loads(Path(result_path).read_text())
    if int(result.get("checkpoint_step", -1)) != 250:
        raise ValueError("start-noise audit requires checkpoint step 250")
    if Path(result.get("checkpoint", "")).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("checkpoint path mismatch")
    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    if int(result.get("delta_frames", -1)) != 1:
        raise ValueError("start-noise audit requires delta=1")
    panel = result.get("domain_panel", {})
    if panel.get("sha256") != domain_list_sha256 or int(panel.get("evaluated_count", -1)) != 1:
        raise ValueError("domain panel identity mismatch")
    settings = result.get("settings", {})
    expected = {
        "domains": 1,
        "starts": EXPECTED_STARTS,
        "steps": EXPECTED_STEPS,
        "methods": "mean",
        "teacher_forced_mean": True,
        "seed": 20260718,
        "integrator": "euler",
        "tau_max": 1.0,
        "drift_anchor": "state",
    }
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"start-noise settings mismatch: {mismatches}")
    if result.get("preprocessing", {}).get("canon_symmetric") is not True:
        raise ValueError("start-noise audit requires canonical symmetric atom slots")
    domains = result.get("domains", [])
    if len(domains) != 1:
        raise ValueError("start-noise audit requires one domain row")

    model = _final_start_values(domains[0], "mean")
    noop = _final_start_values(domains[0], "noop")
    deltas = [model_value - noop_value for model_value, noop_value in zip(model, noop)]
    mean_delta = statistics.fmean(deltas)
    standard_error = statistics.stdev(deltas) / math.sqrt(EXPECTED_STARTS)
    magnitude = abs(mean_delta)
    if mean_delta < 0 and (standard_error == 0 or magnitude >= 2 * standard_error):
        status = "ROBUST_ADVANTAGE"
    elif mean_delta > 0 and (standard_error == 0 or magnitude >= 2 * standard_error):
        status = "ROBUST_DISADVANTAGE"
    elif (standard_error == 0 and mean_delta == 0) or magnitude < standard_error:
        status = "NOISE_DOMINATED"
    else:
        status = "INCONCLUSIVE_START_NOISE"
    return {
        "status": status,
        "scope": "single-domain five-start H6 paired noise audit only",
        "model_minus_noop_by_start": deltas,
        "mean_model_minus_noop": mean_delta,
        "standard_error": standard_error,
        "absolute_mean_over_standard_error": (
            magnitude / standard_error if standard_error > 0 else None
        ),
        "decision_rule": {
            "ROBUST_ADVANTAGE": "mean delta < 0 and |mean delta| >= 2 SE",
            "ROBUST_DISADVANTAGE": "mean delta > 0 and |mean delta| >= 2 SE",
            "NOISE_DOMINATED": "|mean delta| < 1 SE, including exact equality with zero variance",
            "INCONCLUSIVE": "all remaining outcomes",
        },
        "second_seed_authorized": status == "ROBUST_ADVANTAGE",
        "formal_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.domain_list_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
