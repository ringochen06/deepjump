#!/usr/bin/env python
"""Deterministically select TensorCloud01 checkpoints for scientific triage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_STEPS = [250, 500, 750, 1000]


def select_checkpoints(
    history: Any,
    config: Any,
    *,
    expected_delta: int,
    require_vector_only: bool = False,
    count: int = 2,
) -> dict[str, Any]:
    """Rank only by frozen validation loss; RMSD remains a diagnostic.

    Scientific results are deliberately absent from this selection stage. This
    prevents choosing a checkpoint after looking at no-op comparison outcomes.
    """
    if not isinstance(history, list):
        raise ValueError("history must be a JSON list")
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    if count < 1 or count > len(EXPECTED_STEPS):
        raise ValueError("selection count must be between 1 and 4")

    data = config.get("data", {})
    model = config.get("model", {})
    train = config.get("train", {})
    if data.get("delta_frames") != expected_delta:
        raise ValueError("config delta_frames does not match the requested calibration")
    if model.get("tensor_cloud01") is not True:
        raise ValueError("config is not the reviewed TensorCloud01 architecture")
    if require_vector_only and model.get("tensor_cloud01_vector_only_attention") is not True:
        raise ValueError("config is not the reviewed vector-only attention candidate")
    if train.get("max_steps") != 1000:
        raise ValueError("config is not the reviewed 1000-step calibration")

    by_step: dict[int, dict[str, Any]] = {}
    for row in history:
        if not isinstance(row, dict) or row.get("step") not in EXPECTED_STEPS:
            raise ValueError("history contains an unexpected validation record")
        step = int(row["step"])
        if step in by_step:
            raise ValueError(f"history contains duplicate step {step}")
        for name in ("val_loss", "val_rmsd", "noop_rmsd"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"history {name} at step {step} is not numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"history {name} at step {step} is not finite")
        by_step[step] = row
    if sorted(by_step) != EXPECTED_STEPS:
        raise ValueError("history must contain exactly steps 250, 500, 750, and 1000")

    ranked = sorted(
        by_step.values(),
        key=lambda row: (float(row["val_loss"]), int(row["step"])),
    )
    return {
        "status": "PASS",
        "scope": "checkpoint_selection_only",
        "delta_frames": expected_delta,
        "vector_only_attention": bool(
            model.get("tensor_cloud01_vector_only_attention", False)
        ),
        "selection_rule": "lowest_frozen_validation_loss_then_earlier_step",
        "scientific_metrics_used_for_selection": False,
        "candidate_count": count,
        "selected": [
            {
                "step": int(row["step"]),
                "checkpoint": f"ckpt_{int(row['step'])}.pt",
                "val_loss": float(row["val_loss"]),
                "val_rmsd": float(row["val_rmsd"]),
                "noop_rmsd": float(row["noop_rmsd"]),
            }
            for row in ranked[:count]
        ],
        "ranked_steps": [int(row["step"]) for row in ranked],
    }


def _load(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-delta", required=True, type=int)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--require-vector-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = select_checkpoints(
        _load(args.history),
        _load(args.config),
        expected_delta=args.expected_delta,
        require_vector_only=args.require_vector_only,
        count=args.count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
