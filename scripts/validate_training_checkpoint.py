#!/usr/bin/env python
"""Fail-closed validation for a bounded DDP training checkpoint and history."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def validate_checkpoint(
    checkpoint_path: Path,
    expected_step: int,
    expected_world_size: int,
    history_path: Path,
    history_mode: str = "final",
    expected_delta: int | None = None,
    require_vector_only: bool = False,
    require_full_tensor: bool = False,
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if checkpoint.get("step") != expected_step:
        errors.append(f"checkpoint step {checkpoint.get('step')} != {expected_step}")
    if checkpoint.get("checkpoint_schema") != 2:
        errors.append(f"checkpoint schema {checkpoint.get('checkpoint_schema')} != 2")

    train_state = checkpoint.get("train_state") or {}
    if train_state.get("world_size") != expected_world_size:
        errors.append(
            f"checkpoint world_size {train_state.get('world_size')} != {expected_world_size}"
        )

    config = checkpoint.get("cfg") or {}
    data_config = config.get("data") or {}
    model_config = config.get("model") or {}
    if expected_delta is not None and data_config.get("delta_frames") != expected_delta:
        errors.append(
            f"checkpoint delta_frames {data_config.get('delta_frames')} != {expected_delta}"
        )
    if require_vector_only:
        if model_config.get("tensor_cloud01") is not True:
            errors.append("checkpoint is not the reviewed TensorCloud01 architecture")
        if model_config.get("tensor_cloud01_vector_only_attention") is not True:
            errors.append("checkpoint is not the reviewed vector-only attention candidate")
    if require_full_tensor:
        if model_config.get("tensor_cloud01") is not True:
            errors.append("checkpoint is not the reviewed TensorCloud01 architecture")
        if model_config.get("tensor_cloud01_vector_only_attention", False) is not False:
            errors.append("checkpoint is not the reviewed full-tensor attention candidate")

    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        errors.append("checkpoint model state is missing or empty")
        nonfinite_parameters = []
    else:
        nonfinite_parameters = [
            name
            for name, value in model_state.items()
            if torch.is_tensor(value) and not torch.isfinite(value).all()
        ]
        if nonfinite_parameters:
            errors.append(f"non-finite model tensors: {nonfinite_parameters[:5]}")

    history = json.loads(history_path.read_text())
    if not isinstance(history, list) or not history:
        errors.append("history is missing validation records")
        selected_history = {}
    else:
        if history_mode == "final":
            selected_history = history[-1]
        elif history_mode == "contains":
            matches = [entry for entry in history if entry.get("step") == expected_step]
            selected_history = matches[0] if len(matches) == 1 else {}
            if len(matches) != 1:
                errors.append(
                    f"history contains {len(matches)} records for step {expected_step}, expected 1"
                )
        else:
            raise ValueError(f"unsupported history mode: {history_mode}")
        if selected_history.get("step") != expected_step:
            errors.append(f"history step {selected_history.get('step')} != {expected_step}")
        for name in ("val_loss", "val_rmsd", "noop_rmsd"):
            value = selected_history.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"history {name} is not finite: {value!r}")

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_schema": checkpoint.get("checkpoint_schema"),
        "world_size": train_state.get("world_size"),
        "delta_frames": data_config.get("delta_frames"),
        "vector_only_attention": bool(
            model_config.get("tensor_cloud01_vector_only_attention", False)
        ),
        "model_tensors": len(model_state) if isinstance(model_state, dict) else 0,
        "nonfinite_model_tensors": nonfinite_parameters,
        "history": selected_history,
        "history_mode": history_mode,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--expected-world-size", required=True, type=int)
    parser.add_argument("--history-mode", choices=("final", "contains"), default="final")
    parser.add_argument("--expected-delta", type=int)
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument("--require-vector-only", action="store_true")
    architecture.add_argument("--require-full-tensor", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report, errors = validate_checkpoint(
            args.checkpoint,
            args.expected_step,
            args.expected_world_size,
            args.history,
            args.history_mode,
            args.expected_delta,
            args.require_vector_only,
            args.require_full_tensor,
        )
    except Exception as exc:  # noqa: BLE001 - convert corrupt artifacts into a gate failure
        report = {
            "checkpoint": str(args.checkpoint),
            "status": "FAIL",
            "errors": [f"checkpoint readback failed: {exc}"],
        }
        errors = report["errors"]

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
