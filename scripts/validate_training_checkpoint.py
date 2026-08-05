#!/usr/bin/env python
"""Fail-closed validation for a bounded DDP training checkpoint and history."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from deepjump.config import Config, ModelConfig, _from_dict, to_dict
from deepjump.data_contract import _read_regular_bytes
from deepjump.model import DeepJumpLite
try:
    from scripts.train_ddp import training_semantics_sha256
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    # Direct execution places this script's directory, rather than the
    # repository root, on sys.path. Support that execution mode without
    # masking import failures from train_ddp or its dependencies.
    from train_ddp import training_semantics_sha256


_HEX = frozenset("0123456789abcdef")
_TRAIN_STATE_FIELDS = frozenset(
    {
        "world_size",
        "train_dataset_size",
        "sampler_num_samples",
        "sampler_seed",
        "sampler_epoch",
        "samples_consumed_per_rank",
        "batch_size",
        "grad_accum",
        "train_fingerprint",
        "full_training_data_contract",
        "training_semantics_sha256",
        "crop_resume",
    }
)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _nonfinite_paths(value: object, path: str) -> list[str]:
    """Return every recursively non-finite numeric leaf in a serialized state."""

    if torch.is_tensor(value):
        if (
            value.is_floating_point() or value.is_complex()
        ) and not torch.isfinite(value).all():
            return [path]
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _nonfinite_paths(child, f"{path}.{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            item
            for index, child in enumerate(value)
            for item in _nonfinite_paths(child, f"{path}[{index}]")
        ]
    return []


def _load_expected_config(path: Path) -> tuple[dict, str, str]:
    raw = _read_regular_bytes(path, "expected sealed config")
    try:
        payload = yaml.safe_load(raw.decode("utf-8")) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("expected sealed config is not valid UTF-8 YAML") from exc
    if not isinstance(payload, dict):
        raise ValueError("expected sealed config must be a mapping")
    cfg = _from_dict(Config, payload)
    return (
        to_dict(cfg),
        hashlib.sha256(raw).hexdigest(),
        training_semantics_sha256(cfg),
    )


def _load_expected_contract_verification(
    path: Path, expected_sha256: str
) -> tuple[dict, str]:
    if len(expected_sha256) != 64 or set(expected_sha256) - _HEX:
        raise ValueError("expected contract-verification SHA256 is invalid")
    raw = _read_regular_bytes(path, "expected contract verification")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("expected contract-verification SHA256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("expected contract verification is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("status") != (
        "PASS_FULL_TRAINING_DATA_CONTRACT"
    ):
        raise ValueError("expected contract verification is not a qualifying PASS")
    return payload, actual_sha256


def validate_checkpoint(
    checkpoint_path: Path,
    expected_step: int,
    expected_world_size: int,
    history_path: Path,
    history_mode: str = "final",
    expected_delta: int | None = None,
    require_vector_only: bool = False,
    require_full_tensor: bool = False,
    require_vector_scalar_value: bool = False,
    expected_lr_horizon_steps: int | None = None,
    expected_config_path: Path | None = None,
    expected_contract_verification_path: Path | None = None,
    expected_contract_verification_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    strict_restart = expected_checkpoint_sha256 is not None
    checkpoint_raw = _read_regular_bytes(checkpoint_path, "training checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_raw).hexdigest()
    if expected_checkpoint_sha256 is not None:
        if not _valid_sha256(expected_checkpoint_sha256):
            errors.append("expected checkpoint SHA256 is invalid")
        elif checkpoint_sha256 != expected_checkpoint_sha256:
            errors.append("training checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must contain a mapping")

    if checkpoint.get("step") != expected_step:
        errors.append(f"checkpoint step {checkpoint.get('step')} != {expected_step}")
    if checkpoint.get("checkpoint_schema") != 2:
        errors.append(f"checkpoint schema {checkpoint.get('checkpoint_schema')} != 2")

    train_state = checkpoint.get("train_state") or {}
    if not isinstance(train_state, dict):
        errors.append("checkpoint train_state is missing or invalid")
        train_state = {}
    missing_train_state = sorted(_TRAIN_STATE_FIELDS - set(train_state))
    if strict_restart and missing_train_state:
        errors.append(f"checkpoint train_state fields missing: {missing_train_state}")
    if train_state.get("world_size") != expected_world_size:
        errors.append(
            f"checkpoint world_size {train_state.get('world_size')} != {expected_world_size}"
        )

    config = checkpoint.get("cfg") or {}
    data_config = config.get("data") or {}
    model_config = config.get("model") or {}
    train_config = config.get("train") or {}
    if expected_delta is not None and data_config.get("delta_frames") != expected_delta:
        errors.append(
            f"checkpoint delta_frames {data_config.get('delta_frames')} != {expected_delta}"
        )
    if require_vector_only:
        if model_config.get("tensor_cloud01") is not True:
            errors.append("checkpoint is not the reviewed TensorCloud01 architecture")
        if model_config.get("tensor_cloud01_vector_only_attention") is not True:
            errors.append("checkpoint is not the reviewed vector-only attention candidate")
        if model_config.get("tensor_cloud01_vector_only_scalar_value", False) is not False:
            errors.append("checkpoint is not the pure vector-only attention candidate")
    if require_full_tensor:
        if model_config.get("tensor_cloud01") is not True:
            errors.append("checkpoint is not the reviewed TensorCloud01 architecture")
        if model_config.get("tensor_cloud01_vector_only_attention", False) is not False:
            errors.append("checkpoint is not the reviewed full-tensor attention candidate")
        if model_config.get("tensor_cloud01_vector_only_scalar_value", False) is not False:
            errors.append("full-tensor checkpoint has an invalid scalar-value variant flag")
    if require_vector_scalar_value:
        if model_config.get("tensor_cloud01") is not True:
            errors.append("checkpoint is not the reviewed TensorCloud01 architecture")
        if model_config.get("tensor_cloud01_vector_only_attention") is not True:
            errors.append("checkpoint does not use vector-only attention logits")
        if model_config.get("tensor_cloud01_vector_only_scalar_value") is not True:
            errors.append("checkpoint is not the normalized scalar-value candidate")
    if (
        expected_lr_horizon_steps is not None
        and train_config.get("lr_horizon_steps") != expected_lr_horizon_steps
    ):
        errors.append(
            "checkpoint lr_horizon_steps "
            f"{train_config.get('lr_horizon_steps')} != {expected_lr_horizon_steps}"
        )

    if strict_restart:
        for name in ("train_dataset_size", "sampler_num_samples"):
            value = train_state.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"checkpoint {name} must be a positive integer")
        for name in ("sampler_epoch", "samples_consumed_per_rank"):
            value = train_state.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"checkpoint {name} must be a non-negative integer")
        for name, expected in (
            ("sampler_seed", train_config.get("seed")),
            ("batch_size", train_config.get("batch_size")),
            ("grad_accum", train_config.get("grad_accum")),
        ):
            if train_state.get(name) != expected:
                errors.append(
                    f"checkpoint {name} {train_state.get(name)} != config value {expected}"
                )
        if not _valid_sha256(train_state.get("train_fingerprint")):
            errors.append("checkpoint train_fingerprint is invalid")
        expected_crop_resume = (
            "state_consistent_non_bitwise_crop_and_noise"
            if train_config.get("run_class") == "formal"
            else "stochastic_worker_rng_not_bitwise"
        )
        if train_state.get("crop_resume") != expected_crop_resume:
            errors.append("checkpoint crop_resume contract is invalid")

    expected_config_sha256 = None
    expected_training_semantics_sha256 = None
    if expected_config_path is not None:
        try:
            (
                expected_config,
                expected_config_sha256,
                expected_training_semantics_sha256,
            ) = _load_expected_config(expected_config_path)
        except Exception as exc:  # noqa: BLE001 - convert identity drift into gate failure
            errors.append(f"expected sealed config verification failed: {exc}")
        else:
            if config != expected_config:
                errors.append("checkpoint configuration differs from the exact sealed config")
            if train_state.get("training_semantics_sha256") != (
                expected_training_semantics_sha256
            ):
                errors.append(
                    "checkpoint training_semantics_sha256 differs from the sealed config"
                )

    expected_contract_verification_digest = None
    contract_arguments_complete = (
        expected_contract_verification_path is not None
        and expected_contract_verification_sha256 is not None
    )
    if (expected_contract_verification_path is None) != (
        expected_contract_verification_sha256 is None
    ):
        errors.append(
            "expected contract verification path and SHA256 must be provided together"
        )
    elif contract_arguments_complete:
        try:
            (
                expected_contract_verification,
                expected_contract_verification_digest,
            ) = _load_expected_contract_verification(
                expected_contract_verification_path,
                expected_contract_verification_sha256,
            )
        except Exception as exc:  # noqa: BLE001 - convert identity drift into gate failure
            errors.append(f"expected contract verification failed: {exc}")
        else:
            if train_state.get("full_training_data_contract") != (
                expected_contract_verification
            ):
                errors.append(
                    "checkpoint full_training_data_contract differs from the exact verification"
                )

    expected_model = None
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

        expected_model_state = {}
        if strict_restart:
            try:
                expected_model = DeepJumpLite(
                    _from_dict(ModelConfig, model_config),
                    noise_sigma=float(data_config.get("noise_sigma", 0.1)),
                    predict_heavy=bool(model_config.get("predict_heavy", False)),
                )
                expected_model_state = expected_model.state_dict()
            except Exception as exc:  # noqa: BLE001 - malformed architecture is a gate failure
                errors.append(f"cannot instantiate checkpoint architecture: {exc}")
        if expected_model_state:
            missing_model = sorted(set(expected_model_state) - set(model_state))
            extra_model = sorted(set(model_state) - set(expected_model_state))
            if missing_model or extra_model:
                errors.append(
                    "checkpoint model keys differ from instantiated architecture: "
                    f"missing={missing_model[:5]} extra={extra_model[:5]}"
                )
            incompatible = []
            for name in set(expected_model_state) & set(model_state):
                value = model_state[name]
                expected_value = expected_model_state[name]
                if not torch.is_tensor(value):
                    incompatible.append(f"{name}:not_tensor")
                elif value.shape != expected_value.shape:
                    incompatible.append(
                        f"{name}:shape={tuple(value.shape)} expected={tuple(expected_value.shape)}"
                    )
                elif value.dtype != expected_value.dtype:
                    incompatible.append(
                        f"{name}:dtype={value.dtype} expected={expected_value.dtype}"
                    )
            if incompatible:
                errors.append(f"checkpoint model tensors are incompatible: {incompatible[:5]}")
    optimizer_state = checkpoint.get("opt")
    if strict_restart and not isinstance(optimizer_state, dict):
        errors.append("checkpoint optimizer state is missing or invalid")
    elif strict_restart:
        if not isinstance(optimizer_state.get("state"), dict) or not optimizer_state.get(
            "state"
        ):
            errors.append("checkpoint optimizer state mapping is missing")
        groups = optimizer_state.get("param_groups")
        if not isinstance(groups, list) or not groups:
            errors.append("checkpoint optimizer parameter groups are missing")
        elif any(
            not isinstance(group, dict) or not isinstance(group.get("params"), list)
            for group in groups
        ):
            errors.append("checkpoint optimizer parameter groups are invalid")
        else:
            optimizer_parameters = [
                parameter for group in groups for parameter in group["params"]
            ]
            expected_parameter_count = (
                sum(parameter.requires_grad for parameter in expected_model.parameters())
                if expected_model is not None
                else 0
            )
            if (
                not optimizer_parameters
                or len(optimizer_parameters) != len(set(optimizer_parameters))
                or len(optimizer_parameters) != expected_parameter_count
                or set(optimizer_parameters) != set(range(expected_parameter_count))
            ):
                errors.append(
                    "checkpoint optimizer parameter mapping does not match the architecture"
                )
            elif expected_model is not None:
                named_parameters = list(expected_model.named_parameters())
                final_transport = int(model_config.get("transport_layers", 0)) - 1
                allowed_unused = {
                    f"transport.blocks.{final_transport}.feedforward.scalar_out.weight",
                    f"transport.blocks.{final_transport}.feedforward.scalar_out.bias",
                }
                expected_state_ids = {
                    index
                    for index, (name, parameter) in enumerate(named_parameters)
                    if parameter.requires_grad and name not in allowed_unused
                }
                actual_state = optimizer_state.get("state")
                actual_state_ids = set(actual_state) if isinstance(actual_state, dict) else set()
                if actual_state_ids != expected_state_ids:
                    errors.append(
                        "checkpoint optimizer state does not cover every expected-used parameter"
                    )
                else:
                    invalid_adam_state = []
                    for index in sorted(expected_state_ids):
                        parameter_state = actual_state[index]
                        parameter = named_parameters[index][1]
                        if not isinstance(parameter_state, dict) or not {
                            "step", "exp_avg", "exp_avg_sq"
                        }.issubset(parameter_state):
                            invalid_adam_state.append(f"{index}:fields")
                            continue
                        step = parameter_state["step"]
                        if (
                            not torch.is_tensor(step)
                            or step.numel() != 1
                            or not step.is_floating_point()
                            or not torch.isfinite(step).all()
                            or step.item() < 0
                        ):
                            invalid_adam_state.append(f"{index}:step")
                        for field in ("exp_avg", "exp_avg_sq"):
                            value = parameter_state[field]
                            if (
                                not torch.is_tensor(value)
                                or not value.is_floating_point()
                                or value.shape != parameter.shape
                                or value.dtype != parameter.dtype
                            ):
                                invalid_adam_state.append(f"{index}:{field}")
                    if invalid_adam_state:
                        errors.append(
                            "checkpoint Adam state is incomplete or shape-incompatible: "
                            f"{invalid_adam_state[:5]}"
                        )
        optimizer_nonfinite = _nonfinite_paths(optimizer_state, "opt")
        if optimizer_nonfinite:
            errors.append(f"non-finite optimizer state: {optimizer_nonfinite[:5]}")

    scaler_state = checkpoint.get("scaler")
    if strict_restart and not isinstance(scaler_state, dict):
        errors.append("checkpoint scaler state is missing or invalid")
    elif strict_restart:
        scaler_nonfinite = _nonfinite_paths(scaler_state, "scaler")
        if scaler_nonfinite:
            errors.append(f"non-finite scaler state: {scaler_nonfinite[:5]}")

    history_raw = _read_regular_bytes(history_path, "training history")
    try:
        history = json.loads(history_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training history is not valid UTF-8 JSON") from exc
    if not isinstance(history, list):
        errors.append("history must be a JSON list")
        selected_history = {}
        history_steps = []
    elif not history and history_mode != "through":
        errors.append("history is missing validation records")
        selected_history = {}
        history_steps = []
    else:
        history_steps: list[int] = []
        noop_values: list[float] = []
        val_every = train_config.get("val_every")
        max_steps = train_config.get("max_steps")
        for index, entry in enumerate(history):
            if not isinstance(entry, dict):
                errors.append(f"history record {index} is not an object")
                continue
            step = entry.get("step")
            if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
                errors.append(f"history record {index} step is invalid: {step!r}")
            else:
                history_steps.append(step)
                if strict_restart and not (
                    isinstance(val_every, int)
                    and val_every > 0
                    and (step % val_every == 0 or step == max_steps)
                ):
                    errors.append(f"history step {step} violates validation cadence")
            for name in ("val_loss", "val_rmsd", "noop_rmsd"):
                value = entry.get(name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    errors.append(
                        f"history record {index} {name} is not finite: {value!r}"
                    )
                elif name == "noop_rmsd":
                    noop_values.append(float(value))
        if len(history_steps) != len(history) or any(
            current <= previous for previous, current in zip(history_steps, history_steps[1:])
        ):
            errors.append("history steps must be unique and strictly increasing")
        if strict_restart and isinstance(val_every, int) and val_every > 0:
            if (
                history_mode == "final"
                and isinstance(max_steps, int)
                and max_steps > 0
                and expected_step == max_steps
            ):
                expected_history_steps = list(
                    range(val_every, max_steps + 1, val_every)
                )
                if (
                    not expected_history_steps
                    or expected_history_steps[-1] != max_steps
                ):
                    expected_history_steps.append(max_steps)
                if history_steps != expected_history_steps:
                    errors.append(
                        "history does not contain the exact validation cadence "
                        "through final step"
                    )
            elif history_mode == "through":
                expected_history_steps = list(
                    range(val_every, expected_step + 1, val_every)
                )
                if history_steps != expected_history_steps:
                    errors.append(
                        "history does not contain the exact completed validation "
                        "cadence through checkpoint step"
                    )
        if noop_values and any(value != noop_values[0] for value in noop_values[1:]):
            errors.append("history noop_rmsd must remain constant")
        if history_mode == "final":
            selected_history = (
                history[-1] if history and isinstance(history[-1], dict) else {}
            )
        elif history_mode == "contains":
            matches = [
                entry
                for entry in history
                if isinstance(entry, dict) and entry.get("step") == expected_step
            ]
            selected_history = matches[0] if len(matches) == 1 else {}
            if len(matches) != 1:
                errors.append(
                    f"history contains {len(matches)} records for step {expected_step}, expected 1"
                )
        elif history_mode == "through":
            selected_history = (
                history[-1] if history and isinstance(history[-1], dict) else {}
            )
        else:
            raise ValueError(f"unsupported history mode: {history_mode}")
        if (
            history_mode != "through"
            and selected_history.get("step") != expected_step
        ):
            errors.append(f"history step {selected_history.get('step')} != {expected_step}")

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_schema": checkpoint.get("checkpoint_schema"),
        "world_size": train_state.get("world_size"),
        "delta_frames": data_config.get("delta_frames"),
        "vector_only_attention": bool(
            model_config.get("tensor_cloud01_vector_only_attention", False)
        ),
        "vector_only_scalar_value": bool(
            model_config.get("tensor_cloud01_vector_only_scalar_value", False)
        ),
        "lr_horizon_steps": train_config.get("lr_horizon_steps"),
        "expected_config_sha256": expected_config_sha256,
        "training_semantics_sha256": train_state.get("training_semantics_sha256"),
        "expected_training_semantics_sha256": expected_training_semantics_sha256,
        "contract_verification_sha256": expected_contract_verification_digest,
        "model_tensors": len(model_state) if isinstance(model_state, dict) else 0,
        "nonfinite_model_tensors": nonfinite_parameters,
        "optimizer_present": isinstance(optimizer_state, dict),
        "scaler_present": isinstance(scaler_state, dict),
        "train_state_fields": sorted(train_state),
        "history": selected_history,
        "history_records": len(history) if isinstance(history, list) else 0,
        "history_steps": history_steps if isinstance(history, list) and history else [],
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
    parser.add_argument(
        "--history-mode",
        choices=("final", "contains", "through"),
        default="final",
    )
    parser.add_argument("--expected-delta", type=int)
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument("--require-vector-only", action="store_true")
    architecture.add_argument("--require-full-tensor", action="store_true")
    architecture.add_argument("--require-vector-scalar-value", action="store_true")
    parser.add_argument("--expected-lr-horizon-steps", type=int)
    parser.add_argument("--expected-config", type=Path)
    parser.add_argument("--expected-contract-verification", type=Path)
    parser.add_argument("--expected-contract-verification-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
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
            args.require_vector_scalar_value,
            args.expected_lr_horizon_steps,
            args.expected_config,
            args.expected_contract_verification,
            args.expected_contract_verification_sha256,
            args.expected_checkpoint_sha256,
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
