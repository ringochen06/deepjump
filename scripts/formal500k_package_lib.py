"""Fail-closed construction of one exact closer-to-paper formal500k package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z")
CLASSIFICATION = "closer_to_paper_not_exact_reproduction"
RECOVERY_SEMANTICS = "state_consistent_non_bitwise_crop_and_noise"
PRICE_QUOTE_KEYS = {
    "schema",
    "provider",
    "region",
    "instance_id",
    "flavor",
    "billing_mode",
    "currency",
    "hourly_rate",
    "observed_at",
    "source",
    "reference_url",
    "reference_hourly_rate",
    "reference_quantity",
    "historical_console_hourly_rate",
    "derivation",
}
PRICE_QUOTE_MAX_AGE_SECONDS = 24 * 60 * 60
GPU_INSTANCE_ID = "4c2273f2-4763-4827-839b-27d2c79cd76a"
GPU_FLAVOR = "p2v.16xlarge.8"
GPU_REGION = "cn-north-4"
GPU_HOSTNAME = "deepjump-v100-8gpu-20260716"
GPU_MODEL = "Tesla V100-SXM2-16GB"
PRICE_REFERENCE_URL = "https://support.huaweicloud.com/mineru-ctf/ctf-mineru.pdf"
PRICE_REFERENCE_HOURLY_RATE = 16.51
PRICE_REFERENCE_QUANTITY = 8
HISTORICAL_CONSOLE_HOURLY_RATE = 133.58356
PRICE_BOUND_SOURCE = (
    "official_huawei_cloud_reference_with_historical_console_upper_bound"
)
PRICE_BOUND_DERIVATION = (
    "max(historical_exact_8gpu_console_rate, "
    "official_single_v100_reference_rate * 8)"
)

SPEC_KEYS = {
    "schema",
    "package_id",
    "created_at",
    "classification",
    "formal_candidate",
    "data_identity",
    "source_identity",
    "runtime_identity",
    "checkpoint_plan",
    "stop_plan",
    "recovery_plan",
    "obs_plan",
    "scientific_policy",
    "estimate_budget",
    "known_deviations",
    "prerequisites",
}

EXECUTION_KEYS = {
    "reviewed_commit",
    "run_id",
    "obs_dst",
    "data_uuid",
    "world_size",
    "repo_root",
    "run_dir",
    "config_path",
    "config_sha256",
    "contract_verification_path",
    "contract_verification_sha256",
    "full_training_contract_path",
    "full_training_contract_sha256",
    "supervisor_path",
    "supervisor_sha256",
    "archiver_path",
    "archiver_sha256",
    "validator_path",
    "validator_sha256",
    "empty_prefix_validator_path",
    "empty_prefix_validator_sha256",
    "trainer_path",
    "trainer_sha256",
    "soft_stop_minutes",
    "hard_stop_minutes",
    "archive_kill_grace_seconds",
    "archive_poll_seconds",
    "toolchain",
}

CHECKPOINT_PLAN = {
    "ckpt_every": 1000,
    "trainer_keep_last_k": 501,
    "archiver_keep_local_verified": 3,
    "immutable_numbered": True,
    "local_strict_validator": True,
    "forced_obs_readback": True,
    "remote_strict_validator": True,
    "verified_remote_required_for_retention": True,
    "latest_verified_required": True,
}

RECOVERY_PLAN = {
    "separate_attempt_required": True,
    "max_recovery_attempts": 16,
    "resume_history_required": True,
    "strict_checkpoint_preflight": True,
    "latest_verified_only": True,
    "resume_semantics": RECOVERY_SEMANTICS,
}


def _exact_keys(payload: object, expected: set[str], label: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return payload


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    return parsed


def read_regular(path_value: str | Path, label: str) -> tuple[Path, bytes]:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        raw = handle.read()
        after = os.fstat(handle.fileno())
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ValueError(f"{label} changed during its verified read")
    return path.resolve(), raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _verify_bound_file(path: object, digest: object, label: str) -> Path:
    resolved, raw = read_regular(str(path), label)
    if sha256_bytes(raw) != _sha(digest, f"{label} SHA256"):
        raise ValueError(f"{label} SHA256 mismatch")
    return resolved


def _verify_toolchain(toolchain: object) -> None:
    tools = _exact_keys(toolchain, {"python", "torchrun", "obsutil"}, "toolchain")
    for name, entry_value in tools.items():
        entry = _exact_keys(
            entry_value, {"path", "sha256", "version_args", "version"}, f"toolchain.{name}"
        )
        path = _verify_bound_file(entry["path"], entry["sha256"], f"toolchain.{name}")
        if str(path) != entry["path"]:
            raise ValueError(f"toolchain.{name} path must be canonical")
        args = entry["version_args"]
        if (
            not isinstance(args, list)
            or not args
            or any(not isinstance(value, str) for value in args)
        ):
            raise ValueError(f"toolchain.{name}.version_args is invalid")
        completed = subprocess.run(
            [str(path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0 or completed.stdout.strip() != entry["version"]:
            raise ValueError(f"toolchain.{name} version output mismatch")


def _verify_config(config_raw: bytes, contract_sha: str) -> dict:
    config = yaml.safe_load(config_raw)
    _exact_keys(config, {"data", "model", "train"}, "formal config")
    data = config["data"]
    model = config["model"]
    train = config["train"]
    required_train = {
        "run_class": "formal",
        "batch_size": 2,
        "grad_accum": 8,
        "lr": 5.0e-3,
        "lr_final": 3.0e-3,
        "warmup_steps": 200,
        "lr_horizon_steps": 500000,
        "grad_clip": 0.1,
        "max_steps": 500000,
        "val_every": 10000,
        "ckpt_every": 1000,
        "keep_last_k": 501,
        "amp": False,
        "num_workers": 8,
        "seed": 0,
    }
    for field, expected in required_train.items():
        if train.get(field) != expected:
            raise ValueError(f"formal config train.{field} differs from {expected!r}")
    required_data = {
        "crop_length": 256,
        "val_fraction": 0.02,
        "delta_frames": 1,
        "seed": 0,
    }
    for field, expected in required_data.items():
        if data.get(field) != expected:
            raise ValueError(f"formal config data.{field} differs from {expected!r}")
    required_model = {
        "hidden": 128,
        "vector_channels": 128,
        "cond_layers": 6,
        "transport_layers": 6,
        "tensor_cloud01": True,
        "tensor_cloud01_vector_only_attention": False,
        "tensor_cloud01_vector_only_scalar_value": False,
    }
    for field, expected in required_model.items():
        if model.get(field) != expected:
            raise ValueError(f"formal config model.{field} differs from {expected!r}")
    if train["batch_size"] * train["grad_accum"] * 8 != 128:
        raise ValueError("formal config effective batch must be 128")
    if train.get("resume"):
        raise ValueError("formal config must be fresh-init without train.resume")
    if data.get("domains"):
        raise ValueError("formal config must use the contracted domains_file, not inline domains")
    for field in ("root", "manifest", "domains_file", "full_training_contract"):
        value = data.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"formal config data.{field} must be absolute")
    if data.get("full_training_contract_sha256") != contract_sha:
        raise ValueError("formal config full-training contract SHA256 mismatch")
    return config


def _verify_config_contract_paths(
    config: dict, contract_path: Path, contract_raw: bytes
) -> None:
    try:
        contract = json.loads(contract_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("full training contract is not valid JSON") from exc
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("full training contract artifacts are missing")

    resolved: dict[str, Path] = {}
    for label in ("data_audit", "manifest", "train_list"):
        entry = artifacts.get(label)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
        ):
            raise ValueError(f"full training contract {label} identity is invalid")
        artifact_path = (contract_path.parent / entry["path"]).resolve()
        resolved[label] = _verify_bound_file(
            artifact_path, entry["sha256"], f"contract {label}"
        )

    try:
        audit = json.loads(resolved["data_audit"].read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("contract data audit is not valid JSON") from exc
    configured_root = Path(config["data"]["root"]).expanduser()
    if configured_root.is_symlink():
        raise ValueError("formal config data root must not be a symlink")
    if configured_root.resolve() != Path(audit.get("root", "")).resolve():
        raise ValueError("formal config data root differs from the qualified audit root")
    if Path(config["data"]["manifest"]).expanduser().resolve() != resolved["manifest"]:
        raise ValueError("formal config manifest differs from the contracted manifest")
    if (
        Path(config["data"]["domains_file"]).expanduser().resolve()
        != resolved["train_list"]
    ):
        raise ValueError("formal config domains file differs from the contracted train list")


def build_package_payload(spec_path: Path) -> dict:
    _, spec_raw = read_regular(spec_path.resolve(), "formal500k package spec")
    try:
        spec = json.loads(spec_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal500k package spec is not valid JSON") from exc
    _exact_keys(spec, SPEC_KEYS, "formal500k package spec")
    if spec["schema"] != "deepjump.formal500k.package_spec.v1":
        raise ValueError("formal500k package spec schema mismatch")
    if spec["classification"] != CLASSIFICATION:
        raise ValueError("formal500k classification mismatch")
    if (
        not isinstance(spec["package_id"], str)
        or not spec["package_id"]
        or not isinstance(spec["created_at"], str)
        or not spec["created_at"].endswith("Z")
    ):
        raise ValueError("package identity or timestamp is invalid")

    candidate = _exact_keys(
        spec["formal_candidate"],
        {
            "fresh_init",
            "initial_checkpoint",
            "training_seed",
            "data_seed",
            "target_total_steps",
            "unique_scientific_endpoint_steps",
            "intermediate_checkpoint_policy",
            "config_sha256",
            "execution_plan",
        },
        "formal_candidate",
    )
    if (
        candidate["fresh_init"] is not True
        or candidate["initial_checkpoint"] is not None
        or candidate["training_seed"] != 0
        or candidate["data_seed"] != 0
        or candidate["target_total_steps"] != 500000
        or candidate["unique_scientific_endpoint_steps"] != [500000]
        or candidate["intermediate_checkpoint_policy"]
        != "engineering_recovery_and_finite_monitoring_only_no_selection"
    ):
        raise ValueError("formal candidate is not the frozen fresh-init 500k endpoint")

    plan = _exact_keys(candidate["execution_plan"], EXECUTION_KEYS, "execution_plan")
    if (
        HEX40.fullmatch(str(plan["reviewed_commit"])) is None
        or RUN_ID.fullmatch(str(plan["run_id"])) is None
        or plan["world_size"] != 8
        or not str(plan["obs_dst"]).startswith("obs://")
        or not str(plan["obs_dst"]).rstrip("/").endswith("/" + str(plan["run_id"]))
    ):
        raise ValueError("execution identity is invalid")
    if plan["hard_stop_minutes"] - plan["soft_stop_minutes"] < 30:
        raise ValueError("hard stop must leave at least 30 minutes after soft stop")
    repo_root = Path(plan["repo_root"])
    if not repo_root.is_absolute() or repo_root.resolve() != repo_root:
        raise ValueError("execution repo_root must be canonical and absolute")
    if not Path(plan["run_dir"]).is_absolute():
        raise ValueError("execution run_dir must be absolute")
    for field in (
        "config",
        "contract_verification",
        "full_training_contract",
        "supervisor",
        "archiver",
        "validator",
        "empty_prefix_validator",
        "trainer",
    ):
        resolved = _verify_bound_file(
            plan[f"{field}_path"], plan[f"{field}_sha256"], field
        )
        if str(resolved) != plan[f"{field}_path"]:
            raise ValueError(f"execution {field}_path must be canonical")
    _verify_toolchain(plan["toolchain"])

    config_path, config_raw = read_regular(plan["config_path"], "formal config")
    if sha256_bytes(config_raw) != candidate["config_sha256"]:
        raise ValueError("formal candidate config SHA256 mismatch")
    contract_path, contract_raw = read_regular(
        plan["full_training_contract_path"], "full training contract"
    )
    contract_sha = sha256_bytes(contract_raw)
    if contract_sha != plan["full_training_contract_sha256"]:
        raise ValueError("full training contract SHA256 mismatch")
    config = _verify_config(config_raw, contract_sha)
    _verify_config_contract_paths(config, contract_path, contract_raw)
    if Path(config["train"]["out_dir"]).resolve() != Path(plan["run_dir"]).resolve():
        raise ValueError("formal config out_dir differs from execution run_dir")
    if Path(config["data"]["full_training_contract"]).resolve() != contract_path:
        raise ValueError("formal config contract path differs from execution plan")

    verification_path, verification_raw = read_regular(
        plan["contract_verification_path"], "contract verification"
    )
    if sha256_bytes(verification_raw) != plan["contract_verification_sha256"]:
        raise ValueError("contract verification SHA256 mismatch")
    verification = json.loads(verification_raw)
    if (
        verification.get("status") != "PASS_FULL_TRAINING_DATA_CONTRACT"
        or verification.get("contract_sha256") != contract_sha
        or verification.get("train_domains") != 5218
    ):
        raise ValueError("contract verification is not the exact 5,218-domain PASS")

    data = _exact_keys(
        spec["data_identity"],
        {
            "mount_target",
            "mount_uuid",
            "mount_mode",
            "h5_count",
            "corpus_bytes",
            "train_domain_count",
            "training_pair_count",
            "manifest_sha256",
            "train_list_sha256",
            "contract_verification_sha256",
            "full_training_contract_sha256",
        },
        "data_identity",
    )
    expected_data = {
        "mount_uuid": plan["data_uuid"],
        "mount_mode": "ro",
        "h5_count": 5398,
        "corpus_bytes": 3613998101757,
        "train_domain_count": 5218,
        "training_pair_count": 59154922,
        "manifest_sha256": verification.get("manifest_sha256"),
        "train_list_sha256": verification.get("train_list_sha256"),
        "contract_verification_sha256": sha256_bytes(verification_raw),
        "full_training_contract_sha256": contract_sha,
    }
    for field, expected in expected_data.items():
        if data.get(field) != expected:
            raise ValueError(f"data_identity.{field} mismatch")
    if data["mount_target"] != "/data-full":
        raise ValueError("data mount target must be /data-full")

    if spec["checkpoint_plan"] != CHECKPOINT_PLAN:
        raise ValueError("checkpoint plan differs from the frozen strict plan")
    if spec["recovery_plan"] != RECOVERY_PLAN:
        raise ValueError("recovery plan differs from the frozen state-consistent plan")
    stop = _exact_keys(
        spec["stop_plan"],
        {
            "soft_stop_minutes",
            "hard_stop_minutes",
            "archive_kill_grace_seconds",
            "soft_stop_mechanism",
            "soft_stop_precedes_hard_stop",
            "archive_failure_soft_stop",
        },
        "stop_plan",
    )
    expected_stop = {
        "soft_stop_minutes": plan["soft_stop_minutes"],
        "hard_stop_minutes": plan["hard_stop_minutes"],
        "archive_kill_grace_seconds": plan["archive_kill_grace_seconds"],
        "soft_stop_mechanism": (
            "sealed_attempt_sentinel_at_optimizer_boundary"
        ),
        "soft_stop_precedes_hard_stop": True,
        "archive_failure_soft_stop": True,
    }
    if stop != expected_stop:
        raise ValueError("stop plan differs from execution plan")

    scientific = _exact_keys(
        spec["scientific_policy"],
        {
            "unique_endpoint_steps",
            "checkpoint_selection_forbidden",
            "hyperparameter_changes_before_endpoint_forbidden",
            "post_training_order",
            "formal_run_auto_start_forbidden",
        },
        "scientific_policy",
    )
    if (
        scientific["unique_endpoint_steps"] != [500000]
        or scientific["checkpoint_selection_forbidden"] is not True
        or scientific["hyperparameter_changes_before_endpoint_forbidden"] is not True
        or scientific["post_training_order"]
        != ["development", "external", "second_seed", "untouched"]
        or scientific["formal_run_auto_start_forbidden"] is not True
    ):
        raise ValueError("scientific endpoint policy mismatch")

    estimate = _exact_keys(
        spec["estimate_budget"],
        {
            "throughput_steps_per_second_low",
            "throughput_steps_per_second_high",
            "estimated_hours_low",
            "estimated_hours_high",
            "hard_cap_hours",
            "price_quote_path",
            "price_quote_sha256",
            "price_observed_at",
            "currency",
            "hourly_rate",
            "estimated_cost_low",
            "estimated_cost_high",
            "maximum_authorized_cost",
            "formula",
        },
        "estimate_budget",
    )
    low_rate = float(estimate["throughput_steps_per_second_low"])
    high_rate = float(estimate["throughput_steps_per_second_high"])
    if not (0 < low_rate <= high_rate):
        raise ValueError("throughput bounds are invalid")
    raw_low_hours = 500000 / high_rate / 3600
    raw_high_hours = 500000 / low_rate / 3600
    if (
        float(estimate["estimated_hours_low"]) < raw_low_hours
        or float(estimate["estimated_hours_high"]) < raw_high_hours
        or float(estimate["hard_cap_hours"]) < float(estimate["estimated_hours_high"])
    ):
        raise ValueError("duration estimate or hard cap is below measured-derived runtime")
    quote_path = _verify_bound_file(
        estimate["price_quote_path"], estimate["price_quote_sha256"], "price quote"
    )
    _, quote_raw = read_regular(quote_path, "price quote")
    try:
        quote = json.loads(quote_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("price quote is not valid JSON") from exc
    quote = _exact_keys(quote, PRICE_QUOTE_KEYS, "price quote")
    expected_quote_identity = {
        "schema": "deepjump.huawei_ecs_price_bound.v1",
        "provider": "Huawei Cloud",
        "region": GPU_REGION,
        "instance_id": GPU_INSTANCE_ID,
        "flavor": GPU_FLAVOR,
        "billing_mode": "pay_per_use",
        "currency": "CNY",
        "source": PRICE_BOUND_SOURCE,
        "reference_url": PRICE_REFERENCE_URL,
        "reference_hourly_rate": PRICE_REFERENCE_HOURLY_RATE,
        "reference_quantity": PRICE_REFERENCE_QUANTITY,
        "historical_console_hourly_rate": HISTORICAL_CONSOLE_HOURLY_RATE,
        "derivation": PRICE_BOUND_DERIVATION,
    }
    for field, expected in expected_quote_identity.items():
        if quote.get(field) != expected:
            raise ValueError(f"price quote {field} mismatch")
    package_created_at = _utc_timestamp(spec["created_at"], "package created_at")
    quote_observed_at = _utc_timestamp(quote["observed_at"], "price quote observed_at")
    quote_age_seconds = (package_created_at - quote_observed_at).total_seconds()
    if quote_age_seconds < 0 or quote_age_seconds > PRICE_QUOTE_MAX_AGE_SECONDS:
        raise ValueError("price quote is future-dated or older than 24 hours")
    if (
        estimate["price_observed_at"] != quote["observed_at"]
        or estimate["currency"] != quote["currency"]
        or float(estimate["hourly_rate"]) != float(quote["hourly_rate"])
    ):
        raise ValueError("estimate budget does not bind the exact price quote")
    if not quote_path.is_file() or estimate["currency"] != "CNY":
        raise ValueError("price quote identity or currency is invalid")
    hourly = float(estimate["hourly_rate"])
    conservative_hourly_bound = max(
        HISTORICAL_CONSOLE_HOURLY_RATE,
        PRICE_REFERENCE_HOURLY_RATE * PRICE_REFERENCE_QUANTITY,
    )
    if (
        hourly <= 0
        or hourly != conservative_hourly_bound
        or float(estimate["estimated_cost_low"])
        != round(hourly * float(estimate["estimated_hours_low"]), 2)
        or float(estimate["estimated_cost_high"])
        != round(hourly * float(estimate["estimated_hours_high"]), 2)
        or float(estimate["maximum_authorized_cost"])
        < round(hourly * float(estimate["hard_cap_hours"]), 2)
        or estimate["formula"] != "hourly_rate * powered_on_gpu_instance_hours"
    ):
        raise ValueError("cost estimate or maximum authorization is inconsistent")

    runtime_identity = _exact_keys(
        spec["runtime_identity"],
        {
            "provider",
            "region",
            "instance_id",
            "hostname",
            "product_uuid",
            "product_serial",
            "flavor",
            "gpu_model",
            "gpu_count",
        },
        "runtime_identity",
    )
    expected_runtime_identity = {
        "provider": "Huawei Cloud",
        "region": GPU_REGION,
        "instance_id": GPU_INSTANCE_ID,
        "hostname": GPU_HOSTNAME,
        "product_uuid": GPU_INSTANCE_ID,
        "product_serial": GPU_INSTANCE_ID,
        "flavor": GPU_FLAVOR,
        "gpu_model": GPU_MODEL,
        "gpu_count": 8,
    }
    if runtime_identity != expected_runtime_identity:
        raise ValueError("runtime identity differs from the exact authorized GPU")

    for label in ("source_identity", "obs_plan", "prerequisites"):
        if not isinstance(spec[label], dict) or not spec[label]:
            raise ValueError(f"{label} must be a non-empty mapping")
    if (
        not isinstance(spec["known_deviations"], list)
        or not spec["known_deviations"]
        or any(not isinstance(value, str) or not value for value in spec["known_deviations"])
    ):
        raise ValueError("known_deviations must be a non-empty string list")

    return {
        "schema": "deepjump.formal500k.package.v1",
        "status": "READY_FOR_USER_FORMAL_TRAINING_DECISION",
        "package_ready": True,
        "formal_training_authorized": False,
        "authorization_required": True,
        **{key: value for key, value in spec.items() if key != "schema"},
    }


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
