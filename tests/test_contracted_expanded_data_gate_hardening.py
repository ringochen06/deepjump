import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from deepjump.config import load_config, to_dict
from deepjump.model import DeepJumpLite
from scripts.contracted_guarded_endpoint_panel_eval import (
    PREREQUISITE_SCHEMA,
    SCOPE,
    verify_reserved_evaluation_prerequisite,
)
from scripts.summarize_contracted_expanded_data_gate import summarize
from scripts.train_ddp import training_semantics_sha256
from scripts.validate_training_checkpoint import validate_checkpoint
from scripts.verify_audit_readback import verify_audit_readback


SOURCE_CONFIG = Path("configs/v100_tensorcloud01_full_expanded_d1_smoke100.yaml")
SHA = {
    "result": "2" * 64,
    "checkpoint": "3" * 64,
    "contract": "4" * 64,
    "panel": "5" * 64,
    "prerequisite": "6" * 64,
    "source": "8" * 64,
    "smoke_checkpoint": "a" * 64,
    "calibration_checkpoint": "b" * 64,
}
RUN_ID = "20260723T010203Z"
COMMIT = "c" * 40
OBS = "obs://bucket/prefix"


def _write_json(path: Path, payload: object) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_fixture(tmp_path: Path) -> dict:
    cfg = load_config(SOURCE_CONFIG)
    cfg.model.hidden = 32
    cfg.model.vector_channels = 32
    sealed_config = tmp_path / "smoke.yaml"
    sealed_config.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))

    contract = tmp_path / "contract_verification.json"
    contract_payload = {
        "status": "PASS_FULL_TRAINING_DATA_CONTRACT",
        "contract_sha256": "d" * 64,
        "train_list_sha256": "e" * 64,
    }
    contract_sha256 = _write_json(contract, contract_payload)

    model = DeepJumpLite(
        cfg.model,
        noise_sigma=cfg.data.noise_sigma,
        predict_heavy=cfg.model.predict_heavy,
    )
    named_parameters = list(model.named_parameters())
    parameter_count = sum(parameter.requires_grad for _, parameter in named_parameters)
    unused = {
        "transport.blocks.5.feedforward.scalar_out.weight",
        "transport.blocks.5.feedforward.scalar_out.bias",
    }
    optimizer_states = {
        index: {
            "step": torch.tensor(100.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        for index, (name, parameter) in enumerate(named_parameters)
        if parameter.requires_grad and name not in unused
    }
    checkpoint = tmp_path / "last.ckpt"
    torch.save(
        {
            "step": 100,
            "checkpoint_schema": 2,
            "cfg": to_dict(cfg),
            "train_state": {
                "world_size": 8,
                "train_dataset_size": 5_000,
                "sampler_num_samples": 625,
                "sampler_seed": cfg.train.seed,
                "sampler_epoch": 0,
                "samples_consumed_per_rank": 1_600,
                "batch_size": cfg.train.batch_size,
                "grad_accum": cfg.train.grad_accum,
                "train_fingerprint": "f" * 64,
                "training_semantics_sha256": training_semantics_sha256(cfg),
                "full_training_data_contract": contract_payload,
                "crop_resume": "stochastic_worker_rng_not_bitwise",
            },
            "model": model.state_dict(),
            "opt": {
                "state": optimizer_states,
                "param_groups": [{"params": list(range(parameter_count))}],
            },
            "scaler": {},
        },
        checkpoint,
    )
    history = tmp_path / "history.json"
    _write_json(
        history,
        [
            {"step": 50, "val_loss": 1.1, "val_rmsd": 2.1, "noop_rmsd": 3.0},
            {"step": 100, "val_loss": 1.0, "val_rmsd": 2.0, "noop_rmsd": 3.0},
        ],
    )
    return {
        "checkpoint": checkpoint,
        "history": history,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "config": sealed_config,
    }


def _validate_exact_checkpoint(fixture: dict, *, checkpoint_sha256: str | None = None):
    return validate_checkpoint(
        fixture["checkpoint"],
        100,
        8,
        fixture["history"],
        expected_delta=1,
        require_full_tensor=True,
        expected_lr_horizon_steps=500_000,
        expected_config_path=fixture["config"],
        expected_contract_verification_path=fixture["contract"],
        expected_contract_verification_sha256=fixture["contract_sha256"],
        expected_checkpoint_sha256=(
            checkpoint_sha256 or _file_sha256(fixture["checkpoint"])
        ),
    )


def test_checkpoint_gate_exactly_binds_full_restart_state(tmp_path):
    fixture = _checkpoint_fixture(tmp_path)
    report, errors = _validate_exact_checkpoint(fixture)
    assert errors == []
    assert report["checkpoint_sha256"] == _file_sha256(fixture["checkpoint"])
    assert report["history_steps"] == [50, 100]


@pytest.mark.parametrize("mutation", ["config", "semantics", "contract"])
def test_checkpoint_gate_rejects_identity_drift(tmp_path, mutation):
    fixture = _checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    if mutation == "config":
        payload["cfg"]["train"]["lr"] *= 0.5
    elif mutation == "semantics":
        payload["train_state"]["training_semantics_sha256"] = "0" * 64
    else:
        payload["train_state"]["full_training_data_contract"] = {
            **payload["train_state"]["full_training_data_contract"],
            "train_list_sha256": "0" * 64,
        }
    torch.save(payload, fixture["checkpoint"])
    _, errors = _validate_exact_checkpoint(fixture)
    assert errors


def test_checkpoint_gate_rejects_same_path_sha_drift(tmp_path):
    fixture = _checkpoint_fixture(tmp_path)
    frozen_sha = _file_sha256(fixture["checkpoint"])
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    payload["step"] = 99
    torch.save(payload, fixture["checkpoint"])
    _, errors = _validate_exact_checkpoint(fixture, checkpoint_sha256=frozen_sha)
    assert "training checkpoint SHA256 mismatch" in errors


@pytest.mark.parametrize(
    ("step", "history_steps"),
    [
        (25, []),
        (75, [50]),
    ],
)
def test_checkpoint_gate_through_accepts_only_completed_validation_cadence(
    tmp_path, step, history_steps
):
    fixture = _checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    payload["step"] = step
    torch.save(payload, fixture["checkpoint"])
    _write_json(
        fixture["history"],
        [
            {
                "step": history_step,
                "val_loss": 1.0,
                "val_rmsd": 2.0,
                "noop_rmsd": 3.0,
            }
            for history_step in history_steps
        ],
    )

    _, errors = validate_checkpoint(
        fixture["checkpoint"],
        step,
        8,
        fixture["history"],
        history_mode="through",
        expected_delta=1,
        require_full_tensor=True,
        expected_lr_horizon_steps=500_000,
        expected_config_path=fixture["config"],
        expected_contract_verification_path=fixture["contract"],
        expected_contract_verification_sha256=fixture["contract_sha256"],
        expected_checkpoint_sha256=_file_sha256(fixture["checkpoint"]),
    )
    assert errors == []


def test_checkpoint_gate_through_rejects_future_or_missing_validation_records(tmp_path):
    fixture = _checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    payload["step"] = 75
    torch.save(payload, fixture["checkpoint"])

    _, errors = validate_checkpoint(
        fixture["checkpoint"],
        75,
        8,
        fixture["history"],
        history_mode="through",
        expected_delta=1,
        require_full_tensor=True,
        expected_lr_horizon_steps=500_000,
        expected_config_path=fixture["config"],
        expected_contract_verification_path=fixture["contract"],
        expected_contract_verification_sha256=fixture["contract_sha256"],
        expected_checkpoint_sha256=_file_sha256(fixture["checkpoint"]),
    )
    assert any("exact completed validation cadence" in error for error in errors)


@pytest.mark.parametrize("missing", ["opt", "scaler"])
def test_checkpoint_gate_rejects_missing_restart_state(tmp_path, missing):
    fixture = _checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    payload.pop(missing)
    torch.save(payload, fixture["checkpoint"])
    _, errors = _validate_exact_checkpoint(fixture)
    assert any(missing in error for error in errors)


@pytest.mark.parametrize(
    "missing",
    [
        "train_dataset_size",
        "sampler_num_samples",
        "sampler_seed",
        "sampler_epoch",
        "samples_consumed_per_rank",
        "batch_size",
        "grad_accum",
        "train_fingerprint",
        "crop_resume",
    ],
)
def test_checkpoint_gate_rejects_missing_train_state_contract(tmp_path, missing):
    fixture = _checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    payload["train_state"].pop(missing)
    torch.save(payload, fixture["checkpoint"])
    _, errors = _validate_exact_checkpoint(fixture)
    assert any("train_state fields missing" in error for error in errors)


@pytest.mark.parametrize(
    "mutation",
    [
        "model_key",
        "model_shape",
        "opt_partial",
        "opt_bad_step",
        "opt_bad_dtype",
        "opt_nan",
        "scaler_inf",
    ],
)
def test_checkpoint_gate_rejects_incomplete_or_nonfinite_serialized_state(
    tmp_path, mutation
):
    fixture = _checkpoint_fixture(tmp_path)
    payload = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    first_name = next(iter(payload["model"]))
    if mutation == "model_key":
        payload["model"].pop(first_name)
    elif mutation == "model_shape":
        payload["model"][first_name] = torch.zeros(1)
    elif mutation == "opt_partial":
        payload["opt"]["state"] = {
            next(iter(payload["opt"]["state"])): next(iter(payload["opt"]["state"].values()))
        }
    elif mutation == "opt_bad_step":
        next(iter(payload["opt"]["state"].values()))["step"] = "not-a-step"
    elif mutation == "opt_bad_dtype":
        first_state = next(iter(payload["opt"]["state"].values()))
        first_state["exp_avg"] = first_state["exp_avg"].to(torch.int64)
        first_state["exp_avg_sq"] = first_state["exp_avg_sq"].to(torch.int64)
    elif mutation == "opt_nan":
        first_state = next(iter(payload["opt"]["state"].values()))
        first_state["exp_avg"].view(-1)[0] = float("nan")
    else:
        payload["scaler"] = {"scale": float("inf")}
    torch.save(payload, fixture["checkpoint"])
    _, errors = _validate_exact_checkpoint(fixture)
    assert errors


@pytest.mark.parametrize("mutation", ["early_nan", "duplicate", "cadence", "noop_drift"])
def test_checkpoint_gate_rejects_h20_style_last_record_pseudo_pass(tmp_path, mutation):
    fixture = _checkpoint_fixture(tmp_path)
    history = json.loads(fixture["history"].read_text())
    if mutation == "early_nan":
        history[0]["val_rmsd"] = float("nan")
    elif mutation == "duplicate":
        history[0]["step"] = 100
    elif mutation == "cadence":
        history[0]["step"] = 49
    else:
        history[0]["noop_rmsd"] = 3.1
    fixture["history"].write_text(json.dumps(history))
    _, errors = _validate_exact_checkpoint(fixture)
    assert errors


def _checkpoint_gates(tmp_path: Path) -> tuple[dict, dict, dict]:
    checkpoint_paths = {}
    checkpoints = {}
    for name in ("smoke", "calibration", "development"):
        path = tmp_path / f"{name}.ckpt"
        path.write_bytes(f"sealed-{name}-checkpoint\n".encode())
        checkpoint_paths[name] = path
        checkpoints[name] = _file_sha256(path)
    steps = {"smoke": 100, "calibration": 1_000, "development": 2_000}
    paths = {}
    bindings = {}
    for name in checkpoints:
        path = tmp_path / f"{name}_checkpoint_gate.json"
        digest = _write_json(
            path,
            {
                "status": "PASS",
                "checkpoint_sha256": checkpoints[name],
                "checkpoint_step": steps[name],
            },
        )
        paths[name] = path
        bindings[name] = {
            "status": "PASS",
            "sha256": digest,
            "checkpoint_sha256": checkpoints[name],
            "checkpoint_step": steps[name],
        }
    return checkpoint_paths, paths, bindings


def test_reserved_authorization_preserves_sealed_run_binding(tmp_path):
    _, _, gate_bindings = _checkpoint_gates(tmp_path)
    development_checkpoint_sha = gate_bindings["development"]["checkpoint_sha256"]
    run_binding = {
        "run_id": RUN_ID,
        "commit": COMMIT,
        "obs": OBS,
        "source_identity_manifest_sha256": SHA["source"],
        "checkpoint_gates": gate_bindings,
    }
    authorization = tmp_path / "authorization.json"
    authorization_sha = _write_json(
        authorization,
        {
            "schema": PREREQUISITE_SCHEMA,
            "authorization_id": f"development-{RUN_ID}",
            "consumption_ledger_root": str((tmp_path / "ledger").resolve()),
            "status": "ADVANCE_EXPANDED_DATA_DEVELOPMENT",
            "phase": "development",
            "checkpoint_sha256": development_checkpoint_sha,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": SHA["contract"],
            "panel_name": "legacy_dev20",
            "panel_sha256": SHA["panel"],
            "reserved_panel_authorized": True,
            "formal_training_authorized": False,
            "run_binding": run_binding,
        },
    )
    report = verify_reserved_evaluation_prerequisite(
        authorization,
        authorization_sha,
        phase="development",
        checkpoint_sha256=development_checkpoint_sha,
        checkpoint_step=2_000,
        contract_sha256=SHA["contract"],
        panel_name="legacy_dev20",
        panel_sha256=SHA["panel"],
    )
    assert report["run_binding"] == run_binding


def _decision(tmp_path: Path, status: str) -> tuple[Path, Path, str, str, dict, dict]:
    runtime_path = tmp_path / "runtime.json"
    runtime = {"status": "PASS_RUNTIME_PROBE", "domain": "d1"}
    runtime_sha = _write_json(runtime_path, runtime)
    checkpoint_paths, gate_paths, gate_bindings = _checkpoint_gates(tmp_path)
    development_checkpoint_sha = gate_bindings["development"]["checkpoint_sha256"]
    gate_status = (
        "PASS_CONTRACTED_GUARD_DEVELOPMENT20"
        if status == "ADVANCE_EXPANDED_DATA_EXTERNAL"
        else status
    )
    run_binding = {
        "run_id": RUN_ID,
        "commit": COMMIT,
        "obs": OBS,
        "source_identity_manifest_sha256": SHA["source"],
        "checkpoint_gates": gate_bindings,
    }
    decision = {
        "status": status,
        "gate_status": gate_status,
        "scope": SCOPE,
        "phase": "development",
        "result_sha256": SHA["result"],
        "checkpoint_sha256": development_checkpoint_sha,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": SHA["contract"],
        "panel_name": "legacy_dev20",
        "panel_sha256": SHA["panel"],
        "prerequisite_decision_sha256": SHA["prerequisite"],
        "runtime_probe": {**runtime, "sha256": runtime_sha},
        "identity": {
            "status": "PASS_FROZEN_EVALUATION_IDENTITY",
            "phase": "development",
            "panel_name": "legacy_dev20",
            "panel_sha256": SHA["panel"],
            "checkpoint_sha256": development_checkpoint_sha,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": SHA["contract"],
            "formal_training_authorized": False,
        },
        "prerequisite": {
            "status": "ADVANCE_EXPANDED_DATA_DEVELOPMENT",
            "phase": "development",
            "panel_name": "legacy_dev20",
            "panel_sha256": SHA["panel"],
            "checkpoint_sha256": development_checkpoint_sha,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": SHA["contract"],
            "reserved_panel_authorized": True,
            "formal_training_authorized": False,
            "run_binding": run_binding,
            "sha256": SHA["prerequisite"],
        },
        "consumption_claim": {
            "authorization_sha256": SHA["prerequisite"],
            "phase": "development",
            "panel_name": "legacy_dev20",
            "panel_sha256": SHA["panel"],
            "checkpoint_sha256": development_checkpoint_sha,
            "checkpoint_step": 2_000,
            "full_training_contract_sha256": SHA["contract"],
            "sha256": "9" * 64,
        },
        "formal_training_authorized": False,
    }
    decision_path = tmp_path / "decision.json"
    decision_sha = _write_json(decision_path, decision)
    return (
        decision_path,
        runtime_path,
        decision_sha,
        runtime_sha,
        checkpoint_paths,
        gate_paths,
        gate_bindings,
    )


def _summarize_kwargs(
    runtime, runtime_sha, checkpoint_paths, gate_paths, gate_bindings
) -> dict:
    return {
        "expected_result_sha256": SHA["result"],
        "expected_checkpoint_sha256": gate_bindings["development"][
            "checkpoint_sha256"
        ],
        "expected_contract_sha256": SHA["contract"],
        "expected_panel_name": "legacy_dev20",
        "expected_panel_sha256": SHA["panel"],
        "expected_prerequisite_decision_sha256": SHA["prerequisite"],
        "runtime_probe_path": runtime,
        "expected_runtime_probe_sha256": runtime_sha,
        "run_id": RUN_ID,
        "commit": COMMIT,
        "obs": OBS,
        "source_identity_manifest_sha256": SHA["source"],
        "smoke_checkpoint_gate_path": gate_paths["smoke"],
        "expected_smoke_checkpoint_gate_sha256": gate_bindings["smoke"]["sha256"],
        "smoke_checkpoint_path": checkpoint_paths["smoke"],
        "expected_smoke_checkpoint_sha256": gate_bindings["smoke"][
            "checkpoint_sha256"
        ],
        "calibration_checkpoint_gate_path": gate_paths["calibration"],
        "expected_calibration_checkpoint_gate_sha256": gate_bindings["calibration"]["sha256"],
        "calibration_checkpoint_path": checkpoint_paths["calibration"],
        "expected_calibration_checkpoint_sha256": gate_bindings["calibration"][
            "checkpoint_sha256"
        ],
        "development_checkpoint_gate_path": gate_paths["development"],
        "expected_development_checkpoint_gate_sha256": gate_bindings["development"]["sha256"],
        "development_checkpoint_path": checkpoint_paths["development"],
    }


@pytest.mark.parametrize(
    "status", ["ADVANCE_EXPANDED_DATA_EXTERNAL", "STOP_CONTRACTED_GUARD_NO_ADVANTAGE"]
)
def test_summary_binds_all_decision_and_stage_identities(tmp_path, status):
    (
        decision, runtime, decision_sha, runtime_sha, checkpoint_paths,
        gate_paths, gate_bindings,
    ) = _decision(tmp_path, status)
    summary = summarize(
        decision,
        expected_decision_sha256=decision_sha,
        **_summarize_kwargs(
            runtime, runtime_sha, checkpoint_paths, gate_paths, gate_bindings
        ),
    )
    assert summary["status"] == status
    assert summary["decision_sha256"] == decision_sha
    assert summary["checkpoint_gates"] == gate_bindings
    assert summary["formal_training_authorized"] is False


@pytest.mark.parametrize("field", ["run_id", "commit", "obs", "source_identity_manifest_sha256"])
def test_summary_rejects_free_parameter_identity_drift(tmp_path, field):
    (
        decision, runtime, decision_sha, runtime_sha, checkpoint_paths,
        gate_paths, gate_bindings,
    ) = _decision(tmp_path, "ADVANCE_EXPANDED_DATA_EXTERNAL")
    kwargs = _summarize_kwargs(
        runtime, runtime_sha, checkpoint_paths, gate_paths, gate_bindings
    )
    kwargs[field] = {
        "run_id": "20260723T010204Z",
        "commit": "d" * 40,
        "obs": "obs://bucket/other",
        "source_identity_manifest_sha256": "0" * 64,
    }[field]
    with pytest.raises(ValueError, match="run binding mismatch"):
        summarize(decision, expected_decision_sha256=decision_sha, **kwargs)


def test_summary_rejects_checkpoint_gate_identity_drift(tmp_path):
    (
        decision, runtime, decision_sha, runtime_sha, checkpoint_paths,
        gate_paths, gate_bindings,
    ) = _decision(tmp_path, "ADVANCE_EXPANDED_DATA_EXTERNAL")
    kwargs = _summarize_kwargs(
        runtime, runtime_sha, checkpoint_paths, gate_paths, gate_bindings
    )
    kwargs["expected_smoke_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        summarize(decision, expected_decision_sha256=decision_sha, **kwargs)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_summary_reopens_all_three_checkpoint_bytes(tmp_path, mutation):
    (
        decision, runtime, decision_sha, runtime_sha, checkpoint_paths,
        gate_paths, gate_bindings,
    ) = _decision(tmp_path, "ADVANCE_EXPANDED_DATA_EXTERNAL")
    if mutation == "missing":
        checkpoint_paths["calibration"].unlink()
    else:
        checkpoint_paths["calibration"].write_bytes(b"changed checkpoint\n")
    kwargs = _summarize_kwargs(
        runtime, runtime_sha, checkpoint_paths, gate_paths, gate_bindings
    )
    with pytest.raises((OSError, ValueError)):
        summarize(decision, expected_decision_sha256=decision_sha, **kwargs)


def test_summary_rejects_previously_accepted_minimal_forgery(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime_sha = _write_json(runtime, {"status": "PASS_RUNTIME_PROBE"})
    checkpoint_paths, gate_paths, gate_bindings = _checkpoint_gates(tmp_path)
    decision = tmp_path / "decision.json"
    decision_sha = _write_json(
        decision,
        {"status": "ADVANCE_EXPANDED_DATA_EXTERNAL", "formal_training_authorized": False},
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        summarize(
            decision,
            expected_decision_sha256=decision_sha,
            **_summarize_kwargs(
                runtime, runtime_sha, checkpoint_paths, gate_paths, gate_bindings
            ),
        )


def _audit_tree(tmp_path: Path) -> Path:
    root = tmp_path / "readback"
    (root / "configs").mkdir(parents=True)
    (root / "evidence").mkdir()
    first = root / "configs" / "sealed.yaml"
    second = root / "evidence" / "decision.json"
    first.write_text("sealed\n")
    second.write_text("{}\n")
    lines = []
    for path in (first, second):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (root / "audit_sha256.txt").write_text("".join(lines))
    (root / "runner.log").write_text("operational\n")
    return root


def test_exact_audit_readback_passes_only_declared_tree(tmp_path):
    root = _audit_tree(tmp_path)
    report = verify_audit_readback(root, allowed_relative=("runner.log",))
    assert report["status"] == "PASS_EXACT_AUDIT_READBACK"


@pytest.mark.parametrize("mutation", ["alter", "extra", "missing"])
def test_exact_audit_readback_rejects_tree_drift(tmp_path, mutation):
    root = _audit_tree(tmp_path)
    if mutation == "alter":
        (root / "evidence" / "decision.json").write_text("changed\n")
    elif mutation == "extra":
        (root / "evidence" / "extra.json").write_text("{}\n")
    else:
        (root / "configs" / "sealed.yaml").unlink()
    with pytest.raises(ValueError):
        verify_audit_readback(root, allowed_relative=("runner.log",))
