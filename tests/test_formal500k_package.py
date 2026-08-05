import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.formal500k_package_lib import build_package_payload


ROOT = Path(__file__).parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _sha(path)


def _fixture(tmp_path: Path) -> dict:
    data_root = tmp_path / "data-full/mdcath"
    data_root.mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest_sha = _write_json(manifest, [])
    train_list = tmp_path / "train_eligible_5218.txt"
    train_list.write_text("domain\n")
    train_list_sha = _sha(train_list)
    data_audit = tmp_path / "full_mdcath_audit.json"
    data_audit_sha = _write_json(data_audit, {"root": str(data_root)})
    contract = tmp_path / "full_training_data_contract.json"
    contract_sha = _write_json(
        contract,
        {
            "schema": "test-contract",
            "artifacts": {
                "data_audit": {
                    "path": data_audit.name,
                    "sha256": data_audit_sha,
                },
                "manifest": {
                    "path": manifest.name,
                    "sha256": manifest_sha,
                },
                "train_list": {
                    "path": train_list.name,
                    "sha256": train_list_sha,
                },
            },
        },
    )
    verification = tmp_path / "contract_verification.json"
    verification_sha = _write_json(
        verification,
        {
            "status": "PASS_FULL_TRAINING_DATA_CONTRACT",
            "contract_sha256": contract_sha,
            "manifest_sha256": manifest_sha,
            "train_list_sha256": train_list_sha,
            "train_domains": 5218,
        },
    )

    config = yaml.safe_load(
        (ROOT / "configs/v100_tensorcloud01_full_expanded_formal500k.yaml").read_text()
    )
    run_dir = tmp_path / "formal-run"
    config["data"].update(
        {
            "root": str(data_root),
            "manifest": str(manifest),
            "domains_file": str(train_list),
            "full_training_contract": str(contract),
            "full_training_contract_sha256": contract_sha,
        }
    )
    config["train"]["out_dir"] = str(run_dir)
    config_path = tmp_path / "formal500k.rendered.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    source = tmp_path / "source"
    source.write_text("#!/usr/bin/env python3\n")
    source.chmod(0o755)
    source_sha = _sha(source)
    python = Path(sys.executable).resolve()
    version = subprocess.run(
        [str(python), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    ).stdout.strip()
    tool = {
        "path": str(python),
        "sha256": _sha(python),
        "version_args": ["--version"],
        "version": version,
    }
    quote = tmp_path / "price_quote.json"
    _write_json(
        quote,
        {
            "schema": "deepjump.huawei_ecs_price_bound.v1",
            "provider": "Huawei Cloud",
            "region": "cn-north-4",
            "instance_id": "4c2273f2-4763-4827-839b-27d2c79cd76a",
            "flavor": "p2v.16xlarge.8",
            "billing_mode": "pay_per_use",
            "currency": "CNY",
            "hourly_rate": 133.58356,
            "observed_at": "2026-07-26T12:00:00Z",
            "source": (
                "official_huawei_cloud_reference_with_historical_console_upper_bound"
            ),
            "reference_url": (
                "https://support.huaweicloud.com/mineru-ctf/ctf-mineru.pdf"
            ),
            "reference_hourly_rate": 16.51,
            "reference_quantity": 8,
            "historical_console_hourly_rate": 133.58356,
            "derivation": (
                "max(historical_exact_8gpu_console_rate, "
                "official_single_v100_reference_rate * 8)"
            ),
        },
    )

    run_id = "20260726T120000Z"
    execution = {
        "reviewed_commit": "a" * 40,
        "run_id": run_id,
        "obs_dst": f"obs://bucket/formal500k/{run_id}",
        "data_uuid": "977cb910-fe4b-4ed7-b3d3-7b4098f5e141",
        "world_size": 8,
        "repo_root": str(tmp_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha(config_path),
        "contract_verification_path": str(verification.resolve()),
        "contract_verification_sha256": verification_sha,
        "full_training_contract_path": str(contract.resolve()),
        "full_training_contract_sha256": contract_sha,
        "supervisor_path": str(source.resolve()),
        "supervisor_sha256": source_sha,
        "archiver_path": str(source.resolve()),
        "archiver_sha256": source_sha,
        "validator_path": str(source.resolve()),
        "validator_sha256": source_sha,
        "empty_prefix_validator_path": str(source.resolve()),
        "empty_prefix_validator_sha256": source_sha,
        "trainer_path": str(source.resolve()),
        "trainer_sha256": source_sha,
        "soft_stop_minutes": 8970,
        "hard_stop_minutes": 9000,
        "archive_kill_grace_seconds": 600,
        "archive_poll_seconds": 60,
        "toolchain": {"python": tool, "torchrun": tool, "obsutil": tool},
    }
    spec = {
        "schema": "deepjump.formal500k.package_spec.v1",
        "package_id": "formal500k-test",
        "created_at": "2026-07-26T12:00:00Z",
        "classification": "closer_to_paper_not_exact_reproduction",
        "formal_candidate": {
            "fresh_init": True,
            "initial_checkpoint": None,
            "training_seed": 0,
            "data_seed": 0,
            "target_total_steps": 500000,
            "unique_scientific_endpoint_steps": [500000],
            "intermediate_checkpoint_policy": (
                "engineering_recovery_and_finite_monitoring_only_no_selection"
            ),
            "config_sha256": _sha(config_path),
            "execution_plan": execution,
        },
        "data_identity": {
            "mount_target": "/data-full",
            "mount_uuid": execution["data_uuid"],
            "mount_mode": "ro",
            "h5_count": 5398,
            "corpus_bytes": 3613998101757,
            "train_domain_count": 5218,
            "training_pair_count": 59154922,
            "manifest_sha256": manifest_sha,
            "train_list_sha256": train_list_sha,
            "contract_verification_sha256": verification_sha,
            "full_training_contract_sha256": contract_sha,
        },
        "source_identity": {"tracked_manifest_sha256": "3" * 64},
        "runtime_identity": {
            "provider": "Huawei Cloud",
            "region": "cn-north-4",
            "instance_id": "4c2273f2-4763-4827-839b-27d2c79cd76a",
            "hostname": "deepjump-v100-8gpu-20260716",
            "product_uuid": "4c2273f2-4763-4827-839b-27d2c79cd76a",
            "product_serial": "4c2273f2-4763-4827-839b-27d2c79cd76a",
            "flavor": "p2v.16xlarge.8",
            "gpu_model": "Tesla V100-SXM2-16GB",
            "gpu_count": 8,
        },
        "checkpoint_plan": {
            "ckpt_every": 1000,
            "trainer_keep_last_k": 501,
            "archiver_keep_local_verified": 3,
            "immutable_numbered": True,
            "local_strict_validator": True,
            "forced_obs_readback": True,
            "remote_strict_validator": True,
            "verified_remote_required_for_retention": True,
            "latest_verified_required": True,
        },
        "stop_plan": {
            "soft_stop_minutes": 8970,
            "hard_stop_minutes": 9000,
            "archive_kill_grace_seconds": 600,
            "soft_stop_mechanism": (
                "sealed_attempt_sentinel_at_optimizer_boundary"
            ),
            "soft_stop_precedes_hard_stop": True,
            "archive_failure_soft_stop": True,
        },
        "recovery_plan": {
            "separate_attempt_required": True,
            "max_recovery_attempts": 16,
            "resume_history_required": True,
            "strict_checkpoint_preflight": True,
            "latest_verified_only": True,
            "resume_semantics": "state_consistent_non_bitwise_crop_and_noise",
        },
        "obs_plan": {"atomic_ownership_claim_required": True},
        "scientific_policy": {
            "unique_endpoint_steps": [500000],
            "checkpoint_selection_forbidden": True,
            "hyperparameter_changes_before_endpoint_forbidden": True,
            "post_training_order": [
                "development",
                "external",
                "second_seed",
                "untouched",
            ],
            "formal_run_auto_start_forbidden": True,
        },
        "estimate_budget": {
            "throughput_steps_per_second_low": 1.05,
            "throughput_steps_per_second_high": 1.08,
            "estimated_hours_low": 129.0,
            "estimated_hours_high": 133.0,
            "hard_cap_hours": 150.0,
            "price_quote_path": str(quote.resolve()),
            "price_quote_sha256": _sha(quote),
            "price_observed_at": "2026-07-26T12:00:00Z",
            "currency": "CNY",
            "hourly_rate": 133.58356,
            "estimated_cost_low": 17232.28,
            "estimated_cost_high": 17766.61,
            "maximum_authorized_cost": 20037.54,
            "formula": "hourly_rate * powered_on_gpu_instance_hours",
        },
        "known_deviations": ["paper implementation and exact optimizer defaults unavailable"],
        "prerequisites": {"post_download_qualification": "PASS"},
    }
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, spec)
    return {"spec": spec_path, "payload": spec}


def test_package_builder_is_ready_but_never_authorizes(tmp_path):
    fixture = _fixture(tmp_path)
    package = build_package_payload(fixture["spec"])
    assert package["status"] == "READY_FOR_USER_FORMAL_TRAINING_DECISION"
    assert package["package_ready"] is True
    assert package["formal_training_authorized"] is False
    assert package["formal_candidate"]["target_total_steps"] == 500000


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("formal_candidate", "target_total_steps"), 499999),
        (("checkpoint_plan", "ckpt_every"), 10000),
        (("recovery_plan", "latest_verified_only"), False),
        (("recovery_plan", "max_recovery_attempts"), 17),
        (("runtime_identity", "product_uuid"), "0" * 36),
        (("scientific_policy", "checkpoint_selection_forbidden"), False),
    ],
)
def test_package_builder_rejects_semantic_drift(tmp_path, path, value):
    fixture = _fixture(tmp_path)
    payload = fixture["payload"]
    payload[path[0]][path[1]] = value
    _write_json(fixture["spec"], payload)
    with pytest.raises(ValueError):
        build_package_payload(fixture["spec"])


def test_package_builder_rejects_unknown_fields(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["payload"]["unexpected"] = True
    _write_json(fixture["spec"], fixture["payload"])
    with pytest.raises(ValueError, match="must contain exactly"):
        build_package_payload(fixture["spec"])


def test_package_builder_requires_run_scoped_obs_prefix(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["payload"]["formal_candidate"]["execution_plan"]["obs_dst"] = (
        "obs://bucket/formal500k/not-the-run-id"
    )
    _write_json(fixture["spec"], fixture["payload"])
    with pytest.raises(ValueError, match="execution identity"):
        build_package_payload(fixture["spec"])


def test_package_builder_rejects_data_root_outside_qualified_audit(tmp_path):
    fixture = _fixture(tmp_path)
    payload = fixture["payload"]
    execution = payload["formal_candidate"]["execution_plan"]
    config_path = Path(execution["config_path"])
    config = yaml.safe_load(config_path.read_text())
    config["data"]["root"] = str(Path(config["data"]["root"]) / "data")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    config_sha = _sha(config_path)
    execution["config_sha256"] = config_sha
    payload["formal_candidate"]["config_sha256"] = config_sha
    _write_json(fixture["spec"], payload)
    with pytest.raises(ValueError, match="qualified audit root"):
        build_package_payload(fixture["spec"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_id", "00000000-0000-0000-0000-000000000000"),
        ("flavor", "p2v.8xlarge.8"),
        ("region", "cn-south-1"),
        ("billing_mode", "yearly_monthly"),
        ("hourly_rate", 99.0),
        ("reference_hourly_rate", 16.49),
        ("historical_console_hourly_rate", 133.0),
    ],
)
def test_package_builder_rejects_price_quote_identity_or_value_drift(
    tmp_path, field, value
):
    fixture = _fixture(tmp_path)
    payload = fixture["payload"]
    quote_path = Path(payload["estimate_budget"]["price_quote_path"])
    quote = json.loads(quote_path.read_text())
    quote[field] = value
    _write_json(quote_path, quote)
    payload["estimate_budget"]["price_quote_sha256"] = _sha(quote_path)
    _write_json(fixture["spec"], payload)
    with pytest.raises(ValueError):
        build_package_payload(fixture["spec"])


def test_package_builder_rejects_stale_price_quote(tmp_path):
    fixture = _fixture(tmp_path)
    payload = fixture["payload"]
    quote_path = Path(payload["estimate_budget"]["price_quote_path"])
    quote = json.loads(quote_path.read_text())
    quote["observed_at"] = "2026-07-24T11:59:59Z"
    _write_json(quote_path, quote)
    payload["estimate_budget"]["price_quote_sha256"] = _sha(quote_path)
    payload["estimate_budget"]["price_observed_at"] = quote["observed_at"]
    _write_json(fixture["spec"], payload)
    with pytest.raises(ValueError, match="older than 24 hours"):
        build_package_payload(fixture["spec"])
