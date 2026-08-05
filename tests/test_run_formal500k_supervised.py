import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SUPERVISOR = ROOT / "cloud/huawei/run_formal500k_supervised.sh"


def test_supervisor_is_fail_closed_and_has_separate_attempt_claims():
    text = SUPERVISOR.read_text()
    assert text.startswith("#!/usr/bin/env -S -i ")
    assert "DEEPJUMP_SANITIZED_LAUNCH=1" in text
    assert "supervisor must be executed directly through its sanitized shebang" in text
    assert "export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0" in text
    assert "PASS_FORMAL_TRAINING_AUTHORIZATION" in text
    assert 'package.get("formal_training_authorized") is not False' in text
    assert 'authorization.get("formal_training_authorized") is not True' in text
    assert "atomic fresh-run claim failed" in text
    assert "recovery attempt already exists" in text
    assert "O_EXCL" in text
    assert 'flock -n 9 || fail "another supervisor already owns this formal RUN_ID"' in text
    assert '"run_lock_held": True' in text
    assert "checkpoint archiver readiness timed out" in text
    assert "ARCHIVER_READY_AFTER_INITIAL_ROUND" in text
    assert "checkpoint archiver exited after readiness" in text
    assert "checkpoint archiver failed before trainer launch" in text
    assert 'git -C "$REPO_ROOT" archive "$EXPECTED_COMMIT"' in text
    assert 'export PYTHONPATH="$SOURCE_SNAPSHOT/src"' in text
    assert "runtime tool SHA256 drift" in text
    assert "authorization-expiry hard stop" in text
    assert 'emergency_poweroff "package-bound maximum recovery attempts exhausted"' in text
    assert "RECOVERY_ATTEMPTS_EXHAUSTED_REQUEST_POWER_OFF" in text
    assert "--on-active=2m" in text
    assert "immediate poweroff request failed; emergency_timer_active=" in text
    assert 'path.chmod(mode & ~0o222)' in text
    assert "os.chmod(destination, 0o400)" in text
    assert 'ARCHIVE_ONCE=0 setsid "$ARCHIVER"' in text
    assert 'kill -TERM -- "-$owned_pid"' in text
    assert 'kill -KILL -- "-$owned_pid"' in text
    assert "stop_archiver_group" in text
    assert "FINAL_ARCHIVE_FAILED" in text
    assert "record_archiver_failure" in text
    assert "ARCHIVER_FAILURE_FILE" not in text
    assert "FULL_TRAINING_CONTRACT_SNAPSHOT=$(snapshot_input" in text
    assert (
        'verify_runtime_file "$FULL_TRAINING_CONTRACT_SNAPSHOT"'
        in text
    )
    assert '--full-training-contract "$FULL_TRAINING_CONTRACT"' in text
    assert "--warm-start" not in text
    assert '--resume "$RESUME_CHECKPOINT" --resume-history "$RESUME_HISTORY"' in text


def test_supervisor_freezes_world_size_and_soft_stop_precedes_hard_stop():
    text = SUPERVISOR.read_text()
    assert "--nproc_per_node=8" in text
    assert "effective batch is not 128" in text
    assert "soft stop must precede hard stop" in text
    assert "at least thirty minutes" in text
    assert "hard-stop timer did not become active" in text
    assert "--graceful-stop-file" in text
    assert "GRACEFUL_STOP_REQUESTED" in text
    assert 'kill -TERM -- "-$trainer_pid"' in text
    assert "ARCHIVER_FAILURE_REQUEST_SOFT_STOP" in text
    assert "PASS_STRICT_CHECKPOINT_OBS_READBACK" in text


def test_supervisor_cross_binds_runtime_to_package_execution_plans():
    text = SUPERVISOR.read_text()
    for field in (
        "reviewed_commit",
        "run_id",
        "obs_dst",
        "data_uuid",
        "world_size",
        "supervisor_sha256",
        "archiver_sha256",
        "trainer_sha256",
        "validator_sha256",
        "soft_stop_minutes",
        "hard_stop_minutes",
        "archive_kill_grace_seconds",
        "archive_poll_seconds",
    ):
        assert f'"{field}"' in text
    assert 'package.get("checkpoint_plan")' in text
    assert 'package.get("stop_plan")' in text
    assert 'package.get("recovery_plan")' in text
    assert '"resume_history_required": True' in text
    assert '"separate_attempt_required": True' in text


def test_supervisor_rejects_runtime_overrides_and_recovers_only_from_obs_proof():
    text = SUPERVISOR.read_text()
    assert "exact package and authorization arguments are required" in text
    assert "FORMAL_AUTHORIZATION=${FORMAL_AUTHORIZATION:?" not in text
    assert "RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?" not in text
    assert "RESUME_HISTORY=${RESUME_HISTORY:?" not in text
    assert '"$OBS_DST/LATEST_VERIFIED.json"' in text
    assert "LATEST_VERIFIED step is invalid" in text
    assert "does not exactly match its immutable step marker" in text
    assert "candidate_step=500000" not in text
    assert "forced {label} readback SHA mismatch" in text
    assert "authorization_verification_sha256" in text
    assert "stale_local_unverified" in text
    assert "PRESERVED_NOT_SELECTED_FOR_RECOVERY" in text


def test_supervisor_rejects_subset_and_step_hardcoding():
    text = SUPERVISOR.read_text()
    assert "subset1000" not in text
    assert "/data/mdcath" not in text
    assert "2000" not in text
    assert "10650" not in text


def test_supervisor_claims_empty_obs_prefix_and_verifies_hard_stop():
    text = SUPERVISOR.read_text()
    assert "verify_obsutil_empty_prefix" not in text  # path is package-bound
    assert "empty_prefix_validator_sha256" in text
    assert "CLAIMED_EXACT_HOST_AND_RUN" in text
    assert "recovery OBS ownership does not bind exact host/run/package" in text
    assert '"$OBS_DST/OWNERSHIP.json"' in text
    assert "prefix_file_inventory" in text
    assert "hard-stop service ExecStart is not exact poweroff" in text
    assert "NextElapseUSecRealtime" in text
    assert '--on-calendar="$hard_stop_calendar"' in text
    assert text.count("--timer-property=AccuracySec=1s") >= 3
    assert text.count("-p AccuracyUSec --value") >= 2
    assert "authorization_stop_target_epoch=$((AUTHORIZATION_EXPIRES_EPOCH - 2))" in text
    assert "latest_authorized_hard_stop_epoch=$((AUTHORIZATION_EXPIRES_EPOCH - 2))" in text
    assert "scheduled hard-stop exceeds authorization expiry" in text
    assert "hard-stop timer deadline differs from the absolute deadline" in text
    assert "cleanup_children" in text


def test_supervisor_accepts_only_a_cross_bound_package_before_host_checks(tmp_path):
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    (repo / "tracked").write_text("x\n")
    subprocess.run(["git", "-C", repo, "add", "tracked"], check=True)
    subprocess.run(
        [
            "git", "-C", repo, "-c", "user.name=test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "base",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()

    contract = tmp_path / "contract.json"
    contract.write_text('{"schema":"test"}\n')
    contract_sha = digest(contract)
    verification = tmp_path / "verification.json"
    verification.write_text('{"status":"PASS_FULL_TRAINING_DATA_CONTRACT"}\n')
    verification_sha = digest(verification)
    config = yaml.safe_load(
        (ROOT / "configs/v100_tensorcloud01_full_expanded_formal500k.yaml").read_text()
    )
    config["data"].update(
        {
            "root": "/tmp",
            "manifest": "/tmp/manifest.json",
            "domains_file": "/tmp/domains.txt",
            "full_training_contract": str(contract),
            "full_training_contract_sha256": contract_sha,
        }
    )
    config["train"]["out_dir"] = str(repo / "runs/formal")
    config_path = tmp_path / "formal.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    config_sha = digest(config_path)

    python = str(Path(sys.executable).resolve())
    benign_tool = str(Path("/usr/bin/true").resolve())

    def tool(path, version_args):
        version = subprocess.run(
            [path, *version_args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        ).stdout.strip()
        return {
            "path": path,
            "sha256": digest(path),
            "version_args": version_args,
            "version": version,
        }

    run_id = "20260726T120000Z"
    execution_plan = {
        "reviewed_commit": commit,
        "run_id": run_id,
        "obs_dst": f"obs://bucket/formal/{run_id}",
        "data_uuid": "",
        "world_size": 8,
        "repo_root": str(repo),
        "run_dir": str(repo / "runs/formal"),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "contract_verification_path": str(verification),
        "contract_verification_sha256": verification_sha,
        "full_training_contract_path": str(contract),
        "full_training_contract_sha256": contract_sha,
        "supervisor_path": str(SUPERVISOR),
        "supervisor_sha256": digest(SUPERVISOR),
        "archiver_path": str(ROOT / "cloud/huawei/archive_verified_checkpoints_formal500k.sh"),
        "archiver_sha256": digest(
            ROOT / "cloud/huawei/archive_verified_checkpoints_formal500k.sh"
        ),
        "validator_path": str(ROOT / "scripts/validate_training_checkpoint.py"),
        "validator_sha256": digest(ROOT / "scripts/validate_training_checkpoint.py"),
        "empty_prefix_validator_path": str(ROOT / "scripts/verify_obsutil_empty_prefix.py"),
        "empty_prefix_validator_sha256": digest(
            ROOT / "scripts/verify_obsutil_empty_prefix.py"
        ),
        "trainer_path": str(ROOT / "scripts/train_ddp.py"),
        "trainer_sha256": digest(ROOT / "scripts/train_ddp.py"),
        "soft_stop_minutes": 8970,
        "hard_stop_minutes": 9000,
        "archive_kill_grace_seconds": 600,
        "archive_poll_seconds": 60,
        "toolchain": {
            "python": tool(python, ["--version"]),
            "torchrun": tool(benign_tool, ["--version"]),
            "obsutil": tool(benign_tool, ["--version"]),
        },
    }
    checkpoint_plan = {
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
    stop_plan = {
        "soft_stop_minutes": 8970,
        "hard_stop_minutes": 9000,
        "archive_kill_grace_seconds": 600,
        "soft_stop_mechanism": (
            "sealed_attempt_sentinel_at_optimizer_boundary"
        ),
        "soft_stop_precedes_hard_stop": True,
        "archive_failure_soft_stop": True,
    }
    recovery_plan = {
        "separate_attempt_required": True,
        "max_recovery_attempts": 16,
        "resume_history_required": True,
        "strict_checkpoint_preflight": True,
        "latest_verified_only": True,
        "resume_semantics": "state_consistent_non_bitwise_crop_and_noise",
    }
    package = tmp_path / "package.json"
    package.write_text(
        json.dumps(
            {
                "package_id": "formal500k-test",
                "package_ready": True,
                "formal_training_authorized": False,
                "formal_candidate": {
                    "config_sha256": config_sha,
                    "execution_plan": execution_plan,
                },
                "runtime_identity": {
                    "hostname": "deepjump-v100-8gpu-20260716",
                    "product_uuid": "4c2273f2-4763-4827-839b-27d2c79cd76a",
                    "product_serial": "4c2273f2-4763-4827-839b-27d2c79cd76a",
                    "gpu_model": "Tesla V100-SXM2-16GB",
                },
                "checkpoint_plan": checkpoint_plan,
                "stop_plan": stop_plan,
                "recovery_plan": recovery_plan,
                "estimate_budget": {"hard_cap_hours": 150.0},
            }
        )
    )
    package_sha = digest(package)
    scope = {
        "formal_run_may_start": True,
        "external_or_untouched_access_authorized": False,
    }
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": "deepjump.formal500k.user_authorization.v1",
                "status": "USER_AUTHORIZED_FORMAL_TRAINING",
                "formal_training_authorized": True,
                "authorization_id": "test-auth",
                "issued_at": "2026-07-26T00:00:00Z",
                "expires_at": "2026-08-01T06:00:00Z",
                "authorized_package_sha256": package_sha,
                "authorized_package_id": "formal500k-test",
                "authorized_run_id": run_id,
                "authorized_reviewed_commit": commit,
                "scope": scope,
                "approval_record_sha256": "a" * 64,
            }
        )
    )
    authorization_sha = digest(authorization)
    authorized = tmp_path / "authorization-verification.json"
    authorized.write_text(
        json.dumps(
            {
                "schema": "deepjump.formal500k.authorization_verification.v1",
                "status": "PASS_FORMAL_TRAINING_AUTHORIZATION",
                "formal_training_authorized": True,
                "package_sha256": package_sha,
                "config_sha256": config_sha,
                "target_total_steps": 500000,
                "authorization_sha256": authorization_sha,
                "authorization_verifier_sha256": digest(
                    ROOT / "scripts/verify_formal500k_authorization.py"
                ),
                "package_id": "formal500k-test",
                "run_id": run_id,
                "reviewed_commit": commit,
                "execution_plan": execution_plan,
                "checkpoint_plan": checkpoint_plan,
                "stop_plan": stop_plan,
                "recovery_plan": recovery_plan,
                "scope": scope,
            }
        )
    )
    result = subprocess.run(
        [
            str(SUPERVISOR),
            "fresh",
            str(package),
            package_sha,
            str(authorization),
            authorization_sha,
            str(authorized),
            digest(authorized),
        ],
        env={
            **os.environ,
            "BASH_ENV": str(tmp_path / "must-not-run"),
            "PYTHONPATH": str(tmp_path / "must-not-be-inherited"),
            "LD_PRELOAD": str(tmp_path / "must-not-be-inherited.so"),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "formal package/config/authorization preflight failed" not in result.stderr
    assert "exact formal package, authorization, and execution-plan preflight PASS" in result.stdout
