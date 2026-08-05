import subprocess
from pathlib import Path


RUNNER = Path("cloud/huawei/run_contracted_expanded_data_gate.sh")
SUMMARIZER = Path("scripts/summarize_contracted_expanded_data_gate.py")


def test_contracted_expanded_data_runner_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_contracted_expanded_data_runner_is_development_only():
    text = RUNNER.read_text()
    assert text.count(
        '"$PYTHON" scripts/contracted_guarded_endpoint_panel_eval.py'
    ) == 1
    assert text.count(
        '"$PYTHON" -m scripts.adjudicate_contracted_guarded_endpoint_panel'
    ) == 1
    assert "--phase development" in text
    assert "--phase external" not in text
    assert "--phase untouched" not in text
    assert "scripts/guarded_endpoint_panel_eval.py" not in text.replace(
        "scripts/contracted_guarded_endpoint_panel_eval.py", ""
    )
    assert "formal_training_authorized\": False" in text
    assert '"external_evaluation_started": False' in SUMMARIZER.read_text()


def test_contracted_expanded_data_runner_orders_preregistered_gates():
    text = RUNNER.read_text()
    markers = [
        "gate=contract_verify",
        "gate=sealed_configs",
        'run_stage eight_gpu_smoke',
        'run_stage short_calibration',
        'run_stage fresh_development2000',
        "gate=development20",
        "gate=development_adjudication",
        "gate=obs_archive_and_readback",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert text.rindex("summarize_contracted_expanded_data_gate.py") > text.index(
        "gate=development_adjudication"
    )
    assert "ADVANCE_EXPANDED_DATA_EXTERNAL" in SUMMARIZER.read_text()


def test_contracted_expanded_data_runner_seals_training_identity():
    text = RUNNER.read_text()
    for required in (
        "cfg.data.domains_file = train_list",
        "cfg.data.full_training_contract = contract",
        "cfg.data.full_training_contract_sha256 = contract_sha256",
        "training_semantics_sha256(smoke)",
        "training_semantics_sha256(calibration)",
        "training_semantics_sha256(development)",
        "configs/v100_tensorcloud01_full_expanded_d1_smoke100.yaml",
        "configs/v100_tensorcloud01_full_expanded_d1_calibration1000.yaml",
        "configs/v100_tensorcloud01_full_expanded_d1_development2000.yaml",
        "--expected-lr-horizon-steps 500000",
        "sealed config did not round-trip exactly",
        "--full-training-contract",
        "--expected-full-training-contract-sha256",
        "--require-full-tensor",
        "--expected-config",
        "--expected-contract-verification",
        "--expected-contract-verification-sha256",
    ):
        assert required in text


def test_contracted_expanded_data_runner_is_fail_closed_and_archived():
    text = RUNNER.read_text()
    for required in (
        "set -euo pipefail",
        "systemd-run --quiet",
        "HARD_STOP_MINUTES must remain 480",
        "/usr/bin/systemctl poweroff",
        "SHUTDOWN_ON_EXIT must be 1",
        "export PYTHONNOUSERSITE=1",
        'export PYTHONPATH="$REPO:$REPO/src"',
        "git rev-parse HEAD",
        "git status --porcelain=v1 --untracked-files=all",
        "qualified DATA_ROOT mount must be read-only",
        "GPU count %s != 8",
        "verify_obsutil_empty_prefix.py",
        "obsutil sync",
        "sha256sum -c audit_sha256.txt",
        "readback_completion.sha256",
        "tracked_source_sha256.txt",
        "verify_source_identity",
        "summarize_contracted_expanded_data_gate.py",
        "verify_audit_readback.py",
        "DEVELOPMENT_DECISION_SHA256",
        "decision_sha256",
    ):
        assert required in text


def test_contracted_expanded_data_runner_binds_runtime_probe_to_adjudication():
    text = RUNNER.read_text()
    assert '--runtime-probe-output "$RUN_DIR/evidence/development_runtime_probe.json"' in text
    assert '--runtime-probe "$DEVELOPMENT_RUNTIME_PROBE"' in text
    assert '--expected-runtime-probe-sha256 "$DEVELOPMENT_RUNTIME_PROBE_SHA256"' in text
    assert text.count("smoke, smoke_report = seal(") == 1
    assert text.count("--require-full-tensor") == 3
    assert text.count("--expected-config") == 3
    assert text.count("--expected-contract-verification ") == 3
    assert text.count("sha256sum -c audit_sha256.txt") == 2
    assert text.count("scripts/verify_audit_readback.py") >= 3
    assert text.count("verify_source_identity") >= 10


def test_contracted_expanded_data_runner_binds_all_checkpoint_gate_bytes():
    text = RUNNER.read_text()
    assert text.count("--expected-checkpoint-sha256") >= 5
    for stage in ("SMOKE", "CALIBRATION", "DEVELOPMENT"):
        assert f'{stage}_CHECKPOINT_SHA256=$(sha256sum "${stage}_CHECKPOINT"' in text
        assert f'{stage}_CHECKPOINT_GATE_SHA256=$(sha256sum "${stage}_CHECKPOINT_GATE"' in text
        assert f'--{stage.lower()}-checkpoint "${stage}_CHECKPOINT"' in text
    assert "tests/test_contracted_expanded_data_gate_hardening.py" in text
    assert text.count("verify_checkpoint_identities") >= 4
    assert text.count("verify_readback_checkpoint_identities") >= 3
    assert "verify_checkpoint_chain()" in text
    assert "from deepjump.data_contract import _read_regular_bytes" in text
    assert text.count("verify_checkpoint_chain \"") >= 4


def test_contracted_expanded_data_runner_archives_failure_manifest_and_readback():
    text = RUNNER.read_text()
    trap_body = text[text.index("shutdown_on_exit()") : text.index("trap shutdown_on_exit")]
    for required in (
        "failure_sha256.txt",
        "PASS_OBS_FAILURE_READBACK",
        "failure_readback_status.json",
        'obsutil sync "$RUN_DIR" "$OBS_DST/failure/audit"',
        "sha256sum -c failure_sha256.txt",
        'sudo -n shutdown -h now',
    ):
        assert required in trap_body


def test_contracted_expanded_data_runner_seals_run_and_completion_identity():
    text = RUNNER.read_text()
    for required in (
        '"run_binding": {',
        '"run_id": run_id',
        '"commit": commit',
        '"obs": obs',
        '"source_identity_manifest_sha256": source_sha',
        '"checkpoint_gates": {',
        '"smoke": {',
        '"calibration": {',
        '"development": {',
    ):
        assert required in text
    assert text.count('"checkpoint_gates": {') >= 2


def test_contracted_expanded_data_runner_uses_fresh_step2000_checkpoint():
    text = RUNNER.read_text()
    assert 'run_stage fresh_development2000 "$RUN_DIR/configs/development.yaml" 60' in text
    assert '--resume' not in text
    assert 'DEVELOPMENT_CHECKPOINT="$RUN_DIR/stages/development/last.ckpt"' in text
    assert text.count("--expected-checkpoint-step 2000") == 2
    assert 'expected_checkpoint_step=2_000' in text
    assert '"checkpoint_step": 2_000' in text


def test_contracted_expanded_data_runner_issues_ledger_bound_v2_authorization():
    text = RUNNER.read_text()
    for required in (
        '"schema": "deepjump.reserved_evaluation_authorization.v2"',
        '"authorization_id": f"development-{run_id}"',
        '"consumption_ledger_root": str(Path(ledger_root).resolve())',
        "ordinary obsutil copy/sync cannot provide that",
    ):
        assert required in text
    assert "--consumption-ledger" not in text
