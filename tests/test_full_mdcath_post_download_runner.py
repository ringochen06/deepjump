import os
import subprocess
from pathlib import Path


RUNNER = Path("cloud/huawei/run_full_mdcath_post_download_gate.sh")
EXPANDED_RUNNER = Path("cloud/huawei/run_contracted_expanded_data_gate.sh")


def test_post_download_runner_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_post_download_runner_orders_fail_closed_qualification_before_handoff():
    text = RUNNER.read_text()
    markers = [
        "gate=live_rehash_audit",
        "gate=heldout_partition",
        "gate=full_training_contract",
        "gate=obs_qualification_archive",
        "gate=obs_corpus_upload",
        "gate=obs_corpus_content_readback",
        "gate=obs_readback_journal_archive",
        "gate=sync_and_read_only_mount",
        "gate=completion_marker_readback",
        "gate=handoff_contracted_expanded_data",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert text.count(
        'exec "$REPO/cloud/huawei/run_contracted_expanded_data_gate.sh"'
    ) == 1
    assert text.rindex("run_contracted_expanded_data_gate.sh") > text.index(
        "gate=completion_marker_readback"
    )


def test_post_download_runner_pins_data_identity_and_mutation_quiescence():
    text = RUNNER.read_text()
    for required in (
        "EXPECTED_H5_FILES=5398",
        "EXPECTED_H5_BYTES=3613998101757",
        "EXPECTED_TRAJECTORIES=134950",
        "EXPECTED_SOURCE_REVISION=5e3ed8aec62b689e01751db16275fdcdbc39e47f",
        "EXPECTED_SOURCE_INVENTORY_SHA256=2e6e3602",
        "deepjump-mdcath-download.service deepjump-mdcath-hash.service",
        "deepjump-mdcath-copy.service",
        "git status --porcelain=v1 --untracked-files=all",
        "verify_source_identity",
        "--rehash-payloads",
        "--panel-registry",
        "--expected-panel-registry-sha256",
        "build_full_training_data_contract.py",
        'DATA_MOUNT" == "$EXPECTED_DATA_MOUNT"',
        '"$DATA_MOUNT" != /',
    ):
        assert required in text


def test_post_download_runner_proves_obs_bytes_mount_and_independent_completion():
    text = RUNNER.read_text()
    for required in (
        "verify_obsutil_empty_prefix.py",
        '"$CONTRACT_DIR" "$OBS_QUALIFICATION_DST"',
        '"$DATA_ROOT/data" "$OBS_CORPUS_DST"',
        "prefix_file_inventory",
        "-limit=0 -bf=raw",
        '"$EXPECTED_H5_FILES" "$EXPECTED_H5_BYTES"',
        "verify_full_mdcath_obs_readback.py",
        "PASS_FULL_MDCATH_POST_DOWNLOAD_QUALIFICATION",
        "corpus_readback_completion_sha256",
        "corpus_readback_journal_inventory_sha256",
        '"$READBACK_JOURNAL" "$OBS_DST/corpus_readback_journal"',
        "obs_readback_journal_sha256.txt",
        'mount -o remount,ro "$DATA_MOUNT"',
        "qualified staging mount did not become read-only",
        'obsutil cp "$COMPLETION_MARKER"',
        'obsutil cp "$OBS_DST/completion/post_download_completion.json"',
        'cmp "$COMPLETION_MARKER" "$COMPLETION_READBACK"',
        "formal_training_authorized\": False",
        "RECOVERY_PREVIOUS_COMMIT",
        '"obs_upload_commit": obs_upload_commit',
        "OBS ownership marker recovery identity mismatch",
        "tracked_source_sha256.$actual_commit.txt",
        "verify_expanded_data_partition_recovery.py",
    ):
        assert required in text
    assert 'obsutil ls "$OBS_CORPUS_DST/" -r' not in text


def test_post_download_runner_hard_stop_failure_archive_and_recovery():
    text = RUNNER.read_text()
    assert "HARD_STOP_MINUTES must remain 4320" in text
    assert "/usr/bin/systemctl poweroff" in text
    assert "DATA_REMOUNTED_RO=1" in text
    assert "completion marker resume identity mismatch" in text
    trap_body = text[text.index("shutdown_on_exit()") : text.index("trap shutdown_on_exit")]
    assert 'mount -o remount,rw "$DATA_MOUNT"' in trap_body
    assert "shutdown -h now" in trap_body
    for required in (
        "failure_status.json",
        "failure_archive_persistence.txt",
        "FAILURE_ARCHIVE_LOCAL_ONLY_OBS_NOT_SAFELY_BOUND",
        "PASS_OBS_FAILURE_ARCHIVE_READBACK",
        'obsutil cp "$failure_archive"',
        "sha256sum --check",
    ):
        assert required in trap_body
    assert "full_payload_rehash_journal_v1" not in text


def test_exec_failure_keeps_parent_trap_until_process_replacement():
    text = RUNNER.read_text()
    handoff = text[text.index("gate=handoff_contracted_expanded_data") :]
    assert "trap - EXIT" not in handoff
    assert "HANDED_OFF" not in handoff
    assert "shopt -s execfail" in handoff
    exec_position = handoff.index(
        'exec "$REPO/cloud/huawei/run_contracted_expanded_data_gate.sh"'
    )
    assert handoff.index("handoff_code=$?", exec_position) > exec_position
    assert handoff.index("trap shutdown_on_exit EXIT", exec_position) > exec_position
    assert handoff.index('exit "$handoff_code"', exec_position) > exec_position


def _failure_function(text: str) -> str:
    start = text.index("shutdown_on_exit()")
    end = text.index("\ntrap shutdown_on_exit EXIT", start)
    return text[start:end]


def test_failure_trap_actually_creates_atomic_local_archive_when_obs_unbound(tmp_path):
    text = RUNNER.read_text()
    run_dir = tmp_path / "run"
    run_id = "20990101T010101Z"
    harness = tmp_path / "failure-harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\nset -u\n"
        "DATA_REMOUNTED_RO=0\nOBS_BOUND=0\n"
        "sudo() { return 0; }\n"
        + _failure_function(text)
        + f"\nRUN_DIR={run_dir!s}\nRUN_ID={run_id}\nfalse\nshutdown_on_exit\n"
    )
    before = set(Path("/var/tmp").glob(f"deepjump-full-mdcath-post-download-failure-{run_id}-*.tar.gz"))
    result = subprocess.run(["bash", str(harness)], capture_output=True, text=True)
    assert result.returncode == 1
    created = set(Path("/var/tmp").glob(f"deepjump-full-mdcath-post-download-failure-{run_id}-*.tar.gz")) - before
    try:
        assert len(created) == 1
        archive = created.pop()
        assert Path(str(archive) + ".sha256").is_file()
        assert (run_dir / "failure_status.json").is_file()
        persistence = (run_dir / "failure_archive_persistence.txt").read_text()
        assert "LOCAL_ONLY_OBS_NOT_SAFELY_BOUND" in persistence
    finally:
        for archive in created | (set(Path("/var/tmp").glob(
            f"deepjump-full-mdcath-post-download-failure-{run_id}-*.tar.gz"
        )) - before):
            archive.unlink(missing_ok=True)
            Path(str(archive) + ".sha256").unlink(missing_ok=True)


def test_exec_system_call_failure_runs_parent_failure_trap(tmp_path):
    text = RUNNER.read_text()
    repo = tmp_path / "repo"
    runner = repo / "cloud/huawei/run_contracted_expanded_data_gate.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/definitely/missing/interpreter\n")
    runner.chmod(0o700)
    run_dir = tmp_path / "run"
    run_id = "20990101T020202Z"
    handoff_start = text.index("export REPO PYTHON EXPECTED_REPO_COMMIT")
    handoff = text[handoff_start:]
    harness = tmp_path / "exec-failure-harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\nset -u\n"
        "DATA_REMOUNTED_RO=0\nOBS_BOUND=0\n"
        "sudo() { return 0; }\n"
        + _failure_function(text)
        + "\ntrap shutdown_on_exit EXIT\n"
        + f"RUN_DIR={run_dir!s}\nRUN_ID={run_id}\nREPO={repo!s}\n"
        + "PYTHON=/bin/false\nEXPECTED_REPO_COMMIT=" + "a" * 40 + "\n"
        + "EXPECTED_HOSTNAME=test\nDATA_ROOT=/data-full/mdcath\n"
        + "CONTRACT=/contract\nCONTRACT_SHA256=" + "b" * 64 + "\n"
        + "BUCKET=obs://bucket\nCONTRACT_DIR=/contract-dir\n"
        + handoff
    )
    before = set(Path("/var/tmp").glob(f"deepjump-full-mdcath-post-download-failure-{run_id}-*.tar.gz"))
    result = subprocess.run(["bash", str(harness)], capture_output=True, text=True)
    assert result.returncode != 0
    created = set(Path("/var/tmp").glob(f"deepjump-full-mdcath-post-download-failure-{run_id}-*.tar.gz")) - before
    try:
        assert len(created) == 1
        assert (run_dir / "failure_status.json").is_file()
    finally:
        for archive in created:
            archive.unlink(missing_ok=True)
            Path(str(archive) + ".sha256").unlink(missing_ok=True)


def test_expanded_runner_requires_hash_service_inactive():
    text = EXPANDED_RUNNER.read_text()
    assert "deepjump-mdcath-download.service deepjump-mdcath-hash.service" in text
    assert "deepjump-mdcath-copy.service" in text
