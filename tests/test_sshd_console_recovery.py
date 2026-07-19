from pathlib import Path
import subprocess


RECOVERY = Path("cloud/huawei/recover_sshd_from_console.sh")
RUNNER = Path("cloud/huawei/run_full_tensor_tiny_domain_overfit5000.sh")


def test_console_recovery_is_root_only_bounded_and_does_not_touch_credentials():
    script = RECOVERY.read_text()

    assert "RECOVERY_HARD_STOP_MINUTES=${RECOVERY_HARD_STOP_MINUTES:-20}" in script
    assert '[[ "$RECOVERY_HARD_STOP_MINUTES" == 20 ]]' in script
    assert "run from an already-authenticated root console" in script
    assert script.index("systemd-run --quiet") < script.index("systemctl status ssh")
    assert '--on-active="${RECOVERY_HARD_STOP_MINUTES}m"' in script
    assert 'systemctl is-active --quiet "$RECOVERY_HARD_STOP_UNIT.timer"' in script
    assert "shutdown -c" not in script
    assert '[[ "${APPLY_SSH_REPAIR:-0}" == 1 ]]' in script
    assert "systemctl restart ssh" in script
    assert "SSH_SERVICE_READY" in script

    forbidden = (
        "passwd",
        "chpasswd",
        "ssh-keygen",
        "private key",
        "base64",
        "openssl",
    )
    assert not any(token in script for token in forbidden)
    authorized_key_lines = [
        line.strip() for line in script.splitlines() if "authorized_keys" in line
    ]
    assert authorized_key_lines == [
        "[[ -s /root/.ssh/authorized_keys ]] || {",
        "printf 'ERROR: /root/.ssh/authorized_keys is missing or empty\\n' >&2",
    ]


def test_console_recovery_preserves_public_key_only_root_login():
    script = RECOVERY.read_text()

    assert "passwordauthentication no" in script
    assert "kbdinteractiveauthentication no" in script
    assert "pubkeyauthentication yes" in script
    assert "prohibit-password|without-password" in script
    assert "sshd configuration validation failed; service was not restarted" in script


def test_scripts_are_valid_bash():
    for path in (RECOVERY, RUNNER):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_tiny_domain_runner_hands_off_recovery_cutoff_without_a_gap():
    runner = RUNNER.read_text()

    trap_index = runner.index("trap shutdown_on_exit EXIT")
    arm_index = runner.index("sudo -n systemd-run --quiet", trap_index)
    verify_index = runner.index(
        'sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"', arm_index
    )
    release_index = runner.index(
        "sudo -n systemctl stop 'deepjump-recovery-hard-stop-*.timer'", verify_index
    )
    legacy_cancel_index = runner.index(
        "sudo -n shutdown -c 2>/dev/null || true", release_index
    )
    required_inputs_index = runner.index("EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?")
    assert (
        trap_index
        < arm_index
        < verify_index
        < release_index
        < legacy_cancel_index
        < required_inputs_index
    )
    trap_body = runner[runner.index("shutdown_on_exit()") : trap_index]
    assert "shutdown -c" not in trap_body
    assert "sudo -n shutdown -h now" in trap_body
