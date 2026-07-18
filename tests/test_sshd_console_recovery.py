from pathlib import Path


RECOVERY = Path("cloud/huawei/recover_sshd_from_console.sh")
RUNNER = Path("cloud/huawei/run_full_tensor_tiny_domain_overfit5000.sh")


def test_console_recovery_is_root_only_bounded_and_does_not_touch_credentials():
    script = RECOVERY.read_text()

    assert "RECOVERY_HARD_STOP_MINUTES=${RECOVERY_HARD_STOP_MINUTES:-20}" in script
    assert '[[ "$RECOVERY_HARD_STOP_MINUTES" == 20 ]]' in script
    assert "run from an already-authenticated root console" in script
    assert script.index('shutdown -h "+$RECOVERY_HARD_STOP_MINUTES"') < script.index(
        "systemctl status ssh"
    )
    assert '[[ "${APPLY_SSH_REPAIR:-0}" == 1 ]]' in script
    assert "systemctl restart ssh" in script
    assert "SSH_SERVICE_READY" in script

    forbidden = (
        "passwd ",
        "chpasswd",
        "authorized_keys >",
        "authorized_keys <<",
        "ssh-keygen",
        "private key",
    )
    assert not any(token in script for token in forbidden)


def test_console_recovery_preserves_public_key_only_root_login():
    script = RECOVERY.read_text()

    assert "passwordauthentication no" in script
    assert "kbdinteractiveauthentication no" in script
    assert "pubkeyauthentication yes" in script
    assert "prohibit-password|without-password" in script
    assert "sshd configuration validation failed; service was not restarted" in script


def test_tiny_domain_runner_replaces_recovery_cutoff_fail_closed():
    runner = RUNNER.read_text()

    trap_index = runner.index("trap shutdown_on_exit EXIT")
    cancel_index = runner.index("sudo -n shutdown -c 2>/dev/null || true", trap_index)
    arm_index = runner.index('sudo -n shutdown -h "+$HARD_STOP_MINUTES"', cancel_index)
    required_inputs_index = runner.index("EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?")
    assert trap_index < cancel_index < arm_index < required_inputs_index
