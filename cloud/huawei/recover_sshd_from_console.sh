#!/usr/bin/env bash
# Audit or restart sshd from an already-authenticated root console session.
set -euo pipefail

MODE=${1:-audit}
RECOVERY_HARD_STOP_MINUTES=${RECOVERY_HARD_STOP_MINUTES:-20}
SSHD=${SSHD:-/usr/sbin/sshd}

[[ "$MODE" == audit || "$MODE" == repair ]] || {
  printf 'usage: %s [audit|repair]\n' "$0" >&2
  exit 2
}
[[ "$RECOVERY_HARD_STOP_MINUTES" == 20 ]] || {
  printf 'RECOVERY_HARD_STOP_MINUTES must remain 20\n' >&2
  exit 2
}
[[ "${EUID:-$(id -u)}" == 0 ]] || {
  printf 'run from an already-authenticated root console\n' >&2
  exit 2
}
[[ -x "$SSHD" ]] || {
  printf 'sshd executable not found at %s\n' "$SSHD" >&2
  exit 2
}

# Arm a short independent cutoff before inspecting or restarting the service.
shutdown -c 2>/dev/null || true
shutdown -h "+$RECOVERY_HARD_STOP_MINUTES"

printf 'recovery_mode=%s started_at=%s hard_stop_minutes=%s\n' \
  "$MODE" "$(date -Is)" "$RECOVERY_HARD_STOP_MINUTES"
hostname
uptime
systemctl is-active ssh || true
systemctl status ssh --no-pager -l || true
ss -ltnp 'sport = :22' || true
journalctl -u ssh -b --no-pager -n 120 || true

if ! "$SSHD" -t; then
  printf 'ERROR: sshd configuration validation failed; service was not restarted\n' >&2
  exit 2
fi

effective_config=$(
  "$SSHD" -T | awk '
    $1 == "port" ||
    $1 == "listenaddress" ||
    $1 == "passwordauthentication" ||
    $1 == "kbdinteractiveauthentication" ||
    $1 == "permitrootlogin" ||
    $1 == "pubkeyauthentication" ||
    $1 == "usedns" ||
    $1 == "maxstartups" { print }
  '
)
printf '%s\n' "$effective_config"
grep -qx 'passwordauthentication no' <<<"$effective_config"
grep -qx 'kbdinteractiveauthentication no' <<<"$effective_config"
grep -qx 'pubkeyauthentication yes' <<<"$effective_config"
grep -Eq '^permitrootlogin (prohibit-password|without-password)$' <<<"$effective_config"
[[ -s /root/.ssh/authorized_keys ]] || {
  printf 'ERROR: /root/.ssh/authorized_keys is missing or empty\n' >&2
  exit 2
}

if [[ "$MODE" == repair ]]; then
  [[ "${APPLY_SSH_REPAIR:-0}" == 1 ]] || {
    printf 'set APPLY_SSH_REPAIR=1 to authorize the bounded service restart\n' >&2
    exit 2
  }
  install -d -m 0755 /run/sshd
  systemctl reset-failed ssh
  systemctl restart ssh
fi

systemctl is-active --quiet ssh
ss -ltnp 'sport = :22' | grep -q ':22'
printf 'SSH_SERVICE_READY mode=%s completed_at=%s\n' "$MODE" "$(date -Is)"
printf 'Keep this console open while Codex verifies the remote banner and public-key login.\n'
printf 'Do not cancel the 20-minute cutoff until the experiment runner replaces it.\n'
