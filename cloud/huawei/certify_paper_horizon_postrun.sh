#!/usr/bin/env bash
# Independently re-download and certify the exact dbbc86d paper-horizon run.
set -euo pipefail

REPO=${REPO:-/data/deepjump}
PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:-}
HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}
EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?set strict verifier commit SHA}
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:?set authorized GPU hostname}
BUCKET=${BUCKET:?set BUCKET=obs://your-bucket-name}

SOURCE_RUN_ID=20260722T012922Z
SOURCE_COMMIT=dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b
SOURCE_OBS="$BUCKET/deepjump-calibration/paper-horizon-ab2000/$SOURCE_RUN_ID"
CERT_OBS="$BUCKET/deepjump-certification/paper-horizon-ab2000/$SOURCE_RUN_ID/$EXPECTED_REPO_COMMIT"
WORK_ROOT="/data/deepjump-postrun-certification/$SOURCE_RUN_ID/$EXPECTED_REPO_COMMIT"
READBACK_ONE="$WORK_ROOT/readback_one"
READBACK_TWO="$WORK_ROOT/readback_two"
CERT_DIR="$WORK_ROOT/certificate"
CERT_READBACK="$WORK_ROOT/certificate_readback"
HARD_STOP_UNIT="deepjump-paper-horizon-postcert-$(date -u +%Y%m%dT%H%M%SZ)-$$"

shutdown_on_exit() {
  code=$?
  shutdown_code=0
  trap - EXIT
  sudo -n shutdown -h now || shutdown_code=$?
  if [[ "$shutdown_code" != 0 && "$code" == 0 ]]; then
    code=$shutdown_code
  fi
  exit "$code"
}
trap shutdown_on_exit EXIT

[[ "$SHUTDOWN_ON_EXIT" == 1 ]] || { printf 'SHUTDOWN_ON_EXIT must be 1\n' >&2; exit 2; }
[[ "$HARD_STOP_MINUTES" == 45 ]] || { printf 'HARD_STOP_MINUTES must be 45\n' >&2; exit 2; }
[[ "$BUCKET" == "obs://deepjump-mdcath-cn4-ringochen" ]] || {
  printf 'unexpected OBS bucket\n' >&2; exit 2;
}
sudo -n systemd-run --quiet --unit="$HARD_STOP_UNIT" \
  --on-active="${HARD_STOP_MINUTES}m" /usr/bin/systemctl poweroff
sudo -n systemctl is-active --quiet "$HARD_STOP_UNIT.timer"
sudo -n systemctl show "$HARD_STOP_UNIT.service" --property=ExecStart --no-pager \
  | grep -Fq '/usr/bin/systemctl poweroff'
sudo -n shutdown -c 2>/dev/null || true

[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]] || { printf 'hostname mismatch\n' >&2; exit 2; }
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_REPO_COMMIT" ]] || {
  printf 'verifier commit mismatch\n' >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { printf 'verifier worktree dirty\n' >&2; exit 2; }
[[ ! -e "$WORK_ROOT" ]] || { printf 'refusing to overwrite certification work root\n' >&2; exit 2; }
command -v obsutil >/dev/null
[[ -x "$PYTHON" ]]
if pgrep -af '[s]cripts/(train_ddp|guarded_endpoint_panel_eval|external_endpoint_panel_eval).py'; then
  printf 'training or evaluation process is active\n' >&2
  exit 2
fi
mkdir -p "$WORK_ROOT"
timeout 30s obsutil ls "$CERT_OBS/" -limit=1 \
  | tee "$WORK_ROOT/cert_prefix_preflight.log"
"$PYTHON" scripts/verify_obsutil_empty_prefix.py \
  "$WORK_ROOT/cert_prefix_preflight.log"

download_readback() {
  target=$1
  mkdir -p "$target/baseline" "$target/candidate" "$target/audit"
  timeout --signal=TERM --kill-after=30s 8m \
    obsutil sync "$SOURCE_OBS/baseline" "$target/baseline"
  timeout --signal=TERM --kill-after=30s 8m \
    obsutil sync "$SOURCE_OBS/candidate" "$target/candidate"
  timeout --signal=TERM --kill-after=30s 8m \
    obsutil sync "$SOURCE_OBS/audit" "$target/audit"
}
download_readback "$READBACK_ONE"
download_readback "$READBACK_TWO"

mkdir -p "$CERT_DIR"
cp "$WORK_ROOT/cert_prefix_preflight.log" "$CERT_DIR/cert_prefix_preflight.log"
"$PYTHON" scripts/certify_paper_horizon_postrun.py \
  --readback-one "$READBACK_ONE" \
  --readback-two "$READBACK_TWO" \
  --repo "$REPO" \
  --expected-run-id "$SOURCE_RUN_ID" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-verifier-commit "$EXPECTED_REPO_COMMIT" \
  --expected-obs "$SOURCE_OBS" \
  --certification-obs "$CERT_OBS" \
  --output "$CERT_DIR/certification.json"
(cd "$CERT_DIR" && sha256sum certification.json cert_prefix_preflight.log \
  > certification.sha256)
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$CERT_DIR" "$CERT_OBS"
mkdir -p "$CERT_READBACK"
timeout --signal=TERM --kill-after=30s 4m obsutil sync "$CERT_OBS" "$CERT_READBACK"
(cd "$CERT_READBACK" && sha256sum -c "$CERT_DIR/certification.sha256")
printf 'Paper-horizon post-run certification complete; seed1 was not started.\n'
