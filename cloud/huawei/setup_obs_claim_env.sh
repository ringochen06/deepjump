#!/usr/bin/env bash
# Provision the isolated, pinned OBS SDK used only for atomic external-panel claims.
set -euo pipefail

PYTHON=${PYTHON:-/data/venvs/deepjump/bin/python}
TARGET=${TARGET:-/data/venvs/obs-claim}
EXPECTED_VERSION=3.26.6

[[ -x "$PYTHON" ]] || {
  printf 'base Python runtime is missing: %s\n' "$PYTHON" >&2
  exit 2
}
[[ ! -e "$TARGET" ]] || {
  printf 'refusing to overwrite OBS claim environment: %s\n' "$TARGET" >&2
  exit 2
}

"$PYTHON" -m venv "$TARGET"
"$TARGET/bin/python" -m pip install --no-cache-dir \
  "esdk-obs-python==$EXPECTED_VERSION"
"$TARGET/bin/python" - <<'PY'
import importlib.metadata
import obs

assert importlib.metadata.version("esdk-obs-python") == "3.26.6"
assert hasattr(obs, "ObsClient")
assert hasattr(obs, "AppendObjectContent")
PY
printf 'OBS claim environment ready: %s (%s)\n' "$TARGET" "$EXPECTED_VERSION"
