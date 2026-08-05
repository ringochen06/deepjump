#!/bin/sh
# Exact virtual-environment Python launcher for the reviewed formal500k package.
#
# The venv entrypoint is intentionally invoked by its venv path so CPython
# discovers pyvenv.cfg. Its complete symlink chain and underlying interpreter
# are verified before every execution.

set -eu

VENV_ROOT=/data/venvs/deepjump
PYTHON_LAUNCHER=$VENV_ROOT/bin/python
PYTHON3_LAUNCHER=$VENV_ROOT/bin/python3
SYSTEM_PYTHON_LINK=/usr/bin/python3
SYSTEM_PYTHON=/usr/bin/python3.10
EXPECTED_SYSTEM_PYTHON_SHA256=7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86
EXPECTED_PYVENV_SHA256=db6c8a96f25493eda9f74f23f0b5f248a8b50a5b469b15c5ee7313875b416364

[ "$(readlink "$PYTHON_LAUNCHER")" = python3 ] ||
  { echo "formal500k Python launcher link drift" >&2; exit 70; }
[ "$(readlink "$PYTHON3_LAUNCHER")" = /usr/bin/python3 ] ||
  { echo "formal500k Python3 launcher link drift" >&2; exit 70; }
[ "$(readlink "$SYSTEM_PYTHON_LINK")" = python3.10 ] ||
  { echo "formal500k system Python link drift" >&2; exit 70; }
[ "$(readlink -f "$PYTHON_LAUNCHER")" = "$SYSTEM_PYTHON" ] ||
  { echo "formal500k resolved Python path drift" >&2; exit 70; }
[ "$(sha256sum "$SYSTEM_PYTHON" | awk '{print $1}')" = "$EXPECTED_SYSTEM_PYTHON_SHA256" ] ||
  { echo "formal500k system Python SHA256 drift" >&2; exit 70; }
[ "$(sha256sum "$VENV_ROOT/pyvenv.cfg" | awk '{print $1}')" = "$EXPECTED_PYVENV_SHA256" ] ||
  { echo "formal500k pyvenv.cfg SHA256 drift" >&2; exit 70; }

if [ "${1-}" = "--deepjump-toolchain-version" ]; then
  exec "$PYTHON_LAUNCHER" -c \
    'import json, sys, torch; print(json.dumps({"cuda": torch.version.cuda, "python": sys.version.split()[0], "sys_prefix": sys.prefix, "torch": torch.__version__}, sort_keys=True, separators=(",", ":")))'
fi

exec "$PYTHON_LAUNCHER" "$@"
