import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "cloud/huawei/formal500k_python.sh"


def test_launcher_binds_exact_virtual_environment_chain():
    text = LAUNCHER.read_text()
    assert "VENV_ROOT=/data/venvs/deepjump" in text
    assert "PYTHON_LAUNCHER=$VENV_ROOT/bin/python" in text
    assert "EXPECTED_SYSTEM_PYTHON_SHA256=" in text
    assert "EXPECTED_PYVENV_SHA256=" in text
    assert 'readlink -f "$PYTHON_LAUNCHER"' in text
    assert 'exec "$PYTHON_LAUNCHER" "$@"' in text
    assert "--deepjump-toolchain-version" in text


@pytest.mark.skipif(
    not Path("/data/venvs/deepjump/bin/python").exists(),
    reason="exact cloud virtual environment is unavailable",
)
def test_launcher_executes_inside_exact_cloud_virtual_environment():
    completed = subprocess.run(
        [str(LAUNCHER), "--deepjump-toolchain-version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert completed.stdout.strip() == (
        '{"cuda":"12.1","python":"3.10.12",'
        '"sys_prefix":"/data/venvs/deepjump","torch":"2.5.1+cu121"}'
    )
