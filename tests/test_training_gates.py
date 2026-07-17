import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.train import fast_dev_gate_errors


def test_fast_dev_gate_accepts_finite_improvement():
    report = {
        "initial": {"loss": 10.0, "rmsd": 4.0},
        "final": {"loss": 1.0, "rmsd": 1.5},
        "loss_ratio": 0.1,
        "rmsd_ratio": 0.375,
    }
    assert fast_dev_gate_errors(report, 0.25, 0.5) == []


def test_fast_dev_gate_rejects_nonfinite_and_weak_improvement():
    report = {
        "initial": {"loss": 10.0, "rmsd": 4.0},
        "final": {"loss": float("nan"), "rmsd": 3.0},
        "loss_ratio": 0.8,
        "rmsd_ratio": 0.75,
    }
    errors = fast_dev_gate_errors(report, 0.25, 0.5)
    assert any("non-finite" in error for error in errors)


def test_fast_dev_gate_rejects_finite_but_weak_improvement():
    report = {
        "initial": {"loss": 10.0, "rmsd": 4.0},
        "final": {"loss": 8.0, "rmsd": 3.0},
        "loss_ratio": 0.8,
        "rmsd_ratio": 0.75,
    }
    errors = fast_dev_gate_errors(report, 0.25, 0.5)
    assert any("loss ratio" in error for error in errors)
    assert any("RMSD ratio" in error for error in errors)


def _write_checkpoint(
    tmp_path: Path, *, world_size=8, step=10, finite=True, vector_only=True
):
    checkpoint = tmp_path / "ckpt_10.pt"
    value = torch.tensor([1.0 if finite else float("nan")])
    torch.save(
        {
            "model": {"weight": value},
            "opt": {},
            "scaler": {},
            "step": step,
            "cfg": {
                "data": {"delta_frames": 1},
                "model": {
                    "tensor_cloud01": True,
                    "tensor_cloud01_vector_only_attention": vector_only,
                },
            },
            "checkpoint_schema": 2,
            "train_state": {"world_size": world_size},
        },
        checkpoint,
    )
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps([{"step": step, "val_loss": 1.0, "val_rmsd": 2.0, "noop_rmsd": 3.0}])
    )
    return checkpoint, history


def test_checkpoint_gate_accepts_complete_finite_artifacts(tmp_path):
    checkpoint, history = _write_checkpoint(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--expected-step",
            "10",
            "--expected-world-size",
            "8",
            "--expected-delta",
            "1",
            "--require-vector-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PASS"


def test_checkpoint_gate_rejects_nonfinite_or_wrong_world_size(tmp_path):
    checkpoint, history = _write_checkpoint(tmp_path, world_size=1, finite=False)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--expected-step",
            "10",
            "--expected-world-size",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["status"] == "FAIL"
    assert any("world_size" in error for error in report["errors"])
    assert any("non-finite" in error for error in report["errors"])


def test_checkpoint_gate_rejects_wrong_attention_variant(tmp_path):
    checkpoint, history = _write_checkpoint(tmp_path, vector_only=False)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--expected-step",
            "10",
            "--expected-world-size",
            "8",
            "--expected-delta",
            "1",
            "--require-vector-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "vector-only" in result.stderr


def test_checkpoint_gate_can_select_one_intermediate_history_record(tmp_path):
    checkpoint, history = _write_checkpoint(tmp_path, step=250)
    history.write_text(
        json.dumps(
            [
                {"step": 250, "val_loss": 1.0, "val_rmsd": 2.0, "noop_rmsd": 3.0},
                {"step": 500, "val_loss": 0.8, "val_rmsd": 1.8, "noop_rmsd": 3.0},
            ]
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
            "--history",
            str(history),
            "--expected-step",
            "250",
            "--expected-world-size",
            "8",
            "--history-mode",
            "contains",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["history"]["step"] == 250
    assert report["history_mode"] == "contains"


def test_tensorcloud01_calibration_runner_is_bounded_and_delta_scoped():
    runner = Path("cloud/huawei/run_tensorcloud01_calibration.sh").read_text()
    assert 'DELTA=${DELTA:?' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-30}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 30 ]]' in runner
    assert 'SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?' in runner
    assert runner.index('sudo -n shutdown -h "+$HARD_STOP_MINUTES"') < runner.index(
        "BUCKET=${BUCKET:?"
    )
    assert 'sudo -n shutdown -h now' in runner
    assert '[[ "$code" != 0 ]] || code=$shutdown_code' in runner
    assert 'scripts/train_ddp.py --config "$CONFIG"' in runner
    assert 'timeout --signal=TERM --kill-after=30s 8m' in runner
    assert 'tests/test_tensor_cloud01.py' in runner
    assert 'tests/test_audit_mdcath_staging.py' in runner
    assert 'tests/test_ddp_sync.py' not in runner
    assert 'tests/test_worker_crop_rng.py' not in runner
    assert '"$PYTHON" -m pytest -q |' not in runner
    assert "--warm-start" not in runner
    assert "formal training was not started" in runner
    assert "v100_tensorcloud01_vector_only_d1_calibration.yaml" in runner
    assert "params=4,038,240" in runner
    assert "--require-vector-only" in runner
    assert "vector-only calibration is frozen to delta=1" in runner
