import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.train import fast_dev_gate_errors
from deepjump.config import load_config
from deepjump.training import lr_at


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


def test_lr_horizon_preserves_reference_schedule_for_bounded_probe():
    reference = load_config("configs/v100_tensorcloud01_vector_only_d1_calibration.yaml")
    fp32 = load_config(
        "configs/v100_tensorcloud01_vector_only_fp32_highlr_step230.yaml"
    )
    fp16 = load_config(
        "configs/v100_tensorcloud01_vector_only_fp16_lowlr_step230.yaml"
    )
    for step in (0, 199, 200, 220, 229):
        assert lr_at(step, fp32) == lr_at(step, reference)
        assert lr_at(step, fp16) == lr_at(step, reference) / 10
    assert fp32.train.max_steps == fp16.train.max_steps == 230
    assert fp32.train.lr_horizon_steps == fp16.train.lr_horizon_steps == 1000
    assert not fp32.train.amp
    assert fp16.train.amp and fp16.train.amp_dtype == "fp16"


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


def test_checkpoint_gate_accepts_full_tensor_and_rejects_vector_only(tmp_path):
    checkpoint, history = _write_checkpoint(tmp_path, vector_only=False)
    accepted = subprocess.run(
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
            "--require-full-tensor",
        ],
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)["status"] == "PASS"

    vector_dir = tmp_path / "vector"
    vector_dir.mkdir()
    vector_checkpoint, vector_history = _write_checkpoint(vector_dir, vector_only=True)
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint",
            str(vector_checkpoint),
            "--history",
            str(vector_history),
            "--expected-step",
            "10",
            "--expected-world-size",
            "8",
            "--expected-delta",
            "1",
            "--require-full-tensor",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "full-tensor" in rejected.stderr


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
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'DELTA=${DELTA:?' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-30}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 30 ]]' in runner
    assert 'SHUTDOWN_ON_EXIT=${SHUTDOWN_ON_EXIT:?' in runner
    assert runner.index('sudo -n shutdown -h "+$HARD_STOP_MINUTES"') < runner.index(
        "BUCKET=${BUCKET:?"
    )
    assert 'sudo -n shutdown -h now' in runner
    assert "another train_ddp.py process already exists" in runner
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
    assert "v100_tensorcloud01_vector_only_d1_lowlr_calibration.yaml" in runner
    assert "v100_tensorcloud01_vector_only_d1_fp32_calibration.yaml" in runner
    assert 'LR_PROFILE=${LR_PROFILE:-reference}' in runner
    assert 'unsupported LR_PROFILE=%s; expected reference, lowlr, or fp32' in runner
    assert '"lr_profile":"%s"' in runner
    assert "params=4,038,240" in runner
    assert "--require-vector-only" in runner
    assert "vector-only calibration is frozen to delta=1" in runner


def test_vector_only_numerics_discriminator_is_bounded_and_two_arm_scoped():
    runner = Path("cloud/huawei/run_vector_only_step221_discriminator.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-20}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 20 ]]' in runner
    assert runner.index('sudo -n shutdown -h "+$HARD_STOP_MINUTES"') < runner.index(
        "BUCKET=${BUCKET:?"
    )
    assert 'sudo -n shutdown -h now' in runner
    assert "v100_tensorcloud01_vector_only_fp32_highlr_step230.yaml" in runner
    assert "v100_tensorcloud01_vector_only_fp16_lowlr_step230.yaml" in runner
    assert "timeout --signal=TERM --kill-after=30s 8m" in runner
    assert "--expected-step 230" in runner
    assert "--expected-world-size 8" in runner
    assert "--require-vector-only" in runner
    assert "scaler_skips [1-9][0-9]*" in runner
    assert '"status":"MATRIX_COMPLETE"' in runner
    assert "OBS_DST/$label" in runner
    assert "sha256sum -c" in runner
    assert "formal training was not started" in runner
    assert "--warm-start" not in runner


def test_tensorcloud01_eval_integration_runner_is_bounded_and_pins_source():
    runner = Path("cloud/huawei/run_tensorcloud01_eval_integration.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-30}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 30 ]]' in runner
    assert runner.index('sudo -n shutdown -h "+$HARD_STOP_MINUTES"') < runner.index(
        'cd "$REPO"'
    )
    assert runner.index('trap shutdown_on_exit EXIT') < runner.index(
        '[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]]'
    )
    assert 'sudo -n shutdown -c 2>/dev/null || true' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert '[[ "$code" != 0 ]] || code=$shutdown_code' in runner
    assert "conflicting training/evaluation process exists" in runner
    assert "--domains 1 --starts 1 --draws 2 --methods mean" in runner
    assert "--domains 1 --starts 1 --steps \"$steps\" --methods mean" in runner
    assert "--expected-domains 1" in runner
    assert "sha256sum -c SHA256SUMS" in runner
    assert '"scope":"integration_only"' in runner
    assert "scientific calibration/training was not started" in runner


def test_vector_only_sampling_discriminator_is_bounded_and_inference_only():
    runner = Path("cloud/huawei/run_vector_only_sampling_discriminator.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-10}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 10 ]]' in runner
    assert runner.index('trap shutdown_on_exit EXIT') < runner.index(
        '[[ "$(hostname)" == "$EXPECTED_HOSTNAME" ]]'
    )
    assert 'sudo -n shutdown -c 2>/dev/null || true' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert "conflicting training/evaluation process exists" in runner
    assert "--domains 1 --starts 1 --steps 20" in runner
    assert "--methods mean,ode_1,ode_5,ode_20" in runner
    assert 'for anchor in state conditioner' in runner
    assert '--drift-anchor "$anchor"' in runner
    assert '"scope": "inference_mechanism_probe_only"' in runner
    assert "sha256sum -c SHA256SUMS" in runner
    assert '"$PYTHON" scripts/train_ddp.py' not in runner
    assert "--warm-start" not in runner
    assert "no training or scientific gate was run" in runner


def test_vector_only_paper_loss_continuation_is_bounded_and_fail_closed():
    runner = Path(
        "cloud/huawei/run_vector_only_paper_loss_continuation2000.sh"
    ).read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-90}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 90 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'CHECKPOINT=${CHECKPOINT:?'
    )
    assert 'sudo -n shutdown -h "+$HARD_STOP_MINUTES"' in runner
    assert 'sudo -n shutdown -h now' in runner
    assert 'EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?' in runner
    assert "v100_tensorcloud01_vector_only_d1_fp32_continuation2000.yaml" in runner
    assert 'scripts/train_ddp.py --config "$CONFIG" --resume "$CHECKPOINT"' in runner
    assert "--warm-start" not in runner
    assert "--expected-step 1000" in runner
    assert "for step in $(seq 1100 100 2000)" in runner
    assert "--domains 3 --starts 2 --steps 20 --methods mean,ode_1" in runner
    assert "--drift-anchor state" in runner
    assert "scripts/adjudicate_paper_loss_continuation.py" in runner
    assert '"formal_training_authorized":false' in runner
    assert "sha256sum -c" in runner
    assert "formal training was not started" in runner


def test_full_tensor_paper_loss_discriminator_is_matched_bounded_and_fail_closed():
    runner = Path(
        "cloud/huawei/run_full_tensor_paper_loss_discriminator2000.sh"
    ).read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-135}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 135 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert 'sudo -n shutdown -h "+$HARD_STOP_MINUTES"' in runner
    assert 'sudo -n shutdown -h now' in runner
    assert "v100_tensorcloud01_full_d1_fp32_calibration.yaml" in runner
    assert "v100_tensorcloud01_full_d1_fp32_continuation2000.yaml" in runner
    assert 'scripts/train_ddp.py --config "$CALIBRATION_CONFIG"' in runner
    assert (
        'scripts/train_ddp.py --config "$CONTINUATION_CONFIG" --resume '
        '"$SOURCE_CHECKPOINT"'
    ) in runner
    assert "--warm-start" not in runner
    assert "params=4,840,032" in runner
    assert runner.count("--require-full-tensor") >= 3
    assert "for step in $(seq 1100 100 2000)" in runner
    assert "--domains 3 --starts 2 --steps 20 --methods mean,ode_1" in runner
    assert "--drift-anchor state" in runner
    assert "scripts/adjudicate_full_tensor_discriminator.py" in runner
    assert "VECTOR_BASELINE_OBS_PREFIX" in runner
    assert "VECTOR_BASELINE_SHA256" in runner
    assert "35b73f0d3f0889201fb192735114a7e818e30df41259edf6f4a6f8f8479755ff" in runner
    assert '--vector-baseline "$VECTOR_BASELINE_PATH"' in runner
    assert "expected exactly one frozen vector-only baseline" in runner
    assert '"formal_training_authorized":false' in runner
    assert "sha256sum -c" in runner
    assert "formal training was not started" in runner
