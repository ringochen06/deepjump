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


def test_dev20_endpoint_runner_is_evaluation_only_bounded_and_readback_closed():
    runner = Path("cloud/huawei/run_twenty_domain_endpoint_gate.sh").read_text()

    assert 'export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-75}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 75 ]]' in runner
    assert runner.index("systemd-run") < runner.index("nvidia-smi")
    assert "scripts/endpoint_panel_eval.py" in runner
    assert "--starts 3" in runner
    assert '--runtime-probe-output "$RUN_DIR/runtime_probe.json"' in runner
    assert "runtime_probe.json panel.json panel.log" in runner
    assert "scripts.adjudicate_endpoint_panel" in runner
    assert "tests/test_endpoint_panel_adjudication.py" in runner
    assert "configs/dev_20_length_proportional_seed0.txt" in runner
    assert "4fd7015951fc48598d7beb888670d701b39697cdf62c2982a95b2b7b243474af" in runner
    assert "train_ddp.py" not in runner
    assert "scripts/train.py" not in runner
    assert 'second_seed_authorized":false' in runner
    assert 'formal_training_authorized":false' in runner
    assert "sha256sum -c \"$RUN_DIR/audit_sha256.txt\"" in runner
    assert "readback_completion.sha256" in runner
    assert "sudo -n shutdown -h now" in runner


def test_external_dev20_runner_is_disjoint_evaluation_only_and_fail_closed():
    runner = Path("cloud/huawei/run_external_dev20_endpoint_gate.sh").read_text()

    assert 'export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-120}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 120 ]]' in runner
    assert runner.index("systemd-run") < runner.index("nvidia-smi")
    assert "scripts/external_endpoint_panel_eval.py" in runner
    assert "scripts.adjudicate_external_endpoint_panel" in runner
    assert "configs/subset_1000_length_proportional.txt" in runner
    assert "configs/external_dev_20_length_proportional_seed20260721.txt" in runner
    assert "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734" in runner
    assert "9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245" in runner
    assert "13778143616" in runner
    assert "--expected-h5 1000" in runner
    assert "scripts/download_mdcath.py" in runner
    assert "scripts/audit_external_mdcath.py" in runner
    assert "download.log" in runner
    assert "tests/test_external_endpoint_panel_adjudication.py" in runner
    assert "train_ddp.py" not in runner
    assert "scripts/train.py" not in runner
    assert 'formal_training_authorized": False' in runner
    assert 'second_seed_authorized": bool(decision["second_seed_authorized"])' in runner
    assert "sha256sum -c \"$RUN_DIR/audit_sha256.txt\"" in runner
    assert "readback_completion.sha256" in runner
    assert "sudo -n shutdown -h now" in runner


def test_external_endpoint_root_cause_runner_is_bounded_and_training_free():
    runner = Path("cloud/huawei/run_external_endpoint_root_cause.sh").read_text()
    assert "HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-75}" in runner
    assert "SHUTDOWN_ON_EXIT must be 1" in runner
    assert runner.index("systemd-run") < runner.index("nvidia-smi")
    assert "scripts/external_endpoint_root_cause.py" in runner
    assert "scripts.adjudicate_external_endpoint_root_cause" in runner
    assert "configs/external_context_9_root_cause.txt" in runner
    assert "7ec4af135d80c94764099c201ed1e3283f8bf17579fba34aa208e477bb484573" in runner
    assert "REFERENCE_PANEL_SHA256=6b904ca244242987e28dcc3598a8ad877501f45e9ca4acba26a3e6dedc683b25" in runner
    assert "OBS_READBACK_PASS" in runner
    assert "scripts/train_ddp.py" not in runner
    assert '"formal_training_authorized": False' in runner
    assert "sudo -n shutdown -h now" in runner


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
    assert '"$PYTHON" -m scripts.adjudicate_full_tensor_discriminator' in runner
    assert '"$PYTHON" scripts/adjudicate_full_tensor_discriminator.py' not in runner
    assert "VECTOR_BASELINE_OBS_PREFIX" in runner
    assert "VECTOR_BASELINE_SHA256" in runner
    assert "35b73f0d3f0889201fb192735114a7e818e30df41259edf6f4a6f8f8479755ff" in runner
    assert '--vector-baseline "$VECTOR_BASELINE_PATH"' in runner
    assert "expected exactly one frozen vector-only baseline" in runner
    assert '"formal_training_authorized":false' in runner
    assert "sha256sum -c" in runner
    assert "formal training was not started" in runner


def test_first_party_source_law_runner_is_bounded_and_fail_closed():
    runner = Path(
        "cloud/huawei/run_first_party_source_law1000.sh"
    ).read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-135}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 135 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        '[[ "$SHUTDOWN_ON_EXIT" == 1 ]]'
    )
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'systemctl is-active --quiet "$HARD_STOP_UNIT.timer"' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert "v100_tensorcloud01_full_d1_first_party_source_law1000.yaml" in runner
    assert 'scripts/train_ddp.py --config "$CONFIG"' in runner
    assert '--expected-step 1000' in runner
    assert '--domains 1 --starts 5 --steps 6 --methods ode_150' in runner
    assert '--domains 1 --starts 5 --steps 20 --methods ode_150' in runner
    assert runner.index('--steps 6 --methods ode_150') < runner.index(
        'if [[ "$status" == ADVANCE_SOURCE_LAW_H20 ]]'
    )
    assert 'scripts/adjudicate_source_law_candidate.py' in runner
    assert '"formal_training_authorized":false' in runner
    assert 'sha256sum -c "$RUN_DIR/training_sha256.txt"' in runner
    assert 'sha256sum -c "$RUN_DIR/audit_sha256.txt"' in runner
    assert runner.count("timeout --signal=TERM --kill-after=30s") >= 7
    assert 'cmp "$RUN_DIR/obs_ckpt_gate.json"' in runner
    assert "formal training was not started" in runner


def test_v_mask_projection_runner_is_bounded_conditional_and_inference_only():
    runner = Path(
        "cloud/huawei/run_v_mask_projection_discriminator.sh"
    ).read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-75}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 75 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert 'CHECKPOINT=${CHECKPOINT:?set the existing source-law ckpt_1000.pt path}' in runner
    assert 'CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the frozen checkpoint SHA256}' in runner
    assert "_verify_checkpoint_source_law" in runner
    assert "run_eval 1 current" in runner
    assert "run_eval 1 masked" in runner
    assert runner.index("run_eval 1 masked") < runner.index(
        'if [[ "$status" == ADVANCE_MASKED_H6 ]]'
    )
    assert "run_eval 6 current" in runner
    assert "run_eval 6 masked" in runner
    assert "--project-v-atom-mask" in runner
    assert runner.count('"$PYTHON" -m scripts.adjudicate_v_mask_projection') == 2
    assert '"$PYTHON" scripts/adjudicate_v_mask_projection.py' not in runner
    assert '"twenty_domain_authorized":false' in runner
    assert '"second_seed_authorized":false' in runner
    assert '"confirmation_authorized":false' in runner
    assert '"formal_training_authorized":false' in runner
    assert "scripts/train_ddp.py" not in runner
    assert "torchrun" not in runner.lower()
    assert 'sha256sum -c "$RUN_DIR/audit_sha256.txt"' in runner
    assert "training was not started" in runner


def test_ode_step_scan_runner_is_paired_bounded_and_inference_only():
    runner = Path("cloud/huawei/run_ode_step_scan.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert 'REFERENCE_ODE150=${REFERENCE_ODE150:?' in runner
    assert 'REFERENCE_ODE150_SHA256=${REFERENCE_ODE150_SHA256:?' in runner
    assert 'METHODS=(mean ode_1 ode_2 ode_5 ode_10 ode_20 ode_40 ode_75 ode_150)' in runner
    assert '--domains 1 --starts 5 --steps 1 --methods "$method"' in runner
    assert '--seed 20260718 --integrator euler --tau-max 1.0 --drift-anchor state' in runner
    assert '"$PYTHON" -m scripts.adjudicate_ode_step_scan' in runner
    assert '"twenty_domain_authorized":false' in runner
    assert '"second_seed_authorized":false' in runner
    assert '"confirmation_authorized":false' in runner
    assert '"formal_training_authorized":false' in runner
    assert "scripts/train_ddp.py" not in runner
    assert "torchrun" not in runner.lower()
    assert 'sha256sum -c "$RUN_DIR/audit_sha256.txt"' in runner
    assert "training was not started" in runner


def test_endpoint_grid_runner_is_full_grid_bounded_and_inference_only():
    runner = Path("cloud/huawei/run_endpoint_grid_discriminator.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert 'CHECKPOINT=${CHECKPOINT:?set the existing source-law ckpt_1000.pt path}' in runner
    assert 'CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the frozen checkpoint SHA256}' in runner
    assert "_verify_checkpoint_source_law" in runner
    assert "scripts/endpoint_grid_eval.py" in runner
    assert '--checkpoint-sha256 "$CHECKPOINT_SHA256"' in runner
    assert '--starts 5 --output "$RUN_DIR/grid.json"' in runner
    assert '"$PYTHON" -m scripts.adjudicate_endpoint_grid' in runner
    assert '"twenty_domain_authorized":false' in runner
    assert '"second_seed_authorized":false' in runner
    assert '"confirmation_authorized":false' in runner
    assert '"formal_training_authorized":false' in runner
    assert "scripts/train_ddp.py" not in runner
    assert "torchrun" not in runner.lower()
    assert 'sha256sum -c "$RUN_DIR/audit_sha256.txt"' in runner
    assert "training was not started" in runner


def test_heldout_endpoint_grid_runner_is_preregistered_bounded_and_inference_only():
    runner = Path("cloud/huawei/run_heldout_endpoint_grid_discriminator.sh").read_text()
    assert 'export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert 'DOMAIN_LIST=configs/heldout_endpoint_domain_seed0.txt' in runner
    assert (
        'DOMAIN_LIST_SHA256=a3804fd4bb0fc09b32efd0de790b14862acece82c57a55860d5b6306f41ab61c'
        in runner
    )
    assert "scripts/endpoint_grid_eval.py" in runner
    assert 'assert d["domains"]==["1a0hA01"]' in runner
    assert 'assert int(d["crop_length"])>=86' in runner
    assert '"$PYTHON" -m scripts.adjudicate_heldout_endpoint_grid' in runner
    assert '--starts 5 --output "$RUN_DIR/grid.json"' in runner
    assert '"twenty_domain_authorized":false' in runner
    assert '"second_seed_authorized":false' in runner
    assert '"confirmation_authorized":false' in runner
    assert '"formal_training_authorized":false' in runner
    assert "scripts/train_ddp.py" not in runner
    assert "torchrun" not in runner.lower()
    assert 'sha256sum -c "$RUN_DIR/audit_sha256.txt"' in runner
    assert '"status":"OBS_READBACK_PASS"' in runner
    assert "readback_completion.sha256" in runner
    assert "training was not started" in runner
