import json
import subprocess
import sys
from pathlib import Path

import torch
import pytest

from scripts.train import fast_dev_gate_errors
from scripts.verify_obsutil_empty_prefix import prefix_object_count
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


def test_paper_horizon_ab_freezes_expected_lr_trajectories():
    baseline = load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_horizon_ab_baseline2000.yaml"
    )
    candidate = load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000.yaml"
    )
    assert [lr_at(step, baseline) for step in (0, 199, 200, 1000, 1999)] == pytest.approx(
        [2.5e-5, 5.0e-3, 5.0e-3, 3.0e-3, 3.0e-3]
    )
    assert [lr_at(step, candidate) for step in (0, 199, 200, 1000, 1999)] == pytest.approx(
        [
            2.5e-5,
            5.0e-3,
            5.0e-3,
            0.004996798719487795,
            0.004992801120448179,
        ]
    )


def test_paper_horizon_ab_runner_is_matched_bounded_and_fail_closed():
    runner = Path("cloud/huawei/run_paper_horizon_ab2000.sh").read_text()
    prefix_verifier = Path("scripts/verify_obsutil_empty_prefix.py").read_text()
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-600}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 600 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert "systemctl is-active" in runner
    assert "/usr/bin/systemctl poweroff" in runner
    assert '[[ -z "$(git status --porcelain)" ]]' in runner
    assert "v100_tensorcloud01_full_d1_fp32_horizon_ab_baseline2000.yaml" in runner
    assert "v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000.yaml" in runner
    assert 'scripts/train_ddp.py --config "$config"' in runner
    assert "--resume" not in runner
    assert "--warm-start" not in runner
    assert "run_arm baseline" in runner and "run_arm candidate" in runner
    assert "paper-horizon-ab-baseline1000" in runner
    assert "paper-horizon-500k" in runner
    assert runner.count("scripts/guarded_endpoint_panel_eval.py") == 2
    assert "run_panel baseline" in runner and "run_panel candidate" in runner
    assert "scripts/adjudicate_paper_horizon_ab.py" in runner
    assert 'verify_readback "$READBACK_ONE"' in runner
    assert 'verify_readback "$READBACK_TWO"' in runner
    assert "OBS_DOUBLE_READBACK_PASS" in runner
    assert '"formal_training_authorized": False' in runner
    assert '"second_seed_authorized": False' in runner
    assert "second_seed_scientifically_eligible" in runner
    assert '[[ "$BUCKET" == "obs://deepjump-mdcath-cn4-ringochen" ]]' in runner
    assert "RUN_ID must be UTC basic timestamp" in runner
    assert "refusing to reuse non-empty OBS evidence prefix" in prefix_verifier
    assert "scripts/verify_obsutil_empty_prefix.py" in runner
    assert "Object number" in prefix_verifier
    assert "Folder number" in prefix_verifier
    assert "File number" in prefix_verifier
    assert '"authorization_requires_independent_readback": True' in runner
    assert "final_markers.sha256" in runner
    assert '"$READBACK_TWO/audit/decision.json"' in runner
    assert 'decision.get("status") == "PASS_PAPER_HORIZON_EXTERNAL20"' in runner
    assert 'scientifically_eligible is True' in runner
    assert "ADVANCE_PAPER_HORIZON_EXTERNAL20" in runner
    assert "SKIPPED_PAPER_HORIZON_EXTERNAL20" in runner
    assert "paper_horizon_external_dev_20_length_proportional_seed20260723.txt" in runner
    assert "run_external_panel baseline" in runner
    assert "run_external_panel candidate" in runner
    assert "--panel-kind paper-horizon-external" in runner
    assert "scripts/train_ddp.py" not in runner.split(
        "if [[ \"$training_ab_status\" == ADVANCE_PAPER_HORIZON_EXTERNAL20 ]]"
    )[1]


def test_paper_vector_ab_runner_is_single_arm_bounded_and_fail_closed():
    runner = Path("cloud/huawei/run_paper_vector_ab2000.sh").read_text()
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-660}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 660 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert "systemctl is-active" in runner
    assert "/usr/bin/systemctl poweroff" in runner
    assert '[[ -z "$(git status --porcelain)" ]]' in runner
    assert "scripts/verify_obsutil_empty_prefix.py" in runner
    assert "scripts/verify_paper_vector_readback.py" in runner
    assert "20260722T012922Z" in runner
    assert "dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b" in runner
    assert "fb12d776b106867ca14a8f56476daf776a6296b6dca640f03c2188a75a69bb47" in runner
    assert "868e3e44386163e61e61f6c0da60c160e3cb9f282e20c3ba7a9198208c64fa3f" in runner
    assert "2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38" in runner
    assert "scripts/verify_paper_vector_source_stop.py" in runner
    assert "--source-runner cloud/huawei/run_paper_horizon_ab2000.sh" in runner
    assert 'source_audit_one_sync.log' in runner
    assert 'source_audit_two_sync.log' in runner
    assert "v100_tensorcloud01_vector_only_d1_fp32_paper_horizon500k_2000.yaml" in runner
    assert runner.count("scripts/train_ddp.py --config") == 1
    assert 'scripts/train_ddp.py --config "$CANDIDATE_CONFIG"' in runner
    assert "--resume" not in runner
    assert "--warm-start" not in runner
    assert "world=8 params=4,038,240 effective_batch=128" in runner
    assert runner.count("--require-full-tensor") >= 2
    assert runner.count("--require-vector-only") >= 2
    assert runner.count("scripts/guarded_endpoint_panel_eval.py") == 2
    assert "paper-horizon-500k" in runner
    assert "paper-horizon-vector-only-500k" in runner
    assert "scripts/rollout_robustness_eval.py" in runner
    assert "--domains 3 --starts 2 --steps 20 --methods mean,ode_1" in runner
    assert "scripts/adjudicate_paper_vector_ab.py" in runner
    assert "ADVANCE_PAPER_VECTOR_EXTERNAL20" in runner
    assert "SKIPPED_PAPER_VECTOR_EXTERNAL20" in runner
    assert "paper_horizon_external_dev_20_length_proportional_seed20260723.txt" in runner
    external_branch = runner.split(
        'if [[ "$training_status" == ADVANCE_PAPER_VECTOR_EXTERNAL20 ]]'
    )[1]
    assert "deepjump-governance/external-panel-claims" in runner
    assert 'EXTERNAL_DATA_ROOT=/data/mdcath_paper_vector_external20_seed20260723' in runner
    assert "EXTERNAL_DATA_ROOT=${" not in runner
    assert '[[ ! -e "$EXTERNAL_DATA_ROOT" ]]' in external_branch
    assert "CLAIMED_FOR_SINGLE_USE" in external_branch
    assert "scripts/claim_external_panel.py" in external_branch
    assert "AppendObject" not in runner
    assert "scripts/write_external_download_manifest.py" in external_branch
    assert "--external-claim-sha256" in external_branch
    assert "--external-download-manifest-sha256" in external_branch
    assert "--source-proof-sha256" in external_branch
    assert "--panel-kind paper-vector-external" in external_branch
    assert "--baseline-checkpoint-sha256" in external_branch
    assert "--candidate-checkpoint-sha256" in external_branch
    assert "scripts/train_ddp.py" not in external_branch
    assert 'verify_readback "$READBACK_ONE"' in runner
    assert 'verify_readback "$READBACK_TWO"' in runner
    assert '"$target/audit/audit_sha256.txt"' not in runner
    assert "readback_manifests.sha256" in runner
    assert '--root "$target" --phase initial' in runner
    assert 'verify_final_readback "$FINAL_READBACK_ONE"' in runner
    assert 'verify_final_readback "$FINAL_READBACK_TWO"' in runner
    assert "OBS_PRECOMPLETION_DOUBLE_READBACK_PASS" in runner
    assert '"second_seed_authorized": False' in runner
    assert '"untouched_confirmation_authorized": False' in runner
    assert '"formal_training_authorized": False' in runner
    assert "Paper-vector A/B complete; seed1/untouched/formal training was not started." in runner


def test_scalar_value_ab_runner_is_training_only_and_fail_closed():
    runner = Path("cloud/huawei/run_paper_scalar_value_ab2000.sh").read_text()
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-420}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 420 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert "/usr/bin/systemctl poweroff" in runner
    assert '[[ -z "$(git status --porcelain)" ]]' in runner
    assert "scripts/verify_obsutil_empty_prefix.py" in runner
    assert "scripts/verify_paper_vector_readback.py" in runner
    assert "scripts/verify_paper_scalar_value_readback.py" in runner
    assert "20260722T051048Z" in runner
    assert "fd92112c9ab7c3e941138a95b136f51c29558353" in runner
    assert "19d960826938419e1bf494701a09b395ece729e1c0dc2c8a5d1e6bf36d73053b" in runner
    assert "36f8850ba4e9c094526850370b22371d10df76765eead3e39adf051e68d0d80e" in runner
    assert "0816f94b01bf8b434086677d59c913193a70aa8b802f79b46378590f772af7bf" in runner
    assert "1ceb092102c4c0ad608289a19d924a60e7f55df4fe226a21f8fd27895ab1bac6" in runner
    assert "BASELINE_FINAL_STATUS=STOP_PAPER_VECTOR_ABSOLUTE_GATE" in runner
    assert '--expected-final-decision-sha256 "$BASELINE_FINAL_DECISION_SHA256"' in runner
    assert '--expected-final-status "$BASELINE_FINAL_STATUS"' in runner
    assert "v100_tensorcloud01_vector_scalar_value_d1_fp32_" in runner
    assert runner.count("scripts/train_ddp.py --config") == 1
    assert "--resume" not in runner
    assert "--warm-start" not in runner
    assert "world=8 params=4,443,744 effective_batch=128" in runner
    assert runner.count("--require-vector-only") >= 2
    assert runner.count("--require-vector-scalar-value") >= 2
    assert runner.count("scripts/guarded_endpoint_panel_eval.py") == 1
    assert "run_training_panel baseline" in runner
    assert "run_training_panel candidate" in runner
    assert "paper-horizon-vector-only-500k" in runner
    assert "paper-horizon-vector-scalar-value-500k" in runner
    assert "scripts/adjudicate_paper_scalar_value_ab.py" in runner
    assert "--baseline-replay-decision" in runner
    assert "--evidence-manifest" in runner
    assert '"$SOURCE_READBACK/audit/summary.json"' in runner
    assert '"$SOURCE_READBACK/audit/readback_completion.json"' in runner
    assert '"$SOURCE_READBACK/audit/candidate_decision.json"' in runner
    assert '"$SOURCE_READBACK/audit/decision.json"' in runner
    assert "summary_path, completion_path, candidate_decision_path, decision_path" in runner
    assert 'completion.get("scientific_status")' in runner
    assert 'decision.get("status")' in runner
    assert 'open(decision_path, "rb")' in runner
    assert '"$RUN_DIR/sealed_baseline_decision.json"' in runner
    assert '"$RUN_DIR/candidate_config.yaml"' in runner
    assert "external_development_authorized\": False" in runner
    assert "scripts/download_mdcath.py" not in runner
    assert "claim_external_panel.py" not in runner
    assert 'verify_readback "$READBACK_ONE"' in runner
    assert 'verify_readback "$READBACK_TWO"' in runner
    assert 'verify_final_readback "$FINAL_READBACK_ONE"' in runner
    assert 'verify_final_readback "$FINAL_READBACK_TWO"' in runner
    assert "OBS_PRECOMPLETION_DOUBLE_READBACK_PASS" in runner
    assert "external/seed1/untouched/formal were not started" in runner


def test_paper_horizon_postrun_certifier_is_read_only_bounded_and_source_bound():
    runner = Path("cloud/huawei/certify_paper_horizon_postrun.sh").read_text()
    assert "HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}" in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        '[[ "$SHUTDOWN_ON_EXIT" == 1 ]]'
    )
    assert "/usr/bin/systemctl poweroff" in runner
    assert "20260722T012922Z" in runner
    assert "dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b" in runner
    assert '[[ "$(git rev-parse HEAD)" == "$EXPECTED_REPO_COMMIT" ]]' in runner
    assert '[[ -z "$(git status --porcelain)" ]]' in runner
    assert 'download_readback "$READBACK_ONE"' in runner
    assert 'download_readback "$READBACK_TWO"' in runner
    assert "scripts/certify_paper_horizon_postrun.py" in runner
    assert "scripts/verify_obsutil_empty_prefix.py" in runner
    assert "certification.sha256" in runner
    assert "torchrun" not in runner
    assert "CUDA_VISIBLE_DEVICES" not in runner
    assert "scripts/train_ddp.py --config" not in runner


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ("Object number: 0\n", 0),
        ("Object number is: 2\n", 2),
        ("Folder number: 0\nFile number: 0\n", 0),
        ("Folder number: 1\nFile number: 2\n", 3),
    ],
)
def test_obsutil_prefix_count_supports_cloud_output_variants(report, expected):
    assert prefix_object_count(report) == expected


@pytest.mark.parametrize(
    "report",
    [
        "",
        "Total size of prefix: 0B\n",
        "Folder number: 0\n",
        "File number: 0\n",
        "Object number: 0\nObject number: 3\n",
        "Object number: 0\nFolder number: 1\nFile number: 0\n",
        "Object number: 0\nFolder number: 0\nFile number: 0\n",
        "Folder number: 0\nFolder number: 2\nFile number: 0\n",
        "Folder number: 0\nFile number: 0\nFile number: 2\n",
    ],
)
def test_obsutil_prefix_count_rejects_incomplete_or_unknown_output(report):
    with pytest.raises(ValueError, match="count|format"):
        prefix_object_count(report)


@pytest.mark.parametrize(
    "report",
    [
        "Object number: not-a-number\n",
        "Folder number: 0 objects\nFile number: 0\n",
        "prefix Object number: 0\n",
    ],
)
def test_obsutil_prefix_count_rejects_malformed_count_lines(report):
    with pytest.raises(ValueError, match="malformed count line"):
        prefix_object_count(report)


def _write_checkpoint(
    tmp_path: Path, *, world_size=8, step=10, finite=True, vector_only=True,
    scalar_value=False,
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
                    "tensor_cloud01_vector_only_scalar_value": scalar_value,
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


def test_checkpoint_gate_keeps_vector_and_scalar_value_variants_disjoint(tmp_path):
    checkpoint, history = _write_checkpoint(tmp_path, scalar_value=True)
    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint", str(checkpoint),
            "--history", str(history),
            "--expected-step", "10",
            "--expected-world-size", "8",
            "--require-vector-only",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "pure vector-only" in rejected.stderr

    accepted = subprocess.run(
        [
            sys.executable,
            "scripts/validate_training_checkpoint.py",
            "--checkpoint", str(checkpoint),
            "--history", str(history),
            "--expected-step", "10",
            "--expected-world-size", "8",
            "--require-vector-scalar-value",
        ],
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    report = json.loads(accepted.stdout)
    assert report["vector_only_scalar_value"] is True


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
    assert 'EXPECTED_CHECKPOINT_STEP=${EXPECTED_CHECKPOINT_STEP:-1000}' in runner
    assert runner.count('--expected-checkpoint-step "$EXPECTED_CHECKPOINT_STEP"') == 2
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


def test_teacher_update_projection_runner_is_bounded_bound_and_inference_only():
    runner = Path(
        "cloud/huawei/run_teacher_update_projection_discriminator.sh"
    ).read_text()
    assert 'export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert runner.index("trap shutdown_on_exit EXIT") < runner.index(
        'EXPECTED_REPO_COMMIT=${EXPECTED_REPO_COMMIT:?'
    )
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'sudo -n shutdown -h now || shutdown_code=$?' in runner
    assert "fc5f1e7b5188af4911e518ac0e3d44c2aba4a22431360bde704465c9c1889a73" in runner
    assert "bacf07bdd93119a0b793b67335a520c468c5749d5d9da71d887d5a5fe8aa7753" in runner
    assert "03a953b4bda5e45391f7a06311eceeb84485a1a4b4a54f01edcd8aa7aea2609d" in runner
    assert "70a84d0e6e1bb4491ce51d89bcaf7fccab090ded2dc0415a267792e659794512" in runner
    assert 'for path in "$RUN_DIR" "$RUNNER_LOG"' in runner
    assert "scripts/teacher_update_projection_eval.py" in runner
    assert "--domains 3 --starts 2 --steps 20 --calibration-domain 1gxlA02" in runner
    assert '"$PYTHON" -m scripts.adjudicate_teacher_update_projection' in runner
    assert "scripts/train_ddp.py" not in runner
    assert "torchrun" not in runner.lower()
    assert '"formal_training_authorized": False' in runner
    assert runner.count("verify_teacher_update_projection_readback.py") >= 2
    assert "training was not started" in runner
