import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from deepjump.config import load_config, to_dict
import scripts.adjudicate_scalar_feedback_h6 as discriminator
import scripts.verify_scalar_feedback_h6_readback as readback


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(
    rmsd: list[float],
    *,
    bond_mean: list[float] | None = None,
    bond_max: list[float] | None = None,
) -> dict:
    bond_mean = bond_mean or [3.8] * 7
    bond_max = bond_max or [4.2] * 7
    return {
        "rmsd": rmsd,
        "rmsd_by_start": [[value] * 5 for value in rmsd],
        "fnc": [1.0] * 7,
        "bond_mean": bond_mean,
        "bond_p95": bond_mean,
        "bond_p99": bond_max,
        "bond_max": bond_max,
        "bond_mae_real": [0.1] * 7,
        "angle_cos_mae_real": [0.1] * 7,
    }


def _write_result(path: Path, payload: dict) -> None:
    for method in payload["domains"][0]["methods"].values():
        method["rmsd_by_start"] = [[value] * 5 for value in method["rmsd"]]
    payload["summary"] = {
        name: discriminator.summarize_domains(payload["domains"], name, 6)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    path.write_text(json.dumps(payload))


def _case(tmp_path: Path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = to_dict(load_config(
        "configs/v100_tensorcloud01_vector_scalar_value_d1_fp32_"
        "paper_horizon500k_2000.yaml"
    ))
    checkpoint = tmp_path / "ckpt_2000.pt"
    torch.save({
        "step": 2000,
        "checkpoint_schema": 2,
        "cfg": config,
        "model": {"weight": torch.ones(1)},
        "train_state": {"world_size": 8, "train_fingerprint": "e" * 64},
    }, checkpoint)
    checkpoint_sha = _sha(checkpoint)
    monkeypatch.setattr(discriminator, "EXPECTED_CHECKPOINT_SHA256", checkpoint_sha)

    domain_list = tmp_path / "dev20.txt"
    domain_list.write_text("1gxlA02\n" + "\n".join(
        f"domain{i:02d}" for i in range(19)
    ) + "\n")
    domain_sha = _sha(domain_list)
    monkeypatch.setattr(discriminator, "EXPECTED_DOMAIN_LIST_SHA256", domain_sha)

    source = tmp_path / "training_evidence.json"
    source.write_text(json.dumps({
        "schema": "deepjump.scalar_value_training_evidence.v1",
        "commit": discriminator.EXPECTED_SOURCE_COMMIT,
        "candidate_config_sha256": discriminator.EXPECTED_CONFIG_SHA256,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "domain_list_sha256": domain_sha,
    }, sort_keys=True, separators=(",", ":")) + "\n")
    monkeypatch.setattr(discriminator, "EXPECTED_SOURCE_EVIDENCE_SHA256", _sha(source))

    persistence = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    result = tmp_path / "result.json"
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 2000,
        "settings": {
            "ckpt": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "domain_list": str(domain_list),
            "domain_list_sha256": domain_sha,
            "domains": 1,
            "starts": 5,
            "steps": 6,
            "methods": "mean",
            "seed": 20260718,
            "noise_sigma": None,
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
            "project_v_atom_mask": False,
            "teacher_forced_mean": True,
            "output": str(result),
        },
        "preprocessing": {"canon_symmetric": True},
        "delta_frames": 1,
        "domain_panel": {
            "path": str(domain_list),
            "sha256": domain_sha,
            "count": 20,
            "evaluated_count": 1,
        },
        "domains": [{
            "domain": "1gxlA02",
            "residues_total": 64,
            "residues_evaluated": 64,
            "frames": 11,
            "starts": [0, 1, 2, 3, 4],
            "methods": {
                "noop": _metrics([0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
                "one_step_persistence": _metrics(persistence),
                "mean": _metrics(
                    [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    bond_max=[4.2, 4.2, 4.2, 4.2, 6.0, 6.2, 6.4],
                ),
                "teacher_forced_mean": _metrics(
                    [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
                ),
            },
        }],
    }
    _write_result(result, payload)
    return result, checkpoint, domain_list, source


def _adjudicate(case):
    return discriminator.adjudicate(*case)


def test_scalar_feedback_classifies_distribution_shift(tmp_path, monkeypatch):
    report = _adjudicate(_case(tmp_path, monkeypatch))
    assert report["status"] == "FEEDBACK_DISTRIBUTION_SHIFT"
    assert report["teacher_forced_first_nonphysical_step"] is None
    assert report["autoregressive_first_nonphysical_step"] == 4
    assert report["teacher_forced_steps_better_than_one_step_persistence"] == 6
    for key in (
        "external_development_scientifically_eligible",
        "external_development_authorized",
        "second_seed_scientifically_eligible",
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
    ):
        assert report[key] is False


def test_scalar_feedback_classifies_endpoint_failure_on_teacher_geometry(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["teacher_forced_mean"]["bond_max"][4] = 5.6
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "ENDPOINT_OPERATOR_FAILURE"
    assert report["teacher_forced_first_nonphysical_step"] == 4


def test_scalar_feedback_classifies_endpoint_failure_at_two_persistence_wins(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["mean"]["bond_max"] = [4.2] * 7
    payload["domains"][0]["methods"]["teacher_forced_mean"]["rmsd"] = [
        0.0, 1.0, 1.0, 2.0, 2.0, 2.1, 2.1
    ]
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "ENDPOINT_OPERATOR_FAILURE"
    assert report["teacher_forced_steps_better_than_one_step_persistence"] == 2


def test_scalar_feedback_is_inconclusive_outside_preregistered_regions(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["mean"]["bond_max"] = [4.2] * 7
    payload["domains"][0]["methods"]["teacher_forced_mean"]["rmsd"] = [
        0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0
    ]
    _write_result(case[0], payload)
    assert _adjudicate(case)["status"] == "INCONCLUSIVE_SCALAR_FEEDBACK_H6"


def test_scalar_feedback_compares_teacher_to_persistence_not_noop(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["mean"]["bond_max"] = [4.2] * 7
    payload["domains"][0]["methods"]["teacher_forced_mean"]["rmsd"] = [
        0.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5
    ]
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["teacher_forced_steps_better_than_one_step_persistence"] == 0
    assert report["status"] == "ENDPOINT_OPERATOR_FAILURE"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("checkpoint_step",), 1000, "checkpoint step 2000"),
        (("settings", "seed"), 7, "settings mismatch"),
        (("settings", "teacher_forced_mean"), False, "settings mismatch"),
        (("domain_panel", "count"), 1, "frozen dev20"),
        (("domains", 0, "domain"), "wrong", "domain mismatch"),
    ],
)
def test_scalar_feedback_rejects_identity_mutations(
    tmp_path, monkeypatch, path, value, message
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_result(case[0], payload)
    with pytest.raises(ValueError, match=message):
        _adjudicate(case)


def test_scalar_feedback_rejects_nonfinite_teacher_metric(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["teacher_forced_mean"]["rmsd"][3] = float("nan")
    _write_result(case[0], payload)
    with pytest.raises(ValueError, match="must be finite"):
        _adjudicate(case)


def test_scalar_feedback_rejects_reanchored_aggregate_rmsd_replacement(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    teacher = payload["domains"][0]["methods"]["teacher_forced_mean"]
    teacher["rmsd"] = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    payload["summary"] = {
        name: discriminator.summarize_domains(payload["domains"], name, 6)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    case[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match rmsd_by_start"):
        _adjudicate(case)


@pytest.mark.parametrize("location", ("result", "settings", "method"))
def test_scalar_feedback_rejects_authorization_like_field_injection(
    tmp_path, monkeypatch, location
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    if location == "result":
        payload["formal_training_authorized"] = True
    elif location == "settings":
        payload["settings"]["formal_training_authorized"] = True
    else:
        payload["domains"][0]["methods"]["teacher_forced_mean"][
            "formal_training_authorized"
        ] = True
    payload["summary"] = {
        name: discriminator.summarize_domains(payload["domains"], name, 6)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    case[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing or extra"):
        _adjudicate(case)


def test_scalar_feedback_rejects_source_commit_even_with_reanchored_digest(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    source = case[3]
    payload = json.loads(source.read_text())
    payload["commit"] = "0" * 40
    source.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    monkeypatch.setattr(discriminator, "EXPECTED_SOURCE_EVIDENCE_SHA256", _sha(source))
    with pytest.raises(ValueError, match="commit mismatch"):
        _adjudicate(case)


def test_scalar_feedback_rejects_wrong_scalar_profile(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    checkpoint = case[1]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["cfg"]["model"]["tensor_cloud01_vector_only_scalar_value"] = False
    torch.save(payload, checkpoint)
    checkpoint_sha = _sha(checkpoint)
    monkeypatch.setattr(discriminator, "EXPECTED_CHECKPOINT_SHA256", checkpoint_sha)
    source = case[3]
    evidence = json.loads(source.read_text())
    evidence["candidate_checkpoint_sha256"] = checkpoint_sha
    source.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
    monkeypatch.setattr(discriminator, "EXPECTED_SOURCE_EVIDENCE_SHA256", _sha(source))
    result = json.loads(case[0].read_text())
    result["checkpoint_sha256"] = checkpoint_sha
    result["settings"]["checkpoint_sha256"] = checkpoint_sha
    _write_result(case[0], result)
    with pytest.raises(
        ValueError, match="model.tensor_cloud01_vector_only_scalar_value mismatch"
    ):
        _adjudicate(case)


def test_scalar_feedback_runner_is_bounded_readback_closed_and_training_free():
    runner = Path(
        "cloud/huawei/run_scalar_feedback_h6_discriminator.sh"
    ).read_text()
    assert "HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}" in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert "ExecStart=/usr/bin/systemctl poweroff" in runner
    assert "timeout --signal=TERM --kill-after=30s 15m" in runner
    assert "scripts/rollout_robustness_eval.py" in runner
    assert "--domains 1 --starts 5 --steps 6 --methods mean --teacher-forced-mean" in runner
    assert "scripts.adjudicate_scalar_feedback_h6" in runner
    assert "scripts/verify_obsutil_empty_prefix.py" in runner
    assert runner.count('verify_readback "$') == 4
    assert runner.count("scripts/verify_scalar_feedback_h6_readback.py") >= 3
    assert runner.count("--root-two") == 2
    assert "readback_completion.json" in runner
    assert "audit_sha256.txt" in runner
    assert "runtime_evidence.log" in runner
    assert "hard_stop_evidence.log" in runner
    assert "Maximum declared success-path timeout envelope is 31.5 minutes" in runner
    assert "OBS_PREFIX_EMPTY_VERIFIED" in runner
    assert "shutdown -h now" in runner
    assert '"$PYTHON" scripts/train_ddp.py' not in runner
    assert "formal_training_authorized\": False" in runner


def _make_readback(
    tmp_path: Path,
    monkeypatch,
    *,
    completion: bool,
) -> tuple[Path, Path, Path, str, str, str]:
    case = _case(tmp_path / "case", monkeypatch)
    result, checkpoint, domain_list, source = case
    root = tmp_path / "readback"
    root.mkdir(parents=True)
    shutil.copyfile(result, root / "result.json")
    shutil.copyfile(source, root / "source_training_evidence.json")
    for name in (
        "hard_stop_evidence.log", "obs_prefix_preflight.log", "pytest.log",
        "result.log", "runtime_evidence.log"
    ):
        (root / name).write_text(f"{name}\n")
    decision = discriminator.adjudicate(*case)
    (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    run_id = "20260722T120000Z"
    commit = "d" * 40
    obs = "obs://deepjump-mdcath-cn4-ringochen/deepjump-diagnostics/scalar-feedback-h6/test"
    summary = {
        "status": decision["status"],
        "scope": decision["scope"],
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "run_id": run_id,
        "deployed_commit": commit,
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "checkpoint_source_commit": decision["checkpoint_source_commit"],
        "obs": obs,
        "completed_at": "2026-07-22T12:00:00+00:00",
    }
    (root / "summary.json").write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    covered = sorted(readback.INITIAL_FILES - {"audit_sha256.txt"})
    (root / "audit_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in covered
    ))
    if completion:
        initial_report = readback.verify(
            root,
            checkpoint,
            domain_list,
            phase="initial",
            expected_run_id=run_id,
            expected_deployed_commit=commit,
            expected_obs=obs,
        )
        pair_report = {**initial_report, "independent_readbacks_verified": 2}
        for name, report in (
            ("initial_readback_one.json", initial_report),
            ("initial_readback_two.json", initial_report),
            ("initial_readback_pair.json", pair_report),
        ):
            (root / name).write_text(json.dumps(report, indent=2) + "\n")
        completion_payload = {
            "status": "OBS_DOUBLE_READBACK_PASS",
            "decision_status": decision["status"],
            "run_id": run_id,
            "commit": commit,
            "audit_manifest_sha256": _sha(root / "audit_sha256.txt"),
            "archived_decision_sha256": _sha(root / "decision.json"),
            "archived_summary_sha256": _sha(root / "summary.json"),
            "recomputed_decision_sha256": _sha(root / "decision.json"),
            "initial_readback_one_sha256": _sha(root / "initial_readback_one.json"),
            "initial_readback_two_sha256": _sha(root / "initial_readback_two.json"),
            "initial_readback_pair_sha256": _sha(root / "initial_readback_pair.json"),
            "independent_readbacks_verified": 2,
            "external_development_authorized": False,
            "second_seed_authorized": False,
            "untouched_confirmation_authorized": False,
            "formal_training_authorized": False,
            "completed_at": "2026-07-22T12:01:00+00:00",
        }
        (root / "readback_completion.json").write_text(
            json.dumps(completion_payload, separators=(",", ":")) + "\n"
        )
        completion_names = (
            "initial_readback_one.json",
            "initial_readback_two.json",
            "initial_readback_pair.json",
            "readback_completion.json",
        )
        (root / "completion_sha256.txt").write_text("".join(
            f"{_sha(root / name)}  {name}\n" for name in completion_names
        ))
    return root, checkpoint, domain_list, run_id, commit, obs


def _verify_readback(case, *, phase: str):
    root, checkpoint, domain_list, run_id, commit, obs = case
    return readback.verify(
        root,
        checkpoint,
        domain_list,
        phase=phase,
        expected_run_id=run_id,
        expected_deployed_commit=commit,
        expected_obs=obs,
    )


def test_scalar_feedback_readback_recomputes_initial_and_completion(
    tmp_path, monkeypatch
):
    initial = _make_readback(tmp_path / "initial", monkeypatch, completion=False)
    assert _verify_readback(initial, phase="initial")["status"] == (
        "SCALAR_FEEDBACK_H6_READBACK_PASS"
    )
    completed = _make_readback(tmp_path / "completed", monkeypatch, completion=True)
    report = _verify_readback(completed, phase="completion")
    assert report["decision_status"] == "FEEDBACK_DISTRIBUTION_SHIFT"
    assert report["completion_sha256"] is not None


def test_scalar_feedback_readback_rejects_extra_object(tmp_path, monkeypatch):
    case = _make_readback(tmp_path, monkeypatch, completion=False)
    (case[0] / "injected.json").write_text("{}\n")
    with pytest.raises(ValueError, match="missing or extra"):
        _verify_readback(case, phase="initial")


def test_scalar_feedback_completion_readback_rejects_extra_object(
    tmp_path, monkeypatch
):
    case = _make_readback(tmp_path, monkeypatch, completion=True)
    (case[0] / "injected.json").write_text("{}\n")
    with pytest.raises(ValueError, match="missing or extra"):
        _verify_readback(case, phase="completion")


def test_scalar_feedback_readback_rejects_reanchored_false_decision(
    tmp_path, monkeypatch
):
    case = _make_readback(tmp_path, monkeypatch, completion=False)
    root = case[0]
    decision = json.loads((root / "decision.json").read_text())
    decision["status"] = "ENDPOINT_OPERATOR_FAILURE"
    (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    summary = json.loads((root / "summary.json").read_text())
    summary["status"] = decision["status"]
    (root / "summary.json").write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    covered = sorted(readback.INITIAL_FILES - {"audit_sha256.txt"})
    (root / "audit_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in covered
    ))
    with pytest.raises(ValueError, match="differs from recomputed"):
        _verify_readback(case, phase="initial")


def test_scalar_feedback_completion_binds_decision_and_summary(tmp_path, monkeypatch):
    case = _make_readback(tmp_path, monkeypatch, completion=True)
    root = case[0]
    completion = json.loads((root / "readback_completion.json").read_text())
    completion["archived_summary_sha256"] = "0" * 64
    (root / "readback_completion.json").write_text(
        json.dumps(completion, separators=(",", ":")) + "\n"
    )
    completion_names = (
        "initial_readback_one.json",
        "initial_readback_two.json",
        "initial_readback_pair.json",
        "readback_completion.json",
    )
    (root / "completion_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in completion_names
    ))
    with pytest.raises(ValueError, match="archived_summary_sha256 mismatch"):
        _verify_readback(case, phase="completion")


def _reanchor_completion_reports(root: Path) -> None:
    completion = json.loads((root / "readback_completion.json").read_text())
    completion["initial_readback_one_sha256"] = _sha(
        root / "initial_readback_one.json"
    )
    completion["initial_readback_two_sha256"] = _sha(
        root / "initial_readback_two.json"
    )
    completion["initial_readback_pair_sha256"] = _sha(
        root / "initial_readback_pair.json"
    )
    (root / "readback_completion.json").write_text(
        json.dumps(completion, separators=(",", ":")) + "\n"
    )
    completion_names = (
        "initial_readback_one.json",
        "initial_readback_two.json",
        "initial_readback_pair.json",
        "readback_completion.json",
    )
    (root / "completion_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in completion_names
    ))


def test_scalar_feedback_completion_rejects_reanchored_report_field_injection(
    tmp_path, monkeypatch
):
    case = _make_readback(tmp_path, monkeypatch, completion=True)
    root = case[0]
    report = json.loads((root / "initial_readback_pair.json").read_text())
    report["formal_training_authorized"] = True
    (root / "initial_readback_pair.json").write_text(json.dumps(report, indent=2) + "\n")
    _reanchor_completion_reports(root)
    with pytest.raises(ValueError, match="missing or extra fields"):
        _verify_readback(case, phase="completion")


def test_scalar_feedback_completion_rejects_reanchored_report_inventory_replacement(
    tmp_path, monkeypatch
):
    case = _make_readback(tmp_path, monkeypatch, completion=True)
    root = case[0]
    report = json.loads((root / "initial_readback_pair.json").read_text())
    report["inventory_sha256"] = "0" * 64
    (root / "initial_readback_pair.json").write_text(json.dumps(report, indent=2) + "\n")
    _reanchor_completion_reports(root)
    with pytest.raises(ValueError, match="reports differ"):
        _verify_readback(case, phase="completion")


def test_scalar_feedback_readback_pair_requires_independent_exact_inventories(
    tmp_path, monkeypatch
):
    case = _make_readback(tmp_path / "source", monkeypatch, completion=True)
    first = tmp_path / "one"
    second = tmp_path / "two"
    shutil.copytree(case[0], first)
    shutil.copytree(case[0], second)
    kwargs = {
        "phase": "completion",
        "expected_run_id": case[3],
        "expected_deployed_commit": case[4],
        "expected_obs": case[5],
    }
    assert readback.verify_pair(
        first, second, case[1], case[2], **kwargs
    )["independent_readbacks_verified"] == 2
    with pytest.raises(ValueError, match="roots must differ"):
        readback.verify_pair(first, first, case[1], case[2], **kwargs)
