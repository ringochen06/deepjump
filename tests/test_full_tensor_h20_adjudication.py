import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from deepjump.config import load_config, to_dict
import scripts.adjudicate_full_tensor_h20 as discriminator
from scripts.rollout_robustness_eval import select_validation_domains
import scripts.verify_full_tensor_h20_readback as readback


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(value: float) -> dict:
    rmsd_by_start = [[0.0, 0.0]] + [[value, value] for _ in range(20)]
    rmsd = [sum(row) / 2 for row in rmsd_by_start]
    bond_mean = [3.8] * 21
    bond_max = [4.2] * 21
    return {
        "rmsd": rmsd,
        "rmsd_by_start": rmsd_by_start,
        "fnc": [1.0] * 21,
        "bond_mean": bond_mean,
        "bond_p95": bond_mean,
        "bond_p99": bond_max,
        "bond_max": bond_max,
        "bond_mean_by_start": [[value, value] for value in bond_mean],
        "bond_max_by_start": [[value, value] for value in bond_max],
        "bond_mae_real": [0.1] * 21,
        "angle_cos_mae_real": [0.1] * 21,
    }


def _recompute(payload: dict) -> None:
    for row in payload["domains"]:
        for method in row["methods"].values():
            method["rmsd"] = [sum(values) / 2 for values in method["rmsd_by_start"]]
            method["bond_mean"] = [
                sum(values) / 2 for values in method["bond_mean_by_start"]
            ]
            method["bond_max"] = [max(values) for values in method["bond_max_by_start"]]
    payload["summary"] = {
        name: discriminator.summarize_domains(payload["domains"], name, 20)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }


def _write_result(path: Path, payload: dict) -> None:
    _recompute(payload)
    path.write_text(json.dumps(payload))


def _case(tmp_path: Path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = to_dict(load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000.yaml"
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

    ids = [
        "1gxlA02", "1nh8A02", "1qhdA01", "1qu3A05", "1s5lH00",
        "1vddA03", "1zcaA02", "1zu2A00", "2b5eA03", "2dgmA02",
        "2e9xD02", "2kl5A00", "2nluA02", "2ogyA00", "2xhgA02",
        "3fk5A01", "3ha4B00", "3k6yA01", "4agrB00", "4i9cA01",
    ]
    domain_list = tmp_path / "dev20.txt"
    domain_list.write_text("\n".join(ids) + "\n")
    domain_sha = _sha(domain_list)
    monkeypatch.setattr(discriminator, "EXPECTED_DOMAIN_LIST_SHA256", domain_sha)

    source_decision = tmp_path / "source_decision.json"
    source_decision.write_text(json.dumps({
        "status": discriminator.EXPECTED_SOURCE_STATUS,
        "candidate_checkpoint_sha256": checkpoint_sha,
        "external_development_authorized": False,
        "second_seed_scientifically_eligible": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }, sort_keys=True) + "\n")
    monkeypatch.setattr(
        discriminator, "EXPECTED_SOURCE_DECISION_SHA256", _sha(source_decision)
    )

    source_runner = tmp_path / "source_runner.sh"
    source_runner.write_text(
        'if [[ "$training_ab_status" == ADVANCE_PAPER_HORIZON_EXTERNAL20 ]]; then\n'
        'mkdir -p "$EXTERNAL_DATA_ROOT"\n'
        '"$PYTHON" scripts/download_mdcath.py\n'
        'cp "$RUN_DIR/training_ab_decision.json" "$RUN_DIR/decision.json"\n'
    )
    monkeypatch.setattr(discriminator, "EXPECTED_SOURCE_RUNNER_SHA256", _sha(source_runner))

    result = tmp_path / "result.json"
    rows = []
    for domain in discriminator.EXPECTED_DOMAINS:
        rows.append({
            "domain": domain,
            "residues_total": 64,
            "residues_evaluated": 64,
            "frames": 50,
            "starts": [0, 29],
            "methods": {
                "noop": _metrics(3.0),
                "one_step_persistence": _metrics(2.0),
                "mean": _metrics(1.5),
                "teacher_forced_mean": _metrics(1.0),
            },
        })
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 2000,
        "settings": {
            "ckpt": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "domain_list": str(domain_list),
            "domain_list_sha256": domain_sha,
            "domains": 3,
            "starts": 2,
            "steps": 20,
            "methods": "mean",
            "seed": 20260718,
            "noise_sigma": None,
            "integrator": "euler",
            "tau_max": 1.0,
            "terminal_denoise": False,
            "drift_anchor": "state",
            "project_v_atom_mask": False,
            "teacher_forced_mean": True,
            "per_start_geometry": True,
            "output": str(result),
        },
        "preprocessing": {"canon_symmetric": True},
        "delta_frames": 1,
        "domain_panel": {
            "path": str(domain_list),
            "sha256": domain_sha,
            "count": 20,
            "evaluated_count": 3,
        },
        "domains": rows,
    }
    _write_result(result, payload)
    return result, checkpoint, domain_list, source_decision, source_runner


def _adjudicate(case):
    return discriminator.adjudicate(*case)


def _set_wins(payload: dict, domain: int, start: int, wins: int) -> None:
    rows = payload["domains"][domain]["methods"]
    teacher = rows["teacher_forced_mean"]["rmsd_by_start"]
    persistence = rows["one_step_persistence"]["rmsd_by_start"]
    for step in range(1, 21):
        teacher[step][start] = 1.0 if step <= wins else 2.0
        persistence[step][start] = 2.0


def test_frozen_three_domain_selection_is_spread_0_9_19():
    ids = Path("configs/dev_20_length_proportional_seed0.txt").read_text().splitlines()
    assert select_validation_domains(ids, 3) == list(discriminator.EXPECTED_DOMAINS)


def test_signal_requires_14_wins_for_both_starts_and_aggregate(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    for domain in range(3):
        for start in range(2):
            _set_wins(payload, domain, start, 14)
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "FULL_TENSOR_ENDPOINT_SIGNAL_H20"
    assert report["teacher_strong_domains"] == 3


def test_both_starts_at_14_are_not_signal_when_aggregate_has_only_8_wins(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    for domain in range(3):
        teacher = payload["domains"][domain]["methods"]["teacher_forced_mean"][
            "rmsd_by_start"
        ]
        persistence = payload["domains"][domain]["methods"]["one_step_persistence"][
            "rmsd_by_start"
        ]
        for step in range(1, 21):
            persistence[step] = [2.0, 2.0]
            teacher[step][0] = 1.0 if step <= 14 else 4.0
            teacher[step][1] = 1.0 if step >= 7 else 4.0
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "INCONCLUSIVE_FULL_TENSOR_H20"
    assert all(
        row["teacher_forced_steps_better_than_one_step_persistence_by_start"]
        == [14, 14]
        and row["teacher_forced_steps_better_than_one_step_persistence_aggregate"] == 8
        for row in report["domain_evidence"]
    )


def test_13_wins_is_not_signal_and_6_wins_in_two_domains_is_failure(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    _set_wins(payload, 0, 0, 13)
    _write_result(case[0], payload)
    assert _adjudicate(case)["status"] == "INCONCLUSIVE_FULL_TENSOR_H20"

    case = _case(tmp_path / "failure", monkeypatch)
    payload = json.loads(case[0].read_text())
    _set_wins(payload, 0, 0, 6)
    _set_wins(payload, 1, 1, 6)
    _write_result(case[0], payload)
    assert _adjudicate(case)["status"] == "FULL_TENSOR_ENDPOINT_FAILURE_H20"


def test_bad_start_cannot_be_hidden_by_good_start_or_aggregate(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    for domain in (0, 1):
        _set_wins(payload, domain, 0, 20)
        _set_wins(payload, domain, 1, 0)
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "FULL_TENSOR_ENDPOINT_FAILURE_H20"
    assert report["teacher_failure_domains"] == 2


def test_equal_and_exact_margin_do_not_count_as_wins(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    teacher = payload["domains"][0]["methods"]["teacher_forced_mean"]["rmsd_by_start"]
    persistence = payload["domains"][0]["methods"]["one_step_persistence"]["rmsd_by_start"]
    for step in range(1, 21):
        teacher[step][0] = persistence[step][0]
        teacher[step][1] = persistence[step][1] - discriminator.RMSD_MARGIN
    _write_result(case[0], payload)
    evidence = _adjudicate(case)["domain_evidence"][0]
    assert evidence["teacher_forced_steps_better_than_one_step_persistence_by_start"] == [0, 0]


def test_autoregressive_geometry_does_not_change_endpoint_signal(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    method = payload["domains"][0]["methods"]["mean"]
    method["bond_max_by_start"][10] = [6.0, 6.0]
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "FULL_TENSOR_ENDPOINT_SIGNAL_H20"
    assert report["domain_evidence"][0]["autoregressive_first_nonphysical_step"] == 10


@pytest.mark.parametrize("mutation", ("decision", "runner", "checkpoint"))
def test_rejects_frozen_identity_mutations(tmp_path, monkeypatch, mutation):
    case = _case(tmp_path, monkeypatch)
    path = {"decision": case[3], "runner": case[4], "checkpoint": case[1]}[mutation]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _adjudicate(case)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("rmsd_by_start", -1.0, "nonnegative"),
        ("bond_mean_by_start", -1.0, "nonnegative"),
        ("bond_max_by_start", 3.0, "bond_max must be >= bond_mean"),
    ],
)
def test_rejects_invalid_per_start_values(tmp_path, monkeypatch, metric, value, message):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["teacher_forced_mean"][metric][1][0] = value
    _write_result(case[0], payload)
    with pytest.raises(ValueError, match=message):
        _adjudicate(case)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("checkpoint_step", 2000.9),
        ("delta_frames", 1.9),
        ("panel_count", 20.9),
        ("panel_evaluated_count", 3.0),
        ("settings_domains", 3.0),
        ("settings_terminal_denoise", 0),
    ],
)
def test_rejects_wrong_primitive_types(tmp_path, monkeypatch, mutation, value):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    targets = {
        "checkpoint_step": (payload, "checkpoint_step"),
        "delta_frames": (payload, "delta_frames"),
        "panel_count": (payload["domain_panel"], "count"),
        "panel_evaluated_count": (payload["domain_panel"], "evaluated_count"),
        "settings_domains": (payload["settings"], "domains"),
        "settings_terminal_denoise": (payload["settings"], "terminal_denoise"),
    }
    target, key = targets[mutation]
    target[key] = value
    _write_result(case[0], payload)
    with pytest.raises(ValueError):
        _adjudicate(case)


def test_rejects_settings_output_not_bound_to_result(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["settings"]["output"] = str(tmp_path / "other.json")
    _write_result(case[0], payload)
    with pytest.raises(ValueError, match="output path mismatch"):
        _adjudicate(case)


def test_all_authorization_outputs_are_false(tmp_path, monkeypatch):
    report = _adjudicate(_case(tmp_path, monkeypatch))
    for key in (
        "external_development_scientifically_eligible",
        "external_development_authorized",
        "second_seed_scientifically_eligible",
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
    ):
        assert report[key] is False


def test_runner_is_full_tensor_bounded_readback_closed_and_training_free():
    runner = Path("cloud/huawei/run_full_tensor_h20_discriminator.sh").read_text()
    assert discriminator.EXPECTED_CHECKPOINT_SHA256 in runner
    assert discriminator.EXPECTED_SOURCE_DECISION_SHA256 in runner
    assert discriminator.EXPECTED_SOURCE_RUNNER_SHA256 in runner
    assert "fc5f1e7b" not in runner and "H6_" not in runner
    assert '[[ "$HARD_STOP_MINUTES" == 55 ]]' in runner
    assert "Explicit timeout envelope is 44.5 minutes" in runner
    assert "at least 10.5 minutes" in runner
    assert 'ExecStart="/usr/bin/systemctl" "poweroff"' in runner
    assert "--domains 3 --starts 2 --steps 20 --methods mean --teacher-forced-mean" in runner
    assert "--per-start-geometry" in runner
    assert runner.count('--expected-result-output "$RUN_DIR/result.json"') == 4
    assert runner.count('verify_readback "$') == 4
    assert runner.count("--root-two") == 2
    assert '"$PYTHON" scripts/train_ddp.py' not in runner
    assert "torchrun" not in runner
    assert 'formal_training_authorized": False' in runner


def _initial_readback(tmp_path: Path, monkeypatch):
    case = _case(tmp_path / "case", monkeypatch)
    root = tmp_path / "readback"
    root.mkdir(parents=True)
    for source, name in (
        (case[0], "result.json"),
        (case[3], "source_decision.json"),
        (case[4], "source_runner.sh"),
    ):
        shutil.copyfile(source, root / name)
    for name in (
        "hard_stop_evidence.log", "obs_prefix_preflight.log", "pytest.log",
        "result.log", "runtime_evidence.log",
    ):
        (root / name).write_text(f"{name}\n")
    decision = _adjudicate(case)
    (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    run_id, commit = "20260722T150000Z", "d" * 40
    obs = "obs://deepjump-mdcath-cn4-ringochen/deepjump-diagnostics/full-tensor-h20/test"
    summary = {
        "status": decision["status"], "scope": decision["scope"],
        "source_status": decision["source_status"],
        "source_decision_sha256": decision["source_decision_sha256"],
        "source_runner_sha256": decision["source_runner_sha256"],
        "external_development_authorized": False, "second_seed_authorized": False,
        "untouched_confirmation_authorized": False, "formal_training_authorized": False,
        "run_id": run_id, "deployed_commit": commit,
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "checkpoint_source_commit": decision["checkpoint_source_commit"],
        "obs": obs, "completed_at": "2026-07-22T15:00:00+00:00",
    }
    (root / "summary.json").write_text(json.dumps(summary) + "\n")
    covered = sorted(readback.INITIAL_FILES - {"audit_sha256.txt"})
    (root / "audit_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in covered
    ))
    return root, case[1], case[2], run_id, commit, obs, case[0]


def test_readback_recomputes_and_requires_independent_roots(tmp_path, monkeypatch):
    case = _initial_readback(tmp_path, monkeypatch)
    kwargs = {
        "phase": "initial", "expected_run_id": case[3],
        "expected_deployed_commit": case[4], "expected_obs": case[5],
        "expected_result_output": case[6],
    }
    assert readback.verify(case[0], case[1], case[2], **kwargs)["status"] == (
        "FULL_TENSOR_H20_READBACK_PASS"
    )
    first, second = tmp_path / "one", tmp_path / "two"
    shutil.copytree(case[0], first)
    shutil.copytree(case[0], second)
    assert readback.verify_pair(first, second, case[1], case[2], **kwargs)[
        "independent_readbacks_verified"
    ] == 2
    with pytest.raises(ValueError, match="roots must differ"):
        readback.verify_pair(first, first, case[1], case[2], **kwargs)

    symlink = tmp_path / "symlink"
    symlink.symlink_to(case[0], target_is_directory=True)
    with pytest.raises(ValueError, match="missing readback root"):
        readback.verify(symlink, case[1], case[2], **kwargs)


def _completion_readback(tmp_path: Path, monkeypatch):
    case = _initial_readback(tmp_path, monkeypatch)
    root, checkpoint, domain_list, run_id, commit, obs, result_output = case
    initial_kwargs = {
        "phase": "initial",
        "expected_run_id": run_id,
        "expected_deployed_commit": commit,
        "expected_obs": obs,
        "expected_result_output": result_output,
    }
    initial = readback.verify(root, checkpoint, domain_list, **initial_kwargs)
    reports = {
        "initial_readback_one.json": dict(initial),
        "initial_readback_two.json": dict(initial),
        "initial_readback_pair.json": {
            **initial,
            "independent_readbacks_verified": 2,
        },
    }
    for name, report in reports.items():
        (root / name).write_text(json.dumps(report) + "\n")
    decision = json.loads((root / "decision.json").read_text())
    completion = {
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
        "completed_at": "2026-07-22T15:30:00+00:00",
    }
    (root / "readback_completion.json").write_text(json.dumps(completion) + "\n")
    completion_names = (
        "initial_readback_one.json",
        "initial_readback_two.json",
        "initial_readback_pair.json",
        "readback_completion.json",
    )
    (root / "completion_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in completion_names
    ))
    return case


def test_completion_rejects_reanchored_initial_inventory_digest(tmp_path, monkeypatch):
    case = _completion_readback(tmp_path, monkeypatch)
    root, checkpoint, domain_list, run_id, commit, obs, result_output = case
    kwargs = {
        "phase": "completion",
        "expected_run_id": run_id,
        "expected_deployed_commit": commit,
        "expected_obs": obs,
        "expected_result_output": result_output,
    }
    assert readback.verify(root, checkpoint, domain_list, **kwargs)["status"] == (
        "FULL_TENSOR_H20_READBACK_PASS"
    )

    wrong_digest = "0" * 64
    for name in (
        "initial_readback_one.json",
        "initial_readback_two.json",
        "initial_readback_pair.json",
    ):
        report = json.loads((root / name).read_text())
        report["inventory_sha256"] = wrong_digest
        (root / name).write_text(json.dumps(report) + "\n")
    completion = json.loads((root / "readback_completion.json").read_text())
    completion["initial_readback_one_sha256"] = _sha(root / "initial_readback_one.json")
    completion["initial_readback_two_sha256"] = _sha(root / "initial_readback_two.json")
    completion["initial_readback_pair_sha256"] = _sha(root / "initial_readback_pair.json")
    (root / "readback_completion.json").write_text(json.dumps(completion) + "\n")
    completion_names = (
        "initial_readback_one.json",
        "initial_readback_two.json",
        "initial_readback_pair.json",
        "readback_completion.json",
    )
    (root / "completion_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in completion_names
    ))
    with pytest.raises(ValueError, match="inventory_sha256 mismatch"):
        readback.verify(root, checkpoint, domain_list, **kwargs)
