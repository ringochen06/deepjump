import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from deepjump.config import load_config, to_dict
import scripts.adjudicate_scalar_feedback_h20 as discriminator
from scripts.rollout_robustness_eval import select_validation_domains
import scripts.verify_scalar_feedback_h20_readback as readback


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(
    rmsd: list[float],
    *,
    bond_mean: list[float] | None = None,
    bond_max: list[float] | None = None,
) -> dict:
    bond_mean = bond_mean or [3.8] * 21
    bond_max = bond_max or [4.2] * 21
    return {
        "rmsd": rmsd,
        "rmsd_by_start": [[value, value] for value in rmsd],
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


def _write_result(path: Path, payload: dict) -> None:
    for row in payload["domains"]:
        for method in row["methods"].values():
            method["rmsd_by_start"] = [[value, value] for value in method["rmsd"]]
    payload["summary"] = {
        name: discriminator.summarize_domains(payload["domains"], name, 20)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    path.write_text(json.dumps(payload))


def _h6_payload(checkpoint_sha: str, domain_sha: str) -> dict:
    return {
        "status": discriminator.EXPECTED_H6_STATUS,
        "scope": "scalar_step2000_teacher_forced_vs_autoregressive_h1_h6_no_training",
        "checkpoint_step": 2000,
        "checkpoint_profile": discriminator.PAPER_SCALAR_VALUE_PROFILE,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_source_commit": discriminator.EXPECTED_SOURCE_COMMIT,
        "source_evidence_sha256": discriminator.EXPECTED_SOURCE_EVIDENCE_SHA256,
        "domain_list_sha256": domain_sha,
        "domain": "1gxlA02",
        "starts": 5,
        "steps": 6,
        "teacher_forced_first_nonphysical_step": None,
        "autoregressive_first_nonphysical_step": None,
        "teacher_forced_steps_better_than_one_step_persistence": 4,
        "decision_rule": {},
        "external_development_scientifically_eligible": False,
        "external_development_authorized": False,
        "second_seed_scientifically_eligible": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


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
    ids = [
        "1gxlA02", "1nh8A02", "1qhdA01", "1qu3A05", "1s5lH00",
        "1vddA03", "1zcaA02", "1zu2A00", "2b5eA03", "2dgmA02",
        "2e9xD02", "2kl5A00", "2nluA02", "2ogyA00", "2xhgA02",
        "3fk5A01", "3ha4B00", "3k6yA01", "4agrB00", "4i9cA01",
    ]
    domain_list.write_text("\n".join(ids) + "\n")
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
    source_sha = _sha(source)
    monkeypatch.setattr(discriminator, "EXPECTED_SOURCE_EVIDENCE_SHA256", source_sha)

    h6 = tmp_path / "h6_decision.json"
    h6.write_text(json.dumps(_h6_payload(checkpoint_sha, domain_sha), indent=2) + "\n")
    h6_sha = _sha(h6)
    monkeypatch.setattr(discriminator, "EXPECTED_H6_DECISION_SHA256", h6_sha)
    completion = tmp_path / "h6_readback_completion.json"
    completion.write_text(json.dumps({
        "status": "OBS_DOUBLE_READBACK_PASS",
        "decision_status": discriminator.EXPECTED_H6_STATUS,
        "run_id": "20260722T115322Z",
        "commit": "279b9fd628725f36cd2d1508e7222110ba0fa461",
        "archived_decision_sha256": h6_sha,
        "recomputed_decision_sha256": h6_sha,
        "independent_readbacks_verified": 2,
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }, sort_keys=True) + "\n")
    completion_sha = _sha(completion)
    monkeypatch.setattr(discriminator, "EXPECTED_H6_COMPLETION_SHA256", completion_sha)

    persistence = [0.0] + [2.0] * 20
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
                "noop": _metrics([0.0] + [3.0] * 20),
                "one_step_persistence": _metrics(persistence),
                "mean": _metrics([0.0] + [1.5] * 20),
                "teacher_forced_mean": _metrics([0.0] + [1.0] * 20),
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
    return result, checkpoint, domain_list, source, h6, completion


def _adjudicate(case):
    return discriminator.adjudicate(*case)


def _make_autoregressive_nonphysical(payload: dict, domain_indexes=(0, 1)) -> None:
    for index in domain_indexes:
        payload["domains"][index]["methods"]["mean"]["bond_max"][10] = 5.6
        payload["domains"][index]["methods"]["mean"]["bond_max_by_start"][10] = [5.6, 5.6]


def _teacher_wins(payload: dict, domain_index: int, wins: int) -> None:
    teacher = payload["domains"][domain_index]["methods"]["teacher_forced_mean"]["rmsd"]
    teacher[1:] = [1.0] * wins + [2.0] * (20 - wins)


def test_frozen_three_domain_selection_is_spread_0_9_19():
    ids = Path("configs/dev_20_length_proportional_seed0.txt").read_text().splitlines()
    assert select_validation_domains(ids, 3) == list(discriminator.EXPECTED_DOMAINS)


def test_h20_classifies_feedback_shift_at_14_win_boundary(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    _make_autoregressive_nonphysical(payload)
    for index in range(3):
        _teacher_wins(payload, index, 14)
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "FEEDBACK_DISTRIBUTION_SHIFT_H20"
    assert report["teacher_strong_domains"] == 3
    assert report["autoregressive_failure_domains"] == 2
    assert all(row["teacher_forced_steps_better_than_one_step_persistence"] == 14
               for row in report["domain_evidence"])


def test_h20_classifies_endpoint_failure_at_6_win_boundary(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    _teacher_wins(payload, 0, 6)
    _teacher_wins(payload, 1, 6)
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "ENDPOINT_OPERATOR_FAILURE_H20"
    assert report["teacher_failure_domains"] == 2


def test_h20_endpoint_failure_accepts_teacher_geometry_failure(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["domains"][0]["methods"]["teacher_forced_mean"]["bond_mean"][20] = 3.1
    payload["domains"][0]["methods"]["teacher_forced_mean"]["bond_mean_by_start"][20] = [3.1, 3.1]
    payload["domains"][1]["methods"]["teacher_forced_mean"]["bond_max"][20] = 5.6
    payload["domains"][1]["methods"]["teacher_forced_mean"]["bond_max_by_start"][20] = [5.6, 5.6]
    _write_result(case[0], payload)
    assert _adjudicate(case)["status"] == "ENDPOINT_OPERATOR_FAILURE_H20"


def test_h20_is_inconclusive_for_one_autoregressive_failure(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    _make_autoregressive_nonphysical(payload, (0,))
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["status"] == "INCONCLUSIVE_SCALAR_FEEDBACK_H20"
    assert report["teacher_strong_domains"] == 3
    assert report["autoregressive_failure_domains"] == 1


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("settings", "starts"), 5, "settings mismatch"),
        (("domain_panel", "evaluated_count"), 1, "three domains"),
        (("domains", 1, "domain"), "wrong", "domain identity"),
        (("domains", 0, "starts"), [0, 28], "start identity"),
        (("domains", 0, "residues_total"), 0, "positive integer"),
        (("domains", 0, "residues_evaluated"), 63, "crop length"),
    ],
)
def test_h20_rejects_identity_mutations(tmp_path, monkeypatch, path, value, message):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_result(case[0], payload)
    with pytest.raises(ValueError, match=message):
        _adjudicate(case)


def test_h20_rejects_nonfinite_and_reanchored_aggregate(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    teacher = payload["domains"][0]["methods"]["teacher_forced_mean"]
    teacher["rmsd"][3] = 0.1
    payload["summary"] = {
        name: discriminator.summarize_domains(payload["domains"], name, 20)
        for name in ("noop", "one_step_persistence", "mean", "teacher_forced_mean")
    }
    case[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match rmsd_by_start"):
        _adjudicate(case)


def test_h20_rejects_one_collapsed_start_hidden_by_pooled_bond_mean(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    teacher = payload["domains"][0]["methods"]["teacher_forced_mean"]
    teacher["bond_mean"][10] = 3.4
    teacher["bond_mean_by_start"][10] = [2.6, 4.2]
    teacher["bond_max"][10] = 4.4
    teacher["bond_max_by_start"][10] = [4.0, 4.4]
    _write_result(case[0], payload)
    report = _adjudicate(case)
    assert report["domain_evidence"][0]["teacher_forced_first_nonphysical_step"] == 10
    assert report["teacher_strong_domains"] == 2
    assert report["status"] == "INCONCLUSIVE_SCALAR_FEEDBACK_H20"

    payload = json.loads(case[0].read_text())
    teacher = payload["domains"][0]["methods"]["teacher_forced_mean"]
    teacher["rmsd"][3] = float("nan")
    teacher["rmsd_by_start"][3] = [float("nan"), float("nan")]
    case[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be finite"):
        _adjudicate(case)


@pytest.mark.parametrize("source", ("decision", "completion"))
def test_h20_rejects_h6_sha_or_status_mutation(tmp_path, monkeypatch, source):
    case = _case(tmp_path, monkeypatch)
    path = case[4] if source == "decision" else case[5]
    payload = json.loads(path.read_text())
    if source == "decision":
        payload["status"] = "FEEDBACK_DISTRIBUTION_SHIFT"
    else:
        payload["decision_status"] = "FEEDBACK_DISTRIBUTION_SHIFT"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match=f"H6 {source} SHA256 mismatch"):
        _adjudicate(case)


@pytest.mark.parametrize("location", ("result", "settings", "method"))
def test_h20_rejects_authorization_like_field_injection(tmp_path, monkeypatch, location):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    if location == "result":
        payload["formal_training_authorized"] = True
    elif location == "settings":
        payload["settings"]["formal_training_authorized"] = True
    else:
        payload["domains"][0]["methods"]["mean"]["formal_training_authorized"] = True
    _write_result(case[0], payload)
    with pytest.raises(ValueError, match="missing or extra"):
        _adjudicate(case)


def test_h20_all_authorization_outputs_are_false(tmp_path, monkeypatch):
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


def test_h20_runner_is_bounded_exact_readback_closed_and_training_free():
    runner = Path("cloud/huawei/run_scalar_feedback_h20_discriminator.sh").read_text()
    assert "HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}" in runner
    assert '[[ "$HARD_STOP_MINUTES" == 45 ]]' in runner
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert 'ExecStart="/usr/bin/systemctl" "poweroff"' in runner
    assert "timeout --signal=TERM --kill-after=30s 24m" in runner
    assert "--domains 3 --starts 2 --steps 20 --methods mean --teacher-forced-mean" in runner
    assert "--per-start-geometry" in runner
    assert "1gxlA02" in runner and "2dgmA02" in runner and "4i9cA01" in runner
    assert "H6_DECISION_SHA256=ace2b577" in runner
    assert "H6_COMPLETION_SHA256=90a03a75" in runner
    assert "Maximum declared success-path timeout envelope is 40.5 minutes" in runner
    assert runner.count('verify_readback "$') == 4
    assert runner.count("--root-two") == 2
    assert "scripts/verify_obsutil_empty_prefix.py" in runner
    assert '"$PYTHON" scripts/train_ddp.py' not in runner
    assert 'formal_training_authorized": False' in runner


def _make_readback(tmp_path: Path, monkeypatch, *, completion: bool):
    case = _case(tmp_path / "case", monkeypatch)
    result, checkpoint, domain_list, source, h6, h6_completion = case
    root = tmp_path / "readback"
    root.mkdir(parents=True)
    for source_path, name in (
        (result, "result.json"),
        (source, "source_training_evidence.json"),
        (h6, "h6_decision.json"),
        (h6_completion, "h6_readback_completion.json"),
    ):
        shutil.copyfile(source_path, root / name)
    for name in (
        "hard_stop_evidence.log", "obs_prefix_preflight.log", "pytest.log",
        "result.log", "runtime_evidence.log",
    ):
        (root / name).write_text(f"{name}\n")
    decision = discriminator.adjudicate(*case)
    (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    run_id = "20260722T140000Z"
    commit = "d" * 40
    obs = "obs://deepjump-mdcath-cn4-ringochen/deepjump-diagnostics/scalar-feedback-h20/test"
    summary = {
        "status": decision["status"],
        "scope": decision["scope"],
        "h6_decision_sha256": decision["h6_decision_sha256"],
        "h6_completion_sha256": decision["h6_completion_sha256"],
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "run_id": run_id,
        "deployed_commit": commit,
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "checkpoint_source_commit": decision["checkpoint_source_commit"],
        "obs": obs,
        "completed_at": "2026-07-22T14:00:00+00:00",
    }
    (root / "summary.json").write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    covered = sorted(readback.INITIAL_FILES - {"audit_sha256.txt"})
    (root / "audit_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in covered
    ))
    if completion:
        initial = readback.verify(
            root, checkpoint, domain_list, phase="initial",
            expected_run_id=run_id, expected_deployed_commit=commit, expected_obs=obs,
        )
        for name, report in (
            ("initial_readback_one.json", initial),
            ("initial_readback_two.json", initial),
            ("initial_readback_pair.json", {**initial, "independent_readbacks_verified": 2}),
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
            **{key: False for key in readback.AUTHORIZATION_KEYS},
            "completed_at": "2026-07-22T14:01:00+00:00",
        }
        (root / "readback_completion.json").write_text(
            json.dumps(completion_payload, separators=(",", ":")) + "\n"
        )
        names = (
            "initial_readback_one.json", "initial_readback_two.json",
            "initial_readback_pair.json", "readback_completion.json",
        )
        (root / "completion_sha256.txt").write_text("".join(
            f"{_sha(root / name)}  {name}\n" for name in names
        ))
    return root, checkpoint, domain_list, run_id, commit, obs


def _verify_readback(case, phase: str):
    return readback.verify(
        case[0], case[1], case[2], phase=phase,
        expected_run_id=case[3], expected_deployed_commit=case[4], expected_obs=case[5],
    )


def test_h20_readback_recomputes_initial_and_completion(tmp_path, monkeypatch):
    initial = _make_readback(tmp_path / "initial", monkeypatch, completion=False)
    assert _verify_readback(initial, "initial")["status"] == "SCALAR_FEEDBACK_H20_READBACK_PASS"
    completed = _make_readback(tmp_path / "completed", monkeypatch, completion=True)
    assert _verify_readback(completed, "completion")["completion_sha256"] is not None


def test_h20_readback_rejects_extra_and_reanchored_decision(tmp_path, monkeypatch):
    case = _make_readback(tmp_path / "extra", monkeypatch, completion=False)
    (case[0] / "injected.json").write_text("{}\n")
    with pytest.raises(ValueError, match="missing or extra"):
        _verify_readback(case, "initial")

    case = _make_readback(tmp_path / "decision", monkeypatch, completion=False)
    root = case[0]
    decision = json.loads((root / "decision.json").read_text())
    decision["status"] = "FEEDBACK_DISTRIBUTION_SHIFT_H20"
    (root / "decision.json").write_text(json.dumps(decision) + "\n")
    summary = json.loads((root / "summary.json").read_text())
    summary["status"] = decision["status"]
    (root / "summary.json").write_text(json.dumps(summary) + "\n")
    covered = sorted(readback.INITIAL_FILES - {"audit_sha256.txt"})
    (root / "audit_sha256.txt").write_text("".join(
        f"{_sha(root / name)}  {name}\n" for name in covered
    ))
    with pytest.raises(ValueError, match="differs from recomputed"):
        _verify_readback(case, "initial")


def test_h20_readback_pair_requires_independent_exact_inventories(tmp_path, monkeypatch):
    case = _make_readback(tmp_path / "source", monkeypatch, completion=True)
    first, second = tmp_path / "one", tmp_path / "two"
    shutil.copytree(case[0], first)
    shutil.copytree(case[0], second)
    kwargs = {
        "phase": "completion", "expected_run_id": case[3],
        "expected_deployed_commit": case[4], "expected_obs": case[5],
    }
    assert readback.verify_pair(first, second, case[1], case[2], **kwargs)[
        "independent_readbacks_verified"
    ] == 2
    with pytest.raises(ValueError, match="roots must differ"):
        readback.verify_pair(first, first, case[1], case[2], **kwargs)
