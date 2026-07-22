import hashlib
import json
import shutil

import pytest
import torch

from scripts import adjudicate_teacher_update_projection as adjudicator
from scripts import verify_teacher_update_projection_readback as readback
from scripts.teacher_update_projection_eval import teacher_update_statistics


def _write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(value):
    return [[float(value), float(value)] for _ in range(20)]


def _projection_row(domain, starts, dot=0.25):
    metrics = {
        "dot_uv_by_start": _matrix(dot),
        "u_sq_by_start": _matrix(0.25),
        "v_sq_by_start": _matrix(1.0),
        "cosine_by_start": _matrix(dot / 0.5),
        "rho_by_start": _matrix(0.5),
        "raw_gain_by_start": _matrix(2 * dot - 0.25),
        "scaled_gain_by_start": _matrix(dot - 0.0625),
        "teacher_aligned_rmsd_by_start": _matrix(1.05),
        "persistence_aligned_rmsd_by_start": _matrix(1.0),
        "scaled_aligned_rmsd_by_start": _matrix(0.9),
        "scaled_bond_mean_by_start": _matrix(3.8),
        "scaled_bond_max_by_start": _matrix(4.1),
    }
    for name, values in list(metrics.items()):
        base = name.removesuffix("_by_start")
        metrics[base] = [max(row) if base == "scaled_bond_max" else sum(row) / 2 for row in values]
    return {
        "domain": domain,
        "residues_total": 80,
        "residues_evaluated": 80,
        "frames": starts[-1] + 21,
        "starts": starts,
        "metrics": metrics,
    }


def _h20_row(domain, starts):
    method = {
        "rmsd_by_start": [[0.0, 0.0], *_matrix(1.05)],
    }
    persistence = {
        "rmsd_by_start": [[0.0, 0.0], *_matrix(1.0)],
    }
    return {
        "domain": domain,
        "residues_total": 80,
        "residues_evaluated": 80,
        "frames": starts[-1] + 21,
        "starts": starts,
        "methods": {
            "teacher_forced_mean": method,
            "one_step_persistence": persistence,
        },
    }


def _case(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")
    domain_list = tmp_path / "domains.txt"
    domain_list.write_text("\n".join([f"domain-{index}" for index in range(20)]) + "\n")
    starts = [[0, 99], [0, 199], [0, 299]]
    h20_result = {
        "domains": [
            _h20_row(domain, domain_starts)
            for domain, domain_starts in zip(adjudicator.EXPECTED_DOMAINS, starts)
        ]
    }
    h20_decision = {"status": "ENDPOINT_OPERATOR_FAILURE_H20"}
    h20_completion = {
        "status": "OBS_DOUBLE_READBACK_PASS",
        "archived_decision_sha256": "pending",
    }
    h20_result_path = tmp_path / "h20-result.json"
    h20_decision_path = tmp_path / "h20-decision.json"
    h20_completion_path = tmp_path / "h20-completion.json"
    result_sha = _write_json(h20_result_path, h20_result)
    decision_sha = _write_json(h20_decision_path, h20_decision)
    h20_completion["archived_decision_sha256"] = decision_sha
    completion_sha = _write_json(h20_completion_path, h20_completion)
    monkeypatch.setattr(adjudicator, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(adjudicator, "DOMAIN_LIST_SHA256", hashlib.sha256(domain_list.read_bytes()).hexdigest())
    monkeypatch.setattr(adjudicator, "H20_RESULT_SHA256", result_sha)
    monkeypatch.setattr(adjudicator, "H20_DECISION_SHA256", decision_sha)
    monkeypatch.setattr(adjudicator, "H20_COMPLETION_SHA256", completion_sha)
    rows = [
        _projection_row(domain, domain_starts, dot=0.125)
        for domain, domain_starts in zip(adjudicator.EXPECTED_DOMAINS, starts)
    ]
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": adjudicator.CHECKPOINT_SHA256,
        "checkpoint_step": 2000,
        "settings": {
            "ckpt": str(checkpoint),
            "checkpoint_sha256": adjudicator.CHECKPOINT_SHA256,
            "domain_list": str(domain_list),
            "domain_list_sha256": adjudicator.DOMAIN_LIST_SHA256,
            "domains": 3,
            "starts": 2,
            "steps": 20,
            "calibration_domain": "1gxlA02",
            "output": str(tmp_path / "result.json"),
        },
        "preprocessing": {
            "canon_symmetric": True,
            "target_alignment": "full_structure_target_to_source_before_crop",
            "update_translation": "per_crop_update_mean_removed",
        },
        "delta_frames": 1,
        "domain_panel": {
            "path": str(domain_list),
            "sha256": adjudicator.DOMAIN_LIST_SHA256,
            "count": 20,
            "evaluated_count": 3,
        },
        "calibration": {
            "domain": "1gxlA02",
            "alpha": 0.5,
            "formula": "sum(dot_uv)/sum(u_sq)",
        },
        "domains": rows,
    }
    result_path = tmp_path / "result.json"
    _write_json(result_path, result)
    args = (
        result_path,
        checkpoint,
        domain_list,
        h20_result_path,
        h20_decision_path,
        h20_completion_path,
    )
    return result, result_path, args


def test_teacher_update_statistics_obeys_gain_identity():
    source = torch.zeros(2, 2, 3)
    target = torch.tensor([
        [[-1.0, 0, 0], [1.0, 0, 0]],
        [[-2.0, 0, 0], [2.0, 0, 0]],
    ])
    prediction = 0.5 * target
    stats = teacher_update_statistics(source, prediction, target)
    assert stats["cosine_by_start"] == pytest.approx([1.0, 1.0])
    assert stats["rho_by_start"] == pytest.approx([0.5, 0.5])
    for dot, u_sq, gain in zip(
        stats["dot_uv_by_start"],
        stats["u_sq_by_start"],
        stats["raw_gain_by_start"],
    ):
        assert gain == pytest.approx(2 * dot - u_sq)


def test_positive_scalar_rescale_requires_both_held_out_domains(tmp_path, monkeypatch):
    _, _, args = _case(tmp_path, monkeypatch)
    report = adjudicator.adjudicate(*args)
    assert report["status"] == "POSITIVE_SCALAR_RESCALE_SIGNAL"
    assert report["alpha"] == pytest.approx(0.5)
    assert all(
        row["held_out_pass"] is True
        for row in report["domain_evidence"]
        if row["role"] == "held_out"
    )


def test_directional_failure_when_two_domains_have_nonpositive_dot(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    for row in result["domains"][1:]:
        metrics = row["metrics"]
        metrics["dot_uv_by_start"] = _matrix(-0.1)
        metrics["dot_uv"] = [-0.1] * 20
        metrics["cosine_by_start"] = _matrix(-0.2)
        metrics["cosine"] = [-0.2] * 20
        metrics["raw_gain_by_start"] = _matrix(-0.45)
        metrics["raw_gain"] = [-0.45] * 20
        metrics["scaled_gain_by_start"] = _matrix(-0.1625)
        metrics["scaled_gain"] = [-0.1625] * 20
    _write_json(result_path, result)
    report = adjudicator.adjudicate(*args)
    assert report["status"] == "DIRECTIONAL_CA_ENDPOINT_FAILURE"
    assert report["directional_failure_domains"] == 2


def test_h20_rmsd_preprocessing_drift_is_rejected(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    result["domains"][0]["metrics"]["teacher_aligned_rmsd_by_start"][0][0] = 0.9
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="teacher"):
        adjudicator.adjudicate(*args)


def test_one_bad_scaled_start_blocks_amplitude_status(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    held_out = result["domains"][1]["metrics"]
    held_out["scaled_gain_by_start"] = [[0.1, -0.05] for _ in range(20)]
    held_out["scaled_gain"] = [0.025] * 20
    # Preserve the gain identity by changing the second start's dot product.
    held_out["dot_uv_by_start"] = [[0.1625, 0.0125] for _ in range(20)]
    held_out["dot_uv"] = [0.0875] * 20
    held_out["cosine_by_start"] = [[0.325, 0.025] for _ in range(20)]
    held_out["cosine"] = [0.175] * 20
    held_out["raw_gain_by_start"] = [[0.075, -0.225] for _ in range(20)]
    held_out["raw_gain"] = [-0.075] * 20
    held_out["scaled_aligned_rmsd_by_start"] = [[0.8, 1.1] for _ in range(20)]
    held_out["scaled_aligned_rmsd"] = [0.95] * 20
    _write_json(result_path, result)
    report = adjudicator.adjudicate(*args)
    assert report["status"] == "INCONCLUSIVE_PROJECTION"


def test_authorization_like_result_injection_is_rejected(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    result["formal_training_authorized"] = True
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="missing or extra"):
        adjudicator.adjudicate(*args)


def test_alpha_equal_to_one_is_not_a_rescale_signal(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    result["calibration"]["alpha"] = 1.0
    for row in result["domains"]:
        metrics = row["metrics"]
        metrics["dot_uv_by_start"] = _matrix(0.25)
        metrics["dot_uv"] = [0.25] * 20
        metrics["cosine_by_start"] = _matrix(0.5)
        metrics["cosine"] = [0.5] * 20
        metrics["raw_gain_by_start"] = _matrix(0.25)
        metrics["raw_gain"] = [0.25] * 20
        metrics["scaled_gain_by_start"] = _matrix(0.25)
        metrics["scaled_gain"] = [0.25] * 20
    _write_json(result_path, result)
    report = adjudicator.adjudicate(*args)
    assert report["status"] == "INCONCLUSIVE_PROJECTION"


def test_scaled_must_improve_raw_teacher(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    for index in (1, 2):
        metrics = result["domains"][index]["metrics"]
        metrics["teacher_aligned_rmsd_by_start"] = _matrix(0.95)
        metrics["teacher_aligned_rmsd"] = [0.95] * 20
        metrics["scaled_aligned_rmsd_by_start"] = _matrix(0.98)
        metrics["scaled_aligned_rmsd"] = [0.98] * 20
        result_path_h20 = args[3]
        h20 = json.loads(result_path_h20.read_text())
        h20["domains"][index]["methods"]["teacher_forced_mean"]["rmsd_by_start"] = [
            [0.0, 0.0], *_matrix(0.95)
        ]
        _write_json(result_path_h20, h20)
        monkeypatch.setattr(
            adjudicator,
            "H20_RESULT_SHA256",
            hashlib.sha256(result_path_h20.read_bytes()).hexdigest(),
        )
    _write_json(result_path, result)
    report = adjudicator.adjudicate(*args)
    assert report["status"] == "INCONCLUSIVE_PROJECTION"


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("scaled_aligned_rmsd_by_start", -1.0, "non-negative"),
        ("scaled_bond_max_by_start", -1.0, "non-negative"),
        ("scaled_bond_max_by_start", 3.7, "below its mean"),
    ],
)
def test_nonphysical_metric_cannot_pass(tmp_path, monkeypatch, metric, value, message):
    result, result_path, args = _case(tmp_path, monkeypatch)
    metrics = result["domains"][0]["metrics"]
    metrics[metric] = _matrix(value)
    aggregate = metric.removesuffix("_by_start")
    metrics[aggregate] = [value] * 20
    _write_json(result_path, result)
    with pytest.raises(ValueError, match=message):
        adjudicator.adjudicate(*args)


def test_cauchy_violating_projection_is_rejected(tmp_path, monkeypatch):
    result, result_path, args = _case(tmp_path, monkeypatch)
    metrics = result["domains"][0]["metrics"]
    metrics["dot_uv_by_start"] = _matrix(1.0)
    metrics["dot_uv"] = [1.0] * 20
    _write_json(result_path, result)
    with pytest.raises(ValueError, match="Cauchy"):
        adjudicator.adjudicate(*args)


def test_initial_readback_recomputes_exact_decision(tmp_path, monkeypatch):
    case_root = tmp_path / "case"
    case_root.mkdir()
    _, result_path, args = _case(case_root, monkeypatch)
    checkpoint, domain_list = args[1], args[2]
    root = tmp_path / "readback"
    root.mkdir()
    for source, name in (
        (result_path, "result.json"),
        (args[3], "h20_result.json"),
        (args[4], "h20_decision.json"),
        (args[5], "h20_readback_completion.json"),
    ):
        shutil.copyfile(source, root / name)
    for name in (
        "hard_stop_evidence.log", "obs_prefix_preflight.log", "pytest.log",
        "result.log", "runtime_evidence.log",
    ):
        (root / name).write_text(f"{name}\n")
    decision = adjudicator.adjudicate(*args)
    (root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    run_id = "20260722T160000Z"
    commit = "d" * 40
    obs = "obs://deepjump-mdcath-cn4-ringochen/deepjump-diagnostics/teacher-update-projection/test"
    summary = {
        "status": decision["status"],
        "scope": decision["scope"],
        "h20_result_sha256": decision["h20_result_sha256"],
        "h20_decision_sha256": decision["h20_decision_sha256"],
        "h20_completion_sha256": decision["h20_completion_sha256"],
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "run_id": run_id,
        "deployed_commit": commit,
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "obs": obs,
        "completed_at": "2026-07-22T16:00:00+00:00",
    }
    (root / "summary.json").write_text(json.dumps(summary) + "\n")
    covered = sorted(readback.INITIAL_FILES - {"audit_sha256.txt"})
    (root / "audit_sha256.txt").write_text("".join(
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n"
        for name in covered
    ))
    report = readback.verify(
        root,
        checkpoint,
        domain_list,
        phase="initial",
        expected_run_id=run_id,
        expected_deployed_commit=commit,
        expected_obs=obs,
    )
    assert report["status"] == "TEACHER_UPDATE_PROJECTION_READBACK_PASS"
    (root / "result.json").write_text((root / "result.json").read_text() + " \n")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        readback.verify(
            root,
            checkpoint,
            domain_list,
            phase="initial",
            expected_run_id=run_id,
            expected_deployed_commit=commit,
            expected_obs=obs,
        )
