#!/usr/bin/env python
"""Independently certify a paper-horizon run before seed-1 execution."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.adjudicate_guarded_endpoint_panel import adjudicate as adjudicate_guarded
from scripts.adjudicate_paper_horizon_ab import adjudicate as adjudicate_horizon
from scripts.guarded_endpoint_panel_eval import (
    HORIZON_AB_BASELINE_PROFILE,
    PAPER_HORIZON_PROFILE,
)
from scripts.validate_training_checkpoint import validate_checkpoint
from scripts.verify_obsutil_empty_prefix import prefix_object_count


EXPECTED_PASS_AUDIT_MANIFEST = frozenset(
    {
        "obs_prefix_preflight.log",
        "pytest.log",
        "data_audit.json",
        "train_baseline.log",
        "baseline_checkpoint_1000_gate.json",
        "baseline_checkpoint_2000_gate.json",
        "train_candidate.log",
        "candidate_checkpoint_1000_gate.json",
        "candidate_checkpoint_2000_gate.json",
        "baseline_runtime_probe.json",
        "baseline_panel.json",
        "baseline_panel.log",
        "baseline_decision.json",
        "candidate_runtime_probe.json",
        "candidate_panel.json",
        "candidate_panel.log",
        "candidate_decision.json",
        "training_ab_decision.json",
        "external_download.log",
        "external_data_audit.json",
        "external_baseline_runtime_probe.json",
        "external_baseline_panel.json",
        "external_baseline_panel.log",
        "external_baseline_decision.json",
        "external_candidate_runtime_probe.json",
        "external_candidate_panel.json",
        "external_candidate_panel.log",
        "external_candidate_decision.json",
        "decision.json",
        "external_status.json",
        "summary.json",
    }
)
EXPECTED_UNMANIFESTED_AUDIT = frozenset(
    {
        "runner.log",
        "audit_sha256.txt",
        "baseline_sha256.txt",
        "candidate_sha256.txt",
        "authorization.json",
        "readback_completion.json",
        "final_markers.sha256",
    }
)
AUTHORIZATION_KEYS = frozenset(
    {
        "run_id",
        "commit",
        "scientific_status",
        "decision_sha256",
        "audit_manifest_sha256",
        "baseline_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "second_seed_scientifically_eligible",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
        "completed_at",
        "status",
        "second_seed_authorized",
        "authorization_requires_independent_readback",
    }
)
COMPLETION_KEYS = AUTHORIZATION_KEYS - {"authorization_requires_independent_readback"}
SUMMARY_KEYS = frozenset(
    {
        "status",
        "external_development_authorized",
        "second_seed_scientifically_eligible",
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
        "run_id",
        "commit",
        "baseline_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "obs",
        "completed_at",
    }
)
EXPECTED_TRAINING_DATA_AUDIT = {
    "status": "PASS",
    "h5_files": 1000,
    "h5_bytes": 668131379559,
    "manifest_domains": 1000,
    "manifest_trajectories": 25000,
    "subset_sha256": "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734",
    "unresolved_failures": 0,
    "hdf5_samples": 5,
}
EXPECTED_EXTERNAL_DATA_AUDIT = {
    "status": "PASS",
    "domain_list_sha256": "9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb",
    "h5_files": 20,
    "total_bytes": 14236836972,
    "trajectories": 500,
    "unresolved_failures": 0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"{path}:{line_number}: malformed sha256 line")
        digest, name = line[:64], line[66:]
        if any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: malformed sha256 digest")
        if not name or name in entries:
            raise ValueError(f"{path}:{line_number}: empty or duplicate path")
        entries[name] = digest
    if not entries:
        raise ValueError(f"{path}: empty sha256 manifest")
    return entries


def _safe_member(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe manifest path: {name}")
    root_resolved = root.resolve()
    member = (root / relative).resolve()
    try:
        member.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes root: {name}") from exc
    if not member.is_file():
        raise ValueError(f"manifest member is missing or not a file: {name}")
    return member


def _verify_manifest(
    manifest: Path,
    root: Path,
    *,
    required: set[str] | None = None,
    exact: set[str] | None = None,
) -> dict[str, str]:
    entries = _manifest_entries(manifest)
    normalized = {name.removeprefix("./") for name in entries}
    if len(normalized) != len(entries):
        raise ValueError(f"{manifest}: duplicate normalized paths")
    if required is not None and not required.issubset(normalized):
        missing = sorted(required - normalized)
        raise ValueError(f"{manifest}: missing required entries: {missing}")
    if exact is not None and normalized != exact:
        raise ValueError(f"{manifest}: expected exact entries {sorted(exact)}")
    for name, expected in entries.items():
        actual = _sha256(_safe_member(root, name))
        if actual != expected:
            raise ValueError(f"{manifest}: sha256 mismatch for {name}")
    return {name.removeprefix("./"): digest for name, digest in entries.items()}


def _inventory(root: Path) -> list[tuple[str, int, str]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"readback contains a symlink: {path}")
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path)))
    if not rows:
        raise ValueError(f"empty readback root: {root}")
    return rows


def _inventory_sha256(rows: list[tuple[str, int, str]]) -> str:
    encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inode_set(root: Path) -> set[tuple[int, int]]:
    return {
        (path.stat().st_dev, path.stat().st_ino)
        for path in root.rglob("*")
        if path.is_file()
    }


def _require_exact_json(actual_path: Path, recomputed: dict, label: str) -> None:
    if _load_object(actual_path) != recomputed:
        raise ValueError(f"{label}: recomputed adjudication mismatch")


def _require_false(payload: dict, keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if payload.get(key) is not False:
            raise ValueError(f"{label}: expected {key}=false")


def _require_exact_keys(payload: dict, expected: frozenset[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label}: exact schema mismatch")


def _require_utc_timestamp(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label}: completed_at must be a string")
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label}: completed_at is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        raise ValueError(f"{label}: completed_at must be UTC")


def _certify_one(
    readback_root: Path,
    repo: Path,
    *,
    expected_run_id: str,
    expected_source_commit: str,
    expected_verifier_commit: str,
    expected_obs: str,
) -> dict:
    audit_root = readback_root / "audit"
    baseline_root = readback_root / "baseline"
    candidate_root = readback_root / "candidate"
    for root in (audit_root, baseline_root, candidate_root):
        if not root.is_dir():
            raise ValueError(f"missing readback directory: {root}")

    audit_manifest = audit_root / "audit_sha256.txt"
    audit_entries = _verify_manifest(
        audit_manifest, audit_root, exact=set(EXPECTED_PASS_AUDIT_MANIFEST)
    )
    actual_audit_files = {
        path.relative_to(audit_root).as_posix()
        for path in audit_root.rglob("*")
        if path.is_file()
    }
    expected_audit_files = set(EXPECTED_PASS_AUDIT_MANIFEST) | set(
        EXPECTED_UNMANIFESTED_AUDIT
    )
    if actual_audit_files != expected_audit_files:
        raise ValueError("audit readback has missing or extra files")
    checkpoint_files = {
        "config.json",
        "history.json",
        "last.ckpt",
        "ckpt_1000.pt",
        "ckpt_2000.pt",
    }
    _verify_manifest(
        audit_root / "baseline_sha256.txt",
        baseline_root,
        exact=checkpoint_files,
    )
    if _sha256(baseline_root / "last.ckpt") != _sha256(
        baseline_root / "ckpt_2000.pt"
    ):
        raise ValueError("baseline last.ckpt is not ckpt_2000.pt")
    if _sha256(candidate_root / "last.ckpt") != _sha256(
        candidate_root / "ckpt_2000.pt"
    ):
        raise ValueError("candidate last.ckpt is not ckpt_2000.pt")
    _verify_manifest(
        audit_root / "candidate_sha256.txt",
        candidate_root,
        exact=checkpoint_files,
    )
    for root, label in ((baseline_root, "baseline"), (candidate_root, "candidate")):
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual != checkpoint_files:
            raise ValueError(f"{label} readback has missing or extra files")
    _verify_manifest(
        audit_root / "final_markers.sha256",
        audit_root,
        exact={"authorization.json", "readback_completion.json"},
    )

    if prefix_object_count(
        (audit_root / "obs_prefix_preflight.log").read_text()
    ) != 0:
        raise ValueError("OBS prefix preflight was not uniquely empty")

    decision_path = audit_root / "decision.json"
    summary_path = audit_root / "summary.json"
    authorization_path = audit_root / "authorization.json"
    completion_path = audit_root / "readback_completion.json"
    decision = _load_object(decision_path)
    summary = _load_object(summary_path)
    authorization = _load_object(authorization_path)
    completion = _load_object(completion_path)
    data_audit = _load_object(audit_root / "data_audit.json")
    external_data_audit = _load_object(audit_root / "external_data_audit.json")
    baseline_gate = _load_object(audit_root / "baseline_checkpoint_2000_gate.json")
    candidate_gate = _load_object(audit_root / "candidate_checkpoint_2000_gate.json")
    _require_exact_keys(authorization, AUTHORIZATION_KEYS, "authorization")
    _require_exact_keys(completion, COMPLETION_KEYS, "completion")
    _require_exact_keys(summary, SUMMARY_KEYS, "summary")
    _require_utc_timestamp(authorization["completed_at"], "authorization")
    _require_utc_timestamp(completion["completed_at"], "completion")
    _require_utc_timestamp(summary["completed_at"], "summary")
    if authorization["completed_at"] != completion["completed_at"]:
        raise ValueError("authorization/completion completed_at mismatch")

    for label, payload, expected in (
        ("training", data_audit, EXPECTED_TRAINING_DATA_AUDIT),
        ("external", external_data_audit, EXPECTED_EXTERNAL_DATA_AUDIT),
    ):
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"{label} data audit {key} mismatch")
    gate_expectations = (
        (baseline_gate, 1000, "baseline"),
        (candidate_gate, 500000, "candidate"),
    )
    for gate, horizon, label in gate_expectations:
        if (
            gate.get("status") != "PASS"
            or gate.get("checkpoint_step") != 2000
            or gate.get("world_size") != 8
            or gate.get("lr_horizon_steps") != horizon
            or gate.get("nonfinite_model_tensors") != []
        ):
            raise ValueError(f"{label} checkpoint gate mismatch")

    permission_keys = (
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
    )
    if decision.get("status") != "PASS_PAPER_HORIZON_EXTERNAL20":
        raise ValueError("scientific decision does not authorize seed-1 eligibility")
    if decision.get("second_seed_scientifically_eligible") is not True:
        raise ValueError("scientific decision lacks seed-1 eligibility")
    _require_false(decision, permission_keys, "decision")
    if summary.get("status") != decision["status"]:
        raise ValueError("summary scientific status mismatch")
    if summary.get("second_seed_scientifically_eligible") is not True:
        raise ValueError("summary lacks seed-1 eligibility")
    if summary.get("external_development_authorized") != decision.get(
        "external_development_authorized"
    ):
        raise ValueError("summary external authorization mismatch")
    _require_false(summary, permission_keys, "summary")

    baseline_sha = _sha256(baseline_root / "ckpt_2000.pt")
    candidate_sha = _sha256(candidate_root / "ckpt_2000.pt")
    for root, horizon, label in (
        (baseline_root, 1000, "baseline"),
        (candidate_root, 500000, "candidate"),
    ):
        _, errors = validate_checkpoint(
            root / "ckpt_2000.pt",
            2000,
            8,
            root / "history.json",
            history_mode="contains",
            expected_delta=1,
            require_full_tensor=True,
            expected_lr_horizon_steps=horizon,
        )
        if errors:
            raise ValueError(f"{label} checkpoint validation failed: {errors}")
    expected_common = {
        "run_id": expected_run_id,
        "commit": expected_source_commit,
        "scientific_status": decision["status"],
        "decision_sha256": _sha256(decision_path),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "second_seed_scientifically_eligible": True,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }
    for label, payload in (("authorization", authorization), ("completion", completion)):
        for key, expected in expected_common.items():
            if payload.get(key) != expected:
                raise ValueError(f"{label}: {key} mismatch")
    if (
        authorization.get("status") != "SECOND_SEED_AUTHORIZED"
        or authorization.get("second_seed_authorized") is not True
        or authorization.get("authorization_requires_independent_readback") is not True
    ):
        raise ValueError("authorization marker is not independently gated")
    if (
        completion.get("status") != "OBS_DOUBLE_READBACK_PASS"
        or completion.get("second_seed_authorized") is not False
    ):
        raise ValueError("completion marker mismatch")

    decision_checkpoint_fields = {
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
    }
    for key, expected in decision_checkpoint_fields.items():
        if decision.get(key) != expected or summary.get(key) != expected:
            raise ValueError(f"decision/summary {key} mismatch")
    summary_expected = {
        "run_id": expected_run_id,
        "commit": expected_source_commit,
        "obs": expected_obs,
    }
    for key, expected in summary_expected.items():
        if summary.get(key) != expected:
            raise ValueError(f"summary {key} mismatch")

    training_list = repo / "configs/subset_1000_length_proportional.txt"
    training_panel = repo / "configs/dev_20_length_proportional_seed0.txt"
    prior_external = repo / "configs/external_dev_20_length_proportional_seed20260721.txt"
    prior_fresh = repo / "configs/guarded_external_dev_20_length_proportional_seed20260722.txt"
    untouched = repo / "configs/confirmation_100_length_proportional_seed20260717.txt"
    external_panel = repo / "configs/paper_horizon_external_dev_20_length_proportional_seed20260723.txt"
    training_sha = _sha256(training_list)
    training_panel_sha = _sha256(training_panel)
    prior_external_sha = _sha256(prior_external)
    prior_fresh_sha = _sha256(prior_fresh)
    untouched_sha = _sha256(untouched)
    external_panel_sha = _sha256(external_panel)

    recomputed_training: dict[str, dict] = {}
    for label, root, checkpoint_sha, profile in (
        ("baseline", baseline_root, baseline_sha, HORIZON_AB_BASELINE_PROFILE),
        ("candidate", candidate_root, candidate_sha, PAPER_HORIZON_PROFILE),
    ):
        recomputed = adjudicate_guarded(
            audit_root / f"{label}_panel.json",
            root / "ckpt_2000.pt",
            checkpoint_sha,
            training_list,
            training_sha,
            training_panel,
            training_panel_sha,
            checkpoint_profile=profile,
        )
        _require_exact_json(
            audit_root / f"{label}_decision.json",
            recomputed,
            f"{label} training decision",
        )
        recomputed_training[label] = recomputed
    training_ab = adjudicate_horizon(
        audit_root / "baseline_decision.json",
        audit_root / "candidate_decision.json",
        baseline_root / "history.json",
        candidate_root / "history.json",
    )
    _require_exact_json(
        audit_root / "training_ab_decision.json", training_ab, "training A/B"
    )
    if training_ab.get("status") != "ADVANCE_PAPER_HORIZON_EXTERNAL20":
        raise ValueError("training A/B did not authorize the frozen external panel")
    training_ab_sha = _sha256(audit_root / "training_ab_decision.json")

    for label, root, checkpoint_sha, profile in (
        ("baseline", baseline_root, baseline_sha, HORIZON_AB_BASELINE_PROFILE),
        ("candidate", candidate_root, candidate_sha, PAPER_HORIZON_PROFILE),
    ):
        recomputed = adjudicate_guarded(
            audit_root / f"external_{label}_panel.json",
            root / "ckpt_2000.pt",
            checkpoint_sha,
            training_list,
            training_sha,
            external_panel,
            external_panel_sha,
            panel_kind="paper-horizon-external",
            prior_external_domain_list=prior_external,
            prior_external_domain_list_sha256=prior_external_sha,
            prior_fresh_external_domain_list=prior_fresh,
            prior_fresh_external_domain_list_sha256=prior_fresh_sha,
            untouched_domain_list=untouched,
            untouched_domain_list_sha256=untouched_sha,
            prerequisite_decision=audit_root / "training_ab_decision.json",
            prerequisite_decision_sha256=training_ab_sha,
            candidate_checkpoint_sha256=candidate_sha,
            checkpoint_profile=profile,
        )
        _require_exact_json(
            audit_root / f"external_{label}_decision.json",
            recomputed,
            f"{label} external decision",
        )
    final_ab = adjudicate_horizon(
        audit_root / "external_baseline_decision.json",
        audit_root / "external_candidate_decision.json",
        baseline_root / "history.json",
        candidate_root / "history.json",
        panel_kind="paper-horizon-external",
    )
    _require_exact_json(decision_path, final_ab, "final external A/B")

    return {
        "status": "PASS_PAPER_HORIZON_POSTRUN_CERTIFICATION",
        "run_id": expected_run_id,
        "source_commit": expected_source_commit,
        "verifier_commit": expected_verifier_commit,
        "obs": expected_obs,
        "authorization_sha256": _sha256(authorization_path),
        "readback_completion_sha256": _sha256(completion_path),
        "decision_sha256": _sha256(decision_path),
        "summary_sha256": _sha256(summary_path),
        "audit_manifest_sha256": _sha256(audit_manifest),
        "baseline_checkpoint_sha256": baseline_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "audit_members_verified": len(audit_entries),
        "second_seed_authorized": True,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
    }


def certify(
    readback_one: Path,
    readback_two: Path,
    repo: Path,
    *,
    expected_run_id: str,
    expected_source_commit: str,
    expected_verifier_commit: str,
    expected_obs: str,
    certification_obs: str | None = None,
) -> dict:
    if readback_one.resolve() == readback_two.resolve():
        raise ValueError("independent OBS readback roots must be different")
    first_root_stat = readback_one.stat()
    second_root_stat = readback_two.stat()
    if (first_root_stat.st_dev, first_root_stat.st_ino) == (
        second_root_stat.st_dev,
        second_root_stat.st_ino,
    ):
        raise ValueError("independent OBS readback roots share an inode")
    if _inode_set(readback_one) & _inode_set(readback_two):
        raise ValueError("independent OBS readbacks share file inodes")
    inventory_one = _inventory(readback_one)
    inventory_two = _inventory(readback_two)
    if inventory_one != inventory_two:
        raise ValueError("independent OBS readback inventories differ")
    first = _certify_one(
        readback_one,
        repo,
        expected_run_id=expected_run_id,
        expected_source_commit=expected_source_commit,
        expected_verifier_commit=expected_verifier_commit,
        expected_obs=expected_obs,
    )
    second = _certify_one(
        readback_two,
        repo,
        expected_run_id=expected_run_id,
        expected_source_commit=expected_source_commit,
        expected_verifier_commit=expected_verifier_commit,
        expected_obs=expected_obs,
    )
    if first != second:
        raise ValueError("independent OBS readback certifications differ")
    if _inventory(readback_one) != inventory_one or _inventory(readback_two) != inventory_two:
        raise ValueError("independent OBS readback changed during certification")
    first["readback_inventory_sha256"] = _inventory_sha256(inventory_one)
    first["independent_readbacks_verified"] = 2
    if certification_obs is not None:
        first["certification_obs"] = certification_obs
    return first


def _git_identity(repo: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
    )
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readback-one", required=True, type=Path)
    parser.add_argument("--readback-two", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-verifier-commit", required=True)
    parser.add_argument("--expected-obs", required=True)
    parser.add_argument("--certification-obs", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    commit, dirty = _git_identity(args.repo)
    if commit != args.expected_verifier_commit or dirty:
        raise SystemExit("verifier repository identity mismatch or dirty worktree")
    report = certify(
        args.readback_one,
        args.readback_two,
        args.repo,
        expected_run_id=args.expected_run_id,
        expected_source_commit=args.expected_source_commit,
        expected_verifier_commit=args.expected_verifier_commit,
        expected_obs=args.expected_obs,
        certification_obs=args.certification_obs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
