#!/usr/bin/env python
"""Verify exact teacher-update projection OBS readbacks and recompute the decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.adjudicate_teacher_update_projection import adjudicate
from scripts.verify_scalar_feedback_h20_readback import (
    AUTHORIZATION_KEYS,
    _inodes,
    _inventory,
    _load_object,
    _manifest,
    _sha256,
)


BOUND_FILES = {
    "h20_result.json",
    "h20_decision.json",
    "h20_readback_completion.json",
}
INITIAL_FILES = {
    "audit_sha256.txt",
    "decision.json",
    "hard_stop_evidence.log",
    "obs_prefix_preflight.log",
    "pytest.log",
    "result.json",
    "result.log",
    "runtime_evidence.log",
    "summary.json",
    *BOUND_FILES,
}
COMPLETION_FILES = INITIAL_FILES | {
    "completion_sha256.txt",
    "initial_readback_one.json",
    "initial_readback_pair.json",
    "initial_readback_two.json",
    "readback_completion.json",
}


def verify(
    root: str | Path,
    checkpoint: str | Path,
    domain_list: str | Path,
    *,
    phase: str,
    expected_run_id: str,
    expected_deployed_commit: str,
    expected_obs: str,
) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise ValueError("missing readback root")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("readback contains a symlink")
    expected_files = INITIAL_FILES if phase == "initial" else COMPLETION_FILES
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("readback has missing or extra files")
    manifest = _manifest(root / "audit_sha256.txt")
    if set(manifest) != INITIAL_FILES - {"audit_sha256.txt"}:
        raise ValueError("audit manifest has missing or extra entries")
    for name, expected_digest in manifest.items():
        if _sha256(root / name) != expected_digest:
            raise ValueError(f"audit manifest SHA256 mismatch: {name}")

    archived_decision = _load_object(root / "decision.json", "decision")
    recomputed = adjudicate(
        root / "result.json",
        checkpoint,
        domain_list,
        root / "h20_result.json",
        root / "h20_decision.json",
        root / "h20_readback_completion.json",
    )
    if recomputed != archived_decision:
        raise ValueError("archived decision differs from recomputed decision")
    summary = _load_object(root / "summary.json", "summary")
    expected_summary = {
        "status": archived_decision["status"],
        "scope": archived_decision["scope"],
        "h20_result_sha256": archived_decision["h20_result_sha256"],
        "h20_decision_sha256": archived_decision["h20_decision_sha256"],
        "h20_completion_sha256": archived_decision["h20_completion_sha256"],
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "run_id": expected_run_id,
        "deployed_commit": expected_deployed_commit,
        "checkpoint_sha256": archived_decision["checkpoint_sha256"],
        "obs": expected_obs,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value or (
            isinstance(value, bool) and type(summary.get(key)) is not bool
        ):
            raise ValueError(f"summary {key} mismatch")
    if set(summary) != set(expected_summary) | {"completed_at"}:
        raise ValueError("summary has missing or extra fields")
    if not isinstance(summary.get("completed_at"), str) or not summary["completed_at"]:
        raise ValueError("summary completed_at is missing")

    completion = None
    if phase == "completion":
        reports = {
            name: _load_object(root / name, name)
            for name in (
                "initial_readback_one.json",
                "initial_readback_two.json",
                "initial_readback_pair.json",
            )
        }
        for name, report in reports.items():
            expected = {
                "status": "TEACHER_UPDATE_PROJECTION_READBACK_PASS",
                "phase": "initial",
                "decision_status": archived_decision["status"],
                "decision_sha256": _sha256(root / "decision.json"),
                "summary_sha256": _sha256(root / "summary.json"),
                "completion_sha256": None,
            }
            for key, value in expected.items():
                if report.get(key) != value:
                    raise ValueError(f"{name} {key} mismatch")
            extra = {"inventory_sha256"}
            if name == "initial_readback_pair.json":
                extra.add("independent_readbacks_verified")
                if report.get("independent_readbacks_verified") != 2:
                    raise ValueError("initial pair was not independently verified")
            if set(report) != set(expected) | extra:
                raise ValueError(f"{name} has missing or extra fields")
        first = reports["initial_readback_one.json"]
        second = reports["initial_readback_two.json"]
        pair = dict(reports["initial_readback_pair.json"])
        pair.pop("independent_readbacks_verified")
        if first != second or first != pair:
            raise ValueError("initial readback reports differ")

        completion = _load_object(root / "readback_completion.json", "completion")
        expected_completion = {
            "status": "OBS_DOUBLE_READBACK_PASS",
            "decision_status": archived_decision["status"],
            "run_id": expected_run_id,
            "commit": expected_deployed_commit,
            "audit_manifest_sha256": _sha256(root / "audit_sha256.txt"),
            "archived_decision_sha256": _sha256(root / "decision.json"),
            "archived_summary_sha256": _sha256(root / "summary.json"),
            "recomputed_decision_sha256": _sha256(root / "decision.json"),
            "initial_readback_one_sha256": _sha256(root / "initial_readback_one.json"),
            "initial_readback_two_sha256": _sha256(root / "initial_readback_two.json"),
            "initial_readback_pair_sha256": _sha256(root / "initial_readback_pair.json"),
            "independent_readbacks_verified": 2,
            **{key: False for key in AUTHORIZATION_KEYS},
        }
        for key, value in expected_completion.items():
            if completion.get(key) != value or (
                isinstance(value, bool) and type(completion.get(key)) is not bool
            ):
                raise ValueError(f"completion {key} mismatch")
        if set(completion) != set(expected_completion) | {"completed_at"}:
            raise ValueError("completion has missing or extra fields")
        completion_names = {
            "initial_readback_one.json",
            "initial_readback_two.json",
            "initial_readback_pair.json",
            "readback_completion.json",
        }
        if _manifest(root / "completion_sha256.txt") != {
            name: _sha256(root / name) for name in completion_names
        }:
            raise ValueError("completion SHA256 manifest mismatch")

    return {
        "status": "TEACHER_UPDATE_PROJECTION_READBACK_PASS",
        "phase": phase,
        "decision_status": archived_decision["status"],
        "decision_sha256": _sha256(root / "decision.json"),
        "summary_sha256": _sha256(root / "summary.json"),
        "inventory_sha256": hashlib.sha256(
            json.dumps(_inventory(root), separators=(",", ":")).encode()
        ).hexdigest(),
        "completion_sha256": (
            _sha256(root / "readback_completion.json") if completion else None
        ),
    }


def verify_pair(left, right, checkpoint, domain_list, **kwargs) -> dict:
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        raise ValueError("independent readback roots must differ")
    if os.stat(left).st_ino == os.stat(right).st_ino or _inodes(left) & _inodes(right):
        raise ValueError("independent readbacks share inodes")
    first = verify(left, checkpoint, domain_list, **kwargs)
    second = verify(right, checkpoint, domain_list, **kwargs)
    if _inventory(left) != _inventory(right) or first != second:
        raise ValueError("independent readbacks differ")
    return {**first, "independent_readbacks_verified": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--root-two", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--domain-list", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("initial", "completion"))
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-deployed-commit", required=True)
    parser.add_argument("--expected-obs", required=True)
    args = parser.parse_args()
    kwargs = {
        "phase": args.phase,
        "expected_run_id": args.expected_run_id,
        "expected_deployed_commit": args.expected_deployed_commit,
        "expected_obs": args.expected_obs,
    }
    if args.root_two is None:
        report = verify(args.root, args.checkpoint, args.domain_list, **kwargs)
    else:
        report = verify_pair(
            args.root, args.root_two, args.checkpoint, args.domain_list, **kwargs
        )
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
