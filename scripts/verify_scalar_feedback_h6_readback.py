#!/usr/bin/env python
"""Verify exact scalar-feedback OBS readbacks and recompute their decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts.adjudicate_scalar_feedback_h6 import adjudicate


INITIAL_FILES = {
    "audit_sha256.txt",
    "decision.json",
    "hard_stop_evidence.log",
    "obs_prefix_preflight.log",
    "pytest.log",
    "result.json",
    "result.log",
    "runtime_evidence.log",
    "source_training_evidence.json",
    "summary.json",
}
COMPLETION_FILES = INITIAL_FILES | {
    "completion_sha256.txt",
    "initial_readback_one.json",
    "initial_readback_pair.json",
    "initial_readback_two.json",
    "readback_completion.json",
}
AUTHORIZATION_KEYS = (
    "external_development_authorized",
    "second_seed_authorized",
    "untouched_confirmation_authorized",
    "formal_training_authorized",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _inventory(root: Path) -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(root)), _sha256(path))
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _inodes(root: Path) -> set[tuple[int, int]]:
    return {
        (path.stat().st_dev, path.stat().st_ino)
        for path in root.rglob("*")
        if path.is_file()
    }


def _manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("audit manifest contains a malformed line")
        digest, name = parts[0], parts[1].lstrip(" *")
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not name
            or "/" in name
            or name in entries
        ):
            raise ValueError("audit manifest contains an invalid entry")
        entries[name] = digest
    return entries


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
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
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
    recomputed_decision = adjudicate(
        root / "result.json",
        checkpoint,
        domain_list,
        root / "source_training_evidence.json",
    )
    if recomputed_decision != archived_decision:
        raise ValueError("archived decision differs from recomputed decision")

    summary = _load_object(root / "summary.json", "summary")
    expected_summary = {
        "status": archived_decision["status"],
        "scope": archived_decision["scope"],
        "external_development_authorized": False,
        "second_seed_authorized": False,
        "untouched_confirmation_authorized": False,
        "formal_training_authorized": False,
        "run_id": expected_run_id,
        "deployed_commit": expected_deployed_commit,
        "checkpoint_sha256": archived_decision["checkpoint_sha256"],
        "checkpoint_source_commit": archived_decision["checkpoint_source_commit"],
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
        initial_reports = {
            name: _load_object(root / name, name)
            for name in (
                "initial_readback_one.json",
                "initial_readback_two.json",
                "initial_readback_pair.json",
            )
        }
        base_report_keys = {
            "status",
            "phase",
            "decision_status",
            "decision_sha256",
            "summary_sha256",
            "inventory_sha256",
            "completion_sha256",
        }
        for name, report in initial_reports.items():
            expected_keys = (
                base_report_keys | {"independent_readbacks_verified"}
                if name == "initial_readback_pair.json"
                else base_report_keys
            )
            if set(report) != expected_keys:
                raise ValueError(f"{name} has missing or extra fields")
            expected_report = {
                "status": "SCALAR_FEEDBACK_H6_READBACK_PASS",
                "phase": "initial",
                "decision_status": archived_decision["status"],
                "decision_sha256": _sha256(root / "decision.json"),
                "summary_sha256": _sha256(root / "summary.json"),
                "completion_sha256": None,
            }
            for key, value in expected_report.items():
                if report.get(key) != value:
                    raise ValueError(f"{name} {key} mismatch")
            if not isinstance(report.get("inventory_sha256"), str):
                raise ValueError(f"{name} inventory_sha256 is missing")
        if initial_reports["initial_readback_pair.json"].get(
            "independent_readbacks_verified"
        ) != 2:
            raise ValueError("initial readback pair was not independently verified")
        for name in ("initial_readback_one.json", "initial_readback_two.json"):
            if "independent_readbacks_verified" in initial_reports[name]:
                raise ValueError(f"{name} has an unexpected pair-only field")
        first = initial_reports["initial_readback_one.json"]
        second = initial_reports["initial_readback_two.json"]
        pair = dict(initial_reports["initial_readback_pair.json"])
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
            "initial_readback_one_sha256": _sha256(
                root / "initial_readback_one.json"
            ),
            "initial_readback_two_sha256": _sha256(
                root / "initial_readback_two.json"
            ),
            "initial_readback_pair_sha256": _sha256(
                root / "initial_readback_pair.json"
            ),
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
        completion_manifest = _manifest(root / "completion_sha256.txt")
        if completion_manifest != {
            name: _sha256(root / name) for name in completion_names
        }:
            raise ValueError("completion SHA256 manifest mismatch")

    return {
        "status": "SCALAR_FEEDBACK_H6_READBACK_PASS",
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


def verify_pair(
    left: str | Path,
    right: str | Path,
    checkpoint: str | Path,
    domain_list: str | Path,
    **kwargs,
) -> dict:
    left = Path(left)
    right = Path(right)
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
    report = (
        verify_pair(
            args.root,
            args.root_two,
            args.checkpoint,
            args.domain_list,
            **kwargs,
        )
        if args.root_two is not None
        else verify(args.root, args.checkpoint, args.domain_list, **kwargs)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
