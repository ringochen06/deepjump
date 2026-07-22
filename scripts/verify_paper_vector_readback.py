#!/usr/bin/env python
"""Verify exact paper-vector OBS readback manifests and inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE_FILES = frozenset(
    {"config.json", "history.json", "last.ckpt", "ckpt_1000.pt", "ckpt_2000.pt"}
)
CANDIDATE_FILES = BASELINE_FILES
COMMON_AUDIT_FILES = frozenset(
    {
        "obs_prefix_preflight.log",
        "external_claim_preflight_initial.log",
        "pytest.log",
        "data_audit.json",
        "source_audit_one_sync.log",
        "source_audit_two_sync.log",
        "source_proof.json",
        "baseline_checkpoint_gate.json",
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
        "candidate_h20.json",
        "candidate_h20.log",
        "training_ab_decision.json",
        "decision.json",
        "external_status.json",
        "summary.json",
        "runner.log",
    }
)
EXTERNAL_AUDIT_FILES = frozenset(
    {
        "external_claim_preflight_final.log",
        "external_claim.json",
        "external_claim_readback.json",
        "external_download.log",
        "external_data_audit.json",
        "external_download_manifest.json",
        "external_baseline_runtime_probe.json",
        "external_baseline_panel.json",
        "external_baseline_panel.log",
        "external_baseline_decision.json",
        "external_candidate_runtime_probe.json",
        "external_candidate_panel.json",
        "external_candidate_panel.log",
        "external_candidate_decision.json",
    }
)
INITIAL_UNMANIFESTED = frozenset(
    {
        "audit_sha256.txt",
        "baseline_sha256.txt",
        "candidate_sha256.txt",
        "readback_manifests.sha256",
    }
)
COMPLETION_ADDITIONS = frozenset(
    {"readback_completion.json", "final_marker.sha256"}
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise ValueError(f"{path}:{line_number}: malformed sha256 line")
        digest, name = line[:64], line[66:].removeprefix("./")
        if any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}:{line_number}: malformed digest")
        if not name or name in entries:
            raise ValueError(f"{path}:{line_number}: empty or duplicate path")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"{path}:{line_number}: unsafe path")
        entries[name] = digest
    if not entries:
        raise ValueError(f"{path}: empty manifest")
    return entries


def _inventory(root: Path) -> set[str]:
    if not root.is_dir():
        raise ValueError(f"missing readback directory: {root}")
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"readback contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if "/" in relative:
                raise ValueError(f"unexpected nested readback file: {relative}")
            names.add(relative)
    if not names:
        raise ValueError(f"empty readback directory: {root}")
    return names


def _verify_manifest(path: Path, root: Path, expected: frozenset[str]) -> None:
    entries = _manifest(path)
    if set(entries) != set(expected):
        raise ValueError(f"{path}: exact manifest members mismatch")
    for name, digest in entries.items():
        member = root / name
        if not member.is_file() or _sha256(member) != digest:
            raise ValueError(f"{path}: member mismatch: {name}")


def verify(root: Path, phase: str) -> dict:
    baseline = root / "baseline"
    candidate = root / "candidate"
    audit = root / "audit"
    status = json.loads((audit / "external_status.json").read_text()).get("status")
    if status == "EXECUTED_PAPER_VECTOR_EXTERNAL20":
        expected_audit = COMMON_AUDIT_FILES | EXTERNAL_AUDIT_FILES
    elif status == "SKIPPED_PAPER_VECTOR_EXTERNAL20":
        expected_audit = COMMON_AUDIT_FILES
    else:
        raise ValueError("external_status.json has an unknown status")

    manifests = {
        "baseline_sha256.txt": BASELINE_FILES,
        "candidate_sha256.txt": CANDIDATE_FILES,
        "audit_sha256.txt": expected_audit,
    }
    anchor = _manifest(audit / "readback_manifests.sha256")
    if set(anchor) != set(manifests):
        raise ValueError("readback manifest anchor members mismatch")
    for name, expected in manifests.items():
        manifest_path = audit / name
        if _sha256(manifest_path) != anchor[name]:
            raise ValueError(f"readback manifest anchor mismatch: {name}")
        target = audit if name == "audit_sha256.txt" else root / name.split("_")[0]
        _verify_manifest(manifest_path, target, expected)

    expected_audit_inventory = expected_audit | INITIAL_UNMANIFESTED
    if phase == "completion":
        expected_audit_inventory |= COMPLETION_ADDITIONS
        final_marker = _manifest(audit / "final_marker.sha256")
        if set(final_marker) != {"readback_completion.json", "readback_manifests.sha256"}:
            raise ValueError("final marker members mismatch")
        for name, digest in final_marker.items():
            if _sha256(audit / name) != digest:
                raise ValueError(f"final marker mismatch: {name}")
    if _inventory(baseline) != set(BASELINE_FILES):
        raise ValueError("baseline exact inventory mismatch")
    if _inventory(candidate) != set(CANDIDATE_FILES):
        raise ValueError("candidate exact inventory mismatch")
    if _inventory(audit) != set(expected_audit_inventory):
        raise ValueError("audit exact inventory mismatch")
    return {
        "status": "PASS",
        "phase": phase,
        "external_status": status,
        "baseline_files": len(BASELINE_FILES),
        "candidate_files": len(CANDIDATE_FILES),
        "audit_files": len(expected_audit_inventory),
        "manifest_anchor_sha256": _sha256(audit / "readback_manifests.sha256"),
    }


def _snapshot(root: Path) -> list[tuple[str, int, str]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"readback contains a symlink: {path}")
        if path.is_file():
            rows.append((
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                _sha256(path),
            ))
    return rows


def verify_pair(first: Path, second: Path, phase: str) -> dict:
    first_report = verify(first, phase)
    second_report = verify(second, phase)
    if first.resolve() == second.resolve() or _snapshot(first) != _snapshot(second):
        raise ValueError("independent readback snapshots differ or share a root")
    first_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in first.rglob("*") if path.is_file()
    }
    second_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in second.rglob("*") if path.is_file()
    }
    if first_inodes & second_inodes:
        raise ValueError("independent readbacks share file inodes")
    return {
        "status": "PASS_INDEPENDENT_DOUBLE_READBACK",
        "phase": phase,
        "external_status": first_report["external_status"],
        "files": len(_snapshot(first)),
        "first_manifest_anchor_sha256": first_report["manifest_anchor_sha256"],
        "second_manifest_anchor_sha256": second_report["manifest_anchor_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--root-two", type=Path)
    parser.add_argument("--phase", required=True, choices=("initial", "completion"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = (
        verify_pair(args.root, args.root_two, args.phase)
        if args.root_two is not None else verify(args.root, args.phase)
    )
    if args.output is not None:
        args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
