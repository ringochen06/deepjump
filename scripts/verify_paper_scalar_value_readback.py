#!/usr/bin/env python
"""Verify exact scalar-value A/B OBS readback manifests and inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.adjudicate_paper_scalar_value_ab import adjudicate


ARM_FILES = frozenset(
    {"config.json", "history.json", "last.ckpt", "ckpt_1000.pt", "ckpt_2000.pt"}
)
AUDIT_FILES = frozenset(
    {
        "obs_prefix_preflight.log",
        "pytest.log",
        "data_audit.json",
        "source_baseline_sync.log",
        "source_candidate_sync.log",
        "source_audit_sync.log",
        "source_readback_gate.json",
        "sealed_baseline_decision.json",
        "candidate_config.yaml",
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
        "training_evidence.json",
        "decision.json",
        "summary.json",
        "runner.log",
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
        relative = Path(name)
        if (
            not name
            or name in entries
            or relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 1
        ):
            raise ValueError(f"{path}:{line_number}: unsafe or duplicate path")
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


def _absolute_decision(
    path: Path, *, profile: str, checkpoint_sha256: str
) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: decision must be an object")
    expected = {
        "checkpoint_profile": profile,
        "checkpoint_step": 2000,
        "checkpoint_sha256": checkpoint_sha256,
        "domains": 20,
        "starts": 1500,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key} mismatch")
    values = payload.get("domain_mean_guarded_minus_noop")
    if not isinstance(values, list) or len(values) != 20 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"{path}: invalid domain means")
    for key in ("status", "training_domain_list_sha256", "domain_list_sha256"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: missing {key}")
    return payload


def verify(root: Path, phase: str) -> dict:
    baseline = root / "baseline"
    candidate = root / "candidate"
    audit = root / "audit"
    manifests = {
        "baseline_sha256.txt": ARM_FILES,
        "candidate_sha256.txt": ARM_FILES,
        "audit_sha256.txt": AUDIT_FILES,
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

    expected_audit = AUDIT_FILES | INITIAL_UNMANIFESTED
    if phase == "completion":
        expected_audit |= COMPLETION_ADDITIONS
        final_marker = _manifest(audit / "final_marker.sha256")
        if set(final_marker) != {"readback_completion.json", "readback_manifests.sha256"}:
            raise ValueError("final marker members mismatch")
        for name, digest in final_marker.items():
            if _sha256(audit / name) != digest:
                raise ValueError(f"final marker mismatch: {name}")
        decision = json.loads((audit / "decision.json").read_text())
        summary = json.loads((audit / "summary.json").read_text())
        evidence = json.loads((audit / "training_evidence.json").read_text())
        completion = json.loads((audit / "readback_completion.json").read_text())
        if completion.get("status") != "OBS_PRECOMPLETION_DOUBLE_READBACK_PASS":
            raise ValueError("completion status mismatch")
        for key in (
            "external_development_authorized",
            "second_seed_authorized",
            "untouched_confirmation_authorized",
            "formal_training_authorized",
        ):
            for name, payload in (
                ("decision", decision),
                ("summary", summary),
                ("completion", completion),
            ):
                if payload.get(key) is not False:
                    raise ValueError(f"{name} must keep {key}=false")
        if not (
            decision.get("status")
            == summary.get("status")
            == completion.get("scientific_status")
        ):
            raise ValueError("scientific status mismatch")
        for key in ("run_id", "commit"):
            if not (
                evidence.get(key)
                == summary.get(key)
                == completion.get(key)
            ):
                raise ValueError(f"{key} mismatch")
        checkpoint_paths = {
            "baseline_checkpoint_sha256": baseline / "ckpt_2000.pt",
            "candidate_checkpoint_sha256": candidate / "ckpt_2000.pt",
        }
        for key, checkpoint_path in checkpoint_paths.items():
            expected = _sha256(checkpoint_path)
            if any(payload.get(key) != expected for payload in (
                evidence, decision, summary, completion
            )):
                raise ValueError(f"{key} mismatch")
        if evidence.get("schema") != "deepjump.scalar_value_training_evidence.v1":
            raise ValueError("training evidence schema mismatch")
        baseline_checkpoint_sha = _sha256(baseline / "ckpt_2000.pt")
        candidate_checkpoint_sha = _sha256(candidate / "ckpt_2000.pt")
        sealed_baseline = _absolute_decision(
            audit / "sealed_baseline_decision.json",
            profile="paper-horizon-vector-only-500k",
            checkpoint_sha256=baseline_checkpoint_sha,
        )
        baseline_replay = _absolute_decision(
            audit / "baseline_decision.json",
            profile="paper-horizon-vector-only-500k",
            checkpoint_sha256=baseline_checkpoint_sha,
        )
        candidate_absolute = _absolute_decision(
            audit / "candidate_decision.json",
            profile="paper-horizon-vector-scalar-value-500k",
            checkpoint_sha256=candidate_checkpoint_sha,
        )
        evidence_files = {
            "candidate_config_sha256": audit / "candidate_config.yaml",
            "baseline_decision_sha256": audit / "sealed_baseline_decision.json",
            "baseline_replay_decision_sha256": audit / "baseline_decision.json",
            "candidate_decision_sha256": audit / "candidate_decision.json",
            "baseline_history_sha256": baseline / "history.json",
            "candidate_history_sha256": candidate / "history.json",
            "candidate_h20_sha256": audit / "candidate_h20.json",
        }
        for key, evidence_path in evidence_files.items():
            if evidence.get(key) != _sha256(evidence_path):
                raise ValueError(f"training evidence {key} mismatch")
        for key in ("training_domain_list_sha256", "domain_list_sha256"):
            if not (
                evidence.get(key)
                == sealed_baseline.get(key)
                == baseline_replay.get(key)
                == candidate_absolute.get(key)
                == decision.get(key)
            ):
                raise ValueError(f"training evidence {key} mismatch")
        replay_fields = (
            "status",
            "checkpoint_step",
            "checkpoint_sha256",
            "training_domain_list_sha256",
            "domain_list_sha256",
            "domains",
            "starts",
            "domain_mean_guarded_minus_noop",
        )
        replay_mismatches = [
            key for key in replay_fields
            if sealed_baseline.get(key) != baseline_replay.get(key)
        ]
        baseline_reproduced = not replay_mismatches
        if decision.get("baseline_reproduced") is not baseline_reproduced or decision.get(
            "baseline_replay_mismatches"
        ) != replay_mismatches:
            raise ValueError("baseline replay decision mismatch")
        if not baseline_reproduced and decision.get("status") != (
            "STOP_SCALAR_VALUE_BASELINE_REPRODUCIBILITY"
        ):
            raise ValueError("baseline replay mismatch did not stop")
        status_fields = {
            "baseline_absolute_status": sealed_baseline["status"],
            "baseline_replay_absolute_status": baseline_replay["status"],
            "candidate_absolute_status": candidate_absolute["status"],
        }
        for key, value in status_fields.items():
            if decision.get(key) != value:
                raise ValueError(f"{key} mismatch")
        if decision.get("training_evidence") != evidence or decision.get(
            "training_evidence_manifest_sha256"
        ) != _sha256(audit / "training_evidence.json"):
            raise ValueError("training evidence cross-file mismatch")
        if completion.get("decision_sha256") != _sha256(audit / "decision.json"):
            raise ValueError("completion decision SHA256 mismatch")
        if completion.get("audit_manifest_sha256") != _sha256(
            audit / "audit_sha256.txt"
        ):
            raise ValueError("completion audit manifest SHA256 mismatch")
        baseline_history = json.loads((baseline / "history.json").read_text())
        if not isinstance(baseline_history, list) or not baseline_history:
            raise ValueError("baseline history is empty")
        recomputed = adjudicate(
            audit / "sealed_baseline_decision.json",
            audit / "baseline_decision.json",
            audit / "candidate_decision.json",
            baseline / "history.json",
            candidate / "history.json",
            audit / "candidate_h20.json",
            audit / "training_evidence.json",
        )
        if recomputed != decision:
            raise ValueError("final scientific decision does not match recomputation")
        expected_eligible = decision.get("status") == "ADVANCE_SCALAR_VALUE_EXTERNAL20"
        if not (
            decision.get("external_development_scientifically_eligible")
            is expected_eligible
            and summary.get("external_development_scientifically_eligible")
            is expected_eligible
        ):
            raise ValueError("external scientific eligibility mismatch")
    if _inventory(baseline) != set(ARM_FILES):
        raise ValueError("baseline exact inventory mismatch")
    if _inventory(candidate) != set(ARM_FILES):
        raise ValueError("candidate exact inventory mismatch")
    if _inventory(audit) != set(expected_audit):
        raise ValueError("audit exact inventory mismatch")
    return {
        "status": "PASS",
        "phase": phase,
        "baseline_files": len(ARM_FILES),
        "candidate_files": len(ARM_FILES),
        "audit_files": len(expected_audit),
        "manifest_anchor_sha256": _sha256(audit / "readback_manifests.sha256"),
    }


def _snapshot(root: Path) -> list[tuple[str, int, str]]:
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def verify_pair(first: Path, second: Path, phase: str) -> dict:
    first_report = verify(first, phase)
    second_report = verify(second, phase)
    if first.resolve() == second.resolve() or _snapshot(first) != _snapshot(second):
        raise ValueError("independent readback snapshots differ or share a root")
    first_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in first.rglob("*")
        if path.is_file()
    }
    second_inodes = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in second.rglob("*")
        if path.is_file()
    }
    if first_inodes & second_inodes:
        raise ValueError("independent readbacks share file inodes")
    return {
        "status": "PASS_INDEPENDENT_DOUBLE_READBACK",
        "phase": phase,
        "files": len(_snapshot(first)),
        "first_manifest_anchor_sha256": first_report["manifest_anchor_sha256"],
        "second_manifest_anchor_sha256": second_report["manifest_anchor_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--root-two", type=Path)
    parser.add_argument("--phase", required=True, choices=("initial", "completion"))
    args = parser.parse_args()
    report = (
        verify_pair(args.root, args.root_two, args.phase)
        if args.root_two is not None
        else verify(args.root, args.phase)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
