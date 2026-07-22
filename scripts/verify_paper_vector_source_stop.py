#!/usr/bin/env python
"""Prove the frozen source runner could not consume external20 after its STOP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_RUN_ID = "20260722T012922Z"
SOURCE_COMMIT = "dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b"
SOURCE_AUDIT_OBS_URI = (
    "obs://deepjump-mdcath-cn4-ringochen/deepjump-calibration/"
    f"paper-horizon-ab2000/{SOURCE_RUN_ID}/audit"
)
SOURCE_DECISION_SHA256 = (
    "2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38"
)
SOURCE_RUNNER_SHA256 = (
    "2c8eedad191a814080303b6a30204fbb9bee522937c3a0cb5087e3439b6bd75f"
)
SOURCE_CANDIDATE_CHECKPOINT_SHA256 = (
    "fb12d776b106867ca14a8f56476daf776a6296b6dca640f03c2188a75a69bb47"
)
SOURCE_STOP_STATUS = "STOP_PAPER_HORIZON_OBJECTIVE_GAIN"
SOURCE_ADVANCE_STATUS = "ADVANCE_PAPER_HORIZON_EXTERNAL20"
PROOF_SCHEMA = "deepjump.prior_source_control_flow_proof.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_decision(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source decision is missing or symlinked")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_DECISION_SHA256:
        raise ValueError("source decision SHA256 mismatch")
    decision = json.loads(payload)
    if decision.get("status") != SOURCE_STOP_STATUS:
        raise ValueError("source decision did not stop before external20")
    if decision.get("candidate_checkpoint_sha256") != (
        SOURCE_CANDIDATE_CHECKPOINT_SHA256
    ):
        raise ValueError("source candidate checkpoint mismatch")
    for key in (
        "external_development_authorized",
        "second_seed_scientifically_eligible",
        "second_seed_authorized",
        "untouched_confirmation_authorized",
        "formal_training_authorized",
    ):
        if decision.get(key) is not False:
            raise ValueError(f"source decision must keep {key}=false")
    return payload


def _validate_source_runner(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("source runner is missing or symlinked")
    if _sha(path) != SOURCE_RUNNER_SHA256:
        raise ValueError("source runner SHA256 mismatch")
    runner = path.read_text()
    gate = (
        'if [[ "$training_ab_status" == '
        "ADVANCE_PAPER_HORIZON_EXTERNAL20 ]]; then"
    )
    mkdir = 'mkdir -p "$EXTERNAL_DATA_ROOT"'
    download = '"$PYTHON" scripts/download_mdcath.py'
    stop_copy = 'cp "$RUN_DIR/training_ab_decision.json" "$RUN_DIR/decision.json"'
    positions = [runner.index(fragment) for fragment in (gate, mkdir, download, stop_copy)]
    if positions != sorted(positions):
        raise ValueError("source runner external gate order mismatch")


def verify(first: Path, second: Path, source_runner: Path) -> dict:
    first_path = first / "decision.json"
    second_path = second / "decision.json"
    first_payload = _validate_decision(first_path)
    second_payload = _validate_decision(second_path)
    if first_payload != second_payload:
        raise ValueError("source independent decision readbacks differ")
    if first_path.resolve() == second_path.resolve():
        raise ValueError("source decision readbacks use the same path")
    first_identity = (first_path.stat().st_dev, first_path.stat().st_ino)
    second_identity = (second_path.stat().st_dev, second_path.stat().st_ino)
    if first_identity == second_identity:
        raise ValueError("source decision readbacks are not independent files")
    _validate_source_runner(source_runner)
    return {
        "schema": PROOF_SCHEMA,
        "status": "PASS_PRIOR_AUTHORITATIVE_RUN_EXTERNAL_UNCONSUMED",
        "source_run_id": SOURCE_RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "source_audit_obs_uri": SOURCE_AUDIT_OBS_URI,
        "source_decision_sha256": SOURCE_DECISION_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "source_status": SOURCE_STOP_STATUS,
        "required_advance_status": SOURCE_ADVANCE_STATUS,
        "proof_basis": "fixed_decision_and_fixed_runner_control_flow",
        "prior_authoritative_run_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readback-one", required=True, type=Path)
    parser.add_argument("--readback-two", required=True, type=Path)
    parser.add_argument("--source-runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.readback_one, args.readback_two, args.source_runner)
    args.output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
