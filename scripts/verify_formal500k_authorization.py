#!/usr/bin/env python
"""Bind a separate user approval to one exact formal500k package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from formal500k_package_lib import (
    atomic_write_json,
    build_package_payload,
    read_regular,
    sha256_bytes,
)


def _exact_keys(payload: object, expected: set[str], label: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        args.spec.resolve(),
        args.package.resolve(),
        args.authorization.resolve(),
        args.output.resolve(),
    }
    if len(paths) != 4:
        raise ValueError("spec, package, authorization, and output must be separate")

    _, package_raw = read_regular(args.package.resolve(), "formal500k package")
    if hashlib.sha256(package_raw).hexdigest() != args.package_sha256:
        raise ValueError("formal500k package SHA256 mismatch")
    package = json.loads(package_raw)
    rebuilt = build_package_payload(args.spec.resolve())
    if package != rebuilt:
        raise ValueError("formal500k package differs from direct source re-verification")

    _, authorization_raw = read_regular(
        args.authorization.resolve(), "formal500k user authorization"
    )
    if sha256_bytes(authorization_raw) != args.authorization_sha256:
        raise ValueError("formal500k authorization SHA256 mismatch")
    authorization = json.loads(authorization_raw)
    _exact_keys(
        authorization,
        {
            "schema",
            "status",
            "formal_training_authorized",
            "authorization_id",
            "issued_at",
            "expires_at",
            "authorized_package_sha256",
            "authorized_package_id",
            "authorized_run_id",
            "authorized_reviewed_commit",
            "scope",
            "approval_record_sha256",
        },
        "formal500k authorization",
    )
    if (
        authorization["schema"] != "deepjump.formal500k.user_authorization.v1"
        or authorization["status"] != "USER_AUTHORIZED_FORMAL_TRAINING"
        or authorization["formal_training_authorized"] is not True
    ):
        raise ValueError("formal500k authorization status is invalid")
    candidate = package["formal_candidate"]
    plan = candidate["execution_plan"]
    expected_scope = {
        "classification": package["classification"],
        "fresh_init": True,
        "target_step": 500000,
        "unique_scientific_endpoint_steps": [500000],
        "world_size": 8,
        "effective_batch": 128,
        "train_domain_count": 5218,
        "full_training_contract_sha256": package["data_identity"][
            "full_training_contract_sha256"
        ],
        "data_uuid": plan["data_uuid"],
        "hard_cap_hours": package["estimate_budget"]["hard_cap_hours"],
        "maximum_authorized_cost": {
            "currency": package["estimate_budget"]["currency"],
            "amount": package["estimate_budget"]["maximum_authorized_cost"],
        },
        "formal_run_may_start": True,
        "external_or_untouched_access_authorized": False,
    }
    bindings = {
        "authorized_package_sha256": args.package_sha256,
        "authorized_package_id": package["package_id"],
        "authorized_run_id": plan["run_id"],
        "authorized_reviewed_commit": plan["reviewed_commit"],
        "scope": expected_scope,
    }
    mismatches = {
        field: (authorization.get(field), expected)
        for field, expected in bindings.items()
        if authorization.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"authorization does not bind the exact package: {mismatches}")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(authorization["approval_record_sha256"])
    ) is None:
        raise ValueError("approval_record_sha256 is invalid")
    try:
        issued_at = datetime.fromisoformat(
            authorization["issued_at"].replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            authorization["expires_at"].replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("authorization timestamps are invalid") from exc
    authorized_hours = (expires_at - issued_at).total_seconds() / 3600
    if (
        authorized_hours <= 0
        or authorized_hours > float(package["estimate_budget"]["hard_cap_hours"])
    ):
        raise ValueError("authorization lifetime exceeds the package GPU-hour cap")

    verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_write_json(
        args.output.resolve(),
        {
            "schema": "deepjump.formal500k.authorization_verification.v1",
            "status": "PASS_FORMAL_TRAINING_AUTHORIZATION",
            "formal_training_authorized": True,
            "package_sha256": args.package_sha256,
            "authorization_sha256": args.authorization_sha256,
            "authorization_verifier_sha256": verifier_sha,
            "package_id": package["package_id"],
            "run_id": plan["run_id"],
            "reviewed_commit": plan["reviewed_commit"],
            "config_sha256": candidate["config_sha256"],
            "target_total_steps": 500000,
            "execution_plan": plan,
            "checkpoint_plan": package["checkpoint_plan"],
            "stop_plan": package["stop_plan"],
            "recovery_plan": package["recovery_plan"],
            "scope": expected_scope,
        },
    )


if __name__ == "__main__":
    main()
