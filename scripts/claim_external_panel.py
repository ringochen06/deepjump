#!/usr/bin/env python
"""Atomically claim one frozen OBS external panel using an ECS agency."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_id",
        "commit",
        "panel_sha256",
        "panel_count",
        "expected_total_bytes",
        "source_stop_decision_sha256",
        "source_proof_sha256",
        "training_ab_decision_sha256",
        "baseline_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "prior_authoritative_run_consumed",
        "claimed_at",
    }
)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return value


def _commit(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError("claim commit must be 40 lowercase hex characters")
    return value


def _claim_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if not raw:
        raise ValueError("claim must be non-empty")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != REQUIRED_KEYS:
        raise ValueError("claim exact schema mismatch")
    if payload.get("schema") != "deepjump.external_panel_claim.v1":
        raise ValueError("claim schema mismatch")
    if payload.get("status") != "CLAIMED_FOR_SINGLE_USE":
        raise ValueError("claim status mismatch")
    if payload.get("panel_count") != 20 or type(payload.get("panel_count")) is not int:
        raise ValueError("claim panel count mismatch")
    if (
        payload.get("expected_total_bytes") != 14236836972
        or type(payload.get("expected_total_bytes")) is not int
    ):
        raise ValueError("claim byte count mismatch")
    if payload.get("prior_authoritative_run_consumed") is not False:
        raise ValueError("prior authoritative consumption flag mismatch")
    _commit(payload.get("commit"))
    for key in (
        "panel_sha256",
        "source_stop_decision_sha256",
        "source_proof_sha256",
        "training_ab_decision_sha256",
        "baseline_checkpoint_sha256",
        "candidate_checkpoint_sha256",
    ):
        _digest(payload.get(key), f"claim {key}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(
        r"[0-9]{8}T[0-9]{6}Z", run_id
    ) is None:
        raise ValueError("claim run_id must be a UTC basic timestamp")
    claimed_at = payload.get("claimed_at")
    if not isinstance(claimed_at, str):
        raise ValueError("claim claimed_at must be a UTC timestamp")
    try:
        timestamp = datetime.fromisoformat(claimed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("claim claimed_at is malformed") from exc
    if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError("claim claimed_at must use UTC")
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if raw != canonical:
        raise ValueError("claim JSON is not canonical")
    return raw


def _response_bytes(response: object) -> bytes:
    status = int(getattr(response, "status", 0))
    if not 200 <= status < 300:
        raise RuntimeError(f"OBS claim readback failed with HTTP {status}")
    body = getattr(response, "body", None)
    buffer = getattr(body, "buffer", None)
    if isinstance(buffer, str):
        return buffer.encode()
    if isinstance(buffer, bytes):
        return buffer
    raise RuntimeError("OBS claim readback did not return in-memory bytes")


def claim(
    client: object,
    *,
    bucket: str,
    key: str,
    claim_bytes: bytes,
) -> tuple[bytes, bytes]:
    from obs import AppendObjectContent

    response = client.appendObject(
        bucket,
        key,
        AppendObjectContent(content=io.BytesIO(claim_bytes), position=0),
    )
    status = int(getattr(response, "status", 0))
    if not 200 <= status < 300:
        raise RuntimeError(f"atomic OBS claim failed with HTTP {status}")
    next_position = getattr(getattr(response, "body", None), "nextPosition", None)
    if int(next_position or -1) != len(claim_bytes):
        raise RuntimeError("atomic OBS claim next position mismatch")
    first = _response_bytes(client.getObject(bucket, key, loadStreamInMemory=True))
    second = _response_bytes(client.getObject(bucket, key, loadStreamInMemory=True))
    if first != claim_bytes or second != claim_bytes or first != second:
        raise RuntimeError("atomic OBS claim independent readbacks differ")
    return first, second


def _client(endpoint: str) -> object:
    from obs import ObsClient

    return ObsClient(server=endpoint, security_provider_policy="ECS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--claim-json", required=True, type=Path)
    parser.add_argument("--readback-one", required=True, type=Path)
    parser.add_argument("--readback-two", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    content = _claim_bytes(args.claim_json)
    client = _client(args.endpoint)
    try:
        first, second = claim(
            client, bucket=args.bucket, key=args.key, claim_bytes=content
        )
    finally:
        client.close()
    args.readback_one.write_bytes(first)
    args.readback_two.write_bytes(second)
    digest = hashlib.sha256(content).hexdigest()
    report = {
        "status": "OBS_ATOMIC_SINGLE_USE_CLAIM_DOUBLE_READBACK_PASS",
        "bucket": args.bucket,
        "key": args.key,
        "claim_sha256": digest,
        "claim_bytes": len(content),
        "readback_one_sha256": hashlib.sha256(first).hexdigest(),
        "readback_two_sha256": hashlib.sha256(second).hexdigest(),
    }
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
