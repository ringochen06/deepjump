import io
import json
import sys
import types
from pathlib import Path

import pytest

from scripts.claim_external_panel import _claim_bytes, claim


def _payload() -> dict:
    return {
        "schema": "deepjump.external_panel_claim.v1",
        "status": "CLAIMED_FOR_SINGLE_USE",
        "run_id": "20260722T120000Z",
        "commit": "a" * 40,
        "panel_sha256": "b" * 64,
        "panel_count": 20,
        "expected_total_bytes": 14236836972,
        "source_stop_decision_sha256": "c" * 64,
        "source_proof_sha256": "0" * 64,
        "training_ab_decision_sha256": "d" * 64,
        "baseline_checkpoint_sha256": "e" * 64,
        "candidate_checkpoint_sha256": "f" * 64,
        "prior_authoritative_run_consumed": False,
        "claimed_at": "2026-07-22T12:00:00+00:00",
    }


def _write(path: Path, payload: dict) -> bytes:
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(content)
    return content


def test_claim_json_requires_nonempty_canonical_exact_schema(tmp_path):
    path = tmp_path / "claim.json"
    expected = _write(path, _payload())
    assert _claim_bytes(path) == expected
    path.write_text(json.dumps(_payload(), indent=2))
    with pytest.raises(ValueError, match="not canonical"):
        _claim_bytes(path)
    payload = _payload()
    payload["extra"] = True
    _write(path, payload)
    with pytest.raises(ValueError, match="schema"):
        _claim_bytes(path)
    payload = _payload()
    payload["run_id"] = "mutable-run"
    _write(path, payload)
    with pytest.raises(ValueError, match="run_id"):
        _claim_bytes(path)
    payload = _payload()
    payload["claimed_at"] = "2026-07-22T12:00:00+08:00"
    _write(path, payload)
    with pytest.raises(ValueError, match="UTC"):
        _claim_bytes(path)


class _AppendObjectContent:
    def __init__(self, content, position):
        self.content = content
        self.position = position


class _Response:
    def __init__(self, status, *, next_position=None, content=None):
        self.status = status
        self.body = types.SimpleNamespace(
            nextPosition=next_position,
            buffer=content,
        )


class _Client:
    def __init__(self, content: bytes, *, append_status=200, second=None):
        self.content = content
        self.append_status = append_status
        self.second = content if second is None else second
        self.reads = 0

    def appendObject(self, bucket, key, request):
        assert request.position == 0
        assert isinstance(request.content, io.BytesIO)
        assert request.content.read() == self.content
        return _Response(
            self.append_status,
            next_position=len(self.content) if self.append_status == 200 else None,
        )

    def getObject(self, bucket, key, loadStreamInMemory):
        assert loadStreamInMemory is True
        self.reads += 1
        return _Response(200, content=self.content if self.reads == 1 else self.second)


def test_atomic_claim_requires_first_writer_and_two_exact_readbacks(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "obs",
        types.SimpleNamespace(AppendObjectContent=_AppendObjectContent),
    )
    content = b'{"nonempty":true}\n'
    first, second = claim(
        _Client(content), bucket="bucket", key="claim.json", claim_bytes=content
    )
    assert first == second == content
    with pytest.raises(RuntimeError, match="HTTP 409"):
        claim(
            _Client(content, append_status=409),
            bucket="bucket",
            key="claim.json",
            claim_bytes=content,
        )
    with pytest.raises(RuntimeError, match="readbacks differ"):
        claim(
            _Client(content, second=b"changed"),
            bucket="bucket",
            key="claim.json",
            claim_bytes=content,
        )
