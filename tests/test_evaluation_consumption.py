import json

import pytest

from deepjump.evaluation_consumption import (
    CLAIM_SCHEMA,
    claim_reserved_evaluation,
    verify_reserved_evaluation_claim,
)


AUTHORIZATION_SHA = "a" * 64


def _authorization(tmp_path, *, authorization_id="external-once"):
    ledger = tmp_path / "ledger"
    ledger.mkdir(exist_ok=True)
    return {
        "authorization_id": authorization_id,
        "consumption_ledger_root": str(ledger.resolve()),
        "phase": "external",
        "checkpoint_sha256": "1" * 64,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": "2" * 64,
        "panel_name": "external20",
        "panel_sha256": "3" * 64,
        "sha256": AUTHORIZATION_SHA,
    }


def test_claim_is_atomic_and_same_authorization_cannot_change_outputs(tmp_path):
    authorization = _authorization(tmp_path)
    first = claim_reserved_evaluation(
        authorization,
        AUTHORIZATION_SHA,
        runtime_probe_output=tmp_path / "runtime-1.json",
        output=tmp_path / "result-1.json",
    )
    assert first["schema"] == CLAIM_SCHEMA
    assert first["authorization_id"] == "external-once"
    assert first["authorization_sha256"] == AUTHORIZATION_SHA
    assert json.loads((tmp_path / "ledger" / f"{AUTHORIZATION_SHA}.claim.json").read_text()) == {
        key: value for key, value in first.items() if key not in {"path", "sha256"}
    }

    with pytest.raises(FileExistsError, match="already consumed"):
        claim_reserved_evaluation(
            authorization,
            AUTHORIZATION_SHA,
            runtime_probe_output=tmp_path / "runtime-2.json",
            output=tmp_path / "result-2.json",
        )


def test_claim_is_not_rolled_back_after_later_failure(tmp_path):
    authorization = _authorization(tmp_path)
    claim_reserved_evaluation(
        authorization,
        AUTHORIZATION_SHA,
        runtime_probe_output=tmp_path / "runtime.json",
        output=tmp_path / "result.json",
    )
    # Simulate any identity, CUDA, payload, or runtime failure after the claim.
    with pytest.raises(RuntimeError, match="later failure"):
        raise RuntimeError("later failure")
    with pytest.raises(FileExistsError, match="already consumed"):
        claim_reserved_evaluation(
            authorization,
            AUTHORIZATION_SHA,
            runtime_probe_output=tmp_path / "retry-runtime.json",
            output=tmp_path / "retry-result.json",
        )


def test_adjudicator_readback_binds_exact_claim_and_ledger(tmp_path):
    authorization = _authorization(tmp_path)
    claim = claim_reserved_evaluation(
        authorization,
        AUTHORIZATION_SHA,
        runtime_probe_output=tmp_path / "runtime.json",
        output=tmp_path / "result.json",
    )
    assert verify_reserved_evaluation_claim(
        authorization,
        AUTHORIZATION_SHA,
        claim,
        runtime_probe_output=tmp_path / "runtime.json",
        output=tmp_path / "result.json",
    ) == claim

    path = tmp_path / "ledger" / f"{AUTHORIZATION_SHA}.claim.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_reserved_evaluation_claim(
            authorization,
            AUTHORIZATION_SHA,
            claim,
            runtime_probe_output=tmp_path / "runtime.json",
            output=tmp_path / "result.json",
        )


def test_adjudicator_rejects_claim_rebound_to_copied_result_path(tmp_path):
    authorization = _authorization(tmp_path)
    claim = claim_reserved_evaluation(
        authorization,
        AUTHORIZATION_SHA,
        runtime_probe_output=tmp_path / "runtime.json",
        output=tmp_path / "result.json",
    )
    with pytest.raises(ValueError, match="exact authorization"):
        verify_reserved_evaluation_claim(
            authorization,
            AUTHORIZATION_SHA,
            claim,
            runtime_probe_output=tmp_path / "runtime.json",
            output=tmp_path / "copied-result.json",
        )


def test_ledger_root_cannot_be_redirected_or_selected_per_replay(tmp_path):
    authorization = _authorization(tmp_path)
    authorization["consumption_ledger_root"] = str(tmp_path / "ledger" / ".." / "ledger")
    with pytest.raises(ValueError, match="canonical"):
        claim_reserved_evaluation(
            authorization,
            AUTHORIZATION_SHA,
            runtime_probe_output=tmp_path / "runtime.json",
            output=tmp_path / "result.json",
        )


def test_authorization_sha_must_match_verified_authorization(tmp_path):
    authorization = _authorization(tmp_path)
    with pytest.raises(ValueError, match="binding mismatch"):
        claim_reserved_evaluation(
            authorization,
            "b" * 64,
            runtime_probe_output=tmp_path / "runtime.json",
            output=tmp_path / "result.json",
        )
