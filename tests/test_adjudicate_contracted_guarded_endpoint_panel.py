import hashlib
import json
import statistics
from pathlib import Path

import pytest

import scripts.adjudicate_contracted_guarded_endpoint_panel as adjudicator
from deepjump.evaluation_consumption import claim_reserved_evaluation
from scripts.contracted_guarded_endpoint_panel_eval import PREREQUISITE_STATUS, SCOPE


TEMPERATURES = [320, 348, 379, 413, 450]
REPLICAS = [0, 1, 2, 3, 4]
CHECKPOINT_SHA = "1" * 64
CONTRACT_SHA = "2" * 64
PANEL_SHA = "3" * 64
PREREQUISITE_SHA = "4" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _domain_rows(
    panel_ids: list[str], fallback_cells: set[tuple[int, int]] | None = None
) -> list[dict]:
    fallback_cells = fallback_cells or set()
    domains = []
    for domain_index, domain_id in enumerate(panel_ids):
        effect = -0.2 + 0.0005 * domain_index
        cells = []
        for cell_index, (temperature, replica) in enumerate(
            (temperature, replica)
            for temperature in TEMPERATURES
            for replica in REPLICAS
        ):
            starts = []
            for start_index, start_frame in enumerate([0, 50, 100]):
                fallback = (
                    (domain_index, cell_index) in fallback_cells and start_index == 0
                )
                noop = 2.0 + cell_index * 0.01 + start_index * 0.001
                raw_rmsd = noop + effect
                guarded_rmsd = noop if fallback else raw_rmsd
                starts.append({
                    "start_index": start_index,
                    "start_frame": start_frame,
                    "target_position_finite": True,
                    "noop_rmsd": noop,
                    "accepted": not fallback,
                    "fallback": fallback,
                    "selected_position_exact": True,
                    "selected_vector_exact": True,
                    "source": {
                        "position_finite": True,
                        "vector_finite": True,
                        "bond_mean": 3.8,
                        "bond_max": 4.0,
                        "physical": True,
                    },
                    "raw": {
                        "position_finite": True,
                        "vector_finite": True,
                        "rmsd": raw_rmsd,
                        "minus_noop": effect,
                        "bond_mean": 3.8,
                        "bond_max": 6.0 if fallback else 4.0,
                        "physical": not fallback,
                    },
                    "guarded": {
                        "position_finite": True,
                        "vector_finite": True,
                        "rmsd": guarded_rmsd,
                        "minus_noop": guarded_rmsd - noop,
                        "bond_mean": 3.8,
                        "bond_max": 4.0,
                        "physical": True,
                    },
                })
            deltas = [row["guarded"]["minus_noop"] for row in starts]
            cells.append({
                "domain": domain_id,
                "temperature": temperature,
                "replica": replica,
                "frames": 102,
                "starts": [0, 50, 100],
                "by_start": starts,
                "mean_guarded_minus_noop": statistics.fmean(deltas),
                "source_cell_physical": True,
                "raw_cell_physical": all(row["raw"]["physical"] for row in starts),
                "guarded_cell_physical": True,
                "fallback_starts": sum(row["fallback"] for row in starts),
            })
        cell_deltas = [cell["mean_guarded_minus_noop"] for cell in cells]
        domains.append({
            "domain": domain_id,
            "preprocessing": {
                "canon_symmetric": True,
                "residues_total": 80 + domain_index,
                "residues_evaluated": 80 + domain_index,
            },
            "summary": {
                "cells": 25,
                "mean_guarded_minus_noop": statistics.fmean(cell_deltas),
                "cells_better_than_noop": sum(value < 0 for value in cell_deltas),
                "fallback_starts": sum(cell["fallback_starts"] for cell in cells),
                "fallback_cells": sum(cell["fallback_starts"] > 0 for cell in cells),
            },
            "cells": cells,
        })
    return domains


def _case(
    tmp_path: Path,
    monkeypatch,
    phase: str,
    *,
    fallback_cells: set[tuple[int, int]] | None = None,
    mechanism_passes: bool = True,
):
    count = 100 if phase == "untouched" else 20
    panel_ids = [f"domain-{index:03d}" for index in range(count)]
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(panel_ids) + "\n")
    panel_sha = _sha(panel)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint is verified by the patched identity gate")
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    prerequisite_path = tmp_path / "prerequisite.json"
    prerequisite_path.write_text("{}")
    ledger_root = tmp_path / "consumption-ledger"
    ledger_root.mkdir()
    panel_name = f"{phase}-panel"
    identity = {
        "status": "PASS_FROZEN_EVALUATION_IDENTITY",
        "phase": phase,
        "panel_name": panel_name,
        "panel_role": {
            "development": "development_seen",
            "external": "external_reserved",
            "untouched": "untouched_confirmation_reserved",
        }[phase],
        "panel_sha256": panel_sha,
        "panel_domains": count,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": CONTRACT_SHA,
        "train_list_sha256": "5" * 64,
        "formal_training_authorized": False,
    }
    prerequisite = {
        "schema": "deepjump.reserved_evaluation_authorization.v2",
        "authorization_id": f"{phase}-test-authorization",
        "consumption_ledger_root": str(ledger_root.resolve()),
        "status": PREREQUISITE_STATUS[phase],
        "phase": phase,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_step": 2_000,
        "full_training_contract_sha256": CONTRACT_SHA,
        "panel_name": panel_name,
        "panel_sha256": panel_sha,
        "reserved_panel_authorized": True,
        "formal_training_authorized": False,
        "path": str(prerequisite_path.resolve()),
        "sha256": PREREQUISITE_SHA,
    }
    payload = {
        "status": "PASS_CONTRACTED_PANEL_LIVE_PAYLOAD_REHASH",
        "manifest_sha256": "6" * 64,
        "panel_domains": count,
        "panel_bytes": count * 100,
        "payloads": [
            {
                "domain": domain,
                "file": f"mdcath_dataset_{domain}.h5",
                "bytes": 100,
                "sha256": "7" * 64,
            }
            for domain in panel_ids
        ],
    }
    runtime = {
        "status": "PASS_RUNTIME_PROBE",
        "domain": panel_ids[-1],
        "residues": 100,
        "cell_seconds": 0.5,
        "projected_panel_minutes": 10.0,
        "peak_memory_bytes": 50,
        "total_memory_bytes": 100,
        "peak_memory_fraction": 0.5,
        "limits": {
            "max_peak_memory_fraction": 0.8,
            "max_projected_minutes": 50.0 if phase == "external" else 250.0,
        },
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    runtime_sha = _sha(runtime_path)
    result = tmp_path / "result.json"
    consumption_claim = claim_reserved_evaluation(
        prerequisite,
        PREREQUISITE_SHA,
        runtime_probe_output=runtime_path,
        output=result,
    )
    mechanism = {
        "domain": panel_ids[0],
        "temperature": 320,
        "replica": 0,
        "target_slot": 0,
        "target_start": 0,
        "same_shape_peer_position_bitwise_equal": mechanism_passes,
        "same_shape_peer_vector_bitwise_equal": mechanism_passes,
        "fp32_b1_b3_position_max_abs_diff": 1e-5,
        "fp32_b1_b3_vector_max_abs_diff": 1e-5,
        "fp32_accept_b1": True,
        "fp32_accept_b3": False,
        "fp64_b1_b3_position_max_abs_diff": 1e-14,
        "fp64_b1_b3_vector_max_abs_diff": 1e-14,
        "fp64_accept_b1": True,
        "fp64_accept_b3": True,
    }
    result_payload = {
        "status": "EVALUATION_COMPLETE_NOT_ADJUDICATED",
        "scope": SCOPE,
        "identity": identity,
        "prerequisite": prerequisite,
        "consumption_claim": consumption_claim,
        "payload_verification": json.loads(json.dumps(payload)),
        "runtime_probe": runtime,
        "runtime_probe_sha256": runtime_sha,
        "mechanism_probe": mechanism,
        "evidence_completeness": {
            "status": "PASS_RESERVED_EVALUATION_EVIDENCE_COMPLETENESS",
            "domains": count,
            "cells": count * 25,
            "starts": count * 25 * 3,
            "scientific_adjudication_performed": False,
        },
        "delta_frames": 1,
        "settings": {
            "starts": 3,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
            "policy": "reject_to_exact_source_per_start",
            "strict_thresholds": {
                "bond_mean_gt": 3.2,
                "bond_mean_lt": 4.5,
                "bond_max_lt": 5.5,
            },
            "fallback_caps": {"max_starts": 3, "max_cells": 1},
        },
        "grid": {"temperatures": TEMPERATURES, "replicas": REPLICAS},
        "domains": _domain_rows(panel_ids, fallback_cells),
        "formal_training_authorized": False,
    }
    result.write_text(json.dumps(result_payload))

    monkeypatch.setattr(
        adjudicator, "verify_frozen_evaluation_identity", lambda *args, **kwargs: identity
    )
    monkeypatch.setattr(
        adjudicator,
        "verify_reserved_evaluation_prerequisite",
        lambda *args, **kwargs: prerequisite,
    )
    monkeypatch.setattr(
        adjudicator,
        "_load_verified_checkpoint",
        lambda *args, **kwargs: ({
            "cfg": {
                "data": {
                    "root": "/data/mdcath",
                    "delta_frames": 1,
                    "temperatures": TEMPERATURES,
                    "replicas": REPLICAS,
                }
            }
        }, CHECKPOINT_SHA),
    )
    monkeypatch.setattr(
        adjudicator,
        "rehash_contracted_panel_payloads",
        lambda *args, **kwargs: {**payload, "paths": [], "pins": []},
    )
    return {
        "result": result,
        "result_payload": result_payload,
        "checkpoint": checkpoint,
        "contract": contract,
        "panel": panel,
        "panel_name": panel_name,
        "prerequisite": prerequisite_path,
        "runtime": runtime_path,
        "runtime_sha": runtime_sha,
    }


def _run(case: dict, phase: str) -> dict:
    return adjudicator.adjudicate(
        case["result"],
        _sha(case["result"]),
        case["checkpoint"],
        CHECKPOINT_SHA,
        2_000,
        case["contract"],
        CONTRACT_SHA,
        phase,
        case["panel_name"],
        case["panel"],
        case["prerequisite"],
        PREREQUISITE_SHA,
        case["runtime"],
        case["runtime_sha"],
    )


@pytest.mark.parametrize(
    ("phase", "expected_status", "expected_domains_better"),
    [
        ("development", "ADVANCE_EXPANDED_DATA_EXTERNAL", 20),
        ("external", "PASS_CONTRACTED_GUARD_EXTERNAL20", 20),
        ("untouched", "PASS_CONTRACTED_GUARD_UNTOUCHED100", 100),
    ],
)
def test_contracted_adjudicator_passes_each_exact_phase(
    tmp_path, monkeypatch, phase, expected_status, expected_domains_better
):
    case = _case(tmp_path, monkeypatch, phase)
    report = _run(case, phase)
    assert report["status"] == expected_status
    assert report["gate_status"] == {
        "development": "PASS_CONTRACTED_GUARD_DEVELOPMENT20",
        "external": "PASS_CONTRACTED_GUARD_EXTERNAL20",
        "untouched": "PASS_CONTRACTED_GUARD_UNTOUCHED100",
    }[phase]
    assert report["primary_domains_better_than_noop"] == expected_domains_better
    assert report["fallback_starts"] == 0
    assert report["formal_training_authorized"] is False


def test_contracted_adjudicator_enforces_global_fallback_caps(tmp_path, monkeypatch):
    case = _case(
        tmp_path, monkeypatch, "development", fallback_cells={(0, 0), (0, 1)}
    )
    report = _run(case, "development")
    assert report["status"] == "STOP_CONTRACTED_GUARD_FALLBACK_CAP"
    assert report["fallback_starts"] == 2
    assert report["fallback_cells"] == 2
    assert report["formal_training_authorized"] is False


def test_contracted_adjudicator_stops_on_mechanism_failure(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, "external", mechanism_passes=False)
    report = _run(case, "external")
    assert report["status"] == "STOP_CONTRACTED_GUARD_MECHANISM"
    assert report["formal_training_authorized"] is False


def test_contracted_adjudicator_rejects_nested_arithmetic_drift(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, "development")
    case["result_payload"]["domains"][0]["cells"][0]["by_start"][0]["guarded"][
        "minus_noop"
    ] += 0.01
    case["result"].write_text(json.dumps(case["result_payload"]))
    with pytest.raises(ValueError, match="guarded-minus-noop"):
        _run(case, "development")


def test_contracted_adjudicator_rejects_nested_physical_flag_drift(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, "development")
    case["result_payload"]["domains"][0]["cells"][0]["raw_cell_physical"] = False
    case["result"].write_text(json.dumps(case["result_payload"]))
    with pytest.raises(ValueError, match="raw cell physical flag mismatch"):
        _run(case, "development")


def test_contracted_adjudicator_rejects_result_sha_drift(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, "development")
    with pytest.raises(ValueError, match="result SHA256 mismatch"):
        adjudicator.adjudicate(
            case["result"],
            "f" * 64,
            case["checkpoint"],
            CHECKPOINT_SHA,
            2_000,
            case["contract"],
            CONTRACT_SHA,
            "development",
            case["panel_name"],
            case["panel"],
            case["prerequisite"],
            PREREQUISITE_SHA,
            case["runtime"],
            case["runtime_sha"],
        )


def test_contracted_adjudicator_rejects_payload_report_drift(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, "development")
    case["result_payload"]["payload_verification"]["panel_bytes"] += 1
    case["result"].write_text(json.dumps(case["result_payload"]))
    with pytest.raises(ValueError, match="payload report mismatch"):
        _run(case, "development")
