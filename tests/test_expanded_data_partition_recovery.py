import copy
import json

import pytest

from scripts.verify_expanded_data_partition_recovery import (
    verify_recovery_equivalence,
)


def _report(repo: str, temporary: str) -> dict:
    return {
        "schema": "deepjump.expanded_data_partition.v1",
        "status": "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION",
        "official": {
            "path": "/data-full/mdcath/mdCATH_domains.txt",
            "sha256": "a" * 64,
            "domains": 5398,
        },
        "held_out_panels": [
            {
                "name": "external20",
                "path": f"{repo}/configs/external20.txt",
                "sha256": "b" * 64,
                "domains": 20,
            }
        ],
        "held_out_panel_contract_sha256": "c" * 64,
        "partition": {
            "official_domains": 5398,
            "excluded_domains": 20,
            "train_eligible_domains": 5378,
            "train_eligible_sha256": "d" * 64,
        },
        "domain_assignments": [
            {"domain": "1abcA00", "partition": "train_eligible", "panel": None}
        ],
        "outputs": {
            "train_eligible_list": {
                "path": f"{temporary}/train_eligible_5218.txt",
                "sha256": "e" * 64,
                "domains": 5218,
            },
            "audit": {"path": f"{temporary}/expanded_data_partition.json"},
        },
        "panel_registry": {
            "path": f"{repo}/configs/full_mdcath_evaluation_exclusion_registry.json",
            "sha256": "f" * 64,
            "schema": "deepjump.full_mdcath_evaluation_exclusion_registry.v1",
        },
        "partition_contract_sha256": "1" * 64,
    }


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_recovery_accepts_only_deployment_and_atomic_output_directory_changes(tmp_path):
    sealed = tmp_path / "sealed.json"
    candidate = tmp_path / "candidate.json"
    _write(sealed, _report("/data/deepjump-old", "/qualification/.partition.old"))
    _write(candidate, _report("/data/deepjump-new", "/qualification/.partition.new"))

    verify_recovery_equivalence(sealed, candidate)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("partition", "train_eligible_domains"), 5377),
        (("held_out_panels", 0, "sha256"), "0" * 64),
        (("partition_contract_sha256",), "0" * 64),
        (("domain_assignments", 0, "partition"), "excluded"),
    ],
)
def test_recovery_rejects_semantic_drift(tmp_path, path, value):
    sealed_payload = _report("/data/deepjump-old", "/qualification/.partition.old")
    candidate_payload = copy.deepcopy(sealed_payload)
    target = candidate_payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    sealed = tmp_path / "sealed.json"
    candidate = tmp_path / "candidate.json"
    _write(sealed, sealed_payload)
    _write(candidate, candidate_payload)

    with pytest.raises(ValueError, match="differs from sealed semantics"):
        verify_recovery_equivalence(sealed, candidate)


def test_recovery_rejects_referenced_basename_drift(tmp_path):
    sealed_payload = _report("/data/deepjump-old", "/qualification/.partition.old")
    candidate_payload = _report("/data/deepjump-new", "/qualification/.partition.new")
    candidate_payload["held_out_panels"][0]["path"] = (
        "/data/deepjump-new/configs/wrong-panel.txt"
    )
    sealed = tmp_path / "sealed.json"
    candidate = tmp_path / "candidate.json"
    _write(sealed, sealed_payload)
    _write(candidate, candidate_payload)

    with pytest.raises(ValueError, match="differs from sealed semantics"):
        verify_recovery_equivalence(sealed, candidate)
