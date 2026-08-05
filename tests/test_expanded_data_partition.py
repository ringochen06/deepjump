from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.build_expanded_data_partition as partitioner


PanelSpec = partitioner.PanelSpec


def _write(path: Path, domains: list[str]) -> str:
    path.write_text("".join(f"{domain}\n" for domain in domains))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path):
    official = tmp_path / "official.txt"
    official_sha = _write(official, ["d5", "d1", "d4", "d2", "d3"])
    historical = tmp_path / "historical.txt"
    historical_sha = _write(historical, ["d1", "d3"])
    sealed_future = tmp_path / "sealed-future.txt"
    sealed_future_sha = _write(sealed_future, ["d5"])
    panels = [
        PanelSpec("historical", historical, historical_sha),
        PanelSpec("sealed_future", sealed_future, sealed_future_sha),
    ]
    return official, official_sha, historical, sealed_future, panels


def _expected_panel_contract_sha256(panels: list[PanelSpec]) -> str:
    audits = [
        {
            "name": panel.name,
            "sha256": panel.expected_sha256,
            "domains": len(panel.path.read_text().splitlines()),
        }
        for panel in panels
    ]
    return partitioner._panel_contract_sha256(audits)


def _build_partition(official, official_sha, panels):
    return partitioner.build_partition(
        official,
        official_sha,
        panels,
        _expected_panel_contract_sha256(panels),
    )


def _write_partition(official, official_sha, panels, train_path, audit_path):
    return partitioner.write_partition(
        official,
        official_sha,
        panels,
        _expected_panel_contract_sha256(panels),
        train_path,
        audit_path,
    )


def _write_registry(path: Path, official_sha: str, panels: list[PanelSpec]) -> str:
    audits = [
        {
            "name": panel.name,
            "sha256": panel.expected_sha256,
            "domains": len(panel.path.read_text().splitlines()),
        }
        for panel in panels
    ]
    payload = {
        "official_domain_list_sha256": official_sha,
        "panel_contract_sha256": partitioner._panel_contract_sha256(audits),
        "panels": [
            {
                "domains": audit["domains"],
                "name": panel.name,
                "path": panel.path.name,
                "role": "development_seen",
                "sha256": panel.expected_sha256,
            }
            for panel, audit in zip(panels, audits, strict=True)
        ],
        "schema": partitioner.PANEL_REGISTRY_SCHEMA,
        "source_revision": partitioner.EXPECTED_SOURCE_REVISION,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partition_preserves_official_order_and_emits_domain_level_audit(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    train, report = _build_partition(official, official_sha, panels)

    assert train == ["d4", "d2"]
    assert report["status"] == "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION"
    assert report["partition"] == {
        "official_domains": 5,
        "excluded_domains": 3,
        "train_eligible_domains": 2,
        "excluded_union_sha256": hashlib.sha256(b"d5\nd1\nd3\n").hexdigest(),
        "train_eligible_sha256": hashlib.sha256(b"d4\nd2\n").hexdigest(),
        "panels_are_pairwise_disjoint": True,
        "all_excluded_domains_are_official": True,
    }
    assert report["domain_assignments"] == [
        {"domain": "d5", "partition": "excluded", "panel": "sealed_future"},
        {"domain": "d1", "partition": "excluded", "panel": "historical"},
        {"domain": "d4", "partition": "train_eligible", "panel": None},
        {"domain": "d2", "partition": "train_eligible", "panel": None},
        {"domain": "d3", "partition": "excluded", "panel": "historical"},
    ]
    assert len(report["partition_contract_sha256"]) == 64


def test_contract_sha_is_independent_of_input_file_locations(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_inputs = _inputs(left)
    right_inputs = _inputs(right)
    _, left_report = _build_partition(left_inputs[0], left_inputs[1], left_inputs[4])
    _, right_report = _build_partition(right_inputs[0], right_inputs[1], right_inputs[4])
    assert left_report["official"]["path"] != right_report["official"]["path"]
    assert (
        left_report["partition_contract_sha256"]
        == right_report["partition_contract_sha256"]
    )


def test_contract_sha_is_independent_of_panel_argument_order(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    expected = _expected_panel_contract_sha256(panels)
    _, left = partitioner.build_partition(
        official, official_sha, panels, expected
    )
    _, right = partitioner.build_partition(
        official, official_sha, list(reversed(panels)), expected
    )
    assert left["partition_contract_sha256"] == right["partition_contract_sha256"]


def test_missing_required_panel_fails_frozen_contract(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    expected = _expected_panel_contract_sha256(panels)
    with pytest.raises(ValueError, match="panel contract SHA256 mismatch"):
        partitioner.build_partition(
            official,
            official_sha,
            [panels[0]],
            expected,
        )


def test_panel_registry_binds_exact_file_and_contract(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    registry = tmp_path / "registry.json"
    registry_sha = _write_registry(registry, official_sha, panels)
    loaded, contract_sha, identity = partitioner._load_panel_registry(
        registry,
        registry_sha,
        official_sha,
    )
    assert loaded == panels
    assert contract_sha == _expected_panel_contract_sha256(panels)
    assert identity["sha256"] == registry_sha

    registry.write_text(registry.read_text() + " ")
    with pytest.raises(ValueError, match="panel registry SHA256 mismatch"):
        partitioner._load_panel_registry(registry, registry_sha, official_sha)


def test_write_partition_binds_outputs_and_refuses_overwrite(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    train_path, audit_path = tmp_path / "train.txt", tmp_path / "audit.json"
    report = _write_partition(
        official, official_sha, panels, train_path, audit_path
    )

    assert train_path.read_text() == "d4\nd2\n"
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == report["outputs"][
        "train_eligible_list"
    ]["sha256"]
    assert json.loads(audit_path.read_text()) == report
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _write_partition(official, official_sha, panels, train_path, tmp_path / "new.json")


@pytest.mark.parametrize("which", ["official", "panel"])
def test_duplicate_domains_fail_closed(tmp_path, which):
    official, official_sha, historical, _, panels = _inputs(tmp_path)
    if which == "official":
        official_sha = _write(official, ["d1", "d1", "d2"])
    else:
        duplicated_sha = _write(historical, ["d1", "d1"])
        panels[0] = PanelSpec("historical", historical, duplicated_sha)
    with pytest.raises(ValueError, match="duplicate domains"):
        _build_partition(official, official_sha, panels)


def test_unknown_panel_domain_fails_closed(tmp_path):
    official, official_sha, historical, _, panels = _inputs(tmp_path)
    unknown_sha = _write(historical, ["notOfficial"])
    panels[0] = PanelSpec("historical", historical, unknown_sha)
    with pytest.raises(ValueError, match="absent from the official list"):
        _build_partition(official, official_sha, panels)


def test_any_cross_panel_overlap_fails_closed(tmp_path):
    official, official_sha, _, sealed_future, panels = _inputs(tmp_path)
    overlap_sha = _write(sealed_future, ["d3", "d5"])
    panels[1] = PanelSpec("sealed_future", sealed_future, overlap_sha)
    with pytest.raises(ValueError, match="overlaps another held-out panel"):
        _build_partition(official, official_sha, panels)


def test_sha_mismatch_and_invalid_sha_fail_closed(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _build_partition(official, "0" * 64, panels)
    panels[0] = PanelSpec(panels[0].name, panels[0].path, "ABC")
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        _build_partition(official, official_sha, panels)


def test_panel_names_paths_and_symlinks_cannot_alias(tmp_path):
    official, official_sha, historical, _, panels = _inputs(tmp_path)
    with pytest.raises(ValueError, match="duplicate panel name"):
        _build_partition(official, official_sha, [panels[0], panels[0]])
    alias = PanelSpec("alias", historical, panels[0].expected_sha256)
    with pytest.raises(ValueError, match="aliases another panel path"):
        _build_partition(official, official_sha, [panels[0], alias])
    symlink = tmp_path / "link.txt"
    symlink.symlink_to(historical)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _build_partition(
            official,
            official_sha,
            [PanelSpec("linked", symlink, panels[0].expected_sha256)],
        )


def test_requires_a_panel_and_leaves_at_least_one_train_domain(tmp_path):
    official, official_sha, historical, _, panels = _inputs(tmp_path)
    with pytest.raises(ValueError, match="at least one held-out panel"):
        _build_partition(official, official_sha, [])
    all_sha = _write(historical, ["d5", "d1", "d4", "d2", "d3"])
    with pytest.raises(ValueError, match="exclude every official domain"):
        _build_partition(
            official,
            official_sha,
            [PanelSpec("all", historical, all_sha)],
        )


def test_outputs_must_be_distinct(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    output = tmp_path / "same.txt"
    with pytest.raises(ValueError, match="must be different"):
        _write_partition(official, official_sha, panels, output, output)


@pytest.mark.parametrize("content", ["d1", "d1\n\n", " d1\n", "d/1\n"])
def test_list_format_is_strict(tmp_path, content):
    official, official_sha, historical, _, panels = _inputs(tmp_path)
    historical.write_text(content)
    panels[0] = PanelSpec(
        "historical", historical, hashlib.sha256(historical.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        _build_partition(official, official_sha, panels)


def test_cli_accepts_a_sealed_future_panel_path_without_embedding_contents(tmp_path):
    official, official_sha, historical, sealed_future, panels = _inputs(tmp_path)
    train_path, audit_path = tmp_path / "train.txt", tmp_path / "audit.json"
    command = [
        sys.executable,
        "scripts/build_expanded_data_partition.py",
        "--official-list",
        str(official),
        "--official-sha256",
        official_sha,
        "--exclude-panel",
        f"historical={historical}={panels[0].expected_sha256}",
        "--exclude-panel",
        f"sealed_future={sealed_future}={panels[1].expected_sha256}",
        "--expected-panel-contract-sha256",
        _expected_panel_contract_sha256(panels),
        "--train-output",
        str(train_path),
        "--audit-output",
        str(audit_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "PASS_EXPANDED_DATA_HELDOUT_EXCLUSION"
    assert train_path.read_text() == "d4\nd2\n"


def test_cli_panel_path_may_contain_equals(tmp_path):
    official, official_sha, historical, _, panels = _inputs(tmp_path)
    directory = tmp_path / "sealed=location"
    directory.mkdir()
    sealed = directory / "future=list.txt"
    sealed_sha = _write(sealed, ["d5"])
    train_path, audit_path = tmp_path / "train.txt", tmp_path / "audit.json"
    command = [
        sys.executable,
        "scripts/build_expanded_data_partition.py",
        "--official-list",
        str(official),
        "--official-sha256",
        official_sha,
        "--exclude-panel",
        f"historical={historical}={panels[0].expected_sha256}",
        "--exclude-panel",
        f"sealed_future={sealed}={sealed_sha}",
        "--expected-panel-contract-sha256",
        _expected_panel_contract_sha256(
            [panels[0], PanelSpec("sealed_future", sealed, sealed_sha)]
        ),
        "--train-output",
        str(train_path),
        "--audit-output",
        str(audit_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert train_path.read_text() == "d4\nd2\n"


def test_cli_accepts_frozen_panel_registry(tmp_path):
    official, official_sha, _, _, panels = _inputs(tmp_path)
    registry = tmp_path / "registry.json"
    registry_sha = _write_registry(registry, official_sha, panels)
    train_path, audit_path = tmp_path / "train.txt", tmp_path / "audit.json"
    command = [
        sys.executable,
        "scripts/build_expanded_data_partition.py",
        "--official-list",
        str(official),
        "--official-sha256",
        official_sha,
        "--panel-registry",
        str(registry),
        "--expected-panel-registry-sha256",
        registry_sha,
        "--train-output",
        str(train_path),
        "--audit-output",
        str(audit_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["panel_registry"]["sha256"] == registry_sha
    assert report["held_out_panel_contract_sha256"] == (
        _expected_panel_contract_sha256(panels)
    )
    assert train_path.read_text() == "d4\nd2\n"


def test_production_panel_registry_identity_is_frozen():
    registry = Path("configs/full_mdcath_evaluation_exclusion_registry.json")
    expected_sha = "65f14cb45c1af84ca6a7e97affe6974232fd3ec12da69a875e9a089525943097"
    panels, contract_sha, identity = partitioner._load_panel_registry(
        registry,
        expected_sha,
        "295c6da1c9f8846a1ea3993eca12a3232d16a2b3a4b0d8791c7c45392186709b",
    )
    assert identity["sha256"] == expected_sha
    assert contract_sha == "f5a772daa77a1f3118cc2e9151363d6a5a9f9737634b4a45561f09c302db2865"
    assert [(panel.name, len(panel.path.read_text().splitlines())) for panel in panels] == [
        ("fresh_external20", 20),
        ("guarded_external20", 20),
        ("legacy_dev20", 20),
        ("paper_horizon_external20", 20),
        ("untouched_confirmation100", 100),
    ]
