from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import freeze_mdcath_source_inventory as inventory


def _entry(domain: str, size: int, ordinal: int) -> SimpleNamespace:
    return SimpleNamespace(
        path=f"data/mdcath_dataset_{domain}.h5",
        size=size,
        blob_id=f"{ordinal:040x}",
        lfs=SimpleNamespace(
            pointer_size=130 + ordinal,
            sha256=f"{ordinal:064x}",
            size=size,
        ),
        xet_hash=f"{ordinal + 100:064x}",
    )


def _expectations(rows: list[dict], domain_list: bytes) -> inventory.Expectations:
    payload = inventory.canonical_jsonl(rows)
    return inventory.Expectations(
        revision="a" * 40,
        h5_count=len(rows),
        h5_bytes=sum(row["size"] for row in rows),
        domain_list_sha256=inventory._sha256(domain_list),
        canonical_inventory_sha256=inventory._sha256(payload),
    )


def test_canonical_inventory_validates_exact_domain_list_and_lfs_identity():
    rows = inventory.rows_from_repo_tree(
        [_entry("2defB00", 7, 2), _entry("1abcA00", 5, 1)]
    )
    domain_list = b"1abcA00\n2defB00\n"
    summary = inventory.validate_inventory(
        rows,
        domain_list_bytes=domain_list,
        expectations=_expectations(rows, domain_list),
    )

    assert [row["path"] for row in rows] == [
        "data/mdcath_dataset_1abcA00.h5",
        "data/mdcath_dataset_2defB00.h5",
    ]
    assert summary["h5_count"] == 2
    assert summary["total_bytes"] == 12
    assert summary["unique_lfs_sha256"] == 2


def test_inventory_rejects_non_alphanumeric_domain_path():
    with pytest.raises(inventory.InventoryError, match="unexpected mdCATH source path"):
        inventory.rows_from_repo_tree([_entry("bad.domain", 5, 1)])


@pytest.mark.parametrize("mutation", ["duplicate_lfs", "wrong_size", "wrong_domains"])
def test_inventory_fails_closed_on_identity_mismatch(mutation):
    rows = inventory.rows_from_repo_tree(
        [_entry("1abcA00", 5, 1), _entry("2defB00", 7, 2)]
    )
    domain_list = b"1abcA00\n2defB00\n"
    expectations = _expectations(rows, domain_list)
    if mutation == "duplicate_lfs":
        rows[1]["lfs_sha256"] = rows[0]["lfs_sha256"]
    elif mutation == "wrong_size":
        rows[1]["lfs_size"] += 1
    else:
        domain_list = b"1abcA00\n3ghiC00\n"

    with pytest.raises(inventory.InventoryError):
        inventory.validate_inventory(
            rows,
            domain_list_bytes=domain_list,
            expectations=expectations,
        )


def test_offline_verification_rejects_noncanonical_or_tampered_cache(tmp_path, monkeypatch):
    rows = inventory.rows_from_repo_tree(
        [_entry("1abcA00", 5, 1), _entry("2defB00", 7, 2)]
    )
    domain_list = b"1abcA00\n2defB00\n"
    expectations = _expectations(rows, domain_list)
    monkeypatch.setattr(inventory, "Expectations", lambda: expectations)

    payload = inventory.canonical_jsonl(rows)
    summary = inventory.validate_inventory(
        rows,
        domain_list_bytes=domain_list,
        expectations=expectations,
    )
    metadata = {
        **summary,
        "inventory_schema": inventory.INVENTORY_SCHEMA,
        "source_domain_list": inventory.SOURCE_DOMAIN_LIST,
        "source_domain_list_matches_inventory": True,
        "source_endpoint": inventory.SOURCE_ENDPOINT,
        "source_repo": inventory.SOURCE_REPO,
        "source_revision": inventory.SOURCE_REVISION,
    }
    inventory_path = tmp_path / inventory.INVENTORY_NAME
    metadata_path = tmp_path / inventory.METADATA_NAME
    inventory_path.write_bytes(payload)
    metadata_path.write_text(json.dumps(metadata))

    assert inventory.verify_offline(tmp_path)["canonical_inventory_sha256"] == inventory._sha256(payload)

    first, *rest = payload.splitlines()
    noncanonical = json.dumps(json.loads(first), sort_keys=False).encode() + b"\n"
    inventory_path.write_bytes(noncanonical + b"\n".join(rest) + b"\n")
    with pytest.raises(inventory.InventoryError, match="canonically serialized"):
        inventory.verify_offline(tmp_path)


def test_offline_verification_rejects_metadata_revision(tmp_path, monkeypatch):
    rows = inventory.rows_from_repo_tree([_entry("1abcA00", 5, 1)])
    domain_list = b"1abcA00\n"
    expectations = _expectations(rows, domain_list)
    monkeypatch.setattr(inventory, "Expectations", lambda: expectations)
    payload = inventory.canonical_jsonl(rows)
    summary = inventory.validate_inventory(
        rows,
        domain_list_bytes=domain_list,
        expectations=expectations,
    )
    (tmp_path / inventory.INVENTORY_NAME).write_bytes(payload)
    (tmp_path / inventory.METADATA_NAME).write_text(
        json.dumps(
            {
                **summary,
                "inventory_schema": inventory.INVENTORY_SCHEMA,
                "source_domain_list": inventory.SOURCE_DOMAIN_LIST,
                "source_domain_list_matches_inventory": True,
                "source_endpoint": inventory.SOURCE_ENDPOINT,
                "source_repo": inventory.SOURCE_REPO,
                "source_revision": "b" * 40,
            }
        )
    )

    with pytest.raises(inventory.InventoryError, match="source_revision"):
        inventory.verify_offline(tmp_path)
