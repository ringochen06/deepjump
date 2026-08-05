import hashlib
import json
import shutil
from pathlib import Path

import pytest

import scripts.verify_full_mdcath_obs_readback as verifier


def _case(tmp_path: Path):
    source = tmp_path / "obs"
    source.mkdir()
    rows = []
    for domain, content in (("1abcA00", b"payload-one"), ("2defB00", b"payload-two")):
        filename = f"mdcath_dataset_{domain}.h5"
        (source / filename).write_bytes(content)
        rows.append({
            "domain": domain,
            "file": filename,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows, sort_keys=True) + "\n")
    return source, manifest, rows


def _fake_obsutil(source: Path, calls: list[str]):
    def run(command, check):
        assert check is True
        assert command[:2] == ["obsutil", "cp"]
        filename = command[2].rsplit("/", 1)[-1]
        calls.append(filename)
        shutil.copyfile(source / filename, command[3])

    return run


def test_readback_revalidates_remote_content_when_resuming_exact_records(tmp_path, monkeypatch):
    source, manifest, rows = _case(tmp_path)
    calls = []
    monkeypatch.setattr(verifier.subprocess, "run", _fake_obsutil(source, calls))
    expected_bytes = sum(row["size"] for row in rows)
    report = verifier.verify_obs_readback(
        manifest=manifest,
        obs_prefix="obs://bucket/corpus",
        work_dir=tmp_path / "journal",
        expected_count=2,
        expected_bytes=expected_bytes,
    )
    assert report["status"] == "PASS_FULL_MDCATH_OBS_CONTENT_READBACK"
    assert report["records_created"] == 2
    assert calls == [row["file"] for row in rows]
    calls.clear()
    resumed = verifier.verify_obs_readback(
        manifest=manifest,
        obs_prefix="obs://bucket/corpus",
        work_dir=tmp_path / "journal",
        expected_count=2,
        expected_bytes=expected_bytes,
    )
    assert resumed == report
    assert calls == [row["file"] for row in rows]
    session = json.loads((tmp_path / "journal" / "session.json").read_text())
    completion = json.loads((tmp_path / "journal" / "completion.json").read_text())
    assert session["schema"] == verifier.SESSION_SCHEMA
    assert completion["record_inventory_sha256"]


def test_same_size_good_to_evil_obs_resume_cannot_reuse_old_record(tmp_path, monkeypatch):
    source, manifest, rows = _case(tmp_path)
    monkeypatch.setattr(verifier.subprocess, "run", _fake_obsutil(source, []))
    work = tmp_path / "journal"
    expected_bytes = sum(row["size"] for row in rows)
    verifier.verify_obs_readback(
        manifest=manifest,
        obs_prefix="obs://bucket/corpus",
        work_dir=work,
        expected_count=2,
        expected_bytes=expected_bytes,
    )
    original = source / rows[0]["file"]
    original.write_bytes(b"evil-bytes!")
    assert original.stat().st_size == rows[0]["size"]
    calls = []
    monkeypatch.setattr(verifier.subprocess, "run", _fake_obsutil(source, calls))
    with pytest.raises(ValueError, match="OBS content readback mismatch"):
        verifier.verify_obs_readback(
            manifest=manifest,
            obs_prefix="obs://bucket/corpus",
            work_dir=work,
            expected_count=2,
            expected_bytes=expected_bytes,
        )
    assert calls == [rows[0]["file"]]


def test_readback_rejects_changed_obs_content_without_publishing_record(tmp_path, monkeypatch):
    source, manifest, rows = _case(tmp_path)
    (source / rows[0]["file"]).write_bytes(b"forged")
    monkeypatch.setattr(verifier.subprocess, "run", _fake_obsutil(source, []))
    with pytest.raises(ValueError, match="OBS content readback mismatch"):
        verifier.verify_obs_readback(
            manifest=manifest,
            obs_prefix="obs://bucket/corpus",
            work_dir=tmp_path / "journal",
            expected_count=2,
            expected_bytes=sum(row["size"] for row in rows),
        )
    assert not list((tmp_path / "journal" / "records").iterdir())


def test_readback_rejects_manifest_count_bytes_and_extra_journal_entry(tmp_path, monkeypatch):
    source, manifest, rows = _case(tmp_path)
    monkeypatch.setattr(verifier.subprocess, "run", _fake_obsutil(source, []))
    with pytest.raises(ValueError, match="object count"):
        verifier.verify_obs_readback(
            manifest=manifest,
            obs_prefix="obs://bucket/corpus",
            work_dir=tmp_path / "count",
            expected_count=3,
            expected_bytes=sum(row["size"] for row in rows),
        )
    with pytest.raises(ValueError, match="byte count"):
        verifier.verify_obs_readback(
            manifest=manifest,
            obs_prefix="obs://bucket/corpus",
            work_dir=tmp_path / "bytes",
            expected_count=2,
            expected_bytes=1,
        )
    journal = tmp_path / "journal"
    verifier.verify_obs_readback(
        manifest=manifest,
        obs_prefix="obs://bucket/corpus",
        work_dir=journal,
        expected_count=2,
        expected_bytes=sum(row["size"] for row in rows),
    )
    (journal / "records" / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="exact inventory"):
        verifier.verify_obs_readback(
            manifest=manifest,
            obs_prefix="obs://bucket/corpus",
            work_dir=journal,
            expected_count=2,
            expected_bytes=sum(row["size"] for row in rows),
        )
