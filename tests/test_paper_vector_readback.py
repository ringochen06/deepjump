import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_paper_vector_readback import (
    BASELINE_FILES,
    CANDIDATE_FILES,
    COMMON_AUDIT_FILES,
    COMPLETION_ADDITIONS,
    EXTERNAL_AUDIT_FILES,
    INITIAL_UNMANIFESTED,
    verify,
    verify_pair,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_files(root: Path, names: set[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text(f"fixture:{name}\n")


def _write_manifest(path: Path, root: Path, names: set[str]) -> None:
    path.write_text("".join(f"{_sha(root / name)}  {name}\n" for name in sorted(names)))


def _fixture(root: Path, *, external: bool, completion: bool) -> Path:
    baseline = root / "baseline"
    candidate = root / "candidate"
    audit = root / "audit"
    audit_files = set(COMMON_AUDIT_FILES)
    if external:
        audit_files.update(EXTERNAL_AUDIT_FILES)
    _write_files(baseline, set(BASELINE_FILES))
    _write_files(candidate, set(CANDIDATE_FILES))
    _write_files(audit, audit_files)
    (audit / "external_status.json").write_text(json.dumps({
        "status": (
            "EXECUTED_PAPER_VECTOR_EXTERNAL20"
            if external else "SKIPPED_PAPER_VECTOR_EXTERNAL20"
        )
    }))
    _write_manifest(audit / "baseline_sha256.txt", baseline, set(BASELINE_FILES))
    _write_manifest(audit / "candidate_sha256.txt", candidate, set(CANDIDATE_FILES))
    _write_manifest(audit / "audit_sha256.txt", audit, audit_files)
    _write_manifest(
        audit / "readback_manifests.sha256",
        audit,
        {"baseline_sha256.txt", "candidate_sha256.txt", "audit_sha256.txt"},
    )
    if completion:
        (audit / "readback_completion.json").write_text("{}\n")
        _write_manifest(
            audit / "final_marker.sha256",
            audit,
            {"readback_completion.json", "readback_manifests.sha256"},
        )
    expected = audit_files | set(INITIAL_UNMANIFESTED)
    if completion:
        expected |= set(COMPLETION_ADDITIONS)
    assert {path.name for path in audit.iterdir()} == expected
    return root


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("completion", [False, True])
def test_paper_vector_readback_requires_downloaded_manifests_and_exact_inventory(
    tmp_path, external, completion
):
    root = _fixture(tmp_path, external=external, completion=completion)
    report = verify(root, "completion" if completion else "initial")
    assert report["status"] == "PASS"
    assert report["external_status"].startswith(
        "EXECUTED" if external else "SKIPPED"
    )


def test_paper_vector_readback_rejects_corrupt_downloaded_manifest(tmp_path):
    root = _fixture(tmp_path, external=False, completion=False)
    manifest = root / "audit" / "baseline_sha256.txt"
    manifest.write_text("0" * 64 + "  config.json\n")
    with pytest.raises(ValueError, match="anchor mismatch"):
        verify(root, "initial")


def test_paper_vector_readback_rejects_extra_obs_object(tmp_path):
    root = _fixture(tmp_path, external=True, completion=True)
    (root / "audit" / "unexpected.txt").write_text("must fail\n")
    with pytest.raises(ValueError, match="exact inventory"):
        verify(root, "completion")


def test_paper_vector_readback_pair_rejects_shared_inodes(tmp_path):
    first = _fixture(tmp_path / "one", external=False, completion=False)
    second = _fixture(tmp_path / "two", external=False, completion=False)
    report = verify_pair(first, second, "initial")
    assert report["status"] == "PASS_INDEPENDENT_DOUBLE_READBACK"
    target = second / "audit" / "summary.json"
    target.unlink()
    target.hardlink_to(first / "audit" / "summary.json")
    with pytest.raises(ValueError, match="share file inodes"):
        verify_pair(first, second, "initial")
