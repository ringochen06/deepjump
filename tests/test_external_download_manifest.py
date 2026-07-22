import hashlib
import json
from pathlib import Path

import pytest

from scripts.external_endpoint_identity import verify_paper_vector_external_evidence
from scripts.write_external_download_manifest import build_manifest


PANEL = Path("configs/paper_horizon_external_dev_20_length_proportional_seed20260723.txt")
PANEL_SHA = "9c53aa3a5ccbc08531dea066b8ba09914f1a6b45bf3a3500d24d966ed21381bb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path) -> tuple[Path, Path, Path, int]:
    root = tmp_path / "external"
    data = root / "data"
    data.mkdir(parents=True)
    domains = PANEL.read_text().splitlines()
    rows = []
    for index, domain in enumerate(domains):
        path = data / f"mdcath_dataset_{domain}.h5"
        path.write_bytes((domain + str(index)).encode())
        rows.append({
            "domain": domain,
            "file": path.name,
            "bytes": path.stat().st_size,
            "residues": 10 + index,
            "trajectories": 25,
            "min_frames": 3,
        })
    total = sum(row["bytes"] for row in rows)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "status": "PASS",
        "domain_list_sha256": PANEL_SHA,
        "h5_files": 20,
        "total_bytes": total,
        "trajectories": 500,
        "unresolved_failures": 0,
        "domains": rows,
    }))
    claim = tmp_path / "claim.json"
    claim.write_text(json.dumps({
        "status": "CLAIMED_FOR_SINGLE_USE",
        "panel_sha256": PANEL_SHA,
    }))
    return root, audit, claim, total


def test_external_download_manifest_hashes_exact_ordered_hdf5_inventory(tmp_path):
    root, audit, claim, total = _case(tmp_path)
    report = build_manifest(
        root=root,
        domain_list=PANEL,
        domain_list_sha256=PANEL_SHA,
        expected_bytes=total,
        audit_path=audit,
        claim_path=claim,
        claim_sha256=_sha(claim),
        run_id="20260722T120000Z",
        commit="a" * 40,
    )
    assert report["status"] == "PASS"
    assert report["files_count"] == 20
    assert report["trajectories"] == 500
    assert [row["domain"] for row in report["files"]] == PANEL.read_text().splitlines()
    assert all(len(row["sha256"]) == 64 for row in report["files"])


def test_external_download_manifest_rejects_extra_or_symlink_hdf5(tmp_path):
    root, audit, claim, total = _case(tmp_path)
    extra = root / "data" / "mdcath_dataset_extra.h5"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="exact inventory"):
        build_manifest(
            root=root, domain_list=PANEL, domain_list_sha256=PANEL_SHA,
            expected_bytes=total, audit_path=audit, claim_path=claim,
            claim_sha256=_sha(claim), run_id="run", commit="commit",
        )
    extra.unlink()
    target = root / "data" / f"mdcath_dataset_{PANEL.read_text().splitlines()[0]}.h5"
    target.unlink()
    target.symlink_to(claim)
    with pytest.raises(ValueError, match="exact inventory|symlink"):
        build_manifest(
            root=root, domain_list=PANEL, domain_list_sha256=PANEL_SHA,
            expected_bytes=total, audit_path=audit, claim_path=claim,
            claim_sha256=_sha(claim), run_id="run", commit="commit",
        )


def test_external_evidence_rechecks_exact_fixed_source_proof(tmp_path):
    source_proof = {
        "schema": "deepjump.prior_source_control_flow_proof.v1",
        "status": "PASS_PRIOR_AUTHORITATIVE_RUN_EXTERNAL_UNCONSUMED",
        "source_run_id": "20260722T012922Z",
        "source_commit": "dbbc86daa1bc7dd123d52924f7ab6eed21c96b9b",
        "source_audit_obs_uri": (
            "obs://deepjump-mdcath-cn4-ringochen/deepjump-calibration/"
            "paper-horizon-ab2000/20260722T012922Z/audit"
        ),
        "source_decision_sha256": (
            "2367d8d29fc02e9a53ec8672b6cb4e2ef9f06ef9ae265f2cffd9f905dcd91d38"
        ),
        "source_runner_sha256": (
            "2c8eedad191a814080303b6a30204fbb9bee522937c3a0cb5087e3439b6bd75f"
        ),
        "source_status": "STOP_PAPER_HORIZON_OBJECTIVE_GAIN",
        "required_advance_status": "ADVANCE_PAPER_HORIZON_EXTERNAL20",
        "proof_basis": "fixed_decision_and_fixed_runner_control_flow",
        "prior_authoritative_run_consumed": False,
    }
    proof_path = tmp_path / "source_proof.json"
    proof_path.write_text(json.dumps(source_proof))
    proof_sha = _sha(proof_path)
    files = [
        {
            "domain": f"domain-{index}",
            "relative_path": f"data/domain-{index}.h5",
            "bytes": index + 1,
            "sha256": f"{index:064x}",
            "residues": 10,
            "trajectories": 25,
            "min_frames": 3,
        }
        for index in range(20)
    ]
    inventory_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "deepjump.external_download_inventory.v1",
        "status": "PASS",
        "panel_sha256": PANEL_SHA,
        "claim_sha256": "placeholder",
        "run_id": "20260722T120000Z",
        "commit": "a" * 40,
        "root": "/not-used",
        "files_count": 20,
        "total_bytes": 14236836972,
        "trajectories": 500,
        "unresolved_failures": 0,
        "inventory_sha256": inventory_sha,
        "files": files,
    }))
    claim_path = tmp_path / "claim.json"
    claim = {
        "schema": "deepjump.external_panel_claim.v1",
        "status": "CLAIMED_FOR_SINGLE_USE",
        "run_id": "20260722T120000Z",
        "commit": "a" * 40,
        "panel_sha256": PANEL_SHA,
        "panel_count": 20,
        "expected_total_bytes": 14236836972,
        "source_stop_decision_sha256": source_proof["source_decision_sha256"],
        "source_proof_sha256": proof_sha,
        "training_ab_decision_sha256": "b" * 64,
        "baseline_checkpoint_sha256": "c" * 64,
        "candidate_checkpoint_sha256": "d" * 64,
        "prior_authoritative_run_consumed": False,
        "claimed_at": "2026-07-22T12:00:00+00:00",
    }
    claim_path.write_text(json.dumps(claim))
    claim_sha = _sha(claim_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["claim_sha256"] = claim_sha
    manifest_path.write_text(json.dumps(manifest))
    kwargs = dict(
        claim_path=claim_path,
        expected_claim_sha256=claim_sha,
        manifest_path=manifest_path,
        expected_manifest_sha256=_sha(manifest_path),
        expected_panel_sha256=PANEL_SHA,
        expected_prerequisite_decision_sha256="b" * 64,
        expected_baseline_checkpoint_sha256="c" * 64,
        expected_candidate_checkpoint_sha256="d" * 64,
        source_proof_path=proof_path,
        expected_source_proof_sha256=proof_sha,
    )
    assert verify_paper_vector_external_evidence(**kwargs)["source_proof_sha256"] == proof_sha
    source_proof["forged_extra"] = True
    proof_path.write_text(json.dumps(source_proof))
    forged_sha = _sha(proof_path)
    claim["source_proof_sha256"] = forged_sha
    claim_path.write_text(json.dumps(claim))
    forged_claim_sha = _sha(claim_path)
    manifest["claim_sha256"] = forged_claim_sha
    manifest_path.write_text(json.dumps(manifest))
    kwargs.update(
        expected_source_proof_sha256=forged_sha,
        expected_claim_sha256=forged_claim_sha,
        expected_manifest_sha256=_sha(manifest_path),
    )
    with pytest.raises(ValueError, match="source proof exact schema"):
        verify_paper_vector_external_evidence(**kwargs)
