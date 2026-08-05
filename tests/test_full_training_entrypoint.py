import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from deepjump.config import Config, load_config
from deepjump.data.mdcath import _DomainHandle
from scripts import train as train_single
from scripts import train_ddp
from scripts.train import validate_single_process_scope
from scripts.train_ddp import (
    bind_full_training_contract,
    training_semantics_sha256,
    validate_checkpoint_mode,
)


def test_formal_training_requires_contract():
    cfg = Config()
    cfg.train.run_class = "formal"
    with pytest.raises(ValueError, match="sealed full training data contract"):
        bind_full_training_contract(cfg)


def test_full_data_stage_requires_contract():
    cfg = Config()
    cfg.train.run_class = "full_data_stage"
    with pytest.raises(ValueError, match="sealed full training data contract"):
        bind_full_training_contract(cfg)


def test_long_run_cannot_hide_as_development():
    cfg = Config()
    cfg.train.max_steps = 100_000
    with pytest.raises(ValueError, match="run_class=formal"):
        bind_full_training_contract(cfg)


def test_contract_cli_identity_must_be_complete():
    cfg = Config()
    with pytest.raises(ValueError, match="must be set together"):
        bind_full_training_contract(cfg, "contract.json", None)


@pytest.mark.parametrize(
    ("warm_start", "allow_legacy", "message"),
    [
        ("old.pt", False, "cannot warm-start"),
        (None, True, "cannot allow legacy resume"),
    ],
)
def test_contracted_run_rejects_contaminated_checkpoint_modes(
    warm_start, allow_legacy, message
):
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_mode(
            {"status": "PASS_FULL_TRAINING_DATA_CONTRACT"},
            warm_start,
            allow_legacy,
        )


def test_production_stage_configs_declare_contract_requirement():
    assert load_config("configs/v100_ddp_smoke.yaml").train.run_class == "full_data_stage"
    assert load_config("configs/v100_ddp_calibration.yaml").train.run_class == "full_data_stage"
    assert load_config("configs/v100_paper_d1.yaml").train.run_class == "formal"
    for name in ("paper_h128_d1.yaml", "paper_h128_d10.yaml", "paper_h128_d100.yaml"):
        assert load_config(f"configs/{name}").train.run_class == "formal"


def test_legacy_single_process_trainer_rejects_formal_config():
    cfg = Config()
    cfg.train.run_class = "formal"
    with pytest.raises(ValueError, match="development-only"):
        validate_single_process_scope(cfg)


def test_legacy_single_process_trainer_rejects_sealed_domains_file():
    cfg = Config()
    cfg.data.domains_file = "train_eligible_5218.txt"
    with pytest.raises(ValueError, match="sealed domains files"):
        validate_single_process_scope(cfg)


def test_training_semantics_hash_changes_with_scientific_configuration():
    baseline = Config()
    candidate = Config()
    candidate.train.lr = baseline.train.lr / 2
    assert training_semantics_sha256(candidate) != training_semantics_sha256(baseline)
    candidate = Config()
    candidate.model.tensor_cloud01 = True
    assert training_semantics_sha256(candidate) != training_semantics_sha256(baseline)
    candidate = Config()
    candidate.data.unroll = 3
    assert training_semantics_sha256(candidate) != training_semantics_sha256(baseline)
    candidate = Config()
    candidate.train.max_steps = baseline.train.max_steps * 2
    assert training_semantics_sha256(candidate) != training_semantics_sha256(baseline)


def test_training_semantics_hash_allows_only_operational_cadence_changes():
    baseline = Config()
    baseline.train.lr_horizon_steps = 5_000
    candidate = Config()
    candidate.train.lr_horizon_steps = 5_000
    candidate.train.out_dir = "another-run"
    candidate.train.max_steps = 2_000
    candidate.train.log_every = 1
    candidate.train.val_every = 2
    candidate.train.ckpt_every = 3
    candidate.train.keep_last_k = 9
    candidate.train.num_workers = 4
    assert training_semantics_sha256(candidate) == training_semantics_sha256(baseline)


def test_contracted_dataset_rejects_post_audit_fingerprint_drift(tmp_path):
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    payload = data / "mdcath_dataset_d1.h5"
    payload.write_bytes(b"same-size")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '[{"file":"mdcath_dataset_d1.h5","domain":"d1","size":9,'
        '"local_fingerprint":{"device":0,"inode":0,"size":9,"mtime_ns":0,"ctime_ns":0}}]'
    )
    cfg = Config()
    cfg.data.root = str(root)
    cfg.data.manifest = str(manifest)
    domains = tmp_path / "train.txt"
    domains.write_text("d1\n")
    cfg.data.domains_file = str(domains)
    cfg.data.full_training_contract = "contract.json"
    verification = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "train_list_sha256": hashlib.sha256(domains.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        train_ddp.build_datasets(cfg, verification)


def test_contracted_dataset_parses_the_exact_verified_artifact_bytes(tmp_path):
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    payload = data / "mdcath_dataset_d1.h5"
    payload.write_bytes(b"payload")
    stat_result = payload.stat()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "file": payload.name,
        "domain": "d1",
        "local_fingerprint": {
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "ctime_ns": stat_result.st_ctime_ns,
        },
    }]))
    domains = tmp_path / "train.txt"
    domains.write_text("d1\n")
    cfg = Config()
    cfg.data.root = str(root)
    cfg.data.manifest = str(manifest)
    cfg.data.domains_file = str(domains)
    cfg.data.full_training_contract = "contract.json"
    verification = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "train_list_sha256": hashlib.sha256(domains.read_bytes()).hexdigest(),
    }
    domains.write_text("d2\n")
    with pytest.raises(ValueError, match="train list SHA256 mismatch"):
        train_ddp.build_datasets(cfg, verification)


def _write_coords(path: Path, values: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        domain = handle.create_group("d1")
        temperature = domain.create_group("320")
        replica = temperature.create_group("0")
        replica.create_dataset("coords", data=values)


def _fingerprint(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def test_lazy_hdf5_open_rejects_post_construction_replacement(tmp_path):
    payload = tmp_path / "mdcath_dataset_d1.h5"
    _write_coords(payload, np.zeros((2, 1, 3), dtype=np.float32))
    expected = _fingerprint(payload)
    replacement = tmp_path / "replacement.h5"
    _write_coords(replacement, np.ones((2, 1, 3), dtype=np.float32))
    handle = _DomainHandle(payload, expected)
    os.replace(replacement, payload)
    with pytest.raises(ValueError, match="fingerprint changed before first open"):
        _ = handle.f


def test_lazy_hdf5_open_rejects_symlink(tmp_path):
    target = tmp_path / "target.h5"
    _write_coords(target, np.zeros((2, 1, 3), dtype=np.float32))
    payload = tmp_path / "mdcath_dataset_d1.h5"
    payload.symlink_to(target)
    handle = _DomainHandle(payload)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _ = handle.f


def test_lazy_hdf5_open_uses_pinned_descriptor_after_path_replacement(tmp_path):
    payload = tmp_path / "mdcath_dataset_d1.h5"
    original = np.zeros((2, 1, 3), dtype=np.float32)
    replacement_values = np.ones((2, 1, 3), dtype=np.float32)
    _write_coords(payload, original)
    descriptor = os.open(payload, os.O_RDONLY)
    replacement = tmp_path / "replacement.h5"
    _write_coords(replacement, replacement_values)
    os.replace(replacement, payload)
    expected = _DomainHandle._fingerprint(os.fstat(descriptor))
    handle = _DomainHandle(payload, expected, descriptor=descriptor)
    try:
        assert np.array_equal(handle.f["d1"]["320"]["0"]["coords"][:], original)
        handle.close()
        assert os.fstat(descriptor).st_ino == expected["inode"]
    finally:
        handle.close()
        os.close(descriptor)


def test_every_coordinate_read_must_be_finite(tmp_path):
    payload = tmp_path / "mdcath_dataset_d1.h5"
    values = np.zeros((3, 1, 3), dtype=np.float32)
    values[1, 0, 2] = np.nan
    _write_coords(payload, values)
    handle = _DomainHandle(payload, _fingerprint(payload))
    try:
        assert np.isfinite(handle.coords(320, 0, 0)).all()
        with pytest.raises(ValueError, match="numeric and finite"):
            handle.coords(320, 0, 1)
    finally:
        handle.close()


def test_more_than_1000_domains_cannot_use_uncontracted_entrypoint(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(
        train_ddp,
        "discover_domains",
        lambda _root: [Path(f"mdcath_dataset_d{i}.h5") for i in range(1_001)],
    )
    with pytest.raises(ValueError, match="more than 1,000 domains"):
        train_ddp.build_datasets(cfg)


def test_legacy_single_process_trainer_rejects_more_than_1000_domains(monkeypatch):
    cfg = Config()
    monkeypatch.setattr(
        train_single,
        "discover_domains",
        lambda _root: [Path(f"mdcath_dataset_d{i}.h5") for i in range(1_001)],
    )
    with pytest.raises(ValueError, match="more than 1,000 domains"):
        train_single.build_loaders(cfg)
