import json
from pathlib import Path

import pytest
import torch

from deepjump.config import Config
from scripts import train_ddp


class _State:
    def __init__(self, payload):
        self.payload = payload

    def state_dict(self):
        return self.payload


def _checkpoint_args():
    cfg = Config()
    cfg.train.run_class = "formal"
    cfg.train.max_steps = 500_000
    cfg.train.lr_horizon_steps = 500_000
    return (
        _State({"weight": torch.ones(1)}),
        _State({"state": {}}),
        _State({}),
        cfg,
        {
            "world_size": 8,
            "crop_resume": "state_consistent_non_bitwise_crop_and_noise",
        },
    )


def test_training_semantics_include_num_workers():
    baseline = Config()
    candidate = Config()
    candidate.train.num_workers = baseline.train.num_workers + 1
    assert train_ddp.training_semantics_sha256(candidate) != (
        train_ddp.training_semantics_sha256(baseline)
    )


def test_numbered_checkpoint_is_durable_and_refuses_overwrite(tmp_path, monkeypatch):
    core, opt, scaler, cfg, state = _checkpoint_args()
    checkpoint = tmp_path / "ckpt_1000.pt"
    fsync_calls = []
    real_fsync = train_ddp.os.fsync

    def recording_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(train_ddp.os, "fsync", recording_fsync)
    train_ddp.save_ckpt(checkpoint, core, opt, scaler, 1000, cfg, state)
    assert checkpoint.is_file()
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["step"] == 1000
    assert len(fsync_calls) >= 2

    with pytest.raises(FileExistsError):
        train_ddp.save_ckpt(checkpoint, core, opt, scaler, 1000, cfg, state)
    assert not list(tmp_path.glob(".ckpt_1000.pt.tmp.*"))


def test_last_checkpoint_can_be_atomically_replaced(tmp_path):
    core, opt, scaler, cfg, state = _checkpoint_args()
    checkpoint = tmp_path / "last.ckpt"
    train_ddp.save_ckpt(checkpoint, core, opt, scaler, 1000, cfg, state)
    train_ddp.save_ckpt(checkpoint, core, opt, scaler, 2000, cfg, state)
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["step"] == 2000


def test_failed_checkpoint_write_cleans_unique_temporary(tmp_path, monkeypatch):
    core, opt, scaler, cfg, state = _checkpoint_args()

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(train_ddp.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="synthetic"):
        train_ddp.save_ckpt(
            tmp_path / "ckpt_1000.pt", core, opt, scaler, 1000, cfg, state
        )
    assert list(tmp_path.iterdir()) == []


def test_resume_history_requires_exact_completed_cadence(tmp_path):
    history_path = tmp_path / "history.json"
    records = [
        {"step": 10_000, "val_loss": 1.0, "val_rmsd": 2.0, "noop_rmsd": 3.0},
        {"step": 20_000, "val_loss": 0.9, "val_rmsd": 1.9, "noop_rmsd": 3.0},
    ]
    history_path.write_text(json.dumps(records))
    assert train_ddp._load_resume_history(
        history_path, start_step=20_500, val_every=10_000
    ) == records

    records.pop()
    history_path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="exact completed validation cadence"):
        train_ddp._load_resume_history(
            history_path, start_step=20_500, val_every=10_000
        )


def test_formal_graceful_stop_contract_is_optimizer_boundary_checkpoint():
    source = Path("scripts/train_ddp.py").read_text()
    assert "signal.SIGUSR1" in source
    assert "continue_state = torch.tensor(" in source
    assert source.index("continue_state = torch.tensor(") < source.index(
        "for g in opt.param_groups:"
    )
    assert "if not numbered.exists():" in source
    assert "publish_checkpoint(step)" in source
    assert "scaler.step(opt); scaler.update()" in source
    assert source.index("scaler.step(opt); scaler.update()") < source.index(
        "step += 1"
    )
    assert "return 75 if stopped_early else 0" in source
