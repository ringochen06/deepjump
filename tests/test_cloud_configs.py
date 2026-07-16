from __future__ import annotations

from dataclasses import asdict

from deepjump.config import load_config


def test_v100_stage_configs_share_training_footprint():
    smoke = load_config("configs/v100_ddp_smoke.yaml")
    calibration = load_config("configs/v100_ddp_calibration.yaml")
    formal = load_config("configs/v100_paper_d1.yaml")

    assert asdict(smoke.data) == asdict(calibration.data) == asdict(formal.data)
    assert asdict(smoke.model) == asdict(calibration.model) == asdict(formal.model)

    scheduling_fields = {"max_steps", "val_every", "log_every", "ckpt_every", "out_dir"}
    for config in (smoke, calibration):
        staged = asdict(config.train)
        reference = asdict(formal.train)
        for field in scheduling_fields:
            staged.pop(field)
            reference.pop(field)
        assert staged == reference

    assert smoke.train.max_steps == 100
    assert calibration.train.max_steps == 1000
    assert formal.train.max_steps == 100000
    assert smoke.train.batch_size * 8 * smoke.train.grad_accum == 128
