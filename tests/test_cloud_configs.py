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


def test_lr_diagnostics_change_only_bounded_schedule_fields():
    calibration = load_config("configs/v100_ddp_calibration.yaml")
    nowarmup = load_config("configs/v100_lr_diag_nowarmup.yaml")
    warmup200 = load_config("configs/v100_lr_diag_warmup200.yaml")

    assert asdict(nowarmup.data) == asdict(warmup200.data) == asdict(calibration.data)
    assert asdict(nowarmup.model) == asdict(warmup200.model) == asdict(calibration.model)
    assert nowarmup.train.max_steps == warmup200.train.max_steps == 500
    assert nowarmup.train.warmup_steps == 0
    assert warmup200.train.warmup_steps == 200
    assert nowarmup.train.lr == warmup200.train.lr == 5e-3
    assert nowarmup.train.lr_final == warmup200.train.lr_final == 3e-3


def test_robustness_diagnostics_preserve_core_footprint():
    baseline = load_config("configs/v100_lr_diag_warmup200.yaml")
    aug = load_config("configs/v100_robust_aug1.yaml")
    unroll = load_config("configs/v100_robust_unroll2.yaml")

    assert asdict(aug.data) == asdict(baseline.data)
    assert aug.model.input_aug_sigma == 1.0
    aug_model = asdict(aug.model)
    base_model = asdict(baseline.model)
    aug_model.pop("input_aug_sigma")
    base_model.pop("input_aug_sigma")
    assert aug_model == base_model

    unroll_data = asdict(unroll.data)
    base_data = asdict(baseline.data)
    unroll_data.pop("unroll")
    base_data.pop("unroll")
    assert unroll_data == base_data
    assert unroll.data.unroll == 2 and unroll.train.w_unroll == 0.5
    assert unroll.train.batch_size * 8 * unroll.train.grad_accum == 128
    assert aug.train.batch_size * 8 * aug.train.grad_accum == 128
    assert aug.train.max_steps == unroll.train.max_steps == 500


def test_followup_robustness_configs_preserve_effective_batch_and_bounds():
    unroll3 = load_config("configs/v100_robust_unroll3.yaml")
    unroll3_1000 = load_config("configs/v100_robust_unroll3_1000.yaml")
    unroll3_w1 = load_config("configs/v100_robust_unroll3_w1.yaml")
    unroll3_ca1 = load_config("configs/v100_robust_unroll3_ca1.yaml")
    unroll3_bond05 = load_config("configs/v100_robust_unroll3_bond05.yaml")
    unroll3_bond01 = load_config("configs/v100_robust_unroll3_bond01.yaml")
    corrected_bond = load_config("configs/v100_robust_unroll3_corrected_bond025.yaml")
    unroll5 = load_config("configs/v100_robust_unroll5.yaml")
    combo = load_config("configs/v100_robust_aug05_unroll2.yaml")

    assert unroll3.data.unroll == 3
    assert unroll3.model.input_aug_sigma == 0.0
    assert unroll3.train.batch_size * 8 * unroll3.train.grad_accum == 128
    assert combo.data.unroll == 2
    assert combo.model.input_aug_sigma == 0.5
    assert combo.train.batch_size * 8 * combo.train.grad_accum == 128
    assert unroll3.train.max_steps == combo.train.max_steps == 500
    assert unroll3_1000.data.unroll == 3
    assert unroll3_1000.model.input_aug_sigma == 0.0
    assert unroll3_1000.train.batch_size * 8 * unroll3_1000.train.grad_accum == 128
    assert unroll3_1000.train.max_steps == 1000
    assert unroll3_w1.data.unroll == 3
    assert unroll3_w1.train.w_unroll == 1.0
    assert unroll3_w1.train.batch_size * 8 * unroll3_w1.train.grad_accum == 128
    assert unroll3_w1.train.max_steps == 500
    assert asdict(unroll3_ca1.data) == asdict(unroll3.data)
    assert asdict(unroll3_ca1.model) == asdict(unroll3.model)
    assert unroll3_ca1.train.w_ca == 1.0
    assert unroll3_ca1.train.w_allatom == 1.0
    assert unroll3_ca1.train.w_unroll == 0.5
    assert unroll3_ca1.train.batch_size * 8 * unroll3_ca1.train.grad_accum == 128
    assert unroll3_ca1.train.max_steps == 500
    assert asdict(unroll3_bond05.data) == asdict(unroll3.data)
    assert asdict(unroll3_bond05.model) == asdict(unroll3.model)
    assert unroll3_bond05.train.w_ca == 0.0
    assert unroll3_bond05.train.w_bond == 0.5
    assert unroll3_bond05.train.w_allatom == 1.0
    assert unroll3_bond05.train.w_unroll == 0.5
    assert unroll3_bond05.train.batch_size * 8 * unroll3_bond05.train.grad_accum == 128
    assert unroll3_bond05.train.max_steps == 500
    assert asdict(unroll3_bond01.data) == asdict(unroll3.data)
    assert asdict(unroll3_bond01.model) == asdict(unroll3.model)
    assert unroll3_bond01.train.w_ca == 0.0
    assert unroll3_bond01.train.w_bond == 0.1
    assert unroll3_bond01.train.w_allatom == 1.0
    assert unroll3_bond01.train.w_unroll == 0.5
    assert unroll3_bond01.train.batch_size * 8 * unroll3_bond01.train.grad_accum == 128
    assert unroll3_bond01.train.max_steps == 500
    assert asdict(corrected_bond.data) == asdict(unroll3.data)
    assert asdict(corrected_bond.model) == asdict(unroll3.model)
    assert corrected_bond.train.w_bond == 0.0
    assert corrected_bond.train.w_bond_unroll == 0.25
    assert corrected_bond.train.w_allatom == 1.0
    assert corrected_bond.train.w_unroll == 0.5
    assert corrected_bond.train.batch_size * 8 * corrected_bond.train.grad_accum == 128
    assert corrected_bond.train.max_steps == 500
    assert unroll5.data.unroll == 5
    assert unroll5.train.w_unroll == 0.25
    assert unroll5.train.w_bond == unroll5.train.w_bond_unroll == 0.0

    localgeom = load_config("configs/v100_robust_unroll3_localgeom.yaml")
    assert localgeom.data.unroll == 3
    assert localgeom.train.max_steps == 1000
    assert localgeom.train.w_geom_length_unroll > 0
    assert localgeom.train.w_geom_angle_unroll > 0
    assert localgeom.train.w_bond == localgeom.train.w_bond_unroll == 0.0
    jointnoise = load_config("configs/v100_robust_unroll5_jointnoise.yaml")
    assert jointnoise.data.unroll == 5
    assert jointnoise.model.source_noise_v
    assert jointnoise.train.max_steps == 750
    assert jointnoise.train.w_unroll == 0.25
    paperstyle = load_config("configs/v100_paperstyle_d1_500.yaml")
    paperstyle_smoke = load_config("configs/v100_paperstyle_d1_smoke.yaml")
    paperstyle_2000 = load_config("configs/v100_paperstyle_d1_2000.yaml")
    paperstyle_unroll3 = load_config("configs/v100_paperstyle_unroll3_1000.yaml")
    tensorcloud_smoke = load_config("configs/v100_tensorcloud_d1_smoke.yaml")
    tensorcloud_adapt = load_config("configs/v100_tensorcloud_unroll3_adapt1000.yaml")
    tensorcloud01_overfit = load_config("configs/v100_tensorcloud01_overfit.yaml")
    tensorcloud01_smoke = load_config("configs/v100_tensorcloud01_d1_smoke.yaml")
    assert paperstyle.data.unroll == 1
    assert paperstyle.model.source_noise_v
    assert paperstyle.model.vector_qk and paperstyle.model.paper_ff
    assert paperstyle.train.batch_size * 8 * paperstyle.train.grad_accum == 128
    assert paperstyle.train.w_unroll == 0.0
    assert asdict(paperstyle_smoke.data) == asdict(paperstyle.data)
    assert asdict(paperstyle_smoke.model) == asdict(paperstyle.model)
    assert paperstyle_smoke.train.max_steps == 10
    assert paperstyle_smoke.train.batch_size * 8 * paperstyle_smoke.train.grad_accum == 128
    assert paperstyle_smoke.train.warmup_steps == 0
    assert paperstyle_smoke.train.lr == paperstyle_smoke.train.lr_final == 5e-3
    assert asdict(paperstyle_2000.data) == asdict(paperstyle.data)
    assert asdict(paperstyle_2000.model) == asdict(paperstyle.model)
    assert paperstyle_2000.train.max_steps == 2000
    assert paperstyle_2000.train.batch_size * 8 * paperstyle_2000.train.grad_accum == 128
    assert asdict(paperstyle_unroll3.model) == asdict(paperstyle.model)
    assert paperstyle_unroll3.data.unroll == 3
    assert paperstyle_unroll3.train.w_unroll == 0.5
    assert paperstyle_unroll3.train.max_steps == 1000
    assert paperstyle_unroll3.train.batch_size * 8 * paperstyle_unroll3.train.grad_accum == 128
    assert tensorcloud_smoke.model.tensor_qkv
    assert not tensorcloud_smoke.model.vector_qk
    assert tensorcloud_smoke.model.paper_ff
    assert tensorcloud_smoke.train.max_steps == 10
    assert tensorcloud_smoke.train.batch_size * 8 * tensorcloud_smoke.train.grad_accum == 128
    assert tensorcloud_smoke.train.lr == tensorcloud_smoke.train.lr_final == 5e-3
    assert asdict(tensorcloud_adapt.model) == asdict(tensorcloud_smoke.model)
    assert tensorcloud_adapt.data.unroll == 3
    assert tensorcloud_adapt.train.w_unroll == 0.5
    assert tensorcloud_adapt.train.max_steps == 1000
    assert tensorcloud_adapt.train.warmup_steps == 200
    assert tensorcloud01_overfit.data.domains == ["1a0hA01"]
    assert tensorcloud01_overfit.model.tensor_cloud01
    assert tensorcloud01_overfit.model.hidden == tensorcloud01_overfit.model.vector_channels == 32
    assert not tensorcloud01_overfit.train.amp
    assert tensorcloud01_overfit.train.batch_size == 1
    assert tensorcloud01_smoke.model.tensor_cloud01
    assert tensorcloud01_smoke.model.hidden == tensorcloud01_smoke.model.vector_channels == 128
    assert not tensorcloud01_smoke.model.vector_qk
    assert not tensorcloud01_smoke.model.tensor_qkv
    assert not tensorcloud01_smoke.model.paper_ff
    assert tensorcloud01_smoke.train.max_steps == 10
    assert tensorcloud01_smoke.train.batch_size * 8 * tensorcloud01_smoke.train.grad_accum == 128
    assert tensorcloud01_smoke.train.amp_dtype == "fp16"
    assert tensorcloud01_smoke.train.lr == tensorcloud01_smoke.train.lr_final == 5e-3
    assert unroll5.train.batch_size * 8 * unroll5.train.grad_accum == 128
    assert unroll5.train.max_steps == 500
