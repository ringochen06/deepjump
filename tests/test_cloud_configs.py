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
    tensorcloud01_fp32 = load_config("configs/v100_tensorcloud01_fp32_lr5e3_probe.yaml")
    tensorcloud01_warmup = load_config("configs/v100_tensorcloud01_fp16_warmup20_probe.yaml")
    tensorcloud01_lowlr = load_config("configs/v100_tensorcloud01_fp16_lr5e4_probe.yaml")
    tensorcloud01_vector_only = load_config(
        "configs/v100_tensorcloud01_vector_only_d1_calibration.yaml"
    )
    tensorcloud01_vector_only_lowlr = load_config(
        "configs/v100_tensorcloud01_vector_only_d1_lowlr_calibration.yaml"
    )
    tensorcloud01_vector_only_fp32 = load_config(
        "configs/v100_tensorcloud01_vector_only_d1_fp32_calibration.yaml"
    )
    tensorcloud01_vector_only_fp32_continuation = load_config(
        "configs/v100_tensorcloud01_vector_only_d1_fp32_continuation2000.yaml"
    )
    vector_fp32_highlr = load_config(
        "configs/v100_tensorcloud01_vector_only_fp32_highlr_step230.yaml"
    )
    vector_fp16_lowlr = load_config(
        "configs/v100_tensorcloud01_vector_only_fp16_lowlr_step230.yaml"
    )
    full_tiny_domain = load_config(
        "configs/v100_tensorcloud01_full_d1_tiny_domain5000.yaml"
    )
    full_feedback_adapt = load_config(
        "configs/v100_tensorcloud01_full_d1_unroll3_adapt250.yaml"
    )
    full_feedback_bond17 = load_config(
        "configs/v100_tensorcloud01_full_d1_unroll3_bond17_adapt250.yaml"
    )
    first_party_source_law = load_config(
        "configs/v100_tensorcloud01_full_d1_first_party_source_law1000.yaml"
    )
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
    assert tensorcloud01_vector_only.model.tensor_cloud01
    assert tensorcloud01_vector_only.model.tensor_cloud01_vector_only_attention
    assert tensorcloud01_vector_only.model.hidden == 128
    assert tensorcloud01_vector_only.model.vector_channels == 128
    assert tensorcloud01_vector_only.data.delta_frames == 1
    assert tensorcloud01_vector_only.train.max_steps == 1000
    assert (
        tensorcloud01_vector_only.train.batch_size
        * 8
        * tensorcloud01_vector_only.train.grad_accum
        == 128
    )
    assert tensorcloud01_smoke.train.max_steps == 30
    assert tensorcloud01_smoke.train.warmup_steps == 20
    assert tensorcloud01_smoke.train.val_every == 30
    assert tensorcloud01_smoke.train.ckpt_every == 30
    assert tensorcloud01_smoke.train.batch_size * 8 * tensorcloud01_smoke.train.grad_accum == 128
    assert tensorcloud01_smoke.train.amp_dtype == "fp16"
    assert tensorcloud01_smoke.train.lr == tensorcloud01_smoke.train.lr_final == 5e-3
    for probe in (tensorcloud01_fp32, tensorcloud01_warmup, tensorcloud01_lowlr):
        assert asdict(probe.data) == asdict(tensorcloud01_smoke.data)
        assert asdict(probe.model) == asdict(tensorcloud01_smoke.model)
        assert probe.train.batch_size * 8 * probe.train.grad_accum == 128
    assert not tensorcloud01_fp32.train.amp
    assert tensorcloud01_fp32.train.lr == tensorcloud01_fp32.train.lr_final == 5e-3
    assert tensorcloud01_fp32.train.max_steps == 3
    assert tensorcloud01_warmup.train.amp
    assert tensorcloud01_warmup.train.warmup_steps == 20
    assert tensorcloud01_warmup.train.max_steps == 30
    assert tensorcloud01_warmup.train.lr == tensorcloud01_warmup.train.lr_final == 5e-3
    assert tensorcloud01_lowlr.train.amp
    assert tensorcloud01_lowlr.train.warmup_steps == 0
    assert tensorcloud01_lowlr.train.max_steps == 30
    assert tensorcloud01_lowlr.train.lr == tensorcloud01_lowlr.train.lr_final == 5e-4
    calibrations = [
        load_config("configs/v100_tensorcloud01_d1_calibration.yaml"),
        load_config("configs/v100_tensorcloud01_d10_calibration.yaml"),
        load_config("configs/v100_tensorcloud01_d100_calibration.yaml"),
    ]
    full_d1 = calibrations[0]
    assert asdict(tensorcloud01_vector_only.data) == asdict(full_d1.data)
    vector_model = asdict(tensorcloud01_vector_only.model)
    full_model = asdict(full_d1.model)
    assert vector_model.pop("tensor_cloud01_vector_only_attention")
    assert not full_model.pop("tensor_cloud01_vector_only_attention")
    assert vector_model == full_model
    vector_train = asdict(tensorcloud01_vector_only.train)
    full_train = asdict(full_d1.train)
    vector_train.pop("out_dir")
    full_train.pop("out_dir")
    assert vector_train == full_train
    assert asdict(tensorcloud01_vector_only_lowlr.data) == asdict(
        tensorcloud01_vector_only.data
    )
    assert asdict(tensorcloud01_vector_only_lowlr.model) == asdict(
        tensorcloud01_vector_only.model
    )
    reference_train = asdict(tensorcloud01_vector_only.train)
    lowlr_train = asdict(tensorcloud01_vector_only_lowlr.train)
    for key in ("lr", "lr_final", "lr_horizon_steps", "out_dir"):
        reference_train.pop(key)
        lowlr_train.pop(key)
    assert reference_train == lowlr_train
    assert tensorcloud01_vector_only_lowlr.train.lr == 5e-4
    assert tensorcloud01_vector_only_lowlr.train.lr_final == 3e-4
    assert tensorcloud01_vector_only_lowlr.train.lr_horizon_steps == 1000
    assert asdict(tensorcloud01_vector_only_fp32.data) == asdict(
        tensorcloud01_vector_only.data
    )
    assert asdict(tensorcloud01_vector_only_fp32.model) == asdict(
        tensorcloud01_vector_only.model
    )
    reference_train = asdict(tensorcloud01_vector_only.train)
    fp32_train = asdict(tensorcloud01_vector_only_fp32.train)
    for key in ("amp", "lr_horizon_steps", "out_dir"):
        reference_train.pop(key)
        fp32_train.pop(key)
    assert reference_train == fp32_train
    assert not tensorcloud01_vector_only_fp32.train.amp
    assert tensorcloud01_vector_only_fp32.train.lr == 5e-3
    assert tensorcloud01_vector_only_fp32.train.lr_final == 3e-3
    assert tensorcloud01_vector_only_fp32.train.lr_horizon_steps == 1000
    assert asdict(tensorcloud01_vector_only_fp32_continuation.data) == asdict(
        tensorcloud01_vector_only_fp32.data
    )
    assert asdict(tensorcloud01_vector_only_fp32_continuation.model) == asdict(
        tensorcloud01_vector_only_fp32.model
    )
    continuation_train = asdict(tensorcloud01_vector_only_fp32_continuation.train)
    fp32_reference_train = asdict(tensorcloud01_vector_only_fp32.train)
    for key in ("max_steps", "val_every", "ckpt_every", "keep_last_k", "out_dir"):
        continuation_train.pop(key)
        fp32_reference_train.pop(key)
    assert continuation_train == fp32_reference_train
    assert tensorcloud01_vector_only_fp32_continuation.train.max_steps == 2000
    assert tensorcloud01_vector_only_fp32_continuation.train.val_every == 100
    assert tensorcloud01_vector_only_fp32_continuation.train.ckpt_every == 100
    assert tensorcloud01_vector_only_fp32_continuation.train.keep_last_k == 10
    assert tensorcloud01_vector_only_fp32_continuation.train.lr_horizon_steps == 1000
    assert not tensorcloud01_vector_only_fp32_continuation.train.amp
    full_reference = load_config("configs/v100_tensorcloud01_full_d1_fp32_calibration.yaml")
    assert asdict(full_tiny_domain.model) == asdict(full_reference.model)
    assert full_tiny_domain.data.domains == ["1a0hA01"]
    assert full_tiny_domain.data.temperatures == full_reference.data.temperatures
    assert full_tiny_domain.data.replicas == full_reference.data.replicas
    assert full_tiny_domain.data.delta_frames == full_reference.data.delta_frames == 1
    assert full_tiny_domain.data.crop_length == full_reference.data.crop_length == 256
    assert full_tiny_domain.data.noise_sigma == full_reference.data.noise_sigma == 0.1
    assert full_tiny_domain.train.batch_size * 8 * full_tiny_domain.train.grad_accum == 128
    assert full_tiny_domain.train.max_steps == 5000
    assert full_tiny_domain.train.lr_horizon_steps == 500000
    assert full_tiny_domain.train.warmup_steps == 200
    assert full_tiny_domain.train.val_every == 500
    assert full_tiny_domain.train.ckpt_every == 500
    assert full_tiny_domain.train.keep_last_k == 3
    assert full_tiny_domain.train.w_ca == 0.0
    assert full_tiny_domain.train.w_allatom == 1.0
    assert not full_tiny_domain.train.amp
    source_data = asdict(first_party_source_law.data)
    baseline_data = asdict(full_tiny_domain.data)
    assert source_data.pop("noise_sigma") == 1.5
    assert baseline_data.pop("noise_sigma") == 0.1
    assert source_data == baseline_data
    source_model = asdict(first_party_source_law.model)
    baseline_model = asdict(full_tiny_domain.model)
    assert source_model.pop("source_noise_sigma_v") == 1.0
    assert baseline_model.pop("source_noise_sigma_v") is None
    assert source_model == baseline_model
    source_train = asdict(first_party_source_law.train)
    baseline_train = asdict(full_tiny_domain.train)
    for key in ("max_steps", "val_every", "ckpt_every", "keep_last_k", "out_dir"):
        source_train.pop(key)
        baseline_train.pop(key)
    assert source_train == baseline_train
    assert first_party_source_law.train.max_steps == 1000
    assert first_party_source_law.train.batch_size * 8 * first_party_source_law.train.grad_accum == 128
    assert asdict(full_feedback_adapt.model) == asdict(full_tiny_domain.model)
    assert full_feedback_adapt.data.domains == full_tiny_domain.data.domains
    assert full_feedback_adapt.data.unroll == 3
    assert full_feedback_adapt.train.w_unroll == 0.5
    assert full_feedback_adapt.train.max_steps == 250
    assert full_feedback_adapt.train.batch_size * 8 * full_feedback_adapt.train.grad_accum == 128
    assert not full_feedback_adapt.train.amp
    assert asdict(full_feedback_bond17.data) == asdict(full_feedback_adapt.data)
    assert asdict(full_feedback_bond17.model) == asdict(full_feedback_adapt.model)
    bond17_train = asdict(full_feedback_bond17.train)
    feedback_train = asdict(full_feedback_adapt.train)
    assert bond17_train.pop("w_bond_unroll") == 1.7
    assert feedback_train.pop("w_bond_unroll") == 0.0
    assert bond17_train.pop("out_dir") != feedback_train.pop("out_dir")
    assert bond17_train == feedback_train
    for probe in (vector_fp32_highlr, vector_fp16_lowlr):
        assert asdict(probe.data) == asdict(tensorcloud01_vector_only.data)
        assert asdict(probe.model) == asdict(tensorcloud01_vector_only.model)
        assert probe.train.batch_size * 8 * probe.train.grad_accum == 128
        assert probe.train.max_steps == 230
        assert probe.train.lr_horizon_steps == 1000
        assert probe.train.warmup_steps == 200
        assert probe.train.val_every == probe.train.ckpt_every == 230
        assert probe.train.keep_last_k == 1
    fp32_train = asdict(vector_fp32_highlr.train)
    fp16_train = asdict(vector_fp16_lowlr.train)
    for key in ("amp", "lr", "lr_final", "out_dir"):
        fp32_train.pop(key)
        fp16_train.pop(key)
    assert fp32_train == fp16_train
    assert not vector_fp32_highlr.train.amp
    assert vector_fp32_highlr.train.lr == 5e-3
    assert vector_fp32_highlr.train.lr_final == 3e-3
    assert vector_fp16_lowlr.train.amp
    assert vector_fp16_lowlr.train.amp_dtype == "fp16"
    assert vector_fp16_lowlr.train.lr == 5e-4
    assert vector_fp16_lowlr.train.lr_final == 3e-4
    for calibration, expected_delta in zip(calibrations, (1, 10, 100), strict=True):
        calibration_data = asdict(calibration.data)
        smoke_data = asdict(tensorcloud01_smoke.data)
        calibration_data.pop("delta_frames")
        smoke_data.pop("delta_frames")
        assert calibration_data == smoke_data
        assert asdict(calibration.model) == asdict(tensorcloud01_smoke.model)
        assert calibration.data.delta_frames == expected_delta
        assert calibration.train.batch_size * 8 * calibration.train.grad_accum == 128
        assert calibration.train.max_steps == 1000
        assert calibration.train.warmup_steps == 200
        assert calibration.train.lr == 5e-3
        assert calibration.train.lr_final == 3e-3
        assert calibration.train.val_every == calibration.train.ckpt_every == 250
        assert calibration.train.keep_last_k == 4
        assert calibration.train.amp and calibration.train.amp_dtype == "fp16"
        assert calibration.train.out_dir.endswith(f"d{expected_delta}_calibration")
    assert unroll5.train.batch_size * 8 * unroll5.train.grad_accum == 128
    assert unroll5.train.max_steps == 500


def test_full_tensor_fp32_discriminator_matches_vector_only_budget():
    vector_calibration = load_config(
        "configs/v100_tensorcloud01_vector_only_d1_fp32_calibration.yaml"
    )
    vector_continuation = load_config(
        "configs/v100_tensorcloud01_vector_only_d1_fp32_continuation2000.yaml"
    )
    full_calibration = load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_calibration.yaml"
    )
    full_continuation = load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_continuation2000.yaml"
    )

    for vector, full in (
        (vector_calibration, full_calibration),
        (vector_continuation, full_continuation),
    ):
        assert asdict(vector.data) == asdict(full.data)
        vector_model = asdict(vector.model)
        full_model = asdict(full.model)
        assert vector_model.pop("tensor_cloud01_vector_only_attention") is True
        assert full_model.pop("tensor_cloud01_vector_only_attention") is False
        assert vector_model == full_model
        vector_train = asdict(vector.train)
        full_train = asdict(full.train)
        vector_train.pop("out_dir")
        full_train.pop("out_dir")
        assert vector_train == full_train

    assert not full_calibration.train.amp
    assert not full_continuation.train.amp
    assert full_calibration.train.max_steps == 1000
    assert full_continuation.train.max_steps == 2000
    assert full_calibration.train.lr_horizon_steps == 1000
    assert full_continuation.train.lr_horizon_steps == 1000


def test_paper_horizon_ab_configs_are_matched_fresh_continuous_runs():
    baseline = load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_horizon_ab_baseline2000.yaml"
    )
    candidate = load_config(
        "configs/v100_tensorcloud01_full_d1_fp32_paper_horizon500k_2000.yaml"
    )
    assert asdict(baseline.data) == asdict(candidate.data)
    assert asdict(baseline.model) == asdict(candidate.model)
    baseline_train = asdict(baseline.train)
    candidate_train = asdict(candidate.train)
    assert baseline_train.pop("lr_horizon_steps") == 1000
    assert candidate_train.pop("lr_horizon_steps") == 500000
    assert baseline_train.pop("out_dir").endswith("horizon_ab_baseline2000")
    assert candidate_train.pop("out_dir").endswith("paper_horizon500k_2000")
    assert baseline_train == candidate_train
    for config in (baseline, candidate):
        assert config.train.max_steps == 2000
        assert config.train.resume == ""
        assert config.train.batch_size * 8 * config.train.grad_accum == 128
        assert config.train.ckpt_every == 1000
        assert config.train.keep_last_k == 2
        assert config.train.amp is False
