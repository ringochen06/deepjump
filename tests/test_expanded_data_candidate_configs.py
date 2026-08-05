from __future__ import annotations

from dataclasses import asdict

from deepjump.config import load_config
from scripts.train_ddp import training_semantics_sha256


CONFIG_PATHS = {
    "smoke": "configs/v100_tensorcloud01_full_expanded_d1_smoke100.yaml",
    "calibration": (
        "configs/v100_tensorcloud01_full_expanded_d1_calibration1000.yaml"
    ),
    "development": (
        "configs/v100_tensorcloud01_full_expanded_d1_development2000.yaml"
    ),
}

ALLOWED_STAGE_TRAIN_DELTAS = {
    "max_steps",
    "val_every",
    "log_every",
    "ckpt_every",
    "keep_last_k",
    "out_dir",
}


def _configs():
    return {name: load_config(path) for name, path in CONFIG_PATHS.items()}


def test_expanded_data_candidate_has_frozen_full_tensor_scientific_semantics():
    configs = _configs()
    expected_model = {
        "hidden": 128,
        "vector_channels": 128,
        "num_heads": 4,
        "cond_layers": 6,
        "transport_layers": 6,
        "seq_embed_ks": 32,
        "num_dist_basis": 16,
        "dist_cutoff": 25.0,
        "predict_heavy": True,
        "input_aug_sigma": 0.0,
        "source_noise_v": True,
        "source_noise_sigma_v": None,
        "vector_qk": False,
        "tensor_qkv": False,
        "paper_ff": False,
        "tensor_cloud01": True,
        "tensor_cloud01_vector_only_attention": False,
        "tensor_cloud01_vector_only_scalar_value": False,
    }
    expected_data = {
        "root": "",
        "domains": [],
        "domains_file": "",
        "temperatures": [320, 348, 379, 413, 450],
        "replicas": [0, 1, 2, 3, 4],
        "delta_frames": 1,
        "crop_length": 256,
        "val_fraction": 0.02,
        "noise_sigma": 0.1,
        "unroll": 1,
        "canon_symmetric": True,
        "manifest": "",
        "full_training_contract": "",
        "full_training_contract_sha256": "",
        "max_open_files": 96,
        "seed": 0,
    }

    for cfg in configs.values():
        assert asdict(cfg.model) == expected_model
        assert asdict(cfg.data) == expected_data
        assert cfg.train.run_class == "full_data_stage"
        assert cfg.train.batch_size == 2
        assert cfg.train.batch_size * 8 * cfg.train.grad_accum == 128
        assert cfg.train.lr == 5e-3
        assert cfg.train.lr_final == 3e-3
        assert cfg.train.warmup_steps == 200
        assert cfg.train.lr_horizon_steps == 500_000
        assert cfg.train.amp is False
        assert cfg.train.w_allatom == 1.0
        assert cfg.train.w_ca == 0.0
        assert cfg.train.w_offset == 0.0
        assert cfg.train.w_unroll == 0.0
        assert cfg.train.w_bond == 0.0
        assert cfg.train.w_bond_unroll == 0.0
        assert cfg.train.w_geom_length_unroll == 0.0
        assert cfg.train.w_geom_angle_unroll == 0.0
        assert cfg.train.resume == ""


def test_expanded_data_stages_differ_only_in_operational_fields():
    configs = _configs()
    baseline = configs["smoke"]

    for cfg in configs.values():
        assert asdict(cfg.data) == asdict(baseline.data)
        assert asdict(cfg.model) == asdict(baseline.model)
        candidate_train = asdict(cfg.train)
        baseline_train = asdict(baseline.train)
        for field in ALLOWED_STAGE_TRAIN_DELTAS:
            candidate_train.pop(field)
            baseline_train.pop(field)
        assert candidate_train == baseline_train

    assert configs["smoke"].train.max_steps == 100
    assert configs["calibration"].train.max_steps == 1_000
    assert configs["development"].train.max_steps == 2_000


def test_expanded_data_stages_share_training_semantics_identity():
    configs = _configs()
    semantics = {
        name: training_semantics_sha256(cfg) for name, cfg in configs.items()
    }

    assert len(set(semantics.values())) == 1
    assert len(next(iter(semantics.values()))) == 64


def test_expanded_data_candidate_excludes_legacy_config_confounds():
    configs = _configs()

    for cfg in configs.values():
        assert cfg.model.vector_channels == cfg.model.hidden == 128
        assert cfg.train.amp is False
        assert cfg.train.lr_horizon_steps == 500_000
        assert cfg.train.lr_horizon_steps != 100_000
        assert cfg.train.warmup_steps == 200
        assert cfg.train.batch_size == 2
        assert cfg.train.grad_accum == 8
