from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
DEVELOPMENT = ROOT / "configs/v100_tensorcloud01_full_expanded_d1_development2000.yaml"
FORMAL = ROOT / "configs/v100_tensorcloud01_full_expanded_formal500k.yaml"


def test_formal500k_changes_only_class_endpoint_and_operational_cadence():
    development = yaml.safe_load(DEVELOPMENT.read_text())
    formal = yaml.safe_load(FORMAL.read_text())

    assert formal["train"]["run_class"] == "formal"
    assert formal["train"]["max_steps"] == 500_000
    assert formal["train"]["lr_horizon_steps"] == 500_000
    assert formal["train"]["batch_size"] * formal["train"]["grad_accum"] * 8 == 128
    assert formal["data"]["crop_length"] == 256
    assert formal["data"]["seed"] == formal["train"]["seed"] == 0
    assert formal["train"]["amp"] is False
    assert formal["model"]["hidden"] == 128
    assert formal["model"]["cond_layers"] == 6
    assert formal["model"]["transport_layers"] == 6

    normalized = deepcopy(formal)
    for field in (
        "run_class",
        "max_steps",
        "val_every",
        "log_every",
        "ckpt_every",
        "keep_last_k",
        "out_dir",
    ):
        normalized["train"][field] = development["train"][field]
    assert normalized == development


def test_formal500k_trainer_retention_cannot_preempt_remote_verification():
    config = yaml.safe_load(FORMAL.read_text())
    train = config["train"]
    total_numbered = train["max_steps"] // train["ckpt_every"]
    assert train["max_steps"] % train["ckpt_every"] == 0
    assert train["keep_last_k"] > total_numbered
    assert train["ckpt_every"] == 1_000
    assert train["val_every"] == 10_000
