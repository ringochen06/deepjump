import torch

from scripts.adjudicate_external_endpoint_root_cause import (
    CONTEXT_DOMAINS,
    classify_context,
    classify_outlier,
)
from scripts.external_endpoint_root_cause import _padded_crop_batch, centered_slice


def _shifts(default: float = 0.1) -> dict[str, float]:
    return {domain: default for domain in CONTEXT_DOMAINS}


def test_centered_slice_is_exact_and_deterministic():
    assert centered_slice(335, 128) == slice(103, 231)
    assert centered_slice(128, 64) == slice(32, 96)


def test_masked_padding_preserves_real_crop_and_masks_added_residues():
    batch = {
        "P_t": torch.randn(3, 4, 3),
        "V_t": torch.randn(3, 4, 2, 3),
        "res_index": torch.arange(4).repeat(3, 1),
        "delta_ns": torch.ones(3),
        "residue_mask": torch.ones(3, 4, dtype=torch.bool),
        "atom_mask": torch.ones(3, 4, 2, dtype=torch.bool),
        "bond_mask": torch.ones(3, 3, dtype=torch.bool),
    }
    padded = _padded_crop_batch(batch, 7)
    for key in ("P_t", "V_t", "res_index", "residue_mask", "atom_mask"):
        assert torch.equal(padded[key][:, :4], batch[key])
    assert torch.equal(padded["bond_mask"][:, :3], batch["bond_mask"])
    assert not padded["residue_mask"][:, 4:].any()
    assert not padded["atom_mask"][:, 4:].any()
    assert not padded["bond_mask"][:, 3:].any()


def test_context_classifier_requires_broad_within_crop_improvement():
    shifts = _shifts()
    for domain in ("1jq5A02", "1sqgA01", "2kinA00", "2q0xA01", "3u75A01", "4clcA00"):
        shifts[domain] = -0.1
    assert classify_context(shifts, 0.01, -0.01) == "SUPPORT_DENSE_CONTEXT_LENGTH_MECHANISM"

    crop_only = _shifts()
    crop_only["2q0xA01"] = crop_only["3u75A01"] = -0.1
    assert classify_context(crop_only, 0.01, 0.02) == "SUPPORT_TRAIN_CROP_EXTRAPOLATION_ONLY"


def test_outlier_classifier_fails_closed_before_confirming_model_outlier():
    assert classify_outlier(5.712, 0.0, 0.0, 0.0) == "CONFIRM_MODEL_OUTPUT_OUTLIER"
    assert classify_outlier(5.712, 2e-5, 0.0, 0.0) == "STOP_NONDETERMINISTIC_OUTLIER_REPLAY"
    assert classify_outlier(5.712, 0.0, 2e-5, 0.0) == "STOP_BATCH_CONTEXT_COUPLING"
    assert classify_outlier(5.712, 0.0, 0.0, 2e-5) == "STOP_BOND_PRECISION_MISMATCH"
    assert classify_outlier(5.49, 0.0, 0.0, 0.0) == "STOP_OUTLIER_IDENTITY_MISMATCH"
