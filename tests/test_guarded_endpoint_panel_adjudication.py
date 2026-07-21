import json
import statistics
from pathlib import Path

import pytest

import scripts.adjudicate_guarded_endpoint_panel as adjudicator
from scripts.guarded_endpoint_panel_eval import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PANEL_SHA256,
    EXPECTED_TRAINING_SHA256,
    SCOPE,
)


TRAINING_LIST = Path("configs/subset_1000_length_proportional.txt")
PANEL_LIST = Path("configs/dev_20_length_proportional_seed0.txt")
TEMPERATURES = [320, 348, 379, 413, 450]
REPLICAS = [0, 1, 2, 3, 4]
TRAIN_FINGERPRINT = "a" * 64


def _result_payload(
    checkpoint: Path,
    *,
    fallback_cells: set[tuple[int, int]] | None = None,
    raw_nonfinite: bool = False,
    mechanism_passes: bool = True,
) -> dict:
    fallback_cells = fallback_cells or set()
    training_ids = TRAINING_LIST.read_text().splitlines()
    panel_ids = PANEL_LIST.read_text().splitlines()
    domains = []
    for domain_index, domain_id in enumerate(panel_ids):
        domain_effect = -0.20 + 0.005 * domain_index
        cells = []
        for cell_index, (temperature, replica) in enumerate(
            (temperature, replica)
            for temperature in TEMPERATURES
            for replica in REPLICAS
        ):
            by_start = []
            for start_index, start_frame in enumerate([0, 50, 100]):
                fallback = (domain_index, cell_index) in fallback_cells and start_index == 0
                noop = 2.0 + 0.01 * cell_index + 0.001 * start_index
                raw_rmsd = noop + domain_effect
                guarded_rmsd = noop if fallback else raw_rmsd
                raw_finite = not (
                    raw_nonfinite and domain_index == cell_index == start_index == 0
                )
                if not raw_finite:
                    fallback = True
                    guarded_rmsd = noop
                raw_physical = raw_finite and not fallback
                by_start.append({
                    "start_index": start_index,
                    "start_frame": start_frame,
                    "target_position_finite": True,
                    "noop_rmsd": noop,
                    "accepted": raw_physical,
                    "fallback": not raw_physical,
                    "selected_position_exact": True,
                    "selected_vector_exact": True,
                    "source": {
                        "position_finite": True,
                        "vector_finite": True,
                        "bond_mean": 3.8,
                        "bond_max": 4.0,
                        "physical": True,
                    },
                    "raw": {
                        "position_finite": raw_finite,
                        "vector_finite": raw_finite,
                        "rmsd": raw_rmsd if raw_finite else None,
                        "minus_noop": domain_effect if raw_finite else None,
                        "bond_mean": 3.8 if raw_finite else None,
                        "bond_max": 6.0 if fallback and raw_finite else (4.0 if raw_finite else None),
                        "physical": raw_physical,
                    },
                    "guarded": {
                        "position_finite": True,
                        "vector_finite": True,
                        "rmsd": guarded_rmsd,
                        "minus_noop": guarded_rmsd - noop,
                        "bond_mean": 3.8,
                        "bond_max": 4.0,
                        "physical": True,
                    },
                })
            deltas = [row["guarded"]["minus_noop"] for row in by_start]
            cells.append({
                "domain": domain_id,
                "temperature": temperature,
                "replica": replica,
                "frames": 102,
                "starts": [0, 50, 100],
                "by_start": by_start,
                "mean_guarded_minus_noop": statistics.fmean(deltas),
                "source_cell_physical": True,
                "raw_cell_physical": all(row["raw"]["physical"] for row in by_start),
                "guarded_cell_physical": True,
                "fallback_starts": sum(row["fallback"] for row in by_start),
            })
        values = [cell["mean_guarded_minus_noop"] for cell in cells]
        domains.append({
            "domain": domain_id,
            "preprocessing": {
                "canon_symmetric": True,
                "residues_total": 80 + domain_index,
                "residues_evaluated": 80 + domain_index,
            },
            "summary": {
                "cells": 25,
                "mean_guarded_minus_noop": statistics.fmean(values),
                "cells_better_than_noop": sum(value < 0 for value in values),
                "fallback_starts": sum(cell["fallback_starts"] for cell in cells),
                "fallback_cells": sum(cell["fallback_starts"] > 0 for cell in cells),
            },
            "cells": cells,
        })
    return {
        "scope": SCOPE,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_step": 2000,
        "checkpoint_schema": 2,
        "checkpoint_train_seed": 0,
        "checkpoint_train_fingerprint": TRAIN_FINGERPRINT,
        "delta_frames": 1,
        "settings": {
            "starts": 3,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
            "policy": "reject_to_exact_source_per_start",
            "strict_thresholds": {
                "bond_mean_gt": 3.2,
                "bond_mean_lt": 4.5,
                "bond_max_lt": 5.5,
            },
            "fallback_caps": {"max_starts": 3, "max_cells": 1},
        },
        "training_subset": {
            "sha256": EXPECTED_TRAINING_SHA256,
            "ids": training_ids,
            "domains_total": 1000,
            "train_domains": 980,
            "validation_domains": 20,
            "train_fingerprint": TRAIN_FINGERPRINT,
        },
        "domain_panel": {
            "sha256": EXPECTED_PANEL_SHA256,
            "ids": panel_ids,
            "subset_of_training1000": True,
            "h5_files": 20,
            "total_bytes": 123,
        },
        "grid": {"temperatures": TEMPERATURES, "replicas": REPLICAS},
        "mechanism_probe": {
            "domain": panel_ids[0],
            "temperature": 320,
            "replica": 0,
            "target_slot": 0,
            "target_start": 0,
            "same_shape_peer_position_bitwise_equal": mechanism_passes,
            "same_shape_peer_vector_bitwise_equal": mechanism_passes,
            "fp32_b1_b3_position_max_abs_diff": 1e-5,
            "fp32_b1_b3_vector_max_abs_diff": 1e-5,
            "fp32_accept_b1": True,
            "fp32_accept_b3": True,
            "fp64_b1_b3_position_max_abs_diff": 1e-14,
            "fp64_b1_b3_vector_max_abs_diff": 1e-14,
            "fp64_accept_b1": True,
            "fp64_accept_b3": True,
        },
        "runtime_probe": {
            "status": "PASS_RUNTIME_PROBE",
            "domain": panel_ids[-1],
            "batch_size": 3,
            "peak_memory_fraction": 0.5,
            "projected_500_cell_minutes": 10.0,
            "limits": {
                "max_peak_memory_fraction": 0.8,
                "max_projected_minutes": 50.0,
            },
        },
        "domains": domains,
    }


def _run(tmp_path: Path, monkeypatch, **kwargs) -> dict:
    checkpoint = tmp_path / "ckpt_2000.pt"
    checkpoint.write_bytes(b"identity is supplied by the frozen hash contract")
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_result_payload(checkpoint, **kwargs)))
    monkeypatch.setattr(
        adjudicator,
        "verify_multidomain_checkpoint",
        lambda *args, **kw: ({"step": 2000}, TRAIN_FINGERPRINT),
    )
    return adjudicator.adjudicate(
        result,
        checkpoint,
        EXPECTED_CHECKPOINT_SHA256,
        TRAINING_LIST,
        EXPECTED_TRAINING_SHA256,
        PANEL_LIST,
        EXPECTED_PANEL_SHA256,
    )


def test_guarded_panel_pass_authorizes_only_new_external_development(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch)
    assert report["status"] == "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"
    assert report["raw_finite_starts"] == 1500
    assert report["source_physical_cells"] == 500
    assert report["guarded_physical_cells"] == 500
    assert report["external_development_authorized"] is True
    assert report["second_seed_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_guarded_panel_allows_one_rare_fallback_without_hiding_it(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch, fallback_cells={(0, 0)})
    assert report["status"] == "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"
    assert report["fallback_starts"] == 1
    assert report["fallback_cells"] == 1


def test_guarded_panel_rejects_fallbacks_spread_across_two_cells(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch, fallback_cells={(0, 0), (0, 1)})
    assert report["status"] == "STOP_CONDITIONAL_SAFEGUARD_FALLBACK_CAP"
    assert report["external_development_authorized"] is False


def test_guarded_panel_rejects_raw_nonfinite_even_after_exact_fallback(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch, raw_nonfinite=True)
    assert report["status"] == "STOP_CONDITIONAL_SAFEGUARD_RAW_NONFINITE"
    assert report["guarded_physical_starts"] == 1500


def test_guarded_panel_rejects_mechanism_failure(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch, mechanism_passes=False)
    assert report["status"] == "STOP_CONDITIONAL_SAFEGUARD_MECHANISM"


def test_guarded_panel_keeps_fp32_batch_acceptance_descriptive(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ckpt_2000.pt"
    checkpoint.write_bytes(b"x")
    payload = _result_payload(checkpoint)
    payload["mechanism_probe"]["fp32_accept_b3"] = False
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload))
    monkeypatch.setattr(
        adjudicator,
        "verify_multidomain_checkpoint",
        lambda *args, **kw: ({"step": 2000}, TRAIN_FINGERPRINT),
    )

    report = adjudicator.adjudicate(
        result,
        checkpoint,
        EXPECTED_CHECKPOINT_SHA256,
        TRAINING_LIST,
        EXPECTED_TRAINING_SHA256,
        PANEL_LIST,
        EXPECTED_PANEL_SHA256,
    )

    assert report["mechanism_passes"] is True
    assert report["status"] == "PASS_CONDITIONAL_SAFEGUARD_TRAINING_DEV20"


@pytest.mark.parametrize("bad_value", [None, "yes"])
def test_guarded_panel_rejects_missing_or_nonboolean_fp64_acceptance(
    tmp_path, monkeypatch, bad_value
):
    checkpoint = tmp_path / "ckpt_2000.pt"
    checkpoint.write_bytes(b"x")
    payload = _result_payload(checkpoint)
    if bad_value is None:
        payload["mechanism_probe"].pop("fp64_accept_b1")
        payload["mechanism_probe"].pop("fp64_accept_b3")
    else:
        payload["mechanism_probe"]["fp64_accept_b1"] = bad_value
        payload["mechanism_probe"]["fp64_accept_b3"] = bad_value
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload))
    monkeypatch.setattr(
        adjudicator,
        "verify_multidomain_checkpoint",
        lambda *args, **kw: ({"step": 2000}, TRAIN_FINGERPRINT),
    )

    report = adjudicator.adjudicate(
        result,
        checkpoint,
        EXPECTED_CHECKPOINT_SHA256,
        TRAINING_LIST,
        EXPECTED_TRAINING_SHA256,
        PANEL_LIST,
        EXPECTED_PANEL_SHA256,
    )

    assert report["mechanism_passes"] is False
    assert report["status"] == "STOP_CONDITIONAL_SAFEGUARD_MECHANISM"


def test_guarded_panel_fails_closed_on_branch_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ckpt_2000.pt"
    checkpoint.write_bytes(b"x")
    payload = _result_payload(checkpoint, fallback_cells={(0, 0)})
    payload["domains"][0]["cells"][0]["by_start"][0]["guarded"]["rmsd"] += 0.1
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload))
    monkeypatch.setattr(
        adjudicator,
        "verify_multidomain_checkpoint",
        lambda *args, **kw: ({"step": 2000}, TRAIN_FINGERPRINT),
    )
    with pytest.raises(ValueError, match="guarded-minus-noop"):
        adjudicator.adjudicate(
            result,
            checkpoint,
            EXPECTED_CHECKPOINT_SHA256,
            TRAINING_LIST,
            EXPECTED_TRAINING_SHA256,
            PANEL_LIST,
            EXPECTED_PANEL_SHA256,
        )


def test_guarded_runner_is_inference_only_and_fail_closed():
    runner = Path("cloud/huawei/run_guarded_training_dev20.sh").read_text()
    assert "SHUTDOWN_ON_EXIT" in runner
    assert "systemd-run" in runner
    assert 'HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-95}' in runner
    assert '[[ "$HARD_STOP_MINUTES" == 95 ]]' in runner
    assert '--on-active="${HARD_STOP_MINUTES}m"' in runner
    assert "scripts/guarded_endpoint_panel_eval.py" in runner
    assert "scripts.adjudicate_guarded_endpoint_panel" in runner
    assert "audit_mdcath_staging.py" in runner
    assert "obsutil sync" in runner
    assert "sha256sum -c" in runner
    assert "train_ddp.py" not in runner
