import copy
import json
import statistics
from pathlib import Path

import pytest
import torch

import scripts.adjudicate_external_step2000_outlier_replay as adjudicator
from scripts.external_endpoint_identity import _sha256
from scripts.external_step2000_outlier_replay import (
    OUTLIER_DOMAIN,
    OUTLIER_REPLICA,
    OUTLIER_TEMPERATURE,
    SCOPE,
    build_bond_provenance,
)


TRAINING_LIST = Path("configs/subset_1000_length_proportional.txt")
TRAINING_SHA = "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734"
EXTERNAL_LIST = Path("configs/external_dev_20_length_proportional_seed20260721.txt")
EXTERNAL_SHA = "9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245"
TRAIN_FINGERPRINT = "a" * 64
STARTS = [0, 50, 100]


def _checkpoint(path: Path) -> str:
    torch.save({
        "step": 2000,
        "checkpoint_schema": 2,
        "cfg": {
            "data": {
                "domains": [],
                "val_fraction": 0.02,
                "seed": 0,
                "delta_frames": 1,
                "canon_symmetric": True,
            },
            "model": {
                "tensor_cloud01": True,
                "tensor_cloud01_vector_only_attention": False,
            },
            "train": {"seed": 0, "amp": False},
        },
        "train_state": {"world_size": 8, "train_fingerprint": TRAIN_FINGERPRINT},
        "model": {"weight": torch.tensor([1.0])},
    }, path)
    return _sha256(path)


def _cell() -> dict:
    model = [2.0, 2.1, 2.2]
    noop = [1.9, 2.0, 2.1]
    paired = [left - right for left, right in zip(model, noop)]
    return {
        "domain": OUTLIER_DOMAIN,
        "temperature": OUTLIER_TEMPERATURE,
        "replica": OUTLIER_REPLICA,
        "frames": 102,
        "starts": STARTS,
        "model_rmsd_by_start": model,
        "noop_rmsd_by_start": noop,
        "model_minus_noop_by_start": paired,
        "mean_model_minus_noop": statistics.fmean(paired),
        "bond_mean": 4.9,
        "bond_max": 6.0,
    }


def _max_record() -> dict:
    return {
        "bond_index": 2,
        "res_index_pair": [15, 16],
        "source_positions": [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]],
        "target_positions": [[0.0, 0.0, 0.0], [3.9, 0.0, 0.0]],
        "predicted_positions": [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        "source_length": 3.8,
        "target_length": 3.9,
        "predicted_length_fp32": 6.0,
        "predicted_length_fp64": 6.0,
    }


def _payload(checkpoint: Path, digest: str, reference: Path, reference_sha: str) -> dict:
    external_ids = EXTERNAL_LIST.read_text().splitlines()
    cell = _cell()
    topology = {
        "residues": 4,
        "res_index": [10, 11, 15, 16],
        "res_index_sha256": adjudicator._json_sha256([10, 11, 15, 16]),
        "bond_mask": [True, False, True],
        "bond_mask_sha256": adjudicator._json_sha256([True, False, True]),
        "valid_bond_count": 2,
        "valid_bonds": [
            {"bond_index": 0, "res_index_pair": [10, 11], "consecutive_res_index": True},
            {"bond_index": 2, "res_index_pair": [15, 16], "consecutive_res_index": True},
        ],
    }
    per_start = []
    for index, start in enumerate(STARTS):
        per_start.append({
            "start_index": index,
            "start_frame": start,
            "source_positions_sha256": "1" * 64,
            "target_positions_sha256": "2" * 64,
            "predicted_positions_sha256": "3" * 64,
            "valid_bond_lengths": [
                {"bond_index": 0, "source_length": 3.8, "target_length": 3.9, "predicted_length": 3.8},
                {"bond_index": 2, "source_length": 3.8, "target_length": 3.9, "predicted_length": 6.0},
            ],
            "max_predicted_bond": _max_record(),
        })
    paired = cell["model_minus_noop_by_start"]
    return {
        "scope": SCOPE,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_step": 2000,
        "checkpoint_train_seed": 0,
        "checkpoint_train_fingerprint": TRAIN_FINGERPRINT,
        "training_domain_list_sha256": TRAINING_SHA,
        "external_panel": {"sha256": EXTERNAL_SHA, "ids": external_ids},
        "reference_panel": {"path": str(reference), "sha256": reference_sha},
        "cell": {
            "domain": OUTLIER_DOMAIN,
            "temperature": OUTLIER_TEMPERATURE,
            "replica": OUTLIER_REPLICA,
            "frames": 102,
            "starts": STARTS,
        },
        "settings": {"delta_frames": 1, "steps": 1, "method": "mean", "source_noise": False},
        "replay": {
            "model_rmsd_by_start": cell["model_rmsd_by_start"],
            "noop_rmsd_by_start": cell["noop_rmsd_by_start"],
            "model_minus_noop_by_start": paired,
            "mean_model_minus_noop": statistics.fmean(paired),
            "bond_mean": 4.9,
            "bond_max": 6.0,
            "repeat_max_abs_prediction_difference": 0.0,
            "batched_vs_individual_max_abs_prediction_difference": 0.0,
            "topology": topology,
            "per_start": per_start,
        },
        "frozen_panel_cell": cell,
        "formal_training_authorized": False,
    }


def _case(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str, Path, str]:
    checkpoint = tmp_path / "ckpt_2000.pt"
    digest = _checkpoint(checkpoint)
    reference = tmp_path / "panel.json"
    reference.write_text(json.dumps({
        "checkpoint_sha256": digest,
        "domains": [{"domain": OUTLIER_DOMAIN, "cells": [_cell()]}],
    }))
    reference_sha = _sha256(reference)
    monkeypatch.setattr(adjudicator, "EXPECTED_CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(adjudicator, "EXPECTED_REFERENCE_PANEL_SHA256", reference_sha)
    result = tmp_path / "replay.json"
    result.write_text(json.dumps(_payload(checkpoint, digest, reference, reference_sha)))
    return result, checkpoint, digest, reference, reference_sha


def _adjudicate(case):
    result, checkpoint, digest, reference, reference_sha = case
    return adjudicator.adjudicate(
        result,
        checkpoint,
        digest,
        TRAINING_LIST,
        TRAINING_SHA,
        EXTERNAL_LIST,
        EXTERNAL_SHA,
        reference,
        reference_sha,
    )


def test_bond_provenance_records_only_masked_consecutive_bonds():
    source = torch.tensor([[[0.0, 0, 0], [3.8, 0, 0], [8.0, 0, 0], [11.8, 0, 0]]])
    target = source.clone()
    prediction = source.clone()
    prediction[0, 3, 0] = 14.0
    mask = torch.tensor([[True, False, True]])
    res_index = torch.tensor([10, 11, 15, 16])

    topology, records = build_bond_provenance(
        prediction, source, target, mask, res_index, [7]
    )

    assert [item["bond_index"] for item in topology["valid_bonds"]] == [0, 2]
    assert records[0]["start_frame"] == 7
    assert records[0]["max_predicted_bond"]["bond_index"] == 2
    assert records[0]["max_predicted_bond"]["predicted_length_fp64"] == pytest.approx(6.0)


def test_step2000_replay_confirms_real_model_output_outlier(tmp_path, monkeypatch):
    decision = _adjudicate(_case(tmp_path, monkeypatch))
    assert decision["status"] == "CONFIRM_STEP2000_MODEL_OUTPUT_OUTLIER"
    assert decision["formal_training_authorized"] is False
    assert decision["source_bond_max"] == pytest.approx(3.8)
    assert decision["target_bond_max"] == pytest.approx(3.9)


def test_step2000_replay_fails_closed_on_topology_corruption(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["replay"]["topology"]["valid_bonds"][1]["res_index_pair"] = [15, 99]
    case[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="res_index pair"):
        _adjudicate(case)


def test_step2000_replay_classifies_nondeterminism_and_source_anomaly(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["replay"]["repeat_max_abs_prediction_difference"] = 2e-5
    case[0].write_text(json.dumps(payload))
    assert _adjudicate(case)["status"] == "STOP_NONDETERMINISTIC_STEP2000_OUTLIER_REPLAY"

    payload["replay"]["repeat_max_abs_prediction_difference"] = 0.0
    payload["replay"]["per_start"][0]["valid_bond_lengths"][0]["source_length"] = 6.1
    case[0].write_text(json.dumps(payload))
    assert _adjudicate(case)["status"] == "STOP_SOURCE_TARGET_GEOMETRY_ANOMALY"


def test_step2000_replay_rejects_frozen_panel_mismatch(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    payload = json.loads(case[0].read_text())
    payload["replay"]["bond_max"] = 5.9
    case[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="panel bond max"):
        _adjudicate(case)


def test_step2000_replay_runner_is_bounded_readback_verified_and_training_free():
    runner = Path("cloud/huawei/run_external_step2000_outlier_replay.sh").read_text()
    assert "HARD_STOP_MINUTES=${HARD_STOP_MINUTES:-45}" in runner
    assert "timeout --signal=TERM --kill-after=30s 15m" in runner
    assert "external_step2000_outlier_replay.py" in runner
    assert "scripts.adjudicate_external_step2000_outlier_replay" in runner
    assert "obsutil sync" in runner
    assert "sha256sum -c" in runner
    assert "readback_completion.json" in runner
    assert "shutdown -h now" in runner
    assert '"$PYTHON" scripts/train_ddp.py' not in runner
    assert "untouched" not in runner.lower()
