import json
import statistics
from pathlib import Path

import pytest
import torch

from scripts.adjudicate_external_endpoint_panel import adjudicate
from deepjump.utils import split_domains
from scripts.external_endpoint_identity import (
    _sha256,
    load_disjoint_panels,
    verify_training_fingerprint,
)
from scripts.train_ddp import dataset_fingerprint


TRAINING_LIST = Path("configs/subset_1000_length_proportional.txt")
TRAINING_SHA = "39278d6dc3de52065b19dffb2438eae53fca3730572bba30496c1b116d597734"
EXTERNAL_LIST = Path("configs/external_dev_20_length_proportional_seed20260721.txt")
EXTERNAL_SHA = "9fb229049aec41ac9b376b447938930e434c94b7e106dfe5dc1ae1ac8cdaf245"
TRAIN_FINGERPRINT = "a" * 64
TEMPERATURES = [320, 348, 379, 413, 450]
REPLICAS = [0, 1, 2, 3, 4]


def _checkpoint(path: Path) -> str:
    torch.save({
        "step": 1000,
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
        "train_state": {
            "world_size": 8,
            "train_fingerprint": TRAIN_FINGERPRINT,
        },
        "model": {"weight": torch.tensor([1.0])},
    }, path)
    return _sha256(path)


def _write_result(
    path: Path,
    checkpoint: Path,
    domain_deltas: list[float],
    *,
    nonphysical: bool = False,
) -> None:
    training_ids = TRAINING_LIST.read_text().splitlines()
    external_ids = EXTERNAL_LIST.read_text().splitlines()
    domains = []
    for domain_index, (domain_id, domain_delta) in enumerate(
        zip(external_ids, domain_deltas)
    ):
        cells = []
        for cell_index, (temperature, replica) in enumerate(
            (temperature, replica)
            for temperature in TEMPERATURES
            for replica in REPLICAS
        ):
            delta = domain_delta + 0.0001 * (cell_index - 12)
            noop = [2.0 + 0.01 * cell_index + 0.001 * start for start in range(3)]
            model = [value + delta for value in noop]
            cells.append({
                "domain": domain_id,
                "temperature": temperature,
                "replica": replica,
                "frames": 102,
                "starts": [0, 50, 100],
                "model_rmsd_by_start": model,
                "noop_rmsd_by_start": noop,
                "model_minus_noop_by_start": [delta] * 3,
                "mean_model_minus_noop": delta,
                "bond_mean": 3.8,
                "bond_max": 6.0 if nonphysical and domain_index == cell_index == 0 else 4.1,
            })
        cell_deltas = [cell["mean_model_minus_noop"] for cell in cells]
        domains.append({
            "domain": domain_id,
            "preprocessing": {
                "canon_symmetric": True,
                "residues_total": 80 + domain_index,
                "residues_evaluated": 80 + domain_index,
            },
            "summary": {
                "cells": 25,
                "mean_model_minus_noop": statistics.fmean(cell_deltas),
                "cells_better_than_noop": sum(value < 0 for value in cell_deltas),
            },
            "cells": cells,
        })
    payload = {
        "scope": "external_multidomain_fp32_pilot_h1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": 1000,
        "checkpoint_schema": 2,
        "checkpoint_train_fingerprint": TRAIN_FINGERPRINT,
        "delta_frames": 1,
        "settings": {
            "starts": 3,
            "start_strategy": "valid_source_linspace",
            "method": "mean",
            "source_noise": False,
        },
        "training_subset": {
            "sha256": TRAINING_SHA,
            "ids": training_ids,
            "domains_total": 1000,
            "train_domains": 980,
            "validation_domains": 20,
            "train_fingerprint": TRAIN_FINGERPRINT,
        },
        "domain_panel": {
            "sha256": EXTERNAL_SHA,
            "ids": external_ids,
            "h5_files": 20,
            "total_bytes": 13_778_143_616,
        },
        "grid": {"temperatures": TEMPERATURES, "replicas": REPLICAS},
        "runtime_probe": {
            "status": "PASS_RUNTIME_PROBE",
            "domain": external_ids[-1],
            "residues": 300,
            "batch_size": 3,
            "cell_seconds": 1.0,
            "projected_500_cell_minutes": 8.333333333333334,
            "peak_memory_bytes": 8_000_000_000,
            "total_memory_bytes": 16_000_000_000,
            "peak_memory_fraction": 0.5,
            "limits": {
                "max_peak_memory_fraction": 0.8,
                "max_projected_minutes": 50.0,
            },
        },
        "domains": domains,
    }
    path.write_text(json.dumps(payload))


def _adjudicate(tmp_path: Path, effects: list[float], **kwargs) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, effects, **kwargs)
    return adjudicate(
        result,
        checkpoint,
        digest,
        TRAINING_LIST,
        TRAINING_SHA,
        EXTERNAL_LIST,
        EXTERNAL_SHA,
    )


def test_external_panel_pass_authorizes_only_second_seed(tmp_path):
    report = _adjudicate(
        tmp_path, [-0.20 + 0.005 * index for index in range(20)]
    )

    assert report["status"] == "PASS_EXTERNAL_DEV20_ENDPOINT"
    assert report["domains"] == 20
    assert report["cells"] == 500
    assert report["starts"] == 1500
    assert report["primary"]["ci95_model_minus_noop"][1] < 0
    assert all(item["passes_negative"] for item in report["leave_one_domain_out"])
    assert report["second_seed_authorized"] is True
    assert report["untouched_confirmation_authorized"] is False
    assert report["formal_training_authorized"] is False


def test_external_panel_zero_variance_is_inconclusive(tmp_path):
    report = _adjudicate(tmp_path, [-0.1] * 20)
    assert report["status"] == "INCONCLUSIVE_ZERO_VARIANCE_EXTERNAL_DEV20_ENDPOINT"
    assert report["second_seed_authorized"] is False


def test_external_panel_stops_disadvantage_or_nonphysical(tmp_path):
    disadvantage = _adjudicate(
        tmp_path / "disadvantage", [0.10 + 0.005 * index for index in range(20)]
    )
    assert disadvantage["status"] == "STOP_DOMAIN_DISADVANTAGE_EXTERNAL_DEV20_ENDPOINT"
    nonphysical = _adjudicate(
        tmp_path / "nonphysical",
        [-0.20 + 0.005 * index for index in range(20)],
        nonphysical=True,
    )
    assert nonphysical["status"] == "STOP_NONPHYSICAL_EXTERNAL_DEV20_ENDPOINT"
    assert nonphysical["physical_cells"] == 499


def test_external_panel_rejects_overlap(tmp_path):
    overlapping = tmp_path / "external.txt"
    ids = EXTERNAL_LIST.read_text().splitlines()
    ids[0] = TRAINING_LIST.read_text().splitlines()[0]
    overlapping.write_text("\n".join(ids) + "\n")
    with pytest.raises(ValueError, match="overlaps"):
        load_disjoint_panels(
            TRAINING_LIST,
            TRAINING_SHA,
            overlapping,
            _sha256(overlapping),
        )


def test_training_fingerprint_reconstructs_the_manifest_order(tmp_path):
    root = tmp_path / "mdcath"
    data = root / "data"
    data.mkdir(parents=True)
    entries = []
    files = []
    for index, domain_id in enumerate(TRAINING_LIST.read_text().splitlines()):
        path = data / f"mdcath_dataset_{domain_id}.h5"
        path.write_bytes(bytes([index % 251]) * (index % 7 + 1))
        files.append(path)
        entries.append({"file": path.name})
    (root / "manifest.json").write_text(json.dumps(entries))
    train_files, _ = split_domains(files, 0.02, 0)
    fingerprint = dataset_fingerprint(train_files)
    checkpoint = {"cfg": {"data": {"val_fraction": 0.02, "seed": 0}}}

    identity = verify_training_fingerprint(
        checkpoint,
        fingerprint,
        root,
        TRAINING_LIST.read_text().splitlines(),
    )

    assert identity["domains_total"] == 1000
    assert identity["train_domains"] == 980
    assert identity["validation_domains"] == 20
    assert identity["train_fingerprint"] == fingerprint


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload["domain_panel"]["ids"].reverse(), "identity or order"),
        (lambda payload: payload["training_subset"].update(train_domains=979), "split count"),
        (lambda payload: payload["domain_panel"].update(total_bytes=1), "byte count"),
        (
            lambda payload: payload["domains"][0]["cells"][0].update(starts=[0, 1, 2]),
            "start panel",
        ),
    ],
)
def test_external_panel_rejects_identity_or_evidence_corruption(
    tmp_path, mutation, match
):
    checkpoint = tmp_path / "ckpt_1000.pt"
    digest = _checkpoint(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-0.2 + 0.005 * index for index in range(20)])
    payload = json.loads(result.read_text())
    mutation(payload)
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        adjudicate(
            result,
            checkpoint,
            digest,
            TRAINING_LIST,
            TRAINING_SHA,
            EXTERNAL_LIST,
            EXTERNAL_SHA,
        )


def test_external_panel_rejects_nonfinite_checkpoint(tmp_path):
    checkpoint = tmp_path / "ckpt_1000.pt"
    _checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["weight"] = torch.tensor([float("nan")])
    torch.save(payload, checkpoint)
    digest = _sha256(checkpoint)
    result = tmp_path / "result.json"
    _write_result(result, checkpoint, [-0.2 + 0.005 * index for index in range(20)])
    with pytest.raises(ValueError, match="non-finite"):
        adjudicate(
            result,
            checkpoint,
            digest,
            TRAINING_LIST,
            TRAINING_SHA,
            EXTERNAL_LIST,
            EXTERNAL_SHA,
        )
