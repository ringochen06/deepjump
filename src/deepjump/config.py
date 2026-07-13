"""Dataclass config with minimal YAML loading (no Hydra)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    root: str = "~/hkucds/data/mdcath"  # where downloaded *.h5 live
    domains: list[str] = field(default_factory=list)  # empty => use all found under root
    temperatures: list[int] = field(default_factory=lambda: [320])
    replicas: list[int] = field(default_factory=lambda: [0])
    delta_frames: object = 1  # int, or list e.g. [1,10,100] for multi-scale delta training
    crop_length: int = 128
    val_fraction: float = 0.2  # fraction of domains held out for validation
    noise_sigma: float = 0.1  # sigma of gaussian added to X_t at tau=0 (Angstrom)
    unroll: int = 1  # number of future steps per sample (2 => self-conditioning training)
    canon_symmetric: bool = False  # canonicalise symmetric sidechain atom labelling
    manifest: str = ""  # path to manifest.json (build_manifest.py); "" => scan files at init
    max_open_files: int = 64  # per-worker LRU cap on open h5 handles (ulimit-safe)
    seed: int = 0


@dataclass
class ModelConfig:
    hidden: int = 32  # scalar channel width H (=~260k params at H=32)
    vector_channels: int = 16  # vector feature channels
    num_heads: int = 4
    cond_layers: int = 6
    transport_layers: int = 6
    seq_embed_ks: int = 32  # sequence-distance embedding half-window
    num_dist_basis: int = 16  # gaussian spatial distance basis
    dist_cutoff: float = 25.0  # Angstrom, used for gaussian basis range
    predict_heavy: bool = False  # also predict heavy-atom offsets V_hat_1
    input_aug_sigma: float = 0.0  # train-time noise on conditioner input X_t (rollout robustness)


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 1e-3
    grad_clip: float = 0.1
    max_steps: int = 500
    val_every: int = 50
    log_every: int = 10
    huber_delta: float = 1.0
    w_offset: float = 0.0  # weight on heavy-atom offset loss (0 => Ca-only)
    w_allatom: float = 0.0  # weight on 25A all-atom pairwise Huber loss
    w_unroll: float = 0.0  # weight on self-conditioned unroll step losses
    device: str = "auto"  # auto -> mps if available else cpu
    out_dir: str = "runs/ca_delta1"
    seed: int = 0
    # ---- distributed / scale (train_ddp.py) --------------------------------
    num_workers: int = 0  # dataloader workers per process (cloud: 8-16)
    grad_accum: int = 1  # gradient accumulation steps (effective_batch = batch*world*accum)
    amp: bool = False  # mixed precision (bf16/fp16 autocast) -- enable on A100/V100
    amp_dtype: str = "bf16"  # bf16 (A100, no scaler) or fp16 (V100, needs GradScaler)
    lr_final: float = 0.0  # if >0, linearly decay lr -> lr_final over max_steps (paper: 5e-3->3e-3)
    warmup_steps: int = 0  # linear LR warmup
    ckpt_every: int = 5000  # steps between full (model+opt+sched) checkpoints
    keep_last_k: int = 3  # rolling checkpoints to keep
    resume: str = ""  # path to a checkpoint to resume optimizer/scheduler/step from


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not is_dataclass(cls):
        return d
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)  # resolves string annotations to real types
    for key, val in d.items():
        if key not in type_hints:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = type_hints[key]
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[key] = _from_dict(ftype, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(Path(path).expanduser()) as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
