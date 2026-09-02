"""Small training helpers: device resolution, domain split, batch moving."""

from __future__ import annotations

import dataclasses
import os

import torch


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        # let unsupported ops fall back to CPU instead of erroring
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_domains(files, val_fraction: float, seed: int = 0):
    """Split domain files into (train, val). Guarantees >=1 val file when possible."""
    import numpy as np

    files = list(files)
    if len(files) == 1:
        # Tiny-data diagnostic: train and validation intentionally reuse the same
        # trajectories. This is in-sample by construction, not a held-out split.
        return files, files
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train = [f for i, f in enumerate(files) if i not in val_idx]
    val = [f for i, f in enumerate(files) if i in val_idx]
    return train, val


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


def model_config_kwargs(checkpoint_model_cfg, config_class):
    """Filter a checkpoint's model config down to fields the current class declares.

    Checkpoints outlive the code that wrote them, so a stored config can name
    options that have since been removed. Dropping any of them silently would
    change the architecture under the caller, so only *falsy* unknowns are
    dropped and anything else raises: a removed option that was actually on is a
    real incompatibility, not a compatibility shim.

    Callers should still load the state dict with ``strict=True`` so real drift
    surfaces as a load error rather than a quietly different model.
    """
    known = {field.name for field in dataclasses.fields(config_class)}
    for key, value in checkpoint_model_cfg.items():
        if key not in known and value:
            raise ValueError(f"checkpoint sets unknown model option {key}={value!r}")
    return {k: v for k, v in checkpoint_model_cfg.items() if k in known}
