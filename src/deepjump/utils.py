"""Small training helpers: device resolution, domain split, batch moving."""

from __future__ import annotations

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
