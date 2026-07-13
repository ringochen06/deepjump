"""DDP gradient-synchronization regression test (CPU / gloo — no GPU needed).

Spawns `world` processes, feeds each rank DIFFERENT data, and checks that:
  * a DDP-wrapped DeepJumpLite, called through the wrapper, all-reduces gradients so they
    are IDENTICAL across ranks;
  * an UNWRAPPED model produces DIFFERENT gradients across ranks (no sync).
The first assertion is exactly what breaks if the trainer forwards through `model.module`
(a.k.a. `core`) instead of the DDP-wrapped `model`. Run: `python tests/test_ddp_sync.py`
or via pytest (which shells out so it doesn't nest mp.spawn).
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def _toy_batch(seed, N=6, B=2):
    g = torch.Generator().manual_seed(seed)
    return {
        "P_t": torch.randn(B, N, 3, generator=g),
        "V_t": torch.randn(B, N, 13, 3, generator=g) * 0.3,
        "P_1": torch.randn(B, N, 3, generator=g),
        "V_1": torch.randn(B, N, 13, 3, generator=g) * 0.3,
        "res_index": torch.randint(0, 20, (B, N), generator=g),
        "atom_mask": torch.ones(B, N, 13, dtype=torch.bool),
        "residue_mask": torch.ones(B, N, dtype=torch.bool),
        "delta_ns": torch.ones(B),
    }


def _loss(out, batch):
    # mirror real training: P pairwise + heavy-atom offset, so V_hat_1 (head_v) is USED
    # -> no unused parameters -> DDP reducer covers every param.
    from deepjump.losses import heavy_atom_offset_loss, pairwise_vector_huber_loss
    return (pairwise_vector_huber_loss(out["P_hat_1"], batch["P_1"], batch["residue_mask"])
            + heavy_atom_offset_loss(out["V_hat_1"], batch["V_1"], batch["atom_mask"]))


def _all_same(g, world):
    g = g.contiguous()
    gathered = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(gathered, g)
    return all(torch.allclose(gathered[0], x, atol=1e-5) for x in gathered)


def _frac_synced(model, world):
    """Fraction of parameters whose grad is identical across ranks; plus first mismatch."""
    total = same = 0
    first_bad = None
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        total += 1
        if _all_same(p.grad.detach(), world):
            same += 1
        elif first_bad is None:
            first_bad = name
    return same, total, first_bad


def _worker(rank, world, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    from deepjump.config import ModelConfig
    from deepjump.model import DeepJumpLite

    cfg = ModelConfig(hidden=16, vector_channels=8, num_heads=2, cond_layers=1, transport_layers=1)
    batch = _toy_batch(seed=100 + rank)  # DIFFERENT data per rank
    tau = torch.zeros(batch["P_t"].shape[0])

    # (1) DDP-wrapped, called through the wrapper -> grads must be synced (identical)
    torch.manual_seed(0)  # identical init across ranks
    m = DeepJumpLite(cfg, predict_heavy=True); m.noise_sigma = 0.0
    ddp = DDP(m)  # CPU/gloo: no device_ids
    out = ddp(batch, tau=tau)
    _loss(out, batch).backward()
    same, total, first_bad = _frac_synced(m, world)
    synced = same == total

    # (2) unwrapped model -> grads differ across ranks (no sync); proves the test has signal
    torch.manual_seed(0)
    m2 = DeepJumpLite(cfg, predict_heavy=True); m2.noise_sigma = 0.0
    _loss(m2(batch, tau=tau), batch).backward()
    same2, total2, _ = _frac_synced(m2, world)
    unwrapped_differs = same2 < total2

    if rank == 0:
        ret["synced"] = bool(synced)
        ret["synced_frac"] = f"{same}/{total}"
        ret["first_unsynced"] = first_bad
        ret["unwrapped_differs"] = bool(unwrapped_differs)
    dist.destroy_process_group()


def run(world=2):
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(world, ret), nprocs=world, join=True)
    return dict(ret)


def test_ddp_grad_sync():
    # shell out so pytest itself does not nest mp.spawn (avoids re-import issues)
    r = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr


if __name__ == "__main__":
    res = run(2)
    print("ddp sync result:", res)
    ok = res.get("synced") and res.get("unwrapped_differs")
    sys.exit(0 if ok else 1)
