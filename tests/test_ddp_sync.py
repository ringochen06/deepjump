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
        "P_2": torch.randn(B, N, 3, generator=g),
        "V_2": torch.randn(B, N, 13, 3, generator=g) * 0.3,
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

    # (3) honest tau=0 self-conditioning performs extra DDP forwards before one
    # backward; gradients must still synchronize across ranks.
    from deepjump.config import Config
    from deepjump.training import total_loss

    torch.manual_seed(0)
    m3 = DeepJumpLite(cfg, predict_heavy=True); m3.noise_sigma = 0.0
    ddp3 = DDP(m3)
    primary = ddp3(batch, tau=tau)
    loss3_cfg = Config()
    loss3_cfg.train.w_ca = 1.0
    loss3_cfg.train.w_offset = 1.0
    loss3_cfg.train.w_unroll = 0.5
    loss3, _ = total_loss(primary, batch, loss3_cfg, ddp3)
    loss3.backward()
    same3, total3, first_bad3 = _frac_synced(m3, world)
    unroll_synced = same3 == total3

    # (4) Gated vector q/k adds parameters inside every attention block; all
    # gradients, including an initially zero gate/projection path, must reduce.
    vector_cfg = ModelConfig(
        hidden=16, vector_channels=8, num_heads=2, cond_layers=1,
        transport_layers=1, vector_qk=True,
    )
    torch.manual_seed(0)
    m4 = DeepJumpLite(vector_cfg, predict_heavy=True); m4.noise_sigma = 0.0
    ddp4 = DDP(m4)
    _loss(ddp4(batch, tau=tau), batch).backward()
    same4, total4, first_bad4 = _frac_synced(m4, world)
    vector_qk_synced = same4 == total4

    paper_cfg = ModelConfig(
        hidden=16, vector_channels=8, num_heads=2, cond_layers=1,
        transport_layers=1, vector_qk=True, paper_ff=True,
    )
    torch.manual_seed(0)
    m5 = DeepJumpLite(paper_cfg, predict_heavy=True); m5.noise_sigma = 0.0
    ddp5 = DDP(m5, find_unused_parameters=True)
    _loss(ddp5(batch, tau=tau), batch).backward()
    same5, total5, first_bad5 = _frac_synced(m5, world)
    paper_ff_synced = same5 == total5

    # (6) Algorithm-1 Tensor-Cloud q/k uses the vector projections without a
    # legacy gate; their gradients must also reduce across ranks.
    tensor_cfg = ModelConfig(
        hidden=16, vector_channels=8, num_heads=2, cond_layers=1,
        transport_layers=1, tensor_qkv=True,
    )
    torch.manual_seed(0)
    m6 = DeepJumpLite(tensor_cfg, predict_heavy=True); m6.noise_sigma = 0.0
    ddp6 = DDP(m6)
    _loss(ddp6(batch, tau=tau), batch).backward()
    same6, total6, first_bad6 = _frac_synced(m6, world)
    tensor_qkv_synced = same6 == total6

    # (7) The dedicated equal-multiplicity TensorCloud01 path has a structurally
    # unused final scalar FF projection because the transport head is vector-only.
    # DDP must traverse the graph and synchronize every parameter that is used.
    tensor01_cfg = ModelConfig(
        hidden=16, vector_channels=16, num_heads=2, cond_layers=1,
        transport_layers=1, tensor_cloud01=True,
    )
    torch.manual_seed(0)
    m7 = DeepJumpLite(tensor01_cfg, predict_heavy=True); m7.noise_sigma = 0.0
    ddp7 = DDP(m7, find_unused_parameters=True)
    _loss(ddp7(batch, tau=tau), batch).backward()
    same7, total7, first_bad7 = _frac_synced(m7, world)
    tensor_cloud01_synced = same7 == total7

    if rank == 0:
        ret["synced"] = bool(synced)
        ret["synced_frac"] = f"{same}/{total}"
        ret["first_unsynced"] = first_bad
        ret["unwrapped_differs"] = bool(unwrapped_differs)
        ret["unroll_synced"] = bool(unroll_synced)
        ret["unroll_synced_frac"] = f"{same3}/{total3}"
        ret["unroll_first_unsynced"] = first_bad3
        ret["vector_qk_synced"] = bool(vector_qk_synced)
        ret["vector_qk_synced_frac"] = f"{same4}/{total4}"
        ret["vector_qk_first_unsynced"] = first_bad4
        ret["paper_ff_synced"] = bool(paper_ff_synced)
        ret["paper_ff_synced_frac"] = f"{same5}/{total5}"
        ret["paper_ff_first_unsynced"] = first_bad5
        ret["tensor_qkv_synced"] = bool(tensor_qkv_synced)
        ret["tensor_qkv_synced_frac"] = f"{same6}/{total6}"
        ret["tensor_qkv_first_unsynced"] = first_bad6
        ret["tensor_cloud01_synced"] = bool(tensor_cloud01_synced)
        ret["tensor_cloud01_synced_frac"] = f"{same7}/{total7}"
        ret["tensor_cloud01_first_unsynced"] = first_bad7
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
    ok = (res.get("synced") and res.get("unwrapped_differs")
          and res.get("unroll_synced") and res.get("vector_qk_synced"))
    ok = ok and res.get("paper_ff_synced") and res.get("tensor_qkv_synced")
    ok = ok and res.get("tensor_cloud01_synced")
    sys.exit(0 if ok else 1)
