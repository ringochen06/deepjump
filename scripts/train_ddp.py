#!/usr/bin/env python
"""Multi-GPU (DDP) trainer for near-paper-scale DeepJump reproduction on NVIDIA GPUs.

Launch with torchrun (single node, N GPUs):
    torchrun --nproc_per_node=8 scripts/train_ddp.py --config configs/paper_h128_d1.yaml
Multi-node: set --nnodes/--node_rank/--master_addr per torchrun docs.

Features: NCCL DDP, DistributedSampler, AMP (bf16/fp16), gradient accumulation to a
target effective batch, warmup + linear LR decay (paper 5e-3->3e-3), rank-0 validation
/ logging / checkpointing, and resume (model+optimizer+step). Data is streamed via the
scale-safe MdcathPairDataset (manifest + lazy LRU handles), so 5398 domains x 5 temps x
5 replicas fits in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from deepjump.config import load_config, to_dict
from deepjump.data import (
    MdcathPairDataset,
    ResumableDistributedSampler,
    collate_pairs,
    discover_domains,
)
from deepjump.metrics import masked_ca_rmsd
from deepjump.model import DeepJumpLite
from deepjump.model.deepjump import count_parameters
from deepjump.training import lr_at, total_loss
from deepjump.utils import move_batch, split_domains


# ---- distributed helpers ----------------------------------------------------
def ddp_setup():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
    return rank, world, local


def is_main(rank):
    return rank == 0


def load_manifest(cfg):
    if cfg.data.manifest:
        return json.loads(Path(cfg.data.manifest).expanduser().read_text())
    return None


def build_datasets(cfg):
    """Split domains into train/val (by file, so val domains are unseen)."""
    manifest = load_manifest(cfg)
    if manifest is not None:
        files = [Path(cfg.data.root).expanduser() / "data" / e["file"] for e in manifest]
        # fall back to any layout: resolve names against root recursively if needed
        if not files or not files[0].exists():
            found = {f.name: f for f in discover_domains(cfg.data.root)}
            files = [found[e["file"]] for e in manifest if e["file"] in found]
    else:
        files = discover_domains(cfg.data.root)
    if cfg.data.domains:
        wanted = set(cfg.data.domains)
        files = [f for f in files if f.stem.replace("mdcath_dataset_", "") in wanted]
    if not files:
        raise SystemExit(f"no mdCATH files under {cfg.data.root}")
    train_files, val_files = split_domains(files, cfg.data.val_fraction, cfg.data.seed)

    def make(fs, seed):
        return MdcathPairDataset(
            fs, cfg.data.temperatures, cfg.data.replicas, cfg.data.delta_frames,
            cfg.data.crop_length, align=True, unroll=cfg.data.unroll,
            canon_symmetric=cfg.data.canon_symmetric, manifest=manifest,
            max_open_files=cfg.data.max_open_files, seed=seed,
        )
    # Data split remains tied to data.seed, while crop/order randomness follows
    # train.seed so independent training seeds are genuinely distinct without
    # changing the held-out domains or validation panel.
    return make(train_files, cfg.train.seed), make(val_files, cfg.data.seed + 1), train_files, val_files


def dataset_fingerprint(files) -> str:
    """Fingerprint the ordered domain identity and size without reading HDF5 payloads."""
    digest = hashlib.sha256()
    for path in files:
        path = Path(path)
        digest.update(f"{path.name}\t{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def build_frozen_validation_loader(val_ds, cfg):
    """Build the reusable validation panel without multiprocessing IPC state."""
    val_subset = Subset(
        val_ds,
        val_ds.stratified_indices(seed=cfg.data.seed + 2),
    )
    # Rank 0 retains every validation batch for all checkpoints. Worker-backed
    # tensors keep shared-memory file descriptors alive for the lifetime of that
    # list and fail on full-domain panels (``received 0 items of ancdata``).
    # This one-time sequential load is deterministic and does not affect the
    # steady-state training loader.
    return DataLoader(
        val_subset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
    )


@torch.no_grad()
def evaluate(model, loader, device, cfg, amp_dtype, max_batches=None):
    model.eval()
    core = model.module if hasattr(model, "module") else model
    saved = core.noise_sigma
    saved_v = core.noise_sigma_v
    core.noise_sigma = 0.0
    core.noise_sigma_v = 0.0
    losses, rmsds, noop = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        tau0 = torch.zeros(batch["P_t"].shape[0], device=device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.train.amp):
            out = core(batch, tau=tau0)
            loss, _ = total_loss(out, batch, cfg, core)
        losses.append(loss.item())
        rmsds.append(masked_ca_rmsd(out["P_hat_1"].float(), batch["P_1"], batch["residue_mask"]).mean().item())
        noop.append(masked_ca_rmsd(batch["P_t"], batch["P_1"], batch["residue_mask"]).mean().item())
    core.noise_sigma = saved
    core.noise_sigma_v = saved_v
    model.train()
    return {"val_loss": sum(losses) / len(losses), "val_rmsd": sum(rmsds) / len(rmsds),
            "noop_rmsd": sum(noop) / len(noop)}


def save_ckpt(path, core, opt, scaler, step, cfg, train_state):
    # Write atomically: torch.save to a temp file in the same dir, then os.replace (atomic on
    # POSIX). A concurrent reader (e.g. cloud/huawei/ckpt_to_obs.sh syncing to OBS) then always sees a
    # complete file -- old or new, never a half-written last.ckpt.
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": core.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "cfg": to_dict(cfg),
                "checkpoint_schema": 2, "train_state": train_state}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--warm-start", default=None,
                    help="load compatible model weights only; use a fresh optimizer and step 0")
    ap.add_argument(
        "--allow-legacy-resume", action="store_true",
        help="resume a checkpoint without sampler state (safe but repeats data; non-bitwise)",
    )
    args = ap.parse_args()
    if args.resume and args.warm_start:
        ap.error("--resume and --warm-start are mutually exclusive")
    cfg = load_config(args.config)
    resume = args.resume or cfg.train.resume
    if resume and not Path(resume).exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume}")

    rank, world, local = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed + rank)
    amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16
    use_scaler = cfg.train.amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    train_ds, val_ds, train_files, val_files = build_datasets(cfg)
    sampler = ResumableDistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True,
        seed=cfg.train.seed,
    )
    train_fingerprint = dataset_fingerprint(train_files)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
                              num_workers=cfg.train.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=collate_pairs, persistent_workers=cfg.train.num_workers > 0)
    # One deterministic frame from every validation trajectory prevents the
    # former first-30-batches evaluation from collapsing onto one trajectory.
    val_loader = build_frozen_validation_loader(val_ds, cfg)
    # Freeze validation crops once on rank 0. Re-iterating a multi-worker dataset
    # advances each worker's crop RNG, so checkpoints would otherwise see subtly
    # different panels (observable as a changing no-op RMSD).
    val_batches = list(val_loader) if is_main(rank) else []

    model = DeepJumpLite(cfg.model, noise_sigma=cfg.data.noise_sigma,
                         predict_heavy=cfg.model.predict_heavy).to(device)
    if is_main(rank):
        eff = cfg.train.batch_size * world * cfg.train.grad_accum
        print(f"world={world} params={count_parameters(model):,} effective_batch={eff} "
              f"train_samples={len(train_ds)} val_samples={len(val_ds)} "
              f"train_domains={len(train_files)} val_domains={len(val_files)}")
    if world > 1:
        # find_unused_parameters is needed if any parameter gets no gradient in a step:
        #  - w_unroll>0 does multiple forwards; or
        #  - predict_heavy=True with NO V-loss (w_offset=w_allatom=0) leaves head_v unused.
        # Without it, DDP's reducer stalls and gradients are silently under-synchronised.
        heavy_unused = cfg.model.predict_heavy and cfg.train.w_offset == 0 and cfg.train.w_allatom == 0
        # PaperFeedForward's branched scalar/vector graph requires the reducer's
        # graph traversal on current PyTorch/NCCL; without it a real two-rank
        # regression shows only 8/70 parameter gradients synchronized. The final
        # TensorCloud01 transport FF scalar projection is structurally unused
        # because the transport head consumes vectors only.
        find_unused = (
            cfg.train.w_unroll > 0
            or heavy_unused
            or getattr(cfg.model, "paper_ff", False)
            or getattr(cfg.model, "tensor_cloud01", False)
        )
        model = DDP(model, device_ids=[local], find_unused_parameters=find_unused)
    core = model.module if hasattr(model, "module") else model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    start_step = 0
    epoch = 0
    consumed_local = 0
    if args.warm_start:
        ck = torch.load(args.warm_start, map_location=device, weights_only=False)
        incompatible = core.load_state_dict(ck["model"], strict=False)
        allowed_suffixes = (
            ".attn.vector_qk_gate", ".attn.to_qv.weight", ".attn.to_kv.weight",
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.endswith(allowed_suffixes)
        ]
        # A Tensor-Cloud adaptation can reuse the already-trained scalar/vector
        # q/k projections from a gated vector-q/k checkpoint.  The per-head gate
        # has no counterpart in the joint-inner-product path and is the only
        # source key that may be deliberately dropped.
        allowed_unexpected = []
        if getattr(cfg.model, "tensor_qkv", False):
            allowed_unexpected = [
                key for key in incompatible.unexpected_keys
                if key.endswith(".attn.vector_qk_gate")
            ]
        invalid_unexpected = [
            key for key in incompatible.unexpected_keys
            if key not in allowed_unexpected
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "unsafe warm start: "
                f"missing={invalid_missing}, unexpected={invalid_unexpected}"
            )
        if is_main(rank):
            print(
                f"warm-started model from {args.warm_start}; fresh optimizer/step 0; "
                f"new_keys={len(incompatible.missing_keys)} "
                f"dropped_keys={len(allowed_unexpected)}"
            )
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_step = ck["step"]
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        state = ck.get("train_state")
        if state is None:
            if not args.allow_legacy_resume:
                raise RuntimeError(
                    "checkpoint has no sampler state; pass --allow-legacy-resume "
                    "for a non-bitwise continuation that may repeat data"
                )
            if is_main(rank):
                print("warning: legacy resume resets sampler to epoch 0 / offset 0")
        else:
            expected = {
                "world_size": world,
                "train_dataset_size": len(train_ds),
                "sampler_num_samples": sampler.num_samples,
                "sampler_seed": cfg.train.seed,
                "batch_size": cfg.train.batch_size,
                "grad_accum": cfg.train.grad_accum,
                "train_fingerprint": train_fingerprint,
            }
            mismatches = {
                key: (state.get(key), value)
                for key, value in expected.items() if state.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"unsafe resume state mismatch: {mismatches}")
            epoch = int(state["sampler_epoch"])
            consumed_local = int(state["samples_consumed_per_rank"])
            if consumed_local % cfg.train.batch_size:
                raise RuntimeError(
                    f"resume sample offset {consumed_local} is not batch aligned"
                )
            sampler.set_epoch(epoch)
            sampler.set_start_offset(consumed_local)
        if is_main(rank):
            print(
                f"resumed from {resume} at step {start_step}; "
                f"sampler_epoch={epoch} samples_per_rank={consumed_local}"
            )

    out_dir = Path(cfg.train.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
    history, saved_ckpts = [], []
    clip_count_window = 0
    scaler_skips_window = 0

    model.train()
    step = start_step
    accum = cfg.train.grad_accum
    data_iter = iter(train_loader)
    t0 = time.time()
    data_wait = 0.0  # seconds spent waiting on the dataloader per log window
    while step < cfg.train.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        comp_sums = {}
        for micro in range(accum):
            t_data = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                consumed_local = 0
                sampler.set_start_offset(0)
                sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            data_wait += time.time() - t_data
            batch = move_batch(batch, device)
            sync = micro == accum - 1 or world == 1
            ctx = model.no_sync() if (world > 1 and not sync) else _nullctx()
            with ctx:
                with torch.autocast("cuda", dtype=amp_dtype, enabled=cfg.train.amp):
                    # MUST call the DDP-wrapped `model` (not `core`) so the backward
                    # autograd hooks fire and gradients are all-reduced across ranks.
                    out = model(batch)
                    loss, comps = total_loss(out, batch, cfg, model)
                    loss = loss / accum
                scaler.scale(loss).backward()
            consumed_local += int(batch["P_t"].shape[0])
            loss_val += loss.item()
            for name, value in comps.items():
                comp_sums[name] = comp_sums.get(name, 0.0) + value / accum
        if use_scaler:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        grad_norm_value = float(grad_norm)
        finite_step = torch.tensor(
            int(math.isfinite(loss_val) and math.isfinite(grad_norm_value)),
            device=device,
            dtype=torch.int32,
        )
        if world > 1:
            dist.all_reduce(finite_step, op=dist.ReduceOp.MIN)
        if not finite_step.item():
            raise FloatingPointError(
                f"non-finite loss or gradient at step {step + 1}: "
                f"rank={rank} loss={loss_val} grad={grad_norm_value}"
            )
        clip_count_window += int(grad_norm_value > cfg.train.grad_clip)
        scale_before = scaler.get_scale() if use_scaler else 1.0
        scaler.step(opt); scaler.update()
        scale_after = scaler.get_scale() if use_scaler else 1.0
        scaler_skips_window += int(scale_after < scale_before)
        step += 1

        if step % cfg.train.log_every == 0:
            local_peak = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0
            )
            peak_tensor = torch.tensor(local_peak, device=device, dtype=torch.float64)
            if world > 1:
                gathered_peaks = [torch.zeros_like(peak_tensor) for _ in range(world)]
                dist.all_gather(gathered_peaks, peak_tensor)
                peak_by_rank = [float(value) for value in gathered_peaks]
            else:
                peak_by_rank = [local_peak]
        if is_main(rank) and step % cfg.train.log_every == 0:
            elapsed = time.time() - t0
            rate = cfg.train.log_every / elapsed
            data_pct = 100 * data_wait / elapsed if elapsed > 0 else 0
            # Cumulative high-water mark on every rank; max_memory_allocated is
            # deliberately never reset. This proves that every GPU fits, rather
            # than reporting only rank 0 and assuming the others are equivalent.
            peak_text = ",".join(f"{value:.1f}" for value in peak_by_rank)
            t0 = time.time(); data_wait = 0.0
            comp_text = " ".join(f"{name}={value:.4f}" for name, value in sorted(comp_sums.items()))
            print(f"step {step:>7}/{cfg.train.max_steps}  loss {loss_val:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {rate:.2f} it/s  "
                  f"{1000/rate:.0f} ms/step  grad {grad_norm_value:.3f}  "
                  f"clip {clip_count_window}/{cfg.train.log_every}  "
                  f"scaler_skips {scaler_skips_window}  "
                  f"peakGPU_by_rank(cum) [{peak_text}]GB  "
                  f"data {data_pct:.0f}%  {comp_text}")
            clip_count_window = 0
            scaler_skips_window = 0

        did_val_or_ckpt = False
        if is_main(rank) and (step % cfg.train.val_every == 0 or step == cfg.train.max_steps):
            m = evaluate(model, val_batches, device, cfg, amp_dtype)
            m["step"] = step
            history.append(m)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            print(f"  [val] step {step}  loss {m['val_loss']:.4f}  rmsd {m['val_rmsd']:.3f} "
                  f"(no-op {m['noop_rmsd']:.3f})")
            did_val_or_ckpt = True
        if is_main(rank) and (step % cfg.train.ckpt_every == 0 or step == cfg.train.max_steps):
            train_state = {
                "world_size": world,
                "train_dataset_size": len(train_ds),
                "sampler_num_samples": sampler.num_samples,
                "sampler_seed": cfg.train.seed,
                "sampler_epoch": epoch,
                "samples_consumed_per_rank": consumed_local,
                "batch_size": cfg.train.batch_size,
                "grad_accum": cfg.train.grad_accum,
                "train_fingerprint": train_fingerprint,
                "crop_resume": "stochastic_worker_rng_not_bitwise",
            }
            p = out_dir / f"ckpt_{step}.pt"
            save_ckpt(p, core, opt, scaler, step, cfg, train_state)
            save_ckpt(out_dir / "last.ckpt", core, opt, scaler, step, cfg, train_state)
            saved_ckpts.append(p)
            while len(saved_ckpts) > cfg.train.keep_last_k:
                old = saved_ckpts.pop(0)
                old.unlink(missing_ok=True)
            did_val_or_ckpt = True
        # Exclude validation/checkpoint time from the NEXT it/s window (else the step after a
        # val/ckpt looks artificially slow). Reset the timing baseline here.
        if did_val_or_ckpt:
            t0 = time.time(); data_wait = 0.0

    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    if is_main(rank):
        print(f"done. artifacts in {out_dir}")


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
