#!/usr/bin/env python
"""How long does geometry survive, and what happens from an extended chain?

Two questions the ground-truth-anchored evaluators cannot answer:

1. **Geometric horizon.** `rollout_robustness_eval.py` needs a real MD frame per
   step to score against, so it cannot roll past the trajectory length (440
   frames in mdCATH). Chemical validity needs no reference, so a free rollout can
   run as long as we like and report where, if ever, the structure stops being a
   protein.

2. **Off-equilibrium behaviour.** Ab initio folding starts from an extended
   chain, which is maximally outside a training distribution built from
   equilibrium MD. Feeding one in and watching where it goes is a direct probe of
   the regime folding actually requires, rather than an inference from
   near-equilibrium rollouts.

Reports per checkpoint step: CA-CA bond statistics, radius of gyration, and the
fraction of non-adjacent CA pairs closer than 4 A (a steric clash proxy). None of
these need a reference structure. Rg is the informative one from an extended
start: a model that folds must contract it, and one that merely stays legal will
not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deepjump.config import ModelConfig
from deepjump.data.mdcath import _DomainHandle
from deepjump.metrics import aligned_ca_rmsd, contact_fraction_native
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_model_layout
from deepjump.utils import model_config_kwargs, resolve_device

REAL_CA_BOND_A = 3.8
CLASH_RADIUS_A = 4.0


def geometry(positions: torch.Tensor) -> dict:
    """Reference-free chemical-validity diagnostics for one CA trace."""
    bonds = (positions[1:] - positions[:-1]).norm(dim=-1)
    centred = positions - positions.mean(0, keepdim=True)
    distances = torch.cdist(positions, positions)
    n = positions.shape[0]
    # Only non-adjacent pairs can clash; |i-j|>2 avoids counting real backbone geometry.
    far = torch.triu(torch.ones(n, n, dtype=torch.bool, device=positions.device), diagonal=3)
    clashes = ((distances < CLASH_RADIUS_A) & far).sum().item()
    pairs = far.sum().item()
    return {
        "bond_mean": float(bonds.mean()),
        "bond_max": float(bonds.max()),
        "bond_min": float(bonds.min()),
        "radius_of_gyration": float(centred.norm(dim=-1).pow(2).mean().sqrt()),
        "clash_fraction": clashes / max(pairs, 1),
        "finite": bool(torch.isfinite(positions).all()),
    }


def load_model(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg, data_cfg = checkpoint["cfg"]["model"], checkpoint["cfg"]["data"]
    model = DeepJumpLite(
        ModelConfig(**model_config_kwargs(model_cfg, ModelConfig)),
        noise_sigma=data_cfg["noise_sigma"],
        predict_heavy=model_cfg["predict_heavy"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, data_cfg


def random_compact_chain(n_residues: int, radius_a: float, device, seed: int = 0) -> torch.Tensor:
    """A self-avoiding chain with correct bond lengths, confined to a sphere.

    This is the floor for interpreting a native-contact fraction. Any chain of the
    right length folded into the right volume recovers *some* native contacts by
    geometry alone; without knowing how many, an FNC of 0.6 cannot be called
    folding.
    """
    generator = torch.Generator().manual_seed(seed)
    points = [torch.zeros(3)]
    for _ in range(n_residues - 1):
        for _ in range(200):
            step = torch.randn(3, generator=generator)
            candidate = points[-1] + REAL_CA_BOND_A * step / step.norm()
            if candidate.norm() > radius_a:
                continue
            if len(points) < 3:
                break
            far = torch.stack(points[:-2])
            if (far - candidate).norm(dim=-1).min() > CLASH_RADIUS_A:
                break
        points.append(candidate)
    chain = torch.stack(points)
    return (chain - chain.mean(0, keepdim=True)).to(device)


def extended_chain(n_residues: int, device, rise_a: float = 3.6) -> torch.Tensor:
    """A nearly straight CA trace: the maximally off-equilibrium starting point."""
    index = torch.arange(n_residues, dtype=torch.float32, device=device)
    # A slight zig-zag rather than a mathematical line, so the first frame is not
    # degenerate for any alignment or SVD downstream.
    return torch.stack([index * rise_a, (index % 2) * 0.5, torch.zeros_like(index)], dim=-1)


def score_against_native(positions: torch.Tensor, native: torch.Tensor) -> dict:
    """Is a compact structure actually folded, or merely collapsed?

    Radius of gyration alone cannot tell those apart: a random globule and the
    native fold can share a size. RMSD after Kabsch superposition and the
    recovered fraction of native contacts can.
    """
    mask = torch.ones(1, positions.shape[0], dtype=torch.bool, device=positions.device)
    return {
        "rmsd_to_native": float(aligned_ca_rmsd(positions, native)),
        "native_contact_fraction": float(
            contact_fraction_native(positions[None], native[None], mask)[0]
        ),
    }


@torch.no_grad()
def free_rollout(model, batch, n_steps: int, record_every: int, native=None,
                 sampler_seed: int = 0, **sample_kwargs):
    generator = torch.Generator().manual_seed(sampler_seed)
    current = dict(batch)

    def record(positions):
        entry = geometry(positions)
        if native is not None and entry["finite"]:
            entry.update(score_against_native(positions, native))
        return entry

    trace = [(0, record(current["P_t"][0]))]
    for step in range(1, n_steps + 1):
        positions, vectors = model.sample(current, generator=generator, **sample_kwargs)
        if not torch.isfinite(positions).all():
            trace.append((step, record(positions[0])))
            break
        current = {**current, "P_t": positions, "V_t": vectors}
        if step % record_every == 0 or step == n_steps:
            trace.append((step, record(positions[0])))
    return trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", required=True, help="one mdCATH domain, for topology and a real start")
    ap.add_argument("--start", choices=("native", "extended"), default="native")
    ap.add_argument("--temperature", type=int, default=320)
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--shuffle-sequence", type=int, default=None, metavar="SEED",
                    help="permute res_index with this seed: keeps composition and length "
                         "but destroys sequence-specific signal, so a still-high native "
                         "contact fraction means generic compaction rather than folding")
    ap.add_argument("--seed", type=int, default=0, help="sampler seed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, data_cfg = load_model(Path(args.ckpt), device)
    delta = data_cfg["delta_frames"]
    delta = delta if isinstance(delta, int) else int(delta[0])

    handle = _DomainHandle(Path(args.h5))
    layout = handle.layout
    coords = torch.from_numpy(np.asarray(handle.coords(args.temperature, args.replica, args.frame)))
    positions, vectors = apply_model_layout(
        coords, layout, canon_symmetric=bool(data_cfg.get("canon_symmetric", False))
    )
    positions = positions - positions.mean(0, keepdim=True)
    native_reference = positions.clone().to(device)
    if args.start == "extended":
        # Keep the real side-chain offsets and topology; only the CA trace is replaced,
        # so the model sees a valid protein that is simply unfolded.
        positions = extended_chain(positions.shape[0], device=positions.device)
        positions = positions - positions.mean(0, keepdim=True)

    n = positions.shape[0]
    res_index = torch.as_tensor(layout.res_index)
    if args.shuffle_sequence is not None:
        permutation = torch.randperm(n, generator=torch.Generator().manual_seed(args.shuffle_sequence))
        res_index = res_index[permutation]
        print(f"sequence permuted with seed {args.shuffle_sequence}: same composition, "
              "no sequence-specific signal")
    batch = {
        "P_t": positions[None].to(device),
        "V_t": vectors[None].to(device),
        "res_index": res_index[None].to(device),
        "atom_mask": torch.as_tensor(layout.atom_mask)[None].to(device),
        "residue_mask": torch.ones(1, n, dtype=torch.bool, device=device),
        "delta_ns": torch.tensor([float(delta)], device=device),
    }

    print(f"domain {handle.name}  N={n}  start={args.start}  delta={delta} ns  "
          f"steps={args.steps} x ode_{args.ode_steps}")
    print(f"reference: CA-CA bond {REAL_CA_BOND_A} A; clash = non-adjacent CA pair < {CLASH_RADIUS_A} A\n")

    # Two baselines make the numbers readable. The native frame against itself is
    # the floor; a different frame of the same trajectory is what "same protein,
    # different conformer" scores; both bound what a collapse-without-folding
    # result would look like.
    other = torch.from_numpy(np.asarray(handle.coords(args.temperature, args.replica, 200)))
    other_p, _ = apply_model_layout(
        other, layout, canon_symmetric=bool(data_cfg.get("canon_symmetric", False))
    )
    other_p = (other_p - other_p.mean(0, keepdim=True)).to(device)
    far_apart = score_against_native(other_p, native_reference)
    native_rg = float((native_reference - native_reference.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt())
    globules = [
        score_against_native(random_compact_chain(n, native_rg * 1.3, device, seed=k), native_reference)
        for k in range(5)
    ]
    globule_fnc = float(np.mean([g["native_contact_fraction"] for g in globules]))
    globule_rmsd = float(np.mean([g["rmsd_to_native"] for g in globules]))
    print(f"baseline  real MD, 200 ns later : RMSD {far_apart['rmsd_to_native']:.2f} A, "
          f"FNC {far_apart['native_contact_fraction']:.3f}")
    print(f"floor     random compact chain  : RMSD {globule_rmsd:.2f} A, FNC {globule_fnc:.3f}  "
          f"(mean of 5; anything at or below this is compaction, not folding)\n")

    trace = free_rollout(model, batch, args.steps, args.record_every,
                         native=native_reference, sampler_seed=args.seed,
                         steps=args.ode_steps, mode="ode")

    print(f"{'step':>7}{'bond mean':>11}{'Rg (A)':>9}{'clash':>9}"
          f"{'RMSD nat':>10}{'FNC':>8}  finite")
    for step, g in trace:
        rmsd = f"{g['rmsd_to_native']:>10.2f}" if "rmsd_to_native" in g else f"{'-':>10}"
        fnc = f"{g['native_contact_fraction']:>8.3f}" if "native_contact_fraction" in g else f"{'-':>8}"
        print(f"{step:>7}{g['bond_mean']:>11.3f}{g['radius_of_gyration']:>9.2f}"
              f"{g['clash_fraction']:>9.4f}{rmsd}{fnc}  {g['finite']}")

    last = trace[-1][1]
    reached = trace[-1][0]
    print(f"\nsurvived {reached} of {args.steps} requested steps; "
          f"final bond {last['bond_mean']:.3f} A, Rg {last['radius_of_gyration']:.2f} A")
    if args.start == "extended":
        print(f"Rg went {trace[0][1]['radius_of_gyration']:.2f} -> "
              f"{last['radius_of_gyration']:.2f} A. Folding requires a large contraction; "
              "staying flat means the model holds the chain legal without folding it.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "domain": handle.name, "start": args.start, "steps_requested": args.steps,
            "steps_survived": reached, "ode_steps": args.ode_steps,
            "sampler_seed": args.seed, "shuffled_sequence_seed": args.shuffle_sequence,
            "baseline_real_md_200ns": far_apart,
            "floor_random_compact_chain": {"rmsd_to_native": globule_rmsd,
                                           "native_contact_fraction": globule_fnc},
            "trace": [{"step": s, **g} for s, g in trace],
        }, indent=2))
        print(f"wrote {args.json_out}")
    handle.close()


if __name__ == "__main__":
    main()
