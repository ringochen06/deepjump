#!/usr/bin/env python
"""Does the model use sequence at all, at the level of a single update?

The folding probe compared native-contact fractions after 600 chained jumps, where
sequence effects and seed noise are both compounded. This asks the question
directly: hold the structure fixed, change only `res_index`, and measure how far
the predicted next state moves.

The scale that makes the answer readable is the size of the update itself. If
swapping the sequence perturbs the prediction by a small fraction of how far the
model moves the structure in one jump, the sequence channel is not being used,
and no amount of further training on the same objective will make it fold.

Read-only; writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from deepjump.config import ModelConfig  # noqa: E402
from deepjump.data.mdcath import _DomainHandle  # noqa: E402
from deepjump.model import DeepJumpLite  # noqa: E402
from deepjump.representation import apply_model_layout  # noqa: E402
from deepjump.utils import model_config_kwargs, resolve_device  # noqa: E402


def load(ckpt: Path, device):
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_cfg, data_cfg = checkpoint["cfg"]["model"], checkpoint["cfg"]["data"]
    model = DeepJumpLite(
        ModelConfig(**model_config_kwargs(model_cfg, ModelConfig)),
        noise_sigma=data_cfg["noise_sigma"],
        predict_heavy=model_cfg["predict_heavy"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, data_cfg


@torch.no_grad()
def predict(model, batch, res_index, seed: int):
    """One deterministic jump under a given sequence assignment."""
    generator = torch.Generator().manual_seed(seed)
    return model.sample({**batch, "res_index": res_index}, steps=20, mode="ode",
                        generator=generator)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(REPO / "artifacts/formal500k/20260726T164217Z/ckpt_500000.pt"))
    ap.add_argument("--h5", required=True)
    ap.add_argument("--temperature", type=int, default=320)
    ap.add_argument("--replica", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()

    device = resolve_device("auto")
    model, data_cfg = load(Path(args.ckpt), device)
    delta = data_cfg["delta_frames"]
    delta = delta if isinstance(delta, int) else int(delta[0])

    handle = _DomainHandle(Path(args.h5))
    layout = handle.layout
    coords = torch.from_numpy(np.asarray(handle.coords(args.temperature, args.replica, args.frame)))
    positions, vectors = apply_model_layout(
        coords, layout, canon_symmetric=bool(data_cfg.get("canon_symmetric", False))
    )
    positions = (positions - positions.mean(0, keepdim=True)).to(device)
    n = positions.shape[0]
    real_seq = torch.as_tensor(layout.res_index).to(device)

    batch = {
        "P_t": positions[None], "V_t": vectors[None].to(device),
        "atom_mask": torch.as_tensor(layout.atom_mask)[None].to(device),
        "residue_mask": torch.ones(1, n, dtype=torch.bool, device=device),
        "delta_ns": torch.tensor([float(delta)], device=device),
    }

    variants = {"real": real_seq}
    for k in range(3):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(100 + k))
        variants[f"shuffled_{k}"] = real_seq[perm].to(device)
    variants["all_one_type"] = torch.full_like(real_seq, int(real_seq[0]))

    print(f"domain {handle.name}  N={n}  delta={delta} ns  ode_20, {args.seeds} sampler seeds\n")

    reference = {s: predict(model, batch, real_seq, s) for s in range(args.seeds)}
    update = float(np.mean([float((reference[s] - positions).norm(dim=-1).mean()) for s in reference]))
    seed_pairs = [(a, b) for a in reference for b in reference if a < b]
    seed_noise = float(np.mean([
        float((reference[a] - reference[b]).norm(dim=-1).mean()) for a, b in seed_pairs
    ])) if seed_pairs else float("nan")

    print(f"scale 1  how far one jump moves each CA        : {update:.4f} A")
    print(f"scale 2  spread between sampler seeds, same seq: {seed_noise:.4f} A\n")

    print(f"{'sequence':<16}{'shift vs real':>15}{'/ update':>10}{'/ seed noise':>14}")
    for name, seq in variants.items():
        if name == "real":
            continue
        shifts = [
            float((predict(model, batch, seq, s) - reference[s]).norm(dim=-1).mean())
            for s in range(args.seeds)
        ]
        shift = float(np.mean(shifts))
        print(f"{name:<16}{shift:>15.4f}{shift / update:>10.3f}{shift / seed_noise:>14.3f}")

    print("\nA sequence change that moves the prediction less than resampling the")
    print("noise does is not being used. Folding needs the opposite: sequence must")
    print("dominate, because it is the only thing distinguishing one fold from another.")
    handle.close()


if __name__ == "__main__":
    main()
