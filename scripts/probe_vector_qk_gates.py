#!/usr/bin/env python
"""Report learned gated vector-q/k attention strength from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    rows = []
    for name, value in checkpoint["model"].items():
        if name.endswith("attn.vector_qk_gate"):
            gate = torch.tanh(value.float())
            rows.append({
                "name": name,
                "raw": value.float().tolist(),
                "tanh": gate.tolist(),
                "mean_abs_tanh": float(gate.abs().mean().item()),
                "max_abs_tanh": float(gate.abs().max().item()),
            })
    if not rows:
        raise ValueError("checkpoint has no vector_qk_gate parameters")
    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint["step"],
        "gate_count": len(rows),
        "overall_mean_abs_tanh": float(sum(r["mean_abs_tanh"] for r in rows) / len(rows)),
        "overall_max_abs_tanh": max(r["max_abs_tanh"] for r in rows),
        "gates": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
