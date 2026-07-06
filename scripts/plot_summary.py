#!/usr/bin/env python
"""Consolidate the key DeepJump-lite results into one summary figure.

Reads saved JSON artifacts (training history + rollout evals) and produces
runs/summary.png. No checkpoints needed.

    python scripts/plot_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(p):
    return json.load(open(p)) if Path(p).exists() else None


def main():
    root = Path("runs")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: honest tau=0 training curve (val) vs no-op, delta=1 Ca-only
    hist = _load(root / "ca_delta1" / "history.json")
    if hist:
        steps = [h["step"] for h in hist]
        ax[0].plot(steps, [h["val_rmsd"] for h in hist], "o-", label="model (τ=0)")
        ax[0].axhline(hist[-1]["noop_rmsd"], color="orange", ls="--", label="no-op")
        ax[0].set(xlabel="train step", ylabel="val CA RMSD (Å)",
                  title="honest τ=0 single-step\n(≈ no-op: conditional mean)")
        ax[0].legend(); ax[0].grid(alpha=.3)

    # Panel 2: rollout RMSD vs step (log) - stability story
    styles = {"ode": ("C3", "-", "ODE"), "mean": ("C0", "-", "mean"),
              "ode_gated": ("C3", ":", "ODE+gate"), "mean_gated": ("C0", ":", "mean+gate")}
    for key, (c, ls, lab) in styles.items():
        d = _load(root / "rollout" / f"rollout_{key}.json")
        if d:
            ax[1].plot(d["steps"], d["rmsd"], ls, color=c, marker="o", ms=3, label=lab)
    d = _load(root / "rollout" / "rollout_mean.json")
    if d:
        ax[1].plot(d["steps"], d["true_drift"], "k--", label="true drift (no-op)")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="rollout step (ns)", ylabel="CA RMSD vs truth (Å, log)",
              title="rollout stability\n(gate bounds divergence)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

    # Panel 3: rollout bond geometry vs step (log) - explosion vs bounded
    for key, (c, ls, lab) in styles.items():
        d = _load(root / "rollout" / f"rollout_{key}.json")
        if d:
            ax[2].plot(d["steps"], d["bond"], ls, color=c, marker="o", ms=3, label=lab)
    ax[2].axhline(3.8, color="gray", ls=":", label="ideal 3.8 Å")
    ax[2].set_yscale("log")
    ax[2].set(xlabel="rollout step (ns)", ylabel="mean CA–CA bond (Å, log)",
              title="chemical validity\n(ungated explodes)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both")

    fig.suptitle("DeepJump-lite — key results (H=32, 30 mdCATH domains, MPS)", y=1.03)
    fig.tight_layout()
    out = Path("docs"); out.mkdir(exist_ok=True)
    path = out / "summary.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
