#!/usr/bin/env python
"""Figure: how rollout stability improves across fixes. Saves docs/stability.png.

Reads rollout JSONs produced by rollout_eval.py at the paths below.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


CURVES = [
    ("runs/rollout/rollout_ode.json", "C3", "-", "baseline ODE"),
    ("runs/rollout/rollout_mean.json", "C1", "-", "baseline mean"),
    ("runs/rc_unroll/rollout_mean.json", "C0", "-", "2-step unroll (ungated)"),
    ("runs/rc_unroll3/rollout_mean.json", "C4", "-", "3-step unroll (ungated)"),
    ("runs/rc_unroll3g/rollout_mean_gated.json", "C2", "-", "3-step unroll + gate"),
]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for path, c, ls, lab in CURVES:
        d = load(path)
        if not d:
            continue
        ax[0].plot(d["steps"], d["rmsd"], ls, color=c, marker="o", ms=3, label=lab)
        ax[1].plot(d["steps"], d["bond"], ls, color=c, marker="o", ms=3, label=lab)
    d0 = load("runs/rc_unroll3/rollout_mean.json")
    if d0:
        ax[0].plot(d0["steps"], d0["true_drift"], "k--", label="true drift")
    for a in ax:
        a.set_yscale("log"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
        a.set_xlabel("rollout step (ns)")
    ax[0].set(ylabel="CA RMSD vs truth (Å, log)", title="rollout accuracy")
    ax[1].axhline(3.8, color="gray", ls=":")
    ax[1].set(ylabel="mean CA–CA bond (Å, log)", title="chemical validity")
    fig.suptitle("Rollout stability: deeper unrolled training extends the horizon (+ gate)", y=1.02)
    fig.tight_layout()
    out = Path("docs"); out.mkdir(exist_ok=True)
    fig.savefig(out / "stability.png", dpi=120, bbox_inches="tight")
    print(f"saved {out/'stability.png'}")


if __name__ == "__main__":
    main()
