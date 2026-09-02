#!/usr/bin/env python
"""Does TICA JSD track training exposure, or just how hard each domain's landscape is?

Four held-out domains scored better than four trained-on ones (0.41 vs 0.52 mean),
which is evidence against memorisation but is easy to over-read: the held-out four
happened to have simpler real free energy surfaces. This widens the comparison to
every locally available held-out domain and measures a landscape-difficulty proxy
alongside the score, so the two explanations can be told apart.

Difficulty proxy: the Shannon entropy of the *real MD* occupancy on the same grid
the JSD uses. A landscape spread over many bins is harder to reproduce than one
concentrated in a few, independently of whether the model trained on it.

Read-only; writes a JSON summary to the path given by --json-out.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_panel_module():
    """Import scripts/tica_panel.py for build_domain and the fixed JSD."""
    spec = importlib.util.spec_from_file_location(
        "tica_panel_under_test", REPO / "scripts" / "tica_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def occupancy_entropy(tic, rng, bins=24):
    """Shannon entropy (bits) of the reference occupancy on the JSD's own grid."""
    edges = [np.linspace(rng[0][0], rng[0][1], bins + 1),
             np.linspace(rng[1][0], rng[1][1], bins + 1)]
    hist, _, _ = np.histogram2d(tic[:, 0], tic[:, 1], bins=edges)
    p = hist.ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(REPO / "artifacts/formal500k/20260726T164217Z/ckpt_500000.pt"))
    ap.add_argument("--root", required=True, help="mdCATH staging root")
    ap.add_argument("--train-list", default=str(REPO / "docs/provenance/train_eligible_5218.txt"))
    ap.add_argument("--held-out", nargs="+", required=True)
    ap.add_argument("--trained-on", nargs="+", required=True)
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    panel = load_panel_module()
    from deepjump.config import ModelConfig
    from deepjump.data.mdcath import _DomainHandle
    from deepjump.model import DeepJumpLite
    from deepjump.utils import resolve_device

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg, data_cfg = checkpoint["cfg"]["model"], checkpoint["cfg"]["data"]
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    for key, value in model_cfg.items():
        if key not in known and value:
            raise SystemExit(f"checkpoint sets unknown model option {key}={value!r}")
    device = resolve_device("auto")
    model = DeepJumpLite(
        ModelConfig(**{k: v for k, v in model_cfg.items() if k in known}),
        noise_sigma=data_cfg["noise_sigma"],
        predict_heavy=model_cfg["predict_heavy"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    trained_ids = {line.strip() for line in Path(args.train_list).read_text().split() if line.strip()}
    temp, rep = data_cfg["temperatures"][0], data_cfg["replicas"][0]
    canon = bool(data_cfg.get("canon_symmetric", False))
    sample_kwargs = dict(integrator="euler", tau_max=0.9, terminal_denoise=True)

    rows = []
    for name in list(args.held_out) + list(args.trained_on):
        expected_held_out = name in args.held_out
        actually_trained = name in trained_ids
        if expected_held_out == actually_trained:
            raise SystemExit(
                f"{name}: caller says held_out={expected_held_out} but the training "
                f"list says trained={actually_trained}"
            )
        handle = _DomainHandle(Path(args.root) / "data" / f"mdcath_dataset_{name}.h5")
        print(f"  {name} ({'held out' if not actually_trained else 'trained on'}) ...", flush=True)
        block = panel.build_domain(
            model, handle, temp, rep, device, args.starts, args.K, args.steps,
            data_cfg["noise_sigma"], canon_symmetric=canon, sample_kwargs=sample_kwargs,
        )
        real_tic, model_tic = block["real_tic"], block["model_tic"]
        low, high = real_tic.min(0), real_tic.max(0)
        pad = 0.05 * (high - low + 1e-6)
        rng = [[low[0] - pad[0], high[0] + pad[0]], [low[1] - pad[1], high[1] + pad[1]]]
        rows.append({
            "domain": name,
            "trained_on": actually_trained,
            "jsd": float(panel.hist2d_jsd(real_tic, model_tic, rng, bins=24)),
            "real_entropy_bits": occupancy_entropy(real_tic, rng),
            "residues": int(handle.layout.num_residues),
        })
        handle.close()

    rows.sort(key=lambda r: r["jsd"])
    print(f"\n{'domain':<10}{'status':<12}{'JSD':>7}{'real entropy':>14}{'residues':>10}")
    for r in rows:
        print(f"{r['domain']:<10}{'trained on' if r['trained_on'] else 'held out':<12}"
              f"{r['jsd']:>7.3f}{r['real_entropy_bits']:>14.2f}{r['residues']:>10}")

    held = [r for r in rows if not r["trained_on"]]
    seen = [r for r in rows if r["trained_on"]]
    mean = lambda xs, k: float(np.mean([x[k] for x in xs]))
    print(f"\nheld out   n={len(held)}  mean JSD {mean(held,'jsd'):.3f}  "
          f"mean entropy {mean(held,'real_entropy_bits'):.2f}")
    print(f"trained on n={len(seen)}  mean JSD {mean(seen,'jsd'):.3f}  "
          f"mean entropy {mean(seen,'real_entropy_bits'):.2f}")

    jsd = np.array([r["jsd"] for r in rows])
    ent = np.array([r["real_entropy_bits"] for r in rows])
    trained = np.array([float(r["trained_on"]) for r in rows])
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print(f"\ncorr(JSD, real-landscape entropy) = {corr(jsd, ent):+.3f}")
    print(f"corr(JSD, trained-on)             = {corr(jsd, trained):+.3f}")
    print("\nIf the first is the stronger association, the held-out advantage is a")
    print("difficulty artefact and says nothing about generalisation either way.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "rows": rows,
            "corr_jsd_entropy": corr(jsd, ent),
            "corr_jsd_trained": corr(jsd, trained),
        }, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
