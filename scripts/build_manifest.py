#!/usr/bin/env python
"""Scan mdCATH h5 files once and write a manifest.json (frame counts per trajectory).

Training then loads the manifest instead of opening all 5398 files at startup.
Run once after downloading data (or after each incremental download).

    python scripts/build_manifest.py --root /data/mdcath --out /data/mdcath/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from tqdm import tqdm

TEMPERATURES = ("320", "348", "379", "413", "450")


def scan_file(path: Path) -> dict | None:
    try:
        with h5py.File(path, "r") as f:
            dom = next(iter(f.keys()))
            g = f[dom]
            trajectories = []
            for temp in TEMPERATURES:
                if temp not in g:
                    continue
                for rep in g[temp]:
                    grp = g[temp][rep]
                    if "coords" in grp:
                        trajectories.append({
                            "temp": int(temp), "replica": int(rep),
                            "num_frames": int(grp.attrs["numFrames"]),
                        })
            return {
                "file": path.name,
                "domain": dom,
                "num_residues": int(g.attrs.get("numResidues", 0)),
                "num_atoms": int(g.attrs.get("numProteinAtoms", 0)),
                "trajectories": trajectories,
            }
    except Exception as e:  # skip corrupt/partial downloads, log at the end
        print(f"  [skip] {path.name}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing mdcath_dataset_*.h5 (recursive)")
    ap.add_argument("--out", default=None, help="output manifest path (default: <root>/manifest.json)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "manifest.json"
    files = sorted(root.rglob("mdcath_dataset_*.h5"))
    print(f"scanning {len(files)} files under {root}")

    manifest, n_traj = [], 0
    for f in tqdm(files):
        entry = scan_file(f)
        if entry and entry["trajectories"]:
            manifest.append(entry)
            n_traj += len(entry["trajectories"])

    out.write_text(json.dumps(manifest, indent=0))
    print(f"wrote {out}: {len(manifest)} domains, {n_traj} trajectories")


if __name__ == "__main__":
    main()
