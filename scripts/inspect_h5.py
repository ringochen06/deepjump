#!/usr/bin/env python
"""Dump the structure of one mdCATH HDF5 file.

Run this BEFORE writing the dataloader so the real key hierarchy / dtypes /
attrs are confirmed against an actual file (not assumed from the paper).

Usage:
    python scripts/inspect_h5.py ~/hkucds/data/mdcath/data/mdcath_dataset_1a02F00.h5
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np


def _fmt_attrs(obj) -> str:
    items = []
    for k, v in obj.attrs.items():
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode(errors="replace")
        if isinstance(v, np.ndarray):
            v = f"ndarray{v.shape}:{v.dtype}"
        items.append(f"{k}={v}")
    return ("  attrs: " + ", ".join(items)) if items else ""


def walk(name: str, obj, max_depth: int, depth: int = 0) -> None:
    indent = "  " * depth
    if isinstance(obj, h5py.Group):
        print(f"{indent}[grp] {name.split('/')[-1] or '/'}")
        a = _fmt_attrs(obj)
        if a:
            print(f"{indent}{a}")
        if depth >= max_depth:
            print(f"{indent}  ... (truncated at depth {max_depth})")
            return
        for key in obj:
            walk(f"{name}/{key}", obj[key], max_depth, depth + 1)
    else:  # dataset
        print(f"{indent}[dat] {name.split('/')[-1]}  shape={obj.shape} dtype={obj.dtype}")
        a = _fmt_attrs(obj)
        if a:
            print(f"{indent}{a}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"file: {args.path}")
        a = _fmt_attrs(f)
        if a:
            print(a)
        for key in f:
            walk(key, f[key], args.max_depth)

        # Convenience: show a coords shape somewhere deep, if the expected
        # domain -> temperature -> replica -> coords layout holds.
        print("\n--- probing for coords ---")
        found = False

        def _probe(name, obj):
            nonlocal found
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == "coords":
                print(f"coords at '{name}': shape={obj.shape} dtype={obj.dtype}")
                found = True
                return True  # stop after first

        f.visititems(lambda n, o: _probe(n, o))
        if not found:
            print("no dataset named 'coords' found", file=sys.stderr)


if __name__ == "__main__":
    main()
