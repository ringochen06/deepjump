#!/usr/bin/env python
"""Build, but never authorize, one exact closer-to-paper formal500k package."""

from __future__ import annotations

import argparse
from pathlib import Path

from formal500k_package_lib import atomic_write_json, build_package_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = args.spec.resolve()
    output = args.output.resolve()
    if spec == output:
        raise ValueError("package output must be separate from its reviewed spec")
    atomic_write_json(output, build_package_payload(spec))


if __name__ == "__main__":
    main()
