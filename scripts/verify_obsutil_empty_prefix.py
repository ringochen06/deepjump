#!/usr/bin/env python
"""Fail closed unless an ``obsutil ls`` report proves a prefix is empty."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def prefix_object_count(text: str) -> int:
    """Return the parsed object count for supported ``obsutil ls`` formats."""
    object_match = re.search(r"Object number\s*(?:is)?\s*:\s*([0-9]+)", text)
    if object_match:
        return int(object_match.group(1))

    folder_match = re.search(r"Folder number\s*:\s*([0-9]+)", text)
    file_match = re.search(r"File number\s*:\s*([0-9]+)", text)
    if folder_match and file_match:
        return int(folder_match.group(1)) + int(file_match.group(1))
    raise ValueError("OBS prefix preflight did not return parseable object counts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    count = prefix_object_count(Path(args.report).read_text())
    if count != 0:
        raise SystemExit("refusing to reuse non-empty OBS evidence prefix")


if __name__ == "__main__":
    main()
