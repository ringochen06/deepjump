#!/usr/bin/env python
"""Fail closed unless an ``obsutil ls`` report proves a prefix is empty."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def prefix_object_count(text: str) -> int:
    """Return the parsed object count for supported ``obsutil ls`` formats."""
    count_names = ("Object number", "Folder number", "File number")
    counts: dict[str, list[int]] = {name: [] for name in count_names}
    pattern = re.compile(
        r"(Object number|Folder number|File number)\s*(?:is)?\s*:\s*([0-9]+)"
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not any(name in line for name in count_names):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError("OBS prefix preflight contains a malformed count line")
        counts[match.group(1)].append(int(match.group(2)))

    objects = counts["Object number"]
    folders = counts["Folder number"]
    files = counts["File number"]
    if len(objects) == 1 and not folders and not files:
        return objects[0]
    if not objects and len(folders) == len(files) == 1:
        return folders[0] + files[0]
    raise ValueError("OBS prefix preflight requires one unique, unmixed count format")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    count = prefix_object_count(Path(args.report).read_text())
    if count != 0:
        raise SystemExit("refusing to reuse non-empty OBS evidence prefix")


if __name__ == "__main__":
    main()
