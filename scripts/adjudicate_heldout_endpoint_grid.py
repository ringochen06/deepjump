#!/usr/bin/env python
"""Fail-closed adjudication for the frozen held-out H1 endpoint grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.adjudicate_endpoint_grid import _adjudicate_grid


EXPECTED_DOMAIN_ID = "1gxlA02"
EXPECTED_RESIDUE_COUNT = 86


def adjudicate(
    result_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    domain_list_sha256: str,
) -> dict:
    return _adjudicate_grid(
        result_path,
        checkpoint_path,
        checkpoint_sha256,
        domain_list_sha256,
        expected_domain_id=EXPECTED_DOMAIN_ID,
        expected_residue_count=EXPECTED_RESIDUE_COUNT,
        pass_status="PASS_HELDOUT_ENDPOINT_GRID",
        null_status="STOP_NULL_HELDOUT_ENDPOINT_GRID",
        nonphysical_status="STOP_NONPHYSICAL_HELDOUT_ENDPOINT_GRID",
        scope="one-held-out-domain 5x5-cell clean-source H1 endpoint discriminator only",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result,
        args.checkpoint,
        args.checkpoint_sha256,
        args.domain_list_sha256,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
