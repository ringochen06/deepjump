#!/usr/bin/env python
"""Audit frozen per-delta/pathwise eligibility from an mdCATH manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepjump.evaluation import audit_manifest_eligibility, load_frozen_domain_ids


def _integers(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _nonnegative_integers(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--domain-list", required=True)
    parser.add_argument("--domain-list-sha256", required=True)
    parser.add_argument("--temperatures", type=_integers, default=_integers("320,348,379,413,450"))
    parser.add_argument(
        "--replicas", type=_nonnegative_integers,
        default=_nonnegative_integers("0,1,2,3,4"),
    )
    parser.add_argument("--deltas", type=_integers, default=_integers("1,10,100"))
    parser.add_argument("--unrolls", type=_integers, default=_integers("1,20,100"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    domain_ids, digest = load_frozen_domain_ids(
        args.domain_list, args.domain_list_sha256
    )
    manifest = json.loads(Path(args.manifest).expanduser().read_text())
    audits = [
        audit_manifest_eligibility(
            manifest, domain_ids, args.temperatures, args.replicas,
            delta=delta, unroll=unroll,
        )
        for delta in args.deltas for unroll in args.unrolls
    ]
    result = {
        "domain_panel": {
            "path": args.domain_list,
            "sha256": digest,
            "count": len(domain_ids),
        },
        "temperatures": args.temperatures,
        "replicas": args.replicas,
        "audits": audits,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
