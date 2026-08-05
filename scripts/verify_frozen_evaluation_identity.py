#!/usr/bin/env python
"""CLI for the contracted external/untouched evaluation identity gate."""

from __future__ import annotations

import argparse
import json

from deepjump.evaluation_contract import verify_frozen_evaluation_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step", required=True, type=int)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument(
        "--phase", choices=("development", "external", "untouched"), required=True
    )
    parser.add_argument("--panel-name", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify_frozen_evaluation_identity(
        args.checkpoint,
        args.contract,
        args.expected_contract_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_step=args.expected_checkpoint_step,
        phase=args.phase,
        panel_name=args.panel_name,
        panel_file=args.panel_file,
    )
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
