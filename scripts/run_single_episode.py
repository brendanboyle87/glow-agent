#!/usr/bin/env python3
"""Run one scaffolded episode from the command line.

TODO: keep this script thin and push real logic into the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zork_agent.cli import run_single_command


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for single-episode runs."""

    parser = argparse.ArgumentParser(description="Run one scaffolded Zork episode.")
    parser.add_argument("config", type=Path, help="Path to a YAML config file.")
    parser.add_argument("--episode-id", type=str, default=None, help="Optional explicit episode id.")
    return parser


def main() -> None:
    """Parse arguments, execute the runner, and print JSON."""

    args = build_parser().parse_args()
    result = run_single_command(args.config, episode_id=args.episode_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

