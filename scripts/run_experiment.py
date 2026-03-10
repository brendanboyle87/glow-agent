#!/usr/bin/env python3
"""Run a scaffolded batch experiment from the command line.

TODO: replace this thin wrapper only if script-specific automation needs diverge from the CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zork_agent.cli import run_batch_command


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for batch runs."""

    parser = argparse.ArgumentParser(description="Run a batch of scaffolded Zork episodes.")
    parser.add_argument("config", type=Path, help="Path to a YAML config file.")
    parser.add_argument("--episodes", type=int, default=None, help="Optional episode count override.")
    return parser


def main() -> None:
    """Parse arguments, execute the batch runner, and print JSON."""

    args = build_parser().parse_args()
    result = run_batch_command(args.config, episodes=args.episodes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

