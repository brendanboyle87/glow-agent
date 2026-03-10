#!/usr/bin/env python3
"""Inspect a saved JSONL trajectory from the command line.

TODO: add richer formatting only when real trajectories make this worthwhile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zork_agent.cli import inspect_trajectory_command


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for replay inspection."""

    parser = argparse.ArgumentParser(description="Inspect a scaffold trajectory JSONL file.")
    parser.add_argument("trajectory", type=Path, help="Path to a trajectory JSONL file.")
    return parser


def main() -> None:
    """Parse arguments, inspect the trajectory, and print a summary."""

    args = build_parser().parse_args()
    print(inspect_trajectory_command(args.trajectory))


if __name__ == "__main__":
    main()

