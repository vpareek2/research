"""Jaxtitan command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from jaxtitan import __version__
from jaxtitan.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaxtitan", description="JAX-native LM training contracts and tools.")
    parser.add_argument("--version", action="version", version=f"jaxtitan {__version__}")

    commands = parser.add_subparsers(dest="command")
    config_parser = commands.add_parser("config", help="Inspect and validate TOML configs.")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)

    check_parser = config_commands.add_parser("check", help="Validate a TOML config against Jaxtitan contracts.")
    check_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "config" and args.config_command == "check":
        spec = load_config(args.path)
        print(f"valid: {spec.run_id}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
