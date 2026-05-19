"""Jaxtitan command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from jaxtitan import __version__
from jaxtitan.config import load_config, run_spec_to_json
from jaxtitan.errors import JaxtitanError
from jaxtitan.services import initialize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaxtitan", description="JAX-native LM training contracts and tools.")
    parser.add_argument("--version", action="version", version=f"jaxtitan {__version__}")

    commands = parser.add_subparsers(dest="command")
    config_parser = commands.add_parser("config", help="Inspect and validate TOML configs.")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)

    check_parser = config_commands.add_parser("check", help="Validate a TOML config against Jaxtitan contracts.")
    check_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")
    check_parser.add_argument("--json", action="store_true", help="Print resolved RunSpec JSON.")

    run_parser = commands.add_parser("run", help="Create and inspect local run artifacts.")
    run_commands = run_parser.add_subparsers(dest="run_command", required=True)

    init_parser = run_commands.add_parser("init", help="Initialize a local run directory from a TOML config.")
    init_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "config" and args.config_command == "check":
            spec = load_config(args.path)
            if args.json:
                print(run_spec_to_json(spec))
            else:
                print(f"valid: {spec.run_id}")
            return 0

        if args.command == "run" and args.run_command == "init":
            manifest = initialize_run(args.path)
            print(manifest.run_dir)
            return 0
    except JaxtitanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
