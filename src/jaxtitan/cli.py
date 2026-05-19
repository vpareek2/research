"""Jaxtitan command-line interface."""

import argparse
from collections.abc import Sequence
import sys

from jaxtitan import __version__
from jaxtitan.config import load_config, run_spec_to_json
from jaxtitan.data import prepared_dataset_manifest_to_json, validate_dataset_manifest
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

    data_parser = commands.add_parser("data", help="Inspect and validate prepared data artifacts.")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)

    data_check_parser = data_commands.add_parser("check", help="Validate a prepared-token manifest.")
    data_check_parser.add_argument("path", help="Path to a prepared-token manifest JSON file.")
    data_check_parser.add_argument("--tokenizer", required=True, help="Expected tokenizer id.")
    data_check_parser.add_argument("--verify-checksums", action="store_true", help="Verify shard and token-byte checksums.")
    data_check_parser.add_argument("--json", action="store_true", help="Print validated manifest JSON.")

    run_parser = commands.add_parser("run", help="Create and inspect local run artifacts.")
    run_commands = run_parser.add_subparsers(dest="run_command", required=True)

    init_parser = run_commands.add_parser("init", help="Initialize a local run directory from a TOML config.")
    init_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")

    train_parser = run_commands.add_parser("train", help="Run a minimal local training loop from a TOML config.")
    train_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")

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

        if args.command == "data" and args.data_command == "check":
            manifest = validate_dataset_manifest(
                args.path,
                tokenizer_id=args.tokenizer,
                verify_checksums=args.verify_checksums,
            )
            if args.json:
                print(prepared_dataset_manifest_to_json(manifest))
            else:
                print(f"valid: {manifest.manifest_path} tokens={manifest.num_tokens}")
            return 0

        if args.command == "run" and args.run_command == "init":
            manifest = initialize_run(args.path)
            print(manifest.run_dir)
            return 0

        if args.command == "run" and args.run_command == "train":
            from jaxtitan.runtime import run_training

            summary = run_training(args.path)
            print(summary.run_dir)
            return 0
    except JaxtitanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
