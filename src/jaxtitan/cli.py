"""Jaxtitan command-line interface."""

import argparse
from collections.abc import Sequence
import sys

from jaxtitan import __version__
from jaxtitan.config import load_config, run_spec_to_json
from jaxtitan.errors import JaxtitanError


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

    preflight_parser = run_commands.add_parser("preflight", help="Check whether a config is ready to run locally.")
    preflight_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")
    preflight_parser.add_argument("--json", action="store_true", help="Print preflight report JSON.")

    train_parser = run_commands.add_parser("train", help="Run a minimal local training loop from a TOML config.")
    train_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")
    train_parser.add_argument("--resume", action="store_true", help="Resume from the latest local checkpoint.")

    inspect_parser = run_commands.add_parser("inspect", help="Inspect local run artifacts.")
    inspect_parser.add_argument("run_dir", help="Path to a local Jaxtitan run directory.")
    inspect_parser.add_argument("--json", action="store_true", help="Print inspection JSON.")

    eval_parser = commands.add_parser("eval", help="Run deterministic evals over local artifacts.")
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)

    checkpoint_eval_parser = eval_commands.add_parser("checkpoint", help="Evaluate a retained checkpoint.")
    checkpoint_eval_parser.add_argument("run_dir", help="Path to a local Jaxtitan run directory.")
    checkpoint_eval_parser.add_argument("--checkpoint", required=True, help="'best', 'latest', or a checkpoint step.")
    checkpoint_eval_parser.add_argument("--json", action="store_true", help="Print checkpoint eval JSON.")

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
            from jaxtitan.data import prepared_dataset_manifest_to_json, validate_dataset_manifest

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
            from jaxtitan.services import initialize_run

            manifest = initialize_run(args.path)
            print(manifest.run_dir)
            return 0

        if args.command == "run" and args.run_command == "preflight":
            from jaxtitan.runtime.preflight import format_preflight_report, preflight_report_to_json, run_preflight

            report = run_preflight(args.path)
            if args.json:
                print(preflight_report_to_json(report))
            else:
                print(format_preflight_report(report))
            return 0

        if args.command == "run" and args.run_command == "train":
            from jaxtitan.runtime import run_training

            summary = run_training(args.path, resume=args.resume)
            print(summary.run_dir)
            return 0

        if args.command == "run" and args.run_command == "inspect":
            from jaxtitan.runtime.inspect import format_run_inspection, inspect_run, run_inspection_to_json

            inspection = inspect_run(args.run_dir)
            if args.json:
                print(run_inspection_to_json(inspection))
            else:
                print(format_run_inspection(inspection))
            return 0

        if args.command == "eval" and args.eval_command == "checkpoint":
            from jaxtitan.runtime.checkpoint_eval import (
                checkpoint_eval_to_json,
                evaluate_checkpoint,
                format_checkpoint_eval,
            )

            payload = evaluate_checkpoint(args.run_dir, args.checkpoint)
            if args.json:
                print(checkpoint_eval_to_json(payload))
            else:
                print(format_checkpoint_eval(payload))
            return 0
    except JaxtitanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
