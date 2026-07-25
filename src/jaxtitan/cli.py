"""Jaxtitan command-line interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from jaxtitan import __version__
from jaxtitan.config import load_config, run_spec_to_json
from jaxtitan.errors import ContractError, JaxtitanError


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

    data_prepare_parser = data_commands.add_parser("prepare", help="Prepare a text dataset for training.")
    data_prepare_parser.add_argument("path", help="Path to a Jaxtitan data-prepare TOML config.")
    data_prepare_parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    data_prepare_parser.add_argument("--json", action="store_true", help="Print prepared manifest summary JSON.")

    data_check_parser = data_commands.add_parser("check", help="Validate a prepared-token manifest.")
    data_check_parser.add_argument("path", help="Path to a prepared-token manifest JSON file.")
    data_check_parser.add_argument("--tokenizer", required=True, help="Expected tokenizer id.")
    data_check_parser.add_argument("--verify-checksums", action="store_true", help="Verify shard and token-byte checksums.")
    data_check_parser.add_argument("--json", action="store_true", help="Print validated manifest JSON.")

    data_inspect_parser = data_commands.add_parser("inspect", help="Inspect a prepared-token manifest.")
    data_inspect_parser.add_argument("path", help="Path to a prepared-token manifest JSON file.")
    data_inspect_parser.add_argument("--tokenizer", help="Expected tokenizer id.")
    data_inspect_parser.add_argument("--verify-checksums", action="store_true", help="Verify shard and token-byte checksums.")
    data_inspect_parser.add_argument("--seq-len", type=int, help="Report record counts for this sequence length.")
    data_inspect_parser.add_argument("--json", action="store_true", help="Print inspection JSON.")

    kernels_parser = commands.add_parser("kernels", help="Inspect Jaxtitan kernel backend plans.")
    kernels_commands = kernels_parser.add_subparsers(dest="kernels_command", required=True)

    kernels_list_parser = kernels_commands.add_parser("list", help="List Jaxtitan-owned kernel candidates.")
    kernels_list_parser.add_argument("--json", action="store_true", help="Print kernel registry JSON.")

    kernels_check_parser = kernels_commands.add_parser("check", help="Resolve the kernel plan for a TOML config.")
    kernels_check_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")
    kernels_check_parser.add_argument("--cache-dir", help="Kernel cache directory.")
    kernels_check_parser.add_argument("--json", action="store_true", help="Print kernel plan JSON.")

    kernels_compile_parser = kernels_commands.add_parser("compile", help="Compile buildable kernels for a TOML config.")
    kernels_compile_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")
    kernels_compile_parser.add_argument("--arch", help="Kernel architecture, such as SM90 or SM121.")
    kernels_compile_parser.add_argument("--cache-dir", help="Kernel cache directory.")
    kernels_compile_parser.add_argument("--json", action="store_true", help="Print compiled kernel plan JSON.")

    kernels_bench_parser = kernels_commands.add_parser("bench", help="Benchmark a compiled Jaxtitan kernel.")
    kernels_bench_parser.add_argument("op", choices=["rmsnorm"], help="Kernel operation to benchmark.")
    kernels_bench_parser.add_argument("path", help="Path to a Jaxtitan TOML config.")
    kernels_bench_parser.add_argument("--cache-dir", help="Kernel cache directory.")
    kernels_bench_parser.add_argument("--rows", default="1,4,17,64,256", help="Comma-separated row counts.")
    kernels_bench_parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per shape.")
    kernels_bench_parser.add_argument("--iters", type=int, default=20, help="Measured iterations per shape.")
    kernels_bench_parser.add_argument("--json", action="store_true", help="Print benchmark JSON.")

    profile_parser = commands.add_parser("profile", help="Analyze and benchmark Jaxtitan performance paths.")
    profile_commands = profile_parser.add_subparsers(dest="profile_command", required=True)

    profile_analyze_parser = profile_commands.add_parser(
        "analyze",
        help="Analyze completed profiling artifacts below a directory.",
    )
    profile_analyze_parser.add_argument("root", help="Run directory or capture root to analyze recursively.")
    profile_analyze_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Initial steps excluded before steady-state analysis.",
    )
    profile_analyze_parser.add_argument("--json", action="store_true", help="Print stable analysis JSON.")

    profile_bench_parser = profile_commands.add_parser(
        "bench",
        help="Run a deterministic local performance microbenchmark.",
    )
    profile_bench_parser.add_argument("component", choices=["moe", "muon"], help="Component to benchmark.")
    profile_bench_parser.add_argument("--warmup", type=int, default=3, help="Warmup executions per case.")
    profile_bench_parser.add_argument("--iters", type=int, default=10, help="Measured executions per case.")
    profile_bench_parser.add_argument(
        "--artifact-dir",
        help="Optional Muon benchmark directory for canonical JSON, optimized HLO, and profiler artifacts.",
    )
    profile_bench_parser.add_argument(
        "--trace",
        action="store_true",
        help="Capture a profiler trace while timing; requires --artifact-dir and is not for canonical selection timing.",
    )
    profile_bench_parser.add_argument("--json", action="store_true", help="Print stable benchmark JSON.")

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
    train_parser.add_argument("--overwrite", action="store_true", help="Delete an existing fresh-run directory before training.")

    inspect_parser = run_commands.add_parser("inspect", help="Inspect local run artifacts.")
    inspect_parser.add_argument("run_dir", help="Path to a local Jaxtitan run directory.")
    inspect_parser.add_argument("--json", action="store_true", help="Print inspection JSON.")

    eval_parser = commands.add_parser("eval", help="Run deterministic evals over local artifacts.")
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)

    checkpoint_eval_parser = eval_commands.add_parser("checkpoint", help="Evaluate a retained checkpoint.")
    checkpoint_eval_parser.add_argument("run_dir", help="Path to a local Jaxtitan run directory.")
    checkpoint_eval_parser.add_argument("--checkpoint", required=True, help="'best', 'latest', or a checkpoint step.")
    checkpoint_eval_parser.add_argument("--json", action="store_true", help="Print checkpoint eval JSON.")

    sample_parser = commands.add_parser("sample", help="Generate token samples from local artifacts.")
    sample_commands = sample_parser.add_subparsers(dest="sample_command", required=True)

    checkpoint_sample_parser = sample_commands.add_parser("checkpoint", help="Sample token ids from a retained checkpoint.")
    checkpoint_sample_parser.add_argument("run_dir", help="Path to a local Jaxtitan run directory.")
    checkpoint_sample_parser.add_argument("--checkpoint", required=True, help="'best', 'latest', or a checkpoint step.")
    checkpoint_sample_parser.add_argument("--prompt-ids", required=True, help="Comma-separated prompt token ids.")
    checkpoint_sample_parser.add_argument("--max-new-tokens", required=True, type=int, help="Number of tokens to generate.")
    checkpoint_sample_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    checkpoint_sample_parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    checkpoint_sample_parser.add_argument("--json", action="store_true", help="Print checkpoint sample JSON.")

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

        if args.command == "data" and args.data_command == "prepare":
            from jaxtitan.data import format_prepare_result, prepare_dataset, prepare_result_to_json

            result = prepare_dataset(args.path, overwrite=args.overwrite, quiet=args.json)
            if args.json:
                print(prepare_result_to_json(result))
            else:
                print(format_prepare_result(result))
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

        if args.command == "data" and args.data_command == "inspect":
            from jaxtitan.data import data_inspection_to_json, format_data_inspection, inspect_dataset_manifest

            inspection = inspect_dataset_manifest(
                args.path,
                tokenizer_id=args.tokenizer,
                verify_checksums=args.verify_checksums,
                seq_len=args.seq_len,
            )
            if args.json:
                print(data_inspection_to_json(inspection))
            else:
                print(format_data_inspection(inspection))
            return 0

        if args.command == "kernels" and args.kernels_command == "list":
            from jaxtitan.kernels import format_kernel_registry, kernel_plan_to_json, kernel_registry_payload

            payload = kernel_registry_payload()
            if args.json:
                print(kernel_plan_to_json(payload))
            else:
                print(format_kernel_registry(payload))
            return 0

        if args.command == "kernels" and args.kernels_command == "check":
            from jaxtitan.kernels import (
                enrich_kernel_plan_with_cache,
                format_kernel_plan,
                kernel_plan,
                kernel_plan_to_json,
                require_kernel_plan_supported,
            )

            spec = load_config(args.path)
            plan = enrich_kernel_plan_with_cache(kernel_plan(spec), root=args.cache_dir)
            require_kernel_plan_supported(plan)
            if args.json:
                print(kernel_plan_to_json(plan))
            else:
                print(format_kernel_plan(plan))
            return 0

        if args.command == "kernels" and args.kernels_command == "compile":
            from jaxtitan.kernels import compile_kernel_plan, format_compile_result, kernel_plan_to_json

            spec = load_config(args.path)
            plan = compile_kernel_plan(spec, arch=args.arch, root=args.cache_dir)
            if args.json:
                print(kernel_plan_to_json(plan))
            else:
                print(format_compile_result(plan))
            return 0

        if args.command == "kernels" and args.kernels_command == "bench":
            from jaxtitan.kernels.bench import (
                benchmark_rmsnorm,
                benchmark_to_json,
                format_rmsnorm_benchmark,
                parse_rows,
            )

            spec = load_config(args.path)
            rows = parse_rows(args.rows)
            payload = benchmark_rmsnorm(
                spec,
                cache_root=args.cache_dir,
                rows=rows,
                warmup=args.warmup,
                iters=args.iters,
            )
            if args.json:
                print(benchmark_to_json(payload))
            else:
                print(format_rmsnorm_benchmark(payload))
            return 0

        if args.command == "profile" and args.profile_command == "analyze":
            from jaxtitan.runtime.profile_analysis import (
                analyze_profile_root,
                format_profile_analysis,
                profile_analysis_to_json,
            )

            payload = analyze_profile_root(args.root, warmup_steps=args.warmup_steps)
            if args.json:
                print(profile_analysis_to_json(payload))
            else:
                print(format_profile_analysis(payload))
            return 0

        if args.command == "profile" and args.profile_command == "bench":
            from jaxtitan.runtime.profile_bench import benchmark_component, benchmark_to_json, format_benchmark

            payload = benchmark_component(
                args.component,
                warmup=args.warmup,
                iters=args.iters,
                artifact_dir=args.artifact_dir,
                trace=args.trace,
            )
            if args.json:
                print(benchmark_to_json(payload))
            else:
                print(format_benchmark(payload))
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

            if not args.resume:
                _prepare_fresh_train_run(args.path, overwrite=args.overwrite)
            run_training(args.path, resume=args.resume, progress=_training_progress_printer())
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

        if args.command == "sample" and args.sample_command == "checkpoint":
            from jaxtitan.runtime.sampling import (
                checkpoint_sample_to_json,
                format_checkpoint_sample,
                sample_checkpoint,
            )

            payload = sample_checkpoint(
                args.run_dir,
                args.checkpoint,
                args.prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            if args.json:
                print(checkpoint_sample_to_json(payload))
            else:
                print(format_checkpoint_sample(payload))
            return 0
    except JaxtitanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


def _prepare_fresh_train_run(config_path: str | Path, *, overwrite: bool) -> None:
    spec = load_config(config_path)
    run_dir = spec.dirs.run_dir
    if not run_dir.exists():
        return
    if overwrite:
        shutil.rmtree(run_dir)
        return
    if not sys.stdin.isatty():
        raise ContractError(
            f"run directory already exists: {run_dir}; rerun with --overwrite to replace it"
        )
    print(f"run directory already exists: {run_dir}", file=sys.stderr)
    answer = input("Overwrite this run directory? [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        raise ContractError(f"run directory already exists: {run_dir}")
    shutil.rmtree(run_dir)


def _training_progress_printer():
    state: dict[str, Any] = {}

    def print_progress(event: str, payload: dict[str, Any]) -> None:
        if event == "start":
            spec = payload["spec"]
            config_path = Path(payload["config_path"])
            state["started_at"] = time.perf_counter()
            print("=========================")
            print("JAX TITAN TRAINING")
            print("=========================")
            print(f"run: {spec.run_id}")
            print(f"config: {config_path.as_posix()}")
            print(f"resume: {str(payload['resume']).lower()}")
            print("initializing runtime...")
            return

        if event == "initialized":
            spec = payload["spec"]
            diagnostics = payload["diagnostics"]
            model = diagnostics["model"]
            mesh = diagnostics["mesh"]
            jax_info = diagnostics["jax"]
            data = diagnostics["data_pipeline"]
            print("")
            print("config:")
            print(
                "  model   | "
                f"{spec.model.name}/{spec.model.variant} | params {_format_count(model['parameters'])} | "
                f"seq {spec.training.seq_len} | dtype {spec.model.compute_dtype}/{spec.model.param_dtype} | "
                f"remat {spec.model.remat}"
            )
            print(
                "  train   | "
                f"target {_format_int(spec.training.target_tokens)} | "
                f"batch {spec.training.global_batch_size} x accum {spec.training.gradient_accumulation_steps} "
                f"= {mesh['effective_global_batch_size']} | "
                f"tokens/step {_format_int(mesh['effective_tokens_per_step'])}"
            )
            print(
                "  optim   | "
                f"{spec.optimizer.name} | schedule {spec.optimizer.schedule.name} | "
                f"peak_lr {_format_number(spec.optimizer.schedule.peak_lr)}"
            )
            print(
                "  data    | "
                f"{data['backend']} | order {data['order']} | docs {_format_bool(data['document_aware'])} | "
                f"records {_format_int(data['num_records'])} | {data['manifest_path']}"
            )
            print(
                "  mesh    | "
                f"{jax_info['backend']} | devices {jax_info['selected_device_count']} | "
                f"axes {_format_mesh_axes(mesh)}"
            )
            print(f"  run_dir | {payload['run_dir']}")
            print("-------------------------")
            return

        if event == "resumed":
            print(
                "resumed: "
                f"checkpoint_step={payload['checkpoint_step']} "
                f"step={payload['step']} tokens={_format_int(payload['tokens_seen'])} "
                f"path={payload['checkpoint_path']}"
            )
            return

        if event == "compile_start":
            phase = payload["phase"]
            if phase == "train":
                print(f"step: {0:<6} | compiling train step...")
            else:
                print(f"{phase}: compiling step...")
            return

        if event == "train":
            row = payload["row"]
            started_at = float(state.get("started_at", time.perf_counter()))
            total_time = time.perf_counter() - started_at
            print(
                f"step: {row['step']:<6} | "
                f"loss: {_format_metric(row['loss']):>9} | "
                f"grad_norm: {_format_metric(row.get('grad_norm')):>9} | "
                f"mfu: {_format_percent(row.get('mfu')):>7} | "
                f"lr: {_format_number(row['lr']):>9} | "
                f"tps: {_format_tps(row.get('train_tokens_per_sec')):>9} | "
                f"total_time: {_format_runtime(total_time):>9}"
            )
            return

        if event == "eval":
            row = payload["row"]
            print(
                f"eval step {row['step']}: "
                f"loss={_format_metric(row['loss'])} "
                f"tokens={_format_int(row['token_count'])} "
                f"batches={row['num_batches']} "
                f"tps={_format_metric(row.get('eval_tokens_per_sec'))} "
                f"sec={_format_metric(row.get('eval_sec'))}"
            )
            return

        if event == "checkpoint":
            eval_loss = payload.get("eval_loss")
            eval_part = "" if eval_loss is None else f" eval_loss={_format_metric(eval_loss)}"
            print(
                f"checkpoint step {payload['step']}: "
                f"{payload['checkpoint_path']} reason={payload['reason']} "
                f"sec={_format_metric(payload['checkpoint_sec'])}{eval_part}"
            )
            return

        if event == "completed":
            summary = payload["summary"]
            eval_part = "" if summary.final_eval_loss is None else f" final_eval={_format_metric(summary.final_eval_loss)}"
            print("-------------------------")
            print(
                "completed: "
                f"step={summary.steps} "
                f"tokens={_format_int(summary.tokens_seen)}/{_format_int(summary.target_tokens)} "
                f"loss={_format_metric(summary.final_loss)}{eval_part} "
                f"steady_tps={_format_metric(summary.steady_train_tokens_per_sec)} "
                f"mfu={_format_percent(summary.final_mfu)}"
            )
            print(f"run_dir: {summary.run_dir}")

    return print_progress


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_mesh_axes(mesh: dict[str, Any]) -> str:
    return ",".join(
        f"{name}={size}"
        for name, size in zip(mesh["axis_names"], mesh["axis_sizes"], strict=True)
    )


def _format_count(value: int | float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(int(value))


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 100_000:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _format_tps(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10_000:
        return f"{value:.3e}"
    return f"{value:.6g}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{_format_metric(value)}s"


def _format_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


if __name__ == "__main__":
    raise SystemExit(main())
