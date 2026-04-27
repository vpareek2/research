"""
Summarize timing metrics for a completed run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import shlex
import shutil
from statistics import mean, median
import subprocess
import sys
from typing import Any

from config import load_config


DEFAULT_WARMUP_STEPS = 20

PRIMARY_KEYS = [
    "time/data_sec",
    "time/batch_log_sec",
    "time/shard_sec",
    "time/train_step_sec",
    "time/metrics_sync_sec",
    "time/eval_sec",
    "time/sample_sec",
    "time/checkpoint_sec",
    "time/log_sec",
    "time/step_sec",
    "time/tokens_per_sec",
    "time/train_tokens_per_sec",
]


@dataclass(frozen=True)
class FieldStats:
    count: int
    mean: float
    p50: float
    p95: float
    max: float


@dataclass(frozen=True)
class NsysPlan:
    config_path: Path
    run_dir: Path
    temp_output_prefix: Path
    temp_report_path: Path
    final_report_path: Path
    command: list[str]


def load_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics rows found in {path}")
    return rows


def summarize_metrics(rows: list[dict[str, Any]], *, warmup_steps: int = DEFAULT_WARMUP_STEPS) -> dict[str, Any]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    used_rows = [row for row in rows if int(row.get("step", -1)) >= warmup_steps]
    if not used_rows:
        raise ValueError(f"No metrics rows remain after warmup_steps={warmup_steps}")

    groups = {
        "all": used_rows,
        "normal": [row for row in used_rows if "val/loss" not in row and "sample/path" not in row and "time/checkpoint_sec" not in row],
        "eval": [row for row in used_rows if "val/loss" in row],
        "sample": [row for row in used_rows if "sample/path" in row or "time/sample_sec" in row],
        "checkpoint": [row for row in used_rows if "time/checkpoint_sec" in row],
    }

    timing_keys = _ordered_timing_keys(used_rows)
    return {
        "warmup_steps": warmup_steps,
        "rows_total": len(rows),
        "rows_used": len(used_rows),
        "step_min": min(int(row["step"]) for row in used_rows),
        "step_max": max(int(row["step"]) for row in used_rows),
        "groups": {
            name: {
                "rows": len(group_rows),
                "fields": _summarize_fields(group_rows, timing_keys),
            }
            for name, group_rows in groups.items()
            if group_rows
        },
    }


def _ordered_timing_keys(rows: list[dict[str, Any]]) -> list[str]:
    available = {key for row in rows for key in row if key.startswith("time/")}
    ordered = [key for key in PRIMARY_KEYS if key in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _summarize_fields(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float | int]]:
    fields = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            fields[key] = _stats(values).__dict__
    return fields


def _stats(values: list[float]) -> FieldStats:
    ordered = sorted(values)
    p95_index = max(0, ceil(0.95 * len(ordered)) - 1)
    return FieldStats(
        count=len(values),
        mean=mean(values),
        p50=median(values),
        p95=ordered[p95_index],
        max=max(values),
    )


def format_summary(run_dir: str | Path, summary: dict[str, Any]) -> str:
    lines = [
        "Timing Summary",
        f"  run:          {Path(run_dir)}",
        f"  warmup_steps: {summary['warmup_steps']}",
        f"  rows:         {summary['rows_used']} / {summary['rows_total']}",
        f"  step_range:   {summary['step_min']}..{summary['step_max']}",
        "",
    ]
    for group_name, group in summary["groups"].items():
        lines.append(f"{group_name} steps ({group['rows']} rows)")
        lines.append(f"{'field':<30} {'count':>7} {'mean':>12} {'p50':>12} {'p95':>12} {'max':>12}")
        lines.append("-" * 89)
        for field, stats in group["fields"].items():
            lines.append(
                f"{field:<30} "
                f"{stats['count']:>7} "
                f"{_format_number(stats['mean']):>12} "
                f"{_format_number(stats['p50']):>12} "
                f"{_format_number(stats['p95']):>12} "
                f"{_format_number(stats['max']):>12}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_number(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    if abs(value) >= 1.0:
        return f"{value:.4f}"
    return f"{value:.6f}"


def write_summary(run_dir: str | Path, summary: dict[str, Any], markdown: str, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    profile_dir = Path(output_dir) if output_dir is not None else Path(run_dir) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    json_path = profile_dir / "timing_summary.json"
    md_path = profile_dir / "timing_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def run_summary(run_dir: str | Path, *, warmup_steps: int = DEFAULT_WARMUP_STEPS, output_dir: str | Path | None = None) -> tuple[dict[str, Any], Path, Path, str]:
    rows = load_metrics(run_dir)
    summary = summarize_metrics(rows, warmup_steps=warmup_steps)
    markdown = format_summary(run_dir, summary)
    json_path, md_path = write_summary(run_dir, summary, markdown, output_dir)
    return summary, json_path, md_path, markdown


def build_nsys_plan(
    config_path: str | Path,
    *,
    nsys_output_dir: str | Path = "profiles",
    extra_nsys_args: list[str] | None = None,
) -> NsysPlan:
    config_path = Path(config_path)
    config = load_config(config_path)
    if not config.profiling.enabled or config.profiling.profiler != "nsys":
        raise ValueError("profile nsys requires [profiling] enabled = true and profiler = \"nsys\"")

    run_dir = Path(config.experiment.out_dir) / config.experiment.name
    temp_output_prefix = Path(nsys_output_dir) / config.experiment.name
    temp_report_path = temp_output_prefix.with_suffix(".nsys-rep")
    final_report_path = run_dir / "profiles" / "nsys.nsys-rep"
    command = [
        "nsys",
        "profile",
        "--force-overwrite",
        "true",
        "-t",
        "cuda,nvtx,osrt,cudnn,cublas",
    ]
    command.extend(extra_nsys_args or [])
    command.extend(
        [
            "-o",
            str(temp_output_prefix),
            "uv",
            "run",
            "pretrain",
            str(config_path),
        ]
    )
    return NsysPlan(
        config_path=config_path,
        run_dir=run_dir,
        temp_output_prefix=temp_output_prefix,
        temp_report_path=temp_report_path,
        final_report_path=final_report_path,
        command=command,
    )


def run_nsys(
    config_path: str | Path,
    *,
    force_run_dir: bool = False,
    dry_run: bool = False,
    nsys_output_dir: str | Path = "profiles",
    extra_nsys_args: list[str] | None = None,
    runner=subprocess.run,
) -> NsysPlan:
    plan = build_nsys_plan(config_path, nsys_output_dir=nsys_output_dir, extra_nsys_args=extra_nsys_args)
    if dry_run:
        return plan

    if plan.run_dir.exists():
        if not force_run_dir:
            raise FileExistsError(f"Run directory already exists: {plan.run_dir}. Use --force-run-dir to remove it first.")
        shutil.rmtree(plan.run_dir)

    plan.temp_output_prefix.parent.mkdir(parents=True, exist_ok=True)

    runner(plan.command, check=True)
    if not plan.temp_report_path.exists():
        raise FileNotFoundError(f"NSys did not produce expected report: {plan.temp_report_path}")

    plan.final_report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plan.temp_report_path, plan.final_report_path)
    return plan


def print_summary(run_dir: str | Path, *, warmup_steps: int, output_dir: str | Path | None):
    _, json_path, md_path, markdown = run_summary(run_dir, warmup_steps=warmup_steps, output_dir=output_dir)
    print(markdown)
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


def print_nsys_plan(plan: NsysPlan, *, dry_run: bool):
    print("NSys Profile")
    print(f"  config:       {plan.config_path}")
    print(f"  run_dir:      {plan.run_dir}")
    print(f"  temp_report:  {plan.temp_report_path}")
    print(f"  final_report: {plan.final_report_path}")
    print(f"  command:      {shlex.join(plan.command)}")
    if dry_run:
        print("dry run: command not executed")
    else:
        print(f"wrote: {plan.final_report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile and summarize training runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Summarize timing metrics from a completed run.")
    summary.add_argument("run_dir", help="Run directory containing metrics.jsonl.")
    summary.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Ignore metric rows before this step.")
    summary.add_argument("--output-dir", help="Directory for timing_summary.json and timing_summary.md. Defaults to RUN_DIR/profiles.")

    nsys = subparsers.add_parser("nsys", help="Run Nsight Systems around a profiling config.")
    nsys.add_argument("config", help="Config TOML with [profiling] profiler = \"nsys\".")
    nsys.add_argument("--force-run-dir", action="store_true", help="Remove the configured run directory before launching.")
    nsys.add_argument("--nsys-output-dir", default="profiles", help="Temporary directory for NSys reports outside runs/.")
    nsys.add_argument("--dry-run", action="store_true", help="Print resolved paths and command without executing NSys.")
    nsys.add_argument("--extra-nsys-arg", action="append", default=[], help="Additional argument passed to nsys profile. Repeat as needed.")
    return parser


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"summary", "nsys", "-h", "--help"}:
        argv = ["summary", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        print_summary(args.run_dir, warmup_steps=args.warmup_steps, output_dir=args.output_dir)
        return
    if args.command == "nsys":
        plan = run_nsys(
            args.config,
            force_run_dir=args.force_run_dir,
            dry_run=args.dry_run,
            nsys_output_dir=args.nsys_output_dir,
            extra_nsys_args=args.extra_nsys_arg,
        )
        print_nsys_plan(plan, dry_run=args.dry_run)
        return


if __name__ == "__main__":
    main()
