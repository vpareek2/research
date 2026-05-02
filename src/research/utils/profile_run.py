"""
Summarize timing metrics for a completed run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import shlex
import shutil
import sqlite3
from statistics import mean, median
import subprocess
import sys
from typing import Any

from research.config import load_config


DEFAULT_WARMUP_STEPS = 20
REPORT_VERSION = 1

NSYS_PHASES = [
    "step",
    "data",
    "batch_log",
    "shard",
    "train_step",
    "metrics_sync",
    "eval",
    "sample",
    "checkpoint",
    "log",
]

CUDA_SYNC_RUNTIME_NAMES = {
    "cuStreamSynchronize",
    "cuEventSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
}

CUDA_MEMCPY_RUNTIME_NAMES = {
    "cuMemcpyDtoHAsync_v2",
    "cuMemcpyHtoDAsync_v2",
    "cudaMemcpyAsync",
    "cudaMemcpy",
}

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


@dataclass(frozen=True)
class AnalysisResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    markdown: str


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


def run_analysis(
    run_dir: str | Path,
    *,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    output_dir: str | Path | None = None,
    include_nsys: bool = True,
    runner=subprocess.run,
    nsys_executable: str | None = "auto",
) -> AnalysisResult:
    run_dir = Path(run_dir)
    rows = load_metrics(run_dir)
    timing_summary = summarize_metrics(rows, warmup_steps=warmup_steps)
    warnings: list[str] = []
    profile_dir = Path(output_dir) if output_dir is not None else run_dir / "profiles"
    inputs: dict[str, Any] = {
        "metrics": str(run_dir / "metrics.jsonl"),
        "config": None,
        "nsys_report": None,
        "nsys_sqlite": None,
    }
    config_summary = load_run_config_summary(run_dir, warnings)
    if config_summary is not None:
        inputs["config"] = str(run_dir / "config.toml")

    nsys_summary = None
    nsys_report = run_dir / "profiles" / "nsys.nsys-rep"
    if include_nsys:
        if nsys_report.exists():
            inputs["nsys_report"] = str(nsys_report)
            resolved_nsys = shutil.which("nsys") if nsys_executable == "auto" else nsys_executable
            if resolved_nsys is None:
                warnings.append("NSys report found, but `nsys` is not available; generated metrics-only analysis.")
            else:
                sqlite_path = profile_dir / "nsys.sqlite"
                try:
                    export_nsys_sqlite(nsys_report, sqlite_path, runner=runner, nsys_executable=resolved_nsys)
                    inputs["nsys_sqlite"] = str(sqlite_path)
                    nsys_summary = summarize_nsys_sqlite(sqlite_path)
                except (OSError, sqlite3.Error, subprocess.CalledProcessError) as exc:
                    warnings.append(f"Failed to analyze NSys report: {exc}")
        else:
            warnings.append("No NSys report found; generated metrics-only analysis.")

    findings = build_findings(timing_summary, nsys_summary)
    report = {
        "version": REPORT_VERSION,
        "run_dir": str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "config": config_summary,
        "timing_summary": timing_summary,
        "nsys_summary": nsys_summary,
        "findings": findings,
        "warnings": warnings,
    }
    markdown = format_profile_report(report)
    json_path, markdown_path = write_profile_report(run_dir, report, markdown, output_dir=output_dir)
    return AnalysisResult(report=report, json_path=json_path, markdown_path=markdown_path, markdown=markdown)


def load_run_config_summary(run_dir: Path, warnings: list[str]) -> dict[str, Any] | None:
    config_path = run_dir / "config.toml"
    if not config_path.exists():
        return None
    try:
        config = load_config(config_path)
    except Exception as exc:  # Keep analysis best-effort if an old config shape no longer parses.
        warnings.append(f"Failed to parse run config: {exc}")
        return None
    return {
        "experiment": {
            "name": config.experiment.name,
            "out_dir": config.experiment.out_dir,
        },
        "model": {
            "layers": config.model.n_layers,
            "hidden_size": config.model.hidden_size,
            "heads": config.model.n_heads,
            "kv_heads": config.model.n_kv_heads,
            "seq_len": config.model.seq_len,
            "vocab_size": config.model.vocab_size,
        },
        "train": {
            "steps": config.train.steps,
            "batch_size": config.train.batch_size,
            "seq_len": config.train.seq_len,
            "tokens_per_step": config.train.batch_size * config.train.seq_len,
            "lr": config.train.lr,
            "log_every": config.train.log_every,
            "eval_every": config.train.eval_every,
        },
        "precision": {
            "compute_dtype": config.precision.compute_dtype,
            "param_dtype": config.precision.param_dtype,
            "loss_dtype": config.precision.loss_dtype,
        },
        "profiling": {
            "enabled": config.profiling.enabled,
            "profiler": config.profiling.profiler,
            "start_step": config.profiling.start_step,
            "steps": config.profiling.steps,
        },
    }


def export_nsys_sqlite(
    nsys_report: str | Path,
    sqlite_path: str | Path,
    *,
    runner=subprocess.run,
    nsys_executable: str = "nsys",
):
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    runner(
        [
            nsys_executable,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "--quiet=true",
            "--output",
            str(sqlite_path),
            str(nsys_report),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def summarize_nsys_sqlite(sqlite_path: str | Path, *, top_n: int = 10) -> dict[str, Any]:
    sqlite_path = Path(sqlite_path)
    with sqlite3.connect(sqlite_path) as con:
        con.row_factory = sqlite3.Row
        tables = _sqlite_tables(con)
        return {
            "sqlite_path": str(sqlite_path),
            "nvtx_phases": _summarize_nvtx_phases(con, tables),
            "cuda": {
                "top_kernels": _top_cuda_kernel_rows(con, tables, top_n),
                "top_runtime_apis": _top_cuda_runtime_rows(con, tables, top_n),
                "memory": _cuda_memory_summary(con, tables),
            },
            "diagnostics": _diagnostic_rows(con, tables, top_n),
            "xla_compile_events": _count_xla_compile_strings(con, tables),
        }


def _sqlite_tables(con: sqlite3.Connection) -> set[str]:
    return {row[0] for row in con.execute("select name from sqlite_master where type='table'")}


def _summarize_nvtx_phases(con: sqlite3.Connection, tables: set[str]) -> dict[str, dict[str, float | int]]:
    if "NVTX_EVENTS" not in tables:
        return {}
    if "StringIds" in tables:
        rows = con.execute(
            """
            select coalesce(n.text, s.value) as name, (n.end - n.start) / 1000000.0 as duration_ms
            from NVTX_EVENTS n
            left join StringIds s on n.textId = s.id
            where n.end is not null
            """
        ).fetchall()
    else:
        rows = con.execute(
            """
            select n.text as name, (n.end - n.start) / 1000000.0 as duration_ms
            from NVTX_EVENTS n
            where n.end is not null
            """
        ).fetchall()
    by_name: dict[str, list[float]] = {name: [] for name in NSYS_PHASES}
    for row in rows:
        name = row["name"]
        if name in by_name:
            by_name[name].append(float(row["duration_ms"]))
    return {name: _stats(values).__dict__ for name, values in by_name.items() if values}


def _top_cuda_kernel_rows(con: sqlite3.Connection, tables: set[str], top_n: int) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return []
    if "StringIds" in tables:
        query = """
            select coalesce(s.value, cast(k.demangledName as text)) as name,
                   count(*) as count,
                   sum((k.end - k.start) / 1000000.0) as total_ms,
                   avg((k.end - k.start) / 1000000.0) as mean_ms,
                   max((k.end - k.start) / 1000000.0) as max_ms
            from CUPTI_ACTIVITY_KIND_KERNEL k
            left join StringIds s on k.demangledName = s.id
            group by name
            order by total_ms desc
            limit ?
            """
    else:
        query = """
            select cast(k.demangledName as text) as name,
                   count(*) as count,
                   sum((k.end - k.start) / 1000000.0) as total_ms,
                   avg((k.end - k.start) / 1000000.0) as mean_ms,
                   max((k.end - k.start) / 1000000.0) as max_ms
            from CUPTI_ACTIVITY_KIND_KERNEL k
            group by name
            order by total_ms desc
            limit ?
            """
    return _query_top_duration_rows(con, query, top_n)


def _top_cuda_runtime_rows(con: sqlite3.Connection, tables: set[str], top_n: int) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
        return []
    if "StringIds" in tables:
        query = """
            select coalesce(s.value, cast(r.nameId as text)) as name,
                   count(*) as count,
                   sum((r.end - r.start) / 1000000.0) as total_ms,
                   avg((r.end - r.start) / 1000000.0) as mean_ms,
                   max((r.end - r.start) / 1000000.0) as max_ms
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            left join StringIds s on r.nameId = s.id
            group by name
            order by total_ms desc
            limit ?
            """
    else:
        query = """
            select cast(r.nameId as text) as name,
                   count(*) as count,
                   sum((r.end - r.start) / 1000000.0) as total_ms,
                   avg((r.end - r.start) / 1000000.0) as mean_ms,
                   max((r.end - r.start) / 1000000.0) as max_ms
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            group by name
            order by total_ms desc
            limit ?
            """
    return _query_top_duration_rows(con, query, top_n)


def _query_top_duration_rows(con: sqlite3.Connection, query: str, top_n: int) -> list[dict[str, Any]]:
    rows = []
    for row in con.execute(query, (top_n,)):
        rows.append(
            {
                "name": row["name"],
                "count": int(row["count"]),
                "total_ms": float(row["total_ms"] or 0.0),
                "mean_ms": float(row["mean_ms"] or 0.0),
                "max_ms": float(row["max_ms"] or 0.0),
            }
        )
    return rows


def _cuda_memory_summary(con: sqlite3.Connection, tables: set[str]) -> dict[str, Any]:
    summary = {}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        row = con.execute(
            """
            select count(*) as count,
                   sum(bytes) as total_bytes,
                   sum((end - start) / 1000000.0) as total_ms,
                   max((end - start) / 1000000.0) as max_ms
            from CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ).fetchone()
        summary["memcpy"] = {
            "count": int(row["count"] or 0),
            "total_bytes": int(row["total_bytes"] or 0),
            "total_ms": float(row["total_ms"] or 0.0),
            "max_ms": float(row["max_ms"] or 0.0),
        }
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        row = con.execute(
            """
            select count(*) as count,
                   sum(bytes) as total_bytes,
                   sum((end - start) / 1000000.0) as total_ms,
                   max((end - start) / 1000000.0) as max_ms
            from CUPTI_ACTIVITY_KIND_MEMSET
            """
        ).fetchone()
        summary["memset"] = {
            "count": int(row["count"] or 0),
            "total_bytes": int(row["total_bytes"] or 0),
            "total_ms": float(row["total_ms"] or 0.0),
            "max_ms": float(row["max_ms"] or 0.0),
        }
    return summary


def _diagnostic_rows(con: sqlite3.Connection, tables: set[str], top_n: int) -> list[dict[str, Any]]:
    if "DIAGNOSTIC_EVENT" not in tables:
        return []
    rows = con.execute(f"select * from DIAGNOSTIC_EVENT limit {int(top_n)}").fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _count_xla_compile_strings(con: sqlite3.Connection, tables: set[str]) -> int:
    if "StringIds" not in tables:
        return 0
    row = con.execute(
        """
        select count(*) as count
        from StringIds
        where value like '%XlaCompile%'
           or value like '%autotun%'
           or value like '%XlaPass%'
        """
    ).fetchone()
    return int(row["count"] or 0)


def build_findings(timing_summary: dict[str, Any], nsys_summary: dict[str, Any] | None) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    normal_fields = timing_summary.get("groups", {}).get("normal", {}).get("fields", {})
    all_fields = timing_summary.get("groups", {}).get("all", {}).get("fields", {})

    step_p50 = _field_stat(normal_fields, "time/step_sec", "p50") or _field_stat(all_fields, "time/step_sec", "p50")
    metrics_sync_p50 = _field_stat(normal_fields, "time/metrics_sync_sec", "p50")
    if step_p50 and metrics_sync_p50 and metrics_sync_p50 / step_p50 >= 0.4:
        findings.append(
            _finding(
                "Host/device synchronization dominates logged steps",
                f"normal p50 metrics_sync={metrics_sync_p50:.4f}s vs step={step_p50:.4f}s ({metrics_sync_p50 / step_p50:.0%}).",
                "Keep scalar reads batched, avoid extra host conversions, and inspect whether the sync is mostly queued GPU work.",
            )
        )

    for key, label in [
        ("time/data_sec", "Data loading"),
        ("time/batch_log_sec", "Batch provenance logging"),
        ("time/shard_sec", "Batch sharding"),
        ("time/log_sec", "Metric logging"),
    ]:
        value = _field_stat(normal_fields, key, "p50")
        if step_p50 and value and value / step_p50 >= 0.1:
            findings.append(
                _finding(
                    f"{label} is a meaningful part of normal-step time",
                    f"normal p50 {key}={value:.4f}s vs step={step_p50:.4f}s ({value / step_p50:.0%}).",
                    "Optimize this phase before lower-level kernel work if it remains visible in longer runs.",
                )
            )

    normal_tps = _field_stat(normal_fields, "time/tokens_per_sec", "p50")
    for group_name, label in [("eval", "Eval"), ("sample", "Sampling"), ("checkpoint", "Checkpointing")]:
        group_fields = timing_summary.get("groups", {}).get(group_name, {}).get("fields", {})
        group_tps = _field_stat(group_fields, "time/tokens_per_sec", "p50")
        group_step = _field_stat(group_fields, "time/step_sec", "p50")
        if normal_tps and group_tps and group_tps < normal_tps * 0.25:
            findings.append(
                _finding(
                    f"{label} rows distort full-loop throughput",
                    f"{group_name} p50 tok/s={group_tps:.0f} vs normal p50 tok/s={normal_tps:.0f}.",
                    "Use the normal-step group for speed comparisons, and treat these rows as workload overhead.",
                )
            )
        elif step_p50 and group_step and group_step > step_p50 * 4.0:
            findings.append(
                _finding(
                    f"{label} rows are much slower than normal rows",
                    f"{group_name} p50 step={group_step:.4f}s vs normal p50 step={step_p50:.4f}s.",
                    "Separate this work from steady-state throughput analysis.",
                )
            )

    if nsys_summary is not None:
        nvtx = nsys_summary.get("nvtx_phases", {})
        nvtx_step_total = _phase_stat(nvtx, "step", "mean")
        nvtx_sync_total = _phase_stat(nvtx, "metrics_sync", "mean")
        if nvtx_step_total and nvtx_sync_total and nvtx_sync_total / nvtx_step_total >= 0.4:
            findings.append(
                _finding(
                    "NSys NVTX also shows sync-heavy captured steps",
                    f"NVTX mean metrics_sync={nvtx_sync_total:.3f}ms vs step={nvtx_step_total:.3f}ms ({nvtx_sync_total / nvtx_step_total:.0%}).",
                    "Use the CUDA runtime and kernel tables below to determine whether this is host waiting, copies, or GPU kernels.",
                )
            )

        runtime_rows = nsys_summary.get("cuda", {}).get("top_runtime_apis", [])
        sync_rows = [row for row in runtime_rows if row["name"] in CUDA_SYNC_RUNTIME_NAMES]
        memcpy_rows = [row for row in runtime_rows if row["name"] in CUDA_MEMCPY_RUNTIME_NAMES]
        if sync_rows:
            names = ", ".join(row["name"] for row in sync_rows[:3])
            total_ms = sum(float(row["total_ms"]) for row in sync_rows)
            findings.append(
                _finding(
                    "CUDA runtime synchronization is prominent",
                    f"Top runtime APIs include {names} with {total_ms:.3f}ms total in the report summary.",
                    "Correlate these calls with metrics_sync and eval/sample boundaries before changing model kernels.",
                )
            )
        if memcpy_rows:
            names = ", ".join(row["name"] for row in memcpy_rows[:3])
            total_ms = sum(float(row["total_ms"]) for row in memcpy_rows)
            findings.append(
                _finding(
                    "CUDA memcpy API time is visible",
                    f"Top runtime APIs include {names} with {total_ms:.3f}ms total.",
                    "Check host metric reads, data transfer, and any device-to-host paths before deeper kernel work.",
                )
            )

        if int(nsys_summary.get("xla_compile_events", 0)) > 0:
            findings.append(
                _finding(
                    "XLA compile/autotune strings are present in the NSys export",
                    f"Found {nsys_summary['xla_compile_events']} StringIds matching XLA compile/autotune patterns.",
                    "Make sure optimization conclusions are based on the intended steady-state capture window.",
                )
            )

    if not findings:
        findings.append(
            _finding(
                "No obvious bottleneck crossed the built-in thresholds",
                "The deterministic rules did not identify a dominant timing or CUDA runtime issue.",
                "Compare this run against another config or inspect the raw NSys/Perfetto artifact for finer-grained questions.",
            )
        )
    return findings


def _field_stat(fields: dict[str, Any], key: str, stat: str) -> float | None:
    if key not in fields:
        return None
    value = fields[key].get(stat)
    return float(value) if value is not None else None


def _phase_stat(phases: dict[str, Any], key: str, stat: str) -> float | None:
    if key not in phases:
        return None
    value = phases[key].get(stat)
    return float(value) if value is not None else None


def _finding(title: str, evidence: str, action: str) -> dict[str, str]:
    return {"title": title, "evidence": evidence, "action": action}


def format_profile_report(report: dict[str, Any]) -> str:
    lines = [
        "# Profile Report",
        "",
        f"- run: `{report['run_dir']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- report_version: `{report['version']}`",
    ]
    config = report.get("config")
    if config is not None:
        model = config["model"]
        train = config["train"]
        precision = config["precision"]
        lines.extend(
            [
                f"- model: layers={model['layers']} hidden={model['hidden_size']} heads={model['heads']} kv_heads={model['kv_heads']}",
                f"- train: steps={train['steps']} batch={train['batch_size']} seq_len={train['seq_len']} tokens_per_step={train['tokens_per_step']}",
                f"- precision: compute={precision['compute_dtype']} params={precision['param_dtype']} loss={precision['loss_dtype']}",
            ]
        )

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    lines.extend(["", "## Key Findings"])
    for index, finding in enumerate(report["findings"], start=1):
        lines.extend(
            [
                f"{index}. **{finding['title']}**",
                f"   Evidence: {finding['evidence']}",
                f"   Next action: {finding['action']}",
            ]
        )

    lines.extend(["", "## Timing Summary"])
    summary = report["timing_summary"]
    lines.append(
        f"Rows used: {summary['rows_used']} / {summary['rows_total']} after warmup_steps={summary['warmup_steps']} "
        f"(steps {summary['step_min']}..{summary['step_max']})."
    )
    lines.append("")
    lines.extend(_format_group_table(summary))

    lines.extend(
        [
            "",
            "## JAX Async Interpretation",
            "- `time/train_step_sec` mostly measures Python enqueue/dispatch time for the jitted train step.",
            "- `time/metrics_sync_sec` is the explicit host synchronization point for logged scalars and can include queued device work.",
            "- Use the `normal` group for speed comparisons; eval, sample, and checkpoint rows intentionally include extra work.",
        ]
    )

    nsys_summary = report.get("nsys_summary")
    if nsys_summary is not None:
        lines.extend(["", "## NSys NVTX Phases", ""])
        lines.extend(_format_nsys_phase_table(nsys_summary.get("nvtx_phases", {})))
        lines.extend(["", "## CUDA Summary", ""])
        lines.extend(_format_named_rows("Top Kernels", nsys_summary.get("cuda", {}).get("top_kernels", [])))
        lines.extend(["", ""])
        lines.extend(_format_named_rows("Top Runtime APIs", nsys_summary.get("cuda", {}).get("top_runtime_apis", [])))
        memory = nsys_summary.get("cuda", {}).get("memory", {})
        if memory:
            lines.extend(["", "### Memory Activity"])
            for name, stats in memory.items():
                lines.append(
                    f"- {name}: count={stats['count']} total_bytes={stats['total_bytes']} "
                    f"total_ms={_format_number(stats['total_ms'])} max_ms={_format_number(stats['max_ms'])}"
                )
        diagnostics = nsys_summary.get("diagnostics", [])
        if diagnostics:
            lines.extend(["", "### Diagnostics"])
            for row in diagnostics:
                lines.append(f"- `{json.dumps(row, sort_keys=True)}`")

    return "\n".join(lines).rstrip() + "\n"


def _format_group_table(summary: dict[str, Any]) -> list[str]:
    lines = ["| group | rows | step p50 | metrics_sync p50 | tok/s p50 | train tok/s p50 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group_name, group in summary["groups"].items():
        fields = group["fields"]
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(group["rows"]),
                    _stat_text(fields, "time/step_sec", "p50"),
                    _stat_text(fields, "time/metrics_sync_sec", "p50"),
                    _stat_text(fields, "time/tokens_per_sec", "p50"),
                    _stat_text(fields, "time/train_tokens_per_sec", "p50"),
                ]
            )
            + " |"
        )
    return lines


def _stat_text(fields: dict[str, Any], key: str, stat: str) -> str:
    value = _field_stat(fields, key, stat)
    return "n/a" if value is None else _format_number(value)


def _format_nsys_phase_table(phases: dict[str, Any]) -> list[str]:
    if not phases:
        return ["No NVTX phase rows found."]
    lines = ["| phase | count | mean ms | p50 ms | p95 ms | max ms |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for phase in NSYS_PHASES:
        if phase not in phases:
            continue
        stats = phases[phase]
        lines.append(
            f"| {phase} | {stats['count']} | {_format_number(stats['mean'])} | "
            f"{_format_number(stats['p50'])} | {_format_number(stats['p95'])} | {_format_number(stats['max'])} |"
        )
    return lines


def _format_named_rows(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {title}"]
    if not rows:
        lines.append("No rows found.")
        return lines
    lines.extend(["| name | count | total ms | mean ms | max ms |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in rows:
        name = str(row["name"]).replace("|", "\\|")
        lines.append(
            f"| `{name}` | {row['count']} | {_format_number(row['total_ms'])} | "
            f"{_format_number(row['mean_ms'])} | {_format_number(row['max_ms'])} |"
        )
    return lines


def write_profile_report(
    run_dir: str | Path,
    report: dict[str, Any],
    markdown: str,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    profile_dir = Path(output_dir) if output_dir is not None else Path(run_dir) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    json_path = profile_dir / "profile_report.json"
    markdown_path = profile_dir / "profile_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


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


def print_analysis(
    run_dir: str | Path,
    *,
    warmup_steps: int,
    output_dir: str | Path | None,
    runner=subprocess.run,
    nsys_executable: str | None = "auto",
) -> AnalysisResult:
    result = run_analysis(
        run_dir,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        runner=runner,
        nsys_executable=nsys_executable,
    )
    print("Profile Analysis")
    print(f"  run:      {run_dir}")
    print(f"  findings: {len(result.report['findings'])}")
    print(f"  warnings: {len(result.report['warnings'])}")
    print(f"wrote: {result.json_path}")
    print(f"wrote: {result.markdown_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile and summarize training runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Summarize timing metrics from a completed run.")
    summary.add_argument("run_dir", help="Run directory containing metrics.jsonl.")
    summary.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Ignore metric rows before this step.")
    summary.add_argument("--output-dir", help="Directory for timing_summary.json and timing_summary.md. Defaults to RUN_DIR/profiles.")

    analyze = subparsers.add_parser("analyze", help="Generate an LLM-readable profile report for a completed run.")
    analyze.add_argument("run_dir", help="Run directory containing metrics.jsonl.")
    analyze.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Ignore metric rows before this step.")
    analyze.add_argument("--output-dir", help="Directory for profile_report.json and profile_report.md. Defaults to RUN_DIR/profiles.")

    nsys = subparsers.add_parser("nsys", help="Run Nsight Systems around a profiling config.")
    nsys.add_argument("config", help="Config TOML with [profiling] profiler = \"nsys\".")
    nsys.add_argument("--force-run-dir", action="store_true", help="Remove the configured run directory before launching.")
    nsys.add_argument("--nsys-output-dir", default="profiles", help="Temporary directory for NSys reports outside runs/.")
    nsys.add_argument("--dry-run", action="store_true", help="Print resolved paths and command without executing NSys.")
    nsys.add_argument("--extra-nsys-arg", action="append", default=[], help="Additional argument passed to nsys profile. Repeat as needed.")
    nsys.add_argument("--analyze", action="store_true", help="Generate profile_report.json/md after a successful NSys run.")
    nsys.add_argument("--analyze-warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help="Warmup cutoff used by --analyze.")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner=subprocess.run,
    analysis_runner=subprocess.run,
    nsys_executable: str | None = "auto",
):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"summary", "analyze", "nsys", "-h", "--help"}:
        argv = ["summary", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        print_summary(args.run_dir, warmup_steps=args.warmup_steps, output_dir=args.output_dir)
        return
    if args.command == "analyze":
        print_analysis(
            args.run_dir,
            warmup_steps=args.warmup_steps,
            output_dir=args.output_dir,
            runner=analysis_runner,
            nsys_executable=nsys_executable,
        )
        return
    if args.command == "nsys":
        plan = run_nsys(
            args.config,
            force_run_dir=args.force_run_dir,
            dry_run=args.dry_run,
            nsys_output_dir=args.nsys_output_dir,
            extra_nsys_args=args.extra_nsys_arg,
            runner=runner,
        )
        print_nsys_plan(plan, dry_run=args.dry_run)
        if args.analyze and not args.dry_run:
            print_analysis(
                plan.run_dir,
                warmup_steps=args.analyze_warmup_steps,
                output_dir=None,
                runner=analysis_runner,
                nsys_executable=nsys_executable,
            )
        return


if __name__ == "__main__":
    main()
