"""Read-only analysis for Jaxtitan profiling artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable, Mapping

from jaxtitan.errors import ContractError


PROFILE_ANALYSIS_SCHEMA_VERSION = 1
_METRIC_FIELDS = (
    "train_step_sec",
    "step_sec",
    "train_tokens_per_sec",
    "data_sec",
    "placement_sec",
)
_EXCLUDED_EVENT_TYPES = frozenset({"eval_started", "checkpoint_saved"})
_HLO_OPS = (
    "collective-permute-start",
    "reduce-scatter",
    "all-gather-start",
    "all-reduce-start",
    "all-to-all-start",
    "collective-permute",
    "all-gather",
    "all-reduce",
    "all-to-all",
    "dot-general",
    "dot",
    "gather",
    "scatter",
)
_HLO_INSTRUCTION = re.compile(
    r"^\s*(?:ROOT\s+)?%[^=]+\s*=\s+(?P<result>.*?)\b(?P<op>"
    + "|".join(map(re.escape, _HLO_OPS))
    + r")\("
)
_HLO_ARRAY_SHAPE = re.compile(r"\b(?P<dtype>pred|[sufc]\d+|bf16)\[(?P<dimensions>\d+(?:,\d+)*)?\]")
_HLO_DTYPE_BYTES = {
    "pred": 1,
    "s8": 1,
    "u8": 1,
    "f8": 1,
    "s16": 2,
    "u16": 2,
    "f16": 2,
    "bf16": 2,
    "s32": 4,
    "u32": 4,
    "f32": 4,
    "c64": 8,
    "s64": 8,
    "u64": 8,
    "f64": 8,
    "c128": 16,
}
_MODULE_NUMBER = re.compile(r"module_(\d+)\.")


def analyze_profile_root(root: str | Path, *, warmup_steps: int = 2) -> dict[str, Any]:
    """Analyze every Jaxtitan run artifact found below ``root``."""

    source_root = Path(root)
    if warmup_steps < 0:
        raise ContractError(f"profile warmup steps must be non-negative, got {warmup_steps}")
    if not source_root.is_dir():
        raise ContractError(f"profile analysis root is not a directory: {source_root}")

    run_dirs = discover_profile_runs(source_root)
    if not run_dirs:
        raise ContractError(f"profile analysis found no metrics/train.jsonl below {source_root}")
    hlo_roots = _discover_hlo_roots(source_root)
    runs = [_analyze_run(run_dir, warmup_steps=warmup_steps, hlo_dir=hlo_roots.get(run_dir.name)) for run_dir in run_dirs]
    runs.sort(key=lambda item: item["run_id"])
    return {
        "schema_version": PROFILE_ANALYSIS_SCHEMA_VERSION,
        "source": source_root.name,
        "measurement_policy": {
            "warmup_steps": warmup_steps,
            "steady_start": "max(warmup_steps + 1, trace_end_step + 1)",
            "excluded_event_types": sorted(_EXCLUDED_EVENT_TYPES),
            "scalar_reducer": "median",
            "trace_device": "GPU:0",
        },
        "run_count": len(runs),
        "hardware": _hardware_groups(runs),
        "runs": runs,
        "comparisons": _paired_comparisons(runs),
    }


def discover_profile_runs(root: str | Path) -> list[Path]:
    """Return deterministic run roots containing canonical train metrics."""

    source_root = Path(root)
    candidates = {path.parent.parent for path in source_root.rglob("metrics/train.jsonl") if path.is_file()}
    return sorted(candidates, key=lambda path: path.as_posix())


def profile_analysis_to_json(payload: Mapping[str, Any]) -> str:
    """Serialize a profile report deterministically."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def format_profile_analysis(payload: Mapping[str, Any]) -> str:
    """Format the high-signal portion of a profile report for humans."""

    lines = [
        f"profile analysis: runs={payload['run_count']} source={payload['source']}",
        "",
        "run                                             steps    train ms    step ms      tok/s",
    ]
    for run in payload["runs"]:
        medians = run["steady"]["medians"]
        lines.append(
            f"{run['run_id']:<47} "
            f"{run['steady']['start_step']:>2}-{run['steady']['end_step']:<3} "
            f"{_milliseconds(medians.get('train_step_sec')):>11} "
            f"{_milliseconds(medians.get('step_sec')):>10} "
            f"{_number(medians.get('train_tokens_per_sec')):>10}"
        )
    if payload["comparisons"]:
        lines.extend(["", "paired deltas:"])
        for item in payload["comparisons"]:
            lines.append(
                f"  {item['kind']} {item['candidate']} vs {item['baseline']}: "
                f"train={_percent(item['train_step_delta'])} step={_percent(item['step_delta'])}"
            )
    return "\n".join(lines)


def summarize_hlo_text(text: str) -> dict[str, Any]:
    """Count semantic HLO instruction definitions without counting references."""

    counts: Counter[str] = Counter()
    estimated_result_bytes: Counter[str] = Counter()
    examples: defaultdict[str, list[str]] = defaultdict(list)
    result_shapes: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in text.splitlines():
        match = _HLO_INSTRUCTION.match(line)
        if match is None:
            continue
        op = match.group("op")
        counts[op] += 1
        result_shape = _first_hlo_array_shape(match.group("result"))
        if result_shape is not None:
            estimated_result_bytes[op] += result_shape["estimated_bytes"]
            if len(result_shapes[op]) < 3 and result_shape not in result_shapes[op]:
                result_shapes[op].append(result_shape)
        if len(examples[op]) < 3:
            examples[op].append(line.strip()[:500])
    return {
        "instruction_counts": dict(sorted(counts.items())),
        "estimated_result_bytes": dict(sorted(estimated_result_bytes.items())),
        "estimate_scope": "sum_of_first_array_result_per_instruction",
        "instruction_result_shapes": {name: values for name, values in sorted(result_shapes.items())},
        "instruction_examples": {name: values for name, values in sorted(examples.items())},
    }


def _first_hlo_array_shape(result: str) -> dict[str, Any] | None:
    match = _HLO_ARRAY_SHAPE.search(result)
    if match is None:
        return None
    dtype = match.group("dtype")
    dtype_bytes = _HLO_DTYPE_BYTES.get(dtype)
    if dtype_bytes is None:
        return None
    dimensions_text = match.group("dimensions")
    dimensions = [] if not dimensions_text else [int(value) for value in dimensions_text.split(",")]
    return {
        "dtype": dtype,
        "shape": dimensions,
        "estimated_bytes": math.prod(dimensions) * dtype_bytes,
    }


def summarize_perfetto_trace(path: str | Path) -> dict[str, Any]:
    """Summarize complete GPU-0 events from a Perfetto JSON trace."""

    trace_path = Path(path)
    try:
        opener = gzip.open if trace_path.suffix == ".gz" else open
        with opener(trace_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"failed to read Perfetto trace {trace_path}: {exc}") from exc
    events = payload.get("traceEvents") if isinstance(payload, Mapping) else None
    if not isinstance(events, list):
        raise ContractError(f"Perfetto trace must contain a traceEvents array: {trace_path}")

    gpu_zero_pid = None
    for event in events:
        if not isinstance(event, Mapping) or event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        if event.get("args", {}).get("name") == "/device:GPU:0":
            gpu_zero_pid = event.get("pid")
            break
    if gpu_zero_pid is None:
        raise ContractError(f"Perfetto trace has no /device:GPU:0 process metadata: {trace_path}")

    by_name: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_category: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    total_us = 0.0
    for event in events:
        if not isinstance(event, Mapping) or event.get("pid") != gpu_zero_pid or event.get("ph") != "X":
            continue
        duration = event.get("dur")
        name = event.get("name")
        if not isinstance(duration, (int, float)) or not isinstance(name, str) or duration < 0:
            continue
        duration_us = float(duration)
        total_us += duration_us
        by_name[name][0] += 1
        by_name[name][1] += duration_us
        category = _kernel_category(name)
        by_category[category][0] += 1
        by_category[category][1] += duration_us
        by_category[category][2] = max(by_category[category][2], duration_us)

    top = sorted(by_name.items(), key=lambda item: (-item[1][1], item[0]))[:20]
    return {
        "device": "GPU:0",
        "event_count": int(sum(value[0] for value in by_name.values())),
        "event_sum_sec": total_us / 1_000_000.0,
        "categories": {
            name: {
                "count": int(value[0]),
                "duration_sec": value[1] / 1_000_000.0,
                "max_duration_sec": value[2] / 1_000_000.0,
                "event_sum_fraction": value[1] / total_us if total_us else 0.0,
            }
            for name, value in sorted(by_category.items())
        },
        "top_events": [
            {"name": name, "count": int(value[0]), "duration_sec": value[1] / 1_000_000.0}
            for name, value in top
        ],
    }


def _analyze_run(run_dir: Path, *, warmup_steps: int, hlo_dir: Path | None) -> dict[str, Any]:
    final = _load_json(run_dir / "summaries" / "final.json", label="final summary")
    runtime = _load_json(run_dir / "diagnostics" / "runtime.json", label="runtime diagnostics")
    resolved = _load_json(run_dir / "config" / "resolved.json", label="resolved config")
    run_id = final.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ContractError(f"final summary is missing run_id: {run_dir}")
    if final.get("status") != "completed":
        raise ContractError(f"profile run {run_id} is not completed: status={final.get('status')!r}")
    if int(final.get("final_optimizer_nonfinite_group_count", 0)) != 0:
        raise ContractError(f"profile run {run_id} contains nonfinite optimizer groups")

    rows = _load_jsonl(run_dir / "metrics" / "train.jsonl", label="train metrics")
    events = _load_jsonl(run_dir / "events.jsonl", label="events")
    profiling = runtime.get("profiling", {})
    trace_start = _optional_int(profiling.get("trace_start_step"))
    trace_end = _optional_int(profiling.get("trace_end_step"))
    start_step = warmup_steps + 1
    if profiling.get("enabled") and trace_end is not None:
        start_step = max(start_step, trace_end + 1)
    excluded_steps = {
        int(event["step"])
        for event in events
        if event.get("type") in _EXCLUDED_EVENT_TYPES and isinstance(event.get("step"), int)
    }
    steady_rows = [
        row
        for row in rows
        if isinstance(row.get("step"), int) and row["step"] >= start_step and row["step"] not in excluded_steps
    ]
    if not steady_rows:
        raise ContractError(f"profile run {run_id} has no steady rows after applying the measurement policy")
    traced_rows = []
    if trace_start is not None and trace_end is not None:
        traced_rows = [row for row in rows if trace_start <= int(row.get("step", -1)) <= trace_end]

    trace_paths = sorted(run_dir.rglob("perfetto_trace.json.gz"))
    trace_summary = summarize_perfetto_trace(trace_paths[0]) if trace_paths else None
    hlo_summary = _summarize_hlo_dir(hlo_dir) if hlo_dir is not None else None
    steady_medians = _metric_medians(steady_rows)
    trace_medians = _metric_medians(traced_rows) if traced_rows else None
    return {
        "run_id": run_id,
        "status": "completed",
        "model": {
            "name": runtime.get("model", {}).get("name"),
            "parameters": runtime.get("model", {}).get("parameters"),
            "signature": _model_signature(resolved.get("model", {})),
        },
        "optimizer": resolved.get("optimizer", {}).get("name"),
        "layout": _layout_name(resolved),
        "hardware": {
            "backend": runtime.get("jax", {}).get("backend"),
            "device_kind": runtime.get("performance", {}).get("device_kind"),
            "device_count": runtime.get("performance", {}).get("device_count"),
            "mesh": runtime.get("parallelism", {}).get("mesh"),
        },
        "steady": {
            "start_step": min(row["step"] for row in steady_rows),
            "end_step": max(row["step"] for row in steady_rows),
            "row_count": len(steady_rows),
            "excluded_steps": sorted(excluded_steps),
            "medians": steady_medians,
        },
        "trace_window": {
            "start_step": trace_start,
            "end_step": trace_end,
            "row_count": len(traced_rows),
            "medians": trace_medians,
            "tax": _metric_deltas(trace_medians, steady_medians) if trace_medians else None,
        },
        "trace": trace_summary,
        "hlo": hlo_summary,
    }


def _paired_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    by_key = {(run["model"]["signature"], run["layout"], run["optimizer"]): run for run in runs}
    for run in runs:
        signature = run["model"]["signature"]
        optimizer = run["optimizer"]
        layout = run["layout"]
        if optimizer == "muon":
            baseline = by_key.get((signature, layout, "adamw"))
            if baseline is not None:
                comparisons.append(_comparison("optimizer", baseline, run))
        layout_baseline = {
            "tp": "ddp",
            "fsdp+tp": "tp",
            "zero2+tp": "tp",
            "tp+ep": "ep",
        }.get(layout)
        if layout_baseline is not None:
            baseline = by_key.get((signature, layout_baseline, optimizer))
            if baseline is not None:
                comparisons.append(_comparison("layout", baseline, run))
    return sorted(comparisons, key=lambda item: (item["kind"], item["candidate"], item["baseline"]))


def _comparison(kind: str, baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    base = baseline["steady"]["medians"]
    cand = candidate["steady"]["medians"]
    return {
        "kind": kind,
        "baseline": baseline["run_id"],
        "candidate": candidate["run_id"],
        "train_step_delta": _relative_delta(cand.get("train_step_sec"), base.get("train_step_sec")),
        "step_delta": _relative_delta(cand.get("step_sec"), base.get("step_sec")),
        "tokens_per_sec_delta": _relative_delta(
            cand.get("train_tokens_per_sec"), base.get("train_tokens_per_sec")
        ),
    }


def _metric_medians(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    row_list = list(rows)
    return {
        field: _median_or_none(
            float(row[field])
            for row in row_list
            if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
        )
        for field in _METRIC_FIELDS
    }


def _metric_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    return {field: _relative_delta(candidate.get(field), baseline.get(field)) for field in _METRIC_FIELDS}


def _relative_delta(candidate: Any, baseline: Any) -> float | None:
    if not isinstance(candidate, (int, float)) or not isinstance(baseline, (int, float)) or baseline == 0:
        return None
    return float(candidate) / float(baseline) - 1.0


def _median_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(median(materialized)) if materialized else None


def _discover_hlo_roots(root: Path) -> dict[str, Path]:
    result = {}
    hlo_roots = (
        path
        for path in root.rglob("*")
        if path.is_dir() and (path.name == "hlo" or path.name.endswith("_hlo"))
    )
    for path in hlo_roots:
        for child in path.iterdir():
            if child.is_dir():
                result.setdefault(child.name, child)
    return result


def _summarize_hlo_dir(hlo_dir: Path) -> dict[str, Any] | None:
    candidates = list(hlo_dir.glob("*jit__compiled_impl*gpu_after_optimizations.txt"))
    if not candidates:
        return None
    selected = max(candidates, key=lambda path: (path.stat().st_size, _module_number(path), path.name))
    summary = summarize_hlo_text(selected.read_text(encoding="utf-8", errors="replace"))
    summary.update({"file": selected.name, "bytes": selected.stat().st_size})
    return summary


def _module_number(path: Path) -> int:
    match = _MODULE_NUMBER.search(path.name)
    return int(match.group(1)) if match else -1


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        if "allgather" in lowered:
            return "nccl_all_gather"
        if "allreduce" in lowered:
            return "nccl_all_reduce"
        if "reducescatter" in lowered:
            return "nccl_reduce_scatter"
        if "alltoall" in lowered:
            return "nccl_all_to_all"
        return "nccl_other"
    if "input_scatter_fusion" in lowered or "input_reduce_fusion" in lowered:
        return "scatter_reduce_fusion"
    if "gemm" in lowered or "cublas" in lowered:
        return "gemm"
    if "memcpy" in lowered:
        return "memcpy"
    if "fusion" in lowered:
        return "fusion_other"
    return "other"


def _model_signature(model: Any) -> str:
    if not isinstance(model, Mapping):
        return "unknown"
    structural = {key: value for key, value in model.items() if key not in {"variant", "remat"}}
    return json.dumps(structural, sort_keys=True, separators=(",", ":"))


def _layout_name(resolved: Mapping[str, Any]) -> str:
    parallelism = resolved.get("parallelism", {})
    mode = parallelism.get("mode", "ddp")
    axes = []
    if mode in {"fsdp", "zero2"}:
        axes.append(str(mode))
    if parallelism.get("tensor_parallel"):
        axes.append("tp")
    if parallelism.get("expert_parallel"):
        axes.append("ep")
    return "+".join(axes) if axes else "ddp"


def _hardware_groups(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for run in runs:
        hardware = run["hardware"]
        key = json.dumps(hardware, sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, {**hardware, "run_count": 0})["run_count"] += 1
    return sorted(groups.values(), key=lambda item: json.dumps(item, sort_keys=True))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"profile run is missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"failed to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"profile run is missing {label}: {path}")
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ContractError(f"{label} row {line_number} must be a JSON object: {path}")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"failed to read {label} {path}: {exc}") from exc
    if not rows:
        raise ContractError(f"profile run has empty {label}: {path}")
    return rows


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def _milliseconds(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value * 1000.0:.1f}"


def _number(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value:,.0f}"


def _percent(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value * 100.0:+.1f}%"
