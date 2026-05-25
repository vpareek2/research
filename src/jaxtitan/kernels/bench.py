"""Benchmark helpers for Jaxtitan-owned kernel POCs."""

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp

from jaxtitan.errors import ContractError
from jaxtitan.kernels._ffi import rmsnorm_shared_object_path
from jaxtitan.kernels.build import MANIFEST_NAME, _sha256, cache_dir, load_cache_manifest
from jaxtitan.kernels.rmsnorm import RMSNORM_HIDDEN_SIZE, rmsnorm_reference, rmsnorm_tk_forward
from jaxtitan.specs.run import RunSpec


@dataclass(frozen=True, slots=True)
class RmsNormBenchmarkConfig:
    rows: tuple[int, ...] = (1, 4, 17, 64, 256)
    warmup: int = 5
    iters: int = 20


def benchmark_rmsnorm(
    spec: RunSpec,
    *,
    cache_root: str | Path | None = None,
    rows: tuple[int, ...] = (1, 4, 17, 64, 256),
    warmup: int = 5,
    iters: int = 20,
    write_artifact: bool = True,
) -> dict[str, Any]:
    """Benchmark pure JAX RMSNorm against the cached TK FFI RMSNorm."""

    config = RmsNormBenchmarkConfig(rows=rows, warmup=warmup, iters=iters)
    _validate_benchmark_config(config)
    shared_object = rmsnorm_shared_object_path(cache_root)
    manifest = load_cache_manifest(cache_root)
    if manifest is None:
        raise ContractError("kernel cache manifest is missing")
    manifest_path = cache_dir(cache_root) / MANIFEST_NAME
    _validate_runtime()

    row_results = [_benchmark_rows(row_count, cache_root=cache_root, config=config) for row_count in rows]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "op": "rmsnorm",
        "kind": "forward_only_benchmark",
        "run_id": spec.run_id,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "hidden_size": RMSNORM_HIDDEN_SIZE,
        "dtype": "bfloat16",
        "warmup": warmup,
        "iters": iters,
        "cache_dir": cache_dir(cache_root).as_posix(),
        "cache_manifest": manifest_path.as_posix(),
        "cache_manifest_sha256": _sha256(manifest_path) if manifest_path.exists() else None,
        "shared_object": shared_object.relative_to(cache_dir(cache_root)).as_posix(),
        "shared_object_sha256": _sha256(shared_object),
        "rows": row_results,
    }
    if write_artifact:
        artifact_path = spec.dirs.run_dir / "diagnostics" / "kernel_benchmarks" / "rmsnorm.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload["artifact_path"] = artifact_path.as_posix()
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def benchmark_to_json(payload: dict[str, Any]) -> str:
    """Serialize a benchmark payload as stable JSON."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def format_rmsnorm_benchmark(payload: dict[str, Any]) -> str:
    """Format an RMSNorm benchmark summary for humans."""

    lines = [
        (
            "rmsnorm benchmark: "
            f"backend={payload['backend']} dtype={payload['dtype']} hidden={payload['hidden_size']} "
            f"warmup={payload['warmup']} iters={payload['iters']}"
        )
    ]
    for row in payload["rows"]:
        lines.append(
            "  "
            f"rows={row['rows']}: "
            f"jax_p50={row['jax_ms']['p50']:.4f}ms "
            f"tk_p50={row['tk_ms']['p50']:.4f}ms "
            f"speedup={row['speedup_p50']:.3f}x "
            f"max_abs_error={row['max_abs_error']:.6f}"
        )
    if "artifact_path" in payload:
        lines.append(f"artifact: {payload['artifact_path']}")
    return "\n".join(lines)


def parse_rows(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive row list."""

    try:
        rows = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ContractError("rows must be a comma-separated list of positive integers") from exc
    if not rows or any(row <= 0 for row in rows):
        raise ContractError("rows must contain at least one positive integer")
    return rows


def _validate_benchmark_config(config: RmsNormBenchmarkConfig) -> None:
    if not config.rows or any(row <= 0 for row in config.rows):
        raise ContractError("RMSNorm benchmark rows must be positive")
    if config.warmup < 0:
        raise ContractError("RMSNorm benchmark warmup must be non-negative")
    if config.iters <= 0:
        raise ContractError("RMSNorm benchmark iters must be positive")


def _validate_runtime() -> None:
    try:
        backend = jax.default_backend()
    except RuntimeError as exc:
        raise ContractError(f"RMSNorm TK benchmark requires JAX gpu backend: {exc}") from exc
    if backend != "gpu":
        raise ContractError("RMSNorm TK benchmark requires JAX gpu backend")


def _benchmark_rows(
    rows: int,
    *,
    cache_root: str | Path | None,
    config: RmsNormBenchmarkConfig,
) -> dict[str, Any]:
    x = jnp.arange(rows * RMSNORM_HIDDEN_SIZE, dtype=jnp.float32).reshape(rows, RMSNORM_HIDDEN_SIZE)
    x = ((x % 251.0) / 31.0).astype(jnp.bfloat16)
    weight = (1.0 + (jnp.arange(RMSNORM_HIDDEN_SIZE, dtype=jnp.float32) % 97.0) / 257.0).astype(jnp.bfloat16)

    jax_fn = jax.jit(lambda a, b: rmsnorm_reference(a, b))
    tk_fn = jax.jit(lambda a, b: rmsnorm_tk_forward(a, b, cache_root=cache_root))

    jax_compile_ms = _compile_ms(jax_fn, x, weight)
    tk_compile_ms = _compile_ms(tk_fn, x, weight)
    _warmup(jax_fn, x, weight, config.warmup)
    _warmup(tk_fn, x, weight, config.warmup)
    jax_times = _measure(jax_fn, x, weight, config.iters)
    tk_times = _measure(tk_fn, x, weight, config.iters)

    expected = jax_fn(x, weight).block_until_ready()
    got = tk_fn(x, weight).block_until_ready()
    max_abs_error = float(jnp.max(jnp.abs(got.astype(jnp.float32) - expected.astype(jnp.float32))))
    return {
        "rows": rows,
        "shape": [rows, RMSNORM_HIDDEN_SIZE],
        "jax_compile_ms": jax_compile_ms,
        "tk_compile_ms": tk_compile_ms,
        "jax_ms": _latency_stats(jax_times),
        "tk_ms": _latency_stats(tk_times),
        "speedup_p50": _safe_speedup(_latency_stats(jax_times)["p50"], _latency_stats(tk_times)["p50"]),
        "max_abs_error": max_abs_error,
    }


def _compile_ms(fn, x: jax.Array, weight: jax.Array) -> float:
    start = time.perf_counter()
    fn(x, weight).block_until_ready()
    return (time.perf_counter() - start) * 1000.0


def _warmup(fn, x: jax.Array, weight: jax.Array, warmup: int) -> None:
    for _ in range(warmup):
        fn(x, weight).block_until_ready()


def _measure(fn, x: jax.Array, weight: jax.Array, iters: int) -> list[float]:
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(x, weight).block_until_ready()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _latency_stats(times: list[float]) -> dict[str, float]:
    values = sorted(times)
    return {
        "min": values[0],
        "p50": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
    }


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _safe_speedup(jax_ms: float, tk_ms: float) -> float | None:
    if tk_ms <= 0.0:
        return None
    return jax_ms / tk_ms
