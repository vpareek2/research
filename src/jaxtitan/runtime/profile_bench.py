"""Deterministic, opt-in microbenchmarks for performance-sensitive paths."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
import time
from typing import Any, Callable, Mapping

from jaxtitan.errors import ContractError


PROFILE_BENCHMARK_SCHEMA_VERSION = 2


def benchmark_component(
    component: str,
    *,
    warmup: int = 3,
    iters: int = 10,
    artifact_dir: str | Path | None = None,
    trace: bool = False,
) -> dict[str, Any]:
    """Benchmark fixed synthetic cases for ``moe`` or ``muon``."""

    if component not in {"moe", "muon"}:
        raise ContractError(f"unsupported profile benchmark component {component!r}")
    if warmup < 0:
        raise ContractError(f"profile benchmark warmup must be non-negative, got {warmup}")
    if iters <= 0:
        raise ContractError(f"profile benchmark iterations must be positive, got {iters}")
    if artifact_dir is not None and component != "muon":
        raise ContractError("--artifact-dir is currently supported only for the Muon profile benchmark")
    if trace and artifact_dir is None:
        raise ContractError("profile benchmark tracing requires --artifact-dir")
    if trace and component != "muon":
        raise ContractError("profile benchmark tracing is currently supported only for Muon")

    import jax

    if jax.local_device_count() < 4:
        raise ContractError(
            "profile benchmark requires four local JAX devices; for CPU use "
            "JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4"
        )
    if component == "moe":
        cases = _benchmark_moe(warmup=warmup, iters=iters)
        correctness_is_checked = False
        known_correctness_constraint = None
        measurement_contract = None
    else:
        from jaxtitan.runtime.muon_bench import (
            artifact_manifest,
            benchmark_contract,
            benchmark_muon,
            validate_artifact_manifest,
            write_benchmark_artifacts,
        )

        cases = benchmark_muon(
            warmup=warmup,
            iters=iters,
            artifact_dir=artifact_dir,
            trace=trace,
        )
        correctness_is_checked = True
        known_correctness_constraint = "duplicated_five_step_calibrated_envelope"
        measurement_contract = benchmark_contract()
    device = jax.devices()[0]
    payload = {
        "schema_version": PROFILE_BENCHMARK_SCHEMA_VERSION,
        "component": component,
        "kind": "directional_local_microbenchmark",
        "backend": jax.default_backend(),
        "device_kind": getattr(device, "device_kind", str(device)),
        "device_count": 4,
        "warmup": warmup,
        "iters": iters,
        "timing_is_acceptance_gate": False,
        "correctness_is_checked": correctness_is_checked,
        "known_correctness_constraint": known_correctness_constraint,
        "measurement_contract": measurement_contract,
        "cases": cases,
    }
    if artifact_dir is not None:
        manifest = artifact_manifest(artifact_dir)
        if component == "muon":
            validate_artifact_manifest(cases, manifest)
        payload["artifacts"] = manifest
        write_benchmark_artifacts(artifact_dir, payload=payload)
    return payload


def benchmark_to_json(payload: Mapping[str, Any]) -> str:
    """Serialize benchmark output deterministically."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def format_benchmark(payload: Mapping[str, Any]) -> str:
    """Format benchmark cases for humans."""

    lines = [
        f"profile benchmark: component={payload['component']} backend={payload['backend']} "
        f"device={payload['device_kind']} timing_gate=false "
        f"correctness_check={str(bool(payload['correctness_is_checked'])).lower()}"
    ]
    for case in payload["cases"]:
        if "candidates" in case:
            lines.append(
                f"  {case['name']}: shape={case['shape']} placement={case['partition_spec']}"
            )
            for candidate in case["candidates"]:
                lines.append(
                    f"    {candidate['execution']}: compile={candidate['compile_sec']:.3f}s "
                    f"median={candidate['timing_ms']['median']:.3f}ms "
                    f"p95={candidate['timing_ms']['p95']:.3f}ms "
                    f"correct={str(candidate['correctness_gate']).lower()}"
                )
        else:
            lines.append(
                f"  {case['name']}: compile={case['compile_sec']:.3f}s "
                f"median={case['timing_ms']['median']:.3f}ms "
                f"p10={case['timing_ms']['p10']:.3f}ms p90={case['timing_ms']['p90']:.3f}ms"
            )
    return "\n".join(lines)


def _benchmark_moe(*, warmup: int, iters: int) -> list[dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from jaxtitan.models.components.moe import _all_to_all_expert_swiglu

    hidden_size = 32
    intermediate_size = 64
    num_experts = 4
    batch_size = 1
    seq_len = 16
    top_k = 2
    base_gate = jnp.arange(num_experts * hidden_size * intermediate_size, dtype=jnp.float32).reshape(
        num_experts,
        hidden_size,
        intermediate_size,
    ) / 4096.0
    base_up = jnp.flip(base_gate, axis=-1)
    base_down = jnp.arange(num_experts * intermediate_size * hidden_size, dtype=jnp.float32).reshape(
        num_experts,
        intermediate_size,
        hidden_size,
    ) / 4096.0
    base_x = jnp.arange(batch_size * seq_len * hidden_size, dtype=jnp.float32).reshape(
        batch_size,
        seq_len,
        hidden_size,
    ) / 512.0
    weights = jnp.full((batch_size, seq_len, top_k), 0.5, dtype=jnp.float32)
    balanced_ids = jnp.reshape(jnp.arange(batch_size * seq_len * top_k, dtype=jnp.int32), weights.shape) % num_experts
    skewed_ids = jnp.zeros_like(balanced_ids)

    cases = []
    case_specs = (
        ("ep4_balanced", (1, 4), ("data", "ep"), None, balanced_ids),
        ("ep4_skewed", (1, 4), ("data", "ep"), None, skewed_ids),
        ("tp2_ep2_balanced", (1, 2, 2), ("data", "tp", "ep"), "tp", balanced_ids),
        ("tp2_ep2_skewed", (1, 2, 2), ("data", "tp", "ep"), "tp", skewed_ids),
    )
    devices = np.asarray(jax.devices()[:4], dtype=object)
    for name, mesh_shape, axis_names, expert_fsdp_axis, ids in case_specs:
        mesh = Mesh(devices.reshape(mesh_shape), axis_names)
        gate_spec = P("ep", None, expert_fsdp_axis)
        down_spec = P("ep", expert_fsdp_axis, None)
        x = jax.device_put(base_x, NamedSharding(mesh, P("data", None, None)))
        expert_ids = jax.device_put(ids, NamedSharding(mesh, P("data", None, None)))
        route_weights = jax.device_put(weights, NamedSharding(mesh, P("data", None, None)))
        gate = jax.device_put(base_gate, NamedSharding(mesh, gate_spec))
        up = jax.device_put(base_up, NamedSharding(mesh, gate_spec))
        down = jax.device_put(base_down, NamedSharding(mesh, down_spec))

        def loss_fn(local_x, local_ids, local_weights, local_gate, local_up, local_down):
            output = _all_to_all_expert_swiglu(
                x=local_x,
                expert_ids=local_ids,
                weights=local_weights,
                gate=local_gate,
                up=local_up,
                down=local_down,
                mesh=mesh,
                axis_name="ep",
                expert_fsdp_axis_name=expert_fsdp_axis,
                context_parallel_axis_name=None,
            )
            return jnp.mean(jnp.square(output))

        compiled, compile_sec, hlo = _compile(
            jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 2, 3, 4, 5))),
            (x, expert_ids, route_weights, gate, up, down),
        )
        timing = _time_calls(compiled, (x, expert_ids, route_weights, gate, up, down), warmup=warmup, iters=iters)
        cases.append(
            {
                "name": name,
                "mesh": dict(zip(axis_names, mesh_shape, strict=True)),
                "shape": {
                    "tokens": batch_size * seq_len,
                    "hidden": hidden_size,
                    "intermediate": intermediate_size,
                    "experts": num_experts,
                    "top_k": top_k,
                },
                "workload": "forward_backward",
                "compile_sec": compile_sec,
                "timing_ms": timing,
                "hlo": hlo,
            }
        )
    return cases


def _compile(function: Any, args: tuple[Any, ...]) -> tuple[Any, float, dict[str, Any]]:
    from jaxtitan.runtime.profile_analysis import summarize_hlo_text

    started = time.perf_counter()
    compiled = function.lower(*args).compile()
    compile_sec = time.perf_counter() - started
    return compiled, compile_sec, summarize_hlo_text(compiled.as_text())


def _time_calls(function: Callable[..., Any], args: tuple[Any, ...], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        _block(function(*args))
    times = []
    for _ in range(iters):
        started = time.perf_counter()
        _block(function(*args))
        times.append((time.perf_counter() - started) * 1000.0)
    return _timing_summary(times)


def _block(value: Any) -> None:
    import jax

    for leaf in jax.tree.leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()


def _timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": float(median(ordered)),
        "p10": _percentile(ordered, 0.10),
        "p90": _percentile(ordered, 0.90),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
