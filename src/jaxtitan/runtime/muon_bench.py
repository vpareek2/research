"""Correctness-checked distributed-Muon leaf microbenchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Any

from jaxtitan.optim.muon import MuonLeafExecutionPlan, distributed_muon_transform, muon_policy_constants
from jaxtitan.runtime.profile_analysis import summarize_hlo_text


MUON_BENCHMARK_UPDATE_ATOL = 6e-4
MUON_BENCHMARK_PARAMETER_ATOL = 1.25e-3


@dataclass(frozen=True, slots=True)
class _Topology:
    name: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Leaf:
    role: str
    shape: tuple[int, int]
    tp_partition_dim: int


@dataclass(slots=True)
class _CompiledCandidate:
    name: str
    execution: str
    compiled: Any
    compile_sec: float
    hlo_text: str
    hlo: dict[str, Any]
    memory: dict[str, int | None]
    params: Any
    grads: Any
    initial_state: Any
    plans: Any
    timing_values: list[float]


_TP4 = _Topology("tp4", (4,), ("tp",))
_FSDP2_TP2 = _Topology("fsdp2_tp2", (2, 2), ("fsdp", "tp"))
_TP2_EP2 = _Topology("tp2_ep2", (2, 2), ("tp", "ep"))

_CURRENT_LEAVES = (
    _Leaf("attention_kv", (1024, 256), 1),
    _Leaf("attention_q_gate", (1024, 1024), 1),
    _Leaf("attention_o", (1024, 1024), 0),
    _Leaf("shared_mlp_gate_up", (1024, 2048), 1),
    _Leaf("shared_mlp_down", (2048, 1024), 0),
    _Leaf("dense_mlp_gate_up", (1024, 4096), 1),
    _Leaf("dense_mlp_down", (4096, 1024), 0),
)
_COMPOSED_CONFIRMATION_LEAVES = (
    _CURRENT_LEAVES[0],
    _CURRENT_LEAVES[2],
    _CURRENT_LEAVES[5],
)
_GRAM_BUCKET_MAX_BYTES = 32 * 1024 * 1024


def benchmark_muon(
    *,
    warmup: int,
    iters: int,
    artifact_dir: str | Path | None = None,
    trace: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark exact Muon candidates for the current production leaf classes."""

    import jax
    import numpy as np

    devices = np.asarray(jax.devices()[:4], dtype=object)
    artifact_root = Path(artifact_dir) if artifact_dir is not None else None
    if artifact_root is not None:
        (artifact_root / "hlo").mkdir(parents=True, exist_ok=True)
        (artifact_root / "profiles").mkdir(parents=True, exist_ok=True)

    cases = []
    matrix = (
        (_TP4, _CURRENT_LEAVES),
        (_FSDP2_TP2, _COMPOSED_CONFIRMATION_LEAVES),
        (_TP2_EP2, _COMPOSED_CONFIRMATION_LEAVES),
    )
    for topology, leaves in matrix:
        mesh = jax.sharding.Mesh(devices.reshape(topology.shape), topology.axes)
        for leaf in leaves:
            cases.append(
                _benchmark_leaf(
                    topology=topology,
                    mesh=mesh,
                    leaf=leaf,
                    warmup=warmup,
                    iters=iters,
                    artifact_root=artifact_root,
                    trace=trace,
                )
            )
        for leaf, leaf_count in (
            (_CURRENT_LEAVES[0], 24),
            (_CURRENT_LEAVES[2], 12),
        ):
            cases.append(
                _benchmark_bucket(
                    topology=topology,
                    mesh=mesh,
                    leaf=leaf,
                    leaf_count=leaf_count,
                    warmup=warmup,
                    iters=iters,
                    artifact_root=artifact_root,
                    trace=trace,
                )
            )
    return cases


def _benchmark_leaf(
    *,
    topology: _Topology,
    mesh: Any,
    leaf: _Leaf,
    warmup: int,
    iters: int,
    artifact_root: Path | None,
    trace: bool,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    partition_spec = P("tp", None) if leaf.tp_partition_dim == 0 else P(None, "tp")
    sharding = NamedSharding(mesh, partition_spec)
    element_count = math.prod(leaf.shape)
    key = jax.random.key(element_count + leaf.tp_partition_dim * 17)
    base_param = jax.random.normal(key, leaf.shape, dtype=jnp.float32)
    base_grad = jnp.arange(element_count, dtype=jnp.float32).reshape(leaf.shape) / float(element_count)
    params = {"w": jax.device_put(base_param, sharding)}
    grads = {"w": jax.device_put(jnp.flip(base_grad, axis=-1), sharding)}
    candidates = [
        _compile_candidate(
            topology=topology,
            leaf=leaf,
            sharding=sharding,
            params=params,
            grads=grads,
            execution=execution,
            artifact_root=artifact_root,
        )
        for execution in _eligible_executions(
            shape=leaf.shape,
            tp_partition_dim=leaf.tp_partition_dim,
            tp_size=int(mesh.shape["tp"]),
        )
    ]
    correctness = _check_candidates(candidates)
    _time_candidates(
        candidates,
        warmup=warmup,
        iters=iters,
        artifact_root=artifact_root,
        trace=trace,
    )

    rows = []
    for candidate in candidates:
        result = correctness[candidate.execution]
        result.update(
            {
                "name": candidate.name,
                "execution": candidate.execution,
                "compile_sec": candidate.compile_sec,
                "timing_ms": _timing_summary(candidate.timing_values),
                "hlo": candidate.hlo,
                "memory": candidate.memory,
                "collective_operand_model": _collective_operand_model(
                    shape=leaf.shape,
                    execution=candidate.execution,
                    tp_size=int(mesh.shape["tp"]),
                ),
            }
        )
        rows.append(result)
    return {
        "kind": "leaf",
        "name": f"{topology.name}_{leaf.role}",
        "topology": topology.name,
        "mesh": dict(zip(topology.axes, topology.shape, strict=True)),
        "role": leaf.role,
        "shape": list(leaf.shape),
        "partition_spec": str(partition_spec),
        "tp_partition_dim": leaf.tp_partition_dim,
        "canonical_tp_dim": candidates[0].plans["w"].canonical_tp_dim,
        "workload": "five_step_correctness_and_optimizer_update_latency",
        "reference": "duplicated",
        "candidates": rows,
    }


def _benchmark_bucket(
    *,
    topology: _Topology,
    mesh: Any,
    leaf: _Leaf,
    leaf_count: int,
    warmup: int,
    iters: int,
    artifact_root: Path | None,
    trace: bool,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    from jax.sharding import NamedSharding, PartitionSpec as P

    partition_spec = P("tp", None) if leaf.tp_partition_dim == 0 else P(None, "tp")
    sharding = NamedSharding(mesh, partition_spec)
    element_count = math.prod(leaf.shape)
    base_param = jax.random.normal(
        jax.random.key(element_count + leaf_count),
        leaf.shape,
        dtype=jnp.float32,
    )
    base_grad = jax.random.normal(
        jax.random.key(element_count + leaf_count + 1),
        leaf.shape,
        dtype=jnp.float32,
    )
    params = {
        f"w{index:03d}": jax.device_put(
            base_param + jnp.float32(index * 1e-4),
            sharding,
        )
        for index in range(leaf_count)
    }
    grads = {
        f"w{index:03d}": jax.device_put(
            base_grad + jnp.float32(index * 2e-4),
            sharding,
        )
        for index in range(leaf_count)
    }
    candidates = []
    for execution in _eligible_executions(
        shape=leaf.shape,
        tp_partition_dim=leaf.tp_partition_dim,
        tp_size=int(mesh.shape["tp"]),
    ):
        gram_dimension = (
            max(leaf.shape)
            if execution == "distributed_large_gram"
            else min(leaf.shape)
        )
        payload_bytes = gram_dimension * gram_dimension * jnp.dtype(jnp.bfloat16).itemsize
        leaves_per_bucket = max(1, _GRAM_BUCKET_MAX_BYTES // payload_bytes)
        plans = {}
        for index in range(leaf_count):
            plan = _make_plan(
                topology=topology,
                leaf=leaf,
                sharding=sharding,
                execution=execution,
            )
            plans[f"w{index:03d}"] = replace(
                plan,
                path=("benchmark", topology.name, leaf.role, f"w{index:03d}"),
                bucket_id=-1 if execution == "duplicated" else index // leaves_per_bucket,
            )
        candidates.append(
            _compile_tree_candidate(
                name=f"{topology.name}_{leaf.role}_bucket{leaf_count}_{execution}",
                execution=execution,
                params=params,
                grads=grads,
                plans=plans,
                artifact_root=artifact_root,
            )
        )

    correctness = _check_candidates(candidates)
    _time_candidates(
        candidates,
        warmup=warmup,
        iters=iters,
        artifact_root=artifact_root,
        trace=trace,
    )
    rows = []
    for candidate in candidates:
        result = correctness[candidate.execution]
        result.update(
            {
                "name": candidate.name,
                "execution": candidate.execution,
                "compile_sec": candidate.compile_sec,
                "timing_ms": _timing_summary(candidate.timing_values),
                "hlo": candidate.hlo,
                "memory": candidate.memory,
                "bucket_count": len(
                    {
                        plan.bucket_id
                        for plan in candidate.plans.values()
                        if plan.bucket_id >= 0
                    }
                ),
                "collective_operand_model": _collective_operand_model(
                    shape=leaf.shape,
                    execution=candidate.execution,
                    tp_size=int(mesh.shape["tp"]),
                ),
            }
        )
        rows.append(result)
    return {
        "kind": "bucket",
        "name": f"{topology.name}_{leaf.role}_bucket{leaf_count}",
        "topology": topology.name,
        "mesh": dict(zip(topology.axes, topology.shape, strict=True)),
        "role": leaf.role,
        "shape": list(leaf.shape),
        "partition_spec": str(partition_spec),
        "tp_partition_dim": leaf.tp_partition_dim,
        "canonical_tp_dim": candidates[0].plans["w000"].canonical_tp_dim,
        "leaf_count": leaf_count,
        "workload": "production_shape_bucket_optimizer_update_latency",
        "reference": "duplicated",
        "candidates": rows,
    }


def _compile_candidate(
    *,
    topology: _Topology,
    leaf: _Leaf,
    sharding: Any,
    params: Any,
    grads: Any,
    execution: str,
    artifact_root: Path | None,
) -> _CompiledCandidate:
    import jax
    import jax.numpy as jnp

    plan = _make_plan(
        topology=topology,
        leaf=leaf,
        sharding=sharding,
        execution=execution,
    )
    name = f"{topology.name}_{leaf.role}_{execution}"
    return _compile_tree_candidate(
        name=name,
        execution=execution,
        params=params,
        grads=grads,
        plans={"w": plan},
        artifact_root=artifact_root,
    )


def _compile_tree_candidate(
    *,
    name: str,
    execution: str,
    params: Any,
    grads: Any,
    plans: Any,
    artifact_root: Path | None,
) -> _CompiledCandidate:
    import jax
    import jax.numpy as jnp

    transform = distributed_muon_transform(
        lambda _count: jnp.asarray(0.02, dtype=jnp.float32),
        weight_decay=0.1,
        execution_plans=plans,
    )
    state = transform.init(params)

    def update_fn(local_params, local_grads, local_state):
        updates, next_state = transform.update(local_grads, local_state, params=local_params)
        next_params = jax.tree.map(lambda param, update: param + update, local_params, updates)
        return next_params, updates, next_state

    started = time.perf_counter()
    compiled = jax.jit(update_fn).lower(params, grads, state).compile()
    compile_sec = time.perf_counter() - started
    hlo_text = compiled.as_text()
    if artifact_root is not None:
        (artifact_root / "hlo" / f"{name}.txt").write_text(hlo_text, encoding="utf-8")
    return _CompiledCandidate(
        name=name,
        execution=execution,
        compiled=compiled,
        compile_sec=compile_sec,
        hlo_text=hlo_text,
        hlo=summarize_hlo_text(hlo_text),
        memory=_memory_analysis(compiled),
        params=params,
        grads=grads,
        initial_state=state,
        plans=plans,
        timing_values=[],
    )


def _make_plan(
    *,
    topology: _Topology,
    leaf: _Leaf,
    sharding: Any,
    execution: str,
) -> MuonLeafExecutionPlan:
    transpose_for_shape = leaf.shape[0] > leaf.shape[1]
    canonical_tp_dim = 1 - leaf.tp_partition_dim if transpose_for_shape else leaf.tp_partition_dim
    replica_axes = tuple(axis for axis in topology.axes if axis != "tp")
    return MuonLeafExecutionPlan(
        path=("benchmark", topology.name, leaf.role, execution),
        logical_shape=leaf.shape,
        parameter_sharding=sharding,
        gradient_sharding=sharding,
        momentum_sharding=sharding,
        update_sharding=sharding,
        parameter_replica_axes=replica_axes,
        gradient_replica_axes=replica_axes,
        momentum_replica_axes=replica_axes,
        update_replica_axes=replica_axes,
        tp_partition_dim=leaf.tp_partition_dim,
        transpose_for_shape=transpose_for_shape,
        canonical_tp_dim=canonical_tp_dim,
        requested_mode="benchmark",
        execution=execution,
        fallback_reason=None,
        bucket_id=0 if execution != "duplicated" else -1,
        weight_decay=True,
    )


def _eligible_executions(
    *,
    shape: tuple[int, int],
    tp_partition_dim: int,
    tp_size: int,
) -> tuple[str, ...]:
    transpose_for_shape = shape[0] > shape[1]
    canonical_tp_dim = 1 - tp_partition_dim if transpose_for_shape else tp_partition_dim
    if canonical_tp_dim == 1:
        return ("duplicated", "distributed_direct")
    candidates = ["duplicated", "distributed_large_gram"]
    if max(shape) % tp_size == 0:
        candidates.append("distributed_exchange")
    return tuple(candidates)


def _check_candidates(candidates: list[_CompiledCandidate]) -> dict[str, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    states = {candidate.execution: candidate.initial_state for candidate in candidates}
    params = {candidate.execution: candidate.params for candidate in candidates}
    maxima = {
        candidate.execution: {
            "max_update_abs_error": 0.0,
            "max_parameter_abs_error": 0.0,
            "momentum_exact": True,
            "finite": True,
            "replica_max_abs_error": 0.0,
        }
        for candidate in candidates
    }
    duplicated = next(candidate for candidate in candidates if candidate.execution == "duplicated")
    repeated_a = duplicated.compiled(duplicated.params, duplicated.grads, duplicated.initial_state)
    repeated_b = duplicated.compiled(duplicated.params, duplicated.grads, duplicated.initial_state)
    _block(repeated_a)
    _block(repeated_b)
    duplicated_deterministic = _trees_array_equal(repeated_a, repeated_b)

    for step_index in range(5):
        outputs = {}
        for candidate in candidates:
            execution = candidate.execution
            grad = jax.tree.map(
                lambda value: value + jnp.sin(value * (step_index + 1)) * jnp.float32(0.03 * step_index),
                candidate.grads,
            )
            next_params, updates, next_state = candidate.compiled(
                params[execution],
                grad,
                states[execution],
            )
            _block((next_params, updates, next_state))
            outputs[execution] = (next_params, updates, next_state)
            params[execution] = next_params
            states[execution] = next_state
            maxima[execution]["finite"] = maxima[execution]["finite"] and _tree_finite(
                (next_params, updates, next_state)
            )
            maxima[execution]["replica_max_abs_error"] = max(
                maxima[execution]["replica_max_abs_error"],
                _tree_replica_difference((next_params, updates, next_state)),
            )

        reference_params, reference_updates, reference_state = outputs["duplicated"]
        for candidate in candidates:
            execution = candidate.execution
            candidate_params, candidate_updates, candidate_state = outputs[execution]
            maxima[execution]["max_update_abs_error"] = max(
                maxima[execution]["max_update_abs_error"],
                _tree_max_abs_difference(candidate_updates, reference_updates),
            )
            maxima[execution]["max_parameter_abs_error"] = max(
                maxima[execution]["max_parameter_abs_error"],
                _tree_max_abs_difference(candidate_params, reference_params),
            )
            maxima[execution]["momentum_exact"] = maxima[execution]["momentum_exact"] and _trees_array_equal(
                candidate_state.momentum,
                reference_state.momentum,
            )

    results = {}
    for candidate in candidates:
        execution = candidate.execution
        candidate_result = maxima[execution]
        candidate_result["deterministic"] = (
            duplicated_deterministic
            if execution == "duplicated"
            else _candidate_is_deterministic(candidate)
        )
        candidate_result["output_partition_specs"] = sorted(
            {
                str(getattr(param, "sharding", None).spec)
                for param in jax.tree.leaves(params[execution])
            }
        )
        candidate_result["correctness_gate"] = bool(
            candidate_result["finite"]
            and candidate_result["momentum_exact"]
            and candidate_result["deterministic"]
            and candidate_result["replica_max_abs_error"] == 0.0
            and candidate_result["max_update_abs_error"] <= MUON_BENCHMARK_UPDATE_ATOL
            and candidate_result["max_parameter_abs_error"] <= MUON_BENCHMARK_PARAMETER_ATOL
            and _output_shardings_match(params[execution], candidate.plans)
        )
        results[execution] = candidate_result
    return results


def _candidate_is_deterministic(candidate: _CompiledCandidate) -> bool:
    first = candidate.compiled(candidate.params, candidate.grads, candidate.initial_state)
    second = candidate.compiled(candidate.params, candidate.grads, candidate.initial_state)
    _block(first)
    _block(second)
    return _trees_array_equal(first, second)


def _time_candidates(
    candidates: list[_CompiledCandidate],
    *,
    warmup: int,
    iters: int,
    artifact_root: Path | None,
    trace: bool,
) -> None:
    import jax

    current = {
        candidate.execution: (candidate.params, candidate.initial_state)
        for candidate in candidates
    }
    for candidate in candidates:
        params, state = current[candidate.execution]
        for _ in range(warmup):
            params, _updates, state = candidate.compiled(params, candidate.grads, state)
            _block((params, state))
        current[candidate.execution] = (params, state)

    tracing = trace and artifact_root is not None and jax.default_backend() == "gpu"
    if tracing:
        jax.profiler.start_trace(str(artifact_root / "profiles"), create_perfetto_link=False)
    try:
        for iteration in range(iters):
            for offset in range(len(candidates)):
                candidate = candidates[(iteration + offset) % len(candidates)]
                params, state = current[candidate.execution]
                started = time.perf_counter()
                with jax.profiler.TraceAnnotation(candidate.name):
                    params, _updates, state = candidate.compiled(params, candidate.grads, state)
                    _block((params, state))
                candidate.timing_values.append((time.perf_counter() - started) * 1000.0)
                current[candidate.execution] = (params, state)
    finally:
        if tracing:
            jax.profiler.stop_trace()


def _timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    center = float(median(ordered))
    deviations = sorted(abs(value - center) for value in ordered)
    return {
        "median": center,
        "p50": center,
        "p95": _percentile(ordered, 0.95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "median_abs_deviation": float(median(deviations)),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _collective_operand_model(
    *,
    shape: tuple[int, int],
    execution: str,
    tp_size: int,
) -> dict[str, Any]:
    elements = math.prod(shape)
    short = min(shape)
    long = max(shape)
    bf16_bytes = 2
    if execution == "duplicated":
        return {
            "kind": "logical_matrix_gather",
            "gather_local_operand_bytes_per_rank": elements * bf16_bytes // tp_size,
            "gather_result_bytes_per_rank": elements * bf16_bytes,
        }
    gram_dimension = long if execution == "distributed_large_gram" else short
    payload = {
        "kind": "scalar_norm_plus_five_gram_reductions",
        "norm_operand_bytes_per_rank": 4,
        "gram_dimension": gram_dimension,
        "gram_operand_bytes_per_rank_per_step": gram_dimension * gram_dimension * bf16_bytes,
        "gram_steps": int(muon_policy_constants()["newton_schulz_steps"]),
    }
    if execution == "distributed_exchange":
        payload["exchange_operand_bytes_per_rank_each_direction"] = elements * bf16_bytes // tp_size
        payload["exchange_directions"] = 2
    return payload


def _memory_analysis(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {
        field: int(getattr(analysis, field)) if getattr(analysis, field, None) is not None else None
        for field in fields
    }


def _block(value: Any) -> None:
    import jax

    for leaf in jax.tree.leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()


def _tree_finite(value: Any) -> bool:
    import jax
    import numpy as np

    return all(np.isfinite(np.asarray(jax.device_get(leaf))).all() for leaf in jax.tree.leaves(value))


def _tree_max_abs_difference(left: Any, right: Any) -> float:
    import jax
    import numpy as np

    differences = []
    for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        differences.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(jax.device_get(left_leaf), dtype=np.float32)
                        - np.asarray(jax.device_get(right_leaf), dtype=np.float32)
                    )
                )
            )
        )
    return max(differences, default=0.0)


def _trees_array_equal(left: Any, right: Any) -> bool:
    import jax
    import numpy as np

    return all(
        np.array_equal(np.asarray(jax.device_get(left_leaf)), np.asarray(jax.device_get(right_leaf)))
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _output_shardings_match(params: Any, plans: Any) -> bool:
    import jax

    return all(
        getattr(param, "sharding", None).spec == plan.update_sharding.spec
        for param, plan in zip(jax.tree.leaves(params), jax.tree.leaves(plans), strict=True)
    )


def _tree_replica_difference(value: Any) -> float:
    import jax

    return max((_array_replica_difference(leaf) for leaf in jax.tree.leaves(value)), default=0.0)


def _array_replica_difference(value: Any) -> float:
    import numpy as np

    shards = getattr(value, "addressable_shards", ())
    grouped: dict[str, list[np.ndarray]] = {}
    for shard in shards:
        grouped.setdefault(str(shard.index), []).append(np.asarray(shard.data))
    maximum = 0.0
    for replicas in grouped.values():
        if len(replicas) < 2:
            continue
        reference = replicas[0].astype(np.float32)
        maximum = max(
            maximum,
            max(float(np.max(np.abs(replica.astype(np.float32) - reference))) for replica in replicas[1:]),
        )
    return maximum


def benchmark_contract() -> dict[str, Any]:
    """Return the stable numerical and measurement policy."""

    return {
        "correctness_steps": 5,
        "update_atol": MUON_BENCHMARK_UPDATE_ATOL,
        "parameter_atol": MUON_BENCHMARK_PARAMETER_ATOL,
        "candidate_order": "rotating",
        "timing_reducer": "median",
        "tail_percentile": 95,
        "stability_metric": "median_abs_deviation_over_median",
        "stability_limit": 0.05,
        "selection_min_speedup": 1.05,
        "selection_tie_fraction": 0.03,
        "timing_is_production_acceptance_gate": False,
    }


def artifact_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """List benchmark artifacts without embedding machine-specific absolute paths."""

    root = Path(artifact_dir)
    return {
        "hlo_files": sorted(path.relative_to(root).as_posix() for path in (root / "hlo").glob("*.txt")),
        "profile_files": sorted(path.relative_to(root).as_posix() for path in (root / "profiles").rglob("*") if path.is_file()),
    }


def write_benchmark_artifacts(
    artifact_dir: str | Path,
    *,
    payload: dict[str, Any],
) -> None:
    """Write canonical JSON and human-readable benchmark summaries."""

    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
