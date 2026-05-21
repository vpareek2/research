"""Runtime diagnostics, timing, and throughput helpers."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
import platform
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jaxtitan import __version__
from jaxtitan.models import ParamMetadata, count_parameters
from jaxtitan.optim import optimizer_policy_summary
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.run import RunSpec

SECONDS_PER_DEVICE_HOUR = 3600.0
EXECUTION_MODE = "replicated_data_parallel"
METRICS_SCOPE = "global"
ARTIFACT_WRITER = "single_host"

# BF16 dense peak FLOPs. More-specific patterns must come before broad ones.
BF16_PEAK_FLOPS_TABLE = (
    (("rtx pro 6000", "blackwell"), 1.0e15),
    (("gb200",), 2.5e15),
    (("grace blackwell",), 2.5e15),
    (("b200",), 2.25e15),
    (("b100",), 1.8e15),
    (("h200", "nvl"), 836e12),
    (("h200", "pcie"), 836e12),
    (("h200",), 989e12),
    (("h100", "nvl"), 835e12),
    (("h100", "pcie"), 756e12),
    (("h100",), 989e12),
    (("h800", "nvl"), 989e12),
    (("h800",), 756e12),
    (("a100",), 312e12),
    (("a800",), 312e12),
    (("a40",), 149.7e12),
    (("a30",), 165e12),
    (("l40s",), 362e12),
    (("l40-s",), 362e12),
    (("l40 s",), 362e12),
    (("l4",), 121e12),
    (("mi355",), 2.5e15),
    (("mi325",), 1.3074e15),
    (("mi300x",), 1.3074e15),
    (("mi300a",), 980.6e12),
    (("mi250x",), 383e12),
    (("mi250",), 362.1e12),
    (("5090",), 209.5e12),
    (("4090",), 165.2e12),
    (("3090",), 71e12),
    (("gb10",), 125e12),
)

_NVML_UNSET = object()


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    """Stable runtime diagnostics artifact payload."""

    payload: dict[str, Any]


class PhaseTimer:
    """Small phase timer for one host-side runtime unit."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._phases: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (time.perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        self._phases[name] = self._phases.get(name, 0.0) + seconds

    def seconds(self, name: str) -> float:
        return self._phases.get(name, 0.0)

    def total_sec(self) -> float:
        return time.perf_counter() - self._started_at

    def to_dict(self) -> dict[str, float]:
        return dict(self._phases)


def sync_and_time(value: Any) -> float:
    """Block on JAX work and return synchronization elapsed seconds."""

    start = time.perf_counter()
    jax.block_until_ready(value)
    return time.perf_counter() - start


def estimate_flops_per_token(spec: ModelSpec, seq_len: int) -> int:
    """Estimate decoder training FLOPs per token from model shape metadata."""

    head_dim = spec.hidden_size // spec.num_heads
    kv_width = spec.n_kv_heads * head_dim
    q_proj = spec.hidden_size * spec.hidden_size
    k_proj = spec.hidden_size * kv_width
    v_proj = spec.hidden_size * kv_width
    o_proj = spec.hidden_size * spec.hidden_size
    mlp = 3 * spec.hidden_size * spec.intermediate_size
    lm_head = 0 if spec.tied_embeddings else spec.hidden_size * spec.vocab_size
    matmul_params = spec.num_layers * (q_proj + k_proj + v_proj + o_proj + mlp) + lm_head
    attention_flops = spec.num_layers * 12 * spec.num_heads * head_dim * seq_len
    return int(6 * matmul_params + attention_flops)


def peak_flops_for_device(device_name: str | None) -> float | None:
    """Return known BF16 dense peak FLOPs for a device name."""

    if not device_name:
        return None
    normalized = device_name.lower().replace("nvidia", "").strip()
    for patterns, flops in BF16_PEAK_FLOPS_TABLE:
        if all(pattern in normalized for pattern in patterns):
            return flops
    return None


def build_runtime_diagnostics(
    spec: RunSpec,
    context: Any,
    metadata: tuple[ParamMetadata, ...],
    *,
    optimizer: Any | None = None,
    sharding: Any | None = None,
    data_pipeline: Mapping[str, Any] | None = None,
) -> RuntimeDiagnostics:
    """Build the canonical runtime diagnostics payload for a run/preflight."""

    device_kind = selected_device_kind(context.devices)
    flops_per_token = estimate_flops_per_token(spec.model, spec.training.seq_len)
    peak_flops_per_device = peak_flops_for_device(device_kind)
    device_count = len(context.devices)
    peak_flops_total = None if peak_flops_per_device is None else peak_flops_per_device * device_count
    micro_tokens_per_step = spec.training.global_batch_size * spec.training.seq_len
    effective_tokens_per_step = micro_tokens_per_step * spec.training.gradient_accumulation_steps
    effective_global_batch_size = spec.training.global_batch_size * spec.training.gradient_accumulation_steps
    payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "run_id": spec.run_id,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "jaxtitan": __version__,
            "jax": getattr(jax, "__version__", None),
            "flax": _package_version("flax"),
            "optax": _package_version("optax"),
            "orbax_checkpoint": _package_version("orbax-checkpoint"),
            "grain": _package_version("grain"),
        },
        "jax": {
            "backend": jax.default_backend(),
            "process_count": context.process_count,
            "process_index": context.process_index,
            "single_process": context.process_count == 1,
            "local_device_count": context.local_device_count,
            "global_device_count": context.global_device_count,
            "selected_device_count": device_count,
            "addressable_device_count": len(jax.local_devices()),
        },
        "mesh": {
            "axis_names": list(spec.mesh.axis_names),
            "axis_sizes": list(spec.mesh.axis_sizes),
            "data_axis_size": context.data_axis_size,
            "fsdp_axis_size": context.fsdp_axis_size,
            "global_mesh_size": _product(spec.mesh.axis_sizes),
            "selected_device_count": device_count,
            "selected_addressable_device_count": device_count,
            "global_batch_size": spec.training.global_batch_size,
            "micro_global_batch_size": spec.training.global_batch_size,
            "effective_global_batch_size": effective_global_batch_size,
            "gradient_accumulation_steps": spec.training.gradient_accumulation_steps,
            "per_device_batch_size": spec.training.global_batch_size // context.data_axis_size,
            "micro_tokens_per_step": micro_tokens_per_step,
            "effective_tokens_per_step": effective_tokens_per_step,
            "global_tokens_per_step": effective_tokens_per_step,
            "per_device_tokens_per_step": effective_tokens_per_step // context.data_axis_size,
            "selected_devices": [_device_summary(device) for device in context.devices],
        },
        "model": {
            "name": spec.model.name,
            "variant": spec.model.variant,
            "parameters": count_parameters(metadata),
            "parameter_leaves": len(metadata),
            "seq_len": spec.training.seq_len,
            "global_batch_size": spec.training.global_batch_size,
            "effective_global_batch_size": effective_global_batch_size,
            "gradient_accumulation_steps": spec.training.gradient_accumulation_steps,
            "remat": spec.model.remat,
        },
        "optimizer": optimizer_policy_summary(
            spec.optimizer,
            None if optimizer is None else optimizer.route_assignments,
        ),
        "performance": {
            "device_kind": device_kind,
            "device_count": device_count,
            "flops_per_token": flops_per_token,
            "peak_flops_per_device": peak_flops_per_device,
            "peak_flops_total": peak_flops_total,
        },
        "compile": compile_contract_summary(spec, sharding),
        "parallelism": parallelism_summary(spec, context),
        "sharding": None if sharding is None else sharding_policy_summary(sharding),
        "data_pipeline": data_pipeline,
        "device_telemetry": sample_device_telemetry(context.devices),
    }
    return RuntimeDiagnostics(payload=_normalize(payload))


def compile_contract_summary(spec: RunSpec, sharding: Any | None) -> dict[str, Any]:
    """Summarize fixed JIT input contracts used by runtime/preflight."""

    train_shape = (
        spec.training.gradient_accumulation_steps,
        spec.training.global_batch_size,
        spec.training.seq_len,
    )
    eval_shape = (spec.training.global_batch_size, spec.training.seq_len)
    train_shardings = None
    eval_shardings = None
    if sharding is not None:
        train_shardings = {
            "state": {"mode": "from_template", **_sharding_summary(sharding.replicated)},
            "input_ids": _sharding_summary(sharding.batch.accumulated_input_ids),
            "target_ids": _sharding_summary(sharding.batch.accumulated_target_ids),
            "loss_mask": _sharding_summary(sharding.batch.accumulated_loss_mask),
        }
        eval_shardings = {
            "state": {"mode": "from_template", **_sharding_summary(sharding.replicated)},
            "input_ids": _sharding_summary(sharding.batch.input_ids),
            "target_ids": _sharding_summary(sharding.batch.target_ids),
            "loss_mask": _sharding_summary(sharding.batch.loss_mask),
        }
    return _normalize(
        {
            "schema_version": 1,
            "train": {
                "donate_state": True,
                "expected_batch_shape": train_shape,
                "input_shardings": train_shardings,
            },
            "eval": {
                "donate_state": False,
                "expected_batch_shape": eval_shape,
                "input_shardings": eval_shardings,
            },
        }
    )


def parallelism_summary(spec: RunSpec, context: Any) -> dict[str, Any]:
    """Summarize the runtime execution and artifact policy."""

    micro_tokens_per_step = spec.training.global_batch_size * spec.training.seq_len
    global_tokens_per_step = micro_tokens_per_step * spec.training.gradient_accumulation_steps
    effective_global_batch_size = spec.training.global_batch_size * spec.training.gradient_accumulation_steps
    return _normalize(
        {
            "schema_version": 1,
            "mode": spec.parallelism.mode,
            "execution_mode": runtime_execution_mode(spec),
            "metrics_scope": METRICS_SCOPE,
            "artifact_writer": ARTIFACT_WRITER,
            "single_process": context.process_count == 1,
            "process": {
                "count": context.process_count,
                "index": context.process_index,
            },
            "devices": {
                "selected": len(context.devices),
                "local": context.local_device_count,
                "global": context.global_device_count,
                "addressable": len(jax.local_devices()),
            },
            "mesh": {
                "axis_names": list(spec.mesh.axis_names),
                "axis_sizes": list(spec.mesh.axis_sizes),
                "data_axis_size": context.data_axis_size,
                "fsdp_axis_size": context.fsdp_axis_size,
            },
            "batch": {
                "global_batch_size": spec.training.global_batch_size,
                "micro_global_batch_size": spec.training.global_batch_size,
                "effective_global_batch_size": effective_global_batch_size,
                "gradient_accumulation_steps": spec.training.gradient_accumulation_steps,
                "per_device_batch_size": spec.training.global_batch_size // context.data_axis_size,
                "micro_tokens_per_step": micro_tokens_per_step,
                "effective_tokens_per_step": global_tokens_per_step,
                "global_tokens_per_step": global_tokens_per_step,
                "per_device_tokens_per_step": global_tokens_per_step // context.data_axis_size,
            },
            "host_artifacts": {
                "writer": ARTIFACT_WRITER,
                "records": METRICS_SCOPE,
            },
        }
    )


def sharding_policy_summary(plan: Any) -> dict[str, Any]:
    """Summarize intended shardings without dumping large trees."""

    replicated = _sharding_summary(plan.replicated)
    model_state = _state_policy_summary(plan, placement="model")
    optimizer_state = _state_policy_summary(plan, placement="optimizer")
    gradients = _state_policy_summary(plan, placement="gradients")
    return _normalize(
        {
            "schema_version": 1,
            "batch": {
                "input_ids": _sharding_summary(plan.batch.input_ids),
                "target_ids": _sharding_summary(plan.batch.target_ids),
                "loss_mask": _sharding_summary(plan.batch.loss_mask),
                "accumulated_input_ids": _sharding_summary(plan.batch.accumulated_input_ids),
                "accumulated_target_ids": _sharding_summary(plan.batch.accumulated_target_ids),
                "accumulated_loss_mask": _sharding_summary(plan.batch.accumulated_loss_mask),
            },
            "train_state": {
                "step": replicated,
                "tokens_seen": replicated,
                "model": model_state,
                "opt_state": optimizer_state,
                "rng": replicated,
                "schedule_state": replicated,
            },
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "gradients": gradients,
            "rng": replicated,
            "metrics": _sharding_summary(plan.metrics),
            "checkpoint": {
                "restore_template": replicated,
            },
            "reserved": {
                "fsdp": None
                if plan.parallelism.mode == "ddp"
                else {"enabled": True, "axis_size": plan.mesh.fsdp_axis_size},
                "tp": None,
                "kv_cache": None,
            },
        }
    )


def _state_policy_summary(plan: Any, *, placement: str) -> dict[str, Any]:
    shardings = tuple(plan.param_shardings.values())
    if placement == "model" and plan.parallelism.mode != "fsdp":
        fsdp_sharded = 0
    elif placement in {"optimizer", "gradients"} and plan.parallelism.mode == "ddp":
        fsdp_sharded = 0
    else:
        fsdp_sharded = sum(1 for sharding in shardings if "fsdp" in str(getattr(sharding, "spec", "")))
    replicated = len(shardings) - fsdp_sharded
    return {
        "mode": plan.parallelism.mode,
        "partition_spec": _partition_spec_string(getattr(plan.replicated, "spec", None)),
        "parameter_leaves": len(shardings),
        "fsdp_sharded_leaves": fsdp_sharded,
        "replicated_leaves": replicated,
    }


def placed_array_summary(value: Any) -> dict[str, Any]:
    """Summarize one placed array's global and addressable shard shape."""

    shards = tuple(getattr(value, "addressable_shards", ()) or ())
    shard_shapes = sorted({tuple(int(dim) for dim in shard.data.shape) for shard in shards})
    return _normalize(
        {
            "global_shape": tuple(int(dim) for dim in getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "unknown")),
            "sharding": _sharding_summary(getattr(value, "sharding", None)),
            "addressable_shard_count": len(shards),
            "unique_addressable_shard_shapes": shard_shapes,
        }
    )


def selected_device_kind(devices: Sequence[Any]) -> str:
    """Return a stable representative device kind string."""

    if not devices:
        return "unknown"
    device = devices[0]
    return str(getattr(device, "device_kind", None) or getattr(device, "platform", None) or device)


def sample_device_telemetry(devices: Sequence[Any], *, nvml_provider: Any = _NVML_UNSET) -> dict[str, Any]:
    """Sample best-effort device telemetry with explicit nulls for unavailable values."""

    jax_memory = _sample_jax_memory(devices)
    nvml = _sample_nvml(len(devices), provider=nvml_provider)
    return {
        "device_memory_used_bytes": jax_memory["memory_used_bytes"],
        "device_memory_peak_bytes": jax_memory["memory_peak_bytes"],
        "device_memory_limit_bytes": jax_memory["memory_limit_bytes"],
        "gpu_memory_used_bytes": nvml["gpu_memory_used_bytes"],
        "gpu_memory_total_bytes": nvml["gpu_memory_total_bytes"],
        "gpu_utilization_pct": nvml["gpu_utilization_pct"],
        "gpu_memory_utilization_pct": nvml["gpu_memory_utilization_pct"],
        "gpu_power_w": nvml["gpu_power_w"],
        "gpu_temperature_c": nvml["gpu_temperature_c"],
    }


def enrich_train_metrics(
    row: Mapping[str, Any],
    *,
    timings: Mapping[str, float],
    runtime: RuntimeDiagnostics | Mapping[str, Any],
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add timing, throughput, performance, and telemetry fields to a train row."""

    payload = runtime.payload if isinstance(runtime, RuntimeDiagnostics) else runtime
    performance = payload["performance"]
    parallelism = payload.get(
        "parallelism",
        {"mode": "ddp", "execution_mode": EXECUTION_MODE},
    )
    data_sec = float(timings.get("data_sec", 0.0))
    placement_sec = float(timings.get("placement_sec", 0.0))
    train_dispatch_sec = float(timings.get("train_dispatch_sec", 0.0))
    metrics_sync_sec = float(timings.get("metrics_sync_sec", 0.0))
    train_step_sec = float(timings.get("train_step_sec", train_dispatch_sec + metrics_sync_sec))
    step_sec = float(timings.get("step_sec", data_sec + placement_sec + train_step_sec))
    target_tokens = int(row["target_tokens"])
    examples = int(row["examples"])
    flops_per_token = performance["flops_per_token"]
    peak_flops_total = performance["peak_flops_total"]
    train_tokens_per_sec = _rate(target_tokens, train_step_sec)
    flops_per_sec = None if train_tokens_per_sec is None else flops_per_token * train_tokens_per_sec
    enriched = dict(row)
    enriched.update(
        {
            "parallelism_mode": parallelism["mode"],
            "execution_mode": parallelism["execution_mode"],
            "metrics_scope": METRICS_SCOPE,
            "artifact_writer": ARTIFACT_WRITER,
            "data_sec": data_sec,
            "placement_sec": placement_sec,
            "train_dispatch_sec": train_dispatch_sec,
            "metrics_sync_sec": metrics_sync_sec,
            "train_step_sec": train_step_sec,
            "step_sec": step_sec,
            "tokens_per_sec": _rate(target_tokens, step_sec),
            "train_tokens_per_sec": train_tokens_per_sec,
            "examples_per_sec": _rate(examples, step_sec),
            "flops_per_token": flops_per_token,
            "flops_per_step": flops_per_token * target_tokens,
            "flops_per_sec": flops_per_sec,
            "peak_flops_per_device": performance["peak_flops_per_device"],
            "peak_flops_total": peak_flops_total,
            "mfu": None if flops_per_sec is None or peak_flops_total in (None, 0) else 100.0 * flops_per_sec / peak_flops_total,
        }
    )
    enriched.update(_telemetry_fields(telemetry or payload["device_telemetry"]))
    return _normalize(enriched)


def enrich_eval_metrics(row: Mapping[str, Any], *, eval_sec: float) -> dict[str, Any]:
    """Add validation timing and throughput fields to an eval row."""

    enriched = dict(row)
    enriched.update(
        {
            "eval_sec": eval_sec,
            "eval_tokens_per_sec": _rate(int(row["target_tokens"]), eval_sec),
            "eval_examples_per_sec": _rate(int(row["examples"]), eval_sec),
        }
    )
    return _normalize(enriched)


def training_diagnostics_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_wall_sec: float,
    runtime: RuntimeDiagnostics | Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize logged training diagnostics for final summaries and events."""

    payload = runtime.payload if isinstance(runtime, RuntimeDiagnostics) else runtime
    performance = payload["performance"]
    parallelism = payload.get("parallelism", {"execution_mode": EXECUTION_MODE})
    final = rows[-1] if rows else {}
    steady_rows = rows[1:] if len(rows) >= 2 else ()
    return _normalize(
        {
            "total_wall_sec": total_wall_sec,
            "avg_train_tokens_per_sec": _mean_number(row.get("train_tokens_per_sec") for row in rows),
            "final_train_tokens_per_sec": _optional_number(final.get("train_tokens_per_sec")),
            "steady_train_tokens_per_sec": _mean_number(row.get("train_tokens_per_sec") for row in steady_rows),
            "avg_mfu": _mean_number(row.get("mfu") for row in rows),
            "final_mfu": _optional_number(final.get("mfu")),
            "avg_batch_het": _mean_number(row.get("batch_het") for row in rows),
            "final_batch_het": _optional_number(final.get("batch_het")),
            "device_kind": performance["device_kind"],
            "device_count": performance["device_count"],
            "runtime_diagnostics_path": "diagnostics/runtime.json",
            "execution_mode": parallelism["execution_mode"],
            "metrics_scope": METRICS_SCOPE,
            "artifact_writer": ARTIFACT_WRITER,
        }
    )


def runtime_execution_mode(spec: RunSpec) -> str:
    """Return the artifact-facing execution mode name."""

    if spec.parallelism.mode in {"zero2", "fsdp"}:
        return spec.parallelism.mode
    return EXECUTION_MODE


def _telemetry_fields(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "device_memory_used_bytes": telemetry.get("device_memory_used_bytes"),
        "device_memory_peak_bytes": telemetry.get("device_memory_peak_bytes"),
        "device_memory_limit_bytes": telemetry.get("device_memory_limit_bytes"),
        "gpu_memory_used_bytes": telemetry.get("gpu_memory_used_bytes"),
        "gpu_memory_total_bytes": telemetry.get("gpu_memory_total_bytes"),
        "gpu_utilization_pct": telemetry.get("gpu_utilization_pct"),
        "gpu_memory_utilization_pct": telemetry.get("gpu_memory_utilization_pct"),
        "gpu_power_w": telemetry.get("gpu_power_w"),
        "gpu_temperature_c": telemetry.get("gpu_temperature_c"),
    }


def _sample_jax_memory(devices: Sequence[Any]) -> dict[str, int | None]:
    used_values = []
    peak_values = []
    limit_values = []
    for device in devices:
        try:
            stats = device.memory_stats()
        except Exception:
            stats = None
        if not isinstance(stats, Mapping):
            continue
        used = _first_int(stats, ("bytes_in_use", "current_bytes_in_use"))
        peak = _first_int(stats, ("peak_bytes_in_use", "bytes_in_use"))
        limit = _first_int(stats, ("bytes_limit", "memory_limit"))
        if used is not None:
            used_values.append(used)
        if peak is not None:
            peak_values.append(peak)
        if limit is not None:
            limit_values.append(limit)
    return {
        "memory_used_bytes": sum(used_values) if used_values else None,
        "memory_peak_bytes": max(peak_values) if peak_values else None,
        "memory_limit_bytes": sum(limit_values) if limit_values else None,
    }


def _sample_nvml(device_count: int, *, provider: Any) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "gpu_memory_used_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_utilization_pct": None,
        "gpu_memory_utilization_pct": None,
        "gpu_power_w": None,
        "gpu_temperature_c": None,
    }
    nvml = _load_nvml() if provider is _NVML_UNSET else provider
    if nvml is None or device_count <= 0:
        return metrics
    try:
        nvml.nvmlInit()
        handles = [nvml.nvmlDeviceGetHandleByIndex(index) for index in range(device_count)]
    except Exception:
        return metrics
    memory_used = 0
    memory_total = 0
    gpu_utils = []
    memory_utils = []
    power_w = 0.0
    temperatures = []
    for handle in handles:
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used += int(memory.used)
            memory_total += int(memory.total)
        except Exception:
            pass
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utils.append(float(util.gpu))
            memory_utils.append(float(util.memory))
        except Exception:
            pass
        try:
            power_w += float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            pass
        try:
            temperatures.append(float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)))
        except Exception:
            pass
    if memory_total:
        metrics["gpu_memory_used_bytes"] = memory_used
        metrics["gpu_memory_total_bytes"] = memory_total
    if gpu_utils:
        metrics["gpu_utilization_pct"] = sum(gpu_utils) / len(gpu_utils)
    if memory_utils:
        metrics["gpu_memory_utilization_pct"] = sum(memory_utils) / len(memory_utils)
    if power_w:
        metrics["gpu_power_w"] = power_w
    if temperatures:
        metrics["gpu_temperature_c"] = max(temperatures)
    return metrics


def _load_nvml() -> Any | None:
    try:
        import pynvml
    except Exception:
        return None
    return pynvml


def _device_summary(device: Any) -> dict[str, Any]:
    return {
        "id": getattr(device, "id", None),
        "platform": str(getattr(device, "platform", "unknown")),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "process_index": getattr(device, "process_index", None),
    }


def _sharding_summary(sharding: Any) -> dict[str, Any] | None:
    if sharding is None:
        return None
    return {
        "type": type(sharding).__name__,
        "partition_spec": _partition_spec_string(getattr(sharding, "spec", None)),
    }


def _partition_spec_string(spec: Any) -> str | None:
    if spec is None:
        return None
    text = str(spec)
    if text == "P()":
        return "PartitionSpec()"
    if text.startswith("P("):
        return f"PartitionSpec({text[2:]}"
    return text


def _product(values: Sequence[int]) -> int:
    product = 1
    for value in values:
        product *= int(value)
    return product


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _first_int(stats: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = stats.get(key)
        if isinstance(value, int):
            return value
    return None


def _rate(numerator: int | float, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return float(numerator) / seconds


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _mean_number(values: Sequence[Any] | Any) -> float | None:
    numbers = [_optional_number(value) for value in values]
    valid = [value for value in numbers if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
