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
from jaxtitan.kernels import kernel_plan
from jaxtitan.models import ParamMetadata, count_parameters
from jaxtitan.models.execution import (
    expert_parallel_capacity_policy,
    expert_parallel_dispatcher_backend,
    expert_parallel_policy_payload,
    moe_tensor_parallel_policy_payload,
)
from jaxtitan.specs.parallelism import resolve_expert_fsdp_axis, resolve_expert_parallel_axis
from jaxtitan.optim import optimizer_policy_summary
from jaxtitan.runtime.profiling import profiling_runtime_summary
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


def _moe_balance_payload(spec: ModelSpec) -> dict[str, Any]:
    trinity = spec.trinity
    if trinity is None or trinity.moe is None:
        return {"name": "none"}
    balance = trinity.moe.balance
    return {
        "name": balance.name,
        "load_lr": balance.load_lr,
        "momentum": balance.momentum,
        "clamp": balance.clamp,
        "sequence_aux_loss_weight": balance.sequence_aux_loss_weight,
    }


def build_runtime_diagnostics(
    spec: RunSpec,
    context: Any,
    metadata: tuple[ParamMetadata, ...],
    *,
    optimizer: Any | None = None,
    sharding: Any | None = None,
    data_pipeline: Mapping[str, Any] | None = None,
    wandb: Mapping[str, Any] | None = None,
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
            "datasets": _package_version("datasets"),
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
            "ep_axis_size": context.ep_axis_size,
            "tp_axis_size": context.tp_axis_size,
            "cp_axis_size": context.cp_axis_size,
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
            "moe_balance": _moe_balance_payload(spec.model),
        },
        "training": {
            "loss": {
                "z_loss_weight": spec.training.loss.z_loss_weight,
            },
        },
        "optimizer": optimizer_policy_summary(
            spec.optimizer,
            None if optimizer is None else optimizer.route_assignments,
            execution_plans=None if optimizer is None else optimizer.muon_execution_plans,
            parallelism_mode=spec.parallelism.mode,
            fsdp_axis_size=context.fsdp_axis_size,
            tp_axis_size=context.tp_axis_size,
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
        "profiling": profiling_runtime_summary(spec),
        "kernels": kernel_plan(spec, device_kind=device_kind),
        "sharding": None if sharding is None else sharding_policy_summary(sharding, has_moe=_model_has_moe(spec)),
        "data_pipeline": data_pipeline,
        "wandb": None if wandb is None else dict(wandb),
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
    axis_sizes = dict(zip(spec.mesh.axis_names, spec.mesh.axis_sizes, strict=True))
    expert_axis = resolve_expert_parallel_axis(spec.parallelism, axis_sizes)
    expert_fsdp_axis = resolve_expert_fsdp_axis(spec.parallelism, axis_sizes)
    has_moe = spec.model.trinity is not None and spec.model.trinity.moe is not None
    return _normalize(
        {
            "schema_version": 1,
            "mode": spec.parallelism.mode,
            "tensor_parallel": spec.parallelism.tensor_parallel,
            "context_parallel": spec.parallelism.context_parallel,
            "expert_parallel": spec.parallelism.expert_parallel,
            "tensor_parallel_policy": {
                "enabled": spec.parallelism.tensor_parallel,
                "axis": "tp" if spec.parallelism.tensor_parallel else None,
                "axis_size": axis_sizes.get("tp", 1) if spec.parallelism.tensor_parallel else 1,
                "residual_stream": _tp_residual_stream(spec),
                "sequence_parallel": {
                    "enabled": spec.parallelism.tensor_parallel and not spec.parallelism.context_parallel,
                    "activation_spec": _tp_sequence_activation_spec(spec),
                },
                "embedding": "replicated",
                "lm_head": "vocab_parallel" if spec.parallelism.tensor_parallel else "replicated",
                "loss_parallel": {
                    "enabled": spec.parallelism.tensor_parallel,
                    "mode": "exact_vocab_parallel" if spec.parallelism.tensor_parallel else None,
                },
                "routed_experts": "not_tensor_parallel_sharded",
                "optimizer": "muon_routes_to_dist_muon_exact" if spec.parallelism.tensor_parallel else None,
                "moe": moe_tensor_parallel_policy_payload(
                    tensor_parallel=spec.parallelism.tensor_parallel,
                    has_moe=has_moe,
                ),
            },
            "context_parallel_policy": {
                "enabled": spec.parallelism.context_parallel,
                "axis": "cp" if spec.parallelism.context_parallel else None,
                "axis_size": axis_sizes.get("cp", 1) if spec.parallelism.context_parallel else 1,
                "attention": "logical_spmd_exact" if spec.parallelism.context_parallel else None,
                "activation_spec": "batch,cp_sequence,hidden" if spec.parallelism.context_parallel else None,
                "batch_sharding": "batch,cp_sequence" if spec.parallelism.context_parallel else None,
                "kv_cache": "cp_sequence_sharded" if spec.parallelism.context_parallel else None,
                "inference": "checkpoint_eval_and_sampling" if spec.parallelism.context_parallel else None,
            },
            "expert_parallel_policy": expert_parallel_policy_payload(
                enabled=spec.parallelism.expert_parallel,
                axis_name=expert_axis.axis,
                axis_size=expert_axis.axis_size,
                axis_sharing=expert_axis.axis_sharing,
                expert_fsdp_axis_name=expert_fsdp_axis.axis,
                expert_fsdp_axis_size=expert_fsdp_axis.axis_size,
                expert_fsdp_axis_sharing=expert_fsdp_axis.axis_sharing,
                num_experts=_moe_num_experts(spec),
            ),
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
                "ep_axis_size": context.ep_axis_size,
                "tp_axis_size": context.tp_axis_size,
                "cp_axis_size": context.cp_axis_size,
                "tensor_parallel_axis": "tp" if spec.parallelism.tensor_parallel else None,
                "tensor_parallel_axis_size": axis_sizes.get("tp", 1) if spec.parallelism.tensor_parallel else 1,
                "context_parallel_axis": "cp" if spec.parallelism.context_parallel else None,
                "context_parallel_axis_size": axis_sizes.get("cp", 1) if spec.parallelism.context_parallel else 1,
                "expert_parallel_axis": expert_axis.axis,
                "expert_parallel_axis_size": expert_axis.axis_size,
                "expert_parallel_axis_sharing": expert_axis.axis_sharing,
                "expert_fsdp_axis": expert_fsdp_axis.axis,
                "expert_fsdp_axis_size": expert_fsdp_axis.axis_size,
                "expert_fsdp_axis_sharing": expert_fsdp_axis.axis_sharing,
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


def sharding_policy_summary(plan: Any, *, has_moe: bool = False) -> dict[str, Any]:
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
                "ep": None
                if not plan.parallelism.expert_parallel
                else {
                    "enabled": True,
                    "axis": plan.expert_parallel_axis,
                    "axis_size": plan.expert_parallel_axis_size,
                    "axis_sharing": plan.expert_parallel_axis_sharing,
                    "expert_fsdp_axis": plan.expert_fsdp_axis,
                    "expert_fsdp_axis_size": plan.expert_fsdp_axis_size,
                    "expert_fsdp_axis_sharing": plan.expert_fsdp_axis_sharing,
                    "dispatcher_backend": expert_parallel_dispatcher_backend(plan.expert_parallel_axis_sharing),
                    "capacity_policy": expert_parallel_capacity_policy(plan.expert_parallel_axis_sharing),
                },
                "tp": None
                if not plan.parallelism.tensor_parallel
                else {
                    "enabled": True,
                    "axis": plan.tensor_parallel_axis,
                    "axis_size": plan.tensor_parallel_axis_size,
                    "residual_stream": "cp_sequence_parallel" if plan.parallelism.context_parallel else "sequence_parallel",
                    "sequence_parallel": {
                        "enabled": not plan.parallelism.context_parallel,
                        "activation_spec": None
                        if plan.parallelism.context_parallel
                        else "batch,sequence,tp_hidden",
                    },
                    "embedding": "replicated",
                    "lm_head": "vocab_parallel",
                    "loss_parallel": {
                        "enabled": True,
                        "mode": "exact_vocab_parallel",
                    },
                    "routed_experts": "not_tensor_parallel_sharded",
                    "optimizer": "muon_routes_to_dist_muon_exact",
                    "moe": moe_tensor_parallel_policy_payload(
                        tensor_parallel=True,
                        has_moe=has_moe,
                    ),
                },
                "cp": None
                if not plan.parallelism.context_parallel
                else {
                    "enabled": True,
                    "axis": plan.context_parallel_axis,
                    "axis_size": plan.context_parallel_axis_size,
                    "attention": "logical_spmd_exact",
                    "activation_spec": "batch,cp_sequence,hidden",
                    "batch_sharding": "batch,cp_sequence",
                    "kv_cache": "cp_sequence_sharded",
                    "inference": "checkpoint_eval_and_sampling",
                },
                "kv_cache": None
                if not plan.parallelism.context_parallel
                else {
                    "enabled": True,
                    "layout": "layer,batch,cp_cache_sequence,kv_heads,head_dim",
                    "partition_spec": "PartitionSpec(None, 'data', 'cp', None, None)",
                    "lengths_partition_spec": "PartitionSpec('data')",
                },
            },
        }
    )


def _tp_residual_stream(spec: RunSpec) -> str:
    if not spec.parallelism.tensor_parallel:
        return "replicated"
    if spec.parallelism.context_parallel:
        return "cp_sequence_parallel"
    return "sequence_parallel"


def _tp_sequence_activation_spec(spec: RunSpec) -> str | None:
    if not spec.parallelism.tensor_parallel or spec.parallelism.context_parallel:
        return None
    return "batch,sequence,tp_hidden"


def _state_policy_summary(plan: Any, *, placement: str) -> dict[str, Any]:
    shardings = tuple(plan.param_shardings.values())
    if placement == "model" and plan.parallelism.mode != "fsdp":
        fsdp_sharded = 0
    elif placement in {"optimizer", "gradients"} and plan.parallelism.mode == "ddp":
        fsdp_sharded = 0
    else:
        fsdp_sharded = sum(1 for sharding in shardings if "fsdp" in str(getattr(sharding, "spec", "")))
    expert_axis = plan.expert_parallel_axis
    if plan.parallelism.expert_parallel and expert_axis is not None:
        expert_shardings = tuple(
            plan.param_shardings[path]
            for path in getattr(plan, "expert_param_paths", ())
            if path in plan.param_shardings
        )
        ep_sharded = sum(1 for sharding in expert_shardings if expert_axis in str(getattr(sharding, "spec", "")))
    else:
        ep_sharded = 0
        expert_shardings = ()
    expert_fsdp_axis = plan.expert_fsdp_axis
    expert_fsdp_sharded = 0
    if expert_fsdp_axis is not None:
        expert_fsdp_sharded = sum(
            1 for sharding in expert_shardings if expert_fsdp_axis in str(getattr(sharding, "spec", ""))
        )
    tp_sharded = sum(1 for sharding in shardings if "tp" in str(getattr(sharding, "spec", "")))
    overlap = ep_sharded if expert_axis == "fsdp" and plan.parallelism.expert_parallel else 0
    replicated = len(shardings) - fsdp_sharded - ep_sharded - expert_fsdp_sharded - tp_sharded + overlap
    return {
        "mode": plan.parallelism.mode,
        "tensor_parallel": plan.parallelism.tensor_parallel,
        "tensor_parallel_axis": plan.tensor_parallel_axis,
        "context_parallel": plan.parallelism.context_parallel,
        "context_parallel_axis": plan.context_parallel_axis,
        "expert_parallel": plan.parallelism.expert_parallel,
        "expert_parallel_axis": plan.expert_parallel_axis,
        "expert_parallel_axis_sharing": plan.expert_parallel_axis_sharing,
        "expert_fsdp_axis": plan.expert_fsdp_axis,
        "expert_fsdp_axis_sharing": plan.expert_fsdp_axis_sharing,
        "partition_spec": _partition_spec_string(getattr(plan.replicated, "spec", None)),
        "parameter_leaves": len(shardings),
        "fsdp_sharded_leaves": fsdp_sharded,
        "ep_sharded_leaves": ep_sharded,
        "expert_fsdp_sharded_leaves": expert_fsdp_sharded,
        "tp_sharded_leaves": tp_sharded,
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

    suffixes = []
    if spec.parallelism.expert_parallel:
        suffixes.append("ep")
    if spec.parallelism.tensor_parallel:
        suffixes.append("tp")
    if spec.parallelism.context_parallel:
        suffixes.append("cp")
    if spec.parallelism.mode in {"zero2", "fsdp"}:
        return spec.parallelism.mode if not suffixes else f"{spec.parallelism.mode}+{'+'.join(suffixes)}"
    if suffixes:
        return f"replicated_data_parallel+{'+'.join(suffixes)}"
    return EXECUTION_MODE


def _moe_num_experts(spec: RunSpec) -> int | None:
    trinity = spec.model.trinity
    if trinity is None or trinity.moe is None:
        return None
    return trinity.moe.num_experts


def _model_has_moe(spec: RunSpec) -> bool:
    trinity = spec.model.trinity
    return trinity is not None and trinity.moe is not None


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
