"""Model execution policy helpers."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxtitan.errors import ContractError

EXPERT_PARALLEL_DISPATCHER_BACKEND = "all_to_all"
RDEP_STATIC_DISPATCHER_BACKEND = "rdep_static"
EXPERT_PARALLEL_CAPACITY_POLICY = "strict_dropless_static_source_buckets"
EXPERT_PARALLEL_TOKEN_PARTITION = "assignment_index_mod_ep"
EXPERT_PARALLEL_COMBINE_POLICY = "reverse_all_to_all_then_psum"
RDEP_STATIC_TOKEN_PARTITION = "route_row_source_data_axis"
RDEP_STATIC_COMBINE_POLICY = "return_by_route_row_identity"
RDEP_STATIC_ROUTE_ROW_IDENTITY = "((source_rank * T) + token) * top_k + slot"


@dataclass(frozen=True, slots=True)
class ModelExecutionContext:
    """Static runtime execution context for model-owned component policies."""

    expert_parallel_mesh: Any | None = None
    expert_parallel_axis_name: str = "ep"
    expert_fsdp_axis_name: str | None = None
    expert_parallel_dispatcher: str = EXPERT_PARALLEL_DISPATCHER_BACKEND
    tensor_parallel_mesh: Any | None = None
    tensor_parallel_axis_name: str = "tp"

    @property
    def expert_parallel_enabled(self) -> bool:
        return self.expert_parallel_mesh is not None

    @property
    def tensor_parallel_enabled(self) -> bool:
        return self.tensor_parallel_mesh is not None


def expert_parallel_dispatcher_backend(axis_sharing: str | None) -> str | None:
    """Resolve the static dispatcher backend for an expert-axis policy."""

    if axis_sharing is None:
        return None
    if axis_sharing == "shared_with_data":
        return RDEP_STATIC_DISPATCHER_BACKEND
    return EXPERT_PARALLEL_DISPATCHER_BACKEND


def expert_parallel_policy_payload(
    *,
    enabled: bool,
    axis_name: str | None,
    axis_size: int,
    axis_sharing: str | None,
    expert_fsdp_axis_name: str | None = None,
    expert_fsdp_axis_size: int = 1,
    expert_fsdp_axis_sharing: str | None = None,
    num_experts: int | None = None,
) -> dict[str, Any]:
    """Build the stable artifact payload for EP dispatcher semantics."""

    experts_per_rank = None
    if enabled and num_experts is not None:
        experts_per_rank = num_experts // axis_size
    backend = expert_parallel_dispatcher_backend(axis_sharing) if enabled else None
    token_partition = RDEP_STATIC_TOKEN_PARTITION if backend == RDEP_STATIC_DISPATCHER_BACKEND else EXPERT_PARALLEL_TOKEN_PARTITION
    combine_policy = RDEP_STATIC_COMBINE_POLICY if backend == RDEP_STATIC_DISPATCHER_BACKEND else EXPERT_PARALLEL_COMBINE_POLICY
    capacity_policy = EXPERT_PARALLEL_CAPACITY_POLICY if enabled else None
    payload = {
        "enabled": enabled,
        "axis": axis_name if enabled else None,
        "axis_size": axis_size if enabled else 1,
        "axis_sharing": axis_sharing if enabled else None,
        "expert_fsdp_axis": expert_fsdp_axis_name if enabled else None,
        "expert_fsdp_axis_size": expert_fsdp_axis_size if enabled else 1,
        "expert_fsdp_axis_sharing": expert_fsdp_axis_sharing if enabled else None,
        "num_experts": num_experts,
        "experts_per_rank": experts_per_rank,
        "dispatcher_backend": backend,
        "capacity_policy": capacity_policy,
        "token_partition": token_partition if enabled else None,
        "combine_policy": combine_policy if enabled else None,
    }
    if backend == RDEP_STATIC_DISPATCHER_BACKEND:
        payload["rdep_pool_axis"] = axis_name
        payload["route_row_identity"] = RDEP_STATIC_ROUTE_ROW_IDENTITY
    return payload


def apply_layer(layer: Any, *args: Any, remat: str) -> Any:
    """Apply one model layer under the requested execution policy."""

    if remat == "none":
        return layer(*args)
    if remat == "block":
        return jax.checkpoint(layer)(*args)
    raise ContractError(f"unsupported model.remat policy {remat!r}")


def column_parallel_linear(linear: Any, x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Apply a linear whose output/features axis is tensor-parallel sharded."""

    if execution is None or not execution.tensor_parallel_enabled:
        return linear(x)
    out = _linear(linear, x)
    return _constrain_activation(out, execution, P("data", None, execution.tensor_parallel_axis_name))


def row_parallel_linear(linear: Any, x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Apply a linear whose input/features axis is tensor-parallel sharded."""

    if execution is None or not execution.tensor_parallel_enabled:
        return linear(x)
    out = _linear(linear, x)
    return _constrain_activation(out, execution, P("data", None, None))


def vocab_parallel_lm_head(linear: Any, x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Apply an LM head whose vocab/output axis is tensor-parallel sharded."""

    if execution is None or not execution.tensor_parallel_enabled:
        return linear(x)
    out = _linear(linear, x)
    return _constrain_activation(out, execution, P("data", None, execution.tensor_parallel_axis_name))


def _linear(linear: Any, x: jax.Array) -> jax.Array:
    kernel = linear.kernel.get_value()
    out = jnp.einsum("...h,ho->...o", x, kernel)
    bias = getattr(linear, "bias", None)
    if bias is not None:
        out = out + bias.get_value()
    return out


def _constrain_activation(x: jax.Array, execution: ModelExecutionContext, spec: P) -> jax.Array:
    if x.ndim != len(spec):
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(execution.tensor_parallel_mesh, spec))
