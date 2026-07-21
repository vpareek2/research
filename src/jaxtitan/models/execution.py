"""Model execution policy helpers."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxtitan.errors import ContractError

EXPERT_PARALLEL_DISPATCHER_BACKEND = "all_to_all"
RDEP_STATIC_DISPATCHER_BACKEND = "rdep_static"
EXPERT_PARALLEL_CAPACITY_POLICY = "strict_dropless_static_worst_case_receive_bound"
EXPERT_PARALLEL_TOKEN_PARTITION = "source_sequence_sharded_over_ep"
EXPERT_PARALLEL_COMBINE_POLICY = "reverse_all_to_all_restore_source_order_then_all_gather"
EXPERT_PARALLEL_EXPERT_EXECUTION = "expert_major_ragged_dot"
RDEP_STATIC_CAPACITY_POLICY = "strict_dropless_static_source_buckets"
RDEP_STATIC_TOKEN_PARTITION = "route_row_source_data_axis"
RDEP_STATIC_COMBINE_POLICY = "return_by_route_row_identity"
RDEP_STATIC_ROUTE_ROW_IDENTITY = "((source_rank * T) + token) * top_k + slot"
MOE_TP_SHARED_EXPERTS = "dense_tensor_parallel"
MOE_TP_ROUTED_EXPERTS = "expert_axis_or_replicated_not_tensor_parallel"
MOE_TP_ROUTED_EXPERT_TENSOR_PARALLEL = "unsupported_until_expert_tp_optimizer"
TP_OPTIMIZER_POLICY = "muon_routes_to_dist_muon_exact"
MOE_TP_OPTIMIZER_POLICY = TP_OPTIMIZER_POLICY


@dataclass(frozen=True, slots=True)
class ModelExecutionContext:
    """Static runtime execution context for model-owned component policies."""

    expert_parallel_mesh: Any | None = None
    expert_parallel_axis_name: str = "ep"
    expert_fsdp_axis_name: str | None = None
    expert_parallel_dispatcher: str = EXPERT_PARALLEL_DISPATCHER_BACKEND
    tensor_parallel_mesh: Any | None = None
    tensor_parallel_axis_name: str = "tp"
    context_parallel_mesh: Any | None = None
    context_parallel_axis_name: str = "cp"
    sequence_parallel: bool = True

    @property
    def expert_parallel_enabled(self) -> bool:
        return self.expert_parallel_mesh is not None

    @property
    def tensor_parallel_enabled(self) -> bool:
        return self.tensor_parallel_mesh is not None

    @property
    def context_parallel_enabled(self) -> bool:
        return self.context_parallel_mesh is not None

    @property
    def spmd_mesh(self) -> Any | None:
        return self.tensor_parallel_mesh or self.context_parallel_mesh or self.expert_parallel_mesh

    @property
    def sequence_parallel_enabled(self) -> bool:
        return self.tensor_parallel_enabled and self.sequence_parallel and not self.context_parallel_enabled


def expert_parallel_dispatcher_backend(axis_sharing: str | None) -> str | None:
    """Resolve the static dispatcher backend for an expert-axis policy."""

    if axis_sharing is None:
        return None
    if axis_sharing == "shared_with_data":
        return RDEP_STATIC_DISPATCHER_BACKEND
    return EXPERT_PARALLEL_DISPATCHER_BACKEND


def expert_parallel_capacity_policy(axis_sharing: str | None) -> str | None:
    """Resolve the capacity contract paired with the selected dispatcher."""

    backend = expert_parallel_dispatcher_backend(axis_sharing)
    if backend is None:
        return None
    if backend == RDEP_STATIC_DISPATCHER_BACKEND:
        return RDEP_STATIC_CAPACITY_POLICY
    return EXPERT_PARALLEL_CAPACITY_POLICY


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
    capacity_policy = expert_parallel_capacity_policy(axis_sharing) if enabled else None
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
    if backend == EXPERT_PARALLEL_DISPATCHER_BACKEND:
        payload["expert_execution"] = EXPERT_PARALLEL_EXPERT_EXECUTION
    if backend == RDEP_STATIC_DISPATCHER_BACKEND:
        payload["rdep_pool_axis"] = axis_name
        payload["route_row_identity"] = RDEP_STATIC_ROUTE_ROW_IDENTITY
    return payload


def moe_tensor_parallel_policy_payload(*, tensor_parallel: bool, has_moe: bool) -> dict[str, Any]:
    """Build the stable artifact payload for MoE tensor-parallel semantics."""

    if not has_moe:
        return {
            "active": False,
            "shared_experts": None,
            "routed_experts": None,
            "routed_expert_tensor_parallel": None,
            "optimizer": None,
        }
    if not tensor_parallel:
        return {
            "active": False,
            "shared_experts": "ordinary_dense_when_present",
            "routed_experts": "expert_axis_or_replicated",
            "routed_expert_tensor_parallel": None,
            "optimizer": None,
        }
    return {
        "active": True,
        "shared_experts": MOE_TP_SHARED_EXPERTS,
        "routed_experts": MOE_TP_ROUTED_EXPERTS,
        "routed_expert_tensor_parallel": MOE_TP_ROUTED_EXPERT_TENSOR_PARALLEL,
        "optimizer": MOE_TP_OPTIMIZER_POLICY,
    }


def apply_layer(layer: Any, *args: Any, remat: str) -> Any:
    """Apply one model layer under the requested execution policy."""

    if remat == "none":
        return layer(*args)
    if remat == "block":
        return jax.checkpoint(layer)(*args)
    raise ContractError(f"unsupported model.remat policy {remat!r}")


def column_parallel_linear(linear: Any, x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Apply a linear whose output/features axis is tensor-parallel sharded."""

    if execution is None or (not execution.tensor_parallel_enabled and not execution.context_parallel_enabled):
        return linear(x)
    if not execution.tensor_parallel_enabled:
        return feature_parallel_activation(_linear(linear, x), execution)
    if execution.sequence_parallel_enabled:
        x = replicated_activation(x, execution)
    out = _linear(linear, x)
    return feature_parallel_activation(out, execution)


def row_parallel_linear(linear: Any, x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Apply a linear whose input/features axis is tensor-parallel sharded."""

    if execution is None or (not execution.tensor_parallel_enabled and not execution.context_parallel_enabled):
        return linear(x)
    if not execution.tensor_parallel_enabled:
        return sequence_parallel_activation(_linear(linear, x), execution)
    out = _linear(linear, x)
    if execution.sequence_parallel_enabled:
        return sequence_parallel_activation(out, execution)
    return replicated_activation(out, execution)


def vocab_parallel_lm_head(linear: Any, x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Apply an LM head whose vocab/output axis is tensor-parallel sharded."""

    if execution is None or (not execution.tensor_parallel_enabled and not execution.context_parallel_enabled):
        return linear(x)
    if not execution.tensor_parallel_enabled:
        return feature_parallel_activation(_linear(linear, x), execution)
    if execution.sequence_parallel_enabled:
        x = replicated_activation(x, execution)
    out = _linear(linear, x)
    return feature_parallel_activation(out, execution)


def replicated_activation(x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Constrain a batch/sequence/hidden activation to replicated sequence/features."""

    if execution is None or (not execution.tensor_parallel_enabled and not execution.context_parallel_enabled):
        return x
    seq_axis = execution.context_parallel_axis_name if execution.context_parallel_enabled else None
    return _constrain_activation(x, execution, P("data", seq_axis, None))


def sequence_parallel_activation(x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Constrain a batch/sequence/hidden activation to sequence parallel placement."""

    if execution is None:
        return x
    if execution.context_parallel_enabled:
        return _constrain_activation(x, execution, P("data", execution.context_parallel_axis_name, None))
    if not execution.sequence_parallel_enabled:
        return x
    return _constrain_activation(x, execution, P("data", execution.tensor_parallel_axis_name, None))


def feature_parallel_activation(x: jax.Array, execution: ModelExecutionContext | None) -> jax.Array:
    """Constrain a batch/sequence/features activation to feature/vocab parallel placement."""

    if execution is None or (not execution.tensor_parallel_enabled and not execution.context_parallel_enabled):
        return x
    seq_axis = execution.context_parallel_axis_name if execution.context_parallel_enabled else None
    feature_axis = execution.tensor_parallel_axis_name if execution.tensor_parallel_enabled else None
    return _constrain_activation(x, execution, P("data", seq_axis, feature_axis))


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
    mesh = execution.spmd_mesh
    if mesh is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))
