"""Model execution policy helpers."""

from dataclasses import dataclass
from typing import Any

import jax

from jaxtitan.errors import ContractError

EXPERT_PARALLEL_DISPATCHER_BACKEND = "all_to_all"
EXPERT_PARALLEL_CAPACITY_POLICY = "strict_dropless_static_source_buckets"
EXPERT_PARALLEL_TOKEN_PARTITION = "assignment_index_mod_ep"
EXPERT_PARALLEL_COMBINE_POLICY = "reverse_all_to_all_then_psum"


@dataclass(frozen=True, slots=True)
class ModelExecutionContext:
    """Static runtime execution context for model-owned component policies."""

    expert_parallel_mesh: Any | None = None
    expert_parallel_axis_name: str = "ep"
    expert_fsdp_axis_name: str | None = None
    expert_parallel_dispatcher: str = EXPERT_PARALLEL_DISPATCHER_BACKEND

    @property
    def expert_parallel_enabled(self) -> bool:
        return self.expert_parallel_mesh is not None


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
    return {
        "enabled": enabled,
        "axis": axis_name if enabled else None,
        "axis_size": axis_size if enabled else 1,
        "axis_sharing": axis_sharing if enabled else None,
        "expert_fsdp_axis": expert_fsdp_axis_name if enabled else None,
        "expert_fsdp_axis_size": expert_fsdp_axis_size if enabled else 1,
        "expert_fsdp_axis_sharing": expert_fsdp_axis_sharing if enabled else None,
        "num_experts": num_experts,
        "experts_per_rank": experts_per_rank,
        "dispatcher_backend": EXPERT_PARALLEL_DISPATCHER_BACKEND if enabled else None,
        "capacity_policy": EXPERT_PARALLEL_CAPACITY_POLICY if enabled else None,
        "token_partition": EXPERT_PARALLEL_TOKEN_PARTITION if enabled else None,
        "combine_policy": EXPERT_PARALLEL_COMBINE_POLICY if enabled else None,
    }


def apply_layer(layer: Any, *args: Any, remat: str) -> Any:
    """Apply one model layer under the requested execution policy."""

    if remat == "none":
        return layer(*args)
    if remat == "block":
        return jax.checkpoint(layer)(*args)
    raise ContractError(f"unsupported model.remat policy {remat!r}")
