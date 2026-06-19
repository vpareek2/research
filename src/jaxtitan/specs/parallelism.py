"""Parallelism specs."""

from dataclasses import dataclass
from typing import Mapping, Literal

from jaxtitan.errors import ContractError

ParallelismMode = Literal["ddp", "zero2", "fsdp"]
ExpertParallelAxis = Literal["auto", "ep", "fsdp"]
_PARALLELISM_MODES = {"ddp", "zero2", "fsdp"}
_EXPERT_PARALLEL_AXES = {"auto", "ep", "fsdp"}


@dataclass(frozen=True, slots=True)
class ExpertParallelAxisPolicy:
    """Resolved semantic axis used for routed expert ownership."""

    enabled: bool
    axis: str | None = None
    axis_size: int = 1
    axis_sharing: str | None = None


@dataclass(frozen=True, slots=True)
class ExpertFSDPPolicy:
    """Resolved FSDP policy for internals of routed expert matrices."""

    enabled: bool
    axis: str | None = None
    axis_size: int = 1
    axis_sharing: str | None = None


@dataclass(frozen=True, slots=True)
class ParallelismSpec:
    """Static distributed execution mode contract."""

    mode: ParallelismMode = "ddp"
    expert_parallel: bool = False
    expert_parallel_axis: ExpertParallelAxis = "auto"

    def __post_init__(self) -> None:
        if self.mode not in _PARALLELISM_MODES:
            raise ContractError(f"parallelism.mode must be one of {sorted(_PARALLELISM_MODES)}, got {self.mode!r}")
        if not isinstance(self.expert_parallel, bool):
            raise ContractError("parallelism.expert_parallel must be a boolean")
        if self.expert_parallel_axis not in _EXPERT_PARALLEL_AXES:
            raise ContractError(
                "parallelism.expert_parallel_axis must be one of "
                f"{sorted(_EXPERT_PARALLEL_AXES)}, got {self.expert_parallel_axis!r}"
            )
        if not self.expert_parallel and self.expert_parallel_axis != "auto":
            raise ContractError("parallelism.expert_parallel_axis requires parallelism.expert_parallel=true")


def resolve_expert_parallel_axis(
    parallelism: ParallelismSpec,
    axis_sizes: Mapping[str, int],
) -> ExpertParallelAxisPolicy:
    """Resolve product-axis or folded-axis expert ownership policy."""

    if not parallelism.expert_parallel:
        return ExpertParallelAxisPolicy(enabled=False)
    requested_axis = parallelism.expert_parallel_axis
    if requested_axis == "auto":
        if "ep" in axis_sizes:
            axis = "ep"
        elif parallelism.mode == "fsdp" and "fsdp" in axis_sizes:
            axis = "fsdp"
        else:
            raise ContractError(
                "parallelism.expert_parallel=true requires a mesh ep axis or "
                "parallelism.mode='fsdp' with a mesh fsdp axis for folded expert parallelism"
            )
    else:
        axis = requested_axis
    if axis == "ep":
        if "ep" not in axis_sizes:
            raise ContractError("parallelism.expert_parallel_axis='ep' requires a mesh ep axis")
        return ExpertParallelAxisPolicy(
            enabled=True,
            axis="ep",
            axis_size=axis_sizes["ep"],
            axis_sharing="dedicated_ep",
        )
    if axis == "fsdp":
        if parallelism.mode != "fsdp":
            raise ContractError("parallelism.expert_parallel_axis='fsdp' requires parallelism.mode='fsdp'")
        if "fsdp" not in axis_sizes:
            raise ContractError("parallelism.expert_parallel_axis='fsdp' requires a mesh fsdp axis")
        return ExpertParallelAxisPolicy(
            enabled=True,
            axis="fsdp",
            axis_size=axis_sizes["fsdp"],
            axis_sharing="shared_with_fsdp",
        )
    raise ContractError(f"unsupported expert parallel axis {axis!r}")


def resolve_expert_fsdp_axis(
    parallelism: ParallelismSpec,
    axis_sizes: Mapping[str, int],
) -> ExpertFSDPPolicy:
    """Resolve optional routed-expert internal FSDP policy."""

    axis_size = axis_sizes.get("expert_fsdp", 1)
    if axis_size == 1:
        return ExpertFSDPPolicy(enabled=False)
    if parallelism.mode != "fsdp":
        raise ContractError("mesh expert_fsdp axis with size > 1 requires parallelism.mode='fsdp'")
    if not parallelism.expert_parallel:
        raise ContractError("mesh expert_fsdp axis with size > 1 requires parallelism.expert_parallel=true")
    expert_axis = resolve_expert_parallel_axis(parallelism, axis_sizes)
    if expert_axis.axis != "ep":
        raise ContractError("mesh expert_fsdp axis with size > 1 requires a dedicated expert_parallel_axis='ep'")
    return ExpertFSDPPolicy(
        enabled=True,
        axis="expert_fsdp",
        axis_size=axis_size,
        axis_sharing="expert_region_internal",
    )
