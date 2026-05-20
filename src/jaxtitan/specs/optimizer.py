"""Optimizer specs and routing contracts."""

from dataclasses import dataclass
from typing import Literal

from jaxtitan.errors import ContractError

ScheduleName = Literal["constant", "cosine", "wsd"]
OptimizerName = Literal["adamw", "muon", "aurora", "riemannian_aurora", "soap"]
_SCHEDULE_NAMES = {"constant", "cosine", "wsd"}
_OPTIMIZER_NAMES = {"adamw", "muon", "aurora", "riemannian_aurora", "soap"}


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    """Static learning-rate schedule contract."""

    peak_lr: float
    name: ScheduleName = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.0
    stable_steps: int | None = None

    def __post_init__(self) -> None:
        if self.name not in _SCHEDULE_NAMES:
            raise ContractError(f"optimizer.schedule.name must be one of {sorted(_SCHEDULE_NAMES)}, got {self.name!r}")
        if self.peak_lr <= 0:
            raise ContractError(f"optimizer.schedule.peak_lr must be positive, got {self.peak_lr}")
        if self.warmup_steps < 0:
            raise ContractError(f"optimizer.schedule.warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ContractError(f"optimizer.schedule.total_steps must be positive, got {self.total_steps}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ContractError(f"optimizer.schedule.min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.stable_steps is not None and self.stable_steps < 0:
            raise ContractError(f"optimizer.schedule.stable_steps must be non-negative, got {self.stable_steps}")


@dataclass(frozen=True, slots=True)
class ParamRouteRule:
    """Declarative optimizer routing rule for a tagged parameter subset."""

    tag: str
    transform: str
    weight_decay: bool = True

    def __post_init__(self) -> None:
        if not self.tag:
            raise ContractError("optimizer route tag must be non-empty")
        if not self.transform:
            raise ContractError("optimizer route transform must be non-empty")


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Static optimizer contract.

    The actual Optax transformation is built later from this spec plus model
    parameter metadata.
    """

    name: OptimizerName
    schedule: ScheduleSpec
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None
    adamw_fallback_schedule: ScheduleSpec | None = None
    route_rules: tuple[ParamRouteRule, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in _OPTIMIZER_NAMES:
            raise ContractError(f"optimizer.name must be one of {sorted(_OPTIMIZER_NAMES)}, got {self.name!r}")
        if self.name != "muon" and self.adamw_fallback_schedule is not None:
            raise ContractError("optimizer.adamw_fallback_schedule is only supported when optimizer.name is 'muon'")
        if self.weight_decay < 0:
            raise ContractError(f"optimizer.weight_decay must be non-negative, got {self.weight_decay}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ContractError(f"optimizer.grad_clip_norm must be positive, got {self.grad_clip_norm}")
        object.__setattr__(self, "route_rules", tuple(self.route_rules))
