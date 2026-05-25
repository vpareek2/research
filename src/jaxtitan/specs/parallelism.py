"""Parallelism specs."""

from dataclasses import dataclass
from typing import Literal

from jaxtitan.errors import ContractError

ParallelismMode = Literal["ddp", "zero2", "fsdp"]
_PARALLELISM_MODES = {"ddp", "zero2", "fsdp"}


@dataclass(frozen=True, slots=True)
class ParallelismSpec:
    """Static distributed execution mode contract."""

    mode: ParallelismMode = "ddp"
    expert_parallel: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _PARALLELISM_MODES:
            raise ContractError(f"parallelism.mode must be one of {sorted(_PARALLELISM_MODES)}, got {self.mode!r}")
        if not isinstance(self.expert_parallel, bool):
            raise ContractError("parallelism.expert_parallel must be a boolean")
