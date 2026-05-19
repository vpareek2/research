"""Evaluation specs."""

from dataclasses import dataclass

from jaxtitan.errors import ContractError


@dataclass(frozen=True, slots=True)
class EvalSpec:
    """Static evaluation contract."""

    name: str
    every_steps: int
    num_batches: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("eval.name must be non-empty")
        if self.every_steps <= 0:
            raise ContractError(f"eval.every_steps must be positive, got {self.every_steps}")
        if self.num_batches <= 0:
            raise ContractError(f"eval.num_batches must be positive, got {self.num_batches}")
