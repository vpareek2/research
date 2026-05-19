"""Generation specs."""

from dataclasses import dataclass

from jaxtitan.errors import ContractError


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Static generation contract."""

    max_new_tokens: int
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ContractError(f"generation.max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ContractError(f"generation.temperature must be positive, got {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ContractError(f"generation.top_k must be positive, got {self.top_k}")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ContractError(f"generation.top_p must be in (0, 1], got {self.top_p}")
