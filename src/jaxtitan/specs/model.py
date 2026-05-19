"""Model architecture specs."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtitan.errors import ContractError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static decoder model contract.

    This is architecture identity and shape metadata only. It does not own a
    Flax NNX module, params, or sharding.
    """

    name: str
    variant: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    n_kv_heads: int | None = None
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for field_name in ("vocab_size", "hidden_size", "num_layers", "num_heads", "max_seq_len"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ContractError(f"model.{field_name} must be positive, got {value}")

        n_kv_heads = self.num_heads if self.n_kv_heads is None else self.n_kv_heads
        if n_kv_heads <= 0:
            raise ContractError(f"model.n_kv_heads must be positive, got {n_kv_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ContractError(
                f"model.hidden_size must be divisible by model.num_heads, got "
                f"{self.hidden_size} and {self.num_heads}"
            )
        if self.num_heads % n_kv_heads != 0:
            raise ContractError(
                f"model.num_heads must be divisible by model.n_kv_heads, got "
                f"{self.num_heads} and {n_kv_heads}"
            )
        object.__setattr__(self, "n_kv_heads", n_kv_heads)
