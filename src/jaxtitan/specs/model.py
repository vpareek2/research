"""Model architecture specs."""

from dataclasses import dataclass

from jaxtitan.errors import ContractError

_DTYPE_NAMES = {"float32", "bfloat16"}
_REMAT_POLICIES = {"none", "block"}


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
    intermediate_size: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    n_kv_heads: int | None = None
    rope_theta: float = 1_000_000.0
    norm_epsilon: float = 1e-6
    tied_embeddings: bool = False
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    remat: str = "none"

    def __post_init__(self) -> None:
        for field_name in ("vocab_size", "hidden_size", "intermediate_size", "num_layers", "num_heads", "max_seq_len"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ContractError(f"model.{field_name} must be positive, got {value}")
        for field_name in ("rope_theta", "norm_epsilon"):
            value = getattr(self, field_name)
            if value <= 0.0:
                raise ContractError(f"model.{field_name} must be positive, got {value}")
        for field_name in ("param_dtype", "compute_dtype"):
            value = getattr(self, field_name)
            if value not in _DTYPE_NAMES:
                raise ContractError(f"model.{field_name} must be one of {sorted(_DTYPE_NAMES)}, got {value!r}")
        if self.remat not in _REMAT_POLICIES:
            raise ContractError(f"model.remat must be one of {sorted(_REMAT_POLICIES)}, got {self.remat!r}")
        if self.tied_embeddings:
            raise ContractError("model.tied_embeddings is not supported yet")

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
