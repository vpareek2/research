"""Model architecture specs."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from jaxtitan.errors import ContractError

_DTYPE_NAMES = {"float32", "bfloat16"}
_REMAT_POLICIES = {"none", "block"}
_TRINITY_NORM_POLICIES = {"depth_scaled_sandwich"}
_TRINITY_EMBEDDING_SCALES = {"sqrt_hidden"}


@dataclass(frozen=True, slots=True)
class TrinityMoeSpec:
    """Trinity sparse feed-forward contract."""

    num_experts: int
    top_k: int
    expert_intermediate_size: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("num_experts", "top_k"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(f"model.trinity.moe.{field_name} must be a positive integer, got {value!r}")
        if self.top_k > self.num_experts:
            raise ContractError("model.trinity.moe.top_k must be <= model.trinity.moe.num_experts")
        if self.expert_intermediate_size is not None:
            value = self.expert_intermediate_size
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ContractError(
                    f"model.trinity.moe.expert_intermediate_size must be a positive integer, got {value!r}"
                )


@dataclass(frozen=True, slots=True)
class TrinitySpec:
    """Trinity recipe-specific dense-block contract."""

    initial_dense_layers: int
    local_window: int
    local_layers_per_global: int
    attention_gate: bool = True
    qk_norm: bool = True
    norm_policy: str = "depth_scaled_sandwich"
    embedding_scale: str = "sqrt_hidden"
    init_std: float | None = None
    moe: TrinityMoeSpec | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        moe = self.moe
        if isinstance(moe, Mapping):
            moe = TrinityMoeSpec(**dict(moe))
            object.__setattr__(self, "moe", moe)
        elif moe is not None and not isinstance(moe, TrinityMoeSpec):
            raise ContractError("model.trinity.moe must be a TrinityMoeSpec or mapping")
        for field_name in ("initial_dense_layers", "local_window", "local_layers_per_global"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ContractError(f"model.trinity.{field_name} must be an integer, got {value!r}")
            invalid = value < 0 if field_name == "initial_dense_layers" else value <= 0
            if invalid:
                raise ContractError(f"model.trinity.{field_name} is invalid: {value}")
        for field_name in ("attention_gate", "qk_norm"):
            if not isinstance(getattr(self, field_name), bool):
                raise ContractError(f"model.trinity.{field_name} must be a boolean")
        if self.norm_policy not in _TRINITY_NORM_POLICIES:
            raise ContractError(
                f"model.trinity.norm_policy must be one of {sorted(_TRINITY_NORM_POLICIES)}, "
                f"got {self.norm_policy!r}"
            )
        if self.embedding_scale not in _TRINITY_EMBEDDING_SCALES:
            raise ContractError(
                f"model.trinity.embedding_scale must be one of {sorted(_TRINITY_EMBEDDING_SCALES)}, "
                f"got {self.embedding_scale!r}"
            )
        if self.init_std is not None and self.init_std <= 0.0:
            raise ContractError(f"model.trinity.init_std must be positive, got {self.init_std}")


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
    trinity: TrinitySpec | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        trinity = self.trinity
        if isinstance(trinity, Mapping):
            trinity = TrinitySpec(**dict(trinity))
            object.__setattr__(self, "trinity", trinity)
        elif trinity is not None and not isinstance(trinity, TrinitySpec):
            raise ContractError("model.trinity must be a TrinitySpec or mapping")

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
        if self.name == "trinity":
            if trinity is None:
                raise ContractError("model.name='trinity' requires [model.trinity]")
            if trinity.moe is not None and trinity.moe.expert_intermediate_size is None:
                trinity = replace(
                    trinity,
                    moe=replace(trinity.moe, expert_intermediate_size=self.intermediate_size),
                )
                object.__setattr__(self, "trinity", trinity)
            if trinity.initial_dense_layers > self.num_layers:
                raise ContractError("model.trinity.initial_dense_layers must be <= model.num_layers")
            if trinity.moe is not None and trinity.initial_dense_layers >= self.num_layers:
                raise ContractError("model.trinity.initial_dense_layers must leave at least one MoE layer")
            if trinity.local_window > self.max_seq_len:
                raise ContractError("model.trinity.local_window must be <= model.max_seq_len")
        elif trinity is not None:
            raise ContractError("[model.trinity] is only valid when model.name='trinity'")

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
