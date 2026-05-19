"""Flax NNX decoder-only language model boundary."""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.specs.model import ModelSpec


@dataclass(frozen=True, slots=True)
class ParamMetadata:
    """Stable metadata for one model parameter leaf."""

    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    count: int
    tag: str


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    """Explicit model graph/state split plus parameter metadata."""

    graph: Any
    state: Any
    metadata: tuple[ParamMetadata, ...]


def dtype_from_name(name: str) -> Any:
    """Resolve supported Jaxtitan model dtype names."""

    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    raise ContractError(f"unsupported dtype {name!r}; expected 'float32' or 'bfloat16'")


def build_model(spec: ModelSpec, seed: int) -> ModelBuildResult:
    """Build a supported model and return explicit NNX graph/state pieces."""

    if spec.name != "decoder":
        raise ContractError(f"unsupported model.name {spec.name!r}; only 'decoder' is available")
    model = DecoderModel(spec, rngs=nnx.Rngs(seed))
    graph, state = nnx.split(model)
    metadata = parameter_metadata(state)
    return ModelBuildResult(graph=graph, state=state, metadata=metadata)


def apply_model(graph: Any, state: Any, input_ids: Any) -> jax.Array:
    """Apply a split model graph/state to token ids."""

    model = nnx.merge(graph, state)
    return model(input_ids)


def count_parameters(metadata: tuple[ParamMetadata, ...]) -> int:
    """Count parameters from metadata, not by re-walking model state."""

    return sum(item.count for item in metadata)


def parameter_metadata(state: Any) -> tuple[ParamMetadata, ...]:
    """Build metadata for every parameter leaf in an NNX State."""

    items = []
    for raw_path, variable in nnx.to_flat_state(state):
        value = variable.get_value()
        shape = tuple(int(dim) for dim in value.shape)
        path = tuple(str(part) for part in raw_path)
        items.append(
            ParamMetadata(
                path=path,
                shape=shape,
                dtype=str(value.dtype),
                count=reduce(mul, shape, 1),
                tag=_tag_for_path(path),
            )
        )
    return tuple(items)


class DecoderSwiGLU(nnx.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        self.gate = nnx.Linear(
            spec.hidden_size,
            spec.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.up = nnx.Linear(
            spec.hidden_size,
            spec.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.down = nnx.Linear(
            spec.intermediate_size,
            spec.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class GroupedQueryAttention(nnx.Module):
    """Grouped-query causal self-attention."""

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        self.hidden_size = spec.hidden_size
        self.num_heads = spec.num_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.hidden_size // spec.num_heads
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)

        self.q = nnx.Linear(
            spec.hidden_size,
            spec.num_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.k = nnx.Linear(
            spec.hidden_size,
            spec.n_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.v = nnx.Linear(
            spec.hidden_size,
            spec.n_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.o = nnx.Linear(
            spec.hidden_size,
            spec.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=spec.norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=spec.norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = x.shape
        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.reshape(batch_size, seq_len, self.hidden_size))


class DecoderBlock(nnx.Module):
    """Single decoder transformer block."""

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        self.pre_norm = nnx.RMSNorm(
            spec.hidden_size,
            epsilon=spec.norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attn = GroupedQueryAttention(spec, rngs=rngs)
        self.post_norm = nnx.RMSNorm(
            spec.hidden_size,
            epsilon=spec.norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp = DecoderSwiGLU(spec, rngs=rngs)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        x = x + self.attn(self.pre_norm(x), cos, sin)
        x = x + self.mlp(self.post_norm(x))
        return x


class DecoderModel(nnx.Module):
    """Decoder-only language model."""

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        if spec.tied_embeddings:
            raise ContractError("model.tied_embeddings is not supported yet")
        self.spec = spec
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        self.embed = nnx.Embed(
            spec.vocab_size,
            spec.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nnx.List([DecoderBlock(spec, rngs=rngs) for _ in range(spec.num_layers)])
        self.norm = nnx.RMSNorm(
            spec.hidden_size,
            epsilon=spec.norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            spec.hidden_size,
            spec.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, input_ids: Any) -> jax.Array:
        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ContractError(f"input_ids must have shape [batch, seq], got {input_ids.shape}")
        _, seq_len = input_ids.shape
        if seq_len > self.spec.max_seq_len:
            raise ContractError(f"input sequence length {seq_len} exceeds model.max_seq_len={self.spec.max_seq_len}")

        x = self.embed(input_ids)
        cos, sin = precompute_rope(
            seq_len=seq_len,
            head_dim=self.spec.hidden_size // self.spec.num_heads,
            theta=self.spec.rope_theta,
            dtype=x.dtype,
        )
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))


def precompute_rope(seq_len: int, head_dim: int, theta: float, dtype: Any) -> tuple[jax.Array, jax.Array]:
    """Precompute RoPE cos/sin tables for a fixed sequence length."""

    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply RoPE to an array shaped [batch, seq, heads, head_dim]."""

    cos = cos[:, None, :]
    sin = sin[:, None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), axis=-1)
    return rotated.reshape(x.shape)


def _tag_for_path(path: tuple[str, ...]) -> str:
    if path[0] == "embed":
        return "embedding"
    if path[0] == "lm_head":
        return "lm_head"
    if path[0] == "norm":
        return "final_norm"
    if "attn" in path:
        return _attention_tag(path)
    if "mlp" in path:
        return _mlp_tag(path)
    if "pre_norm" in path:
        return "block_pre_norm"
    if "post_norm" in path:
        return "block_post_norm"
    raise ContractError(f"unrecognized decoder parameter path {'.'.join(path)}")


def _attention_tag(path: tuple[str, ...]) -> str:
    for component, tag in (
        ("q_norm", "attention_q_norm"),
        ("k_norm", "attention_k_norm"),
        ("q", "attention_q"),
        ("k", "attention_k"),
        ("v", "attention_v"),
        ("o", "attention_o"),
    ):
        if component in path:
            return tag
    raise ContractError(f"unrecognized attention parameter path {'.'.join(path)}")


def _mlp_tag(path: tuple[str, ...]) -> str:
    for component, tag in (("gate", "mlp_gate"), ("up", "mlp_up"), ("down", "mlp_down")):
        if component in path:
            return tag
    raise ContractError(f"unrecognized MLP parameter path {'.'.join(path)}")
