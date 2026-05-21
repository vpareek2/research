"""Flax NNX decoder-only language model boundary."""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models.execution import apply_layer
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
class ParamLayout:
    """Model-owned parameter layout policy for distributed placement."""

    path: tuple[str, ...]
    tag: str
    shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    fsdp_axis: int | None


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    """Explicit model graph/state split plus parameter metadata."""

    graph: Any
    state: Any
    metadata: tuple[ParamMetadata, ...]
    param_layouts: tuple[ParamLayout, ...]


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
    return ModelBuildResult(graph=graph, state=state, metadata=metadata, param_layouts=parameter_layouts(metadata))


def apply_model(graph: Any, state: Any, input_ids: Any) -> jax.Array:
    """Apply a split model graph/state to token ids."""

    model = nnx.merge(graph, state)
    return model(input_ids)


def prefill_model(
    graph: Any,
    state: Any,
    input_ids: Any,
    positions: Any,
    attention_mask: Any,
    cache: Any,
) -> tuple[jax.Array, jax.Array, Any]:
    """Apply model prefill and update a KV cache."""

    model = nnx.merge(graph, state)
    return model.prefill(input_ids, positions, attention_mask, cache)


def decode_model(
    graph: Any,
    state: Any,
    token_ids: Any,
    positions: Any,
    attention_mask: Any,
    cache: Any,
) -> tuple[jax.Array, Any]:
    """Apply one-token decode and update a KV cache."""

    model = nnx.merge(graph, state)
    return model.decode_one(token_ids, positions, attention_mask, cache)


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


def parameter_layouts(metadata: tuple[ParamMetadata, ...]) -> tuple[ParamLayout, ...]:
    """Build model-owned sharding layout metadata from semantic parameter metadata."""

    return tuple(_layout_for_metadata(item) for item in metadata)


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

        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, :, :]
        out = scaled_dot_product_attention(q, k, v, mask)
        return self.o(out.reshape(batch_size, seq_len, self.hidden_size))

    def prefill(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        batch_size, seq_len, _ = x.shape
        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        q = apply_rope_at_positions(self.q_norm(q), positions, self.head_dim, theta=cache.rope_theta)
        k = apply_rope_at_positions(self.k_norm(k), positions, self.head_dim, theta=cache.rope_theta)

        keys, values, lengths = _cache_write(cache, layer_index, positions, attention_mask, k, v)
        next_cache = cache.replace(keys=keys, values=values, lengths=lengths)
        cache_positions = jnp.arange(cache.max_cache_len, dtype=positions.dtype)
        mask = (
            attention_mask.astype(jnp.bool_)[..., None]
            & (cache_positions[None, None, :] <= positions[..., None])
            & (cache_positions[None, None, :] < lengths[:, None, None])
        )
        cached_k = next_cache.keys[layer_index]
        cached_v = next_cache.values[layer_index]
        out = scaled_dot_product_attention(q, cached_k, cached_v, mask)
        return self.o(out.reshape(batch_size, seq_len, self.hidden_size)), next_cache

    def decode_one(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        batch_size, _, _ = x.shape
        positions = positions[:, None]
        q = self.q(x).reshape(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, 1, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, 1, self.n_kv_heads, self.head_dim)
        q = apply_rope_at_positions(self.q_norm(q), positions, self.head_dim, theta=cache.rope_theta)
        k = apply_rope_at_positions(self.k_norm(k), positions, self.head_dim, theta=cache.rope_theta)

        keys, values, lengths = _cache_write(cache, layer_index, positions, jnp.ones_like(positions, dtype=bool), k, v)
        next_cache = cache.replace(keys=keys, values=values, lengths=lengths)
        cache_positions = jnp.arange(cache.max_cache_len, dtype=positions.dtype)
        mask = (
            attention_mask.astype(jnp.bool_)[:, None, :]
            & (cache_positions[None, None, :] <= positions[..., None])
            & (cache_positions[None, None, :] < lengths[:, None, None])
        )
        cached_k = next_cache.keys[layer_index]
        cached_v = next_cache.values[layer_index]
        out = scaled_dot_product_attention(q, cached_k, cached_v, mask)
        return self.o(out.reshape(batch_size, 1, self.hidden_size)), next_cache


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

    def prefill(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        attn_out, cache = self.attn.prefill(self.pre_norm(x), positions, attention_mask, cache, layer_index)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache

    def decode_one(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        attn_out, cache = self.attn.decode_one(self.pre_norm(x), positions, attention_mask, cache, layer_index)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache


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
            x = apply_layer(layer, x, cos, sin, remat=self.spec.remat)
        return self.lm_head(self.norm(x))

    def prefill(self, input_ids: Any, positions: Any, attention_mask: Any, cache: Any) -> tuple[jax.Array, jax.Array, Any]:
        input_ids = jnp.asarray(input_ids)
        positions = jnp.asarray(positions)
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if input_ids.ndim != 2:
            raise ContractError(f"input_ids must have shape [batch, seq], got {input_ids.shape}")
        batch_size, seq_len = input_ids.shape
        if positions.shape != input_ids.shape:
            raise ContractError(f"positions shape {positions.shape} must equal input_ids shape {input_ids.shape}")
        if attention_mask.shape != input_ids.shape:
            raise ContractError(
                f"attention_mask shape {attention_mask.shape} must equal input_ids shape {input_ids.shape}"
            )
        if seq_len > self.spec.max_seq_len:
            raise ContractError(f"input sequence length {seq_len} exceeds model.max_seq_len={self.spec.max_seq_len}")

        x = self.embed(input_ids)
        for layer_index, layer in enumerate(self.layers):
            x, cache = layer.prefill(x, positions, attention_mask, cache, layer_index)
        logits = self.lm_head(self.norm(x))
        return logits, logits[:, -1, :], cache

    def decode_one(self, token_ids: Any, positions: Any, attention_mask: Any, cache: Any) -> tuple[jax.Array, Any]:
        token_ids = jnp.asarray(token_ids)
        positions = jnp.asarray(positions)
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if token_ids.ndim != 1:
            raise ContractError(f"token_ids must have shape [batch], got {token_ids.shape}")
        if positions.shape != token_ids.shape:
            raise ContractError(f"positions shape {positions.shape} must equal token_ids shape {token_ids.shape}")
        if attention_mask.ndim != 2 or attention_mask.shape[0] != token_ids.shape[0]:
            raise ContractError("attention_mask must have shape [batch, max_cache_len]")

        x = self.embed(token_ids[:, None])
        for layer_index, layer in enumerate(self.layers):
            x, cache = layer.decode_one(x, positions, attention_mask, cache, layer_index)
        logits = self.lm_head(self.norm(x))[:, 0, :]
        return logits, cache


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


def apply_rope_at_positions(x: jax.Array, positions: jax.Array, head_dim: int, theta: float) -> jax.Array:
    """Apply RoPE with absolute per-row positions."""

    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq
    cos = jnp.cos(freqs).astype(x.dtype)[..., None, :]
    sin = jnp.sin(freqs).astype(x.dtype)[..., None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), axis=-1)
    return rotated.reshape(x.shape)


def scaled_dot_product_attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array) -> jax.Array:
    """Grouped-query scaled dot-product attention."""

    k = _repeat_kv(k, q.shape[2])
    v = _repeat_kv(v, q.shape[2])
    logits = jnp.einsum("bthd,bshd->bhts", q.astype(jnp.float32), k.astype(jnp.float32))
    logits = logits / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32))
    logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(logits, axis=-1).astype(q.dtype)
    return jnp.einsum("bhts,bshd->bthd", probs, v)


def _repeat_kv(x: jax.Array, num_heads: int) -> jax.Array:
    if x.shape[2] == num_heads:
        return x
    if num_heads % x.shape[2] != 0:
        raise ContractError(f"query heads {num_heads} must be divisible by kv heads {x.shape[2]}")
    return jnp.repeat(x, num_heads // x.shape[2], axis=2)


def _cache_write(
    cache: Any,
    layer_index: int,
    positions: jax.Array,
    attention_mask: jax.Array,
    k: jax.Array,
    v: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_indices = jnp.arange(k.shape[0])[:, None]
    valid = attention_mask.astype(jnp.bool_)
    existing_k = cache.keys[layer_index, batch_indices, positions]
    existing_v = cache.values[layer_index, batch_indices, positions]
    next_k = jnp.where(valid[..., None, None], k, existing_k)
    next_v = jnp.where(valid[..., None, None], v, existing_v)
    keys = cache.keys.at[layer_index, batch_indices, positions].set(next_k)
    values = cache.values.at[layer_index, batch_indices, positions].set(next_v)
    next_lengths = jnp.max(jnp.where(valid, positions + 1, 0), axis=1)
    lengths = jnp.maximum(cache.lengths, next_lengths.astype(cache.lengths.dtype))
    return keys, values, lengths


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


def _layout_for_metadata(item: ParamMetadata) -> ParamLayout:
    logical_axes, fsdp_axis = _layout_policy(item)
    if len(logical_axes) != len(item.shape):
        raise ContractError(
            f"parameter layout for {'.'.join(item.path)!r} has rank {len(logical_axes)}, "
            f"but parameter shape has rank {len(item.shape)}"
        )
    if fsdp_axis is not None and not (0 <= fsdp_axis < len(item.shape)):
        raise ContractError(f"parameter layout for {'.'.join(item.path)!r} has invalid fsdp axis {fsdp_axis}")
    return ParamLayout(
        path=item.path,
        tag=item.tag,
        shape=item.shape,
        logical_axes=logical_axes,
        fsdp_axis=fsdp_axis,
    )


def _layout_policy(item: ParamMetadata) -> tuple[tuple[str, ...], int | None]:
    tag = item.tag
    if tag == "embedding":
        return ("vocab", "hidden"), None
    if tag == "lm_head":
        return ("hidden", "vocab"), 0
    if tag in {"attention_q", "attention_k", "attention_v"}:
        return ("hidden_in", "hidden_out"), 1
    if tag == "attention_o":
        return ("hidden_in", "hidden_out"), 0
    if tag in {"mlp_gate", "mlp_up"}:
        return ("hidden", "intermediate"), 1
    if tag == "mlp_down":
        return ("intermediate", "hidden"), 0
    if tag in {
        "attention_q_norm",
        "attention_k_norm",
        "block_pre_norm",
        "block_post_norm",
        "final_norm",
    }:
        return tuple("norm" for _dim in item.shape), None
    return tuple("replicated" for _dim in item.shape), None
