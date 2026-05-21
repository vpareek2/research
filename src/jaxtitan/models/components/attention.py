"""Attention model components."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.models.components.masks import cache_attention_mask, full_sequence_attention_mask
from jaxtitan.models.components.position import apply_rope, apply_rope_at_positions
from jaxtitan.specs.model import ModelSpec


class FullAttentionContext(NamedTuple):
    """Full-sequence attention inputs that are shared across attention mechanisms."""

    cos: Any | None
    sin: Any | None


class PrefillAttentionContext(NamedTuple):
    """Prefill-time attention inputs plus mechanism-owned state."""

    positions: Any
    attention_mask: Any
    cache: Any
    layer_index: int


class DecodeAttentionContext(NamedTuple):
    """Decode-time attention inputs plus mechanism-owned state."""

    positions: Any
    attention_mask: Any
    cache: Any
    layer_index: int


class GroupedQueryAttention(nnx.Module):
    """Grouped-query causal self-attention."""

    def __init__(
        self,
        spec: ModelSpec,
        rngs: nnx.Rngs,
        *,
        position: str = "rope",
        mask: str = "causal",
        local_window: int | None = None,
        qk_norm: bool = True,
        gate: bool = False,
        kernel_init: Any | None = None,
    ):
        self.hidden_size = spec.hidden_size
        self.num_heads = spec.num_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.hidden_size // spec.num_heads
        self.position = position
        self.mask = mask
        self.local_window = local_window
        self.qk_norm_enabled = qk_norm
        self.gate_enabled = gate
        if position not in {"rope", "none"}:
            raise ContractError(f"unsupported attention position mode {position!r}")
        if mask not in {"causal", "sliding_window"}:
            raise ContractError(f"unsupported attention mask mode {mask!r}")
        if mask == "sliding_window" and (local_window is None or local_window <= 0):
            raise ContractError("sliding-window attention requires a positive local_window")
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        linear_kwargs = {} if kernel_init is None else {"kernel_init": kernel_init}

        self.q = nnx.Linear(
            spec.hidden_size,
            spec.num_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.k = nnx.Linear(
            spec.hidden_size,
            spec.n_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.v = nnx.Linear(
            spec.hidden_size,
            spec.n_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.o = nnx.Linear(
            spec.hidden_size,
            spec.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        if qk_norm:
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
        else:
            self.q_norm = None
            self.k_norm = None
        if gate:
            self.gate = nnx.Linear(
                spec.hidden_size,
                spec.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                **linear_kwargs,
            )
        else:
            self.gate = None

    def __call__(self, x: jax.Array, context: FullAttentionContext) -> jax.Array:
        batch_size, seq_len, _ = x.shape
        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q, k = self._prepare_qk(q, k, context)

        mask = full_sequence_attention_mask(seq_len, mode=self.mask, local_window=self.local_window)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = self._apply_gate(x, out)
        return self.o(out.reshape(batch_size, seq_len, self.hidden_size))

    def prefill(
        self,
        x: jax.Array,
        context: PrefillAttentionContext,
    ) -> tuple[jax.Array, Any]:
        batch_size, seq_len, _ = x.shape
        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        q, k = self._prepare_qk_at_positions(q, k, context.positions, context.cache.rope_theta)

        keys, values, lengths = _cache_write(
            context.cache,
            context.layer_index,
            context.positions,
            context.attention_mask,
            k,
            v,
        )
        next_cache = context.cache.replace(keys=keys, values=values, lengths=lengths)
        cache_positions = jnp.arange(context.cache.max_cache_len, dtype=context.positions.dtype)
        mask = cache_attention_mask(context.positions, cache_positions, mode=self.mask, local_window=self.local_window)
        mask = mask & context.attention_mask.astype(jnp.bool_)[..., None] & (cache_positions[None, None, :] < lengths[:, None, None])
        cached_k = next_cache.keys[context.layer_index]
        cached_v = next_cache.values[context.layer_index]
        out = scaled_dot_product_attention(q, cached_k, cached_v, mask)
        out = self._apply_gate(x, out)
        return self.o(out.reshape(batch_size, seq_len, self.hidden_size)), next_cache

    def decode_one(
        self,
        x: jax.Array,
        context: DecodeAttentionContext,
    ) -> tuple[jax.Array, Any]:
        batch_size, _, _ = x.shape
        positions = context.positions[:, None]
        q = self.q(x).reshape(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k(x).reshape(batch_size, 1, self.n_kv_heads, self.head_dim)
        v = self.v(x).reshape(batch_size, 1, self.n_kv_heads, self.head_dim)
        q, k = self._prepare_qk_at_positions(q, k, positions, context.cache.rope_theta)

        keys, values, lengths = _cache_write(
            context.cache,
            context.layer_index,
            positions,
            jnp.ones_like(positions, dtype=bool),
            k,
            v,
        )
        next_cache = context.cache.replace(keys=keys, values=values, lengths=lengths)
        cache_positions = jnp.arange(context.cache.max_cache_len, dtype=positions.dtype)
        mask = cache_attention_mask(positions, cache_positions, mode=self.mask, local_window=self.local_window)
        mask = mask & context.attention_mask.astype(jnp.bool_)[:, None, :] & (cache_positions[None, None, :] < lengths[:, None, None])
        cached_k = next_cache.keys[context.layer_index]
        cached_v = next_cache.values[context.layer_index]
        out = scaled_dot_product_attention(q, cached_k, cached_v, mask)
        out = self._apply_gate(x, out)
        return self.o(out.reshape(batch_size, 1, self.hidden_size)), next_cache

    def _prepare_qk(
        self,
        q: jax.Array,
        k: jax.Array,
        context: FullAttentionContext,
    ) -> tuple[jax.Array, jax.Array]:
        q = q if self.q_norm is None else self.q_norm(q)
        k = k if self.k_norm is None else self.k_norm(k)
        if self.position == "none":
            return q, k
        if context.cos is None or context.sin is None:
            raise ContractError("RoPE attention requires cos/sin tables")
        return apply_rope(q, context.cos, context.sin), apply_rope(k, context.cos, context.sin)

    def _prepare_qk_at_positions(
        self,
        q: jax.Array,
        k: jax.Array,
        positions: jax.Array,
        rope_theta: float,
    ) -> tuple[jax.Array, jax.Array]:
        q = q if self.q_norm is None else self.q_norm(q)
        k = k if self.k_norm is None else self.k_norm(k)
        if self.position == "none":
            return q, k
        return (
            apply_rope_at_positions(q, positions, self.head_dim, theta=rope_theta),
            apply_rope_at_positions(k, positions, self.head_dim, theta=rope_theta),
        )

    def _apply_gate(self, x: jax.Array, out: jax.Array) -> jax.Array:
        if self.gate is None:
            return out
        gate = jax.nn.sigmoid(self.gate(x)).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        return out * gate


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
