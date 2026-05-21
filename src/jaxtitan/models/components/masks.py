"""Attention mask helpers."""

import jax
import jax.numpy as jnp

from jaxtitan.errors import ContractError


def full_sequence_attention_mask(seq_len: int, *, mode: str, local_window: int | None = None) -> jax.Array:
    """Build a [1, query, key] full-sequence causal attention mask."""

    query_positions = jnp.arange(seq_len, dtype=jnp.int32)[:, None]
    key_positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    return _position_attention_mask(query_positions, key_positions, mode=mode, local_window=local_window)[None, :, :]


def cache_attention_mask(
    query_positions: jax.Array,
    cache_positions: jax.Array,
    *,
    mode: str,
    local_window: int | None = None,
) -> jax.Array:
    """Build an attention mask from absolute query positions to cache positions."""

    return _position_attention_mask(
        query_positions[..., None],
        cache_positions[None, None, :],
        mode=mode,
        local_window=local_window,
    )


def _position_attention_mask(
    query_positions: jax.Array,
    key_positions: jax.Array,
    *,
    mode: str,
    local_window: int | None,
) -> jax.Array:
    causal = key_positions <= query_positions
    if mode == "causal":
        return causal
    if mode == "sliding_window":
        if local_window is None or local_window <= 0:
            raise ContractError("sliding-window attention requires a positive local_window")
        return causal & (key_positions >= query_positions - local_window + 1)
    raise ContractError(f"unsupported attention mask mode {mode!r}")
