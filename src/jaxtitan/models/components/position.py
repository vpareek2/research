"""Position encoding components."""

from typing import Any

import jax
import jax.numpy as jnp


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
