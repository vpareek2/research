"""
Inference KV cache helpers.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from research.config import ModelConfig


class LayerKVCache(NamedTuple):
    k: jax.Array
    v: jax.Array


class KVCache(NamedTuple):
    layers: tuple[LayerKVCache, ...]
    length: jax.Array


def init_kv_cache(config: ModelConfig, batch_size: int, dtype) -> KVCache:
    head_dim = config.hidden_size // config.n_heads
    shape = (batch_size, config.seq_len, config.n_kv_heads, head_dim)
    layers = tuple(
        LayerKVCache(
            k=jnp.zeros(shape, dtype=dtype),
            v=jnp.zeros(shape, dtype=dtype),
        )
        for _ in range(config.n_layers)
    )
    return KVCache(layers=layers, length=jnp.asarray(0, dtype=jnp.int32))
