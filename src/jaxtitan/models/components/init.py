"""Model parameter initialization helpers."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def truncated_normal_init(std: float) -> Callable[[Any, tuple[int, ...], Any], jax.Array]:
    """Build a zero-mean truncated normal initializer clipped at three standard deviations."""

    def _init(key: Any, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        values = jax.random.truncated_normal(key, lower=-3.0, upper=3.0, shape=shape, dtype=dtype)
        return values * jnp.asarray(std, dtype=dtype)

    return _init
