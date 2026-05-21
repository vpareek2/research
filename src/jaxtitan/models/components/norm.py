"""Normalization components."""

from typing import Any

import jax.numpy as jnp
from flax import nnx

from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.specs.model import ModelSpec


def build_rms_norm(
    spec: ModelSpec,
    rngs: nnx.Rngs,
    *,
    features: int | None = None,
    scale_init_value: float = 1.0,
) -> nnx.RMSNorm:
    """Build an RMSNorm module with the model's dtype policy."""

    dtype = dtype_from_name(spec.compute_dtype)
    param_dtype = dtype_from_name(spec.param_dtype)
    scale_init = _scale_init(scale_init_value)
    return nnx.RMSNorm(
        spec.hidden_size if features is None else features,
        epsilon=spec.norm_epsilon,
        dtype=dtype,
        param_dtype=param_dtype,
        scale_init=scale_init,
        rngs=rngs,
    )


def _scale_init(value: float):
    def _init(_key: Any, shape: tuple[int, ...], dtype: Any = jnp.float32):
        return jnp.full(shape, value, dtype=dtype)

    return _init
