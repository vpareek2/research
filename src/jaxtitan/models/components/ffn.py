"""Feed-forward model components."""

import jax
from flax import nnx

from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.specs.model import ModelSpec


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
