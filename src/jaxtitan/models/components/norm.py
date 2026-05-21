"""Normalization components."""

from flax import nnx

from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.specs.model import ModelSpec


def build_rms_norm(spec: ModelSpec, rngs: nnx.Rngs, *, features: int | None = None) -> nnx.RMSNorm:
    """Build an RMSNorm module with the model's dtype policy."""

    dtype = dtype_from_name(spec.compute_dtype)
    param_dtype = dtype_from_name(spec.param_dtype)
    return nnx.RMSNorm(
        spec.hidden_size if features is None else features,
        epsilon=spec.norm_epsilon,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
