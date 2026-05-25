"""Forward-only RMSNorm wrappers for the ThunderKittens FFI POC."""

from pathlib import Path

import jax
import jax.numpy as jnp

from jaxtitan.errors import ContractError
from jaxtitan.kernels._ffi import register_rmsnorm

RMSNORM_HIDDEN_SIZE = 1024


def rmsnorm_reference(x: jax.Array, weight: jax.Array, *, eps: float = 1.0e-6) -> jax.Array:
    """Pure JAX RMSNorm reference with fp32 accumulation."""

    x_f32 = x.astype(jnp.float32)
    weight_f32 = weight.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps)
    return (x_f32 * inv_rms * weight_f32).astype(x.dtype)


def rmsnorm_tk_forward(
    x: jax.Array,
    weight: jax.Array,
    *,
    eps: float = 1.0e-6,
    cache_root: str | Path | None = None,
) -> jax.Array:
    """Run the cached TK RMSNorm FFI target.

    This is intentionally not wired into model code. It proves the JAX FFI
    boundary for one narrow shape and dtype before any training path depends on
    native kernels.
    """

    _validate_inputs(x, weight)
    if jax.default_backend() != "gpu":
        raise ContractError("TK RMSNorm FFI requires the JAX gpu backend")

    target_name = register_rmsnorm(cache_root)
    original_shape = x.shape
    flat_x = jnp.reshape(x, (-1, RMSNORM_HIDDEN_SIZE))
    call = jax.ffi.ffi_call(
        target_name,
        jax.ShapeDtypeStruct(flat_x.shape, flat_x.dtype),
        input_layouts=[(0, 1), (0,)],
        output_layouts=(0, 1),
        vmap_method="sequential",
    )
    out = call(flat_x, weight, eps=float(eps))
    return jnp.reshape(out, original_shape)


def _validate_inputs(x: jax.Array, weight: jax.Array) -> None:
    if jnp.dtype(x.dtype) != jnp.bfloat16:
        raise ContractError("TK RMSNorm FFI currently supports bfloat16 input only")
    if jnp.dtype(weight.dtype) != jnp.bfloat16:
        raise ContractError("TK RMSNorm FFI currently supports bfloat16 weight only")
    if len(x.shape) < 2:
        raise ContractError("TK RMSNorm FFI input rank must be at least 2")
    if x.shape[-1] != RMSNORM_HIDDEN_SIZE:
        raise ContractError("TK RMSNorm FFI hidden size must be 1024")
    if weight.shape != (RMSNORM_HIDDEN_SIZE,):
        raise ContractError("TK RMSNorm FFI weight shape must be [1024]")
