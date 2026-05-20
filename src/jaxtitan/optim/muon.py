"""Native JAX Muon transform."""

from collections.abc import Callable
from typing import Any, NamedTuple
import math

import jax
import jax.numpy as jnp
import optax

PyTree = Any

MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_STEPS = 5
MUON_NS_EPS = 1e-7
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_RMS_MATCH_SCALE = 0.2
MUON_SCALE_MODE = "match_rms_adamw"


class MuonState(NamedTuple):
    """Opaque optimizer state for the Muon transform."""

    count: jax.Array
    momentum: PyTree


def muon_transform(
    learning_rate: Callable[[Any], jax.Array],
    *,
    weight_decay: float,
) -> optax.GradientTransformationExtraArgs:
    """Build a matrix-only Muon gradient transformation."""

    def init_fn(params: PyTree) -> MuonState:
        return MuonState(
            count=jnp.asarray(0, dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(updates: PyTree, state: MuonState, params: PyTree | None = None, **extra_args) -> tuple[PyTree, MuonState]:
        del extra_args
        if params is None:
            raise ValueError("Muon update requires params")
        base_lr = learning_rate(state.count)

        def update_leaf(grad: jax.Array, momentum: jax.Array, param: jax.Array) -> jax.Array:
            next_momentum = MUON_MOMENTUM * momentum + (1.0 - MUON_MOMENTUM) * grad
            muon_input = (
                (1.0 - MUON_MOMENTUM) * grad + MUON_MOMENTUM * next_momentum
                if MUON_NESTEROV
                else next_momentum
            )
            orthogonalized = zeropower_via_newton_schulz(muon_input)
            adjusted_lr = base_lr * _rms_match_scale(param.shape)
            update = -adjusted_lr.astype(orthogonalized.dtype) * orthogonalized
            if weight_decay != 0.0:
                update = update - base_lr.astype(param.dtype) * weight_decay * param
            return update

        def momentum_leaf(grad: jax.Array, momentum: jax.Array) -> jax.Array:
            return MUON_MOMENTUM * momentum + (1.0 - MUON_MOMENTUM) * grad

        next_updates = jax.tree.map(update_leaf, updates, state.momentum, params)
        next_momentum = jax.tree.map(momentum_leaf, updates, state.momentum)
        next_state = MuonState(count=optax.safe_increment(state.count), momentum=next_momentum)
        return next_updates, next_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def zeropower_via_newton_schulz(value: jax.Array) -> jax.Array:
    """Approximate the zeroth power of a 2D matrix using Newton-Schulz iterations."""

    if len(value.shape) != 2:
        raise ValueError(f"Muon Newton-Schulz expects rank-2 arrays, got shape {value.shape}")
    original_dtype = value.dtype
    x = value.astype(jnp.float32)
    x = x / (jnp.linalg.norm(x) + jnp.asarray(MUON_NS_EPS, dtype=jnp.float32))
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x.astype(jnp.bfloat16)
    a, b, c = (jnp.asarray(coeff, dtype=jnp.bfloat16) for coeff in MUON_NS_COEFFICIENTS)
    for _ in range(MUON_NS_STEPS):
        xx_t = x @ x.T
        update = b * xx_t + c * (xx_t @ xx_t)
        x = a * x + update @ x
    if transposed:
        x = x.T
    return x.astype(original_dtype)


def muon_policy_constants() -> dict[str, Any]:
    """Return stable Muon constants recorded in artifacts."""

    return {
        "momentum": MUON_MOMENTUM,
        "nesterov": MUON_NESTEROV,
        "newton_schulz_steps": MUON_NS_STEPS,
        "newton_schulz_eps": MUON_NS_EPS,
        "newton_schulz_coefficients": list(MUON_NS_COEFFICIENTS),
        "scale_mode": MUON_SCALE_MODE,
        "rms_match_scale": MUON_RMS_MATCH_SCALE,
    }


def _rms_match_scale(shape: tuple[int, ...]) -> jax.Array:
    return jnp.asarray(MUON_RMS_MATCH_SCALE * math.sqrt(max(shape)), dtype=jnp.float32)
