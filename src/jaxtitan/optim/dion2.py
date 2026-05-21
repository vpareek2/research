"""Native JAX Dion2 transform used for sharded Muon routes."""

from collections.abc import Callable
from typing import Any, NamedTuple
import math

import jax
import jax.numpy as jnp
import optax

PyTree = Any

DION2_FRACTION = 0.25
DION2_EF_DECAY = 0.95
DION2_EPSILON = 1e-8
DION2_ADJUST_LR = "spectral_norm"
DION2_ORTHOGONALIZER = "polar_express"
DION2_POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


class Dion2State(NamedTuple):
    """Opaque optimizer state for the Dion2 transform."""

    count: jax.Array
    momentum: PyTree


def dion2_transform(
    learning_rate: Callable[[Any], jax.Array],
    *,
    weight_decay: float,
    select_axis: int,
) -> optax.GradientTransformationExtraArgs:
    """Build a matrix-only Dion2 gradient transformation."""

    if select_axis not in {0, 1}:
        raise ValueError(f"Dion2 select_axis must be 0 or 1, got {select_axis}")

    def init_fn(params: PyTree) -> Dion2State:
        return Dion2State(
            count=jnp.asarray(0, dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(
        updates: PyTree,
        state: Dion2State,
        params: PyTree | None = None,
        **extra_args,
    ) -> tuple[PyTree, Dion2State]:
        del extra_args
        if params is None:
            raise ValueError("Dion2 update requires params")
        base_lr = learning_rate(state.count)

        def update_leaf(grad: jax.Array, momentum: jax.Array, param: jax.Array) -> jax.Array:
            next_momentum = momentum + grad.astype(momentum.dtype)
            selected, indices = select_dion2_slices(
                next_momentum,
                select_axis=select_axis,
                fraction=DION2_FRACTION,
            )
            orthogonalized = polar_express(selected, epsilon=DION2_EPSILON).astype(param.dtype)
            update = jnp.zeros_like(param)
            update = scatter_dion2_slices(
                update,
                indices,
                -base_lr.astype(param.dtype) * _spectral_lr_scale(param.shape) * orthogonalized,
                select_axis=select_axis,
                scatter_mode="add",
            )
            if weight_decay != 0.0:
                update = update - base_lr.astype(param.dtype) * weight_decay * param
            return update

        def momentum_leaf(grad: jax.Array, momentum: jax.Array) -> jax.Array:
            next_momentum = momentum + grad.astype(momentum.dtype)
            selected, indices = select_dion2_slices(
                next_momentum,
                select_axis=select_axis,
                fraction=DION2_FRACTION,
            )
            return scatter_dion2_slices(
                next_momentum,
                indices,
                selected * DION2_EF_DECAY,
                select_axis=select_axis,
                scatter_mode="set",
            )

        next_updates = jax.tree.map(update_leaf, updates, state.momentum, params)
        next_momentum = jax.tree.map(momentum_leaf, updates, state.momentum)
        next_state = Dion2State(count=optax.safe_increment(state.count), momentum=next_momentum)
        return next_updates, next_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def select_dion2_slices(
    value: jax.Array,
    *,
    select_axis: int,
    fraction: float = DION2_FRACTION,
) -> tuple[jax.Array, jax.Array]:
    """Select Dion2 top-k rows or columns by L1 norm."""

    if len(value.shape) != 2:
        raise ValueError(f"Dion2 expects rank-2 arrays, got shape {value.shape}")
    if select_axis not in {0, 1}:
        raise ValueError(f"Dion2 select_axis must be 0 or 1, got {select_axis}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Dion2 fraction must be in (0, 1], got {fraction}")
    select_dim = value.shape[select_axis]
    k = max(1, math.ceil(fraction * select_dim))
    norm_axis = 1 if select_axis == 0 else 0
    norms = jnp.sum(jnp.abs(jnp.asarray(value, dtype=jnp.float32)), axis=norm_axis)
    _values, indices = jax.lax.top_k(norms, k)
    selected = value[indices, :] if select_axis == 0 else jnp.take(value, indices, axis=1)
    return selected, indices


def scatter_dion2_slices(
    target: jax.Array,
    indices: jax.Array,
    values: jax.Array,
    *,
    select_axis: int,
    scatter_mode: str,
) -> jax.Array:
    """Scatter selected Dion2 row or column slices back into a matrix."""

    if select_axis == 0:
        scatter = target.at[indices, :]
    elif select_axis == 1:
        scatter = target.at[:, indices]
    else:
        raise ValueError(f"Dion2 select_axis must be 0 or 1, got {select_axis}")
    if scatter_mode == "add":
        return scatter.add(values)
    if scatter_mode == "set":
        return scatter.set(values)
    raise ValueError(f"unsupported Dion2 scatter_mode {scatter_mode!r}")


def polar_express(value: jax.Array, *, epsilon: float = DION2_EPSILON) -> jax.Array:
    """Approximate a polar factor using the Dion2 reference Polar Express coefficients."""

    if len(value.shape) != 2:
        raise ValueError(f"Polar Express expects rank-2 arrays, got shape {value.shape}")
    original_dtype = value.dtype
    x = value.astype(jnp.bfloat16)
    x = x / (jnp.linalg.norm(x) * jnp.asarray(1.02, dtype=x.dtype) + jnp.asarray(epsilon, dtype=x.dtype))
    tall = x.shape[0] > x.shape[1]
    for a, b, c in DION2_POLAR_EXPRESS_COEFFICIENTS:
        if tall:
            gram = x.T @ x
            update = b * gram + c * (gram @ gram)
            x = a * x + x @ update
        else:
            gram = x @ x.T
            update = b * gram + c * (gram @ gram)
            x = a * x + update @ x
    return x.astype(original_dtype)


def dion2_policy_constants() -> dict[str, Any]:
    """Return stable Dion2 constants recorded in artifacts."""

    return {
        "fraction": DION2_FRACTION,
        "ef_decay": DION2_EF_DECAY,
        "epsilon": DION2_EPSILON,
        "adjust_lr": DION2_ADJUST_LR,
        "orthogonalizer": DION2_ORTHOGONALIZER,
        "polar_express_coefficients": [list(coeffs) for coeffs in DION2_POLAR_EXPRESS_COEFFICIENTS],
    }


def _spectral_lr_scale(shape: tuple[int, ...]) -> jax.Array:
    rows, cols = shape[-2:]
    return jnp.asarray(math.sqrt(rows / cols), dtype=jnp.float32)
