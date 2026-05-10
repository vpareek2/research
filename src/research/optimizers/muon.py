"""
Repo-owned AdamW and Muon optimizer implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from research.config import OptimizerConfig
from research.optimizers.routing import OptimClass


class MixedOptimizerState(NamedTuple):
    count: jax.Array
    mu: object
    adam_m: object
    adam_v: object


def mixed_muon_adamw(labels, config: OptimizerConfig, learning_rate: Callable[[int], jax.Array]) -> optax.GradientTransformation:
    adam = config.adamw
    muon = config.muon

    def init_fn(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return MixedOptimizerState(
            count=jnp.zeros([], dtype=jnp.int32),
            mu=zeros,
            adam_m=zeros,
            adam_v=zeros,
        )

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("mixed_muon_adamw requires params for decoupled weight decay")
        count_inc = state.count + jnp.asarray(1, dtype=jnp.int32)
        lr = learning_rate(state.count)
        mu = jax.tree.map(
            lambda g, m: muon.beta * m + (1.0 - muon.beta) * g,
            updates,
            state.mu,
        )
        adam_m = jax.tree.map(
            lambda g, m: adam.b1 * m + (1.0 - adam.b1) * g,
            updates,
            state.adam_m,
        )
        adam_v = jax.tree.map(
            lambda g, v: adam.b2 * v + (1.0 - adam.b2) * jnp.square(g),
            updates,
            state.adam_v,
        )
        next_updates = jax.tree.map(
            lambda g, p, m_mu, m_adam, v_adam, label: _leaf_update(
                g,
                p,
                m_mu,
                m_adam,
                v_adam,
                label,
                count_inc,
                lr,
                config,
            ),
            updates,
            params,
            mu,
            adam_m,
            adam_v,
            labels,
        )
        return next_updates, MixedOptimizerState(
            count=count_inc,
            mu=mu,
            adam_m=adam_m,
            adam_v=adam_v,
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _leaf_update(grad, param, mu, adam_m, adam_v, label, count, lr, config: OptimizerConfig):
    if label == OptimClass.MATRIX.value and config.name == "muon":
        update = _muon_direction(grad, mu, count, config)
        update = _scale_by_width(update)
        if config.weight_decay:
            update = update + config.weight_decay * param
        return -lr * update
    return _adamw_update(param, adam_m, adam_v, count, lr, config)


def _adamw_update(param, m, v, count, lr, config: OptimizerConfig):
    adam = config.adamw
    m_hat = m / (1.0 - adam.b1**count)
    v_hat = v / (1.0 - adam.b2**count)
    update = m_hat / (jnp.sqrt(v_hat) + adam.eps)
    if config.weight_decay:
        update = update + config.weight_decay * param
    return -lr * update


def _muon_direction(grad, mu, count, config: OptimizerConfig):
    muon = config.muon
    if muon.nesterov:
        momentum_hat = mu / (1.0 - muon.beta ** (count + 1))
        grad_hat = grad / (1.0 - muon.beta**count)
        direction = muon.beta * momentum_hat + (1.0 - muon.beta) * grad_hat
    else:
        direction = mu / (1.0 - muon.beta**count)
    return orthogonalize_newton_schulz(
        direction,
        ns_steps=muon.ns_steps,
        ns_coeffs=jnp.asarray(muon.ns_coeffs, dtype=direction.dtype),
        eps=muon.eps,
    )


def orthogonalize_newton_schulz(matrix, *, ns_steps: int, ns_coeffs: jax.Array, eps: float):
    if matrix.ndim != 2:
        raise ValueError(f"Muon expects a matrix update, got shape={matrix.shape}")
    x = matrix.astype(jnp.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x, ord="fro") + eps)
    coeffs = ns_coeffs.astype(x.dtype)

    def step(_, value):
        xx_t = value @ value.T
        return coeffs[0] * value + (coeffs[1] * xx_t + coeffs[2] * xx_t @ xx_t) @ value

    x = jax.lax.fori_loop(0, ns_steps, step, x, unroll=True)
    if transposed:
        x = x.T
    return x.astype(matrix.dtype)


def _scale_by_width(update):
    fan_in = update.shape[0]
    fan_out = update.shape[1]
    scale = jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in))
    return scale * update
