"""
Mixed optimizer dispatch and shared matrix-optimizer utilities.
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


def mixed_matrix_adamw(labels, config: OptimizerConfig, learning_rate: Callable[[int], jax.Array]) -> optax.GradientTransformation:
    adam = config.adamw

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
            raise ValueError("mixed_matrix_adamw requires params for decoupled weight decay")
        count_inc = state.count + jnp.asarray(1, dtype=jnp.int32)
        lr = learning_rate(state.count)
        matrix_beta = matrix_beta_for_config(config)
        mu = jax.tree.map(
            lambda g, m: matrix_beta * m + (1.0 - matrix_beta) * g,
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
    if label == OptimClass.MATRIX.value and config.name in {"muon", "aurora", "riemannian_aurora"}:
        update = matrix_direction(grad, mu, count, config)
        update = scale_by_width(update)
        if config.weight_decay:
            update = update + config.weight_decay * param
        return -lr * update
    return adamw_update(grad, param, adam_m, adam_v, count, lr, config)


def adamw_update(grad, param, m, v, count, lr, config: OptimizerConfig):
    adam = config.adamw
    if config.name == "muon":
        nesterov = config.muon.nesterov
    else:
        nesterov = adam.nesterov
    if nesterov:
        m_hat = adam.b1 * m / (1.0 - adam.b1 ** (count + 1)) + (1.0 - adam.b1) * grad / (1.0 - adam.b1**count)
    else:
        m_hat = m / (1.0 - adam.b1**count)
    v_hat = v / (1.0 - adam.b2**count)
    update = m_hat / (jnp.sqrt(v_hat) + adam.eps)
    if config.weight_decay:
        update = update + config.weight_decay * param
    return -lr * update


def matrix_beta_for_config(config: OptimizerConfig):
    if config.name == "aurora":
        return config.aurora.beta
    if config.name == "riemannian_aurora":
        return config.riemannian_aurora.beta
    return config.muon.beta


def matrix_direction(grad, mu, count, config: OptimizerConfig):
    if config.name == "aurora":
        from research.optimizers.aurora import aurora_direction

        aurora = config.aurora
        direction = matrix_momentum_direction(grad, mu, count, beta=aurora.beta, nesterov=aurora.nesterov)
        return aurora_direction(direction, config)
    if config.name == "riemannian_aurora":
        from research.optimizers.aurora import riemannian_aurora_direction

        aurora = config.riemannian_aurora
        direction = matrix_momentum_direction(grad, mu, count, beta=aurora.beta, nesterov=aurora.nesterov)
        return riemannian_aurora_direction(direction, config)

    from research.optimizers.muon import muon_direction

    return muon_direction(grad, mu, count, config)


def matrix_momentum_direction(grad, mu, count, *, beta: float, nesterov: bool):
    if nesterov:
        momentum_hat = mu / (1.0 - beta ** (count + 1))
        grad_hat = grad / (1.0 - beta**count)
        return beta * momentum_hat + (1.0 - beta) * grad_hat
    return mu / (1.0 - beta**count)


def orthogonalize_newton_schulz(matrix, *, ns_steps: int, ns_coeffs: jax.Array, eps: float):
    if matrix.ndim != 2:
        raise ValueError(f"Matrix polar expects a 2D update, got shape={matrix.shape}")
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


def polar(matrix, config: OptimizerConfig):
    return orthogonalize_newton_schulz(
        matrix,
        ns_steps=config.muon.ns_steps,
        ns_coeffs=jnp.asarray(config.muon.ns_coeffs, dtype=matrix.dtype),
        eps=config.muon.eps,
    )


def scale_by_width(update):
    fan_in = update.shape[0]
    fan_out = update.shape[1]
    scale = jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in))
    return scale * update
